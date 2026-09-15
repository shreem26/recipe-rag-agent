from typing import Any

from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema import Data
from langflow.schema.message import Message


class RecipeFormatter(Component):
    display_name = "Recipe Formatter"
    description = "Convert the enriched recipe data into a readable Markdown response."
    icon = "FileText"
    name = "RecipeFormatter"

    inputs = [DataInput(name="recipe", display_name="Enriched recipe", required=True)]
    outputs = [Output(name="text", display_name="Recipe response", method="format_recipe")]

    def format_recipe(self) -> Message:
        recipe: dict[str, Any] = dict(self.recipe.data if isinstance(self.recipe, Data) else self.recipe)
        lines = [
            f"# {recipe.get('title', 'Recipe')}",
            "",
            recipe.get("summary", ""),
            "",
            f"**Cuisine:** {recipe.get('cuisine', 'Not specified')}  "
            f"**Servings:** {recipe.get('servings', 'Not specified')}  "
            f"**Time:** {recipe.get('prep_time_minutes', 'Not specified')} minutes",
            "",
            "## Ingredients",
        ]
        lines.extend(f"- {ingredient}" for ingredient in recipe.get("ingredients", []))
        lines.extend(["", "## Steps"])
        lines.extend(f"{index}. {step.lstrip('0123456789. ')}" for index, step in enumerate(recipe.get("steps", []), 1))
        substitutions = recipe.get("substitutions", {})
        if substitutions:
            lines.extend(["", "## Substitutions"])
            if isinstance(substitutions, dict):
                lines.extend(f"- **{name}:** {value}" for name, value in substitutions.items())
            else:
                lines.append(str(substitutions))
        nutrition = recipe.get("nutrition_per_serving", {})
        if nutrition:
            lines.extend([
                "",
                "## Nutrition per serving",
                f"- Calories: {nutrition.get('calories', 'N/A')}",
                f"- Protein: {nutrition.get('protein_g', 'N/A')} g",
                f"- Carbohydrates: {nutrition.get('carbs_g', 'N/A')} g",
                f"- Fat: {nutrition.get('fat_g', 'N/A')} g",
            ])
        shopping_list = recipe.get("shopping_list", [])
        if shopping_list:
            lines.extend(["", "## Shopping list"])
            lines.extend(f"- {item}" for item in shopping_list)
        if recipe.get("notes"):
            lines.extend(["", "## Notes", str(recipe["notes"])])
        return Message(text="\n".join(lines), sender="Machine", sender_name="Recipe Formatter")
