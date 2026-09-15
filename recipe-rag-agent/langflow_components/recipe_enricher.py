from typing import Any

from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema import Data

from src.nutrition import estimate_nutrition
from src.shopping_list import build_shopping_list


class RecipeEnricher(Component):
    display_name = "Recipe Enricher"
    description = "Add deterministic nutrition estimates and a shopping list."
    icon = "ListChecks"
    name = "RecipeEnricher"

    inputs = [DataInput(name="recipe_json", display_name="Recipe JSON", required=True)]
    outputs = [Output(name="recipe", display_name="Enriched recipe", method="enrich")]

    def enrich(self) -> Data:
        recipe: dict[str, Any] = dict(self.recipe_json.data if isinstance(self.recipe_json, Data) else self.recipe_json)
        ingredients = recipe.get("ingredients") or []
        servings = int(recipe.get("servings") or 4)
        nutrition = estimate_nutrition(ingredients, servings=servings)
        recipe["nutrition_per_serving"] = {"calories": nutrition.calories, "protein_g": nutrition.protein_g, "carbs_g": nutrition.carbs_g, "fat_g": nutrition.fat_g}
        recipe["shopping_list"] = build_shopping_list(ingredients, recipe.pop("_available_ingredients", []))
        recipe["retrieved_sources"] = recipe.pop("_retrieved_sources", [])
        return Data(data=recipe)