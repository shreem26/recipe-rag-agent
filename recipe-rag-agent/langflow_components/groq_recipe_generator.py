import json

from langflow.custom import Component
from langflow.io import DataInput, DropdownInput, IntInput, MultilineInput, Output
from langflow.schema import Data

from src.llm import generate


def _normalize_hit(hit: object) -> dict[str, str]:
    """Normalize Langflow-serialized retrieval records for prompt building."""
    if isinstance(hit, dict):
        return {
            "title": str(hit.get("title", "Recipe context")),
            "source": str(hit.get("source", "uploaded document")),
            "text": str(hit.get("text", "")),
        }
    if isinstance(hit, (list, tuple)):
        values = [str(value) for value in hit]
        return {
            "title": values[0] if values else "Recipe context",
            "source": values[1] if len(values) > 1 else "uploaded document",
            "text": "\n".join(values[2:]) if len(values) > 2 else "\n".join(values),
        }
    return {"title": "Recipe context", "source": "uploaded document", "text": str(hit)}


class GroqRecipeGenerator(Component):
    display_name = "Groq Recipe Generator"
    description = "Generate a structured recipe with the configured Groq model."
    icon = "MessageSquareText"
    name = "GroqRecipeGenerator"

    inputs = [
        DataInput(name="context", display_name="Retrieved context", required=True),
        MultilineInput(name="dietary_restrictions", display_name="Dietary restrictions", value=""),
        MultilineInput(name="available_ingredients", display_name="Ingredients on hand", value=""),
        IntInput(name="max_time_minutes", display_name="Maximum time", value=30),
        DropdownInput(name="cuisine", display_name="Cuisine", options=["No preference", "Italian", "Indian", "Mexican", "Chinese", "Thai", "Mediterranean", "American", "Japanese"], value="No preference"),
        IntInput(name="servings", display_name="Servings", value=4),
    ]
    outputs = [Output(name="recipe_json", display_name="Recipe JSON", method="generate_recipe")]

    def generate_recipe(self) -> Data:
        payload = self.context.data if isinstance(self.context, Data) else self.context
        if isinstance(payload, dict):
            hits = payload.get("hits", [])
            query = payload.get("query", "Create a recipe")
        else:
            hits = payload if isinstance(payload, list) else [payload]
            query = "Create a recipe"
        normalized_hits = [_normalize_hit(hit) for hit in hits]
        context = "\n\n".join(f"--- {hit['title']} ({hit['source']}) ---\n{hit['text']}" for hit in normalized_hits)
        constraints = f"Dietary restrictions: {self.dietary_restrictions or 'None'}\nIngredients on hand: {self.available_ingredients or 'None'}\nMaximum time: {self.max_time_minutes} minutes\nCuisine: {self.cuisine}\nServings: {self.servings}"
        system = "You are a recipe RAG assistant. Return ONLY valid JSON with keys: title, servings, prep_time_minutes, cuisine, summary, ingredients, steps, substitutions, notes, sources_used."
        raw = generate(system, f"Question: {query}\n\nConstraints:\n{constraints}\n\nRetrieved context:\n{context or '(No matching recipes were found.)'}")
        try:
            recipe = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError:
            recipe = {"title": query.title(), "ingredients": [], "steps": [], "notes": raw}
        recipe["_available_ingredients"] = [item.strip() for item in self.available_ingredients.split(",") if item.strip()]
        recipe["_retrieved_sources"] = sorted({hit["source"] for hit in normalized_hits})
        return Data(data=recipe)