"""
recipe_agent.py
----------------
The agentic layer that ties everything together:

  1. Retrieve  -- pull the most relevant recipe chunks from the vector store
  2. Adapt     -- apply the user's constraints (diet, available ingredients,
                  time budget, cuisine) when asking Claude to answer
  3. Structure -- force the model to return well-formed JSON so the UI can
                  render steps, substitutions, and nutrition consistently
  4. Enrich    -- compute an offline nutrition estimate and a shopping list
                  locally (deterministic, not left to the LLM to invent)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from . import llm
from .nutrition import estimate_nutrition
from .shopping_list import build_shopping_list
from .vector_store import RecipeVectorStore

SYSTEM_PROMPT = """You are an expert recipe assistant working inside a \
Retrieval-Augmented Generation app. You are given excerpts retrieved from \
the user's own cookbook / recipe knowledge base, plus constraints the user \
has specified (dietary restrictions, ingredients on hand, time budget, \
cuisine preference).

Rules:
- Ground your answer in the retrieved excerpts whenever they are relevant. \
If the excerpts don't cover what's being asked, say so plainly and then use \
your own general culinary knowledge to help -- never invent a fake source.
- Always respect hard dietary constraints (e.g. "sugar-free" means no sugar, \
not "less sugar"). If a constraint can't be fully satisfied, explain the \
closest reasonable substitution rather than ignoring the constraint.
- Prefer ingredients the user says they already have; suggest substitutions \
for anything they're missing or that violates a dietary constraint.
- Respond with ONLY a single JSON object -- no markdown fences, no commentary \
before or after it. Match this schema exactly:

{
  "title": "string",
  "servings": integer,
  "prep_time_minutes": integer,
  "cuisine": "string",
  "summary": "one or two sentence description",
  "ingredients": ["quantity + ingredient, e.g. '2 cups flour'", "..."],
  "steps": ["step 1 text", "step 2 text", "..."],
  "substitutions": ["e.g. 'Use maple syrup instead of honey for vegan'", "..."],
  "notes": "any caveats, e.g. constraints that could only be partially met",
  "sources_used": ["title of retrieved recipe used, if any", "..."]
}
"""


@dataclass
class UserConstraints:
    dietary_restrictions: List[str] = field(default_factory=list)
    available_ingredients: List[str] = field(default_factory=list)
    max_time_minutes: Optional[int] = None
    cuisine: Optional[str] = None
    servings: Optional[int] = None

    def as_prompt_block(self) -> str:
        lines = []
        if self.dietary_restrictions:
            lines.append(f"Dietary restrictions: {', '.join(self.dietary_restrictions)}")
        if self.available_ingredients:
            lines.append(
                f"Ingredients already on hand (prefer using these): "
                f"{', '.join(self.available_ingredients)}"
            )
        if self.max_time_minutes:
            lines.append(f"Time budget: {self.max_time_minutes} minutes or less")
        if self.cuisine:
            lines.append(f"Preferred cuisine style: {self.cuisine}")
        if self.servings:
            lines.append(f"Desired servings: {self.servings}")
        return "\n".join(lines) if lines else "No specific constraints given."


@dataclass
class RecipeAnswer:
    title: str
    servings: int
    prep_time_minutes: int
    cuisine: str
    summary: str
    ingredients: List[str]
    steps: List[str]
    substitutions: List[str]
    notes: str
    sources_used: List[str]
    retrieved_sources: List[str]
    nutrition_per_serving: dict
    shopping_list: List[str]


def _extract_json(raw_text: str) -> dict:
    """Be forgiving about stray markdown fences the model might add."""
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    text = text.strip()
    return json.loads(text)


class RecipeRAGAgent:
    def __init__(self, vector_store: RecipeVectorStore):
        self.vector_store = vector_store

    def _build_user_prompt(
        self, query: str, constraints: UserConstraints, retrieved: list
    ) -> str:
        if retrieved:
            context_blocks = "\n\n".join(
                f"--- Retrieved recipe: {hit['title']} (source: {hit['source']}) ---\n{hit['text']}"
                for hit in retrieved
            )
        else:
            context_blocks = "(No matching recipes were found in the knowledge base.)"

        return f"""User question: {query}

User constraints:
{constraints.as_prompt_block()}

Retrieved context from the recipe knowledge base:
{context_blocks}

Using the retrieved context where relevant and your own culinary expertise \
to fill gaps, produce the JSON recipe object described in your instructions."""

    def ask(
        self,
        query: str,
        constraints: Optional[UserConstraints] = None,
        k: int = 4,
    ) -> RecipeAnswer:
        constraints = constraints or UserConstraints()
        retrieved = self.vector_store.search(query, k=k)

        user_prompt = self._build_user_prompt(query, constraints, retrieved)
        raw = llm.generate(SYSTEM_PROMPT, user_prompt)

        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            # Fall back to a minimal structure so the UI doesn't crash;
            # surface the raw model output in `notes` for debugging.
            data = {
                "title": query.title(),
                "servings": constraints.servings or 4,
                "prep_time_minutes": constraints.max_time_minutes or 30,
                "cuisine": constraints.cuisine or "unspecified",
                "summary": "The model's response could not be parsed as JSON.",
                "ingredients": [],
                "steps": [],
                "substitutions": [],
                "notes": raw[:1000],
                "sources_used": [],
            }

        servings = int(data.get("servings") or constraints.servings or 4)
        ingredients = data.get("ingredients") or []

        nutrition = estimate_nutrition(ingredients, servings=servings)
        shopping = build_shopping_list(ingredients, constraints.available_ingredients)

        return RecipeAnswer(
            title=data.get("title", query.title()),
            servings=servings,
            prep_time_minutes=int(data.get("prep_time_minutes") or 0),
            cuisine=data.get("cuisine", "unspecified"),
            summary=data.get("summary", ""),
            ingredients=ingredients,
            steps=data.get("steps") or [],
            substitutions=data.get("substitutions") or [],
            notes=data.get("notes", ""),
            sources_used=data.get("sources_used") or [],
            retrieved_sources=sorted({hit["source"] for hit in retrieved}),
            nutrition_per_serving={
                "calories": nutrition.calories,
                "protein_g": nutrition.protein_g,
                "carbs_g": nutrition.carbs_g,
                "fat_g": nutrition.fat_g,
                "estimated_from": f"{nutrition.matched_ingredients}/{nutrition.total_ingredients} ingredients matched",
            },
            shopping_list=shopping,
        )
