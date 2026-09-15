"""
nutrition.py
------------
A lightweight, offline nutrition estimator.

This intentionally does NOT call a paid nutrition API -- it uses a small
lookup table of per-100g macros for common ingredients plus rough unit
conversions, so the app works fully offline / for free. It's an estimate
for guidance, not a substitute for a verified nutrition label, and the
UI labels it as such.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# Per 100g: (calories, protein_g, carbs_g, fat_g)
NUTRITION_TABLE: Dict[str, tuple] = {
    "flour": (364, 10, 76, 1),
    "sugar": (387, 0, 100, 0),
    "brown sugar": (380, 0, 98, 0),
    "butter": (717, 1, 0, 81),
    "olive oil": (884, 0, 0, 100),
    "vegetable oil": (884, 0, 0, 100),
    "egg": (155, 13, 1, 11),
    "milk": (61, 3.2, 5, 3.3),
    "heavy cream": (340, 2, 3, 36),
    "chocolate": (546, 5, 61, 31),
    "cocoa powder": (228, 20, 58, 14),
    "baking powder": (53, 0, 28, 0),
    "baking soda": (0, 0, 0, 0),
    "salt": (0, 0, 0, 0),
    "vanilla extract": (288, 0, 13, 0),
    "chicken breast": (165, 31, 0, 3.6),
    "chicken thigh": (209, 26, 0, 11),
    "ground beef": (250, 26, 0, 17),
    "rice": (130, 2.7, 28, 0.3),
    "pasta": (131, 5, 25, 1.1),
    "potato": (77, 2, 17, 0.1),
    "onion": (40, 1.1, 9, 0.1),
    "garlic": (149, 6.4, 33, 0.5),
    "tomato": (18, 0.9, 3.9, 0.2),
    "carrot": (41, 0.9, 10, 0.2),
    "bell pepper": (31, 1, 6, 0.3),
    "spinach": (23, 2.9, 3.6, 0.4),
    "broccoli": (34, 2.8, 7, 0.4),
    "cheese": (402, 25, 1.3, 33),
    "parmesan": (431, 38, 4, 29),
    "mozzarella": (280, 28, 3, 17),
    "yogurt": (61, 3.5, 4.7, 3.3),
    "almond": (579, 21, 22, 50),
    "peanut butter": (588, 25, 20, 50),
    "honey": (304, 0.3, 82, 0),
    "maple syrup": (260, 0, 67, 0.2),
    "banana": (89, 1.1, 23, 0.3),
    "apple": (52, 0.3, 14, 0.2),
    "lemon": (29, 1.1, 9, 0.3),
    "beans": (127, 8.7, 23, 0.5),
    "lentils": (116, 9, 20, 0.4),
    "tofu": (76, 8, 1.9, 4.8),
    "soy sauce": (53, 8, 5, 0),
    "bread": (265, 9, 49, 3.2),
    "oats": (389, 17, 66, 7),
}

# Very rough unit -> grams conversions (ingredient-agnostic approximations)
UNIT_GRAMS = {
    "cup": 120,
    "cups": 120,
    "tbsp": 15,
    "tablespoon": 15,
    "tablespoons": 15,
    "tsp": 5,
    "teaspoon": 5,
    "teaspoons": 5,
    "g": 1,
    "gram": 1,
    "grams": 1,
    "kg": 1000,
    "oz": 28,
    "ounce": 28,
    "ounces": 28,
    "lb": 454,
    "pound": 454,
    "pounds": 454,
    "ml": 1,
    "l": 1000,
    "clove": 5,
    "cloves": 5,
    "slice": 30,
    "slices": 30,
}

DEFAULT_ITEM_GRAMS = 50  # fallback when we can't parse a quantity/unit

QUANTITY_RE = re.compile(
    r"^\s*(?P<qty>[\d/.]+)?\s*(?P<unit>[a-zA-Z]+)?\s*(?P<rest>.*)$"
)


@dataclass
class NutritionEstimate:
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float
    matched_ingredients: int
    total_ingredients: int

    def per_serving(self, servings: int) -> "NutritionEstimate":
        servings = max(servings, 1)
        return NutritionEstimate(
            calories=round(self.calories / servings),
            protein_g=round(self.protein_g / servings, 1),
            carbs_g=round(self.carbs_g / servings, 1),
            fat_g=round(self.fat_g / servings, 1),
            matched_ingredients=self.matched_ingredients,
            total_ingredients=self.total_ingredients,
        )


def _parse_qty(qty_str: Optional[str]) -> float:
    if not qty_str:
        return 1.0
    try:
        if "/" in qty_str:
            num, denom = qty_str.split("/")
            return float(num) / float(denom)
        return float(qty_str)
    except ValueError:
        return 1.0


def _match_ingredient(text: str) -> Optional[str]:
    text_lower = text.lower()
    # Prefer the longest matching key so "brown sugar" beats "sugar"
    matches = [name for name in NUTRITION_TABLE if name in text_lower]
    if not matches:
        return None
    return max(matches, key=len)


def estimate_nutrition(ingredient_lines: List[str], servings: int = 4) -> NutritionEstimate:
    total_cal = total_protein = total_carbs = total_fat = 0.0
    matched = 0

    for line in ingredient_lines:
        clean = line.strip("-*• ").strip()
        if not clean:
            continue

        match = QUANTITY_RE.match(clean)
        qty = _parse_qty(match.group("qty") if match else None)
        unit = (match.group("unit") or "").lower() if match else ""
        rest = match.group("rest") if match else clean

        ingredient_key = _match_ingredient(rest if rest else clean)
        if not ingredient_key:
            continue

        grams_per_unit = UNIT_GRAMS.get(unit, DEFAULT_ITEM_GRAMS)
        grams = qty * grams_per_unit
        cal, protein, carbs, fat = NUTRITION_TABLE[ingredient_key]

        factor = grams / 100.0
        total_cal += cal * factor
        total_protein += protein * factor
        total_carbs += carbs * factor
        total_fat += fat * factor
        matched += 1

    return NutritionEstimate(
        calories=round(total_cal),
        protein_g=round(total_protein, 1),
        carbs_g=round(total_carbs, 1),
        fat_g=round(total_fat, 1),
        matched_ingredients=matched,
        total_ingredients=len([l for l in ingredient_lines if l.strip()]),
    ).per_serving(servings)
