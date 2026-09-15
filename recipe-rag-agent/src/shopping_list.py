"""
shopping_list.py
-----------------
Turns a list of ingredient lines (as generated for a recipe) into a
clean shopping checklist, removing anything the user says they already
have on hand.
"""

from __future__ import annotations

import re
from typing import List

_LEADING_BULLET = re.compile(r"^[\-\*\u2022]\s*")
_QTY_PREFIX = re.compile(r"^[\d/.\s]+[a-zA-Z]*\s+")


def _normalize(line: str) -> str:
    line = _LEADING_BULLET.sub("", line.strip())
    return line.strip()


def _core_ingredient_name(line: str) -> str:
    """Strip quantities/units to get a rough matchable ingredient name."""
    stripped = _QTY_PREFIX.sub("", line).strip().lower()
    # Drop trailing prep notes like ", chopped" or "(optional)"
    stripped = re.split(r",|\(", stripped)[0].strip()
    return stripped


def build_shopping_list(
    ingredient_lines: List[str],
    have_on_hand: List[str] | None = None,
) -> List[str]:
    """
    Given raw ingredient lines from a generated recipe, return a deduped
    shopping list with items the user already has removed.
    """
    have_on_hand = have_on_hand or []
    have_normalized = {h.strip().lower() for h in have_on_hand if h.strip()}

    seen = set()
    shopping_list = []

    for raw_line in ingredient_lines:
        line = _normalize(raw_line)
        if not line:
            continue

        core = _core_ingredient_name(line)
        if not core or core in seen:
            continue

        # Skip if the user said they already have this ingredient
        # (substring match in both directions catches "egg" vs "eggs").
        already_have = any(
            h in core or core in h for h in have_normalized if len(h) > 2
        )
        if already_have:
            seen.add(core)
            continue

        seen.add(core)
        shopping_list.append(line)

    return shopping_list


def format_as_markdown(shopping_list: List[str]) -> str:
    if not shopping_list:
        return "_Nothing to buy -- you have everything you need!_"
    return "\n".join(f"- [ ] {item}" for item in shopping_list)
