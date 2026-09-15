"""
Basic tests that don't require an API key -- they cover the
deterministic parts of the pipeline: ingestion, chunking, nutrition
estimation, and shopping list generation.

Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import chunk_documents  # noqa: E402
from src.ingest import load_directory  # noqa: E402
from src.nutrition import estimate_nutrition  # noqa: E402
from src.shopping_list import build_shopping_list  # noqa: E402

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_recipes"


def test_load_sample_recipes():
    docs = load_directory(SAMPLE_DIR)
    assert len(docs) >= 5, "Expected at least 5 bundled sample recipes"
    assert all(doc.text.strip() for doc in docs)


def test_chunking_produces_titled_chunks():
    docs = load_directory(SAMPLE_DIR)
    chunks = chunk_documents(docs)
    assert len(chunks) >= len(docs)
    assert all(c.title for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_nutrition_estimate_reasonable():
    ingredients = ["2 cups flour", "1 cup sugar", "2 egg", "1 cup milk"]
    result = estimate_nutrition(ingredients, servings=4)
    assert result.calories > 0
    assert result.matched_ingredients == 4
    assert result.total_ingredients == 4


def test_nutrition_estimate_handles_unmatched_lines():
    ingredients = ["a pinch of fairy dust", "2 cups flour"]
    result = estimate_nutrition(ingredients, servings=2)
    assert result.matched_ingredients == 1
    assert result.total_ingredients == 2


def test_shopping_list_removes_items_on_hand():
    ingredients = ["2 cups flour", "2 egg", "1 cup milk", "1 tsp vanilla extract"]
    shopping = build_shopping_list(ingredients, have_on_hand=["eggs", "milk"])
    joined = " | ".join(shopping).lower()
    assert "egg" not in joined
    assert "milk" not in joined
    assert "flour" in joined


def test_shopping_list_dedupes():
    ingredients = ["1 cup flour", "1 cup flour", "2 egg"]
    shopping = build_shopping_list(ingredients)
    assert len(shopping) == 2


if __name__ == "__main__":
    test_load_sample_recipes()
    test_chunking_produces_titled_chunks()
    test_nutrition_estimate_reasonable()
    test_nutrition_estimate_handles_unmatched_lines()
    test_shopping_list_removes_items_on_hand()
    test_shopping_list_dedupes()
    print("All tests passed.")
