"""
Document Q&A RAG Agent for Recipe Generator
--------------------------------------------
Streamlit front-end. Run with:

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

load_dotenv()

from src.chunker import chunk_documents  # noqa: E402
from src.ingest import SUPPORTED_EXTENSIONS, load_directory, load_document  # noqa: E402
from src.llm import LLMNotConfigured  # noqa: E402
from src.recipe_agent import RecipeRAGAgent, UserConstraints  # noqa: E402
from src.shopping_list import format_as_markdown  # noqa: E402
from src.vector_store import RecipeVectorStore  # noqa: E402

st.set_page_config(page_title="Recipe RAG Agent", page_icon="🍳", layout="wide")

SAMPLE_DIR = Path("data/sample_recipes")
UPLOAD_DIR = Path("data/uploaded")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource(show_spinner=False)
def get_vector_store() -> RecipeVectorStore:
    return RecipeVectorStore()


def ensure_index_seeded(store: RecipeVectorStore) -> None:
    """On first run, index the bundled sample recipes so the app isn't empty."""
    if store.count() == 0 and SAMPLE_DIR.exists():
        docs = load_directory(SAMPLE_DIR)
        chunks = chunk_documents(docs)
        if chunks:
            store.add_chunks(chunks)


def index_uploaded_files(store: RecipeVectorStore, uploaded_files) -> int:
    added = 0
    for f in uploaded_files:
        dest = UPLOAD_DIR / f.name
        dest.write_bytes(f.getbuffer())
        try:
            doc = load_document(dest)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not read {f.name}: {exc}")
            continue
        chunks = chunk_documents([doc])
        store.add_chunks(chunks)
        added += len(chunks)
    return added


def render_recipe(answer) -> None:
    st.subheader(f"🍽️ {answer.title}")
    if answer.summary:
        st.write(answer.summary)

    meta_cols = st.columns(4)
    meta_cols[0].metric("Servings", answer.servings)
    meta_cols[1].metric("Prep + cook time", f"{answer.prep_time_minutes} min")
    meta_cols[2].metric("Cuisine", answer.cuisine or "—")
    meta_cols[3].metric("Calories / serving*", answer.nutrition_per_serving["calories"])

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.markdown("### 🧂 Ingredients")
        if answer.ingredients:
            for ing in answer.ingredients:
                st.markdown(f"- {ing}")
        else:
            st.write("_No ingredients returned._")

        if answer.substitutions:
            st.markdown("### 🔄 Substitutions")
            for sub in answer.substitutions:
                st.markdown(f"- {sub}")

        st.markdown("### 🛒 Shopping List")
        st.caption("Already-have ingredients are automatically removed.")
        st.markdown(format_as_markdown(answer.shopping_list))

    with col2:
        st.markdown("### 👩‍🍳 Steps")
        if answer.steps:
            for i, step in enumerate(answer.steps, start=1):
                st.markdown(f"**{i}.** {step}")
        else:
            st.write("_No steps returned._")

        st.markdown("### 📊 Estimated Nutrition (per serving)*")
        n = answer.nutrition_per_serving
        nut_cols = st.columns(4)
        nut_cols[0].metric("Calories", n["calories"])
        nut_cols[1].metric("Protein", f"{n['protein_g']} g")
        nut_cols[2].metric("Carbs", f"{n['carbs_g']} g")
        nut_cols[3].metric("Fat", f"{n['fat_g']} g")
        st.caption(
            f"*Rough offline estimate ({n['estimated_from']}), not a verified "
            "nutrition label."
        )

    if answer.notes:
        st.info(answer.notes)

    sources = answer.sources_used or answer.retrieved_sources
    if sources:
        st.caption("📚 Sources referenced: " + ", ".join(sources))


def main() -> None:
    store = get_vector_store()
    ensure_index_seeded(store)

    st.title("🍳 Recipe RAG Agent")
    st.caption(
        "Ask questions about your recipe collection, or request a recipe adapted "
        "to your diet, available ingredients, and time budget."
    )

    with st.sidebar:
        st.header("📚 Knowledge Base")
        st.write(f"**{store.count()}** indexed recipe chunk(s)")

        uploaded_files = st.file_uploader(
            "Upload cookbooks / recipe notes",
            type=[ext.strip(".") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Add to knowledge base"):
            with st.spinner("Indexing uploaded files..."):
                n = index_uploaded_files(store, uploaded_files)
            st.success(f"Indexed {n} new chunk(s).")
            st.rerun()

        if st.button("Reset index to sample recipes only"):
            store.reset()
            ensure_index_seeded(store)
            st.rerun()

        st.divider()
        st.header("🎯 Constraints")
        diet_options = [
            "Vegan", "Vegetarian", "Gluten-free", "Dairy-free",
            "Sugar-free", "Low-carb", "Nut-free", "Halal", "Kosher",
        ]
        dietary = st.multiselect("Dietary restrictions", diet_options)

        available_text = st.text_area(
            "Ingredients you already have (comma-separated)",
            placeholder="e.g. eggs, milk, flour, bananas",
        )
        available = [a.strip() for a in available_text.split(",") if a.strip()]

        max_time = st.slider("Max time (minutes)", 5, 120, 30, step=5)
        cuisine = st.selectbox(
            "Cuisine preference",
            ["No preference", "Italian", "Indian", "Mexican", "Chinese",
             "Thai", "Mediterranean", "American", "Japanese"],
        )
        servings = st.number_input("Servings", min_value=1, max_value=20, value=4)

        st.divider()
        st.caption(
            "Requires an xAI API key (copy `.env.example` to `.env`). "
            "Get one at console.x.ai. "
            "See README.md for setup."
        )

    if "history" not in st.session_state:
        st.session_state.history = []

    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            if entry["role"] == "assistant" and "answer" in entry:
                render_recipe(entry["answer"])
            else:
                st.write(entry["content"])

    query = st.chat_input(
        "Ask e.g. 'How can I make a sugar-free chocolate cake?'"
    )

    if query:
        st.session_state.history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        constraints = UserConstraints(
            dietary_restrictions=dietary,
            available_ingredients=available,
            max_time_minutes=max_time,
            cuisine=None if cuisine == "No preference" else cuisine,
            servings=int(servings),
        )

        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving recipes and generating your answer..."):
                    agent = RecipeRAGAgent(store)
                    answer = agent.ask(query, constraints)
                render_recipe(answer)
                st.session_state.history.append(
                    {"role": "assistant", "content": answer.title, "answer": answer}
                )
            except LLMNotConfigured as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()
