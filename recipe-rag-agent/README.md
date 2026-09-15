# 🍳 Recipe RAG Agent

A Document Q&A RAG (Retrieval-Augmented Generation) agent that acts as an
intelligent recipe generator. Ask natural-language cooking questions, and it
retrieves relevant recipes from your own document collection, adapts them to
your dietary needs / available ingredients / time budget, and returns a
structured recipe with substitutions, a nutrition estimate, and a shopping
list.

Built for **Problem Statement No. 8: Document Q&A RAG Agent for Recipe
Generator**.

## What it does

| Requirement | How it's implemented |
|---|---|
| **Ingest & index recipe documents** | `src/ingest.py` loads `.txt`, `.md`, `.pdf`, `.docx` files. `src/chunker.py` splits them into individual recipe-sized chunks by detecting "Ingredients" headers, so a whole recipe stays together. |
| **Searchable knowledge base** | `src/vector_store.py` indexes chunks with TF-IDF + cosine similarity — fully offline, no model downloads, instant startup. |
| **Conversational Q&A** | `app.py` is a Streamlit chat UI. Ask things like *"How can I make a sugar-free version of chocolate cake?"* and get a precise, recipe-aware answer. |
| **Personalized & agentic layer** | `src/recipe_agent.py` retrieves context, then prompts the LLM with your constraints (diet, ingredients on hand, time limit, cuisine) and forces a structured JSON response. |
| **Step-by-step guidance** | The UI renders numbered steps, ingredient list, and substitution suggestions. |
| **Nutritional facts** | `src/nutrition.py` estimates calories/protein/carbs/fat per serving from an offline ingredient lookup table (labeled as an estimate, not a certified value). |
| **Shopping list** | `src/shopping_list.py` dedupes the recipe's ingredients and removes anything you said you already have. |

## Architecture

```
User question + constraints
        │
        ▼
 ┌──────────────┐   TF-IDF search   ┌──────────────────┐
 │ Vector Store │ ────────────────► │ Top-k recipe      │
 │ (chunks.pkl) │                   │ chunks (context)   │
 └──────────────┘                   └──────────────────┘
                                              │
                                              ▼
                                   ┌─────────────────────┐
                                   │ LLM (xAI Grok / any    │
                                   │ OpenAI-compatible API) │
                                   │ + structured JSON      │
                                   └─────────────────────┘
                                              │
                                              ▼
                        ┌─────────────────────────────────────┐
                        │ Local post-processing (deterministic) │
                        │  • nutrition.py  → calorie estimate    │
                        │  • shopping_list.py → checklist        │
                        └─────────────────────────────────────┘
                                              │
                                              ▼
                                 Structured recipe card (Streamlit)
```

Nutrition and shopping-list logic run **locally**, not inside the LLM prompt —
that keeps the numbers deterministic and reproducible instead of letting the
model "guess" a plausible-sounding shopping list every time.

## Setup

### 1. Get an xAI API key

This project uses **xAI Grok** by default through xAI's OpenAI-compatible API.

1. Go to [console.x.ai](https://console.x.ai/)
2. Create or copy your xAI API key

### 2. Install and run

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your key
cp .env.example .env
# then edit .env and paste your xAI key into LLM_API_KEY

# Run the app
streamlit run app.py
```

The app opens in your browser. On first launch it automatically indexes the
6 sample recipes bundled in `data/sample_recipes/` so you have something to
query immediately.

### Switching providers

The LLM layer is provider-agnostic — it talks to any OpenAI-compatible API.
To switch, just change the variables in `.env` (examples are pre-written in
`.env.example`):

| Provider | Cost | Notes |
|---|---|---|
| **xAI Grok** (default) | Usage-based | Uses `grok-3-mini` through the xAI API. |
| **OpenRouter** | Free tier available | Look for models with a `:free` suffix. |
| **Ollama** | Free forever | Runs entirely on your own machine, works offline, no key. Install from [ollama.com](https://ollama.com), then `ollama pull llama3.1`. |
| **OpenAI / Anthropic** | Paid | Use if you already have credits. |

No code changes are needed for any of these — only `.env`.

## Using your own recipes

You can either:

- **Upload files in the app** — use the "Upload cookbooks / recipe notes"
  control in the sidebar, then click "Add to knowledge base".
- **Add files and rebuild via CLI**:
  ```bash
  # Drop .txt / .md / .pdf / .docx files into a folder, then:
  python scripts/build_index.py path/to/your/recipes --reset
  ```

## Example questions to try

- "How can I make a sugar-free version of chocolate cake?"
- "I have eggs, milk, and bananas — what can I make in 20 minutes?"
- "Give me a gluten-free version of the vegetable stir-fry."
- "What's a high-protein dinner I can make with lentils?"
- "Adapt the tikka masala recipe to be dairy-free."

## Project structure

```
recipe-rag-agent/
├── app.py                    # Streamlit UI (main entry point)
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py              # Load .txt/.md/.pdf/.docx documents
│   ├── chunker.py             # Recipe-aware chunking
│   ├── vector_store.py        # TF-IDF vector store (index + search)
│   ├── llm.py                 # LLM wrapper (xAI Grok / any OpenAI-compatible API)
│   ├── recipe_agent.py        # RAG orchestration + personalization
│   ├── nutrition.py           # Offline nutrition estimator
│   └── shopping_list.py       # Shopping list generation
├── scripts/
│   └── build_index.py         # CLI to (re)build the index
├── data/
│   └── sample_recipes/        # 6 bundled sample recipes
└── tests/
    └── test_basic.py          # Tests for the non-LLM parts of the pipeline
```

## Running tests

The deterministic parts of the pipeline (ingestion, chunking, nutrition,
shopping list) are covered by tests that don't require an API key:

```bash
pip install pytest
python -m pytest tests/ -v
```

## Design notes / known limitations

- **Retrieval is TF-IDF-based**, not a neural embedding model. This keeps the
  project 100% offline and dependency-light, and works well for recipe
  search since queries and documents share a lot of literal vocabulary
  (ingredient names, diet terms). For much larger or more paraphrase-heavy
  collections, swap `src/vector_store.py`'s internals for a semantic
  embedding backend (e.g. Voyage AI embeddings, OpenAI embeddings, or
  `sentence-transformers`) — the `add_chunks()` / `search()` interface is
  designed to make that a self-contained change.
- **Nutrition estimates are approximate.** They come from an offline lookup
  table of ~40 common ingredients with rough unit-to-gram conversions, meant
  for general guidance, not medical or clinical use.
- **JSON parsing has a fallback.** If the model's response cannot be parsed as
  JSON (rare, but possible), the raw text is surfaced in the "notes" field
  so nothing is silently lost.
