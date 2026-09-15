

# Recipe RAG Agent

A Retrieval-Augmented Generation recipe assistant built with Langflow, Python, and Groq. The system allows users to ask natural-language cooking questions and receive personalized recipe recommendations grounded in uploaded recipe documents.

## Features

- Ingests `.txt`, `.md`, `.pdf`, and `.docx` recipe documents
- Retrieves relevant recipes from a local searchable knowledge base
- Supports dietary restrictions, available ingredients, cuisine, servings, and time limits
- Generates personalized recipe recommendations using Groq
- Provides ingredients, cooking steps, substitutions, nutrition estimates, and shopping lists
- Returns readable Markdown responses instead of raw JSON
- Displays the complete workflow through the Langflow canvas and Playground

## Langflow Components

1. **Chat Input**  
   Accepts the user’s recipe request and preferences.

2. **Recipe Retriever**  
   Searches the local recipe knowledge base using TF-IDF similarity and retrieves relevant recipe content.

3. **Groq Recipe Generator**  
   Uses the `openai/gpt-oss-120b` model through Groq’s OpenAI-compatible API to generate a recipe based on retrieved context.

4. **Recipe Enricher**  
   Adds nutrition estimates and creates a shopping list.

5. **Recipe Formatter**  
   Converts the internal recipe data into a readable Markdown response.

6. **Chat Output**  
   Displays the final recipe to the user in the Langflow Playground.

## Architecture

```text
Recipe Documents
      ↓
Document Ingestion
      ↓
Recipe Chunking
      ↓
TF-IDF Knowledge Index
      ↓
Recipe Retriever
      ↓
Groq Recipe Generator
      ↓
Recipe Enricher
      ↓
Recipe Formatter
      ↓
Chat Output
```

## Technologies Used

- **Platform:** Langflow
- **Language:** Python
- **LLM Provider:** Groq
- **Model:** `openai/gpt-oss-120b`
- **Architecture:** Retrieval-Augmented Generation
- **Agent Type:** Single-agent workflow
- **Retrieval:** TF-IDF and cosine similarity
- **Vector Storage:** Local file-based vector index
- **Document Processing:** PDF, DOCX, Markdown, and text files
- **Output:** Markdown recipe response
- **Additional Libraries:** scikit-learn, OpenAI Python SDK, python-dotenv, pypdf, python-docx

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Add your Groq API key to `.env`:

```env
LLM_API_KEY=your-groq-api-key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=openai/gpt-oss-120b
```

Start Langflow:

```powershell
.\run_langflow.ps1
```

Open the Langflow interface:

```text
http://localhost:7860
```

## Workflow Connections

```text
Chat Input
  → Recipe Retriever
  → Groq Recipe Generator
  → Recipe Enricher
  → Recipe Formatter
  → Chat Output
```

## Example Query

```text
How do I make a gluten-free cookie with berries?
I am vegetarian. Give me a recipe that takes a maximum of 30 minutes.
```

## Expected Output

The system generates a readable recipe containing:

- Recipe title
- Summary
- Servings and preparation time
- Ingredients
- Step-by-step instructions
- Ingredient substitutions
- Nutrition estimate
- Shopping list
- Source recipe references

## Limitations

- Retrieval currently uses TF-IDF rather than semantic embeddings.
- Nutrition values are estimates and should not be considered medical advice.
- The system depends on the quality and completeness of the uploaded documents.
- The local vector index is intended for small and medium-sized recipe collections.
- Allergy information should always be verified using product labels and preparation conditions.

## Future Scope

- Replace TF-IDF with semantic embeddings.
- Add a production vector database such as ChromaDB, FAISS, Pinecone, or Qdrant.
- Support voice, image, and scanned-document inputs.
- Add user profiles for allergies, preferences, and meal history.
- Add weekly meal planning and budget-aware grocery lists.
- Add multilingual recipe support.
- Add specialized agents for nutrition, substitutions, meal planning, and allergy checking.
- Deploy the Langflow workflow as a cloud-hosted application or API.
