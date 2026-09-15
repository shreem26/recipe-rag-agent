from langflow.custom import Component
from langflow.io import IntInput, MessageTextInput, Output, StrInput
from langflow.schema import Data

from src.chunker import chunk_documents
from src.ingest import load_directory
from src.vector_store import RecipeVectorStore


class RecipeRetriever(Component):
    display_name = "Recipe Retriever"
    description = "Search the local recipe knowledge base and return relevant recipe context."
    icon = "Search"
    name = "RecipeRetriever"

    inputs = [
        MessageTextInput(name="query", display_name="Question", required=True),
        IntInput(name="top_k", display_name="Results", value=4),
        StrInput(name="recipes_directory", display_name="Recipes directory", value="data/sample_recipes"),
    ]
    outputs = [Output(name="context", display_name="Retrieved context", method="retrieve")]

    def retrieve(self) -> Data:
        store = RecipeVectorStore()
        if store.count() == 0:
            store.add_chunks(chunk_documents(load_directory(self.recipes_directory)))
        hits = store.search(self.query, k=self.top_k)
        return Data(data={"query": self.query, "hits": hits})