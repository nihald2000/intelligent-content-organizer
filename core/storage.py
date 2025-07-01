import chromadb
import os
from mistralai import Mistral
import config

# Initialize ChromaDB client (persistent directory can be set via CHROMA_DB_DIR)
chroma_db_path = os.getenv("CHROMA_DB_DIR", "db/")
client = chromadb.Client()
collection = client.get_or_create_collection("documents")

# Use Mistral API for embeddings

def get_mistral_embedding(text: str) -> list[float]:
    """
    Get embedding for the given text using Mistral API.
    """
    with Mistral(api_key=config.MISTRAL_API_KEY) as client:
        response = client.embeddings.create(
            model="mistral-embed",
            input=text
        )
        # The API returns a list of embeddings (one per input)
        return response['data'][0]['embedding']


def add_document(doc_id: str, text: str, metadata: dict):
    """
    Add a document's text and metadata to the ChromaDB collection.
    """
    embedding = get_mistral_embedding(text)
    collection.add(ids=[doc_id], embeddings=[embedding], documents=[text], metadatas=[metadata])
    # Persist to disk
    client.persist()
    return True


def search_documents(query: str, top_k: int = 5) -> dict:
    """
    Search for documents semantically similar to the query.
    Returns a dictionary of top results.
    """
    query_vec = get_mistral_embedding(query)
    results = collection.query(query_embeddings=[query_vec], n_results=top_k,
                               include=['ids','distances','documents','metadatas'])
    return results


def get_all_documents() -> list:
    """
    Retrieve metadata for all documents in the collection.
    """
    all_ids = collection.get()['ids']
    docs = []
    for doc_id in all_ids:
        res = collection.get(ids=[doc_id])
        if res and res['metadatas']:
            docs.append({"id": doc_id, "metadata": res['metadatas'][0]})
    return docs
