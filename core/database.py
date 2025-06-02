# core/database.py

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import config

def init_chroma():
    """
    Initialize a ChromaDB client and collection with an embedding function.
    Uses OpenAI embeddings if API key is available, otherwise a dummy embedding.
    """
    # Initialize Chroma client (in-memory by default)
    client = chromadb.Client(Settings())

    # Determine embedding function
    embedding_fn = None
    try:
        openai_key = config.OPENAI_API_KEY
    except AttributeError:
        openai_key = None

    if openai_key:
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-ada-002"
        )
    else:
        # Dummy embedding: one-dimensional embedding based on text length
        class DummyEmbedding:
            def __call__(self, texts):
                return [[float(len(text))] for text in texts]
        embedding_fn = DummyEmbedding()

    # Create or get collection named "documents"
    collection = client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_fn
    )
    return collection

def add_document(collection, doc_id: str, text: str, tags: list[str], summary: str, source: str):
    """
    Add a document to the ChromaDB collection with metadata.
    """
    metadata = {"tags": tags, "summary": summary, "source": source}
    # Add document (Chroma will generate embeddings using the collection's embedding function)
    collection.add(
        ids=[doc_id],
        documents=[text],
        metadatas=[metadata]
    )

def search_documents(collection, query: str, top_n: int = 5) -> list[dict]:
    """
    Search for semantically similar documents in the collection.
    Returns top N results with their metadata.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_n,
        include=["metadatas", "documents", "distances"]
    )
    hits = []
    # Extract the results from the Chroma query response
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, doc_id in enumerate(ids):
        hit = {
            "id": doc_id,
            "score": distances[i] if i < len(distances) else None,
            "source": metadatas[i].get("source") if i < len(metadatas) else None,
            "tags": metadatas[i].get("tags") if i < len(metadatas) else None,
            "summary": metadatas[i].get("summary") if i < len(metadatas) else None,
            "document": documents[i] if i < len(documents) else None
        }
        hits.append(hit)
    return hits
