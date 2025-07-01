import json
from core.storage import search_documents
# For Q&A we can use a simple retrieval + QA pipeline (stubbed here)
# In a real app, you might use LangChain or a HuggingFace question-answering model.

def answer_question(question: str) -> str:
    """
    Agent: retrieve relevant docs and answer the question.
    """
    # Retrieve top documents
    results = search_documents(question, top_k=3)
    doc_texts = results.get("documents", [[]])[0]
    combined = " ".join(doc_texts)
    # Stub: just echo the question and number of docs
    if not combined.strip():
        return "No relevant documents found."
    return f"Answered question: '{question}' (based on {len(doc_texts)} documents)."
