# # mcp_tools.py

# from fastmcp import FastMCP
# import core.processing as processing
# import core.ai_enrichment as ai_enrichment
# import core.database as db
# import core.utils as utils

# # Initialize the FastMCP server instance
# mcp = FastMCP(name="IntelligentContentOrganizer")

# # Initialize the ChromaDB collection (shared for all tools)
# collection = db.init_chroma()

# @mcp.tool()
# def process_content(url: str) -> dict:
#     """
#     Process content from a web URL: fetch, enrich, and store.
#     Returns document ID, tags, summary, and source.
#     """
#     content = processing.fetch_web_content(url)
#     text = utils.clean_text(content)
#     tags = ai_enrichment.generate_tags(text) if text else []
#     summary = ai_enrichment.summarize_text(text) if text else ""
#     doc_id = utils.generate_doc_id(url)
#     # Add the document to the database collection
#     db.add_document(collection, doc_id, text, tags, summary, source=url)
#     return {"id": doc_id, "tags": tags, "summary": summary, "source": url}

# @mcp.tool()
# def upload_local_file(file_path: str) -> dict:
#     """
#     Process a local file: parse, enrich, and store.
#     Returns document ID, tags, summary, and source.
#     """
#     content = processing.parse_local_file(file_path)
#     text = utils.clean_text(content)
#     tags = ai_enrichment.generate_tags(text) if text else []
#     summary = ai_enrichment.summarize_text(text) if text else ""
#     doc_id = utils.generate_doc_id(file_path)
#     db.add_document(collection, doc_id, text, tags, summary, source=file_path)
#     return {"id": doc_id, "tags": tags, "summary": summary, "source": file_path}

# @mcp.tool()
# def semantic_search(query: str, top_n: int = 5) -> list:
#     """
#     Search for documents semantically similar to the query.
#     Returns top N results as a list of dictionaries.
#     """
#     results = db.search_documents(collection, query, top_n)
#     return results


from fastmcp import FastMCP
from core.parser import parse_document, parse_url
from core.summarizer import summarize_content, tag_content
from core.storage import add_document, search_documents
from core.agent import answer_question
import json

mcp = FastMCP("IntelligentContentOrganizer_MCP")

@mcp.tool(name="parse_document")
def mcp_parse_document(file_path: str) -> str:
    """
    MCP tool: Parse a document file and return extracted text.
    """
    text = parse_document(file_path)
    return text

@mcp.tool(name="parse_url")
def mcp_parse_url(url: str) -> str:
    """
    MCP tool: Fetch and parse webpage content from a URL.
    """
    text = parse_url(url)
    return text

@mcp.tool(name="summarize")
def mcp_summarize(text: str) -> str:
    """
    MCP tool: Generate a summary of the provided text.
    """
    return summarize_content(text)

@mcp.tool(name="tag")
def mcp_tag(text: str) -> str:
    """
    MCP tool: Generate tags for the provided text (JSON list).
    """
    tags = tag_content(text)
    return json.dumps(tags)

@mcp.tool(name="add_to_db")
def mcp_add_to_db(doc_id: str, text: str, metadata_json: str) -> str:
    """
    MCP tool: Add a document to ChromaDB with given ID and metadata (JSON).
    """
    metadata = json.loads(metadata_json)
    add_document(doc_id, text, metadata)
    return "Document added with ID: " + doc_id

@mcp.tool(name="search_db")
def mcp_search_db(query: str, top_k: int = 5) -> str:
    """
    MCP tool: Search documents using a query (semantic search). Returns JSON results.
    """
    results = search_documents(query, top_k=top_k)
    return json.dumps(results)

@mcp.tool(name="answer_question")
def mcp_answer_question(question: str) -> str:
    """
    MCP tool: Answer a question using the agentic workflow.
    """
    answer = answer_question(question)
    return answer

if __name__ == "__main__":
    # Run the MCP server (streamable HTTP for web integration:contentReference[oaicite:6]{index=6})
    mcp.run(transport="streamable-http", host="0.0.0.0", port=7861, path="/mcp")

