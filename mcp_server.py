from mcp.server.fastmcp import FastMCP
import json
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("intelligent-content-organizer")

@mcp.tool()
async def process_file(file_path: str) -> Dict[str, Any]:
    """
    Process a local file and extract content, generate tags, and create embeddings
    """
    try:
        from mcp_tools import process_local_file
        result = await process_local_file(file_path)
        return result
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def process_url(url: str) -> Dict[str, Any]:
    """
    Fetch and process content from a URL
    """
    try:
        from mcp_tools import process_web_content
        result = await process_web_content(url)
        return result
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def semantic_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Perform semantic search across stored documents
    """
    try:
        from mcp_tools import search_knowledge_base
        results = await search_knowledge_base(query, limit)
        return results
    except Exception as e:
        logger.error(f"Error performing search: {str(e)}")
        return [{"error": str(e)}]

@mcp.tool()
async def get_document_summary(doc_id: str) -> Dict[str, Any]:
    """
    Get summary and metadata for a specific document
    """
    try:
        from mcp_tools import get_document_details
        result = await get_document_details(doc_id)
        return result
    except Exception as e:
        logger.error(f"Error getting document summary: {str(e)}")
        return {"error": str(e)}

@mcp.tool()
async def get_server_info() -> Dict[str, Any]:
    """
    Get information about this MCP server
    """
    return {
        "name": "Intelligent Content Organizer",
        "version": "1.0.0",
        "description": "AI-powered knowledge management system with automatic tagging and semantic search",
        "capabilities": [
            "File processing (20+ formats)",
            "Web content extraction",
            "Automatic tagging",
            "Semantic search",
            "Document summarization"
        ],
        "tools": [
            {
                "name": "process_file",
                "description": "Process local files and extract content"
            },
            {
                "name": "process_url",
                "description": "Fetch and process web content"
            },
            {
                "name": "semantic_search",
                "description": "Search across stored documents"
            },
            {
                "name": "get_document_summary",
                "description": "Get document details"
            },
            {
                "name": "get_server_info",
                "description": "Get server information"
            }
        ]
    }

if __name__ == "__main__":
    # Run the MCP server
    import asyncio
    asyncio.run(mcp.run())
