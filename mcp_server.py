import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from services.vector_store_service import VectorStoreService
from services.document_store_service import DocumentStoreService
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from services.ocr_service import OCRService

from mcp_tools.ingestion_tool import IngestionTool
from mcp_tools.search_tool import SearchTool
from mcp_tools.generative_tool import GenerativeTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing services for FastMCP...")
vector_store_service = VectorStoreService()
document_store_service = DocumentStoreService()
embedding_service_instance = EmbeddingService()
llm_service_instance = LLMService()
ocr_service_instance = OCRService()

ingestion_tool_instance = IngestionTool(
    vector_store=vector_store_service,
    document_store=document_store_service,
    embedding_service=embedding_service_instance,
    ocr_service=ocr_service_instance
)
search_tool_instance = SearchTool(
    vector_store=vector_store_service,
    embedding_service=embedding_service_instance,
    document_store=document_store_service
)
generative_tool_instance = GenerativeTool(
    llm_service=llm_service_instance,
    search_tool=search_tool_instance
)

mcp = FastMCP("content")
logger.info("FastMCP server initialized.")

@mcp.tool()
async def ingest_document(file_path: str, file_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Process and index a document from a local file path for searching.
    Automatically determines file_type if not provided.
    """
    logger.info(f"Tool 'ingest_document' called with file_path: {file_path}, file_type: {file_type}")
    try:
        actual_file_type = file_type
        if not actual_file_type:
            actual_file_type = Path(file_path).suffix.lower().strip('.')
            logger.info(f"Inferred file_type: {actual_file_type}")
        result = await ingestion_tool_instance.process_document(file_path, actual_file_type)
        logger.info(f"Ingestion result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in 'ingest_document' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@mcp.tool()
async def semantic_search(query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Search through indexed content using natural language.
    'filters' can be used to narrow down the search.
    """
    logger.info(f"Tool 'semantic_search' called with query: {query}, top_k: {top_k}, filters: {filters}")
    try:
        results = await search_tool_instance.search(query, top_k, filters)
        return {
            "success": True,
            "query": query,
            "results": [result.to_dict() for result in results],
            "total_results": len(results)
        }
    except Exception as e:
        logger.error(f"Error in 'semantic_search' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e), "results": []}

@mcp.tool()
async def summarize_content(
    content: Optional[str] = None,
    document_id: Optional[str] = None,
    style: str = "concise"
) -> Dict[str, Any]:
    """
    Generate a summary of provided content or a document_id.
    Available styles: concise, detailed, bullet_points, executive.
    """
    logger.info(f"Tool 'summarize_content' called. doc_id: {document_id}, style: {style}, has_content: {content is not None}")
    try:
        text_to_summarize = content
        if document_id and not text_to_summarize:
            doc = await document_store_service.get_document(document_id)
            if not doc:
                return {"success": False, "error": f"Document {document_id} not found"}
            text_to_summarize = doc.content
        if not text_to_summarize:
            return {"success": False, "error": "No content provided for summarization"}
        max_length = 10000
        if len(text_to_summarize) > max_length:
            logger.warning(f"Content for summarization is long ({len(text_to_summarize)} chars), truncating to {max_length}")
            text_to_summarize = text_to_summarize[:max_length] + "..."
        summary = await generative_tool_instance.summarize(text_to_summarize, style)
        return {
            "success": True,
            "summary": summary,
            "original_length": len(text_to_summarize),
            "summary_length": len(summary),
            "style": style
        }
    except Exception as e:
        logger.error(f"Error in 'summarize_content' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@mcp.tool()
async def generate_tags(
    content: Optional[str] = None,
    document_id: Optional[str] = None,
    max_tags: int = 5
) -> Dict[str, Any]:
    """
    Generate relevant tags for content or a document_id.
    Saves tags to document metadata if document_id is provided.
    """
    logger.info(f"Tool 'generate_tags' called. doc_id: {document_id}, max_tags: {max_tags}, has_content: {content is not None}")
    try:
        text_for_tags = content
        if document_id and not text_for_tags:
            doc = await document_store_service.get_document(document_id)
            if not doc:
                return {"success": False, "error": f"Document {document_id} not found"}
            text_for_tags = doc.content
        if not text_for_tags:
            return {"success": False, "error": "No content provided for tag generation"}
        tags = await generative_tool_instance.generate_tags(text_for_tags, max_tags)
        if document_id and tags:
            await document_store_service.update_document_metadata(document_id, {"tags": tags})
            logger.info(f"Tags {tags} saved for document {document_id}")
        return {
            "success": True,
            "tags": tags,
            "content_length": len(text_for_tags),
            "document_id": document_id
        }
    except Exception as e:
        logger.error(f"Error in 'generate_tags' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@mcp.tool()
async def answer_question(question: str, context_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Answer questions using RAG (Retrieval Augmented Generation) over indexed content.
    'context_filter' can be used to narrow down the context search.
    """
    logger.info(f"Tool 'answer_question' called with question: {question}, context_filter: {context_filter}")
    try:
        search_results = await search_tool_instance.search(question, top_k=5, filters=context_filter)
        if not search_results:
            return {
                "success": False,
                "error": "No relevant context found. Please upload relevant documents.",
                "question": question,
                "answer": "I could not find enough information in the documents to answer your question."
            }
        answer = await generative_tool_instance.answer_question(question, search_results)
        return {
            "success": True,
            "question": question,
            "answer": answer,
            "sources": [result.to_dict() for result in search_results],
            "confidence": "high" if len(search_results) >= 3 else "medium"
        }
    except Exception as e:
        logger.error(f"Error in 'answer_question' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

@mcp.tool()
async def list_documents_for_ui(limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    """
    (UI Helper) List documents from the document store.
    Not a standard processing tool, but useful for UI population.
    """
    logger.info(f"Tool 'list_documents_for_ui' called with limit: {limit}, offset: {offset}")
    try:
        documents = await document_store_service.list_documents(limit, offset)
        return {
            "success": True,
            "documents": [doc.to_dict() for doc in documents],
            "total": len(documents)
        }
    except Exception as e:
        logger.error(f"Error in 'list_documents_for_ui' tool: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e), "documents": []}

if __name__ == "__main__":
    logger.info("Starting FastMCP server...")
    asyncio.run(mcp.run())
