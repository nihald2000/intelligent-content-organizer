import gradio as gr
import asyncio
from pathlib import Path
import tempfile
import json
from typing import List, Dict, Any
import logging

from config import Config
from mcp_server import mcp
# Handle imports based on how the app is run
try:
    from mcp_server import mcp
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️ MCP server not available, running in standalone mode")

import mcp_tools

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate configuration on startup
try:
    Config.validate()
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    print(f"⚠️ Configuration error: {e}")
    print("Please set the required API keys in your environment variables or .env file")

# Global state for search results
current_results = []

async def process_file_handler(file):
    """Handle file upload and processing"""
    if file is None:
        return "Please upload a file", "", "", None
    
    try:
        # Process the file
        result = await mcp_tools.process_local_file(file.name)
        
        if result.get("success"):
            tags_display = ", ".join(result["tags"])
            return (
                f"✅ Successfully processed: {result['file_name']}",
                result["summary"],
                tags_display,
                gr.update(visible=True, value=create_result_card(result))
            )
        else:
            return f"❌ Error: {result.get('error', 'Unknown error')}", "", "", None
            
    except Exception as e:
        logger.error(f"Error in file handler: {str(e)}")
        return f"❌ Error: {str(e)}", "", "", None

async def process_url_handler(url):
    """Handle URL processing"""
    if not url:
        return "Please enter a URL", "", "", None
    
    try:
        # Process the URL
        result = await mcp_tools.process_web_content(url)
        
        if result.get("success"):
            tags_display = ", ".join(result["tags"])
            return (
                f"✅ Successfully processed: {url}",
                result["summary"],
                tags_display,
                gr.update(visible=True, value=create_result_card(result))
            )
        else:
            return f"❌ Error: {result.get('error', 'Unknown error')}", "", "", None
            
    except Exception as e:
        logger.error(f"Error in URL handler: {str(e)}")
        return f"❌ Error: {str(e)}", "", "", None

async def search_handler(query):
    """Handle semantic search"""
    if not query:
        return [], "Please enter a search query"
    
    try:
        # Perform search
        results = await mcp_tools.search_knowledge_base(query, limit=10)
        
        if results:
            # Create display cards for each result
            result_cards = []
            for result in results:
                card = f"""
                ### 📄 {result.get('source', 'Unknown Source')}
                **Tags:** {', '.join(result.get('tags', []))}
                
                **Summary:** {result.get('summary', 'No summary available')}
                
                **Relevance:** {result.get('relevance_score', 0):.2%}
                
                ---
                """
                result_cards.append(card)
            
            global current_results
            current_results = results
            
            return result_cards, f"Found {len(results)} results"
        else:
            return [], "No results found"
            
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return [], f"Error: {str(e)}"

def create_result_card(result: Dict[str, Any]) -> str:
    """Create a formatted result card"""
    return f"""
    ### 📋 Processing Complete
    
    **Document ID:** {result.get('doc_id', 'N/A')}
    
    **Source:** {result.get('file_name', result.get('url', 'Unknown'))}
    
    **Tags:** {', '.join(result.get('tags', []))}
    
    **Summary:** {result.get('summary', 'No summary available')}
    
    **Chunks Processed:** {result.get('chunks_processed', 0)}
    """

# Create Gradio interface
with gr.Blocks(title="Intelligent Content Organizer - MCP Agent") as demo:
    gr.Markdown("""
    # 🧠 Intelligent Content Organizer
    ### MCP-Powered Knowledge Management System
    
    This AI-driven system automatically organizes, enriches, and retrieves your digital content.
    Upload files or provide URLs to build your personal knowledge base with automatic tagging and semantic search.
    
    ---
    """)
    
    with gr.Tabs():
        # File Processing Tab
        with gr.TabItem("📁 Process Files"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload Document",
                        file_types=[".pdf", ".txt", ".docx", ".doc", ".html", ".md", ".csv", ".json"]
                    )
                    file_process_btn = gr.Button("Process File", variant="primary")
                
                with gr.Column():
                    file_status = gr.Textbox(label="Status", lines=1)
                    file_summary = gr.Textbox(label="Generated Summary", lines=3)
                    file_tags = gr.Textbox(label="Generated Tags", lines=1)
            
            file_result = gr.Markdown(visible=False)
        
        # URL Processing Tab
        with gr.TabItem("🌐 Process URLs"):
            with gr.Row():
                with gr.Column():
                    url_input = gr.Textbox(
                        label="Enter URL",
                        placeholder="https://example.com/article"
                    )
                    url_process_btn = gr.Button("Process URL", variant="primary")
                
                with gr.Column():
                    url_status = gr.Textbox(label="Status", lines=1)
                    url_summary = gr.Textbox(label="Generated Summary", lines=3)
                    url_tags = gr.Textbox(label="Generated Tags", lines=1)
            
            url_result = gr.Markdown(visible=False)
        
        # Search Tab
        with gr.TabItem("🔍 Semantic Search"):
            search_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter your search query...",
                lines=1
            )
            search_btn = gr.Button("Search", variant="primary")
            search_status = gr.Textbox(label="Status", lines=1)
            
            search_results = gr.Markdown(label="Search Results")
        
        # MCP Server Info Tab
        with gr.TabItem("ℹ️ MCP Server Info"):
            gr.Markdown("""
            ### MCP Server Configuration
            
            This Gradio app also functions as an MCP (Model Context Protocol) server, allowing integration with:
            - Claude Desktop
            - Cursor
            - Other MCP-compatible clients
            
            **Server Name:** intelligent-content-organizer
            
            **Available Tools:**
            - `process_file`: Process local files and extract content
            - `process_url`: Fetch and process web content
            - `semantic_search`: Search across stored documents
            - `get_document_summary`: Get detailed document information
            
            **To use as MCP server:**
            1. Add this server to your MCP client configuration
            2. Use the tools listed above to interact with your knowledge base
            3. All processed content is automatically indexed for semantic search
            
            **Tags:** mcp-server-track
            """)
    
    # Event handlers
    file_process_btn.click(
        fn=lambda x: asyncio.run(process_file_handler(x)),
        inputs=[file_input],
        outputs=[file_status, file_summary, file_tags, file_result]
    )
    
    url_process_btn.click(
        fn=lambda x: asyncio.run(process_url_handler(x)),
        inputs=[url_input],
        outputs=[url_status, url_summary, url_tags, url_result]
    )
    
    search_btn.click(
        fn=lambda x: asyncio.run(search_handler(x)),
        inputs=[search_input],
        outputs=[search_results, search_status]
    )

# Launch configuration
if __name__ == "__main__":
    # Check if running as MCP server
    import sys
    if "--mcp" in sys.argv:
        # Run as MCP server
        import asyncio
        asyncio.run(mcp.run())
    else:
        # Run as Gradio app
        demo.launch(
            server_name="0.0.0.0",
            share=False,
            show_error=True
        )