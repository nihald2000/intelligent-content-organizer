import gradio as gr
import os
import asyncio
import json
import logging
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import nest_asyncio

# Apply nest_asyncio to handle nested event loops in Gradio
nest_asyncio.apply()

# Import our custom modules
from mcp_tools.ingestion_tool import IngestionTool
from mcp_tools.search_tool import SearchTool
from mcp_tools.generative_tool import GenerativeTool
from services.vector_store_service import VectorStoreService
from services.document_store_service import DocumentStoreService
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from services.ocr_service import OCRService
from core.models import SearchResult, Document
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentOrganizerMCPServer:
    def __init__(self):
        # Initialize services
        logger.info("Initializing Content Organizer MCP Server...")
        
        self.vector_store = VectorStoreService()
        self.document_store = DocumentStoreService()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.ocr_service = OCRService()
        
        # Initialize tools
        self.ingestion_tool = IngestionTool(
            vector_store=self.vector_store,
            document_store=self.document_store,
            embedding_service=self.embedding_service,
            ocr_service=self.ocr_service
        )
        self.search_tool = SearchTool(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            document_store=self.document_store
        )
        self.generative_tool = GenerativeTool(
            llm_service=self.llm_service,
            search_tool=self.search_tool
        )
        
        # Track processing status
        self.processing_status = {}
        
        # Document cache for quick access
        self.document_cache = {}
        
        logger.info("Content Organizer MCP Server initialized successfully!")

    def run_async(self, coro):
        """Helper to run async functions in Gradio"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # If loop is already running, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)

    async def ingest_document_async(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """MCP Tool: Ingest and process a document"""
        try:
            task_id = str(uuid.uuid4())
            self.processing_status[task_id] = {"status": "processing", "progress": 0}
            
            result = await self.ingestion_tool.process_document(file_path, file_type, task_id)
            
            if result.get("success"):
                self.processing_status[task_id] = {"status": "completed", "progress": 100}
                # Update document cache
                doc_id = result.get("document_id")
                if doc_id:
                    doc = await self.document_store.get_document(doc_id)
                    if doc:
                        self.document_cache[doc_id] = doc
                
                return result
            else:
                self.processing_status[task_id] = {"status": "failed", "error": result.get("error")}
                return result
                
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to process document"
            }
    
    async def get_document_content_async(self, document_id: str) -> Optional[str]:
        """Get document content by ID"""
        try:
            # Check cache first
            if document_id in self.document_cache:
                return self.document_cache[document_id].content
            
            # Get from store
            doc = await self.document_store.get_document(document_id)
            if doc:
                self.document_cache[document_id] = doc
                return doc.content
            
            return None
        except Exception as e:
            logger.error(f"Error getting document content: {str(e)}")
            return None

    async def semantic_search_async(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """MCP Tool: Perform semantic search"""
        try:
            results = await self.search_tool.search(query, top_k, filters)
            return {
                "success": True,
                "query": query,
                "results": [result.to_dict() for result in results],
                "total_results": len(results)
            }
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "results": []
            }

    async def summarize_content_async(self, content: str = None, document_id: str = None, style: str = "concise") -> Dict[str, Any]:
        """MCP Tool: Summarize content or document"""
        try:
            # If document_id provided, get content from document
            if document_id and document_id != "none":
                content = await self.get_document_content_async(document_id)
                if not content:
                    return {"success": False, "error": f"Document {document_id} not found"}
            
            if not content or not content.strip():
                return {"success": False, "error": "No content provided for summarization"}
            
            # Truncate content if too long (for API limits)
            max_content_length = 4000
            if len(content) > max_content_length:
                content = content[:max_content_length] + "..."
            
            summary = await self.generative_tool.summarize(content, style)
            return {
                "success": True,
                "summary": summary,
                "original_length": len(content),
                "summary_length": len(summary),
                "style": style,
                "document_id": document_id
            }
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def generate_tags_async(self, content: str = None, document_id: str = None, max_tags: int = 5) -> Dict[str, Any]:
        """MCP Tool: Generate tags for content"""
        try:
            # If document_id provided, get content from document
            if document_id and document_id != "none":
                content = await self.get_document_content_async(document_id)
                if not content:
                    return {"success": False, "error": f"Document {document_id} not found"}
            
            if not content or not content.strip():
                return {"success": False, "error": "No content provided for tag generation"}
            
            tags = await self.generative_tool.generate_tags(content, max_tags)
            
            # Update document tags if document_id provided
            if document_id and document_id != "none" and tags:
                await self.document_store.update_document_metadata(document_id, {"tags": tags})
            
            return {
                "success": True,
                "tags": tags,
                "content_length": len(content),
                "document_id": document_id
            }
        except Exception as e:
            logger.error(f"Tag generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def answer_question_async(self, question: str, context_filter: Optional[Dict] = None) -> Dict[str, Any]:
        """MCP Tool: Answer questions using RAG"""
        try:
            # Search for relevant context
            search_results = await self.search_tool.search(question, top_k=5, filters=context_filter)
            
            if not search_results:
                return {
                    "success": False,
                    "error": "No relevant context found in your documents. Please make sure you have uploaded relevant documents.",
                    "question": question
                }
            
            # Generate answer using context
            answer = await self.generative_tool.answer_question(question, search_results)
            
            return {
                "success": True,
                "question": question,
                "answer": answer,
                "sources": [result.to_dict() for result in search_results],
                "confidence": "high" if len(search_results) >= 3 else "medium"
            }
        except Exception as e:
            logger.error(f"Question answering failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "question": question
            }

    def list_documents_sync(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """List stored documents"""
        try:
            documents = self.run_async(self.document_store.list_documents(limit, offset))
            return {
                "success": True,
                "documents": [doc.to_dict() for doc in documents],
                "total": len(documents)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Initialize the MCP server
mcp_server = ContentOrganizerMCPServer()

# Helper functions
def get_document_list():
    """Get list of documents for display"""
    try:
        result = mcp_server.list_documents_sync(limit=100)
        if result["success"]:
            if result["documents"]:
                doc_list = "📚 Documents in Library:\n\n"
                for i, doc in enumerate(result["documents"], 1):
                    doc_list += f"{i}. {doc['filename']} (ID: {doc['id'][:8]}...)\n"
                    doc_list += f"   Type: {doc['doc_type']}, Size: {doc['file_size']} bytes\n"
                    if doc.get('tags'):
                        doc_list += f"   Tags: {', '.join(doc['tags'])}\n"
                    doc_list += f"   Created: {doc['created_at'][:10]}\n\n"
                return doc_list
            else:
                return "No documents in library yet. Upload some documents to get started!"
        else:
            return f"Error loading documents: {result['error']}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_document_choices():
    """Get document choices for dropdown"""
    try:
        result = mcp_server.list_documents_sync(limit=100)
        if result["success"] and result["documents"]:
            choices = []
            for doc in result["documents"]:
                # Create label with filename and shortened ID
                choice_label = f"{doc['filename']} ({doc['id'][:8]}...)"
                # Use full document ID as the value
                choices.append((choice_label, doc['id']))
            
            logger.info(f"Generated {len(choices)} document choices")
            return choices
        return []
    except Exception as e:
        logger.error(f"Error getting document choices: {str(e)}")
        return []

# Gradio Interface Functions
def upload_and_process_file(file):
    """Gradio interface for file upload"""
    if file is None:
        return "No file uploaded", "", get_document_list(), gr.update(choices=get_document_choices())
    
    try:
        # Get file path
        file_path = file.name if hasattr(file, 'name') else str(file)
        file_type = Path(file_path).suffix.lower()
        
        logger.info(f"Processing file: {file_path}")
        
        # Process document
        result = mcp_server.run_async(mcp_server.ingest_document_async(file_path, file_type))
        
        if result["success"]:
            # Get updated document list and choices
            doc_list = get_document_list()
            doc_choices = get_document_choices()
            
            return (
                f"✅ Success: {result['message']}\nDocument ID: {result['document_id']}\nChunks created: {result['chunks_created']}", 
                result["document_id"],
                doc_list,
                gr.update(choices=doc_choices),
                gr.update(choices=doc_choices),
                gr.update(choices=doc_choices)
            )
        else:
            return (
                f"❌ Error: {result.get('error', 'Unknown error')}", 
                "", 
                get_document_list(),
                gr.update(choices=get_document_choices()),
                gr.update(choices=get_document_choices()),
                gr.update(choices=get_document_choices())
            )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return (
            f"❌ Error: {str(e)}", 
            "", 
            get_document_list(),
            gr.update(choices=get_document_choices()),
            gr.update(choices=get_document_choices()),
            gr.update(choices=get_document_choices())
        )

def perform_search(query, top_k):
    """Gradio interface for search"""
    if not query.strip():
        return "Please enter a search query"
    
    try:
        result = mcp_server.run_async(mcp_server.semantic_search_async(query, int(top_k)))
        
        if result["success"]:
            if result["results"]:
                output = f"🔍 Found {result['total_results']} results for: '{query}'\n\n"
                for i, res in enumerate(result["results"], 1):
                    output += f"Result {i}:\n"
                    output += f"📊 Relevance Score: {res['score']:.3f}\n"
                    output += f"📄 Content: {res['content'][:300]}...\n"
                    if 'document_filename' in res.get('metadata', {}):
                        output += f"📁 Source: {res['metadata']['document_filename']}\n"
                    output += f"🔗 Document ID: {res.get('document_id', 'Unknown')}\n"
                    output += "-" * 80 + "\n\n"
                return output
            else:
                return f"No results found for: '{query}'\n\nMake sure you have uploaded relevant documents first."
        else:
            return f"❌ Search failed: {result['error']}"
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return f"❌ Error: {str(e)}"

def summarize_document(doc_choice, custom_text, style):
    """Gradio interface for summarization"""
    try:
        # Debug logging
        logger.info(f"Summarize called with doc_choice: {doc_choice}, type: {type(doc_choice)}")
        
        # Get document ID from dropdown choice
        document_id = None
        if doc_choice and doc_choice != "none" and doc_choice != "":
            # When Gradio dropdown returns a choice, it returns the value part of the (label, value) tuple
            document_id = doc_choice
            logger.info(f"Using document ID: {document_id}")
        
        # Use custom text if provided, otherwise use document
        if custom_text and custom_text.strip():
            logger.info("Using custom text for summarization")
            result = mcp_server.run_async(mcp_server.summarize_content_async(content=custom_text, style=style))
        elif document_id:
            logger.info(f"Summarizing document: {document_id}")
            result = mcp_server.run_async(mcp_server.summarize_content_async(document_id=document_id, style=style))
        else:
            return "Please select a document from the dropdown or enter text to summarize"
        
        if result["success"]:
            output = f"📝 Summary ({style} style):\n\n{result['summary']}\n\n"
            output += f"📊 Statistics:\n"
            output += f"- Original length: {result['original_length']} characters\n"
            output += f"- Summary length: {result['summary_length']} characters\n"
            output += f"- Compression ratio: {(1 - result['summary_length']/result['original_length'])*100:.1f}%\n"
            if result.get('document_id'):
                output += f"- Document ID: {result['document_id']}\n"
            return output
        else:
            return f"❌ Summarization failed: {result['error']}"
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return f"❌ Error: {str(e)}"

def generate_tags_for_document(doc_choice, custom_text, max_tags):
    """Gradio interface for tag generation"""
    try:
        # Debug logging
        logger.info(f"Generate tags called with doc_choice: {doc_choice}, type: {type(doc_choice)}")
        
        # Get document ID from dropdown choice
        document_id = None
        if doc_choice and doc_choice != "none" and doc_choice != "":
            # When Gradio dropdown returns a choice, it returns the value part of the (label, value) tuple
            document_id = doc_choice
            logger.info(f"Using document ID: {document_id}")
        
        # Use custom text if provided, otherwise use document
        if custom_text and custom_text.strip():
            logger.info("Using custom text for tag generation")
            result = mcp_server.run_async(mcp_server.generate_tags_async(content=custom_text, max_tags=int(max_tags)))
        elif document_id:
            logger.info(f"Generating tags for document: {document_id}")
            result = mcp_server.run_async(mcp_server.generate_tags_async(document_id=document_id, max_tags=int(max_tags)))
        else:
            return "Please select a document from the dropdown or enter text to generate tags"
        
        if result["success"]:
            tags_str = ", ".join(result["tags"])
            output = f"🏷️ Generated Tags:\n\n{tags_str}\n\n"
            output += f"📊 Statistics:\n"
            output += f"- Content length: {result['content_length']} characters\n"
            output += f"- Number of tags: {len(result['tags'])}\n"
            if result.get('document_id'):
                output += f"- Document ID: {result['document_id']}\n"
                output += f"\n✅ Tags have been saved to the document."
            return output
        else:
            return f"❌ Tag generation failed: {result['error']}"
    except Exception as e:
        logger.error(f"Tag generation error: {str(e)}")
        return f"❌ Error: {str(e)}"

def ask_question(question):
    """Gradio interface for Q&A"""
    if not question.strip():
        return "Please enter a question"
    
    try:
        result = mcp_server.run_async(mcp_server.answer_question_async(question))
        
        if result["success"]:
            output = f"❓ Question: {result['question']}\n\n"
            output += f"💡 Answer:\n{result['answer']}\n\n"
            output += f"🎯 Confidence: {result['confidence']}\n\n"
            output += f"📚 Sources Used ({len(result['sources'])}):\n"
            for i, source in enumerate(result['sources'], 1):
                filename = source.get('metadata', {}).get('document_filename', 'Unknown')
                output += f"\n{i}. 📄 {filename}\n"
                output += f"   📝 Excerpt: {source['content'][:150]}...\n"
                output += f"   📊 Relevance: {source['score']:.3f}\n"
            return output
        else:
            return f"❌ {result.get('error', 'Failed to answer question')}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def delete_document_from_library(document_id):
        """deleting a document from the library"""
        try:
            # Run the async delete_document method
            result = mcp_server.run_async(mcp_server.document_store.delete_document(document_id))
            if result:
                msg = f"🗑️ Document {document_id[:8]}... deleted successfully."
            else:
                msg = f"❌ Failed to delete document {document_id[:8]}..."
            # Refresh document list and choices
            doc_list = get_document_list()
            doc_choices = get_document_choices()
            return msg, doc_list, gr.update(choices=doc_choices)
        except Exception as e:
            return f"❌ Error: {str(e)}", get_document_list(), gr.update(choices=get_document_choices())

# Create Gradio Interface
# Create Gradio Interface
def create_gradio_interface():
    with gr.Blocks(title="🧠 Intelligent Content Organizer MCP Agent", theme=gr.themes.Soft()) as interface:
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                # 🧠 Intelligent Content Organizer MCP Agent

                A powerful MCP (Model Context Protocol) server for intelligent content management with semantic search, 
                summarization, and Q&A capabilities powered by Anthropic Claude and Mistral AI.

                ## 🚀 Quick Start:
                1. **Upload Documents** → Go to "📄 Upload Documents" tab  
                2. **Search Your Content** → Use "🔍 Search Documents" to find information  
                3. **Get Summaries** → Select any document in "📝 Summarize" tab  
                4. **Ask Questions** → Get answers from your documents in "❓ Ask Questions" tab  
                """)

        with gr.Tabs():
            # 📚 Document Library Tab
            with gr.Tab("📚 Document Library"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Your Document Collection")
                        document_list = gr.Textbox(
                            label="Documents in Library",
                            value="",  # leave empty, filled on load
                            lines=20,
                            interactive=False
                        )
                        refresh_btn = gr.Button("🔄 Refresh Library", variant="secondary")

                        delete_doc_dropdown = gr.Dropdown(
                            label="Select Document to Delete",
                            choices=get_document_choices(),  # initially empty
                            value=None,
                            interactive=True,
                            allow_custom_value=False
                        )
                        delete_btn = gr.Button("🗑️ Delete Selected Document", variant="stop")

                refresh_btn.click(
                    fn=get_document_list,
                    outputs=[document_list]
                )

                delete_btn.click(
                    delete_document_from_library,
                    inputs=[delete_doc_dropdown],
                    outputs=[document_list, delete_doc_dropdown]
                )

            # 📄 Upload Documents Tab
            with gr.Tab("📄 Upload Documents"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Add Documents to Your Library")
                        file_input = gr.File(
                            label="Select Document to Upload",
                            file_types=[".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"],
                            type="filepath"
                        )
                        upload_btn = gr.Button("🚀 Process & Add to Library", variant="primary", size="lg")
                    with gr.Column():
                        upload_output = gr.Textbox(
                            label="Processing Result",
                            lines=6,
                            placeholder="Upload a document to see processing results..."
                        )
                        doc_id_output = gr.Textbox(
                            label="Document ID",
                            placeholder="Document ID will appear here after processing..."
                        )

                upload_btn.click(
                    upload_and_process_file,
                    inputs=[file_input],
                    outputs=[upload_output, doc_id_output]
                )

            # 🔍 Search Documents Tab
            with gr.Tab("🔍 Search Documents"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Search Your Document Library")
                        search_query = gr.Textbox(
                            label="What are you looking for?",
                            placeholder="Enter your search query...",
                            lines=2
                        )
                        search_top_k = gr.Slider(
                            label="Number of Results",
                            minimum=1,
                            maximum=20,
                            value=5,
                            step=1
                        )
                        search_btn = gr.Button("🔍 Search Library", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        search_output = gr.Textbox(
                            label="Search Results",
                            lines=20,
                            placeholder="Search results will appear here..."
                        )

                search_btn.click(
                    perform_search,
                    inputs=[search_query, search_top_k],
                    outputs=[search_output]
                )

            # 📝 Summarize Tab
            with gr.Tab("📝 Summarize"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Generate Document Summaries")

                        doc_dropdown_sum = gr.Dropdown(
                            label="Select Document to Summarize",
                            choices=[],  # empty initially
                            value=None,
                            interactive=True,
                            allow_custom_value=False
                        )

                        summary_text = gr.Textbox(
                            label="Or Paste Text to Summarize",
                            placeholder="Paste any text here to summarize...",
                            lines=8
                        )

                        summary_style = gr.Dropdown(
                            label="Summary Style",
                            choices=["concise", "detailed", "bullet_points", "executive"],
                            value="concise",
                            info="Choose how you want the summary formatted"
                        )
                        summarize_btn = gr.Button("📝 Generate Summary", variant="primary", size="lg")

                    with gr.Column():
                        summary_output = gr.Textbox(
                            label="Generated Summary",
                            lines=20,
                            placeholder="Summary will appear here..."
                        )

                summarize_btn.click(
                    summarize_document,
                    inputs=[doc_dropdown_sum, summary_text, summary_style],
                    outputs=[summary_output]
                )

            # 🏷️ Generate Tags Tab
            with gr.Tab("🏷️ Generate Tags"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Auto-Generate Document Tags")

                        doc_dropdown_tag = gr.Dropdown(
                            label="Select Document to Tag",
                            choices=[],  
                            value=None,
                            interactive=True,
                            allow_custom_value=False
                        )

                        tag_text = gr.Textbox(
                            label="Or Paste Text to Generate Tags",
                            placeholder="Paste any text here to generate tags...",
                            lines=8
                        )

                        max_tags = gr.Slider(
                            label="Number of Tags",
                            minimum=3,
                            maximum=15,
                            value=5,
                            step=1
                        )
                        tag_btn = gr.Button("🏷️ Generate Tags", variant="primary", size="lg")

                    with gr.Column():
                        tag_output = gr.Textbox(
                            label="Generated Tags",
                            lines=10,
                            placeholder="Tags will appear here..."
                        )

                tag_btn.click(
                    generate_tags_for_document,
                    inputs=[doc_dropdown_tag, tag_text, max_tags],
                    outputs=[tag_output]
                )

            # ❓ Ask Questions Tab
            with gr.Tab("❓ Ask Questions"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        ### Ask Questions About Your Documents

                        The AI will search through all your uploaded documents to find relevant information 
                        and provide comprehensive answers with sources.
                        """)
                        qa_question = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask anything about your documents...",
                            lines=3
                        )
                        qa_btn = gr.Button("❓ Get Answer", variant="primary", size="lg")

                    with gr.Column():
                        qa_output = gr.Textbox(
                            label="AI Answer",
                            lines=20,
                            placeholder="Answer will appear here with sources..."
                        )

                qa_btn.click(
                    ask_question,
                    inputs=[qa_question],
                    outputs=[qa_output]
                )

        # ✅ Auto-refresh all dropdowns when the app loads
        interface.load(
            fn=lambda: (
                get_document_list(),
                get_document_choices(),  # for summarize tab
                get_document_choices(),  # for tag tab
                get_document_choices()   # for delete dropdown
            ),
            outputs=[
                document_list,
                doc_dropdown_sum,
                doc_dropdown_tag,
                delete_doc_dropdown
            ]
        )

        return interface
        

# Create and launch the interface
if __name__ == "__main__":
    interface = create_gradio_interface()
    
    # Launch with proper configuration for Hugging Face Spaces
    interface.launch(mcp_server=True)