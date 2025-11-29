import gradio as gr
import os
import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import nest_asyncio

# Apply nest_asyncio to handle nested event loops in Gradio
nest_asyncio.apply()

# Import services and tools from mcp_server
from mcp_server import (
    # Services
    vector_store_service,
    document_store_service,
    embedding_service_instance,
    llm_service_instance,
    ocr_service_instance,
    llamaindex_service_instance,
    elevenlabs_service_instance,
    podcast_generator_instance,
    # Tools
    ingestion_tool_instance,
    search_tool_instance,
    generative_tool_instance,
    voice_tool_instance,
    podcast_tool_instance
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS FOR ASYNC EXECUTION
# ============================================================================

def run_async(coro):
    """Helper to run async functions in Gradio"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return loop.run_until_complete(coro)

# ============================================================================
# DOCUMENT MANAGEMENT FUNCTIONS
# ============================================================================

def get_document_list():
    """Get formatted list of documents"""
    try:
        documents = run_async(document_store_service.list_documents(limit=100))
        if documents:
            doc_list_str = "📚 Documents in Library:\n\n"
            for i, doc in enumerate(documents, 1):
                doc_list_str += f"{i}. {doc.filename} (ID: {doc.id[:8]}...)\n"
                doc_list_str += f"   Type: {doc.doc_type}, Size: {doc.file_size} bytes\n"
                if doc.metadata and doc.metadata.get('tags'):
                    doc_list_str += f"   Tags: {', '.join(doc.metadata['tags'])}\n"
                doc_list_str += f"   Created: {doc.created_at[:10]}\n\n"
            return doc_list_str
        else:
            return "No documents in library yet. Upload some documents to get started!"
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        return f"Error: {str(e)}"

def get_document_choices():
    """Get document choices for dropdowns"""
    try:
        documents = run_async(document_store_service.list_documents(limit=100))
        if documents:
            choices = [(f"{doc.filename} ({doc.id[:8]}...)", doc.id) for doc in documents]
            logger.info(f"Generated {len(choices)} document choices")
            return choices
        return []
    except Exception as e:
        logger.error(f"Error getting document choices: {str(e)}")
        return []

def refresh_library():
    """Refresh library and update all dropdowns"""
    doc_list_refreshed = get_document_list()
    doc_choices_refreshed = get_document_choices()
    logger.info(f"Refreshing library. Found {len(doc_choices_refreshed)} choices.")
    return (
        doc_list_refreshed,
        gr.update(choices=doc_choices_refreshed),
        gr.update(choices=doc_choices_refreshed),
        gr.update(choices=doc_choices_refreshed)
    )

def upload_and_process_file(file):
    """Upload and process a document file"""
    if file is None:
        doc_list_initial = get_document_list()
        doc_choices_initial = get_document_choices()
        return (
            "No file uploaded", "", doc_list_initial,
            gr.update(choices=doc_choices_initial),
            gr.update(choices=doc_choices_initial),
            gr.update(choices=doc_choices_initial)
        )
    try:
        file_path = file.name if hasattr(file, 'name') else str(file)
        file_type = Path(file_path).suffix.lower().strip('.')
        logger.info(f"Processing file: {file_path}, type: {file_type}")
        
        result = run_async(ingestion_tool_instance.process_document(file_path, file_type))
        
        doc_list_updated = get_document_list()
        doc_choices_updated = get_document_choices()

        if result["success"]:
            return (
                f"✅ Success: {result['message']}\nDocument ID: {result['document_id']}\nChunks created: {result['chunks_created']}",
                result["document_id"],
                doc_list_updated,
                gr.update(choices=doc_choices_updated),
                gr.update(choices=doc_choices_updated),
                gr.update(choices=doc_choices_updated)
            )
        else:
            return (
                f"❌ Error: {result.get('error', 'Unknown error')}", "",
                doc_list_updated,
                gr.update(choices=doc_choices_updated),
                gr.update(choices=doc_choices_updated),
                gr.update(choices=doc_choices_updated)
            )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        doc_list_error = get_document_list()
        doc_choices_error = get_document_choices()
        return (
            f"❌ Error: {str(e)}", "",
            doc_list_error,
            gr.update(choices=doc_choices_error),
            gr.update(choices=doc_choices_error),
            gr.update(choices=doc_choices_error)
        )

def delete_document_from_library(document_id):
    """Delete a document from the library"""
    if not document_id:
        doc_list_current = get_document_list()
        doc_choices_current = get_document_choices()
        return (
            "No document selected to delete.",
            doc_list_current,
            gr.update(choices=doc_choices_current),
            gr.update(choices=doc_choices_current),
            gr.update(choices=doc_choices_current)
        )
    try:
        delete_doc_store_result = run_async(document_store_service.delete_document(document_id))
        delete_vec_store_result = run_async(vector_store_service.delete_document(document_id))

        msg = ""
        if delete_doc_store_result:
            msg += f"🗑️ Document {document_id[:8]}... deleted from document store. "
        else:
            msg += f"❌ Failed to delete document {document_id[:8]}... from document store. "
        
        if delete_vec_store_result:
             msg += "Embeddings deleted from vector store."
        else:
             msg += "Failed to delete embeddings from vector store (or no embeddings existed)."

        doc_list_updated = get_document_list()
        doc_choices_updated = get_document_choices()
        return (
            msg,
            doc_list_updated,
            gr.update(choices=doc_choices_updated),
            gr.update(choices=doc_choices_updated),
            gr.update(choices=doc_choices_updated)
        )
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        doc_list_error = get_document_list()
        doc_choices_error = get_document_choices()
        return (
            f"❌ Error deleting document: {str(e)}",
            doc_list_error,
            gr.update(choices=doc_choices_error),
            gr.update(choices=doc_choices_error),
            gr.update(choices=doc_choices_error)
        )

# ============================================================================
# SEARCH FUNCTIONS
# ============================================================================

def perform_search(query, top_k):
    """Perform semantic search"""
    if not query.strip():
        return "Please enter a search query"
    try:
        results = run_async(search_tool_instance.search(query, int(top_k)))
        if results:
            output_str = f"🔍 Found {len(results)} results for: '{query}'\n\n"
            for i, result in enumerate(results, 1):
                output_str += f"Result {i}:\n"
                output_str += f"📊 Relevance Score: {result.score:.3f}\n"
                output_str += f"📄 Content: {result.content[:300]}...\n"
                if result.metadata and 'document_filename' in result.metadata:
                    output_str += f"📁 Source: {result.metadata['document_filename']}\n"
                output_str += f"🔗 Document ID: {result.document_id}\n"
                output_str += "-" * 80 + "\n\n"
            return output_str
        else:
            return f"No results found for: '{query}'\n\nMake sure you have uploaded relevant documents first."
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return f"❌ Error: {str(e)}"

# ============================================================================
# CONTENT STUDIO FUNCTIONS
# ============================================================================

def update_options_visibility(task):
    """Update visibility of options based on selected task"""
    return (
        gr.update(visible=task == "Summarize"),
        gr.update(visible=task == "Generate Outline"),
        gr.update(visible=task == "Generate Outline"),
        gr.update(visible=task == "Explain Concept"),
        gr.update(visible=task == "Explain Concept"),
        gr.update(visible=task == "Paraphrase"),
        gr.update(visible=task == "Categorize"),
        gr.update(visible=task in ["Key Insights", "Generate Questions"]),
        gr.update(visible=task == "Generate Questions")
    )

async def get_document_content(document_id: str) -> Optional[str]:
    """Get document content by ID"""
    try:
        doc = await document_store_service.get_document(document_id)
        if doc:
            return doc.content
        return None
    except Exception as e:
        logger.error(f"Error getting document content: {str(e)}")
        return None

def execute_content_task(task, doc_choice, custom_text, 
                        summary_style, outline_sections, outline_detail,
                        explain_audience, explain_length,
                        paraphrase_style, categories_input,
                        num_items, question_type):
    """Execute content analysis tasks"""
    try:
        # Get content
        content = ""
        if custom_text and custom_text.strip():
            content = custom_text
        elif doc_choice and doc_choice != "none":
            content = run_async(get_document_content(doc_choice))
            if not content:
                return "❌ Error: Document not found or empty"
        else:
            if task == "Generate Outline":
                content = custom_text
            else:
                return "⚠️ Please select a document or enter text"

        # Execute task
        if task == "Summarize":
            summary = run_async(generative_tool_instance.summarize(content, summary_style))
            return f"📝 Summary ({summary_style}):\n\n{summary}"
                
        elif task == "Generate Outline":
            outline = run_async(generative_tool_instance.generate_outline(content, int(outline_sections), outline_detail))
            return f"📝 Outline for '{content}':\n\n{outline}"
                
        elif task == "Explain Concept":
            explanation = run_async(generative_tool_instance.explain_concept(content, explain_audience, explain_length))
            return f"💡 Explanation ({explain_audience}):\n\n{explanation}"
                
        elif task == "Paraphrase":
            paraphrase = run_async(generative_tool_instance.paraphrase_text(content, paraphrase_style))
            return f"🔄 Paraphrased Text ({paraphrase_style}):\n\n{paraphrase}"
                
        elif task == "Categorize":
            categories = [c.strip() for c in categories_input.split(',')] if categories_input else []
            category = run_async(generative_tool_instance.categorize(content, categories))
            return f"🏷️ Category:\n\n{category}"
                
        elif task == "Key Insights":
            insights = run_async(generative_tool_instance.extract_key_insights(content, int(num_items)))
            return f"🔍 Key Insights:\n\n" + "\n".join([f"- {insight}" for insight in insights])
                
        elif task == "Generate Questions":
            questions = run_async(generative_tool_instance.generate_questions(content, question_type, int(num_items)))
            return f"❓ Generated Questions ({question_type}):\n\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
                
        elif task == "Extract Key Info":
            info = run_async(llm_service_instance.extract_key_information(content))
            return f"📊 Key Information:\n\n{json.dumps(info, indent=2)}"

        return "✅ Task completed"

    except Exception as e:
        logger.error(f"Task execution error: {str(e)}")
        return f"❌ Error: {str(e)}"

def generate_tags_for_document(doc_choice, custom_text, max_tags):
    """Generate tags for document or text"""
    try:
        logger.info(f"Generate tags called with doc_choice: {doc_choice}")
        document_id = doc_choice if doc_choice and doc_choice != "none" and doc_choice != "" else None

        if custom_text and custom_text.strip():
            logger.info("Using custom text for tag generation")
            tags = run_async(generative_tool_instance.generate_tags(custom_text, int(max_tags)))
            content_length = len(custom_text)
            doc_id_display = None
        elif document_id:
            logger.info(f"Generating tags for document: {document_id}")
            content = run_async(get_document_content(document_id))
            if not content:
                return "❌ Error: Document not found or empty"
            tags = run_async(generative_tool_instance.generate_tags(content, int(max_tags)))
            if tags:
                run_async(document_store_service.update_document_metadata(document_id, {"tags": tags}))
            content_length = len(content)
            doc_id_display = document_id
        else:
            return "Please select a document from the dropdown or enter text to generate tags"
        
        if tags:
            tags_str = ", ".join(tags)
            output_str = f"🏷️ Generated Tags:\n\n{tags_str}\n\n"
            output_str += f"📊 Statistics:\n"
            output_str += f"- Content length: {content_length} characters\n"
            output_str += f"- Number of tags: {len(tags)}\n"
            if doc_id_display:
                output_str += f"- Document ID: {doc_id_display}\n"
                output_str += f"\n✅ Tags have been saved to the document."
            return output_str
        else:
            return "❌ Tag generation failed"
    except Exception as e:
        logger.error(f"Tag generation error: {str(e)}")
        return f"❌ Error: {str(e)}"

def ask_question(question):
    """Ask question with RAG"""
    if not question.strip():
        return "Please enter a question"
    try:
        search_results = run_async(search_tool_instance.search(question, top_k=5))
        if not search_results:
            return "❌ No relevant context found in your documents. Please make sure you have uploaded relevant documents."
        
        answer = run_async(generative_tool_instance.answer_question(question, search_results))
        
        output_str = f"❓ Question: {question}\n\n"
        output_str += f"💡 Answer:\n{answer}\n\n"
        output_str += f"🎯 Confidence: {'high' if len(search_results) >= 3 else 'medium'}\n\n"
        output_str += f"📚 Sources Used ({len(search_results)}):\n"
        for i, source in enumerate(search_results, 1):
            filename = source.metadata.get('document_filename', 'Unknown') if source.metadata else 'Unknown'
            output_str += f"\n{i}. 📄 {filename}\n"
            output_str += f"   📝 Excerpt: {source.content[:150]}...\n"
            output_str += f"   📊 Relevance: {source.score:.3f}\n"
        return output_str
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================================================
# VOICE ASSISTANT FUNCTIONS
# ============================================================================

voice_conversation_state = {
    "session_id": None,
    "active": False,
    "transcript": []
}

def start_voice_conversation():
    """Start a new voice conversation session"""
    try:
        if not elevenlabs_service_instance.is_available():
            return (
                "⚠️ Voice assistant not configured. Please set ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID in .env",
                gr.update(interactive=False),
                gr.update(interactive=True),
                []
            )
        
        session_id = str(uuid.uuid4())
        result = run_async(elevenlabs_service_instance.start_conversation(session_id))
        
        if result.get("success"):
            voice_conversation_state["session_id"] = session_id
            voice_conversation_state["active"] = True
            voice_conversation_state["transcript"] = []
            
            return (
                "🎙️ Voice assistant is ready. Type your question below.",
                gr.update(interactive=False),
                gr.update(interactive=True),
                []
            )
        else:
            return (
                f"❌ Failed to start conversation: {result.get('error')}",
                gr.update(interactive=True),
                gr.update(interactive=False),
                []
            )
    except Exception as e:
        logger.error(f"Error starting voice conversation: {str(e)}")
        return (
            f"❌ Error: {str(e)}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            []
        )

def stop_voice_conversation():
    """Stop active voice conversation"""
    try:
        if not voice_conversation_state["active"]:
            return (
                "No active conversation",
                gr.update(interactive=True),
                gr.update(interactive=False),
                voice_conversation_state["transcript"]
            )
        
        session_id = voice_conversation_state["session_id"]
        if session_id:
            run_async(elevenlabs_service_instance.end_conversation(session_id))
        
        voice_conversation_state["active"] = False
        voice_conversation_state["session_id"] = None
        
        return (
            "✅ Conversation ended",
            gr.update(interactive=True),
            gr.update(interactive=False),
            voice_conversation_state["transcript"]
        )
    except Exception as e:
        logger.error(f"Error stopping conversation: {str(e)}")
        return (
            f"❌ Error: {str(e)}",
            gr.update(interactive=True),
            gr.update(interactive=False),
            voice_conversation_state["transcript"]
        )

def send_voice_message_v6(message, chat_history):
    """Send message in voice conversation - Gradio 6 format"""
    try:
        if not voice_conversation_state["active"]:
            return chat_history, ""
        
        if not message or not message.strip():
            return chat_history, message
        
        session_id = voice_conversation_state["session_id"]
        
        # Add user message
        chat_history.append({"role": "user", "content": message})
        
        # Get AI response
        result = run_async(voice_tool_instance.voice_qa(message, session_id))
        
        if result.get("success"):
            answer = result.get("answer", "No response")
            chat_history.append({"role": "assistant", "content": answer})
        else:
            chat_history.append({
                "role": "assistant",
                "content": f"❌ Error: {result.get('error')}"
            })
        
        return chat_history, ""
    except Exception as e:
        logger.error(f"Error in voice message: {str(e)}")
        chat_history.append({
            "role": "assistant",
            "content": f"❌ Error: {str(e)}"
        })
        return chat_history, ""

# ============================================================================
# PODCAST GENERATION FUNCTIONS
# ============================================================================

def generate_podcast_ui(doc_ids, style, duration, voice1, voice2):
    """UI wrapper for podcast generation"""
    try:
        if not doc_ids or len(doc_ids) == 0:
            return ("⚠️ Please select at least one document", None, "No documents selected", "")
        
        logger.info(f"Generating podcast: {len(doc_ids)} docs, {style}, {duration}min")
        
        result = run_async(
            podcast_tool_instance.generate_podcast(
                document_ids=doc_ids,
                style=style,
                duration_minutes=int(duration),
                host1_voice=voice1,
                host2_voice=voice2
            )
        )
        
        if result.get("success"):
            audio_file = result.get("audio_file")
            transcript = result.get("transcript", "Transcript not available")
            message = result.get("message", "Podcast generated!")
            formatted_transcript = f"## Podcast Transcript\n\n{transcript}"
            
            return (
                f"✅ {message}",
                audio_file,
                formatted_transcript,
                result.get("podcast_id", "")
            )
        else:
            error = result.get("error", "Unknown error")
            return (f"❌ Error: {error}", None, "Generation failed", "")
    except Exception as e:
        logger.error(f"Podcast UI error: {str(e)}")
        return (f"❌ Error: {str(e)}", None, "An error occurred", "")

# ============================================================================
# DASHBOARD FUNCTIONS
# ============================================================================

def load_dashboard_stats():
    """Load dashboard statistics"""
    try:
        documents = run_async(document_store_service.list_documents(limit=1000))
        doc_count = len(documents) if documents else 0
        total_chunks = 0
        total_size = 0
        recent_data = []
        
        if documents:
            total_chunks = sum(doc.metadata.get("chunk_count", 0) for doc in documents if doc.metadata)
            total_size = sum(doc.file_size for doc in documents)
            storage_mb = round(total_size / (1024 * 1024), 2) if total_size > 0 else 0.0
            
            # Get recent 5 documents
            recent = documents[:5]
            recent_data = [
                [
                    doc.filename, 
                    doc.doc_type, 
                    doc.created_at[:10] if doc.created_at else "N/A", 
                    f"{doc.file_size} bytes"
                ]
                for doc in recent
            ]
        else:
            storage_mb = 0.0
        
        # Service status
        vector_stat = "✅ Online" if vector_store_service else "❌ Offline"
        llm_stat = "✅ Ready" if llm_service_instance else "❌ Offline"
        voice_stat = "✅ Ready" if (elevenlabs_service_instance and elevenlabs_service_instance.is_available()) else "⚠️ Configure API Key"
        
        return (
            doc_count,
            total_chunks,
            storage_mb,
            recent_data,
            vector_stat,
            llm_stat,
            voice_stat,
        )
    except Exception as e:
        logger.error(f"Error loading dashboard stats: {str(e)}")
        return (0, 0, 0.0, [], "❌ Error", "❌ Error", "❌ Error")

# ============================================================================
# GRADIO UI CREATION
# ============================================================================

def create_gradio_interface():
    """Create the Gradio interface"""
    
    # Create custom theme
    custom_theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Fira Code"), "monospace"],
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
        block_title_text_weight="600",
        block_label_text_size="sm",
        block_label_text_weight="500",
    )
    
    with gr.Blocks(title="🧠 AI Digital Library Assistant", theme=custom_theme) as interface:
        gr.Markdown("""
        # 📚 AI Digital Library Assistant
        A powerful AI-powered document management platform with semantic search, voice interaction, 
        podcast generation, and intelligent content analysis capabilities using MCP (Model Context Protocol).
        
        ## 🎯 Key Features:
        - **📄 Smart Document Processing** → Upload PDFs, Word docs, images with OCR support
        - **🔍 Semantic Search** → AI-powered search across your entire library
        - **🎙️ Voice Assistant** → Natural conversation with your documents via ElevenLabs
        - **🎧 Podcast Generation** → Transform documents into engaging audio conversations
        - **📝 Content Studio** → Summarize, outline, explain concepts, and more
        - **🏷️ Auto-Tagging** → Generate intelligent tags for organization
        - **❓ Q&A System** → Get answers with source citations from your documents
        
        ## 🚀 Quick Start:
        1. **📚 Document Library** → View and manage your uploaded documents
        2. **📄 Upload Documents** → Add PDFs, DOCX, TXT, or images (OCR enabled)
        3. **🔍 Search** → Find information using natural language queries
        4. **📝 Content Studio** → Summarize, paraphrase, or analyze your documents
        5. **🏷️ Generate Tags** → Auto-tag documents for better organization
        6. **❓ Ask Questions** → Get AI-powered answers with source citations
        7. **🎙️ Voice Assistant** → Have natural conversations about your content
        8. **🎧 Podcast Studio** → Generate audio podcasts from your documents
        
        ---
        
        🔗 **For MCP Integration** (Claude Desktop, Cline, etc.):  
        Add this endpoint to your MCP client configuration:
        ```
        https://nihal2000-aidigitiallibrary assistant.hf.space/gradio_api/mcp/sse
        ```
        
         💡 **Powered by:** OpenAI, Mistral AI, Claude, ElevenLabs, LlamaIndex
        """)
        
        with gr.Tabs():
            # Dashboard Tab
            with gr.Tab("🏠 Dashboard"):
                gr.Markdown("# Welcome to Your AI Library Assistant")
                gr.Markdown("*Your intelligent document management and analysis platform powered by AI*")
                
                gr.Markdown("## 📊 Quick Stats")
                with gr.Row():
                    total_docs = gr.Number(
                        label="📚 Total Documents",
                        value=0,
                        interactive=False,
                        container=True
                    )
                    total_chunks = gr.Number(
                        label="🧩 Vector Chunks",
                        value=0,
                        interactive=False,
                        container=True
                    )
                    storage_size = gr.Number(
                        label="💾 Storage (MB)",
                        value=0,
                        interactive=False,
                        container=True
                    )
                
                gr.Markdown("## 📊 Recent Activity")
                with gr.Group():
                    recent_docs = gr.Dataframe(
                        headers=["Document", "Type", "Date", "Size"],
                        datatype=["str", "str", "str", "str"],
                        row_count=(5, "fixed"),
                        col_count=(4, "fixed"),
                        interactive=False,
                        label="Recently Added Documents"
                    )
                
                gr.Markdown("## ⚙️ System Status")
                with gr.Row():
                    vector_status = gr.Textbox(
                        label="Vector Store",
                        value="✅ Online",
                        interactive=False,
                        container=True
                    )
                    llm_status = gr.Textbox(
                        label="LLM Service",
                        value="✅ Ready",
                        interactive=False,
                        container=True
                    )
                    voice_status = gr.Textbox(
                        label="Voice Service",
                        value="⚠️ Configure API Key",
                        interactive=False,
                        container=True
                    )
            
            # Document Library Tab
            with gr.Tab("📚 Document Library"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Your Document Collection")
                        document_list_display = gr.Textbox(
                            label="Documents in Library",
                            value=get_document_list(),
                            lines=20,
                            interactive=False
                        )
                        refresh_btn_library = gr.Button("🔄 Refresh Library", variant="secondary")
                        delete_doc_dropdown_visible = gr.Dropdown(
                            label="Select Document to Delete",
                            choices=get_document_choices(),
                            value=None,
                            interactive=True,
                            allow_custom_value=False
                        )
                        delete_btn = gr.Button("🗑️ Delete Selected Document", variant="stop")
                        delete_output_display = gr.Textbox(label="Delete Status", visible=True)
            
            # Upload Documents Tab
            with gr.Tab("📄 Upload Documents"):
                gr.Markdown("""
                ### 📥 Add Documents to Library
                Upload PDFs, Word documents, text files, or images. OCR will extract text from images automatically.
                """)
                
                with gr.Row():
                    with gr.Column():
                        with gr.Group():
                            gr.Markdown("**Supported formats:** PDF, DOCX, TXT, Images (JPG, PNG)")
                            file_input_upload = gr.File(
                                label="Select File",
                                file_types=[".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"],
                                type="filepath",
                                file_count="single"
                            )
                            
                            upload_btn_process = gr.Button("🚀 Upload & Process", variant="primary", size="lg")
                        
                        with gr.Group():
                            upload_output_display = gr.Textbox(
                                label="Status",
                                lines=6,
                                interactive=False,
                                show_copy_button=False
                            )
                            
                            doc_id_output_display = gr.Textbox(
                                label="Document ID",
                                interactive=False,
                                visible=False
                            )

            # Search Documents Tab
            with gr.Tab("🔍 Search Documents"):
                gr.Markdown("""
                ### 🔎 Semantic Search
                Find relevant content across your entire document library using AI-powered semantic search.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            search_query_input = gr.Textbox(
                                label="Search Query",
                                placeholder="What are you looking for?",
                                lines=2,
                                info="Use natural language to describe what you need"
                            )
                            
                            with gr.Accordion("🎛️ Search Options", open=False):
                                search_top_k_slider = gr.Slider(
                                    label="Number of Results",
                                    minimum=1, maximum=20, value=5, step=1,
                                    info="More results = broader search"
                                )
                            
                            search_btn_action = gr.Button("🔍 Search", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        with gr.Group():
                            search_output_display = gr.Textbox(
                                label="Results",
                                lines=20,
                                placeholder="Search results will appear here...",
                                show_copy_button=True
                            )
            
            # Content Studio Tab
            with gr.Tab("📝 Content Studio"):
                gr.Markdown("""
                ### 🎨 Create & Analyze Content
                Transform documents with AI-powered tools: summarize, outline, explain, and more.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("#### 📄 Content Source")
                            doc_dropdown_content = gr.Dropdown(
                                label="Select Document",
                                choices=get_document_choices(),
                                value=None,
                                interactive=True,
                                info="Choose a document from your library"
                            )
                            
                            gr.Markdown("**OR**")
                            
                            content_text_input = gr.Textbox(
                                label="Enter Text or Topic",
                                placeholder="Paste content or enter a topic...",
                                lines=4,
                                info="For outlines, enter a topic. For other tasks, paste text to analyze."
                            )
                        
                        with gr.Group():
                            gr.Markdown("#### 🛠️ Task Configuration")
                            task_dropdown = gr.Dropdown(
                                label="Select Task",
                                choices=[
                                    "Summarize", "Generate Outline", "Explain Concept",
                                    "Paraphrase", "Categorize", "Key Insights",
                                    "Generate Questions", "Extract Key Info"
                                ],
                                value="Summarize",
                                interactive=True,
                                info="Choose the type of analysis to perform"
                            )
                        
                        with gr.Accordion("⚙️ Advanced Options", open=False):
                            summary_style_opt = gr.Dropdown(
                                label="Summary Style",
                                choices=["concise", "detailed", "bullet_points", "executive"],
                                value="concise",
                                visible=True,
                                info="How detailed should the summary be?"
                            )
                            
                            outline_sections_opt = gr.Slider(
                                label="Number of Sections",
                                minimum=3, maximum=10, value=5, step=1,
                                visible=False,
                                info="How many main sections?"
                            )
                            outline_detail_opt = gr.Dropdown(
                                label="Detail Level",
                                choices=["brief", "medium", "detailed"],
                                value="medium",
                                visible=False
                            )
                            
                            explain_audience_opt = gr.Dropdown(
                                label="Target Audience",
                                choices=["general", "technical", "beginner", "expert"],
                                value="general",
                                visible=False,
                                info="Who is this explanation for?"
                            )
                            explain_length_opt = gr.Dropdown(
                                label="Length",
                                choices=["brief", "medium", "detailed"],
                                value="medium",
                                visible=False
                            )
                            
                            paraphrase_style_opt = gr.Dropdown(
                                label="Style",
                                choices=["formal", "casual", "academic", "simple", "technical"],
                                value="formal",
                                visible=False,
                                info="Writing style for paraphrasing"
                            )
                            
                            categories_input_opt = gr.Textbox(
                                label="Categories (comma separated)",
                                placeholder="Technology, Business, Science...",
                                visible=False
                            )
                            
                            num_items_opt = gr.Slider(
                                label="Number of Items",
                                minimum=1, maximum=10, value=5, step=1,
                                visible=False
                            )
                            question_type_opt = gr.Dropdown(
                                label="Question Type",
                                choices=["comprehension", "analysis", "application", "creative", "factual"],
                                value="comprehension",
                                visible=False
                            )
                        
                        run_task_btn = gr.Button("🚀 Run Task", variant="primary", size="lg")
                    
                    with gr.Column(scale=3):
                        with gr.Group():
                            gr.Markdown("#### 📊 Result")
                            content_output_display = gr.Textbox(
                                label="",
                                lines=25,
                                placeholder="Results will appear here...",
                                show_copy_button=True,
                                container=False
                            )

                # Event Handlers for Content Studio
                task_dropdown.change(
                    fn=update_options_visibility,
                    inputs=[task_dropdown],
                    outputs=[
                        summary_style_opt, outline_sections_opt, outline_detail_opt,
                        explain_audience_opt, explain_length_opt, paraphrase_style_opt,
                        categories_input_opt, num_items_opt, question_type_opt
                    ]
                )
                
                run_task_btn.click(
                    fn=execute_content_task,
                    inputs=[
                        task_dropdown, doc_dropdown_content, content_text_input,
                        summary_style_opt, outline_sections_opt, outline_detail_opt,
                        explain_audience_opt, explain_length_opt, paraphrase_style_opt,
                        categories_input_opt, num_items_opt, question_type_opt
                    ],
                    outputs=[content_output_display]
                )

            # Generate Tags Tab
            with gr.Tab("🏷️ Generate Tags"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Generate Document Tags")
                        doc_dropdown_tag_visible = gr.Dropdown(
                            label="Select Document to Tag",
                            choices=get_document_choices(),
                            value=None,
                            interactive=True,
                            allow_custom_value=False
                        )
                        tag_text_input = gr.Textbox(
                            label="Or Paste Text to Generate Tags",
                            placeholder="Paste any text here to generate tags...",
                            lines=8
                        )
                        max_tags_slider = gr.Slider(
                            label="Number of Tags",
                            minimum=3, maximum=15, value=5, step=1
                        )
                        tag_btn_action = gr.Button("🏷️ Generate Tags", variant="primary", size="lg")
                    with gr.Column():
                        tag_output_display = gr.Textbox(
                            label="Generated Tags",
                            lines=10,
                            placeholder="Tags will appear here..."
                        )

            # Voice Assistant Tab
            with gr.Tab("🎙️ Voice Assistant"):
                gr.Markdown("""
                ### 🗣️ Talk to Your AI Librarian
                
                Have a natural conversation about your documents. Ask questions, request summaries,
                or explore your content library through voice-powered interaction.
                
                **Note:** Requires ElevenLabs API configuration.
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Group():
                            voice_status_display = gr.Textbox(
                                label="Status",
                                value="Ready to start",
                                interactive=False,
                                lines=2
                            )
                            
                            with gr.Row():
                                start_voice_btn = gr.Button("🎤 Start Conversation", variant="primary", size="lg")
                                stop_voice_btn = gr.Button("⏹️ Stop", variant="stop", size="lg", interactive=False)
                        
                        with gr.Group():
                            gr.Markdown("#### 💬 Send Message")
                            voice_input_text = gr.Textbox(
                                label="",
                                placeholder="Type your question...",
                                lines=3,
                                container=False,
                                info="Press Enter or click Send"
                            )
                            send_voice_btn = gr.Button("📤 Send", variant="secondary")
                    
                    with gr.Column(scale=3):
                        with gr.Group():
                            voice_chatbot = gr.Chatbot(
                                label="Conversation",
                                type="messages",
                                height=500,
                                show_copy_button=True
                            )
                            
                            clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                
                # Voice Assistant event handlers
                start_voice_btn.click(
                    fn=start_voice_conversation,
                    outputs=[voice_status_display, start_voice_btn, stop_voice_btn, voice_chatbot]
                )
                
                stop_voice_btn.click(
                    fn=stop_voice_conversation,
                    outputs=[voice_status_display, start_voice_btn, stop_voice_btn, voice_chatbot]
                )
                
                send_voice_btn.click(
                    fn=send_voice_message_v6,
                    inputs=[voice_input_text, voice_chatbot],
                    outputs=[voice_chatbot, voice_input_text]
                )
                
                voice_input_text.submit(
                    fn=send_voice_message_v6,
                    inputs=[voice_input_text, voice_chatbot],
                    outputs=[voice_chatbot, voice_input_text]
                )
                
                clear_chat_btn.click(
                    fn=lambda: [],
                    outputs=[voice_chatbot]
                )

            # Podcast Studio Tab
            with gr.Tab("🎧 Podcast Studio"):
                gr.Markdown("""
                ### 🎙️ AI-Powered Podcast Generation
                
                Transform your documents into engaging audio conversations. Select documents,
                customize the style and voices, and let AI create a professional podcast.
                
                **Powered by:** ElevenLabs AI Voice Technology
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Group():
                            gr.Markdown("#### 📚 Select Content")
                            
                            podcast_doc_selector = gr.CheckboxGroup(
                                choices=get_document_choices(),
                                label="Documents to Include",
                                info="Choose 1-5 documents for best results",
                                interactive=True
                            )
                        
                        with gr.Accordion("🎨 Podcast Settings", open=True):
                            with gr.Row():
                                podcast_style = gr.Dropdown(
                                    label="Style",
                                    choices=["conversational", "educational", "technical", "casual"],
                                    value="conversational",
                                    info="Sets the tone and format"
                                )
                                
                                podcast_duration = gr.Slider(
                                    label="Duration (minutes)",
                                    minimum=5,
                                    maximum=30,
                                    value=10,
                                    step=5,
                                    info="Approximate length"
                                )
                            
                            gr.Markdown("#### 🗣️ Voice Selection")
                            with gr.Row():
                                host1_voice_selector = gr.Dropdown(
                                    label="Host 1",
                                    choices=["Rachel", "Adam", "Domi", "Bella", "Antoni", "Elli", "Josh"],
                                    value="Rachel"
                                )
                                host2_voice_selector = gr.Dropdown(
                                    label="Host 2",
                                    choices=["Adam", "Rachel", "Josh", "Sam", "Emily", "Antoni", "Arnold"],
                                    value="Adam"
                                )
                        
                        generate_podcast_btn = gr.Button(
                            "🎙️ Generate Podcast",
                            variant="primary",
                            size="lg"
                        )
                        
                        podcast_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=2
                        )
                        
                        podcast_id_display = gr.Textbox(
                            label="Podcast ID",
                            interactive=False,
                            visible=False
                        )
                    
                    with gr.Column(scale=3):
                        with gr.Group():
                            gr.Markdown("#### 🎵 Generated Podcast")
                            
                            podcast_audio_player = gr.Audio(
                                label="",
                                type="filepath",
                                interactive=False,
                                autoplay=True,
                                container=False
                            )
                        
                        with gr.Accordion("📝 Transcript", open=False):
                            podcast_transcript_display = gr.Markdown(
                                value="*Transcript will appear after generation...*"
                            )
                
                # Event handlers
                generate_podcast_btn.click(
                    fn=generate_podcast_ui,
                    inputs=[
                        podcast_doc_selector,
                        podcast_style,
                        podcast_duration,
                        host1_voice_selector,
                        host2_voice_selector
                    ],
                    outputs=[
                        podcast_status,
                        podcast_audio_player,
                        podcast_transcript_display,
                        podcast_id_display
                    ]
                )
                
            # Ask Questions Tab
            with gr.Tab("❓ Ask Questions"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""### Ask Questions About Your Documents
                        The AI will search through all your uploaded documents to find relevant information 
                        and provide comprehensive answers with sources.""")
                        qa_question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask anything about your documents...",
                            lines=3
                        )
                        qa_btn_action = gr.Button("❓ Get Answer", variant="primary", size="lg")
                    with gr.Column():
                        qa_output_display = gr.Textbox(
                            label="AI Answer",
                            lines=20,
                            placeholder="Answer will appear here with sources..."
                        )

        # Wire up all dropdown updates
        all_dropdowns_to_update = [delete_doc_dropdown_visible, doc_dropdown_content, doc_dropdown_tag_visible]
        
        refresh_outputs = [document_list_display] + [dd for dd in all_dropdowns_to_update]
        refresh_btn_library.click(fn=refresh_library, outputs=refresh_outputs)
        
        upload_outputs = [upload_output_display, doc_id_output_display, document_list_display] + [dd for dd in all_dropdowns_to_update]
        upload_btn_process.click(upload_and_process_file, inputs=[file_input_upload], outputs=upload_outputs)

        delete_outputs = [delete_output_display, document_list_display] + [dd for dd in all_dropdowns_to_update]
        delete_btn.click(delete_document_from_library, inputs=[delete_doc_dropdown_visible], outputs=delete_outputs)
        
        search_btn_action.click(perform_search, inputs=[search_query_input, search_top_k_slider], outputs=[search_output_display])
        tag_btn_action.click(generate_tags_for_document, inputs=[doc_dropdown_tag_visible, tag_text_input, max_tags_slider], outputs=[tag_output_display])
        qa_btn_action.click(ask_question, inputs=[qa_question_input], outputs=[qa_output_display])

        # Load dashboard stats on interface load
        interface.load(
            fn=load_dashboard_stats,
            outputs=[total_docs, total_chunks, storage_size, recent_docs, vector_status, llm_status, voice_status]
        )
        
        interface.load(fn=refresh_library, outputs=refresh_outputs)
        
        return interface

if __name__ == "__main__":
    gradio_interface = create_gradio_interface()
    gradio_interface.launch(mcp_server=True)
