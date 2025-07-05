---
title: Intelligent Content Organizer MCP Agent
emoji: 😻
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.32.0
app_file: app.py
tag : mcp-server-track
    -Agents-MCP-Hackathon
pinned: false
license: mit
---

A powerful Model Context Protocol (MCP) server for intelligent content management with semantic search, summarization, and Q&A capabilities powered by **OpenAI, Mistral AI, and Anthropic Claude**.

## 🎯 Features

### 🔧 MCP Tools Available

- **📄 Document Ingestion**: Upload and process documents (PDF, TXT, DOCX, images with OCR)
- **🔍 Semantic Search**: Find relevant content using natural language queries
- **📝 Summarization**: Generate summaries in different styles (concise, detailed, bullet points, executive)
- **🏷️ Tag Generation**: Automatically generate relevant tags for content
- **❓ Q&A System**: Ask questions about your documents using RAG (Retrieval-Augmented Generation)
- **📊 Categorization**: Classify content into predefined or custom categories
- **🔄 Batch Processing**: Process multiple documents at once
- **📈 Analytics**: Get insights and statistics about your content

### 🚀 Powered By

- **🧠 OpenAI GPT models** for powerful text generation and understanding
- **🔥 Mistral AI** for efficient text processing and analysis
- **🤖 Anthropic Claude** for advanced reasoning (available as a specific choice or fallback)
- **🔗 Sentence Transformers** for semantic embeddings
- **📚 FAISS** for fast similarity search
- **👁️ Tesseract OCR** for image text extraction
- **🎨 Gradio** for the user interface and MCP server functionality

**LLM Strategy**: The agent intelligently selects the best available LLM for most generative tasks when 'auto' model selection is used, prioritizing OpenAI, then Mistral, and finally Anthropic. Users can also specify a particular model family (e.g., 'gpt-', 'mistral-', 'claude-').

## 📋 Complete File Structure
intelligent-content-organizer/
├── app.py                     # Main Gradio app and MCP server
├── config.py                  # Configuration management
├── mcp_server.py              # mcp server tools
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
├── .gitignore                # Git ignore rules
├── core/                     # Core processing logic
│   ├── init.py
│   ├── models.py             # Data models
│   ├── document_parser.py    # Document processing
│   ├── text_preprocessor.py  # Text cleaning and processing
│   └── chunker.py           # Text chunking strategies
├── services/                 # Backend services
│   ├── init.py
│   ├── embedding_service.py  # Sentence transformers integration
│   ├── llm_service.py       # Anthropic + Mistral integration
│   ├── ocr_service.py       # Mistral OCR integration
│   ├── vector_store_service.py # FAISS vector storage
│   └── document_store_service.py # Document metadata storage
└── mcp_tools/               # MCP tool definitions
├── init.py
├── ingestion_tool.py    # Document ingestion tool
├── search_tool.py       # Semantic search tool
├── generative_tool.py   # AI generation tool
└── utils.py            # Utility functions

## 🎯 Key Features Implemented

1. **Full MCP Server**: Complete implementation with all tools exposed
2. **Multi-Modal Processing**: PDF, TXT, DOCX, and image processing with OCR
3. **Advanced Search**: Semantic search with FAISS, filtering, and multi-query support
4. **AI-Powered Features**: Summarization, tagging, categorization, Q&A with RAG
5. **Production Ready**: Error handling, logging, caching, rate limiting
6. **Gradio UI**: Beautiful web interface for testing and direct use
7. **Anthropic + Mistral**: Dual LLM support with fallbacks

## 🎥 Demo Video

[📹 Watch the demo video](https://your-demo-video-url.com)

*The demo shows the MCP server in action, demonstrating document ingestion, semantic search, and Q&A capabilities, utilizing the configured LLM providers.*

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- API keys for OpenAI and Mistral AI. An Anthropic API key.

- **MCP Tools Reference** (Tool parameters like model allow specifying "auto" or a specific model family like "gpt-", "mistral-", "claude-")

- **ingest_document**
  - Process and index a document for searching.
  - **Parameters:**
    - `file_path` (string): Path to the document file (e.g., an uploaded file path).
    - `file_type` (string, optional): File type/extension (e.g., ".pdf", ".txt"). If not provided, it's inferred from file_path.
  - **Returns:**
    - `success` (boolean): Whether the operation succeeded.
    - `document_id` (string): Unique identifier for the processed document.
    - `chunks_created` (integer): Number of text chunks created.
    - `message` (string): Human-readable result message.

- **semantic_search**
  - Search through indexed content using natural language.
  - **Parameters:**
    - `query` (string): Search query.
    - `top_k` (integer, optional): Number of results to return (default: 5).
    - `filters` (object, optional): Search filters (e.g., {"document_id": "some_id"}).
  - **Returns:**
    - `success` (boolean): Whether the search succeeded.
    - `results` (array of objects): Array of search results, each with content and score.
    - `total_results` (integer): Number of results found.

- **summarize_content**
  - Generate a summary of provided content.
  - **Parameters:**
    - `content` (string, optional): Text content to summarize.
    - `document_id` (string, optional): ID of document to summarize. (Either content or document_id must be provided).
    - `style` (string, optional): Summary style: "concise", "detailed", "bullet_points", "executive" (default: "concise").
    - `model` (string, optional): Specific LLM to use (e.g., "gpt-4o-mini", "mistral-large-latest", "auto"). Default: "auto".
  - **Returns:**
    - `success` (boolean): Whether summarization succeeded.
    - `summary` (string): Generated summary.
    - `original_length` (integer): Character length of original content.
    - `summary_length` (integer): Character length of summary.

- **generate_tags**
  - Generate relevant tags for content.
  - **Parameters:**
    - `content` (string, optional): Text content to tag.
    - `document_id` (string, optional): ID of document to tag. (Either content or document_id must be provided).
    - `max_tags` (integer, optional): Maximum number of tags (default: 5).
    - `model` (string, optional): Specific LLM to use. Default: "auto".
  - **Returns:**
    - `success` (boolean): Whether tag generation succeeded.
    - `tags` (array of strings): Array of generated tags.

- **answer_question**
  - Answer questions using RAG over your indexed content.
  - **Parameters:**
    - `question` (string): Question to answer.
    - `context_filter` (object, optional): Filters for context retrieval (e.g., {"document_id": "some_id"}).
    - `model` (string, optional): Specific LLM to use. Default: "auto".
  - **Returns:**
    - `success` (boolean): Whether question answering succeeded.
    - `answer` (string): Generated answer.
    - `sources` (array of objects): Source document chunks used for context, each with document_id, chunk_id, and content.
    - `confidence` (string, optional): Confidence level in the answer (LLM-dependent, might not always be present).

📊 Performance

Embedding Generation: ~100-500ms per document chunk
Search: <50ms for most queries
Summarization: 1-5s depending on content length
Memory Usage: ~200-500MB base + ~1MB per 1000 document chunks
Supported File Types: PDF, TXT, DOCX, PNG, JPG, JPEG, BMP, TIFF

MCP Server:
        "https://nihal2000-intelligent-content-organizer-mcp-agent.hf.space//mcp/sse"





