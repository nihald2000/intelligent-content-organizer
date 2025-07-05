# Intelligent Content Organizer MCP Agent - Complete Codebase Architecture Graph

## 🏗️ System Architecture Overview

```mermaid
graph TB
    %% External Interfaces
    subgraph "External Interfaces"
        UI[Gradio Web UI]
        MCP[MCP Protocol Server]
        CLI[Command Line Interface]
    end

    %% Main Application Layer
    subgraph "Application Layer"
        APP[app.py - Main Application]
        MCP_SERVER[mcp_server.py - MCP Server]
    end

    %% Tool Layer
    subgraph "MCP Tools Layer"
        INGEST[IngestionTool]
        SEARCH[SearchTool]
        GEN[GenerativeTool]
        UTILS[Utils]
    end

    %% Service Layer
    subgraph "Service Layer"
        DS[DocumentStoreService]
        VS[VectorStoreService]
        ES[EmbeddingService]
        LS[LLMService]
        OS[OCRService]
    end

    %% Core Processing Layer
    subgraph "Core Processing Layer"
        DP[DocumentParser]
        TC[TextChunker]
        TP[TextPreprocessor]
        MODELS[Data Models]
    end

    %% Data Storage Layer
    subgraph "Data Storage"
        FS[File System Storage]
        FAISS[FAISS Vector Index]
        CACHE[In-Memory Cache]
    end

    %% External APIs
    subgraph "External APIs"
        ANTHROPIC[Anthropic Claude]
        MISTRAL[Mistral AI]
        OPENAI[OpenAI GPT]
        HF[HuggingFace]
    end

    %% Configuration
    subgraph "Configuration"
        CONFIG[config.py]
        ENV[Environment Variables]
    end

    %% Connections - External to Application
    UI --> APP
    MCP --> MCP_SERVER
    CLI --> APP

    %% Application Layer Connections
    APP --> INGEST
    APP --> SEARCH
    APP --> GEN
    MCP_SERVER --> INGEST
    MCP_SERVER --> SEARCH
    MCP_SERVER --> GEN

    %% Tool Layer to Service Layer
    INGEST --> DS
    INGEST --> VS
    INGEST --> ES
    INGEST --> OS
    SEARCH --> VS
    SEARCH --> ES
    SEARCH --> DS
    GEN --> LS
    GEN --> SEARCH

    %% Service Layer to Core Layer
    INGEST --> DP
    INGEST --> TC
    INGEST --> TP
    DS --> MODELS
    VS --> MODELS
    ES --> MODELS

    %% Service Layer to Storage
    DS --> FS
    VS --> FAISS
    DS --> CACHE
    VS --> CACHE

    %% Service Layer to External APIs
    LS --> ANTHROPIC
    LS --> MISTRAL
    LS --> OPENAI
    ES --> HF

    %% Configuration Connections
    CONFIG --> ENV
    CONFIG --> DS
    CONFIG --> VS
    CONFIG --> ES
    CONFIG --> LS
    CONFIG --> OS

    %% Styling
    classDef external fill:#e1f5fe
    classDef app fill:#f3e5f5
    classDef tool fill:#e8f5e8
    classDef service fill:#fff3e0
    classDef core fill:#fce4ec
    classDef storage fill:#f1f8e9
    classDef api fill:#fafafa
    classDef config fill:#e0f2f1

    class UI,MCP,CLI external
    class APP,MCP_SERVER app
    class INGEST,SEARCH,GEN,UTILS tool
    class DS,VS,ES,LS,OS service
    class DP,TC,TP,MODELS core
    class FS,FAISS,CACHE storage
    class ANTHROPIC,MISTRAL,OPENAI,HF api
    class CONFIG,ENV config
```

## 📊 Detailed Component Relationships

### 1. **Document Processing Pipeline**

```mermaid
sequenceDiagram
    participant User
    participant App
    participant IngestionTool
    participant DocumentParser
    participant TextChunker
    participant EmbeddingService
    participant DocumentStore
    participant VectorStore

    User->>App: Upload Document
    App->>IngestionTool: process_document()
    IngestionTool->>DocumentParser: parse_document()
    DocumentParser->>DocumentParser: Extract text content
    DocumentParser-->>IngestionTool: Document object
    IngestionTool->>DocumentStore: store_document()
    IngestionTool->>TextChunker: chunk_document()
    TextChunker->>TextChunker: Create text chunks
    TextChunker-->>IngestionTool: Chunk list
    IngestionTool->>EmbeddingService: generate_embeddings()
    EmbeddingService->>EmbeddingService: Create vector embeddings
    EmbeddingService-->>IngestionTool: Embeddings
    IngestionTool->>VectorStore: add_chunks()
    VectorStore->>VectorStore: Store in FAISS index
    VectorStore-->>IngestionTool: Success
    IngestionTool-->>App: Processing result
    App-->>User: Document processed
```

### 2. **Search and Retrieval Flow**

```mermaid
sequenceDiagram
    participant User
    participant App
    participant SearchTool
    participant EmbeddingService
    participant VectorStore
    participant DocumentStore

    User->>App: Search Query
    App->>SearchTool: search()
    SearchTool->>EmbeddingService: generate_single_embedding()
    EmbeddingService-->>SearchTool: Query embedding
    SearchTool->>VectorStore: search()
    VectorStore->>VectorStore: FAISS similarity search
    VectorStore-->>SearchTool: Search results
    SearchTool->>DocumentStore: get_document()
    DocumentStore-->>SearchTool: Document metadata
    SearchTool-->>App: Formatted results
    App-->>User: Search results
```

### 3. **Generative AI Flow**

```mermaid
sequenceDiagram
    participant User
    participant App
    participant GenerativeTool
    participant SearchTool
    participant LLMService
    participant ExternalLLM

    User->>App: Generate content request
    App->>GenerativeTool: summarize/generate_tags/answer_question()
    GenerativeTool->>SearchTool: search() [for context]
    SearchTool-->>GenerativeTool: Relevant chunks
    GenerativeTool->>LLMService: call_llm()
    LLMService->>ExternalLLM: API call
    ExternalLLM-->>LLMService: Generated content
    LLMService-->>GenerativeTool: Formatted response
    GenerativeTool-->>App: Generated content
    App-->>User: Final result
```

## 🗂️ File Structure and Dependencies

### **Root Level Files**
```
intelligent-content-organizer-MCP-Agent/
├── app.py                    # Main Gradio web application
├── mcp_server.py            # MCP protocol server
├── config.py                # Configuration management
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

### **Core Modules**
```
core/
├── models.py               # Data models (Document, Chunk, SearchResult)
├── document_parser.py      # Document parsing and extraction
├── chunker.py             # Text chunking strategies
├── text_preprocessor.py   # Text cleaning and preprocessing
└── __init__.py
```

### **Service Layer**
```
services/
├── document_store_service.py  # Document storage and retrieval
├── vector_store_service.py    # FAISS vector database
├── embedding_service.py       # Text embedding generation
├── llm_service.py            # LLM API integration
├── ocr_service.py            # OCR for image processing
└── __init__.py
```

### **MCP Tools**
```
mcp_tools/
├── ingestion_tool.py         # Document ingestion pipeline
├── search_tool.py           # Semantic search functionality
├── generative_tool.py       # AI content generation
├── utils.py                 # Utility functions
└── __init__.py
```

### **Data Storage**
```
data/
├── documents/
│   ├── content/             # Document text content
│   └── metadata/            # Document metadata
└── vector_store/
    ├── content_index.index  # FAISS vector index
    └── content_index_metadata.json
```

## 🔄 Data Flow Architecture

### **Document Ingestion Flow**
```
File Upload → Document Parser → Text Extraction → 
Text Preprocessing → Chunking → Embedding Generation → 
Document Storage → Vector Storage → Index Update
```

### **Search Flow**
```
Query → Text Preprocessing → Embedding Generation → 
Vector Search → Result Ranking → Document Retrieval → 
Result Formatting → Response
```

### **Generation Flow**
```
Request → Context Retrieval → Prompt Construction → 
LLM API Call → Response Processing → Content Generation → 
Result Formatting → Response
```

## 🎯 Key Design Patterns

### **1. Service Layer Pattern**
- **Purpose**: Encapsulate business logic and external dependencies
- **Components**: DocumentStoreService, VectorStoreService, EmbeddingService, etc.
- **Benefits**: Separation of concerns, testability, reusability

### **2. Tool Layer Pattern**
- **Purpose**: Expose functionality through MCP protocol
- **Components**: IngestionTool, SearchTool, GenerativeTool
- **Benefits**: Protocol abstraction, tool composition

### **3. Async/Await Pattern**
- **Purpose**: Non-blocking I/O operations
- **Usage**: Throughout the codebase for API calls and file operations
- **Benefits**: Scalability, responsiveness

### **4. Repository Pattern**
- **Purpose**: Abstract data access layer
- **Components**: DocumentStoreService, VectorStoreService
- **Benefits**: Data access abstraction, testability

### **5. Factory Pattern**
- **Purpose**: Create objects based on configuration
- **Usage**: LLM service selection, embedding model loading
- **Benefits**: Flexibility, configuration-driven behavior

## 🔧 Configuration Management

### **Environment Variables**
```python
# API Keys
ANTHROPIC_API_KEY
MISTRAL_API_KEY
HUGGINGFACE_API_KEY
OPENAI_API_KEY

# Model Configuration
EMBEDDING_MODEL
ANTHROPIC_MODEL
MISTRAL_MODEL
OPENAI_MODEL

# Storage Configuration
VECTOR_STORE_PATH
DOCUMENT_STORE_PATH
INDEX_NAME

# Processing Configuration
CHUNK_SIZE
CHUNK_OVERLAP
MAX_CONCURRENT_REQUESTS
```

## 🚀 Deployment Architecture

### **Development Mode**
```
User → Gradio UI → Local Services → Local Storage
```

### **Production Mode**
```
User → MCP Client → MCP Server → Services → Persistent Storage
```

### **Scalability Considerations**
- **Horizontal Scaling**: Multiple MCP server instances
- **Vertical Scaling**: GPU acceleration for embeddings
- **Caching**: In-memory document and embedding cache
- **Persistence**: File-based storage with backup strategies

## 🔍 Error Handling and Resilience

### **Error Handling Strategy**
1. **Graceful Degradation**: Fallback models and services
2. **Retry Logic**: API call retries with exponential backoff
3. **Circuit Breaker**: Prevent cascading failures
4. **Logging**: Comprehensive error logging and monitoring

### **Data Integrity**
1. **Transaction Safety**: Atomic operations for document processing
2. **Backup Strategies**: Regular index and metadata backups
3. **Validation**: Input validation and data sanitization
4. **Recovery**: Automatic recovery from partial failures

This architecture provides a robust, scalable, and maintainable foundation for intelligent content organization with clear separation of concerns and well-defined interfaces between components. 