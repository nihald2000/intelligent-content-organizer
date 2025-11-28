import os
from typing import Optional
from dotenv import load_dotenv 

load_dotenv()


class Config:
    # API Keys
    NEBIUS_API_KEY: Optional[str] = os.getenv("NEBIUS_API_KEY")
    MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY", os.getenv("HF_TOKEN"))
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # NEBIUS Configuration (OpenAI OSS models)
    NEBIUS_BASE_URL: str = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/")
    NEBIUS_MODEL: str = os.getenv("NEBIUS_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    
    # Model Configuration
    # Using OpenAI managed embeddings for performance/quality
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-large-2407")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.1-chat-latest")
    FAST_MODEL: str = os.getenv("FAST_MODEL", "gpt-5-mini")
    
    # Vector Store Configuration
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    DOCUMENT_STORE_PATH: str = os.getenv("DOCUMENT_STORE_PATH", "./data/documents")
    INDEX_NAME: str = os.getenv("INDEX_NAME", "content_index")
    
    # Processing Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    # Search Configuration
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    
    # OCR Configuration
    TESSERACT_PATH: Optional[str] = os.getenv("TESSERACT_PATH")
    OCR_LANGUAGE: str = os.getenv("OCR_LANGUAGE", "eng")
    
    # ElevenLabs Configuration
    ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_AGENT_ID: Optional[str] = os.getenv("ELEVENLABS_AGENT_ID")
    ELEVENLABS_VOICE_MODEL: str = os.getenv("ELEVENLABS_VOICE_MODEL", "Rachel")
    
    # App Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "7860"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

config = Config()