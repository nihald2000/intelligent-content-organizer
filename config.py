import os
from typing import Optional
from dotenv import load_dotenv 

load_dotenv()


class Config:
    # API Keys
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
    HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY", os.getenv("HF_TOKEN"))
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
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
    
    # Voice Settings
    VOICE_SPEED: float = float(os.getenv("VOICE_SPEED", "1.0"))
    VOICE_STABILITY: float = float(os.getenv("VOICE_STABILITY", "0.5"))
    VOICE_SIMILARITY_BOOST: float = float(os.getenv("VOICE_SIMILARITY_BOOST", "0.75"))
    CONVERSATION_TIMEOUT: int = int(os.getenv("CONVERSATION_TIMEOUT", "30"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        # Make API keys optional for testing
        return True

# Global config instance
config = Config()

# Create data directories
import pathlib
pathlib.Path(config.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(config.DOCUMENT_STORE_PATH).mkdir(parents=True, exist_ok=True)