
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration management for API keys and settings"""
    
    # API Keys - Only 2 needed, both with free tiers!
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    # ChromaDB Settings (completely free local storage)
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    CHROMA_COLLECTION_NAME = "knowledge_base"
    
    # MCP Server Settings
    MCP_SERVER_NAME = "intelligent-content-organizer"
    MCP_SERVER_VERSION = "1.0.0"
    MCP_SERVER_DESCRIPTION = "AI-powered knowledge management with automatic tagging and semantic search"
    
    # Processing Settings
    MAX_FILE_SIZE_MB = 50
    SUPPORTED_FILE_TYPES = [
        ".pdf", ".txt", ".docx", ".doc", ".html", ".htm",
        ".md", ".csv", ".json", ".xml", ".rtf"
    ]
    
    # Model Settings
    MISTRAL_MODEL = "mistral-small-latest"  # Free tier available
    CLAUDE_MODEL = "claude-3-haiku-20240307"  # Free tier available
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Completely free
    
    # Feature Flags - Enable/disable based on API availability
    USE_MISTRAL_FOR_TAGS = bool(MISTRAL_API_KEY)
    USE_CLAUDE_FOR_SUMMARY = bool(ANTHROPIC_API_KEY)
    
    # Free alternatives settings
    ENABLE_FREE_FALLBACKS = True  # Always use free methods when APIs fail
    
    @classmethod
    def validate(cls):
        """Validate configuration - now more flexible"""
        warnings = []
        
        if not cls.MISTRAL_API_KEY:
            warnings.append("MISTRAL_API_KEY not set - will use free tag generation")
        
        if not cls.ANTHROPIC_API_KEY:
            warnings.append("ANTHROPIC_API_KEY not set - will use free summarization")
        
        if warnings:
            print("⚠️  Configuration warnings:")
            for warning in warnings:
                print(f"   - {warning}")
            print("\n✅ The app will still work using free alternatives!")
        else:
            print("✅ All API keys configured")
        
        return True
    
    @classmethod
    def get_status(cls):
        """Get configuration status for display"""
        return {
            "mistral_configured": bool(cls.MISTRAL_API_KEY),
            "anthropic_configured": bool(cls.ANTHROPIC_API_KEY),
            "free_fallbacks_enabled": cls.ENABLE_FREE_FALLBACKS,
            "supported_formats": cls.SUPPORTED_FILE_TYPES,
            "embedding_model": cls.EMBEDDING_MODEL
        }