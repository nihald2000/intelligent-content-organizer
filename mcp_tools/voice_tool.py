import logging
from typing import Dict, Any, Optional
import asyncio

logger = logging.getLogger(__name__)

class VoiceTool:
    """
    MCP Tool for voice-based Q&A using ElevenLabs conversational AI
    """
    
    def __init__(self, elevenlabs_service):
        """
        Initialize Voice Tool
        
        Args:
            elevenlabs_service: ElevenLabs service instance
        """
        self.elevenlabs_service = elevenlabs_service
    
    async def voice_qa(
        self, 
        question: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        MCP Tool: Ask a question using voice assistant
        
        Args:
            question: User's question (text or transcribed from voice)
            session_id: Optional session ID for conversation context
            
        Returns:
            Dictionary with answer, audio URL (if applicable), and sources
        """
        try:
            if not self.elevenlabs_service or not self.elevenlabs_service.is_available():
                return {
                    "success": False,
                    "error": "Voice assistant not configured. Please set ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID"
                }
            
            logger.info(f"Voice QA: {question}")
            
            # For text-based queries, we can use the RAG tool directly
            # This provides the backend for voice queries
            result = await self.elevenlabs_service.llamaindex_service.query(question)
            
            return {
                "success": True,
                "question": question,
                "answer": result,
                "session_id": session_id,
                "mode": "text"  # Could be "voice" if audio processing is involved
            }
            
        except Exception as e:
            logger.error(f"Voice QA failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "question": question
            }
