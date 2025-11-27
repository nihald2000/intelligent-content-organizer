import logging
import asyncio
from typing import Optional, Dict, Any, List
import json

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs.conversational_ai.conversation import Conversation, ClientTools
    from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ElevenLabs SDK not available. Voice features will be disabled.")

import config
from services.llamaindex_service import LlamaIndexService

logger = logging.getLogger(__name__)

class ElevenLabsService:
    """
    Service for integrating ElevenLabs Conversational AI with RAG capabilities.
    Provides voice-based interaction with the document library.
    """
    
    def __init__(self, llamaindex_service: LlamaIndexService):
        """
        Initialize ElevenLabs service with RAG integration
        
        Args:
            llamaindex_service: LlamaIndex service for document queries
        """
        self.config = config.config
        self.llamaindex_service = llamaindex_service
        self.client = None
        self.client_tools = None
        self.active_conversations: Dict[str, Conversation] = {}
        
        if not ELEVENLABS_AVAILABLE:
            logger.error("ElevenLabs SDK not installed. Run: pip install elevenlabs")
            return
        
        if not self.config.ELEVENLABS_API_KEY:
            logger.warning("ELEVENLABS_API_KEY not configured. Voice features will be limited.")
            return
        
        try:
            # Initialize ElevenLabs client
            self.client = ElevenLabs(api_key=self.config.ELEVENLABS_API_KEY)
            logger.info("ElevenLabs client initialized successfully")
            
            # Initialize client tools for custom tool registration
            self.client_tools = ClientTools()
            
            # Register RAG tool
            self._register_rag_tool()
            
            logger.info("ElevenLabs service initialized with RAG tool")
            
        except Exception as e:
            logger.error(f"Error initializing ElevenLabs service: {str(e)}")
    
    def _register_rag_tool(self):
        """Register RAG query tool with ElevenLabs agent"""
        if not self.client_tools:
            return
        
        try:
            # Register the query_documents tool
            self.client_tools.register(
                name="query_documents",
                handler=self._rag_query_tool,
                is_async=True
            )
            
            logger.info("RAG tool 'query_documents' registered successfully")
            
        except Exception as e:
            logger.error(f"Error registering RAG tool: {str(e)}")
    
    async def _rag_query_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Custom tool for querying documents using LlamaIndex agentic RAG
        
        Args:
            params: Dictionary containing the query
                - query (str): The user's question or search query
        
        Returns:
            Dictionary with answer and metadata
        """
        try:
            query = params.get("query", "")
            
            if not query:
                return {
                    "error": "No query provided",
                    "answer": "I didn't receive a question to search for."
                }
            
            logger.info(f"RAG tool called with query: '{query}'")
            
            # Query the LlamaIndex agentic RAG system
            try:
                result = await asyncio.wait_for(
                    self.llamaindex_service.query(query),
                    timeout=self.config.CONVERSATION_TIMEOUT
                )
                
                logger.info(f"RAG query successful")
                
                return {
                    "answer": result,
                    "source": "document_library",
                    "confidence": "high"
                }
                
            except asyncio.TimeoutError:
                logger.error("RAG query timeout")
                return {
                    "error": "timeout",
                    "answer": "The search took too long. Please try a simpler question."
                }
            
        except Exception as e:
            logger.error(f"Error in RAG query tool: {str(e)}")
            return {
                "error": str(e),
                "answer": f"I encountered an error searching the documents: {str(e)}"
            }
    
    def create_conversation(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Optional[Conversation]:
        """
        Create a new conversation session
        
        Args:
            agent_id: ElevenLabs agent ID (uses config default if not provided)
            session_id: Optional session ID for tracking
        
        Returns:
            Conversation object or None if initialization fails
        """
        if not self.client:
            logger.error("ElevenLabs client not initialized")
            return None
        
        try:
            agent_id = agent_id or self.config.ELEVENLABS_AGENT_ID
            
            if not agent_id:
                logger.error("No agent ID provided or configured")
                return None
            
            # Create audio interface for real-time audio
            audio_interface = DefaultAudioInterface()
            
            # Create conversation with RAG tool
            conversation = Conversation(
                client=self.client,
                agent_id=agent_id,
                requires_auth=True,
                audio_interface=audio_interface,
                client_tools=self.client_tools
            )
            
            # Store conversation if session ID provided
            if session_id:
                self.active_conversations[session_id] = conversation
            
            logger.info(f"Created conversation for agent: {agent_id}")
            return conversation
            
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            return None
    
    async def process_voice_query(
        self,
        audio_file_path: str,
        agent_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a voice query file and return response
        
        Args:
            audio_file_path: Path to audio file
            agent_id: Optional agent ID
        
        Returns:
            Dictionary with transcription, answer, and metadata
        """
        try:
            # For now, this is a placeholder for file-based processing
            # ElevenLabs Conversational AI is primarily WebSocket-based
            # This would be used for async/batch processing
            
            logger.info(f"Processing voice query from: {audio_file_path}")
            
            # This would require additional implementation for file upload
            # and processing through ElevenLabs API
            
            return {
                "status": "pending",
                "message": "Voice query processing requires WebSocket connection",
                "file": audio_file_path
            }
            
        except Exception as e:
            logger.error(f"Error processing voice query: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def end_conversation(self, session_id: str) -> bool:
        """
        End an active conversation session
        
        Args:
            session_id: Session identifier
        
        Returns:
            True if conversation ended successfully
        """
        try:
            if session_id in self.active_conversations:
                conversation = self.active_conversations[session_id]
                conversation.end_session()
                del self.active_conversations[session_id]
                logger.info(f"Ended conversation: {session_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error ending conversation: {str(e)}")
            return False
    
    def get_available_voices(self) -> List[Dict[str, str]]:
        """
        Get list of available voice models
        
        Returns:
            List of voice model information
        """
        try:
            if not self.client:
                return []
            
            # Get voices from ElevenLabs API
            voices = self.client.voices.get_all()
            
            return [
                {
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "category": voice.category if hasattr(voice, 'category') else "general"
                }
                for voice in voices.voices
            ]
            
        except Exception as e:
            logger.error(f"Error getting voices: {str(e)}")
            return []
    
    def is_available(self) -> bool:
        """Check if ElevenLabs service is available and configured"""
        return ELEVENLABS_AVAILABLE and self.client is not None

    async def test_connection(self) -> Dict[str, Any]:
        """
        Test ElevenLabs API connection
        
        Returns:
            Dictionary with test results
        """
        try:
            if not self.client:
                return {
                    "status": "error",
                    "message": "Client not initialized"
                }
            
            # Try to fetch user info or voices as a connection test
            voices = self.get_available_voices()
            
            return {
                "status": "success",
                "message": "ElevenLabs API connected",
                "voices_available": len(voices),
                "rag_tool_registered": self.client_tools is not None
            }
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
