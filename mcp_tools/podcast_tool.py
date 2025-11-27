import logging
from typing import Dict, Any, List
from dataclasses import asdict

logger = logging.getLogger(__name__)

class PodcastTool:
    """
    MCP Tool for podcast generation from documents
    """
    
    def __init__(self, podcast_generator):
        """
        Initialize Podcast Tool
        
        Args:
            podcast_generator: PodcastGeneratorService instance
        """
        self.podcast_generator = podcast_generator
    
    async def generate_podcast(
        self,
        document_ids: List[str],
        style: str = "conversational",
        duration_minutes: int = 10,
        host1_voice: str = "Rachel",
        host2_voice: str = "Adam"
    ) -> Dict[str, Any]:
        """
        MCP Tool: Generate podcast from documents
        
        Args:
            document_ids: List of document IDs to generate podcast from
            style: Podcast style (conversational, educational, technical, casual)
            duration_minutes: Target duration in minutes
            host1_voice: Voice name for first host
            host2_voice: Voice name for second host
            
        Returns:
            Dictionary with podcast ID, audio URL, transcript, and metadata
        """
        try:
            if not document_ids or len(document_ids) == 0:
                return {
                    "success": False,
                    "error": "No documents provided. Please select at least one document."
                }
            
            logger.info(f"Generating podcast from {len(document_ids)} documents")
            
            # Generate podcast using service
            result = await self.podcast_generator.generate_podcast(
                document_ids=document_ids,
                style=style,
                duration_minutes=duration_minutes,
                host1_voice=host1_voice,
                host2_voice=host2_voice
            )
            
            if result.success:
                return {
                    "success": True,
                    "podcast_id": result.podcast_id,
                    "audio_file": result.audio_file_path,
                    "audio_url": f"/data/podcasts/{result.podcast_id}.mp3",
                    "transcript": result.transcript,
                    "metadata": asdict(result.metadata) if result.metadata else {},
                    "generation_time": result.generation_time,
                    "message": f"Podcast generated successfully! Duration: {result.metadata.duration_seconds/60:.1f} minutes"
                }
            else:
                return {
                    "success": False,
                    "error": result.error or "Unknown error during podcast generation"
                }
                
        except Exception as e:
            logger.error(f"Podcast generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_podcasts(self, limit: int = 10) -> Dict[str, Any]:
        """
        List previously generated podcasts
        
        Args:
            limit: Maximum number of podcasts to return
            
        Returns:
            Dictionary with list of podcast metadata
        """
        try:
            podcasts = self.podcast_generator.list_podcasts(limit=limit)
            
            return {
                "success": True,
                "podcasts": [asdict(p) for p in podcasts],
                "total": len(podcasts)
            }
        except Exception as e:
            logger.error(f"Failed to list podcasts: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "podcasts": []
            }
    
    def get_podcast(self, podcast_id: str) -> Dict[str, Any]:
        """
        Get specific podcast by ID
        
        Args:
            podcast_id: Podcast identifier
            
        Returns:
            Dictionary with podcast metadata
        """
        try:
            podcast = self.podcast_generator.get_podcast(podcast_id)
            
            if podcast:
                return {
                    "success": True,
                    "podcast": asdict(podcast)
                }
            else:
                return {
                    "success": False,
                    "error": "Podcast not found"
                }
        except Exception as e:
            logger.error(f"Failed to get podcast: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
