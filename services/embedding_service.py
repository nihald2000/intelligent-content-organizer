import logging
import asyncio
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import config

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.config = config.config
        self.model_name = self.config.EMBEDDING_MODEL
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model lazily
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Embedding model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            # Fallback to a smaller model
            try:
                self.model_name = "all-MiniLM-L6-v2"
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Loaded fallback embedding model: {self.model_name}")
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {str(fallback_error)}")
                raise
    
    async def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            return []
        
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            # Filter out empty texts
            non_empty_texts = [text for text in texts if text and text.strip()]
            if not non_empty_texts:
                logger.warning("No non-empty texts provided for embedding")
                return []
            
            logger.info(f"Generating embeddings for {len(non_empty_texts)} texts")
            
            # Process in batches to manage memory
            all_embeddings = []
            for i in range(0, len(non_empty_texts), batch_size):
                batch = non_empty_texts[i:i + batch_size]
                
                # Run embedding generation in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    None, 
                    self._generate_batch_embeddings, 
                    batch
                )
                all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Generated {len(all_embeddings)} embeddings")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts (synchronous)"""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
            
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=len(texts)
            )
            
            # Convert to list of lists
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error in batch embedding generation: {str(e)}")
            raise
    
    async def generate_single_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        if not text or not text.strip():
            return None
        
        try:
            embeddings = await self.generate_embeddings([text])
            return embeddings[0] if embeddings else None
        except Exception as e:
            logger.error(f"Error generating single embedding: {str(e)}")
            return None
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Could not determine embedding dimension")
        return dimension
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            return float(similarity)
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def compute_similarities(self, query_embedding: List[float], embeddings: List[List[float]]) -> List[float]:
        """Compute similarities between a query embedding and multiple embeddings"""
        try:
            query_emb = np.array(query_embedding)
            emb_matrix = np.array(embeddings)
            
            # Compute cosine similarities
            similarities = np.dot(emb_matrix, query_emb) / (
                np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(query_emb)
            )
            
            return similarities.tolist()
        except Exception as e:
            logger.error(f"Error computing similarities: {str(e)}")
            return [0.0] * len(embeddings)
    
    async def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed a list of chunks and add embeddings to them"""
        if not chunks:
            return []
        
        try:
            # Extract texts
            texts = [chunk.get('content', '') for chunk in chunks]
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(texts)
            
            # Add embeddings to chunks
            embedded_chunks = []
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk_copy = chunk.copy()
                    chunk_copy['embedding'] = embeddings[i]
                    embedded_chunks.append(chunk_copy)
                else:
                    logger.warning(f"No embedding generated for chunk {i}")
                    embedded_chunks.append(chunk)
            
            return embedded_chunks
        except Exception as e:
            logger.error(f"Error embedding chunks: {str(e)}")
            raise
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate that an embedding is properly formatted"""
        try:
            if not embedding:
                return False
            
            if not isinstance(embedding, list):
                return False
            
            if len(embedding) != self.get_embedding_dimension():
                return False
            
            # Check for NaN or infinite values
            emb_array = np.array(embedding)
            if np.isnan(emb_array).any() or np.isinf(emb_array).any():
                return False
            
            return True
        except Exception:
            return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        try:
            return {
                "model_name": self.model_name,
                "device": self.device,
                "embedding_dimension": self.get_embedding_dimension(),
                "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown'),
                "model_loaded": self.model is not None
            }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}