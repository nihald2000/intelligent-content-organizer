import logging
import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss
from pathlib import Path
import asyncio
import json

from core.models import SearchResult, Chunk
import config

logger = logging.getLogger(__name__)

class VectorStoreService:
    def __init__(self):
        self.config = config.config
        self.index = None
        self.chunks_metadata = {}  # Maps index position to chunk metadata
        self.dimension = None
        
        # Paths
        self.store_path = Path(self.config.VECTOR_STORE_PATH)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.store_path / f"{self.config.INDEX_NAME}.index"
        self.metadata_path = self.store_path / f"{self.config.INDEX_NAME}_metadata.json"
        
        # Load existing index if available
        self._load_index()
    
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                logger.info("Loading existing FAISS index...")
                
                # Load FAISS index
                self.index = faiss.read_index(str(self.index_path))
                self.dimension = self.index.d
                
                # Load metadata
                with open(self.metadata_path, 'r') as f:
                    self.chunks_metadata = json.load(f)
                
                logger.info(f"Loaded index with {self.index.ntotal} vectors, dimension {self.dimension}")
            else:
                logger.info("No existing index found, will create new one")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
    
    def _initialize_index(self, dimension: int):
        """Initialize a new FAISS index"""
        try:
            # Use IndexFlatIP for cosine similarity (since embeddings are normalized)
            self.index = faiss.IndexFlatIP(dimension)
            self.dimension = dimension
            self.chunks_metadata = {}
            logger.info(f"Initialized new FAISS index with dimension {dimension}")
        except Exception as e:
            logger.error(f"Error initializing index: {str(e)}")
            raise
    
    async def add_chunks(self, chunks: List[Chunk]) -> bool:
        """Add chunks to the vector store"""
        if not chunks:
            return True
        
        try:
            # Extract embeddings and metadata
            embeddings = []
            new_metadata = {}
            
            for chunk in chunks:
                if chunk.embedding and len(chunk.embedding) > 0:
                    embeddings.append(chunk.embedding)
                    # Store metadata using the current index position
                    current_index = len(self.chunks_metadata) + len(embeddings) - 1
                    new_metadata[str(current_index)] = {
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "start_pos": chunk.start_pos,
                        "end_pos": chunk.end_pos,
                        "metadata": chunk.metadata
                    }
            
            if not embeddings:
                logger.warning("No valid embeddings found in chunks")
                return False
            
            # Initialize index if needed
            if self.index is None:
                self._initialize_index(len(embeddings[0]))
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Update metadata
            self.chunks_metadata.update(new_metadata)
            
            # Save index and metadata
            await self._save_index()
            
            logger.info(f"Added {len(embeddings)} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {str(e)}")
            return False
    
    async def search(self, query_embedding: List[float], top_k: int = 5, 
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar chunks"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("No index available or index is empty")
            return []
        
        try:
            # Convert query embedding to numpy array
            query_array = np.array([query_embedding], dtype=np.float32)
            
            # Perform search
            scores, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
            
            # Convert results to SearchResult objects
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                chunk_metadata = self.chunks_metadata.get(str(idx))
                if chunk_metadata:
                    # Apply filters if specified
                    if filters and not self._apply_filters(chunk_metadata, filters):
                        continue
                    
                    result = SearchResult(
                        chunk_id=chunk_metadata["chunk_id"],
                        document_id=chunk_metadata["document_id"],
                        content=chunk_metadata["content"],
                        score=float(score),
                        metadata=chunk_metadata.get("metadata", {})
                    )
                    results.append(result)
            
            # Sort by score (descending)
            results.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def _apply_filters(self, chunk_metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filters to chunk metadata"""
        try:
            for key, value in filters.items():
                if key == "document_id":
                    if chunk_metadata.get("document_id") != value:
                        return False
                elif key == "document_ids":
                    if chunk_metadata.get("document_id") not in value:
                        return False
                elif key == "content_length_min":
                    if len(chunk_metadata.get("content", "")) < value:
                        return False
                elif key == "content_length_max":
                    if len(chunk_metadata.get("content", "")) > value:
                        return False
                # Add more filter types as needed
            
            return True
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return True
    
    async def _save_index(self):
        """Save the FAISS index and metadata to disk"""
        try:
            if self.index is not None:
                # Save FAISS index
                faiss.write_index(self.index, str(self.index_path))
                
                # Save metadata
                with open(self.metadata_path, 'w') as f:
                    json.dump(self.chunks_metadata, f, indent=2)
                
                logger.debug("Saved index and metadata to disk")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            return {
                "total_vectors": self.index.ntotal if self.index else 0,
                "dimension": self.dimension,
                "index_type": type(self.index).__name__ if self.index else None,
                "metadata_entries": len(self.chunks_metadata),
                "index_file_exists": self.index_path.exists(),
                "metadata_file_exists": self.metadata_path.exists()
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a specific document"""
        try:
            # Find indices to remove
            indices_to_remove = []
            for idx, metadata in self.chunks_metadata.items():
                if metadata.get("document_id") == document_id:
                    indices_to_remove.append(int(idx))
            
            if not indices_to_remove:
                logger.warning(f"No chunks found for document {document_id}")
                return False
            
            # FAISS doesn't support removing individual vectors efficiently
            # We need to rebuild the index without the removed vectors
            if self.index and self.index.ntotal > 0:
                # Get all embeddings except the ones to remove
                all_embeddings = []
                new_metadata = {}
                new_index = 0
                
                for old_idx in range(self.index.ntotal):
                    if old_idx not in indices_to_remove:
                        # Get the embedding from FAISS
                        embedding = self.index.reconstruct(old_idx)
                        all_embeddings.append(embedding)
                        
                        # Update metadata with new index
                        old_metadata = self.chunks_metadata.get(str(old_idx))
                        if old_metadata:
                            new_metadata[str(new_index)] = old_metadata
                            new_index += 1
                
                # Rebuild index
                if all_embeddings:
                    self._initialize_index(self.dimension)
                    embeddings_array = np.array(all_embeddings, dtype=np.float32)
                    self.index.add(embeddings_array)
                    self.chunks_metadata = new_metadata
                else:
                    # No embeddings left, create empty index
                    self._initialize_index(self.dimension)
                
                # Save updated index
                await self._save_index()
            
            logger.info(f"Deleted {len(indices_to_remove)} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document chunks: {str(e)}")
            return False
    
    async def clear_all(self) -> bool:
        """Clear all data from the vector store"""
        try:
            self.index = None
            self.chunks_metadata = {}
            self.dimension = None
            
            # Remove files
            if self.index_path.exists():
                self.index_path.unlink()
            if self.metadata_path.exists():
                self.metadata_path.unlink()
            
            logger.info("Cleared all data from vector store")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False