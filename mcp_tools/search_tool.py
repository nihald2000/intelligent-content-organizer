import logging
from typing import List, Dict, Any, Optional
import asyncio

from core.models import SearchResult
from services.vector_store_service import VectorStoreService
from services.embedding_service import EmbeddingService
from services.document_store_service import DocumentStoreService
import config

logger = logging.getLogger(__name__)

class SearchTool:
    def __init__(self, vector_store: VectorStoreService, embedding_service: EmbeddingService,
                 document_store: Optional[DocumentStoreService] = None, llamaindex_service: Any = None):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.document_store = document_store
        self.llamaindex_service = llamaindex_service
        self.config = config.config
    
    async def search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None,
                    similarity_threshold: Optional[float] = None) -> List[SearchResult]:
        """Perform semantic search"""
        try:
            if not query.strip():
                logger.warning("Empty search query provided")
                return []
            
            # Use default threshold if not provided
            if similarity_threshold is None:
                similarity_threshold = self.config.SIMILARITY_THRESHOLD
            
            logger.info(f"Performing semantic search for: '{query}' (top_k={top_k})")
            
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_single_embedding(query)
            
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Perform vector search
            results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.score >= similarity_threshold
            ]
            
            logger.info(f"Found {len(filtered_results)} results above threshold {similarity_threshold}")
            
            # Enhance results with additional metadata if document store is available
            if self.document_store:
                enhanced_results = await self._enhance_results_with_metadata(filtered_results)
                return enhanced_results
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error performing semantic search: {str(e)}")
            return []

    async def agentic_search(self, query: str) -> str:
        """Perform agentic search using LlamaIndex"""
        if not self.llamaindex_service:
            logger.warning("LlamaIndex service not available for agentic search")
            return "Agentic search not available."
        
        try:
            logger.info(f"Performing agentic search for: '{query}'")
            return await self.llamaindex_service.query(query)
        except Exception as e:
            logger.error(f"Error performing agentic search: {str(e)}")
            return f"Error performing agentic search: {str(e)}"
    
    async def _enhance_results_with_metadata(self, results: List[SearchResult]) -> List[SearchResult]:
        """Enhance search results with document metadata"""
        try:
            enhanced_results = []
            
            for result in results:
                try:
                    # Get document metadata
                    document = await self.document_store.get_document(result.document_id)
                    
                    if document:
                        # Add document metadata to result
                        enhanced_metadata = {
                            **result.metadata,
                            "document_filename": document.filename,
                            "document_type": document.doc_type.value,
                            "document_tags": document.tags,
                            "document_category": document.category,
                            "document_created_at": document.created_at.isoformat(),
                            "document_summary": document.summary
                        }
                        
                        enhanced_result = SearchResult(
                            chunk_id=result.chunk_id,
                            document_id=result.document_id,
                            content=result.content,
                            score=result.score,
                            metadata=enhanced_metadata
                        )
                        
                        enhanced_results.append(enhanced_result)
                    else:
                        # Document not found, use original result
                        enhanced_results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Error enhancing result {result.chunk_id}: {str(e)}")
                    enhanced_results.append(result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error enhancing results: {str(e)}")
            return results
    
    async def multi_query_search(self, queries: List[str], top_k: int = 5, 
                               aggregate_method: str = "merge") -> List[SearchResult]:
        """Perform search with multiple queries and aggregate results"""
        try:
            all_results = []
            
            # Perform search for each query
            for query in queries:
                if query.strip():
                    query_results = await self.search(query, top_k)
                    all_results.extend(query_results)
            
            if not all_results:
                return []
            
            # Aggregate results
            if aggregate_method == "merge":
                return await self._merge_results(all_results, top_k)
            elif aggregate_method == "intersect":
                return await self._intersect_results(all_results, top_k)
            elif aggregate_method == "average":
                return await self._average_results(all_results, top_k)
            else:
                # Default to merge
                return await self._merge_results(all_results, top_k)
                
        except Exception as e:
            logger.error(f"Error in multi-query search: {str(e)}")
            return []
    
    async def _merge_results(self, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Merge results and remove duplicates, keeping highest scores"""
        try:
            # Group by chunk_id and keep highest score
            chunk_scores = {}
            chunk_results = {}
            
            for result in results:
                chunk_id = result.chunk_id
                if chunk_id not in chunk_scores or result.score > chunk_scores[chunk_id]:
                    chunk_scores[chunk_id] = result.score
                    chunk_results[chunk_id] = result
            
            # Sort by score and return top_k
            merged_results = list(chunk_results.values())
            merged_results.sort(key=lambda x: x.score, reverse=True)
            
            return merged_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error merging results: {str(e)}")
            return results[:top_k]
    
    async def _intersect_results(self, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Find chunks that appear in multiple queries"""
        try:
            # Count occurrences of each chunk
            chunk_counts = {}
            chunk_results = {}
            
            for result in results:
                chunk_id = result.chunk_id
                chunk_counts[chunk_id] = chunk_counts.get(chunk_id, 0) + 1
                
                if chunk_id not in chunk_results or result.score > chunk_results[chunk_id].score:
                    chunk_results[chunk_id] = result
            
            # Filter chunks that appear more than once
            intersect_results = [
                result for chunk_id, result in chunk_results.items()
                if chunk_counts[chunk_id] > 1
            ]
            
            # Sort by score
            intersect_results.sort(key=lambda x: x.score, reverse=True)
            
            return intersect_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error intersecting results: {str(e)}")
            return []
    
    async def _average_results(self, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Average scores for chunks that appear multiple times"""
        try:
            # Group by chunk_id and calculate average scores
            chunk_groups = {}
            
            for result in results:
                chunk_id = result.chunk_id
                if chunk_id not in chunk_groups:
                    chunk_groups[chunk_id] = []
                chunk_groups[chunk_id].append(result)
            
            # Calculate average scores
            averaged_results = []
            for chunk_id, group in chunk_groups.items():
                avg_score = sum(r.score for r in group) / len(group)
                
                # Use the result with the highest individual score but update the score to average
                best_result = max(group, key=lambda x: x.score)
                averaged_result = SearchResult(
                    chunk_id=best_result.chunk_id,
                    document_id=best_result.document_id,
                    content=best_result.content,
                    score=avg_score,
                    metadata={
                        **best_result.metadata,
                        "query_count": len(group),
                        "score_range": f"{min(r.score for r in group):.3f}-{max(r.score for r in group):.3f}"
                    }
                )
                averaged_results.append(averaged_result)
            
            # Sort by average score
            averaged_results.sort(key=lambda x: x.score, reverse=True)
            
            return averaged_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error averaging results: {str(e)}")
            return results[:top_k]
    
    async def search_by_document(self, document_id: str, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search within a specific document"""
        try:
            filters = {"document_id": document_id}
            return await self.search(query, top_k, filters)
            
        except Exception as e:
            logger.error(f"Error searching within document {document_id}: {str(e)}")
            return []
    
    async def search_by_category(self, category: str, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search within documents of a specific category"""
        try:
            if not self.document_store:
                logger.warning("Document store not available for category search")
                return await self.search(query, top_k)
            
            # Get documents in the category
            documents = await self.document_store.list_documents(
                limit=1000,  # Adjust as needed
                filters={"category": category}
            )
            
            if not documents:
                logger.info(f"No documents found in category '{category}'")
                return []
            
            # Extract document IDs
            document_ids = [doc.id for doc in documents]
            
            # Search with document ID filter
            filters = {"document_ids": document_ids}
            return await self.search(query, top_k, filters)
            
        except Exception as e:
            logger.error(f"Error searching by category {category}: {str(e)}")
            return []
    
    async def search_with_date_range(self, query: str, start_date, end_date, top_k: int = 5) -> List[SearchResult]:
        """Search documents within a date range"""
        try:
            if not self.document_store:
                logger.warning("Document store not available for date range search")
                return await self.search(query, top_k)
            
            # Get documents in the date range
            documents = await self.document_store.list_documents(
                limit=1000,  # Adjust as needed
                filters={
                    "created_after": start_date,
                    "created_before": end_date
                }
            )
            
            if not documents:
                logger.info(f"No documents found in date range")
                return []
            
            # Extract document IDs
            document_ids = [doc.id for doc in documents]
            
            # Search with document ID filter
            filters = {"document_ids": document_ids}
            return await self.search(query, top_k, filters)
            
        except Exception as e:
            logger.error(f"Error searching with date range: {str(e)}")
            return []
    
    async def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on partial query"""
        try:
            # This is a simple implementation
            # In a production system, you might want to use a more sophisticated approach
            
            if len(partial_query) < 2:
                return []
            
            # Search for the partial query
            results = await self.search(partial_query, top_k=20)
            
            # Extract potential query expansions from content
            suggestions = set()
            
            for result in results:
                content_words = result.content.lower().split()
                for i, word in enumerate(content_words):
                    if partial_query.lower() in word:
                        # Add the word itself
                        suggestions.add(word.strip('.,!?;:'))
                        
                        # Add phrases that include this word
                        if i > 0:
                            phrase = f"{content_words[i-1]} {word}".strip('.,!?;:')
                            suggestions.add(phrase)
                        if i < len(content_words) - 1:
                            phrase = f"{word} {content_words[i+1]}".strip('.,!?;:')
                            suggestions.add(phrase)
            
            # Filter and sort suggestions
            filtered_suggestions = [
                s for s in suggestions 
                if len(s) > len(partial_query) and s.startswith(partial_query.lower())
            ]
            
            return sorted(filtered_suggestions)[:limit]
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {str(e)}")
            return []
    
    async def explain_search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Provide detailed explanation of search process and results"""
        try:
            explanation = {
                "query": query,
                "steps": [],
                "results_analysis": {},
                "performance_metrics": {}
            }
            
            # Step 1: Query processing
            explanation["steps"].append({
                "step": "query_processing",
                "description": "Processing and normalizing the search query",
                "details": {
                    "original_query": query,
                    "cleaned_query": query.strip(),
                    "query_length": len(query)
                }
            })
            
            # Step 2: Embedding generation
            import time
            start_time = time.time()
            
            query_embedding = await self.embedding_service.generate_single_embedding(query)
            
            embedding_time = time.time() - start_time
            
            explanation["steps"].append({
                "step": "embedding_generation",
                "description": "Converting query to vector embedding",
                "details": {
                    "embedding_dimension": len(query_embedding) if query_embedding else 0,
                    "generation_time_ms": round(embedding_time * 1000, 2)
                }
            })
            
            # Step 3: Vector search
            start_time = time.time()
            
            results = await self.vector_store.search(query_embedding, top_k)
            
            search_time = time.time() - start_time
            
            explanation["steps"].append({
                "step": "vector_search",
                "description": "Searching vector database for similar content",
                "details": {
                    "search_time_ms": round(search_time * 1000, 2),
                    "results_found": len(results),
                    "top_score": results[0].score if results else 0,
                    "score_range": f"{min(r.score for r in results):.3f}-{max(r.score for r in results):.3f}" if results else "N/A"
                }
            })
            
            # Results analysis
            if results:
                explanation["results_analysis"] = {
                    "total_results": len(results),
                    "average_score": sum(r.score for r in results) / len(results),
                    "unique_documents": len(set(r.document_id for r in results)),
                    "content_lengths": [len(r.content) for r in results]
                }
            
            # Performance metrics
            explanation["performance_metrics"] = {
                "total_time_ms": round((embedding_time + search_time) * 1000, 2),
                "embedding_time_ms": round(embedding_time * 1000, 2),
                "search_time_ms": round(search_time * 1000, 2)
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining search: {str(e)}")
            return {"error": str(e)}