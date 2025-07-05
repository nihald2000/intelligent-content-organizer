import logging
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import pickle
from datetime import datetime
import asyncio

from core.models import Document, DocumentType
import config

logger = logging.getLogger(__name__)

class DocumentStoreService:
    def __init__(self):
        self.config = config.config
        self.store_path = Path(self.config.DOCUMENT_STORE_PATH)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Separate paths for metadata and content
        self.metadata_path = self.store_path / "metadata"
        self.content_path = self.store_path / "content"
        
        self.metadata_path.mkdir(exist_ok=True)
        self.content_path.mkdir(exist_ok=True)
        
        # In-memory cache for frequently accessed documents
        self._cache = {}
        self._cache_size_limit = 100
    
    async def store_document(self, document: Document) -> bool:
        """Store a document and its metadata"""
        try:
            # Store metadata
            metadata_file = self.metadata_path / f"{document.id}.json"
            metadata = {
                "id": document.id,
                "filename": document.filename,
                "doc_type": document.doc_type.value,
                "file_size": document.file_size,
                "created_at": document.created_at.isoformat(),
                "metadata": document.metadata,
                "tags": document.tags,
                "summary": document.summary,
                "category": document.category,
                "language": document.language,
                "content_length": len(document.content)
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Store content separately (can be large)
            content_file = self.content_path / f"{document.id}.txt"
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(document.content)
            
            # Cache the document
            self._add_to_cache(document.id, document)
            
            logger.info(f"Stored document {document.id} ({document.filename})")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document {document.id}: {str(e)}")
            return False
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by ID"""
        try:
            # Check cache first
            if document_id in self._cache:
                return self._cache[document_id]
            
            # Load from disk
            metadata_file = self.metadata_path / f"{document_id}.json"
            content_file = self.content_path / f"{document_id}.txt"
            
            if not metadata_file.exists() or not content_file.exists():
                return None
            
            # Load metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Load content
            with open(content_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create document object
            document = Document(
                id=metadata["id"],
                filename=metadata["filename"],
                content=content,
                doc_type=DocumentType(metadata["doc_type"]),
                file_size=metadata["file_size"],
                created_at=datetime.fromisoformat(metadata["created_at"]),
                metadata=metadata.get("metadata", {}),
                tags=metadata.get("tags", []),
                summary=metadata.get("summary"),
                category=metadata.get("category"),
                language=metadata.get("language")
            )
            
            # Add to cache
            self._add_to_cache(document_id, document)
            
            return document
            
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {str(e)}")
            return None
    
    async def list_documents(self, limit: int = 50, offset: int = 0, 
                           filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """List documents with pagination and filtering"""
        try:
            documents = []
            metadata_files = list(self.metadata_path.glob("*.json"))
            
            # Sort by creation time (newest first)
            metadata_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Apply pagination
            start_idx = offset
            end_idx = offset + limit
            
            for metadata_file in metadata_files[start_idx:end_idx]:
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Apply filters
                    if filters and not self._apply_filters(metadata, filters):
                        continue
                    
                    # Load content if needed (for small documents)
                    content_file = self.content_path / f"{metadata['id']}.txt"
                    if content_file.exists():
                        with open(content_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                    else:
                        content = ""
                    
                    document = Document(
                        id=metadata["id"],
                        filename=metadata["filename"],
                        content=content,
                        doc_type=DocumentType(metadata["doc_type"]),
                        file_size=metadata["file_size"],
                        created_at=datetime.fromisoformat(metadata["created_at"]),
                        metadata=metadata.get("metadata", {}),
                        tags=metadata.get("tags", []),
                        summary=metadata.get("summary"),
                        category=metadata.get("category"),
                        language=metadata.get("language")
                    )
                    
                    documents.append(document)
                    
                except Exception as e:
                    logger.warning(f"Error loading document metadata from {metadata_file}: {str(e)}")
                    continue
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply filters to document metadata"""
        try:
            for key, value in filters.items():
                if key == "doc_type":
                    if metadata.get("doc_type") != value:
                        return False
                elif key == "filename_contains":
                    if value.lower() not in metadata.get("filename", "").lower():
                        return False
                elif key == "created_after":
                    doc_date = datetime.fromisoformat(metadata.get("created_at", ""))
                    if doc_date < value:
                        return False
                elif key == "created_before":
                    doc_date = datetime.fromisoformat(metadata.get("created_at", ""))
                    if doc_date > value:
                        return False
                elif key == "tags":
                    doc_tags = set(metadata.get("tags", []))
                    required_tags = set(value) if isinstance(value, list) else {value}
                    if not required_tags.intersection(doc_tags):
                        return False
                elif key == "category":
                    if metadata.get("category") != value:
                        return False
                elif key == "language":
                    if metadata.get("language") != value:
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return True
    
    async def update_document_metadata(self, document_id: str, updates: Dict[str, Any]) -> bool:
        """Update document metadata"""
        try:
            metadata_file = self.metadata_path / f"{document_id}.json"
            
            if not metadata_file.exists():
                logger.warning(f"Document {document_id} not found")
                return False
            
            # Load existing metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Apply updates
            for key, value in updates.items():
                if key in ["tags", "summary", "category", "language", "metadata"]:
                    metadata[key] = value
            
            # Save updated metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Update cache if document is cached
            if document_id in self._cache:
                document = self._cache[document_id]
                for key, value in updates.items():
                    if hasattr(document, key):
                        setattr(document, key, value)
            
            logger.info(f"Updated metadata for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document metadata: {str(e)}")
            return False
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its metadata"""
        try:
            metadata_file = self.metadata_path / f"{document_id}.json"
            content_file = self.content_path / f"{document_id}.txt"
            
            # Remove files
            if metadata_file.exists():
                metadata_file.unlink()
            if content_file.exists():
                content_file.unlink()
            
            # Remove from cache
            if document_id in self._cache:
                del self._cache[document_id]
            
            logger.info(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    async def search_documents(self, query: str, fields: List[str] = None) -> List[Document]:
        """Simple text search across documents"""
        if not fields:
            fields = ["filename", "content", "tags", "summary"]
        
        try:
            matching_documents = []
            query_lower = query.lower()
            
            # Get all documents
            all_documents = await self.list_documents(limit=1000)  # Adjust limit as needed
            
            for document in all_documents:
                match_found = False
                
                for field in fields:
                    field_value = getattr(document, field, "")
                    if isinstance(field_value, list):
                        field_value = " ".join(field_value)
                    elif field_value is None:
                        field_value = ""
                    
                    if query_lower in str(field_value).lower():
                        match_found = True
                        break
                
                if match_found:
                    matching_documents.append(document)
            
            logger.info(f"Found {len(matching_documents)} documents matching '{query}'")
            return matching_documents
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def _add_to_cache(self, document_id: str, document: Document):
        """Add document to cache with size limit"""
        try:
            # Remove oldest items if cache is full
            if len(self._cache) >= self._cache_size_limit:
                # Remove first item (FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[document_id] = document
        except Exception as e:
            logger.error(f"Error adding to cache: {str(e)}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the document store"""
        try:
            metadata_files = list(self.metadata_path.glob("*.json"))
            content_files = list(self.content_path.glob("*.txt"))
            
            # Calculate total storage size
            total_size = 0
            for file_path in metadata_files + content_files:
                total_size += file_path.stat().st_size
            
            # Count by document type
            type_counts = {}
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    doc_type = metadata.get("doc_type", "unknown")
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                except:
                    continue
            
            return {
                "total_documents": len(metadata_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_size": len(self._cache),
                "document_types": type_counts,
                "storage_path": str(self.store_path),
                "metadata_files": len(metadata_files),
                "content_files": len(content_files)
            }
        except Exception as e:
            logger.error(f"Error getting document store stats: {str(e)}")
            return {"error": str(e)}