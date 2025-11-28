import logging
import asyncio
from typing import Dict, Any, Optional
import tempfile
import os
from pathlib import Path
import uuid

from core.document_parser import DocumentParser
from core.chunker import TextChunker
from core.text_preprocessor import TextPreprocessor
from services.vector_store_service import VectorStoreService
from services.document_store_service import DocumentStoreService
from services.embedding_service import EmbeddingService
from services.ocr_service import OCRService

logger = logging.getLogger(__name__)

class IngestionTool:
    def __init__(self, vector_store: VectorStoreService, document_store: DocumentStoreService, 
             embedding_service: EmbeddingService, ocr_service: OCRService):
        self.vector_store = vector_store
        self.document_store = document_store
        self.embedding_service = embedding_service
        self.ocr_service = ocr_service
        
        self.document_parser = DocumentParser()
        # Pass OCR service to document parser
        self.document_parser.ocr_service = ocr_service
        
        self.text_chunker = TextChunker()
        self.text_preprocessor = TextPreprocessor()
    
    async def process_document(self, file_path: str, file_type: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a document through the full ingestion pipeline"""
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Starting document processing for {file_path}")
            
            # Step 1: Parse the document
            filename = Path(file_path).name
            document = await self.document_parser.parse_document(file_path, filename)
            
            if not document.content:
                logger.warning(f"No content extracted from document {filename}")
                return {
                    "success": False,
                    "error": "No content could be extracted from the document",
                    "task_id": task_id
                }
            
            # Step 2: Store the document
            await self.document_store.store_document(document)
            
            # Step 3: Process content for embeddings
            chunks = await self._create_and_embed_chunks(document)
            
            if not chunks:
                logger.warning(f"No chunks created for document {document.id}")
                return {
                    "success": False,
                    "error": "Failed to create text chunks",
                    "task_id": task_id,
                    "document_id": document.id,
                    "filename": document.filename,
                    "chunks_created": len(chunks),
                    "content_length": len(document.content),
                    "doc_type": document.doc_type.value,
                    "message": f"Successfully processed {filename}"
                }
            
            # Step 4: Store embeddings
            success = await self.vector_store.add_chunks(chunks)
            
            if not success:
                logger.error(f"Failed to store embeddings for document {document.id}")
                return {
                    "success": False,
                    "error": "Failed to store embeddings",
                    "task_id": task_id,
                    "document_id": document.id
                }
            
            # Step 5: Update document metadata with chunk count
            try:
                current_metadata = document.metadata or {}
                current_metadata["chunk_count"] = len(chunks)
                await self.document_store.update_document_metadata(
                    document.id, 
                    {"metadata": current_metadata}
                )
            except Exception as e:
                logger.warning(f"Failed to update chunk count for document {document.id}: {e}")
            
            logger.info(f"Successfully processed document {document.id} with {len(chunks)} chunks")
            
            return {
                "success": True,
                "task_id": task_id,
                "document_id": document.id,
                "filename": document.filename,
                "chunks_created": len(chunks),
                "content_length": len(document.content),
                "doc_type": document.doc_type.value,
                "message": f"Successfully processed {filename}"
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id,
                "message": f"Failed to process document: {str(e)}"
            }
    
    async def _create_and_embed_chunks(self, document) -> list:
        """Create chunks and generate embeddings"""
        try:
            # Step 1: Create chunks
            chunks = self.text_chunker.chunk_document(
                document.id, 
                document.content, 
                method="recursive"
            )
            
            if not chunks:
                return []
            
            # Step 2: Optimize chunks for embedding
            optimized_chunks = self.text_chunker.optimize_chunks_for_embedding(chunks)
            
            # Step 3: Generate embeddings
            texts = [chunk.content for chunk in optimized_chunks]
            embeddings = await self.embedding_service.generate_embeddings(texts)
            
            # Step 4: Add embeddings to chunks
            embedded_chunks = []
            for i, chunk in enumerate(optimized_chunks):
                if i < len(embeddings):
                    chunk.embedding = embeddings[i]
                    embedded_chunks.append(chunk)
            
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Error creating and embedding chunks: {str(e)}")
            return []
    
    async def process_url(self, url: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a document from a URL"""
        try:
            import requests
            from urllib.parse import urlparse
            
            # Download the file
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine file type from URL or content-type
            parsed_url = urlparse(url)
            filename = Path(parsed_url.path).name or "downloaded_file"
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
            
            try:
                # Process the downloaded file
                result = await self.process_document(tmp_file_path, "", task_id)
                result["source_url"] = url
                return result
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id or str(uuid.uuid4()),
                "source_url": url
            }
    
    async def process_text_content(self, content: str, filename: str = "text_content.txt", 
                                 task_id: Optional[str] = None) -> Dict[str, Any]:
        """Process raw text content directly"""
        try:
            from core.models import Document, DocumentType
            from datetime import datetime
            
            # Create document object
            document = Document(
                id=str(uuid.uuid4()),
                filename=filename,
                content=content,
                doc_type=DocumentType.TEXT,
                file_size=len(content.encode('utf-8')),
                created_at=datetime.utcnow(),
                metadata={
                    "source": "direct_text_input",
                    "content_length": len(content),
                    "word_count": len(content.split())
                }
            )
            
            # Store the document
            await self.document_store.store_document(document)
            
            # Process content for embeddings
            chunks = await self._create_and_embed_chunks(document)
            
            if chunks:
                await self.vector_store.add_chunks(chunks)
                
                # Update document metadata with chunk count
                try:
                    current_metadata = document.metadata or {}
                    current_metadata["chunk_count"] = len(chunks)
                    await self.document_store.update_document_metadata(
                        document.id, 
                        {"metadata": current_metadata}
                    )
                except Exception as e:
                    logger.warning(f"Failed to update chunk count for document {document.id}: {e}")
            
            return {
                "success": True,
                "task_id": task_id or str(uuid.uuid4()),
                "document_id": document.id,
                "filename": filename,
                "chunks_created": len(chunks),
                "content_length": len(content),
                "message": f"Successfully processed text content"
            }
            
        except Exception as e:
            logger.error(f"Error processing text content: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id or str(uuid.uuid4())
            }
    
    async def reprocess_document(self, document_id: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Reprocess an existing document (useful for updating embeddings)"""
        try:
            # Get the document
            document = await self.document_store.get_document(document_id)
            
            if not document:
                return {
                    "success": False,
                    "error": f"Document {document_id} not found",
                    "task_id": task_id or str(uuid.uuid4())
                }
            
            # Remove existing chunks from vector store
            await self.vector_store.delete_document(document_id)
            
            # Recreate and embed chunks
            chunks = await self._create_and_embed_chunks(document)
            
            if chunks:
                await self.vector_store.add_chunks(chunks)
                
                # Update document metadata with chunk count
                try:
                    current_metadata = document.metadata or {}
                    current_metadata["chunk_count"] = len(chunks)
                    await self.document_store.update_document_metadata(
                        document.id, 
                        {"metadata": current_metadata}
                    )
                except Exception as e:
                    logger.warning(f"Failed to update chunk count for document {document.id}: {e}")
            
            return {
                "success": True,
                "task_id": task_id or str(uuid.uuid4()),
                "document_id": document_id,
                "filename": document.filename,
                "chunks_created": len(chunks),
                "message": f"Successfully reprocessed {document.filename}"
            }
            
        except Exception as e:
            logger.error(f"Error reprocessing document {document_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id or str(uuid.uuid4()),
                "document_id": document_id
            }
    
    async def batch_process_directory(self, directory_path: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Process multiple documents from a directory"""
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                return {
                    "success": False,
                    "error": f"Directory {directory_path} does not exist",
                    "task_id": task_id or str(uuid.uuid4())
                }
            
            # Supported file extensions
            supported_extensions = {'.txt', '.pdf', '.docx', '.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
            
            # Find all supported files
            files_to_process = []
            for ext in supported_extensions:
                files_to_process.extend(directory.glob(f"*{ext}"))
                files_to_process.extend(directory.glob(f"*{ext.upper()}"))
            
            if not files_to_process:
                return {
                    "success": False,
                    "error": "No supported files found in directory",
                    "task_id": task_id or str(uuid.uuid4())
                }
            
            # Process files
            results = []
            successful = 0
            failed = 0
            
            for file_path in files_to_process:
                try:
                    result = await self.process_document(str(file_path), file_path.suffix)
                    results.append(result)
                    
                    if result.get("success"):
                        successful += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    failed += 1
                    results.append({
                        "success": False,
                        "error": str(e),
                        "filename": file_path.name
                    })
            
            return {
                "success": True,
                "task_id": task_id or str(uuid.uuid4()),
                "directory": str(directory),
                "total_files": len(files_to_process),
                "successful": successful,
                "failed": failed,
                "results": results,
                "message": f"Processed {successful}/{len(files_to_process)} files successfully"
            }
            
        except Exception as e:
            logger.error(f"Error batch processing directory {directory_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id or str(uuid.uuid4())
            }