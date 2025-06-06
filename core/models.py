from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    TEXT = "txt"
    DOCX = "docx"
    IMAGE = "image"
    HTML = "html"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Document(BaseModel):
    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    content: str = Field(..., description="Extracted text content")
    doc_type: DocumentType = Field(..., description="Document type")
    file_size: int = Field(..., description="File size in bytes")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    category: Optional[str] = None
    language: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "doc_type": self.doc_type,
            "file_size": self.file_size,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "tags": self.tags,
            "summary": self.summary,
            "category": self.category,
            "language": self.language
        }

class Chunk(BaseModel):
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Position in document")
    start_pos: int = Field(..., description="Start position in original document")
    end_pos: int = Field(..., description="End position in original document")
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResult(BaseModel):
    chunk_id: str = Field(..., description="Matching chunk ID")
    document_id: str = Field(..., description="Source document ID")
    content: str = Field(..., description="Matching content")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata
        }

class ProcessingTask(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    document_id: Optional[str] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    message: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class SummaryRequest(BaseModel):
    content: Optional[str] = None
    document_id: Optional[str] = None
    style: str = Field(default="concise", description="Summary style")
    max_length: Optional[int] = None

class TagGenerationRequest(BaseModel):
    content: Optional[str] = None
    document_id: Optional[str] = None
    max_tags: int = Field(default=5, ge=1, le=20)

class QuestionAnswerRequest(BaseModel):
    question: str = Field(..., description="Question to answer")
    context_filter: Optional[Dict[str, Any]] = None
    max_context_length: int = Field(default=2000)

class CategorizationRequest(BaseModel):
    content: Optional[str] = None
    document_id: Optional[str] = None
    categories: Optional[List[str]] = None