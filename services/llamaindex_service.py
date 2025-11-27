import logging
import os
from typing import List, Optional, Any
from pathlib import Path
import shutil
import asyncio

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
    Settings,
    SummaryIndex
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import config
from services.document_store_service import DocumentStoreService

logger = logging.getLogger(__name__)

class LlamaIndexService:
    def __init__(self, document_store: DocumentStoreService):
        self.document_store = document_store
        self.config = config.config
        self.storage_dir = Path(self.config.DATA_DIR) / "llamaindex_storage"
        self.index = None
        self.agent = None
        self.is_initialized = False
        
        self._initialize_settings()
        # We don't fully initialize index here because we need async access to doc store
        # But we try to load existing storage if available
        self._try_load_from_storage()

    def _initialize_settings(self):
        """Initialize LlamaIndex settings (LLM, Embeddings)"""
        try:
            # LLM Setup
            if self.config.OPENAI_API_KEY:
                Settings.llm = OpenAI(model="gpt-4o-mini", api_key=self.config.OPENAI_API_KEY)
                logger.info("LlamaIndex using OpenAI GPT-4o-mini")
            elif self.config.ANTHROPIC_API_KEY:
                Settings.llm = Anthropic(model="claude-3-haiku-20240307", api_key=self.config.ANTHROPIC_API_KEY)
                logger.info("LlamaIndex using Anthropic Claude 3 Haiku")
            else:
                logger.warning("No API key found for LlamaIndex LLM. Agentic features may fail.")

            # Embedding Setup
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("LlamaIndex using HuggingFace embeddings")
            
        except Exception as e:
            logger.error(f"Error initializing LlamaIndex settings: {str(e)}")

    def _try_load_from_storage(self):
        """Try to load index from storage synchronously"""
        try:
            if self.storage_dir.exists():
                logger.info("Loading LlamaIndex from storage...")
                storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
                self.index = load_index_from_storage(storage_context)
                self._initialize_agent()
                self.is_initialized = True
            else:
                logger.info("No existing LlamaIndex storage found. Waiting for async initialization.")
        except Exception as e:
            logger.error(f"Error loading LlamaIndex from storage: {str(e)}")

    async def initialize(self):
        """Async initialization to sync documents and build index"""
        try:
            logger.info("Starting LlamaIndex async initialization...")
            
            # If we already have an index, we might still want to sync if it's empty or stale
            # For now, if no index exists, we definitely need to build it
            if self.index is None:
                await self.sync_from_document_store()
            
            self.is_initialized = True
            logger.info("LlamaIndex async initialization complete.")
            
        except Exception as e:
            logger.error(f"Error during LlamaIndex async initialization: {str(e)}")

    async def sync_from_document_store(self):
        """Sync documents from DocumentStore to LlamaIndex"""
        try:
            logger.info("Syncing documents from DocumentStore to LlamaIndex...")
            
            # Fetch documents from async document store
            # Limit to 1000 for now to avoid memory issues
            docs = await self.document_store.list_documents(limit=1000)
            
            if not docs:
                logger.warning("No documents found in DocumentStore to sync.")
                # Create empty index if no docs
                self.index = VectorStoreIndex.from_documents([])
            else:
                # Convert to LlamaIndex documents
                llama_docs = []
                for doc in docs:
                    llama_doc = Document(
                        text=doc.content,
                        metadata={
                            "filename": doc.filename,
                            "document_id": doc.id,
                            **doc.metadata
                        }
                    )
                    llama_docs.append(llama_doc)
                
                logger.info(f"Building LlamaIndex with {len(llama_docs)} documents...")
                self.index = VectorStoreIndex.from_documents(llama_docs)
            
            # Persist storage
            if not self.storage_dir.exists():
                self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.index.storage_context.persist(persist_dir=str(self.storage_dir))
            
            # Re-initialize agent with new index
            self._initialize_agent()
            logger.info("LlamaIndex sync complete.")
            
        except Exception as e:
            logger.error(f"Error syncing LlamaIndex: {str(e)}")

    async def sync_on_demand(self):
        """Manual trigger for syncing documents"""
        await self.sync_from_document_store()
        return True

    def _initialize_agent(self):
        """Initialize the ReAct agent with query engine tools"""
        try:
            if not self.index:
                return

            query_engine = self.index.as_query_engine()
            
            query_engine_tool = QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name="document_search",
                    description="Search and retrieve information from the document library. Use this for specific questions about content."
                )
            )
            
            self.agent = ReActAgent.from_tools(
                [query_engine_tool],
                llm=Settings.llm,
                verbose=True
            )
            logger.info("LlamaIndex ReAct agent initialized")
            
        except Exception as e:
            logger.error(f"Error initializing LlamaIndex agent: {str(e)}")

    async def query(self, query_text: str) -> str:
        """Process a query using the agent"""
        if not self.agent:
            if not self.is_initialized:
                return "Agent is initializing, please try again in a moment."
            return "Agent failed to initialize. Please check logs."
            
        try:
            response = await self.agent.achat(query_text)
            return str(response)
        except Exception as e:
            logger.error(f"Error querying LlamaIndex agent: {str(e)}")
            return f"Error processing query: {str(e)}"
