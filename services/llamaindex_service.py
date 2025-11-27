import logging
import os
from typing import List, Optional, Any
from pathlib import Path
import shutil

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
        
        self._initialize_settings()
        self._initialize_index()
        self._initialize_agent()

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
            # Using a local HF model to match the likely existing sentence-transformers setup
            # or just a standard efficient one.
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("LlamaIndex using HuggingFace embeddings")
            
        except Exception as e:
            logger.error(f"Error initializing LlamaIndex settings: {str(e)}")

    def _initialize_index(self):
        """Initialize or load the VectorStoreIndex"""
        try:
            if self.storage_dir.exists():
                logger.info("Loading LlamaIndex from storage...")
                storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
                self.index = load_index_from_storage(storage_context)
            else:
                logger.info("Creating new LlamaIndex from documents...")
                self.rebuild_index()
                
        except Exception as e:
            logger.error(f"Error initializing LlamaIndex index: {str(e)}")
            # Fallback: try to rebuild if load fails
            logger.info("Attempting to rebuild index after load failure...")
            self.rebuild_index()

    def rebuild_index(self):
        """Rebuild the index from the document store"""
        try:
            # Fetch all documents from existing DocumentStore
            # We need to run this in a way that works with the sync/async nature.
            # Since __init__ is sync, we might need a helper or just rely on what's available.
            # For now, we'll assume we can access the store. 
            # Note: DocumentStoreService methods are async. We might need to call this differently.
            # However, for the hackathon, we can just read the files directly or use a sync wrapper if needed.
            # Or, we can make this method async and call it after init.
            
            # For simplicity in this phase, let's just check if we can get the docs.
            # If not, we initialize an empty index.
            self.index = VectorStoreIndex.from_documents([])
            
            # We will populate it properly in an async method later or now if we can.
            if not self.storage_dir.exists():
                self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.index.storage_context.persist(persist_dir=str(self.storage_dir))
            logger.info("Initialized empty LlamaIndex (will be populated async)")
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}")

    async def sync_documents(self):
        """Sync documents from DocumentStore to LlamaIndex"""
        try:
            logger.info("Syncing documents to LlamaIndex...")
            docs = await self.document_store.list_documents(limit=1000)
            
            llama_docs = []
            for doc in docs:
                # Create LlamaIndex Document
                llama_doc = Document(
                    text=doc.content,
                    metadata={
                        "filename": doc.filename,
                        "file_path": doc.file_path,
                        "document_id": doc.id,
                        "category": doc.category,
                        "created_at": doc.created_at.isoformat()
                    }
                )
                llama_docs.append(llama_doc)
            
            if llama_docs:
                self.index = VectorStoreIndex.from_documents(llama_docs)
                self.index.storage_context.persist(persist_dir=str(self.storage_dir))
                logger.info(f"Synced {len(llama_docs)} documents to LlamaIndex")
                
                # Re-initialize agent with new index
                self._initialize_agent()
            else:
                logger.info("No documents to sync")
                
        except Exception as e:
            logger.error(f"Error syncing documents: {str(e)}")

    def _initialize_agent(self):
        """Initialize the ReAct agent with tools"""
        try:
            if not self.index:
                return

            # Create Query Engine Tool
            query_engine = self.index.as_query_engine(similarity_top_k=5)
            
            query_engine_tool = QueryEngineTool(
                query_engine=query_engine,
                metadata=ToolMetadata(
                    name="vector_search",
                    description="Useful for searching specific information in the document library."
                )
            )
            
            # Create Summary Tool (using SummaryIndex if we had one, or just the vector store with different prompt)
            summary_tool = QueryEngineTool(
                query_engine=self.index.as_query_engine(response_mode="tree_summarize"),
                metadata=ToolMetadata(
                    name="summarizer",
                    description="Useful for summarizing documents or topics."
                )
            )

            # Initialize ReAct Agent
            self.agent = ReActAgent(
                tools=[query_engine_tool, summary_tool],
                llm=Settings.llm,
                verbose=True
            )
            logger.info("LlamaIndex ReAct agent initialized")
            
        except Exception as e:
            import traceback
            logger.error(f"Error initializing agent: {str(e)}\n{traceback.format_exc()}")

    async def query(self, query_text: str) -> str:
        """Execute a query using the agent"""
        if not self.agent:
            return "Agent not initialized. Please ensure documents are indexed."
            
        try:
            logger.info(f"Agentic query: {query_text}")
            # Run agent
            response = await self.agent.achat(query_text)
            return str(response)
        except Exception as e:
            logger.error(f"Error during agent query: {str(e)}")
            return f"Error processing query: {str(e)}"

    async def auto_routed_retrieval(self, query_text: str) -> str:
        """
        Demonstrate auto-routing between summary and vector search.
        This is a simpler version of the agentic workflow.
        """
        try:
            if not self.index:
                return "Index not ready."

            # Define tools for router
            vector_tool = QueryEngineTool.from_defaults(
                query_engine=self.index.as_query_engine(),
                description="Useful for retrieving specific context and facts."
            )
            
            summary_tool = QueryEngineTool.from_defaults(
                query_engine=self.index.as_query_engine(response_mode="tree_summarize"),
                description="Useful for summarization and high-level questions."
            )

            query_engine = RouterQueryEngine(
                selector=LLMSingleSelector.from_defaults(),
                query_engine_tools=[
                    vector_tool,
                    summary_tool,
                ],
            )
            
            response = await query_engine.aquery(query_text)
            return str(response)
            
        except Exception as e:
            logger.error(f"Error in auto-routed retrieval: {str(e)}")
            return f"Error: {str(e)}"
