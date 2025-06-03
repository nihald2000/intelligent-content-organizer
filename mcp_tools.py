import asyncio
import aiohttp
import chromadb
from chromadb.utils import embedding_functions
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
from pathlib import Path
import requests

# Document processing libraries (all free)
import PyPDF2
import docx
from bs4 import BeautifulSoup
import pandas as pd
import markdown
import xml.etree.ElementTree as ET
from newspaper import Article
import trafilatura
from duckduckgo_search import DDGS

# AI libraries
from config import Config
from mistralai.client import MistralClient
import anthropic

# Set up logging
logger = logging.getLogger(__name__)

# Initialize AI clients
mistral_client = MistralClient(api_key=Config.MISTRAL_API_KEY) if Config.MISTRAL_API_KEY else None
anthropic_client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY) if Config.ANTHROPIC_API_KEY else None

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=Config.EMBEDDING_MODEL
)

# Get or create collection
try:
    collection = chroma_client.get_collection(
        name=Config.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function
    )
except:
    collection = chroma_client.create_collection(
        name=Config.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function
    )

class DocumentProcessor:
    """Free document processing without Unstructured API"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF files"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
        return text
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX files"""
        try:
            doc = docx.Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"Error reading DOCX: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_html(file_path: str) -> str:
        """Extract text from HTML files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
        except Exception as e:
            logger.error(f"Error reading HTML: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading TXT: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_csv(file_path: str) -> str:
        """Extract text from CSV files"""
        try:
            df = pd.read_csv(file_path)
            return df.to_string()
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_json(file_path: str) -> str:
        """Extract text from JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return json.dumps(data, indent=2)
        except Exception as e:
            logger.error(f"Error reading JSON: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_markdown(file_path: str) -> str:
        """Extract text from Markdown files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_text = file.read()
                html = markdown.markdown(md_text)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
        except Exception as e:
            logger.error(f"Error reading Markdown: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_xml(file_path: str) -> str:
        """Extract text from XML files"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            def extract_text(element):
                text = element.text or ""
                for child in element:
                    text += " " + extract_text(child)
                return text.strip()
            
            return extract_text(root)
        except Exception as e:
            logger.error(f"Error reading XML: {e}")
            return ""
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """Extract text from any supported file type"""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        extractors = {
            '.pdf': cls.extract_text_from_pdf,
            '.docx': cls.extract_text_from_docx,
            '.doc': cls.extract_text_from_docx,
            '.html': cls.extract_text_from_html,
            '.htm': cls.extract_text_from_html,
            '.txt': cls.extract_text_from_txt,
            '.csv': cls.extract_text_from_csv,
            '.json': cls.extract_text_from_json,
            '.md': cls.extract_text_from_markdown,
            '.xml': cls.extract_text_from_xml,
        }
        
        extractor = extractors.get(extension, cls.extract_text_from_txt)
        return extractor(file_path)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into chunks with overlap"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to find a sentence boundary
        if end < text_length:
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            boundary = max(last_period, last_newline)
            
            if boundary > chunk_size // 2:
                chunk = text[start:start + boundary + 1]
                end = start + boundary + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks

async def fetch_web_content_free(url: str) -> Optional[str]:
    """Fetch content from URL using multiple free methods"""
    
    # Method 1: Try newspaper3k (best for articles)
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        content = f"{article.title}\n\n{article.text}"
        if len(content) > 100:  # Valid content
            return content
    except Exception as e:
        logger.debug(f"Newspaper failed: {e}")
    
    # Method 2: Try trafilatura (great for web scraping)
    try:
        downloaded = trafilatura.fetch_url(url)
        content = trafilatura.extract(downloaded)
        if content and len(content) > 100:
            return content
    except Exception as e:
        logger.debug(f"Trafilatura failed: {e}")
    
    # Method 3: Basic BeautifulSoup scraping
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Try to find main content
            main_content = None
            
            # Common content selectors
            content_selectors = [
                'main', 'article', '[role="main"]', 
                '.content', '#content', '.post', '.entry-content',
                '.article-body', '.story-body'
            ]
            
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup.find('body')
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
                
                # Get title
                title = soup.find('title')
                title_text = title.get_text() if title else "No title"
                
                return f"{title_text}\n\n{text}"
                
    except Exception as e:
        logger.error(f"BeautifulSoup failed: {e}")
    
    return None

async def search_web_free(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Search the web using free methods (DuckDuckGo)"""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    'title': r.get('title', ''),
                    'url': r.get('link', ''),
                    'snippet': r.get('body', '')
                })
        
        return results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []

# In mcp_tools.py

async def generate_tags(content: str) -> List[str]:
    """Generate tags using Mistral AI or fallback to free method"""
    try:
        if mistral_client: # This is MistralClient from mistralai.client
            prompt = f"""Analyze this content and generate 5-7 relevant tags. 
            Return only the tags as a comma-separated list.
            
            Content: {content[:2000]}...
            
            Tags:"""
            
            # For mistralai==0.4.2, pass messages as a list of dicts
            response = mistral_client.chat(
                model=Config.MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}] # <--- CHANGE HERE
            )
            
            tags_text = response.choices[0].message.content.strip()
            tags = [tag.strip() for tag in tags_text.split(",")]
            return tags[:7]
        else:
            # Free fallback: Extract keywords using frequency analysis
            return generate_tags_free(content)
            
    except Exception as e:
        logger.error(f"Error generating tags: {str(e)}")
        return generate_tags_free(content)

def generate_tags_free(content: str) -> List[str]:
    """Free tag generation using keyword extraction"""
    from collections import Counter
    import re
    
    # Simple keyword extraction
    words = re.findall(r'\b[a-z]{4,}\b', content.lower())
    
    # Common stop words
    stop_words = {
        'this', 'that', 'these', 'those', 'what', 'which', 'when', 'where',
        'who', 'whom', 'whose', 'why', 'how', 'with', 'about', 'against',
        'between', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'from', 'down', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'only', 'same', 'than', 'that', 'have', 'has', 'had',
        'been', 'being', 'does', 'doing', 'will', 'would', 'could', 'should'
    }
    
    # Filter and count words
    filtered_words = [w for w in words if w not in stop_words and len(w) > 4]
    word_counts = Counter(filtered_words)
    
    # Get top keywords
    top_keywords = [word for word, _ in word_counts.most_common(7)]
    
    return top_keywords if top_keywords else ["untagged"]

async def generate_summary(content: str) -> str:
    """Generate summary using Claude or fallback to free method"""
    try:
        if anthropic_client:
            message = anthropic_client.messages.create(
                model=Config.CLAUDE_MODEL,
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": f"Summarize this content in 2-3 sentences:\n\n{content[:4000]}..."
                }]
            )
            
            return message.content[0].text.strip()
        else:
            # Free fallback
            return generate_summary_free(content)
            
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return generate_summary_free(content)

def generate_summary_free(content: str) -> str:
    """Free summary generation using simple extraction"""
    sentences = content.split('.')
    # Take first 3 sentences
    summary_sentences = sentences[:3]
    summary = '. '.join(s.strip() for s in summary_sentences if s.strip())
    
    if len(summary) > 300:
        summary = summary[:297] + "..."
    
    return summary if summary else "Content preview: " + content[:200] + "..."

async def process_local_file(file_path: str) -> Dict[str, Any]:
    """Process a local file and store it in the knowledge base"""
    try:
        # Validate file
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() not in Config.SUPPORTED_FILE_TYPES:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        # Extract text using free methods
        full_text = DocumentProcessor.extract_text(file_path)
        
        if not full_text:
            raise ValueError("No text could be extracted from the file")
        
        # Generate document ID
        doc_id = hashlib.md5(f"{path.name}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        # Generate tags
        tags = await generate_tags(full_text[:3000])
        
        # Generate summary
        summary = await generate_summary(full_text[:5000])
        
        # Chunk the text
        chunks = chunk_text(full_text, chunk_size=1000, overlap=100)
        chunks = chunks[:10]  # Limit chunks for demo
        
        # Store in ChromaDB
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        
        metadata = {
            "source": str(path),
            "file_name": path.name,
            "file_type": path.suffix,
            "processed_at": datetime.now().isoformat(),
            "tags": ", ".join(tags),
            "summary": summary,
            "doc_id": doc_id
        }
        
        collection.add(
            documents=chunks,
            ids=chunk_ids,
            metadatas=[metadata for _ in chunks]
        )
        
        return {
            "success": True,
            "doc_id": doc_id,
            "file_name": path.name,
            "tags": tags,
            "summary": summary,
            "chunks_processed": len(chunks),
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

async def process_web_content(url_or_query: str) -> Dict[str, Any]:
    """Process web content from URL or search query"""
    try:
        # Check if it's a URL or search query
        is_url = url_or_query.startswith(('http://', 'https://'))
        
        if is_url:
            content = await fetch_web_content_free(url_or_query)
            source = url_or_query
        else:
            # It's a search query
            search_results = await search_web_free(url_or_query, num_results=3)
            if not search_results:
                raise ValueError("No search results found")
            
            # Process the first result
            first_result = search_results[0]
            content = await fetch_web_content_free(first_result['url'])
            source = first_result['url']
            
            # Add search context
            content = f"Search Query: {url_or_query}\n\n{first_result['title']}\n\n{content}"
        
        if not content:
            raise ValueError("Failed to fetch content")
        
        # Generate document ID
        doc_id = hashlib.md5(f"{source}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        # Generate tags
        tags = await generate_tags(content[:3000])
        
        # Generate summary
        summary = await generate_summary(content[:5000])
        
        # Chunk the content
        chunks = chunk_text(content, chunk_size=1000, overlap=100)
        chunks = chunks[:10]  # Limit for demo
        
        # Store in ChromaDB
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        
        metadata = {
            "source": source,
            "url": source if is_url else f"Search: {url_or_query}",
            "content_type": "web",
            "processed_at": datetime.now().isoformat(),
            "tags": ", ".join(tags),
            "summary": summary,
            "doc_id": doc_id
        }
        
        collection.add(
            documents=chunks,
            ids=chunk_ids,
            metadatas=[metadata for _ in chunks]
        )
        
        return {
            "success": True,
            "doc_id": doc_id,
            "url": source,
            "tags": tags,
            "summary": summary,
            "chunks_processed": len(chunks),
            "metadata": metadata,
            "search_query": url_or_query if not is_url else None
        }
        
    except Exception as e:
        logger.error(f"Error processing web content: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

async def search_knowledge_base(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Perform semantic search in the knowledge base"""
    try:
        results = collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        if not results["ids"][0]:
            return []
        
        # Format results
        formatted_results = []
        seen_docs = set()
        
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            
            # Deduplicate by document
            if metadata["doc_id"] not in seen_docs:
                seen_docs.add(metadata["doc_id"])
                formatted_results.append({
                    "doc_id": metadata["doc_id"],
                    "source": metadata.get("source", "Unknown"),
                    "tags": metadata.get("tags", "").split(", "),
                    "summary": metadata.get("summary", ""),
                    "relevance_score": 1 - results["distances"][0][i],
                    "processed_at": metadata.get("processed_at", "")
                })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        return []

async def get_document_details(doc_id: str) -> Dict[str, Any]:
    """Get detailed information about a document"""
    try:
        results = collection.get(
            where={"doc_id": doc_id},
            limit=1
        )
        
        if not results["ids"]:
            return {"error": "Document not found"}
        
        metadata = results["metadatas"][0]
        return {
            "doc_id": doc_id,
            "source": metadata.get("source", "Unknown"),
            "tags": metadata.get("tags", "").split(", "),
            "summary": metadata.get("summary", ""),
            "processed_at": metadata.get("processed_at", ""),
            "file_type": metadata.get("file_type", ""),
            "content_preview": results["documents"][0][:500] + "..."
        }
        
    except Exception as e:
        logger.error(f"Error getting document details: {str(e)}")
        return {"error": str(e)}