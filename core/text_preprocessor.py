import re
import logging
from typing import List, Optional
import unicodedata

logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        # Common stop words for basic filtering
        self.stop_words = {
            'en': set([
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                'before', 'after', 'above', 'below', 'between', 'among', 'throughout',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'me',
                'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours'
            ])
        }
    
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        try:
            # Normalize unicode characters
            text = unicodedata.normalize('NFKD', text)
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove or replace special characters
            if aggressive:
                # More aggressive cleaning for embedding
                text = re.sub(r'[^\w\s\-.,!?;:]', ' ', text)
                text = re.sub(r'[.,!?;:]+', '.', text)
            else:
                # Basic cleaning for readability
                text = re.sub(r'[^\w\s\-.,!?;:()\[\]{}"\']', ' ', text)
            
            # Remove excessive punctuation
            text = re.sub(r'\.{2,}', '.', text)
            text = re.sub(r'[!?]{2,}', '!', text)
            
            # Clean up whitespace again
            text = re.sub(r'\s+', ' ', text)
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
            return text
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text"""
        if not text:
            return []
        
        try:
            # Simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            
            # Clean and filter sentences
            clean_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Minimum sentence length
                    clean_sentences.append(sentence)
            
            return clean_sentences
        except Exception as e:
            logger.error(f"Error extracting sentences: {str(e)}")
            return [text]
    
    def extract_keywords(self, text: str, language: str = 'en', max_keywords: int = 20) -> List[str]:
        """Extract potential keywords from text"""
        if not text:
            return []
        
        try:
            # Convert to lowercase and split into words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Remove stop words
            stop_words = self.stop_words.get(language, set())
            keywords = [word for word in words if word not in stop_words]
            
            # Count word frequency
            word_freq = {}
            for word in keywords:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and return top keywords
            sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            return [word for word, freq in sorted_keywords[:max_keywords]]
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def prepare_for_embedding(self, text: str) -> str:
        """Prepare text specifically for embedding generation"""
        if not text:
            return ""
        
        try:
            # Clean text aggressively for better embeddings
            clean_text = self.clean_text(text, aggressive=True)
            
            # Remove very short words
            words = clean_text.split()
            filtered_words = [word for word in words if len(word) >= 2]
            
            # Rejoin and ensure reasonable length
            result = ' '.join(filtered_words)
            
            # Truncate if too long (most embedding models have token limits)
            if len(result) > 5000:  # Rough character limit
                result = result[:5000] + "..."
            
            return result
        except Exception as e:
            logger.error(f"Error preparing text for embedding: {str(e)}")
            return text
    
    def extract_metadata_from_text(self, text: str) -> dict:
        """Extract metadata from text content"""
        if not text:
            return {}
        
        try:
            metadata = {}
            
            # Basic statistics
            metadata['character_count'] = len(text)
            metadata['word_count'] = len(text.split())
            metadata['sentence_count'] = len(self.extract_sentences(text))
            metadata['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
            
            # Content characteristics
            metadata['avg_word_length'] = sum(len(word) for word in text.split()) / max(1, len(text.split()))
            metadata['avg_sentence_length'] = metadata['word_count'] / max(1, metadata['sentence_count'])
            
            # Special content detection
            metadata['has_urls'] = bool(re.search(r'https?://\S+', text))
            metadata['has_emails'] = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
            metadata['has_phone_numbers'] = bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
            metadata['has_dates'] = bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text))
            metadata['has_numbers'] = bool(re.search(r'\b\d+\b', text))
            
            # Language indicators
            metadata['punctuation_density'] = len(re.findall(r'[.,!?;:]', text)) / max(1, len(text))
            metadata['caps_ratio'] = len(re.findall(r'[A-Z]', text)) / max(1, len(text))
            
            return metadata
        except Exception as e:
            logger.error(f"Error extracting text metadata: {str(e)}")
            return {}
    
    def normalize_for_search(self, text: str) -> str:
        """Normalize text for search queries"""
        if not text:
            return ""
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters but keep spaces
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Strip leading/trailing whitespace
            text = text.strip()
            
            return text
        except Exception as e:
            logger.error(f"Error normalizing text for search: {str(e)}")
            return text