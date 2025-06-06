import logging
import asyncio
from typing import List, Dict, Any, Optional
import anthropic
from mistralai.client import MistralClient
import config

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.config = config.config
        
        # Initialize clients
        self.anthropic_client = None
        self.mistral_client = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize LLM clients"""
        try:
            if self.config.ANTHROPIC_API_KEY:
                self.anthropic_client = anthropic.Anthropic(
                    api_key=self.config.ANTHROPIC_API_KEY
                )
                logger.info("Anthropic client initialized")
            
            if self.config.MISTRAL_API_KEY:
                self.mistral_client = MistralClient(
                    api_key=self.config.MISTRAL_API_KEY
                )
                logger.info("Mistral client initialized")
            
            if not self.anthropic_client and not self.mistral_client:
                raise ValueError("No LLM clients could be initialized. Check API keys.")
                
        except Exception as e:
            logger.error(f"Error initializing LLM clients: {str(e)}")
            raise
    
    async def generate_text(self, prompt: str, model: str = "auto", max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text using the specified model"""
        try:
            if model == "auto":
                # Use Claude if available, otherwise Mistral
                if self.anthropic_client:
                    return await self._generate_with_claude(prompt, max_tokens, temperature)
                elif self.mistral_client:
                    return await self._generate_with_mistral(prompt, max_tokens, temperature)
                else:
                    raise ValueError("No LLM clients available")
            elif model.startswith("claude"):
                if not self.anthropic_client:
                    raise ValueError("Anthropic client not available")
                return await self._generate_with_claude(prompt, max_tokens, temperature)
            elif model.startswith("mistral"):
                if not self.mistral_client:
                    raise ValueError("Mistral client not available")
                return await self._generate_with_mistral(prompt, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported model: {model}")
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    async def _generate_with_claude(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using Claude"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(
                    model=self.config.ANTHROPIC_MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error with Claude generation: {str(e)}")
            raise
    
    async def _generate_with_mistral(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate text using Mistral"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.mistral_client.chat(
                    model=self.config.MISTRAL_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with Mistral generation: {str(e)}")
            raise
    
    async def summarize(self, text: str, style: str = "concise", max_length: Optional[int] = None) -> str:
        """Generate a summary of the given text"""
        if not text.strip():
            return ""
        
        # Create style-specific prompts
        style_prompts = {
            "concise": "Provide a concise summary of the following text, focusing on the main points:",
            "detailed": "Provide a detailed summary of the following text, including key details and supporting information:",
            "bullet_points": "Summarize the following text as a list of bullet points highlighting the main ideas:",
            "executive": "Provide an executive summary of the following text, focusing on key findings and actionable insights:"
        }
        
        prompt_template = style_prompts.get(style, style_prompts["concise"])
        
        if max_length:
            prompt_template += f" Keep the summary under {max_length} words."
        
        prompt = f"{prompt_template}\n\nText to summarize:\n{text}\n\nSummary:"
        
        try:
            summary = await self.generate_text(prompt, max_tokens=500, temperature=0.3)
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating summary"
    
    async def generate_tags(self, text: str, max_tags: int = 5) -> List[str]:
        """Generate relevant tags for the given text"""
        if not text.strip():
            return []
        
        prompt = f"""Generate {max_tags} relevant tags for the following text. 
        Tags should be concise, descriptive keywords or phrases that capture the main topics, themes, or concepts.
        Return only the tags, separated by commas.

        Text:
        {text}

        Tags:"""
        
        try:
            response = await self.generate_text(prompt, max_tokens=100, temperature=0.5)
            
            # Parse tags from response
            tags = [tag.strip() for tag in response.split(',')]
            tags = [tag for tag in tags if tag and len(tag) > 1]
            
            return tags[:max_tags]
        except Exception as e:
            logger.error(f"Error generating tags: {str(e)}")
            return []
    
    async def categorize(self, text: str, categories: List[str]) -> str:
        """Categorize text into one of the provided categories"""
        if not text.strip() or not categories:
            return "Uncategorized"
        
        categories_str = ", ".join(categories)
        
        prompt = f"""Classify the following text into one of these categories: {categories_str}

        Choose the most appropriate category based on the content and main theme of the text.
        Return only the category name, nothing else.

        Text to classify:
        {text}

        Category:"""
        
        try:
            response = await self.generate_text(prompt, max_tokens=50, temperature=0.1)
            category = response.strip()
            
            # Validate that the response is one of the provided categories
            if category in categories:
                return category
            else:
                # Try to find a close match
                category_lower = category.lower()
                for cat in categories:
                    if cat.lower() in category_lower or category_lower in cat.lower():
                        return cat
                
                return categories[0] if categories else "Uncategorized"
        except Exception as e:
            logger.error(f"Error categorizing text: {str(e)}")
            return "Uncategorized"
    
    async def answer_question(self, question: str, context: str, max_context_length: int = 2000) -> str:
        """Answer a question based on the provided context"""
        if not question.strip():
            return "No question provided"
        
        if not context.strip():
            return "I don't have enough context to answer this question. Please provide more relevant information."
        
        # Truncate context if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        prompt = f"""Based on the following context, answer the question. If the context doesn't contain enough information to answer the question completely, say so and provide what information you can.

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        try:
            answer = await self.generate_text(prompt, max_tokens=300, temperature=0.3)
            return answer.strip()
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return "I encountered an error while trying to answer your question."
    
    async def extract_key_information(self, text: str) -> Dict[str, Any]:
        """Extract key information from text"""
        if not text.strip():
            return {}
        
        prompt = f"""Analyze the following text and extract key information. Provide the response in the following format:
        
        Main Topic: [main topic or subject]
        Key Points: [list 3-5 key points]
        Entities: [important people, places, organizations mentioned]
        Sentiment: [positive/neutral/negative]
        Content Type: [article/document/email/report/etc.]

        Text to analyze:
        {text}

        Analysis:"""
        
        try:
            response = await self.generate_text(prompt, max_tokens=400, temperature=0.4)
            
            # Parse the structured response
            info = {}
            lines = response.split('\n')
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    if value:
                        info[key] = value
            
            return info
        except Exception as e:
            logger.error(f"Error extracting key information: {str(e)}")
            return {}
    
    async def check_availability(self) -> Dict[str, bool]:
        """Check which LLM services are available"""
        availability = {
            "anthropic": False,
            "mistral": False
        }
        
        try:
            if self.anthropic_client:
                # Test Claude availability with a simple request
                test_response = await self._generate_with_claude("Hello", 10, 0.1)
                availability["anthropic"] = bool(test_response)
        except:
            pass
        
        try:
            if self.mistral_client:
                # Test Mistral availability with a simple request
                test_response = await self._generate_with_mistral("Hello", 10, 0.1)
                availability["mistral"] = bool(test_response)
        except:
            pass
        
        return availability