from mistralai import Mistral
import logging
import asyncio
from typing import List, Dict, Any, Optional

import anthropic
import openai
import config

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.config = config.config
        
        self.anthropic_client = None
        self.mistral_client = None # Synchronous Mistral client
        self.openai_async_client = None # Asynchronous OpenAI client
        
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
                self.mistral_client = Mistral( # Standard sync client
                    api_key=self.config.MISTRAL_API_KEY
                )
                logger.info("Mistral client initialized")

            if self.config.OPENAI_API_KEY:
                self.openai_async_client = openai.AsyncOpenAI(
                    api_key=self.config.OPENAI_API_KEY
                )
                logger.info("OpenAI client initialized")
            
            # Check if at least one client is initialized
            if not any([self.openai_async_client, self.mistral_client, self.anthropic_client]):
                logger.warning("No LLM clients could be initialized based on current config. Check API keys.")
            else:
                logger.info("LLM clients initialized successfully (at least one).")
                
        except Exception as e:
            logger.error(f"Error initializing LLM clients: {str(e)}")
            raise
    
    async def generate_text(self, prompt: str, model: str = "auto", max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text using the specified model, with new priority for 'auto'."""
        try:
            selected_model_name_for_call: str = "" # Actual model name passed to the specific generator

            if model == "auto":
                # New Priority: 1. OpenAI, 2. Mistral, 3. Anthropic
                if self.openai_async_client and self.config.OPENAI_MODEL:
                    selected_model_name_for_call = self.config.OPENAI_MODEL
                    logger.debug(f"Auto-selected OpenAI model: {selected_model_name_for_call}")
                    return await self._generate_with_openai(prompt, selected_model_name_for_call, max_tokens, temperature)
                elif self.mistral_client and self.config.MISTRAL_MODEL:
                    selected_model_name_for_call = self.config.MISTRAL_MODEL
                    logger.debug(f"Auto-selected Mistral model: {selected_model_name_for_call}")
                    return await self._generate_with_mistral(prompt, selected_model_name_for_call, max_tokens, temperature)
                elif self.anthropic_client and self.config.ANTHROPIC_MODEL:
                    selected_model_name_for_call = self.config.ANTHROPIC_MODEL
                    logger.debug(f"Auto-selected Anthropic model: {selected_model_name_for_call}")
                    return await self._generate_with_claude(prompt, selected_model_name_for_call, max_tokens, temperature)
                else:
                    logger.error("No LLM clients available for 'auto' mode or default models not configured.")
                    raise ValueError("No LLM clients available for 'auto' mode or default models not configured.")
            
            elif model.startswith("gpt-") or model.lower().startswith("openai/"):
                if not self.openai_async_client:
                    raise ValueError("OpenAI client not available. Check API key or model prefix.")
                actual_model = model.split('/')[-1] if '/' in model else model
                return await self._generate_with_openai(prompt, actual_model, max_tokens, temperature)
            
            elif model.startswith("mistral"):
                if not self.mistral_client:
                    raise ValueError("Mistral client not available. Check API key or model prefix.")
                return await self._generate_with_mistral(prompt, model, max_tokens, temperature)

            elif model.startswith("claude"):
                if not self.anthropic_client:
                    raise ValueError("Anthropic client not available. Check API key or model prefix.")
                return await self._generate_with_claude(prompt, model, max_tokens, temperature)
            
            else:
                raise ValueError(f"Unsupported model: {model}. Must start with 'gpt-', 'openai/', 'claude', 'mistral', or be 'auto'.")
        
        except Exception as e:
            logger.error(f"Error generating text with model '{model}': {str(e)}")
            raise

    async def _generate_with_openai(self, prompt: str, model_name: str, max_tokens: int, temperature: float) -> str:
        """Generate text using OpenAI (Async)"""
        if not self.openai_async_client:
            raise RuntimeError("OpenAI async client not initialized.")
        try:
            logger.debug(f"Generating with OpenAI model: {model_name}, max_tokens: {max_tokens}, temp: {temperature}, prompt: '{prompt[:50]}...'")
            response = await self.openai_async_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            if response.choices and response.choices[0].message:
                 content = response.choices[0].message.content
                 if content is not None:
                     return content.strip()
                 else:
                     logger.warning(f"OpenAI response message content is None for model {model_name}.")
                     return ""
            else:
                logger.warning(f"OpenAI response did not contain expected choices or message for model {model_name}.")
                return ""
        except Exception as e:
            logger.error(f"Error with OpenAI generation (model: {model_name}): {str(e)}")
            raise

    async def _generate_with_claude(self, prompt: str, model_name: str, max_tokens: int, temperature: float) -> str:
        """Generate text using Anthropic/Claude (Sync via run_in_executor)"""
        if not self.anthropic_client:
            raise RuntimeError("Anthropic client not initialized.")
        try:
            logger.debug(f"Generating with Anthropic model: {model_name}, max_tokens: {max_tokens}, temp: {temperature}, prompt: '{prompt[:50]}...'")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.anthropic_client.messages.create(
                    model=model_name, # Use the passed model_name
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
            )
            if response.content and response.content[0].text:
                return response.content[0].text.strip()
            else:
                logger.warning(f"Anthropic response did not contain expected content for model {model_name}.")
                return ""
        except Exception as e:
            logger.error(f"Error with Anthropic (Claude) generation (model: {model_name}): {str(e)}")
            raise
    
    async def _generate_with_mistral(self, prompt: str, model_name: str, max_tokens: int, temperature: float) -> str:
        """Generate text using Mistral (Sync via run_in_executor)"""
        if not self.mistral_client:
            raise RuntimeError("Mistral client not initialized.")
        try:
            logger.debug(f"Generating with Mistral model: {model_name}, temp: {temperature}, prompt: '{prompt[:50]}...' (max_tokens: {max_tokens} - note: not directly used by MistralClient.chat)")
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.mistral_client.chat(
                    model=model_name, # Use the passed model_name
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, 
                    temperature=temperature
                )
            )
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                if content is not None:
                    return content.strip()
                else:
                    logger.warning(f"Mistral response message content is None for model {model_name}.")
                    return ""
            else:
                logger.warning(f"Mistral response did not contain expected choices or message for model {model_name}.")
                return ""
        except Exception as e:
            logger.error(f"Error with Mistral generation (model: {model_name}): {str(e)}")
            raise
    

    async def summarize(self, text: str, style: str = "concise", max_length: Optional[int] = None) -> str:
        if not text.strip():
            return ""
        
        style_prompts = {
            "concise": "Provide a concise summary of the following text, focusing on the main points:",
            "detailed": "Provide a detailed summary of the following text, including key details and supporting information:",
            "bullet_points": "Summarize the following text as a list of bullet points highlighting the main ideas:",
            "executive": "Provide an executive summary of the following text, focusing on key findings and actionable insights:"
        }
        prompt_template = style_prompts.get(style, style_prompts["concise"])
        if max_length:
            prompt_template += f" Keep the summary under approximately {max_length} words."
        
        prompt = f"{prompt_template}\n\nText to summarize:\n{text}\n\nSummary:"
        
        try:
            summary_max_tokens = (max_length * 2) if max_length else 500 
            summary = await self.generate_text(prompt, model="auto", max_tokens=summary_max_tokens, temperature=0.3)
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating summary"
    
    async def generate_tags(self, text: str, max_tags: int = 5) -> List[str]:
        if not text.strip():
            return []
        
        prompt = f"""Generate up to {max_tags} relevant tags for the following text.
        Tags should be concise, descriptive keywords or phrases (1-3 words typically) that capture the main topics or themes.
        Return only the tags, separated by commas. Do not include any preamble or explanation.

        Text:
        {text}

        Tags:"""
        
        try:
            response = await self.generate_text(prompt, model="auto", max_tokens=100, temperature=0.5)
            tags = [tag.strip().lower() for tag in response.split(',') if tag.strip()]
            tags = [tag for tag in tags if tag and len(tag) > 1 and len(tag) < 50]
            return list(dict.fromkeys(tags))[:max_tags]
        except Exception as e:
            logger.error(f"Error generating tags: {str(e)}")
            return []
    
    async def categorize(self, text: str, categories: List[str]) -> str:
        if not text.strip() or not categories:
            return "Uncategorized"
        
        categories_str = ", ".join([f"'{cat}'" for cat in categories])
        prompt = f"""Classify the following text into ONE of these categories: {categories_str}.
        Choose the single most appropriate category based on the content and main theme of the text.
        Return only the category name as a string, exactly as it appears in the list provided. Do not add any other text or explanation.

        Text to classify:
        {text}

        Category:"""
        
        try:
            response = await self.generate_text(prompt, model="auto", max_tokens=50, temperature=0.1) 
            category_candidate = response.strip().strip("'\"")
            
            for cat in categories:
                if cat.lower() == category_candidate.lower():
                    return cat
            
            logger.warning(f"LLM returned category '{category_candidate}' which is not in the provided list: {categories}. Falling back.")
            return categories[0] if categories else "Uncategorized"
        except Exception as e:
            logger.error(f"Error categorizing text: {str(e)}")
            return "Uncategorized"
    
    async def answer_question(self, question: str, context: str, max_context_length: int = 3000) -> str:
        if not question.strip():
            return "No question provided."
        if not context.strip():
            return "I don't have enough context to answer this question. Please provide relevant information."
        
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            logger.warning(f"Context truncated to {max_context_length} characters for question answering.")
        
        prompt = f"""You are a helpful assistant. Answer the following question based ONLY on the provided context.
If the context does not contain the information to answer the question, state that the context does not provide the answer.
Do not make up information or use external knowledge.

Context:
---
{context}
---

Question: {question}

Answer:"""
        
        try:
            answer = await self.generate_text(prompt, model="auto", max_tokens=300, temperature=0.2)
            return answer.strip()
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return "I encountered an error while trying to answer your question."
    
    async def extract_key_information(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            return {}
        
        prompt = f"""Analyze the following text and extract key information.
        Provide the response as a JSON object with the following keys:
        - "main_topic": (string) The main topic or subject of the text.
        - "key_points": (array of strings) A list of 3-5 key points or takeaways.
        - "entities": (array of strings) Important people, places, organizations, or products mentioned.
        - "sentiment": (string) Overall sentiment of the text (e.g., "positive", "neutral", "negative", "mixed").
        - "content_type": (string) The perceived type of content (e.g., "article", "email", "report", "conversation", "advertisement", "other").

        If a piece of information is not found or not applicable, use null or an empty array/string as appropriate for the JSON structure.

        Text to analyze:
        ---
        {text}
        ---

        JSON Analysis:"""
        
        try:
            response_str = await self.generate_text(prompt, model="auto", max_tokens=500, temperature=0.4)
            
            import json
            try:
                if response_str.startswith("```json"):
                    response_str = response_str.lstrip("```json").rstrip("```").strip()
                
                info = json.loads(response_str)
                expected_keys = {"main_topic", "key_points", "entities", "sentiment", "content_type"}
                if not expected_keys.issubset(info.keys()):
                    logger.warning(f"Extracted information missing some expected keys. Got: {info.keys()}")
                return info
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse JSON from LLM response for key_information: {je}")
                logger.debug(f"LLM Response string was: {response_str}")
                info_fallback = {}
                lines = response_str.split('\n')
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key_clean = key.strip().lower().replace(' ', '_')
                        value_clean = value.strip()
                        if value_clean:
                            if key_clean in ["key_points", "entities"] and '[' in value_clean and ']' in value_clean:
                                try:
                                    info_fallback[key_clean] = [item.strip().strip("'\"") for item in value_clean.strip('[]').split(',') if item.strip()]
                                except: info_fallback[key_clean] = value_clean
                            else: info_fallback[key_clean] = value_clean
                if info_fallback:
                    logger.info("Successfully parsed key information using fallback line-based method.")
                    return info_fallback
                return {"error": "Failed to parse LLM output", "raw_response": response_str}
        except Exception as e:
            logger.error(f"Error extracting key information: {str(e)}")
            return {"error": f"General error extracting key information: {str(e)}"}

    async def check_availability(self) -> Dict[str, bool]:
        """Check which LLM services are available by making a tiny test call."""
        availability = {
            "openai": False,
            "mistral": False,
            "anthropic": False
        }
        test_prompt = "Hello"
        test_max_tokens = 5
        test_temp = 0.1

        logger.info("Checking LLM availability...")

        if self.openai_async_client and self.config.OPENAI_MODEL:
            try:
                logger.debug(f"Testing OpenAI availability with model {self.config.OPENAI_MODEL}...")
                test_response = await self._generate_with_openai(test_prompt, self.config.OPENAI_MODEL, test_max_tokens, test_temp)
                availability["openai"] = bool(test_response.strip())
            except Exception as e:
                logger.warning(f"OpenAI availability check failed for model {self.config.OPENAI_MODEL}: {e}")
        logger.info(f"OpenAI available: {availability['openai']}")
        
        if self.mistral_client and self.config.MISTRAL_MODEL:
            try:
                logger.debug(f"Testing Mistral availability with model {self.config.MISTRAL_MODEL}...")
                test_response = await self._generate_with_mistral(test_prompt, self.config.MISTRAL_MODEL, test_max_tokens, test_temp)
                availability["mistral"] = bool(test_response.strip())
            except Exception as e:
                logger.warning(f"Mistral availability check failed for model {self.config.MISTRAL_MODEL}: {e}")
        logger.info(f"Mistral available: {availability['mistral']}")

        if self.anthropic_client and self.config.ANTHROPIC_MODEL:
            try:
                logger.debug(f"Testing Anthropic availability with model {self.config.ANTHROPIC_MODEL}...")
                test_response = await self._generate_with_claude(test_prompt, self.config.ANTHROPIC_MODEL, test_max_tokens, test_temp)
                availability["anthropic"] = bool(test_response.strip())
            except Exception as e:
                logger.warning(f"Anthropic availability check failed for model {self.config.ANTHROPIC_MODEL}: {e}")
        logger.info(f"Anthropic available: {availability['anthropic']}")
        
        logger.info(f"Final LLM Availability: {availability}")
        return availability