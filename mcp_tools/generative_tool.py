import logging
from typing import List, Dict, Any, Optional
import asyncio

from services.llm_service import LLMService
from mcp_tools.search_tool import SearchTool
from core.models import SearchResult

logger = logging.getLogger(__name__)

class GenerativeTool:
    def __init__(self, llm_service: LLMService, search_tool: Optional[SearchTool] = None):
        self.llm_service = llm_service
        self.search_tool = search_tool
    
    async def summarize(self, content: str, style: str = "concise", max_length: Optional[int] = None) -> str:
        """Generate a summary of the given content"""
        try:
            if not content.strip():
                return "No content provided for summarization."
            
            logger.info(f"Generating {style} summary for content of length {len(content)}")
            
            summary = await self.llm_service.summarize(content, style, max_length)
            
            logger.info(f"Generated summary of length {len(summary)}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    async def generate_tags(self, content: str, max_tags: int = 5) -> List[str]:
        """Generate relevant tags for the given content"""
        try:
            if not content.strip():
                return []
            
            logger.info(f"Generating up to {max_tags} tags for content")
            
            tags = await self.llm_service.generate_tags(content, max_tags)
            
            logger.info(f"Generated {len(tags)} tags")
            return tags
            
        except Exception as e:
            logger.error(f"Error generating tags: {str(e)}")
            return []
    
    async def categorize(self, content: str, categories: List[str]) -> str:
        """Categorize content into one of the provided categories"""
        try:
            if not content.strip():
                return "Uncategorized"
            
            if not categories:
                categories = ["Technology", "Business", "Science", "Education", "Entertainment", "News", "Research", "Other"]
            
            logger.info(f"Categorizing content into one of {len(categories)} categories")
            
            category = await self.llm_service.categorize(content, categories)
            
            logger.info(f"Categorized as: {category}")
            return category
            
        except Exception as e:
            logger.error(f"Error categorizing content: {str(e)}")
            return "Uncategorized"
    
    async def answer_question(self, question: str, context_results: List[SearchResult] = None) -> str:
        """Answer a question using the provided context or RAG"""
        try:
            if not question.strip():
                return "No question provided."
            
            logger.info(f"Answering question: {question[:100]}...")
            
            # If no context provided and search tool is available, search for relevant context
            if not context_results and self.search_tool:
                logger.info("No context provided, searching for relevant information")
                context_results = await self.search_tool.search(question, top_k=5)
            
            # Prepare context from search results
            if context_results:
                context_texts = []
                for result in context_results:
                    context_texts.append(f"Source: {result.document_id}\nContent: {result.content}\n")
                
                context = "\n---\n".join(context_texts)
                logger.info(f"Using context from {len(context_results)} sources")
            else:
                context = ""
                logger.info("No context available for answering question")
            
            # Generate answer
            answer = await self.llm_service.answer_question(question, context)
            
            logger.info(f"Generated answer of length {len(answer)}")
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"I encountered an error while trying to answer your question: {str(e)}"
    
    async def generate_outline(self, topic: str, num_sections: int = 5, detail_level: str = "medium") -> str:
        """Generate an outline for the given topic"""
        try:
            if not topic.strip():
                return "No topic provided."
            
            detail_descriptions = {
                "brief": "brief bullet points",
                "medium": "detailed bullet points with descriptions",
                "detailed": "comprehensive outline with sub-sections and explanations"
            }
            
            detail_desc = detail_descriptions.get(detail_level, "detailed bullet points")
            
            prompt = f"""Create a {detail_desc} outline for the topic: "{topic}"
            
            The outline should have {num_sections} main sections and be well-structured and informative.
            
            Format the outline clearly with proper numbering and indentation.
            
            Topic: {topic}
            
            Outline:"""
            
            outline = await self.llm_service.generate_text(prompt, max_tokens=800, temperature=0.7)
            
            logger.info(f"Generated outline for topic: {topic}")
            return outline
            
        except Exception as e:
            logger.error(f"Error generating outline: {str(e)}")
            return f"Error generating outline: {str(e)}"
    
    async def explain_concept(self, concept: str, audience: str = "general", length: str = "medium") -> str:
        """Explain a concept for a specific audience"""
        try:
            if not concept.strip():
                return "No concept provided."
            
            audience_styles = {
                "general": "a general audience using simple, clear language",
                "technical": "a technical audience with appropriate jargon and detail",
                "beginner": "beginners with no prior knowledge, using analogies and examples",
                "expert": "experts in the field with advanced terminology and depth"
            }
            
            length_guidance = {
                "brief": "Keep the explanation concise and to the point (2-3 paragraphs).",
                "medium": "Provide a comprehensive explanation (4-6 paragraphs).",
                "detailed": "Give a thorough, in-depth explanation with examples."
            }
            
            audience_desc = audience_styles.get(audience, "a general audience")
            length_desc = length_guidance.get(length, "Provide a comprehensive explanation.")
            
            prompt = f"""Explain the concept of "{concept}" for {audience_desc}.
            
            {length_desc}
            
            Make sure to:
            - Use appropriate language for the audience
            - Include relevant examples or analogies
            - Structure the explanation logically
            - Ensure clarity and accuracy
            
            Concept to explain: {concept}
            
            Explanation:"""
            
            explanation = await self.llm_service.generate_text(prompt, max_tokens=600, temperature=0.5)
            
            logger.info(f"Generated explanation for concept: {concept}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining concept: {str(e)}")
            return f"Error explaining concept: {str(e)}"
    
    async def compare_concepts(self, concept1: str, concept2: str, aspects: List[str] = None) -> str:
        """Compare two concepts across specified aspects"""
        try:
            if not concept1.strip() or not concept2.strip():
                return "Both concepts must be provided for comparison."
            
            if not aspects:
                aspects = ["definition", "key features", "advantages", "disadvantages", "use cases"]
            
            aspects_str = ", ".join(aspects)
            
            prompt = f"""Compare and contrast "{concept1}" and "{concept2}" across the following aspects: {aspects_str}.
            
            Structure your comparison clearly, addressing each aspect for both concepts.
            
            Format:
            ## Comparison: {concept1} vs {concept2}
            
            For each aspect, provide:
            - **{concept1}**: [description]
            - **{concept2}**: [description]
            - **Key Difference**: [summary]
            
            Concepts to compare:
            1. {concept1}
            2. {concept2}
            
            Comparison:"""
            
            comparison = await self.llm_service.generate_text(prompt, max_tokens=800, temperature=0.6)
            
            logger.info(f"Generated comparison between {concept1} and {concept2}")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing concepts: {str(e)}")
            return f"Error comparing concepts: {str(e)}"
    
    async def generate_questions(self, content: str, question_type: str = "comprehension", num_questions: int = 5) -> List[str]:
        """Generate questions based on the provided content"""
        try:
            if not content.strip():
                return []
            
            question_types = {
                "comprehension": "comprehension questions that test understanding of key concepts",
                "analysis": "analytical questions that require deeper thinking and evaluation",
                "application": "application questions that ask how to use the concepts in practice",
                "creative": "creative questions that encourage original thinking and exploration",
                "factual": "factual questions about specific details and information"
            }
            
            question_desc = question_types.get(question_type, "comprehension questions")
            
            prompt = f"""Based on the following content, generate {num_questions} {question_desc}.
            
            The questions should be:
            - Clear and well-formulated
            - Relevant to the content
            - Appropriate for the specified type
            - Engaging and thought-provoking
            
            Content:
            {content[:2000]}  # Limit content length
            
            Questions:"""
            
            response = await self.llm_service.generate_text(prompt, max_tokens=400, temperature=0.7)
            
            # Parse questions from response
            questions = []
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and ('?' in line or line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*'))):
                    # Clean up the question
                    question = line.lstrip('0123456789.-* ').strip()
                    if question and '?' in question:
                        questions.append(question)
            
            logger.info(f"Generated {len(questions)} {question_type} questions")
            return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []
    
    async def paraphrase_text(self, text: str, style: str = "formal", preserve_meaning: bool = True) -> str:
        """Paraphrase text in a different style while preserving meaning"""
        try:
            if not text.strip():
                return "No text provided for paraphrasing."
            
            style_instructions = {
                "formal": "formal, professional language",
                "casual": "casual, conversational language",
                "academic": "academic, scholarly language",
                "simple": "simple, easy-to-understand language",
                "technical": "technical, precise language"
            }
            
            style_desc = style_instructions.get(style, "clear, appropriate language")
            meaning_instruction = "while preserving the exact meaning and key information" if preserve_meaning else "while maintaining the general intent"
            
            prompt = f"""Paraphrase the following text using {style_desc} {meaning_instruction}.
            
            Original text:
            {text}
            
            Paraphrased text:"""
            
            paraphrase = await self.llm_service.generate_text(prompt, max_tokens=len(text.split()) * 2, temperature=0.6)
            
            logger.info(f"Paraphrased text in {style} style")
            return paraphrase.strip()
            
        except Exception as e:
            logger.error(f"Error paraphrasing text: {str(e)}")
            return f"Error paraphrasing text: {str(e)}"
    
    async def extract_key_insights(self, content: str, num_insights: int = 5) -> List[str]:
        """Extract key insights from the provided content"""
        try:
            if not content.strip():
                return []
            
            prompt = f"""Analyze the following content and extract {num_insights} key insights or takeaways.
            
            Each insight should be:
            - A clear, concise statement
            - Significant and meaningful
            - Based on the content provided
            - Actionable or thought-provoking when possible
            
            Content:
            {content[:3000]}  # Limit content length
            
            Key Insights:"""
            
            response = await self.llm_service.generate_text(prompt, max_tokens=400, temperature=0.6)
            
            # Parse insights from response
            insights = []
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*')) or len(insights) == 0):
                    # Clean up the insight
                    insight = line.lstrip('0123456789.-* ').strip()
                    if insight and len(insight) > 10:  # Minimum insight length
                        insights.append(insight)
            
            logger.info(f"Extracted {len(insights)} key insights")
            return insights[:num_insights]
            
        except Exception as e:
            logger.error(f"Error extracting insights: {str(e)}")
            return []