
import logging
import asyncio
from pathlib import Path
import os
import base64 # For encoding files
from typing import Optional, List, Dict, Any
import json 

from mistralai import Mistral
from mistralai.models import SDKError
# PIL (Pillow) for dummy image creation in main_example
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        self.api_key = os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            logger.error("MISTRAL_API_KEY environment variable not set.")
            raise ValueError("MISTRAL_API_KEY not found in environment variables.")
        
        self.client = Mistral(api_key=self.api_key)
        self.ocr_model_name = "mistral-ocr-latest"
        self.language = 'eng'
        logger.info(f"OCRService (using Mistral AI model {self.ocr_model_name}) initialized.")

    def _encode_file_to_base64(self, file_path: str) -> Optional[str]:
        try:
            with open(file_path, "rb") as file_to_encode:
                return base64.b64encode(file_to_encode.read()).decode('utf-8')
        except FileNotFoundError:
            logger.error(f"Error: The file {file_path} was not found for Base64 encoding.")
            return None
        except Exception as e:
            logger.error(f"Error during Base64 encoding for {file_path}: {e}")
            return None

    # In OCRService class:

    async def _process_file_with_mistral(self, file_path: str, mime_type: str) -> str:
        file_name = Path(file_path).name
        logger.info(f"Preparing to process file: {file_name} (MIME: {mime_type}) with Mistral OCR.")

        base64_encoded_file = self._encode_file_to_base64(file_path)
        if not base64_encoded_file:
            logger.warning(f"Base64 encoding failed for {file_name}, cannot process.")
            return ""

        document_type = "image_url" if mime_type.startswith("image/") else "document_url"
        uri_key = "image_url" if document_type == "image_url" else "document_url"
        data_uri = f"data:{mime_type};base64,{base64_encoded_file}"
        
        document_payload = {
            "type": document_type,
            uri_key: data_uri
        }
        try:
            logger.info(f"Calling Mistral client.ocr.process for {file_name} with model {self.ocr_model_name}.")
            loop = asyncio.get_event_loop()
            
            ocr_response = await loop.run_in_executor(
                None, 
                lambda: self.client.ocr.process(
                    model=self.ocr_model_name,
                    document=document_payload,
                    include_image_base64=False 
                )
            )
            
            logger.info(f"Received OCR response for {file_name}. Type: {type(ocr_response)}")

            extracted_markdown = ""
            if hasattr(ocr_response, 'pages') and ocr_response.pages and isinstance(ocr_response.pages, list):
                all_pages_markdown = []
                for i, page in enumerate(ocr_response.pages):
                    page_content = None
                    if hasattr(page, 'markdown') and page.markdown: # Check for 'markdown' attribute
                        page_content = page.markdown
                        logger.debug(f"Extracted content from page {i} using 'page.markdown'.")
                    elif hasattr(page, 'markdown_content') and page.markdown_content:
                        page_content = page.markdown_content
                        logger.debug(f"Extracted content from page {i} using 'page.markdown_content'.")
                    elif hasattr(page, 'text') and page.text: 
                        page_content = page.text
                        logger.debug(f"Extracted content from page {i} using 'page.text'.")
                    
                    if page_content:
                        all_pages_markdown.append(page_content)
                    else:
                        page_details_for_log = str(page)[:200] # Default to string snippet
                        if hasattr(page, '__dict__'):
                             page_details_for_log = str(vars(page))[:200] # Log part of vars if it's an object
                        logger.warning(f"Page {i} in OCR response for {file_name} has no 'markdown', 'markdown_content', or 'text'. Page details: {page_details_for_log}")
                
                if all_pages_markdown:
                    extracted_markdown = "\n\n---\nPage Break (simulated)\n---\n\n".join(all_pages_markdown) # Simulate page breaks
                else:
                    logger.warning(f"'pages' attribute found but no content extracted from any pages for {file_name}.")

            # Fallbacks if ocr_response doesn't have 'pages' but might have direct text/markdown
            elif hasattr(ocr_response, 'text') and ocr_response.text:
                 extracted_markdown = ocr_response.text
                 logger.info(f"Extracted content from 'ocr_response.text' (no pages structure) for {file_name}.")
            elif hasattr(ocr_response, 'markdown') and ocr_response.markdown:
                 extracted_markdown = ocr_response.markdown
                 logger.info(f"Extracted content from 'ocr_response.markdown' (no pages structure) for {file_name}.")
            elif isinstance(ocr_response, str) and ocr_response:
                 extracted_markdown = ocr_response
                 logger.info(f"OCR response is a direct non-empty string for {file_name}.")
            else:
                logger.warning(f"Could not extract markdown from OCR response for {file_name} using known attributes (pages, text, markdown).")

            if not extracted_markdown.strip():
                logger.warning(f"Extracted markdown is empty for {file_name} after all parsing attempts.")
            
            return extracted_markdown.strip()

        except SDKError as e:
            logger.error(f"Mistral API Exception during client.ocr.process for {file_name}: {e.message}")
            logger.exception("SDKError details:")
            return ""
        except Exception as e:
            logger.error(f"Generic Exception during Mistral client.ocr.process call for {file_name}: {e}")
            logger.exception("Exception details:")
            return ""

    async def extract_text_from_image(self, image_path: str, language: Optional[str] = None) -> str:
        if language: 
            logger.info(f"Language parameter '{language}' provided, but Mistral OCR is broadly multilingual.")
        
        ext = Path(image_path).suffix.lower()
        mime_map = {'.jpeg': 'image/jpeg', '.jpg': 'image/jpeg', '.png': 'image/png', 
                    '.gif': 'image/gif', '.bmp': 'image/bmp', '.tiff': 'image/tiff', '.webp': 'image/webp',
                    '.avif': 'image/avif'} 
        mime_type = mime_map.get(ext)
        if not mime_type:
            logger.warning(f"Unsupported image extension '{ext}' for path '{image_path}'. Attempting with 'application/octet-stream'.")
            mime_type = 'application/octet-stream' 
            
        return await self._process_file_with_mistral(image_path, mime_type)

    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        return await self._process_file_with_mistral(pdf_path, "application/pdf")

    async def extract_text_from_pdf_images(self, pdf_path: str) -> List[str]:
        logger.info("Mistral processes PDFs directly. This method will return the full Markdown content as a single list item.")
        full_markdown = await self._process_file_with_mistral(pdf_path, "application/pdf")
        if full_markdown:
            return [full_markdown]
        return [""] 

    async def extract_text_with_confidence(self, image_path: str, min_confidence: float = 0.5) -> Dict[str, Any]:
        logger.warning("Mistral Document AI API (ocr.process) typically returns structured text (Markdown). Word-level confidence scores are not standard. 'confidence' field is a placeholder.")
        
        ext = Path(image_path).suffix.lower()
        mime_map = {'.jpeg': 'image/jpeg', '.jpg': 'image/jpeg', '.png': 'image/png', '.avif': 'image/avif'}
        mime_type = mime_map.get(ext)
        if not mime_type:
            logger.warning(f"Unsupported image extension '{ext}' in extract_text_with_confidence. Defaulting mime type.")
            mime_type = 'application/octet-stream'
        
        text_markdown = await self._process_file_with_mistral(image_path, mime_type)
        
        return {
            "text": text_markdown, 
            "confidence": 0.0, 
            "word_count": len(text_markdown.split()) if text_markdown else 0, 
            "raw_data": "Mistral ocr.process response contains structured data. See logs from _process_file_with_mistral for details."
        }

    async def detect_language(self, image_path: str) -> str:
        logger.warning("Mistral OCR is multilingual; explicit language detection is not part of client.ocr.process.")
        return 'eng' 

    async def extract_tables_from_image(self, image_path: str) -> List[List[str]]:
        logger.info("Extracting text (Markdown) from image using Mistral. Mistral OCR preserves table structures in Markdown.")
        
        ext = Path(image_path).suffix.lower()
        mime_map = {'.jpeg': 'image/jpeg', '.jpg': 'image/jpeg', '.png': 'image/png', '.avif': 'image/avif'}
        mime_type = mime_map.get(ext)
        if not mime_type:
             logger.warning(f"Unsupported image extension '{ext}' in extract_tables_from_image. Defaulting mime type.")
             mime_type = 'application/octet-stream'
        
        markdown_content = await self._process_file_with_mistral(image_path, mime_type)
        
        if markdown_content:
            logger.info("Attempting basic parsing of Markdown tables. For complex tables, a dedicated parser is recommended.")
            table_data = []
            # Simplified parsing logic for example purposes - can be improved significantly.
            lines = markdown_content.split('\n')
            for line in lines:
                stripped_line = line.strip()
                if stripped_line.startswith('|') and stripped_line.endswith('|') and "---" not in stripped_line:
                    cells = [cell.strip() for cell in stripped_line.strip('|').split('|')]
                    if any(cells):
                        table_data.append(cells)
            
            if table_data:
                 logger.info(f"Extracted {len(table_data)} lines potentially forming tables using basic parsing.")
            else:
                 logger.info("No distinct table structures found with basic parsing from extracted markdown.")
            return table_data
        return []

    async def get_supported_languages(self) -> List[str]:
        logger.info("Mistral OCR is multilingual. Refer to official Mistral AI documentation for details.")
        return ['eng', 'multilingual (refer to Mistral documentation)']

    async def validate_ocr_setup(self) -> Dict[str, Any]:
        try:
            models_response = await asyncio.to_thread(self.client.models.list)
            model_ids = [model.id for model in models_response.data]
            return {
                "status": "operational",
                "message": "Mistral client initialized. API key present. Model listing successful.",
                "mistral_available_models_sample": model_ids[:5],
                "configured_ocr_model": self.ocr_model_name,
            }
        except SDKError as e:
            logger.error(f"Mistral API Exception during setup validation: {e.message}")
            return { "status": "error", "error": f"Mistral API Error: {e.message}"}
        except Exception as e:
            logger.error(f"Generic error during Mistral OCR setup validation: {str(e)}")
            return { "status": "error", "error": str(e) }
    
    def extract_text(self, file_path: str) -> str:
        logger.warning("`extract_text` is a synchronous method. Running async Mistral OCR in a blocking way.")
        try:
            ext = Path(file_path).suffix.lower()
            if ext in ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.avif']:
                result = asyncio.run(self.extract_text_from_image(file_path))
            elif ext == '.pdf':
                result = asyncio.run(self.extract_text_from_pdf(file_path))
            else:
                logger.error(f"Unsupported file type for sync extract_text: {file_path}")
                return "Unsupported file type."
            return result
        except Exception as e:
            logger.error(f"Error in synchronous extract_text for {file_path}: {str(e)}")
            return "Error during sync extraction."

# Example of how to use the OCRService (main execution part)
async def main_example():
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s')

    if not os.environ.get("MISTRAL_API_KEY"):
       logger.error("MISTRAL_API_KEY environment variable is not set. Please set it: export MISTRAL_API_KEY='yourkey'")
       return

    ocr_service = OCRService()
    
    logger.info("--- Validating OCR Service Setup ---")
    validation_status = await ocr_service.validate_ocr_setup()
    logger.info(f"OCR Service Validation: {validation_status}")
    if validation_status.get("status") == "error":
        logger.error("Halting due to validation error.")
        return

    # --- Test with a specific PDF file ---
    pdf_path_to_test = r"C:\path\to\your\certificate.pdf"

    if os.path.exists(pdf_path_to_test):
        logger.info(f"\n--- Extracting text from specific PDF: {pdf_path_to_test} ---")
        # Using the method that aligns with original `extract_text_from_pdf_images` signature
        pdf_markdown_list = await ocr_service.extract_text_from_pdf_images(pdf_path_to_test)
        if pdf_markdown_list and pdf_markdown_list[0]:
            logger.info(f"Extracted Markdown from PDF ({pdf_path_to_test}):\n" + pdf_markdown_list[0])
        else: 
            logger.warning(f"No text extracted from PDF {pdf_path_to_test} or an error occurred.")
    else:
        logger.warning(f"PDF file for specific test '{pdf_path_to_test}' not found. Skipping this test.")
        logger.warning("Please update `pdf_path_to_test` in `main_example` to a valid PDF path.")

    image_path = "dummy_test_image_ocr.png" 
    if os.path.exists(image_path):
        logger.info(f"\n---Extracting text from image: {image_path} ---")
        # ... image processing logic ...
        pass
    else:
        logger.info(f"Dummy image {image_path} not created or found, skipping optional image test.")


if __name__ == '__main__':
    asyncio.run(main_example())