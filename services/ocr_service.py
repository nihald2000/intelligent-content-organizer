import logging
from typing import Optional, List, Dict, Any
import asyncio
from pathlib import Path
import tempfile
import os

from PIL import Image
import pytesseract
import config

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        self.config = config.config
        
        # Configure Tesseract path if specified
        if self.config.TESSERACT_PATH:
            pytesseract.pytesseract.tesseract_cmd = self.config.TESSERACT_PATH
        
        self.language = self.config.OCR_LANGUAGE
        
        # Test OCR availability
        self._test_ocr_availability()
    
    def _test_ocr_availability(self):
        """Test if OCR is available and working"""
        try:
            # Create a simple test image
            test_image = Image.new('RGB', (100, 30), color='white')
            pytesseract.image_to_string(test_image)
            logger.info("OCR service initialized successfully")
        except Exception as e:
            logger.warning(f"OCR may not be available: {str(e)}")
    
    async def extract_text_from_image(self, image_path: str, language: Optional[str] = None) -> str:
        """Extract text from an image file"""
        try:
            # Use specified language or default
            lang = language or self.language
            
            # Load image
            image = Image.open(image_path)
            
            # Perform OCR in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                self._extract_text_sync,
                image,
                lang
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {str(e)}")
            return ""
    
    def _extract_text_sync(self, image: Image.Image, language: str) -> str:
        """Synchronous text extraction"""
        try:
            # Optimize image for OCR
            processed_image = self._preprocess_image(image)
            
            # Configure OCR
            config_string = '--psm 6'  # Assume a single uniform block of text
            
            # Extract text
            text = pytesseract.image_to_string(
                processed_image,
                lang=language,
                config=config_string
            )
            
            return text
        except Exception as e:
            logger.error(f"Error in synchronous OCR: {str(e)}")
            return ""
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image to improve OCR accuracy"""
        try:
            # Convert to grayscale if not already
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize image if too small (OCR works better on larger images)
            width, height = image.size
            if width < 300 or height < 300:
                scale_factor = max(300 / width, 300 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return image
    
    async def extract_text_from_pdf_images(self, pdf_path: str) -> List[str]:
        """Extract text from PDF by converting pages to images and running OCR"""
        try:
            import fitz  # PyMuPDF
            
            texts = []
            
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                try:
                    # Get page
                    page = pdf_document[page_num]
                    
                    # Convert page to image
                    mat = fitz.Matrix(2.0, 2.0)  # Scale factor for better quality
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("ppm")
                    
                    # Create PIL image from bytes
                    with tempfile.NamedTemporaryFile(suffix='.ppm', delete=False) as tmp_file:
                        tmp_file.write(img_data)
                        tmp_file.flush()
                        
                        # Extract text from image
                        page_text = await self.extract_text_from_image(tmp_file.name)
                        texts.append(page_text)
                        
                        # Clean up temporary file
                        os.unlink(tmp_file.name)
                
                except Exception as e:
                    logger.warning(f"Error processing PDF page {page_num}: {str(e)}")
                    texts.append("")
            
            pdf_document.close()
            return texts
            
        except ImportError:
            logger.error("PyMuPDF not available for PDF OCR")
            return []
        except Exception as e:
            logger.error(f"Error extracting text from PDF images: {str(e)}")
            return []
    
    async def extract_text_with_confidence(self, image_path: str, min_confidence: float = 0.5) -> Dict[str, Any]:
        """Extract text with confidence scores"""
        try:
            image = Image.open(image_path)
            
            # Get detailed OCR data with confidence scores
            loop = asyncio.get_event_loop()
            ocr_data = await loop.run_in_executor(
                None,
                self._extract_detailed_data,
                image
            )
            
            # Filter by confidence
            filtered_text = []
            word_confidences = []
            
            for i, confidence in enumerate(ocr_data.get('conf', [])):
                if confidence > min_confidence * 100:  # Tesseract uses 0-100 scale
                    text = ocr_data.get('text', [])[i]
                    if text.strip():
                        filtered_text.append(text)
                        word_confidences.append(confidence / 100.0)  # Convert to 0-1 scale
            
            return {
                "text": " ".join(filtered_text),
                "confidence": sum(word_confidences) / len(word_confidences) if word_confidences else 0.0,
                "word_count": len(filtered_text),
                "raw_data": ocr_data
            }
            
        except Exception as e:
            logger.error(f"Error extracting text with confidence: {str(e)}")
            return {
                "text": "",
                "confidence": 0.0,
                "word_count": 0,
                "error": str(e)
            }
    
    def _extract_detailed_data(self, image: Image.Image) -> Dict[str, Any]:
        """Extract detailed OCR data with positions and confidence"""
        try:
            processed_image = self._preprocess_image(image)
            
            # Get detailed data
            data = pytesseract.image_to_data(
                processed_image,
                lang=self.language,
                config='--psm 6',
                output_type=pytesseract.Output.DICT
            )
            
            return data
        except Exception as e:
            logger.error(f"Error extracting detailed OCR data: {str(e)}")
            return {}
    
    async def detect_language(self, image_path: str) -> str:
        """Detect the language of text in an image"""
        try:
            image = Image.open(image_path)
            
            # Run language detection
            loop = asyncio.get_event_loop()
            languages = await loop.run_in_executor(
                None,
                pytesseract.image_to_osd,
                image
            )
            
            # Parse the output to get the language
            for line in languages.split('\n'):
                if 'Script:' in line:
                    script = line.split(':')[1].strip()
                    # Map script to language code
                    script_to_lang = {
                        'Latin': 'eng',
                        'Arabic': 'ara',
                        'Chinese': 'chi_sim',
                        'Japanese': 'jpn',
                        'Korean': 'kor'
                    }
                    return script_to_lang.get(script, 'eng')
            
            return 'eng'  # Default to English
            
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return 'eng'
    
    async def extract_tables_from_image(self, image_path: str) -> List[List[str]]:
        """Extract table data from an image"""
        try:
            # This is a basic implementation
            # For better table extraction, consider using specialized libraries like table-transformer
            
            image = Image.open(image_path)
            
            # Use specific PSM for tables
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                lambda: pytesseract.image_to_string(
                    image,
                    lang=self.language,
                    config='--psm 6 -c preserve_interword_spaces=1'
                )
            )
            
            # Simple table parsing (assumes space/tab separated)
            lines = text.split('\n')
            table_data = []
            
            for line in lines:
                if line.strip():
                    # Split by multiple spaces or tabs
                    cells = [cell.strip() for cell in line.split() if cell.strip()]
                    if cells:
                        table_data.append(cells)
            
            return table_data
            
        except Exception as e:
            logger.error(f"Error extracting tables from image: {str(e)}")
            return []
    
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported OCR languages"""
        try:
            languages = pytesseract.get_languages()
            return sorted(languages)
        except Exception as e:
            logger.error(f"Error getting supported languages: {str(e)}")
            return ['eng']  # Default to English only
    
    async def validate_ocr_setup(self) -> Dict[str, Any]:
        """Validate OCR setup and return status"""
        try:
            # Test basic functionality
            test_image = Image.new('RGB', (200, 50), color='white')
            
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(test_image)
            
            try:
                # Try to use a default font
                draw.text((10, 10), "Test OCR", fill='black')
            except:
                # Fall back to basic text without font
                draw.text((10, 10), "Test", fill='black')
            
            # Test OCR
            result = pytesseract.image_to_string(test_image)
            
            # Get available languages
            languages = await self.get_supported_languages()
            
            return {
                "status": "operational",
                "tesseract_version": pytesseract.get_tesseract_version(),
                "available_languages": languages,
                "current_language": self.language,
                "test_result": result.strip(),
                "tesseract_path": pytesseract.pytesseract.tesseract_cmd
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "tesseract_path": pytesseract.pytesseract.tesseract_cmd
            }
    
    def extract_text(self, file_path):
        # Dummy implementation for OCR
        return "OCR functionality not implemented yet."