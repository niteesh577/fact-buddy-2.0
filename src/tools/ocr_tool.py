import os
import logging
import pytesseract
from PIL import Image
from typing import Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class OCRTool:
    """Tool for extracting text from images using OCR (Optical Character Recognition)"""
    
    def __init__(self):
        # Check if tesseract is installed and configured
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.warning(f"Tesseract OCR may not be properly installed: {str(e)}")
            logger.warning("Install Tesseract OCR with: brew install tesseract")
    
    def extract_text(self, image_path: str, include_confidence: bool = False) -> Dict[str, Any]:
        """
        Extract text from an image file
        
        Args:
            image_path: Path to the image file
            include_confidence: Whether to include confidence scores in the result
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            logger.info(f"Extracting text from image: {image_path}")
            
            # Check if file exists
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}", "text": ""}
            
            # Open the image
            image = Image.open(image_path)
            
            # Extract text with detailed data
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Extract full text
            full_text = pytesseract.image_to_string(image)
            
            # Process OCR data
            word_confidences = []
            text_boxes = []
            
            for i in range(len(ocr_data["text"])):
                # Skip empty text
                if not ocr_data["text"][i].strip():
                    continue
                
                # Get confidence score
                conf = int(ocr_data["conf"][i])
                if conf > 0:  # Only include valid confidence scores
                    word_confidences.append(conf / 100.0)  # Convert to 0-1 scale
                
                # Get bounding box
                x = ocr_data["left"][i]
                y = ocr_data["top"][i]
                w = ocr_data["width"][i]
                h = ocr_data["height"][i]
                
                text_boxes.append({
                    "text": ocr_data["text"][i],
                    "box": (x, y, x + w, y + h),
                    "confidence": conf / 100.0
                })
            
            result = {
                "text": full_text,
                "language": pytesseract.image_to_osd(image) if len(full_text) > 10 else "Unknown",
                "word_count": len([w for w in full_text.split() if w.strip()]),
                "status": "success"
            }
            
            if include_confidence and word_confidences:
                result["word_confidences"] = word_confidences
                result["avg_confidence"] = sum(word_confidences) / len(word_confidences)
                result["text_boxes"] = text_boxes
            
            logger.info(f"Successfully extracted {result['word_count']} words from image")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return {
                "error": str(e),
                "text": "",
                "status": "failed"
            }