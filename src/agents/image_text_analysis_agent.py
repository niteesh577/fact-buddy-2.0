from typing import Dict, Any
from datetime import datetime
import logging
import os

from src.tools.ocr_tool import OCRTool
from src.tools.search import SearchTool
from src.tools.web_scraper import WebScraperTool
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

class ImageTextAnalysisAgent:
    """
    Agent responsible for extracting and analyzing text content from images.
    """
    
    def __init__(self):
        self.ocr_tool = OCRTool()
        self.search_tool = SearchTool()
        self.web_scraper = WebScraperTool()
        self.llm = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768"
        )
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image_path = state.get("image_path")
            if not image_path or not os.path.exists(image_path):
                raise ValueError(f"Invalid image path: {image_path}")
            
            # Extract text from image
            logger.info(f"Extracting text from image: {image_path}")
            ocr_results = self.ocr_tool.extract_text(image_path)
            extracted_text = ocr_results.get("text", "")
            
            # If no text was found, note this and return
            if not extracted_text or len(extracted_text.strip()) < 10:
                logger.info("No significant text found in the image")
                state["text_content"] = {
                    "has_text": False,
                    "ocr_results": ocr_results
                }
                state.setdefault("messages", []).append({
                    "agent": "image_text_analysis",
                    "content": "No significant text found in the image",
                    "timestamp": datetime.now().isoformat()
                })
                return state
            
            # Analyze the extracted text
            logger.info(f"Analyzing extracted text: {extracted_text[:100]}...")
            
            # Search for information related to the text
            search_query = f"fact check: {extracted_text[:200]}"  # Use first 200 chars for search
            search_results = self.search_tool.search(search_query)
            
            # Scrape top search results
            scraped_results = []
            for result in search_results.get("organic_results", [])[:3]:  # Top 3 results
                url = result.get("link")
                if url:
                    try:
                        scraped_content = self.web_scraper.scrape(url)
                        scraped_results.append({
                            "url": url,
                            "title": result.get("title", ""),
                            "content": scraped_content
                        })
                    except Exception as e:
                        logger.error(f"Error scraping URL {url}: {str(e)}")
            
            # Use LLM to analyze the text content
            text_analysis = self._analyze_text_content(extracted_text, scraped_results)
            
            # Add results to state
            state["text_content"] = {
                "has_text": True,
                "extracted_text": extracted_text,
                "ocr_results": ocr_results,
                "search_results": search_results,
                "scraped_results": scraped_results,
                "analysis": text_analysis
            }
            
            # Add message about completion
            state.setdefault("messages", []).append({
                "agent": "image_text_analysis",
                "content": f"Text analysis completed: {len(extracted_text)} characters extracted",
                "timestamp": datetime.now().isoformat()
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error in image text analysis: {str(e)}")
            state.setdefault("messages", []).append({
                "agent": "image_text_analysis",
                "content": f"Text analysis failed: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            state["text_content"] = {"error": str(e)}
            return state
    
    def _analyze_text_content(self, extracted_text, scraped_results):
        """Use LLM to analyze the extracted text and related content"""
        # Prepare context from scraped results
        context = ""
        for i, result in enumerate(scraped_results):
            content = result.get("content", {}).get("content", "")
            if content:
                context += f"\nSource {i+1} ({result.get('url')}): {content[:1000]}...\n"
        
        # Prepare prompt for LLM
        prompt = f"""
        Analyze the following text extracted from an image:
        
        TEXT FROM IMAGE:
        {extracted_text}
        
        RELATED INFORMATION FROM WEB SOURCES:
        {context}
        
        Please provide:
        1. A factual assessment of the text content
        2. Identification of any claims made in the text
        3. Verification of these claims based on the provided sources
        4. An overall assessment of the text's accuracy and reliability
        
        Format your response as JSON with the following structure:
        {{
            "claims": [list of claims identified],
            "verification": [assessment of each claim],
            "accuracy_score": [0-1 score of overall accuracy],
            "misleading_elements": [any potentially misleading aspects],
            "conclusion": [overall assessment]
        }}
        """
        
        # Get LLM response
        try:
            response = self.llm.invoke(prompt)
            # Extract JSON from response
            import re
            import json
            json_match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(1))
            else:
                # Try to find any JSON-like structure
                json_match = re.search(r'({.*})', response.content, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group(1))
                else:
                    # Fallback to using the whole response
                    analysis = {
                        "claims": [],
                        "verification": [],
                        "accuracy_score": 0.5,
                        "misleading_elements": ["Could not parse structured analysis"],
                        "conclusion": response.content
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing text with LLM: {str(e)}")
            return {
                "error": str(e),
                "claims": [],
                "verification": [],
                "accuracy_score": 0,
                "misleading_elements": ["Analysis failed"],
                "conclusion": f"Failed to analyze text: {str(e)}"
            }
    
    def extract_entities(self, text):
        """Extract named entities from text"""
        try:
            prompt = f"""
            Extract all named entities from the following text:
            
            {text}
            
            Return a JSON object with these categories:
            - people: List of people mentioned
            - organizations: List of organizations mentioned
            - locations: List of locations mentioned
            - dates: List of dates or time periods mentioned
            - products: List of products mentioned
            - events: List of events mentioned
            
            Format as:
            {{
                "people": [],
                "organizations": [],
                "locations": [],
                "dates": [],
                "products": [],
                "events": []
            }}
            """
            
            response = self.llm.invoke(prompt)
            
            # Extract JSON from response
            import re
            import json
            json_match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group(1))
            else:
                # Try to find any JSON-like structure
                json_match = re.search(r'({.*})', response.content, re.DOTALL)
                if json_match:
                    entities = json.loads(json_match.group(1))
                else:
                    entities = {
                        "people": [],
                        "organizations": [],
                        "locations": [],
                        "dates": [],
                        "products": [],
                        "events": [],
                        "error": "Could not parse structured entity data"
                    }
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {
                "people": [],
                "organizations": [],
                "locations": [],
                "dates": [],
                "products": [],
                "events": [],
                "error": str(e)
            }
    
    def detect_manipulated_text(self, image_path, extracted_text):
        """Detect if text in the image appears to be manipulated or artificially added"""
        try:
            # This would ideally use more sophisticated image analysis
            # For now, we'll use a simple heuristic based on OCR confidence
            ocr_results = self.ocr_tool.extract_text(image_path, include_confidence=True)
            
            # Check for very low confidence scores which might indicate manipulation
            confidences = ocr_results.get("word_confidences", [])
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
                
                # High variance in confidence scores can indicate manipulation
                manipulation_score = 0.0
                if avg_confidence < 0.7:
                    manipulation_score += 0.3
                if variance > 0.05:
                    manipulation_score += 0.3
                
                # Check for unnatural text positioning
                text_boxes = ocr_results.get("text_boxes", [])
                if text_boxes:
                    # Analyze alignment and spacing of text boxes
                    # This is a simplified approach - real implementation would be more sophisticated
                    manipulation_score += 0.1
                
                return {
                    "manipulation_detected": manipulation_score > 0.4,
                    "manipulation_score": manipulation_score,
                    "confidence": {
                        "average": avg_confidence,
                        "variance": variance
                    },
                    "analysis": "Text may be manipulated" if manipulation_score > 0.4 else "Text appears natural"
                }
            
            return {
                "manipulation_detected": False,
                "manipulation_score": 0.0,
                "analysis": "Insufficient data to determine manipulation"
            }
            
        except Exception as e:
            logger.error(f"Error detecting manipulated text: {str(e)}")
            return {
                "error": str(e),
                "manipulation_detected": False,
                "manipulation_score": 0.0,
                "analysis": f"Failed to analyze for manipulation: {str(e)}"
            }
    
    def summarize_text_content(self, text):
        """Generate a concise summary of the extracted text"""
        try:
            prompt = f"""
            Provide a concise summary of the following text extracted from an image:
            
            {text}
            
            Keep the summary under 100 words and focus on the main points and claims.
            """
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return f"Failed to summarize text: {str(e)}"