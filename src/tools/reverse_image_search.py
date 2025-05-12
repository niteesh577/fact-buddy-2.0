import os
import logging
import requests
import json
import base64
from typing import Dict, Any, List
import time
from PIL import Image
import io

logger = logging.getLogger(__name__)

class ReverseImageSearchTool:
    """Tool for performing reverse image searches to find similar images online"""
    
    def __init__(self):
        # In a production environment, you would use a proper API
        # For this implementation, we'll simulate results
        self.api_key = os.getenv("SERPAPI_API_KEY")
    
    def search(self, image_path: str) -> Dict[str, Any]:
        """
        Perform a reverse image search
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing search results
        """
        try:
            logger.info(f"Performing reverse image search for: {image_path}")
            
            # Check if file exists
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}", "results": []}
            
            # If we have a SerpAPI key, use it for real results
            if self.api_key:
                return self._search_with_serpapi(image_path)
            else:
                # Otherwise, simulate results
                logger.warning("No SERPAPI_API_KEY found, using simulated results")
                return self._simulate_search_results(image_path)
            
        except Exception as e:
            logger.error(f"Error in reverse image search: {str(e)}")
            return {
                "error": str(e),
                "results": [],
                "status": "failed"
            }
    
    def _search_with_serpapi(self, image_path: str) -> Dict[str, Any]:
        """Use SerpAPI for reverse image search"""
        try:
            # Prepare the image for upload
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
            
            # Encode image to base64
            encoded_image = base64.b64encode(img_data).decode('utf-8')
            
            # Prepare the API request
            url = "https://serpapi.com/search"
            
            # Use POST request instead of GET to handle large images
            # This avoids the 414 Request-URI Too Large error
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "engine": "google_reverse_image",
                "api_key": self.api_key,
                "image_data": encoded_image
            }
            
            # Make the request using POST
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            
            # Parse the response
            data = response.json()
            
            # Extract relevant information
            results = []
            
            # Process image results
            for item in data.get("image_results", []):
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "source": item.get("source", ""),
                    "thumbnail": item.get("thumbnail", ""),
                    "is_product": item.get("is_product", False),
                    "type": "image"
                })
            
            # Process related content
            for item in data.get("inline_images", [])[:5]:  # Limit to 5 related images
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "source": item.get("source", ""),
                    "thumbnail": item.get("thumbnail", ""),
                    "type": "related_image"
                })
            
            # Get pages with similar images
            for item in data.get("pages_with_matching_images", [])[:5]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "source": item.get("source", ""),
                    "type": "page_with_image"
                })
            
            return {
                "results": results,
                "original_image": image_path,
                "status": "success",
                "result_count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error with SerpAPI reverse image search: {str(e)}")
            # Fall back to simulated results
            logger.info("Falling back to simulated results")
            return self._simulate_search_results(image_path)
    
    def _simulate_search_results(self, image_path: str) -> Dict[str, Any]:
        """Simulate reverse image search results for testing"""
        # Get image details to make simulation more realistic
        try:
            img = Image.open(image_path)
            width, height = img.size
            format_name = img.format
            
            # Generate simulated results based on image properties
            results = [
                {
                    "title": f"Similar {width}x{height} {format_name} Image",
                    "link": "https://example.com/similar-image-1",
                    "source": "example.com",
                    "thumbnail": "https://example.com/thumb1.jpg",
                    "similarity_score": 0.92,
                    "type": "image"
                },
                {
                    "title": "Related Content with Similar Visual",
                    "link": "https://news-site.com/article-with-similar-image",
                    "source": "news-site.com",
                    "thumbnail": "https://news-site.com/thumb.jpg",
                    "similarity_score": 0.85,
                    "type": "page_with_image",
                    "published_date": "2023-05-15"
                },
                {
                    "title": "Original Source of Image",
                    "link": "https://original-source.org/image-gallery",
                    "source": "original-source.org",
                    "thumbnail": "https://original-source.org/thumb.jpg",
                    "similarity_score": 0.99,
                    "type": "image",
                    "published_date": "2023-01-10"
                }
            ]
            
            return {
                "results": results,
                "original_image": image_path,
                "status": "success",
                "result_count": len(results),
                "note": "These are simulated results for testing purposes"
            }
            
        except Exception as e:
            logger.error(f"Error generating simulated results: {str(e)}")
            return {
                "results": [],
                "original_image": image_path,
                "status": "failed",
                "error": str(e)
            }