from typing import Dict, Any
from datetime import datetime
import logging
import os

from src.tools.reverse_image_search import ReverseImageSearchTool
from src.tools.web_scraper import WebScraperTool

logger = logging.getLogger(__name__)

class ImageSourceAgent:
    """
    Agent responsible for verifying the source of an image through reverse image search
    and analyzing the credibility of sources where the image appears.
    """
    
    def __init__(self):
        self.reverse_search_tool = ReverseImageSearchTool()
        self.web_scraper = WebScraperTool()
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image_path = state.get("image_path")
            if not image_path or not os.path.exists(image_path):
                raise ValueError(f"Invalid image path: {image_path}")
            
            # Perform reverse image search
            logger.info(f"Performing reverse image search for: {image_path}")
            search_results = self.reverse_search_tool.search(image_path)
            
            # Analyze top sources
            source_analysis = []
            for result in search_results.get("results", [])[:3]:  # Analyze top 3 results
                url = result.get("url")
                if url:
                    try:
                        logger.info(f"Analyzing source: {url}")
                        scraped_content = self.web_scraper.scrape(url)
                        
                        # Add source analysis
                        source_analysis.append({
                            "url": url,
                            "title": result.get("title", ""),
                            "date_found": result.get("date", "Unknown"),
                            "content_snippet": scraped_content.get("content", "")[:500] + "..." if scraped_content.get("content") else "",
                            "trust_score": self._calculate_trust_score(result, scraped_content)
                        })
                    except Exception as e:
                        logger.error(f"Error analyzing source {url}: {str(e)}")
                        source_analysis.append({
                            "url": url,
                            "title": result.get("title", ""),
                            "error": str(e)
                        })
            
            # Add results to state
            state["source_verification"] = {
                "search_results": search_results,
                "source_analysis": source_analysis,
                "earliest_appearance": self._find_earliest_appearance(source_analysis),
                "most_credible_source": self._find_most_credible_source(source_analysis)
            }
            
            # Add message about completion
            state.setdefault("messages", []).append({
                "agent": "image_source",
                "content": f"Source verification completed with {len(source_analysis)} sources analyzed",
                "timestamp": datetime.now().isoformat()
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error in image source verification: {str(e)}")
            state.setdefault("messages", []).append({
                "agent": "image_source",
                "content": f"Source verification failed: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            state["source_verification"] = {"error": str(e)}
            return state
    
    def _calculate_trust_score(self, search_result, scraped_content):
        """Calculate a trust score for a source based on various factors"""
        # This is a simplified implementation - you would want to expand this
        score = 0.5  # Default medium trust
        
        # Domain reputation factor
        domain = search_result.get("url", "").split("/")[2] if len(search_result.get("url", "").split("/")) > 2 else ""
        reputable_domains = ["reuters.com", "apnews.com", "bbc.com", "nytimes.com", "washingtonpost.com"]
        if any(rep_domain in domain for rep_domain in reputable_domains):
            score += 0.3
        
        # Content length factor
        content_length = len(scraped_content.get("content", ""))
        if content_length > 2000:
            score += 0.1
        
        # Cap at 1.0
        return min(score, 1.0)
    
    def _find_earliest_appearance(self, source_analysis):
        """Find the earliest appearance of the image based on source dates"""
        earliest = None
        earliest_date = None
        
        for source in source_analysis:
            date = source.get("date_found")
            if date and (earliest_date is None or date < earliest_date):
                earliest_date = date
                earliest = source
        
        return earliest
    
    def _find_most_credible_source(self, source_analysis):
        """Find the most credible source based on trust scores"""
        most_credible = None
        highest_score = -1
        
        for source in source_analysis:
            score = source.get("trust_score", 0)
            if score > highest_score:
                highest_score = score
                most_credible = source
        
        return most_credible