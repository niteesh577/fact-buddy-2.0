from typing import Dict, Any, Optional
from datetime import datetime
import logging
import os
import json

from src.agents.deepfake_detection_agent import DeepfakeDetectionAgent
from src.agents.image_source_agent import ImageSourceAgent
from src.agents.image_text_analysis_agent import ImageTextAnalysisAgent
from src.agents.image_summary_agent import ImageSummaryAgent

logger = logging.getLogger(__name__)

class ImageSupervisorAgent:
    """
    Supervisor agent that orchestrates the image fact-checking workflow.
    It coordinates multiple specialized agents to analyze different aspects of an image.
    """
    
    def __init__(self):
        self.deepfake_agent = DeepfakeDetectionAgent()
        self.source_agent = ImageSourceAgent()
        self.text_analysis_agent = ImageTextAnalysisAgent()
        self.summary_agent = ImageSummaryAgent()
    
    def run_image_fact_check(self, image_path: str, context: str = "") -> Dict[str, Any]:
        """
        Run the complete image fact-checking workflow.
        
        Args:
            image_path: Path to the image file
            context: Optional text context about the image
            
        Returns:
            Dict containing the results of the fact-checking process
        """
        try:
            # Initialize state
            state = {
                "image_path": image_path,
                "context": context,
                "messages": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Step 1: Deepfake detection
            logger.info("Starting deepfake detection analysis")
            state = self.deepfake_agent.run(state)
            
            # Step 2: Source verification
            logger.info("Starting image source verification")
            state = self.source_agent.run(state)
            
            # Step 3: Text content analysis (if any text is in the image)
            logger.info("Starting image text content analysis")
            state = self.text_analysis_agent.run(state)
            
            # Step 4: Generate final summary and verdict
            logger.info("Generating final summary and verdict")
            state = self.summary_agent.run(state)
            
            # Add completion message
            state["messages"].append({
                "agent": "image_supervisor",
                "content": "Image fact-checking workflow completed",
                "timestamp": datetime.now().isoformat()
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error in image fact-checking workflow: {str(e)}")
            return {
                "error": str(e),
                "messages": [{
                    "agent": "image_supervisor",
                    "content": f"Image fact-checking failed: {str(e)}",
                    "error": True,
                    "timestamp": datetime.now().isoformat()
                }]
            }