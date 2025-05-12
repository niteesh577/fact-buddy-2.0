from typing import Dict, Any
from datetime import datetime
import logging
import os
import requests
from PIL import Image
import io
import base64

from src.tools.deepfake_detector import DeepfakeDetectorTool

logger = logging.getLogger(__name__)

class DeepfakeDetectionAgent:
    """
    Agent responsible for detecting if an image has been manipulated or is a deepfake.
    """
    
    def __init__(self):
        self.deepfake_tool = DeepfakeDetectorTool()
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image_path = state.get("image_path")
            if not image_path or not os.path.exists(image_path):
                raise ValueError(f"Invalid image path: {image_path}")
            
            # Analyze the image for manipulation/deepfake
            logger.info(f"Analyzing image for manipulation: {image_path}")
            deepfake_results = self.deepfake_tool.analyze(image_path)
            
            # Add results to state
            state["deepfake_analysis"] = deepfake_results
            
            # Add message about completion
            state.setdefault("messages", []).append({
                "agent": "deepfake_detection",
                "content": f"Deepfake analysis completed with confidence {deepfake_results.get('confidence', 0):.2f}",
                "timestamp": datetime.now().isoformat()
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error in deepfake detection: {str(e)}")
            state.setdefault("messages", []).append({
                "agent": "deepfake_detection",
                "content": f"Deepfake analysis failed: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            state["deepfake_analysis"] = {"error": str(e)}
            return state