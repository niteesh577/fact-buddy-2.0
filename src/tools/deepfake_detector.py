import os
import logging
from typing import Dict, Any
import cv2
import numpy as np
from PIL import Image, ImageChops
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

class DeepfakeDetectorTool:
    """Tool for detecting potential deepfakes or manipulated images"""
    
    def __init__(self):
        # In a production environment, you might use a pre-trained model
        # For this implementation, we'll use some basic image analysis techniques
        pass
    
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """Alias for analyze_image to maintain compatibility with agent calls"""
        return self.analyze_image(image_path)
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image for signs of manipulation or deepfake
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Analyzing image for manipulation: {image_path}")
            
            # Check if file exists
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}", "is_deepfake": False}
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Failed to load image", "is_deepfake": False}
            
            # Convert to RGB for analysis
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Basic analysis techniques
            
            # 1. Error Level Analysis (ELA)
            ela_score = self._perform_ela(image_path)
            
            # 2. Noise analysis
            noise_score = self._analyze_noise(image_rgb)
            
            # 3. JPEG artifacts analysis
            jpeg_score = self._analyze_jpeg_artifacts(image_rgb)
            
            # 4. Metadata analysis
            metadata_score = self._analyze_metadata(image_path)
            
            # Calculate overall manipulation score
            # This is a simplified approach - a real implementation would use ML models
            manipulation_score = (ela_score * 0.4 + noise_score * 0.3 + 
                                 jpeg_score * 0.2 + metadata_score * 0.1)
            
            # Determine if the image is likely manipulated
            is_deepfake = manipulation_score > 0.6
            
            # Prepare analysis details
            if manipulation_score > 0.8:
                analysis = "High probability of digital manipulation detected."
            elif manipulation_score > 0.6:
                analysis = "Moderate signs of manipulation detected."
            elif manipulation_score > 0.4:
                analysis = "Some inconsistencies detected, but may be due to normal processing."
            else:
                analysis = "No significant signs of manipulation detected."
            
            result = {
                "is_deepfake": is_deepfake,
                "manipulation_score": round(manipulation_score, 2),
                "analysis": analysis,
                "details": {
                    "ela_score": round(ela_score, 2),
                    "noise_score": round(noise_score, 2),
                    "jpeg_score": round(jpeg_score, 2),
                    "metadata_score": round(metadata_score, 2)
                },
                "status": "success"
            }
            
            logger.info(f"Image analysis complete: manipulation score {manipulation_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {
                "error": str(e),
                "is_deepfake": False,
                "manipulation_score": 0.0,
                "analysis": f"Analysis failed: {str(e)}",
                "status": "failed"
            }
    
    def _perform_ela(self, image_path: str) -> float:
        """Perform Error Level Analysis"""
        try:
            # Load original image
            original = Image.open(image_path)
            
            # Save with a specific quality to a temporary file
            temp_path = f"{image_path}_temp.jpg"
            original.save(temp_path, 'JPEG', quality=90)
            
            # Open the saved image
            saved = Image.open(temp_path)
            
            # Calculate the difference
            diff = ImageChops.difference(original, saved)
            
            # Calculate the average difference
            diff_array = np.array(diff)
            ela_score = np.mean(diff_array) / 255.0
            
            # Clean up
            os.remove(temp_path)
            
            return min(ela_score * 10, 1.0)  # Scale up for better sensitivity
        except Exception as e:
            logger.warning(f"ELA analysis failed: {str(e)}")
            return 0.0
    
    def _analyze_noise(self, image: np.ndarray) -> float:
        """Analyze image noise patterns"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply noise extraction filter
            noise = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Calculate statistics
            mean = np.mean(np.abs(noise))
            std = np.std(noise)
            
            # Inconsistent noise patterns can indicate manipulation
            # This is a simplified approach
            regions = []
            h, w = gray.shape
            for i in range(0, h, h//4):
                for j in range(0, w, w//4):
                    region = noise[i:i+h//4, j:j+w//4]
                    regions.append(np.std(region))
            
            # Calculate variance between region noise levels
            if regions:
                region_variance = np.var(regions) / (np.mean(regions) + 1e-10)
                # High variance can indicate manipulation
                return min(region_variance, 1.0)
            return 0.0
        except Exception as e:
            logger.warning(f"Noise analysis failed: {str(e)}")
            return 0.0
    
    def _analyze_jpeg_artifacts(self, image: np.ndarray) -> float:
        """Analyze JPEG compression artifacts"""
        try:
            # Convert to YCrCb color space
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            
            # Extract chroma channels
            _, cr, cb = cv2.split(ycrcb)
            
            # Look for inconsistent JPEG artifacts
            cr_dct = cv2.dct(np.float32(cr))
            cb_dct = cv2.dct(np.float32(cb))
            
            # Simplified analysis of DCT coefficients
            cr_energy = np.sum(np.abs(cr_dct)) / (cr.shape[0] * cr.shape[1])
            cb_energy = np.sum(np.abs(cb_dct)) / (cb.shape[0] * cb.shape[1])
            
            # Inconsistent energy can indicate manipulation
            energy_diff = abs(cr_energy - cb_energy) / max(cr_energy, cb_energy)
            return min(energy_diff * 5, 1.0)  # Scale up for better sensitivity
        except Exception as e:
            logger.warning(f"JPEG artifact analysis failed: {str(e)}")
            return 0.0
    
    def _analyze_metadata(self, image_path: str) -> float:
        """Analyze image metadata for inconsistencies"""
        try:
            # In a real implementation, you would use a library like PIL.ExifTags
            # For this simplified version, we'll return a random score
            # This is just a placeholder
            return 0.2
        except Exception as e:
            logger.warning(f"Metadata analysis failed: {str(e)}")
            return 0.0