from typing import Dict, Any
from datetime import datetime
import logging
import os
from langchain_groq import ChatGroq
from PIL import Image
import io

logger = logging.getLogger(__name__)

class ImageSummaryAgent:
    """
    Agent responsible for generating a descriptive summary of image content.
    """
    
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768"
        )
    
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            image_path = state.get("image_path")
            if not image_path or not os.path.exists(image_path):
                raise ValueError(f"Invalid image path: {image_path}")
            
            # Generate image description
            logger.info(f"Generating summary for image: {image_path}")
            
            # Get image metadata
            img_metadata = self._get_image_metadata(image_path)
            
            # For a real implementation, you would use a vision model here
            # Since we're simulating, we'll generate a generic description
            image_description = self._generate_image_description(image_path, img_metadata)
            
            # Add results to state
            state["image_summary"] = {
                "description": image_description.get("description", ""),
                "subjects": image_description.get("subjects", []),
                "scene_type": image_description.get("scene_type", ""),
                "metadata": img_metadata
            }
            
            # Generate the final summary that the API endpoint expects
            deepfake_analysis = state.get("deepfake_analysis", {})
            source_verification = state.get("source_verification", {})
            text_content = state.get("text_content", {})
            
            # Determine verdict based on analysis results
            verdict = "Unknown"
            confidence_level = 0.5  # Default medium confidence
            
            # Check deepfake analysis
            if deepfake_analysis.get("is_deepfake", False):
                verdict = "False"
                confidence_level = deepfake_analysis.get("manipulation_score", 0.7)
            
            # Check source verification
            most_credible_source = source_verification.get("most_credible_source", {})
            if most_credible_source and most_credible_source.get("trust_score", 0) > 0.7:
                if verdict == "Unknown":
                    verdict = "True"
                    confidence_level = most_credible_source.get("trust_score", 0.6)
            
            # Generate evidence summary
            evidence_summary = image_description.get("description", "")
            if text_content.get("has_text", False):
                evidence_summary += " " + text_content.get("analysis", {}).get("summary", "")
            
            # Generate key findings
            key_findings = []
            
            # Add deepfake finding if relevant
            if deepfake_analysis:
                key_findings.append({
                    "finding": deepfake_analysis.get("analysis", "Image analysis completed"),
                    "category": "Manipulation Detection"
                })
            
            # Add source verification finding if available
            if source_verification.get("earliest_appearance"):
                key_findings.append({
                    "finding": f"Image first appeared online: {source_verification.get('earliest_appearance', {}).get('date_found', 'Unknown date')}",
                    "category": "Source Verification"
                })
            
            # Add text content finding if available
            if text_content.get("has_text", False):
                key_findings.append({
                    "finding": f"Text detected in image: {text_content.get('extracted_text', '')[:100]}...",
                    "category": "Text Analysis"
                })
            
            # Add image description finding
            key_findings.append({
                "finding": image_description.get("description", "Image analyzed"),
                "category": "Visual Content"
            })
            
            # Generate citations
            citations = []
            
            # Add source verification citations
            for source in source_verification.get("source_analysis", []):
                if source.get("url"):
                    citations.append({
                        "source": source.get("url"),
                        "trust_score": source.get("trust_score", "0.5")
                    })
            
            # Create the final summary
            state["final_summary"] = {
                "verdict": verdict,
                "confidence_level": confidence_level,
                "evidence_summary": evidence_summary,
                "key_findings": key_findings,
                "citations": citations,
                "deepfake_analysis": deepfake_analysis,
                "source_verification": source_verification,
                "text_content": text_content
            }
            
            # Add message about completion
            state.setdefault("messages", []).append({
                "agent": "image_summary",
                "content": "Image summary generated successfully",
                "timestamp": datetime.now().isoformat()
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error in image summary generation: {str(e)}")
            state.setdefault("messages", []).append({
                "agent": "image_summary",
                "content": f"Image summary failed: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            state["image_summary"] = {"error": str(e)}
            return state
    
    def _get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract metadata from the image file"""
        try:
            img = Image.open(image_path)
            metadata = {
                "format": img.format,
                "mode": img.mode,
                "size": {
                    "width": img.width,
                    "height": img.height
                }
            }
            
            # Try to extract EXIF data if available
            exif_data = {}
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                if exif:
                    for tag, value in exif.items():
                        tag_name = TAGS.get(tag, tag)
                        exif_data[tag_name] = value
            
            if exif_data:
                metadata["exif"] = exif_data
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Error extracting image metadata: {str(e)}")
            return {
                "error": str(e)
            }
    
    def _generate_image_description(self, image_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a description of the image content
        
        In a real implementation, this would use a vision model.
        For this simulation, we'll return a generic description.
        """
        try:
            # Get basic image properties
            width = metadata.get("size", {}).get("width", 0)
            height = metadata.get("size", {}).get("height", 0)
            img_format = metadata.get("format", "Unknown")
            
            # For a real implementation, you would use a vision model here
            # Since we're simulating, we'll generate a generic description
            
            # Determine if it's a portrait, landscape, or square image
            orientation = "square"
            if width > height * 1.2:
                orientation = "landscape"
            elif height > width * 1.2:
                orientation = "portrait"
            
            # Open and analyze the image directly from the file path
            img = Image.open(image_path)
            
            # Extract dominant colors (simplified approach)
            colors = self._extract_dominant_colors(img)
            
            # Generate a more detailed description based on image properties
            description = f"This is a {width}x{height} {orientation} image in {img_format} format."
            if colors:
                description += f" The dominant colors are {', '.join(colors[:3])}."
            
            # In a real implementation, you would add details about:
            # - People, objects, and scenes in the image
            # - Text visible in the image
            # - Colors and composition
            # - Context and setting
            
            return {
                "description": description,
                "subjects": ["generic image"],
                "scene_type": orientation,
                "colors": colors[:5] if colors else ["unknown"],
                "text_present": False
            }
            
        except Exception as e:
            logger.error(f"Error generating image description: {str(e)}")
            return {
                "description": "Failed to generate image description",
                "error": str(e)
            }
    
    def _extract_dominant_colors(self, img, num_colors=5):
        """Extract dominant colors from an image"""
        try:
            # Resize image to speed up processing
            img = img.copy()
            img.thumbnail((100, 100))
            
            # Convert to RGB if not already
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Get colors from image
            pixels = list(img.getdata())
            
            # Simple color counting (a more sophisticated approach would use clustering)
            color_counts = {}
            for pixel in pixels:
                # Simplify the color space to reduce variations
                simplified = (
                    round(pixel[0] / 50) * 50,
                    round(pixel[1] / 50) * 50,
                    round(pixel[2] / 50) * 50
                )
                if simplified in color_counts:
                    color_counts[simplified] += 1
                else:
                    color_counts[simplified] = 1
            
            # Get the most common colors
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Convert RGB values to color names (simplified)
            color_names = []
            for color, _ in sorted_colors[:num_colors]:
                r, g, b = color
                # Very simple color naming logic
                if r > 200 and g > 200 and b > 200:
                    color_names.append("white")
                elif r < 50 and g < 50 and b < 50:
                    color_names.append("black")
                elif r > 200 and g < 100 and b < 100:
                    color_names.append("red")
                elif r < 100 and g > 200 and b < 100:
                    color_names.append("green")
                elif r < 100 and g < 100 and b > 200:
                    color_names.append("blue")
                elif r > 200 and g > 200 and b < 100:
                    color_names.append("yellow")
                elif r > 200 and g < 100 and b > 200:
                    color_names.append("magenta")
                elif r < 100 and g > 200 and b > 200:
                    color_names.append("cyan")
                elif r > 200 and g > 100 and b < 100:
                    color_names.append("orange")
                else:
                    color_names.append("gray")
            
            return color_names
            
        except Exception as e:
            logger.warning(f"Error extracting colors: {str(e)}")
            return []