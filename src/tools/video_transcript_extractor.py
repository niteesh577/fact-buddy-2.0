import requests
import re
import json
from urllib.parse import urlparse, parse_qs
import logging
from typing import Dict, Any, Optional

# Attempt to import youtube_transcript_api
try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    logging.warning("youtube-transcript-api not installed. YouTube transcript extraction will be limited. Install with: pip install youtube-transcript-api")

logging.basicConfig(level=logging.INFO)

class VideoTranscriptExtractor:
    """Tool for extracting transcripts from video platforms like YouTube and Vimeo."""
    
    def __init__(self):
        self.api_keys = {
            # In a production environment, these would be loaded from environment variables or a config file
            "youtube": None,  # YouTube Data API key (if needed for other metadata)
            "vimeo": None,    # Vimeo API access token (required for Vimeo transcripts)
        }
    
    def extract_transcript(self, video_url: str) -> Dict[str, Any]:
        """Extract transcript from a video URL.
        
        Args:
            video_url: URL of the video to extract transcript from
            
        Returns:
            Dictionary containing transcript data or an error message.
        """
        try:
            logging.info(f"Attempting to extract transcript from video: {video_url}")
            
            # Determine video platform and ID
            video_platform, video_id = self._parse_video_url(video_url)
            
            if not video_id:
                logging.warning(f"Could not extract video ID from URL: {video_url}")
                return {
                    "status": "failed",
                    "error": f"Could not extract video ID from URL: {video_url}",
                    "transcript": None,
                    "segments": []
                }
            
            if video_platform == "youtube":
                return self._extract_youtube_transcript(video_id)
            elif video_platform == "vimeo":
                return self._extract_vimeo_transcript(video_id)
            else:
                logging.warning(f"Unsupported video platform: {video_platform}")
                return {
                    "status": "failed",
                    "error": f"Unsupported video platform: {video_platform}",
                    "transcript": None,
                    "segments": []
                }
                
        except Exception as e:
            logging.error(f"Unexpected error during transcript extraction for {video_url}: {str(e)}")
            return {
                "status": "failed",
                "error": f"Unexpected error: {str(e)}",
                "transcript": None,
                "segments": []
            }
    
    def _parse_video_url(self, url: str) -> tuple:
        """Parse video URL to determine platform and video ID."""
        parsed_url = urlparse(url)
        
        # YouTube URL patterns
        if 'youtube.com' in parsed_url.netloc:
            if '/watch' in parsed_url.path:
                video_id = parse_qs(parsed_url.query).get('v', [''])[0]
                return "youtube", video_id
            elif '/embed/' in parsed_url.path or '/v/' in parsed_url.path:
                video_id = parsed_url.path.split('/')[-1]
                return "youtube", video_id
        elif 'youtu.be' in parsed_url.netloc:
            video_id = parsed_url.path[1:]
            return "youtube", video_id
            
        # Vimeo URL patterns
        elif 'vimeo.com' in parsed_url.netloc:
            video_id = parsed_url.path.lstrip('/')
            # Handle potential player URLs
            if '/video/' in parsed_url.path:
                video_id = parsed_url.path.split('/')[-1]
            return "vimeo", video_id
            
        # Default case - unknown platform
        return "unknown", ""
    
    def _extract_youtube_transcript(self, video_id: str) -> Dict[str, Any]:
        """Extract transcript from YouTube video using youtube-transcript-api.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary containing transcript data or an error message.
        """
        if not YOUTUBE_API_AVAILABLE:
            logging.error("youtube-transcript-api is not installed. Cannot extract YouTube transcript.")
            return {
                "status": "failed",
                "error": "youtube-transcript-api not installed.",
                "video_id": video_id,
                "platform": "youtube",
                "transcript": None,
                "segments": []
            }
            
        try:
            logging.info(f"Extracting transcript for YouTube video: {video_id}")
            
            # Get available transcript list
            # First try to get the transcript directly without listing languages
            try:
                # Direct approach to get transcript in English or any available language
                fetched_transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                language = 'en'
                logging.info(f"Successfully retrieved English transcript for video {video_id}")
            except (TranscriptsDisabled, NoTranscriptFound) as e:
                # Specific handling for when transcripts are disabled or not found
                logging.warning(f"English transcript not available for {video_id}: {str(e)}")
                return {
                    "status": "failed",
                    "error": f"Transcript not available: {str(e)}",
                    "video_id": video_id,
                    "platform": "youtube",
                    "transcript": None,
                    "segments": []
                }
            except Exception as e:
                try:
                    # Try without language specification
                    logging.info(f"Attempting to retrieve transcript in any language for video {video_id}")
                    fetched_transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    language = 'unknown'
                    logging.warning(f"English transcript not found for {video_id}. Using default language")
                except (TranscriptsDisabled, NoTranscriptFound) as inner_e:
                    # Specific handling for when transcripts are disabled or not found
                    logging.error(f"No transcript available for YouTube video {video_id}: {str(inner_e)}")
                    return {
                        "status": "failed",
                        "error": f"No transcript available: {str(inner_e)}",
                        "video_id": video_id,
                        "platform": "youtube",
                        "transcript": None,
                        "segments": []
                    }
                except Exception as inner_e:
                    logging.error(f"Error fetching transcript for YouTube video {video_id}: {str(inner_e)}")
                    return {
                        "status": "failed",
                        "error": f"Error fetching transcript: {str(inner_e)}",
                        "video_id": video_id,
                        "platform": "youtube",
                        "transcript": None,
                        "segments": []
                    }
            
            # Process transcript segments
            if not fetched_transcript:
                logging.error(f"Empty transcript returned for YouTube video {video_id}")
                return {
                    "status": "failed",
                    "error": "Empty transcript returned",
                    "video_id": video_id,
                    "platform": "youtube",
                    "transcript": None,
                    "segments": []
                }
                
            # Join all text segments to create full transcript
            full_transcript = " ".join([item["text"] for item in fetched_transcript])
            
            # Format segments with start time, end time, and text
            segments = []
            for item in fetched_transcript:
                segment = {
                    "start": item["start"],
                    # Calculate end time if duration is present, otherwise estimate or omit
                    "end": item["start"] + item.get("duration", 1.0), # Default duration if missing
                    "text": item["text"]
                }
                segments.append(segment)
            
            return {
                "status": "success",
                "video_id": video_id,
                "platform": "youtube",
                "transcript": full_transcript,
                "segments": segments
            }
            
        except TranscriptsDisabled as e:
            logging.warning(f"Transcripts are disabled for video {video_id}: {str(e)}")
            return {
                "status": "failed",
                "error": f"Transcripts disabled: {str(e)}",
                "video_id": video_id,
                "platform": "youtube",
                "transcript": None,
                "segments": []
            }
        except Exception as e:
            logging.error(f"Error fetching transcript for YouTube video {video_id}: {str(e)}")
            return {
                "status": "failed",
                "error": f"Error fetching transcript: {str(e)}",
                "video_id": video_id,
                "platform": "youtube",
                "transcript": None,
                "segments": []
            }
    
    # Removed _generate_simulated_transcript as we now return errors or actual data.
    
    def _extract_vimeo_transcript(self, video_id: str) -> Dict[str, Any]:
        """Extract transcript from Vimeo video.
        
        In a production environment, this would use the Vimeo API
        to access captions/subtitles if available.
        
        Args:
            video_id: Vimeo video ID
            
        Returns:
            Dictionary containing transcript data or an error message.
        """
        # Placeholder for Vimeo transcript extraction
        # Requires Vimeo API access token and implementation using their API
        logging.warning(f"Vimeo transcript extraction is not implemented. Video ID: {video_id}")
        
        # Check if API key is available (though not used in this placeholder)
        if not self.api_keys.get("vimeo"):
            logging.warning("Vimeo API access token not configured.")
            # You might return an error here or proceed with a placeholder

        # Return an 'unimplemented' status
        return {
            "status": "failed",
            "error": "Vimeo transcript extraction not implemented.",
            "video_id": video_id,
            "platform": "vimeo",
            "transcript": None,
            "segments": []
        }
    
    def extract_video_info(self, video_url: str) -> Dict[str, Any]:
        """Extract metadata information from a video URL.
        
        Args:
            video_url: URL of the video to extract information from
            
        Returns:
            Dictionary containing video metadata
        """
        try:
            logging.info(f"Extracting video info from: {video_url}")
            
            # Determine video platform and ID
            video_platform, video_id = self._parse_video_url(video_url)
            
            if video_platform == "youtube":
                return self._extract_youtube_info(video_id)
            elif video_platform == "vimeo":
                return self._extract_vimeo_info(video_id)
            else:
                return {
                    "status": "failed",
                    "error": f"Unsupported video platform: {video_platform}",
                    "info": {}
                }
                
        except Exception as e:
            logging.error(f"Error extracting video info: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "info": {}
            }
    
    def _extract_youtube_info(self, video_id: str) -> Dict[str, Any]:
        """Extract metadata from YouTube video.
        
        In a production environment, this would use the YouTube Data API.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary containing video metadata
        """
        try:
            # In a production environment, you would use the YouTube Data API
            # For this implementation, we'll simulate the API response
            logging.info(f"Extracting info for YouTube video: {video_id}")
            
            # Simulate API response
            return {
                "status": "success",
                "video_id": video_id,
                "platform": "youtube",
                "info": {
                    "title": f"Simulated YouTube Video {video_id}",
                    "description": "This is a simulated video description.",
                    "channel": "Simulated Channel",
                    "published_at": "2023-01-01T00:00:00Z",
                    "view_count": 10000,
                    "like_count": 1000,
                    "comment_count": 500,
                    "duration": "PT10M30S"  # ISO 8601 duration format
                }
            }
            
        except Exception as e:
            logging.error(f"Error extracting YouTube info: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "info": {}
            }
    
    def _extract_vimeo_info(self, video_id: str) -> Dict[str, Any]:
        """Extract metadata from Vimeo video.
        
        In a production environment, this would use the Vimeo API.
        
        Args:
            video_id: Vimeo video ID
            
        Returns:
            Dictionary containing video metadata
        """
        try:
            # In a production environment, you would use the Vimeo API
            # For this implementation, we'll simulate the API response
            logging.info(f"Extracting info for Vimeo video: {video_id}")
            
            # Simulate API response
            return {
                "status": "success",
                "video_id": video_id,
                "platform": "vimeo",
                "info": {
                    "title": f"Simulated Vimeo Video {video_id}",
                    "description": "This is a simulated video description.",
                    "user": "Simulated User",
                    "created_time": "2023-01-01T00:00:00Z",
                    "play_count": 5000,
                    "like_count": 500,
                    "comment_count": 200,
                    "duration": 630  # Duration in seconds
                }
            }
            
        except Exception as e:
            logging.error(f"Error extracting Vimeo info: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "info": {}
            }