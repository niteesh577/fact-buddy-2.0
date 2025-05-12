from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from typing import Dict, Any, Optional, List, TypedDict
from datetime import datetime
import logging
import requests
import json
import re
from urllib.parse import urlparse, parse_qs
import logging # Added logging import

from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from src.tools.video_transcript_extractor import VideoTranscriptExtractor

logging.basicConfig(level=logging.INFO)


class VideoResearchState(TypedDict):
    """State for the video research agent."""
    claim: str
    video_url: str
    video_transcript: Dict[str, Any]
    video_research_results: Dict[str, Any]
    messages: List[Dict[str, Any]]


class VideoResearchAgent:
    def __init__(self):
        # Initialize the language model engine with fixed temperature
        self.llm = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768"
        )
        
        # Initialize the video transcript extractor
        self.transcript_extractor = VideoTranscriptExtractor()
        
        # Create tools for video transcript extraction
        self.tools = [
            Tool(
                name="extract_video_transcript",
                func=self.extract_transcript,
                description="Extract transcript from a video URL"
            ),
            Tool(
                name="extract_video_info",
                func=self.extract_video_info,
                description="Extract metadata information from a video URL"
            )
        ]

        # Create a prompt template for the agent
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a video research agent tasked with extracting transcripts and metadata from videos. "
                      "Use the available tools to get the transcript and video information."), # Updated system prompt
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])

        # Create a LangGraph-compatible agent
        # Note: The agent execution logic using create_react_agent might need adjustment 
        # depending on how the graph nodes call the tools vs. how the agent itself is invoked.
        # For now, assuming the graph nodes directly call the methods.
        # self.agent = create_react_agent(
        #     self.llm,
        #     self.tools
        # )
        
        # Create the research graph
        self.graph = self._create_research_graph()

    def _create_research_graph(self):
        """Create a LangGraph workflow for the video research process."""
        workflow = StateGraph(VideoResearchState)
        
        # Add the research node that processes the video URL
        workflow.add_node("video_research", self._process_video_research)
        
        # Set the entry point and connect to END
        workflow.add_edge("video_research", END)
        workflow.set_entry_point("video_research")
        
        return workflow.compile()

    # This method now directly uses the extractor tool
    def extract_transcript(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract transcript from a video URL using the VideoTranscriptExtractor tool.
        
        Args:
            input_data: Dictionary containing the video_url key
            
        Returns:
            Dictionary containing transcript data or an error message
        """
        try:
            # Extract video URL from input data
            video_url = input_data.get("video_url")
            if not video_url:
                logging.error("No video URL provided for transcript extraction")
                return {
                    "status": "failed",
                    "error": "No video URL provided",
                    "transcript": None,
                    "segments": []
                }
                
            logging.info(f"Using VideoTranscriptExtractor for: {video_url}")
            transcript_data = self.transcript_extractor.extract_transcript(video_url)
            
            # Add additional logging to debug transcript extraction
            if transcript_data.get("status") == "failed":
                logging.warning(f"Transcript extraction failed: {transcript_data.get('error')}")
            else:
                logging.info(f"Successfully extracted transcript with {len(transcript_data.get('segments', []))} segments")
                
            return transcript_data
        except Exception as e:
            logging.error(f"Error calling VideoTranscriptExtractor: {str(e)}")
            return {
                "status": "failed",
                "error": f"Error during transcript extraction: {str(e)}",
                "transcript": None,
                "segments": []
            }

    # Method for video info extraction using the transcript extractor
    def extract_video_info(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata information from a video URL.
        
        Args:
            input_data: Dictionary containing the video_url key
            
        Returns:
            Dictionary containing video metadata or an error message
        """
        try:
            # Extract video URL from input data
            video_url = input_data.get("video_url")
            if not video_url:
                logging.error("No video URL provided for video info extraction")
                return {
                    "status": "failed",
                    "error": "No video URL provided",
                    "info": {}
                }
                
            logging.info(f"Attempting to extract video info for: {video_url}")
            # Use the transcript extractor's method to get video info
            video_info = self.transcript_extractor.extract_video_info(video_url)
            
            # Add additional logging
            if video_info.get("status") == "failed":
                logging.warning(f"Video info extraction failed: {video_info.get('error')}")
            else:
                logging.info(f"Successfully extracted video info")
                
            return video_info
        except Exception as e:
            logging.error(f"Error extracting video info: {str(e)}")
            return {
                "status": "failed",
                "error": f"Error during video info extraction: {str(e)}",
                "info": {}
            }

        if video_platform == "youtube":
            # Placeholder: Requires YouTube Data API v3 key and implementation
            logging.warning("YouTube video info extraction requires API key and implementation.")
            # Example API call structure (requires google-api-python-client):
            # youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
            # request = youtube.videos().list(part="snippet,contentDetails,statistics", id=video_id)
            # response = request.execute()
            # metadata = response['items'][0] if response['items'] else {}
            return {
                "status": "unimplemented",
                "video_id": video_id,
                "platform": "youtube",
                "metadata": {"title": f"Simulated Title for {video_id}", "description": "Simulated description."}
            }
        elif video_platform == "vimeo":
            # Placeholder: Requires Vimeo API token and implementation
            logging.warning("Vimeo video info extraction requires API token and implementation.")
            # Example API call structure (requires requests):
            # headers = {'Authorization': f'bearer {VIMEO_ACCESS_TOKEN}'}
            # response = requests.get(f'https://api.vimeo.com/videos/{video_id}', headers=headers)
            # metadata = response.json() if response.ok else {}
            return {
                "status": "unimplemented",
                "video_id": video_id,
                "platform": "vimeo",
                "metadata": {"title": f"Simulated Title for {video_id}", "description": "Simulated description."}
            }
        else:
            return {
                "status": "failed",
                "error": f"Unsupported platform for metadata extraction: {video_platform}",
                "metadata": None
            }

    # Removed redundant _parse_video_url, _extract_youtube_transcript, _extract_vimeo_transcript
    # as this logic is now handled by VideoTranscriptExtractor

    def _process_video_research(self, state: VideoResearchState) -> VideoResearchState:
        """Node to perform video research: extract transcript and metadata."""
        video_url = state.get("video_url")
        if not video_url:
            logging.error("Video URL missing in state for research.")
            state["video_transcript"] = {"status": "failed", "error": "Missing video_url", "transcript": None, "segments": []}
            state["video_research_results"] = {"status": "failed", "error": "Missing video_url", "info": {}}
            state.setdefault("messages", []).append({
                "agent": "video_research",
                "content": "Error: Missing video URL",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            return state

        logging.info(f"Processing video research for: {video_url}")

        # Extract Transcript
        try:
            # Prepare input for transcript extraction
            transcript_input = {"video_url": video_url}
            transcript_data = self.extract_transcript(transcript_input)
            
            # Store transcript data in state
            state["video_transcript"] = transcript_data
            
            # Log transcript extraction result
            if transcript_data.get("status") == "success":
                state.setdefault("messages", []).append({
                    "agent": "video_research",
                    "content": f"Successfully extracted transcript with {len(transcript_data.get('segments', []))} segments",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                state.setdefault("messages", []).append({
                    "agent": "video_research",
                    "content": f"Transcript extraction failed: {transcript_data.get('error')}",
                    "error": True,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Extract Video Info
            video_info_input = {"video_url": video_url}
            video_info = self.extract_video_info(video_info_input)
            
            # Store video info in research results
            state["video_research_results"] = video_info
            
            # Log video info extraction result
            if video_info.get("status") == "success":
                state.setdefault("messages", []).append({
                    "agent": "video_research",
                    "content": "Successfully extracted video information",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                state.setdefault("messages", []).append({
                    "agent": "video_research",
                    "content": f"Video info extraction failed: {video_info.get('error')}",
                    "error": True,
                    "timestamp": datetime.now().isoformat()
                })
                
            # Mark the current agent as completed
            state["current_agent"] = "video_research_completed"
            
        except Exception as e:
            logging.error(f"Error in video research processing: {str(e)}")
            state.setdefault("messages", []).append({
                "agent": "video_research",
                "content": f"Error processing video research: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            state["video_transcript"] = {"status": "failed", "error": str(e), "transcript": None, "segments": []}
            state["video_research_results"] = {"status": "failed", "error": str(e), "info": {}}
        
        return state
        transcript_result = self.extract_transcript(video_url)
        state["video_transcript"] = transcript_result
        if transcript_result.get("status") == "failed":
            logging.warning(f"Transcript extraction failed: {transcript_result.get('error')}")
            # Decide if we should proceed without transcript or stop

        # Extract Video Info (Metadata)
        info_result = self.extract_video_info(video_url)
        # Store metadata under video_research_results for clarity
        state["video_research_results"] = {
            "metadata_status": info_result.get("status"),
            "metadata_error": info_result.get("error"),
            "metadata": info_result.get("metadata")
        }
        if info_result.get("status") != "success" and info_result.get("status") != "unimplemented": # Log warnings for unimplemented, errors for failed
             logging.warning(f"Video info extraction status: {info_result.get('status')} - {info_result.get('error')}")

        state.setdefault("messages", []).append({
            "agent": "video_research",
            "content": f"Video research processing complete for {video_url}. Transcript status: {transcript_result.get('status')}, Info status: {info_result.get('status')}",
            "timestamp": datetime.now().isoformat()
        })

        return state

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the video research agent on the provided state."""
        return self.graph.invoke(state)