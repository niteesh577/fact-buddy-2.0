from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, List, Any, Optional
from src.agents.video_research_agent import VideoResearchAgent
from src.agents.video_verification_agent import VideoVerificationAgent
from src.agents.video_summary_agent import VideoSummaryAgent
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)


class VideoAgentState(TypedDict):
    claim: str
    video_url: str
    video_transcript: Dict[str, Any]
    video_research_results: Dict[str, Any]
    verification_results: Dict[str, Any]
    validation_results: Dict[str, Any]
    final_summary: Dict[str, Any]
    messages: List[Dict[str, Any]]
    current_agent: str  # Track the current agent being executed


class VideoSupervisorAgent:
    def __init__(self):
        self.video_research_agent = VideoResearchAgent()
        self.video_verification_agent = VideoVerificationAgent()
        self.video_summary_agent = VideoSummaryAgent()
        self.workflow = self._create_workflow()
        logging.info("VideoSupervisorAgent initialized with all agent components")

    def _run_video_research_agent(self, state: VideoAgentState) -> VideoAgentState:
        """Run the video research agent and update the workflow state."""
        logging.info("Running video research agent")
        try:
            # Run the video research agent
            updated_state = self.video_research_agent.run(state)
            # Mark this step as completed
            updated_state["current_agent"] = "video_research_completed"
            logging.info("Video research agent completed successfully")
            return updated_state
        except Exception as e:
            logging.error(f"Error in video research agent: {str(e)}")
            state["messages"].append({
                "agent": "video_research",
                "content": f"Error: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            state["current_agent"] = "error"
            return state
            
    def _run_video_verification_agent(self, state: VideoAgentState) -> VideoAgentState:
        """Run the video verification agent and update the workflow state."""
        logging.info("Running video verification agent")
        try:
            # Run the video verification agent
            updated_state = self.video_verification_agent.run(state)
            # Mark this step as completed
            updated_state["current_agent"] = "video_verification_completed"
            logging.info("Video verification agent completed successfully")
            return updated_state
        except Exception as e:
            logging.error(f"Error in video verification agent: {str(e)}")
            state["messages"].append({
                "agent": "video_verification",
                "content": f"Error: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            state["current_agent"] = "error"
            return state

    def _run_video_summary_agent(self, state: VideoAgentState) -> VideoAgentState:
        """Run the video summary agent and update the workflow state."""
        logging.info("Running video summary agent")
        try:
            # Run the video summary agent
            updated_state = self.video_summary_agent.run(state)
            # Mark this step as completed
            updated_state["current_agent"] = "completed"
            logging.info("Video summary agent completed successfully")
            return updated_state
        except Exception as e:
            logging.error(f"Error in video summary agent: {str(e)}")
            state["messages"].append({
                "agent": "video_summary",
                "content": f"Error: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            state["current_agent"] = "error"
            return state

    def _create_workflow(self):
        """Create the LangGraph workflow that orchestrates the video fact-checking process."""
        workflow = StateGraph(VideoAgentState)

        # Define the nodes for each agent
        workflow.add_node("video_research", self._run_video_research_agent)
        workflow.add_node("video_verification", self._run_video_verification_agent)
        workflow.add_node("video_summary", self._run_video_summary_agent)

        # Define the routing logic based on the current_agent state
        def route_next_step(state: VideoAgentState) -> str:
            """Determine the next step based on the current_agent value."""
            current = state.get("current_agent", "")
            
            if current == "":
                # Initial state, start with video research
                return "video_research"
            elif current == "video_research_completed":
                # After video research is done, go to video verification
                return "video_verification"
            elif current == "video_verification_completed":
                # After video verification is done, go to video summary
                return "video_summary"
            elif current == "completed" or current == "error":
                # Workflow is complete or encountered an error
                return END
            else:
                # Unknown state, end the workflow
                logging.warning(f"Unknown current_agent value: {current}, ending workflow")
                return END

        # Add conditional edges for each node
        workflow.add_conditional_edges(
            "video_research",
            route_next_step
        )
        workflow.add_conditional_edges(
            "video_verification",
            route_next_step
        )
        workflow.add_conditional_edges(
            "video_summary",
            route_next_step
        )

        # Set the entry point
        workflow.set_entry_point("video_research")
        logging.info("Video workflow created with all nodes and conditional edges")
        return workflow.compile()

    def run_video_fact_check(self, claim: str, video_url: str) -> dict:
        """Run the complete video fact-checking workflow on a claim and video.
        
        Args:
            claim: The claim to fact-check
            video_url: The URL of the video to analyze
            
        Returns:
            The complete state after workflow execution
        """
        logging.info(f"Starting video fact-check for claim: {claim} with video: {video_url}")
        
        # Initialize the state with proper structure
        initial_state = {
            "claim": claim,
            "video_url": video_url,
            "video_transcript": {},
            "video_research_results": {},
            "verification_results": {},
            "validation_results": {},
            "final_summary": {},
            "messages": [
                {
                    "agent": "video_supervisor",
                    "content": f"Starting video fact-check for: {claim}",
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "current_agent": ""
        }
        
        try:
            # Execute the workflow
            final_state = self.workflow.invoke(initial_state)
            logging.info("Video fact-check workflow completed successfully")
            return final_state
        except Exception as e:
            logging.error(f"Error in video fact-check workflow: {str(e)}")
            initial_state["messages"].append({
                "agent": "video_supervisor",
                "content": f"Workflow error: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            initial_state["current_agent"] = "error"
            return initial_state