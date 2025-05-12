from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, List, Any, Optional
from src.agents.research_agent import ResearchAgent
from src.agents.verification_agent import VerificationAgent
from src.agents.cross_validation_agent import CrossValidationAgent
from src.agents.summary_agent import SummaryAgent
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)


class AgentState(TypedDict):
    claim: str
    source: Optional[str]
    research_results: Dict[str, Any]
    verification_results: Dict[str, Any]
    validation_results: Dict[str, Any]
    final_summary: Dict[str, Any]
    messages: List[Dict[str, Any]]
    current_agent: str  # Track the current agent being executed


class SupervisorAgent:
    def __init__(self):
        self.research_agent = ResearchAgent()
        self.verification_agent = VerificationAgent()
        self.validation_agent = CrossValidationAgent()
        self.summary_agent = SummaryAgent()
        self.workflow = self._create_workflow()
        logging.info("SupervisorAgent initialized with all agent components")

    def _run_research_agent(self, state: AgentState) -> AgentState:
        """Run the research agent and update the workflow state."""
        logging.info("Running research agent")
        try:
            # Run the research agent
            updated_state = self.research_agent.run(state)
            # Mark this step as completed
            updated_state["current_agent"] = "research_completed"
            logging.info("Research agent completed successfully")
            return updated_state
        except Exception as e:
            logging.error(f"Error in research agent: {str(e)}")
            state["messages"].append({
                "agent": "research",
                "content": f"Error: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            state["current_agent"] = "error"
            return state

    def _run_verification_agent(self, state: AgentState) -> AgentState:
        """Run the verification agent and update the workflow state."""
        logging.info("Running verification agent")
        try:
            # Run the verification agent
            updated_state = self.verification_agent.run(state)
            # Mark this step as completed
            updated_state["current_agent"] = "verification_completed"
            logging.info("Verification agent completed successfully")
            return updated_state
        except Exception as e:
            logging.error(f"Error in verification agent: {str(e)}")
            state["messages"].append({
                "agent": "verification",
                "content": f"Error: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            state["current_agent"] = "error"
            return state

    def _run_validation_agent(self, state: AgentState) -> AgentState:
        """Run the validation agent and update the workflow state."""
        logging.info("Running validation agent")
        try:
            # Run the validation agent
            updated_state = self.validation_agent.run(state)
            # Mark this step as completed
            updated_state["current_agent"] = "validation_completed"
            logging.info("Validation agent completed successfully")
            return updated_state
        except Exception as e:
            logging.error(f"Error in validation agent: {str(e)}")
            state["messages"].append({
                "agent": "validation",
                "content": f"Error: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            state["current_agent"] = "error"
            return state

    def _run_summary_agent(self, state: AgentState) -> AgentState:
        """Run the summary agent and update the workflow state."""
        logging.info("Running summary agent")
        try:
            # Run the summary agent
            updated_state = self.summary_agent.run(state)
            # Mark this step as completed
            updated_state["current_agent"] = "completed"
            logging.info("Summary agent completed successfully")
            return updated_state
        except Exception as e:
            logging.error(f"Error in summary agent: {str(e)}")
            state["messages"].append({
                "agent": "summary",
                "content": f"Error: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            state["current_agent"] = "error"
            return state

    def _create_workflow(self):
        """Create the LangGraph workflow that orchestrates the fact-checking process."""
        workflow = StateGraph(AgentState)

        # Define the nodes for each agent
        workflow.add_node("research", self._run_research_agent)
        workflow.add_node("verification", self._run_verification_agent)
        workflow.add_node("validation", self._run_validation_agent)
        workflow.add_node("summary", self._run_summary_agent)

        # Define the routing logic based on the current_agent state
        def route_next_step(state: AgentState) -> str:
            """Determine the next step based on the current_agent value."""
            current = state.get("current_agent", "")
            
            if current == "":
                # Initial state, start with research
                return "research"
            elif current == "research_completed":
                # After research is done, go to verification
                return "verification"
            elif current == "verification_completed":
                # After verification is done, go to validation
                return "validation"
            elif current == "validation_completed":
                # After validation is done, go to summary
                return "summary"
            elif current == "completed" or current == "error":
                # Workflow is complete or encountered an error
                return END
            else:
                # Unknown state, end the workflow
                logging.warning(f"Unknown current_agent value: {current}, ending workflow")
                return END

        # Add conditional edges for each node
        workflow.add_conditional_edges(
            "research",
            route_next_step
        )
        workflow.add_conditional_edges(
            "verification",
            route_next_step
        )
        workflow.add_conditional_edges(
            "validation",
            route_next_step
        )
        workflow.add_conditional_edges(
            "summary",
            route_next_step
        )

        # Set the entry point
        workflow.set_entry_point("research")
        logging.info("Workflow created with all nodes and conditional edges")
        return workflow.compile()

    def run_fact_check(self, claim: str, source: str = None) -> dict:
        """Run the complete fact-checking workflow on a claim.
        
        Args:
            claim: The claim to fact-check
            source: Optional source URL to check specifically
            
        Returns:
            The complete state after workflow execution
        """
        logging.info(f"Starting fact-check for claim: {claim}")
        
        # Initialize the state with proper structure
        initial_state = {
            "claim": claim,
            "source": source,
            "research_results": {},
            "verification_results": {},
            "validation_results": {},
            "final_summary": {},
            "messages": [
                {
                    "agent": "supervisor",
                    "content": f"Starting fact-check for: {claim}",
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "current_agent": ""  # Start with empty current_agent
        }

        # Execute the workflow
        logging.info("Invoking fact-checking workflow")
        try:
            result = self.workflow.invoke(initial_state)
            
            # Check if workflow completed successfully
            if result.get("current_agent") == "completed":
                logging.info("Fact-checking workflow completed successfully")
                result["messages"].append({
                    "agent": "supervisor",
                    "content": "Fact-checking workflow completed successfully",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                logging.warning(f"Workflow ended with status: {result.get('current_agent')}")
                result["messages"].append({
                    "agent": "supervisor",
                    "content": f"Fact-checking workflow ended with status: {result.get('current_agent')}",
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            logging.error(f"Error in fact-checking workflow: {str(e)}")
            initial_state["messages"].append({
                "agent": "supervisor",
                "content": f"Workflow error: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            return initial_state
        
        return result