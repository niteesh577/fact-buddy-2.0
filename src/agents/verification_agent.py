from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from typing import Dict, Any, Optional, List, TypedDict
from src.tools.source_checker import SourceChecker
from datetime import datetime
import logging
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

logging.basicConfig(level=logging.INFO)


class VerificationState(TypedDict):
    """State for the verification agent."""
    claim: str
    research_results: Dict[str, Any]
    verification_results: Dict[str, Any]
    messages: List[Dict[str, Any]]


class VerificationAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768"
        )
        self.source_checker = SourceChecker()

        self.tools = [
            Tool(
                name="source_checker",
                func=self.source_checker.check_domain_authority,
                description="Check the authority and reliability of a source domain"
            ),
            Tool(
                name="content_analyzer",
                func=self.source_checker.analyze_content_quality,
                description="Analyze the quality of content from a source"
            )
        ]

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a verification agent. Analyze sources for credibility "
                      "and assign trust scores based on authority, recency, and reliability."),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])

        # Create a LangGraph-compatible agent
        self.agent = create_react_agent(
            self.llm,
            self.tools
        )
        
        # Create the verification graph
        self.graph = self._create_verification_graph()

    def _create_verification_graph(self):
        """Create a LangGraph workflow for the verification process."""
        workflow = StateGraph(VerificationState)
        
        # Add the verification node that processes the research results
        workflow.add_node("verification", self._process_verification)
        
        # Set the entry point and connect to END
        workflow.add_edge("verification", END)
        workflow.set_entry_point("verification")
        
        return workflow.compile()

    def calculate_trust_score(self, source_data):
        score = 0
        factors = {
            "domain_authority": 0.3,
            "content_quality": 0.3,
            "source_age": 0.2,
            "citation_count": 0.2
        }

        # Implementation of trust score calculation
        if isinstance(source_data, dict):
            # Check for domain authority
            if source_data.get("is_trusted", False):
                score += 0.8 * factors["domain_authority"]
            elif source_data.get("base_trust_score"):
                score += source_data["base_trust_score"] * factors["domain_authority"]
            
            # Check for content quality indicators
            if source_data.get("citation_count", 0) > 5:
                score += 0.7 * factors["citation_count"]
            elif source_data.get("citation_count", 0) > 0:
                score += 0.4 * factors["citation_count"]
            
            # Check for domain age
            age_days = source_data.get("age_days", 0)
            if age_days > 1825:  # Older than 5 years
                score += 0.9 * factors["source_age"]
            elif age_days > 365:  # Older than 1 year
                score += 0.6 * factors["source_age"]
            elif age_days > 30:   # Older than 1 month
                score += 0.3 * factors["source_age"]
            
            # Check for content quality
            if source_data.get("has_citations", False) and source_data.get("has_references", False):
                score += 0.8 * factors["content_quality"]
            elif source_data.get("has_structured_content", False):
                score += 0.5 * factors["content_quality"]

        return {
            "score": score,
            "factors": factors,
            "explanation": "Trust score based on domain authority, content quality, source age, and citations"
        }

    def _process_verification(self, state: VerificationState) -> VerificationState:
        """Process the verification for research results."""
        try:
            # Validate input
            if not state or "research_results" not in state:
                raise ValueError("Research results are required in state")

            research_results = state["research_results"]
            verification_data = {
                "trust_scores": {},
                "source_analysis": {},
                "overall_credibility": 0,
                "timestamp": datetime.now().isoformat()
            }

            try:
                # Analyze each source
                total_score = 0
                valid_sources = 0

                # Process scraped results if available
                if "scraped_results" in research_results:
                    for result in research_results["scraped_results"]:
                        url = result.get("url")
                        if not url or "error" in result:
                            continue

                        try:
                            # Check domain authority
                            domain_info = self.source_checker.check_domain_authority(url)
                            
                            # Analyze content quality if content is available
                            content_quality = {}
                            if "content" in result and "content" in result["content"]:
                                content_text = result["content"]["content"]
                                content_quality = self.source_checker.analyze_content_quality(content_text)
                            
                            # Combine domain and content analysis
                            source_data = {**domain_info, **content_quality}
                            
                            # Calculate trust score
                            trust_score = self.calculate_trust_score(source_data)
                            verification_data["trust_scores"][url] = trust_score
                            verification_data["source_analysis"][url] = source_data

                            if trust_score["score"] is not None:
                                total_score += trust_score["score"]
                                valid_sources += 1

                        except Exception as e:
                            logging.error(f"Error analyzing source {url}: {str(e)}")
                            verification_data["trust_scores"][url] = {
                                "error": str(e),
                                "score": None
                            }

                # Calculate overall credibility
                verification_data["overall_credibility"] = (
                    total_score / valid_sources if valid_sources > 0 else 0
                )

            except Exception as e:
                logging.error(f"Error in source analysis: {str(e)}")
                verification_data["analysis_error"] = str(e)

            state["verification_results"] = verification_data
            state.setdefault("messages", []).append({
                "agent": "verification",
                "content": "Verification completed" + (
                    " with some errors" if "analysis_error" in verification_data else ""
                ),
                "timestamp": datetime.now().isoformat()
            })

            return state

        except Exception as e:
            logging.error(f"Critical error in verification agent: {str(e)}")
            state.setdefault("messages", []).append({
                "agent": "verification",
                "content": f"Verification failed: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            return state

    def validate_input(self, state: Dict[str, Any]) -> Optional[str]:
        if not state:
            return "State cannot be empty"
        if "research_results" not in state:
            return "Research results are required in state"
        if not isinstance(state["research_results"], dict):
            return "Research results must be a dictionary"
        return None

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the verification agent on the given state."""
        # Initialize the verification state
        verification_state = {
            "claim": state["claim"],
            "research_results": state["research_results"],
            "verification_results": {},
            "messages": state.get("messages", [])
        }
        
        # Run the verification graph
        result = self.graph.invoke(verification_state)
        
        # Update the original state with verification results
        state["verification_results"] = result["verification_results"]
        state["messages"] = result["messages"]
        
        return state
