from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
import logging
from langchain.tools import Tool
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

logging.basicConfig(level=logging.INFO)

class ValidationResult(BaseModel):
    biases: List[Dict[str, str]] = Field(description="List of identified biases")
    logical_fallacies: List[Dict[str, str]] = Field(description="List of logical fallacies")
    cross_references: List[str] = Field(description="Cross-referenced sources")
    confidence_score: float = Field(description="Overall confidence score")

class ValidationState(TypedDict):
    """State for the validation agent."""
    claim: str
    research_results: Dict[str, Any]
    verification_results: Dict[str, Any]
    validation_results: Dict[str, Any]
    messages: List[Dict[str, Any]]

class CrossValidationAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a critical analysis agent specialized in identifying:
                      1. Cognitive biases and prejudices in information
                      2. Logical fallacies in arguments
                      3. Cross-referencing different sources
                      4. Evaluating the strength of evidence
                      
                      Analyze the research and verification results thoroughly."""),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        
        # Create tools for the agent
        self.tools = [
            Tool(
                name="bias_detector",
                func=self.identify_biases,
                description="Identify cognitive biases in the research and verification results"
            ),
            Tool(
                name="fallacy_checker",
                func=self.check_logical_fallacies,
                description="Check for logical fallacies in the arguments presented"
            )
        ]
        
        # Create a LangGraph-compatible agent
        self.agent = create_react_agent(
            self.llm,
            self.tools
        )
        
        # Create the validation graph
        self.graph = self._create_validation_graph()
        
        self.output_parser = PydanticOutputParser(pydantic_object=ValidationResult)

    def _create_validation_graph(self):
        """Create a LangGraph workflow for the validation process."""
        workflow = StateGraph(ValidationState)
        
        # Add the validation node that processes the research and verification results
        workflow.add_node("validation", self._process_validation)
        
        # Set the entry point and connect to END
        workflow.add_edge("validation", END)
        workflow.set_entry_point("validation")
        
        return workflow.compile()

    def identify_biases(self, content):
        common_biases = {
            "confirmation_bias": "Looking for information that confirms existing beliefs",
            "selection_bias": "Cherry-picking data that supports a particular view",
            "recency_bias": "Giving too much weight to recent events",
            "authority_bias": "Excessive trust in authority figures"
        }
        
        identified_biases = []
        claim_text = content.get("claim", "").lower()
        research_data = content.get("research", {})
        
        # Simplified implementation to avoid LLM calls during analysis
        if "search_results" in research_data and "organic_results" in research_data["search_results"]:
            results = research_data["search_results"]["organic_results"]
            if len(results) < 3:
                identified_biases.append({
                    "type": "confirmation_bias",
                    "description": "Limited number of sources may indicate confirmation bias"
                })
        
        # Check for authority bias in the claim
        authority_terms = ["expert", "official", "authority", "scientist", "professor", "doctor"]
        if any(term in claim_text for term in authority_terms):
            identified_biases.append({
                "type": "authority_bias",
                "description": "Claim relies on authority figures"
            })
        
        return identified_biases

    def check_logical_fallacies(self, content):
        fallacy_types = {
            "ad_hominem": "Attacking the person instead of the argument",
            "false_causality": "Assuming correlation implies causation",
            "straw_man": "Misrepresenting an argument to make it easier to attack",
            "appeal_to_emotion": "Using emotions instead of facts"
        }
        
        identified_fallacies = []
        claim_text = content.get("claim", "").lower()
        research_data = content.get("research", {})
        
        # Check for ad hominem attacks
        if "search_results" in research_data and "organic_results" in research_data["search_results"]:
            for result in research_data["search_results"]["organic_results"]:
                snippet = result.get("snippet", "").lower()
                if any(phrase in snippet for phrase in ["idiot", "stupid", "incompetent", "fool"]):
                    identified_fallacies.append({
                        "type": "ad_hominem",
                        "description": "Arguments contain personal attacks"
                    })
                    break
        
        # Simplified check for false causality
        causality_terms = ["causes", "because of", "due to", "leads to", "results in"]
        if any(term in claim_text for term in causality_terms):
            identified_fallacies.append({
                "type": "potential_false_causality",
                "description": "Claim may assume causation without sufficient evidence"
            })
        
        return identified_fallacies

    def _process_validation(self, state: ValidationState) -> ValidationState:
        """Process the validation for research and verification results."""
        try:
            # Validate input
            if not state or "research_results" not in state or "verification_results" not in state:
                raise ValueError("Research and verification results are required in state")

            claim = state["claim"]
            research_results = state["research_results"]
            verification_results = state["verification_results"]
            
            # Prepare content for analysis
            content = {
                "claim": claim,
                "research": research_results,
                "verification": verification_results
            }
            
            # Identify biases
            biases = self.identify_biases(content)
            
            # Check for logical fallacies
            fallacies = self.check_logical_fallacies(content)
            
            # Cross-reference sources
            cross_references = []
            if "scraped_results" in research_results:
                for result in research_results["scraped_results"]:
                    if "url" in result and "error" not in result:
                        cross_references.append(result["url"])
            
            # Calculate confidence score based on verification results and identified issues
            base_confidence = verification_results.get("overall_credibility", 0.5)
            confidence_penalty = (len(biases) * 0.05) + (len(fallacies) * 0.1)
            confidence_score = max(0.1, min(1.0, base_confidence - confidence_penalty))
            
            # Create validation results
            validation_data = {
                "biases": biases,
                "logical_fallacies": fallacies,
                "cross_references": cross_references,
                "confidence_score": confidence_score,
                "timestamp": datetime.now().isoformat()
            }
            
            state["validation_results"] = validation_data
            state.setdefault("messages", []).append({
                "agent": "validation",
                "content": f"Validation completed with confidence score: {confidence_score:.2f}",
                "timestamp": datetime.now().isoformat()
            })
            
            return state
            
        except Exception as e:
            logging.error(f"Critical error in validation agent: {str(e)}")
            state.setdefault("messages", []).append({
                "agent": "validation",
                "content": f"Validation failed: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            return state

    def validate_input(self, state: Dict[str, Any]) -> Optional[str]:
        if not state:
            return "State cannot be empty"
        if "research_results" not in state:
            return "Research results are required in state"
        if "verification_results" not in state:
            return "Verification results are required in state"
        return None

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the validation agent on the given state."""
        # Initialize the validation state
        validation_state = {
            "claim": state["claim"],
            "research_results": state["research_results"],
            "verification_results": state["verification_results"],
            "validation_results": {},
            "messages": state.get("messages", [])
        }
        
        # Run the validation graph
        result = self.graph.invoke(validation_state)
        
        # Update the original state with validation results
        state["validation_results"] = result["validation_results"]
        state["messages"] = result["messages"]
        
        return state