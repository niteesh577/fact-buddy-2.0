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

class SummaryOutput(BaseModel):
    verdict: str = Field(description="Final verdict on the claim")
    confidence_level: float = Field(description="Overall confidence in the conclusion")
    key_findings: List[Dict[str, Any]] = Field(description="Main findings with citations and direct quotes")
    evidence_summary: str = Field(description="Detailed summary of supporting and contradicting evidence")
    consensus_analysis: str = Field(description="Analysis of consensus across multiple sources")
    citations: List[Dict[str, Any]] = Field(description="List of citations and sources with credibility metrics")
    contradicting_evidence: List[Dict[str, Any]] = Field(description="Evidence that contradicts the claim")
    supporting_evidence: List[Dict[str, Any]] = Field(description="Evidence that supports the claim")

class SummaryState(TypedDict):
    """State for the summary agent."""
    claim: str
    research_results: Dict[str, Any]
    verification_results: Dict[str, Any]
    validation_results: Dict[str, Any]
    final_summary: Dict[str, Any]
    messages: List[Dict[str, Any]]

class SummaryAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert summary agent that:
                      1. Synthesizes research findings and verification results
                      2. Provides evidence-based conclusions
                      3. Includes proper citations for all claims
                      4. Maintains objectivity in reporting
                      
                      Generate a comprehensive summary with citations."""),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        
        # Create tools for the agent
        self.tools = [
            Tool(
                name="citation_generator",
                func=self.generate_citations,
                description="Generate citations from research and verification results"
            ),
            Tool(
                name="verdict_determiner",
                func=self.determine_verdict,
                description="Determine the final verdict on a claim based on all available evidence"
            )
        ]
        
        # Create a LangGraph-compatible agent
        self.agent = create_react_agent(
            self.llm,
            self.tools
        )
        
        # Create the summary graph
        self.graph = self._create_summary_graph()
        
        self.output_parser = PydanticOutputParser(pydantic_object=SummaryOutput)

    def _create_summary_graph(self):
        """Create a LangGraph workflow for the summary process."""
        workflow = StateGraph(SummaryState)
        
        # Add the summary node that processes all previous results
        workflow.add_node("summary", self._process_summary)
        
        # Set the entry point and connect to END
        workflow.add_edge("summary", END)
        workflow.set_entry_point("summary")
        
        return workflow.compile()

    def generate_citations(self, content):
        citations = []
        research_results = content.get("research_results", {})
        verification_results = content.get("verification_results", {})
        validation_results = content.get("validation_results", {})
        
        # Process scraped results
        if "scraped_results" in research_results:
            for result in research_results["scraped_results"]:
                url = result.get("url")
                title = result.get("title", "")
                
                if url and "error" not in result:
                    # Get trust score if available
                    trust_score = "N/A"
                    trust_factors = {}
                    if verification_results and "trust_scores" in verification_results:
                        if url in verification_results["trust_scores"]:
                            score_data = verification_results["trust_scores"][url]
                            if isinstance(score_data, dict):
                                if "score" in score_data:
                                    trust_score = str(score_data["score"])
                                if "factors" in score_data:
                                    trust_factors = score_data["factors"]
                                if "explanation" in score_data:
                                    trust_explanation = score_data["explanation"]
                    
                    # Extract publication date if available
                    publication_date = "Unknown"
                    if "content" in result and "metadata" in result["content"]:
                        metadata = result["content"]["metadata"]
                        if "date" in metadata:
                            publication_date = metadata["date"]
                    
                    # Check if this source is cross-referenced
                    is_cross_referenced = False
                    if validation_results and "cross_references" in validation_results:
                        if url in validation_results["cross_references"]:
                            is_cross_referenced = True
                    
                    # Extract a short snippet for context
                    snippet = ""
                    if "content" in result and "content" in result["content"]:
                        content_text = result["content"]["content"]
                        snippet = content_text[:150] + "..." if len(content_text) > 150 else content_text
                    
                    citations.append({
                        "source": url,
                        "title": title,
                        "trust_score": trust_score,
                        "trust_factors": trust_factors,
                        "publication_date": publication_date,
                        "is_cross_referenced": is_cross_referenced,
                        "snippet": snippet
                    })
        
        # Sort citations by trust score (highest first)
        citations.sort(key=lambda x: float(x["trust_score"]) if x["trust_score"] != "N/A" else 0, reverse=True)
        
        return citations

    def determine_verdict(self, content):
        # Extract all necessary data
        research_results = content.get("research_results", {})
        verification_results = content.get("verification_results", {})
        validation_results = content.get("validation_results", {})
        claim = content.get("claim", "")
        
        # Get overall credibility from verification
        credibility = verification_results.get("overall_credibility", 0.5)
        
        # Get confidence score from validation
        confidence = validation_results.get("confidence_score", 0.5)
        
        # Enhanced evidence analysis
        supporting_evidence_items = []
        contradicting_evidence_items = []
        neutral_evidence_items = []
        evidence_quality_scores = []
        
        # Check scraped results for evidence with weighted analysis
        if "scraped_results" in research_results:
            for result in research_results["scraped_results"]:
                if "content" in result and "content" in result["content"] and "url" in result:
                    content_text = result["content"]["content"].lower()
                    url = result["url"]
                    title = result.get("title", "")
                    
                    # Get source credibility for weighting
                    source_credibility = 0.5  # Default medium credibility
                    if "trust_scores" in verification_results and url in verification_results["trust_scores"]:
                        score_data = verification_results["trust_scores"][url]
                        if isinstance(score_data, dict) and "score" in score_data:
                            source_credibility = score_data["score"]
                    
                    # More sophisticated evidence classification
                    support_terms = ["true", "confirmed", "verified", "proven", "accurate", "correct", "factual", "supported"]
                    contradict_terms = ["false", "incorrect", "misleading", "wrong", "inaccurate", "untrue", "fabricated", "exaggerated"]
                    neutral_terms = ["unclear", "debated", "disputed", "controversial", "mixed", "uncertain"]
                    
                    # Extract a relevant snippet for evidence
                    max_snippet_length = 200
                    snippet = content_text[:max_snippet_length] + "..." if len(content_text) > max_snippet_length else content_text
                    
                    # Calculate evidence strength based on term frequency and source credibility
                    support_strength = sum(content_text.count(term) for term in support_terms) * source_credibility
                    contradict_strength = sum(content_text.count(term) for term in contradict_terms) * source_credibility
                    neutral_strength = sum(content_text.count(term) for term in neutral_terms) * source_credibility
                    
                    evidence_item = {
                        "source": url,
                        "title": title,
                        "snippet": snippet,
                        "credibility": source_credibility
                    }
                    
                    # Classify evidence with weighted approach
                    if support_strength > contradict_strength and support_strength > neutral_strength:
                        evidence_item["type"] = "supporting"
                        evidence_item["strength"] = support_strength
                        supporting_evidence_items.append(evidence_item)
                        evidence_quality_scores.append(source_credibility * support_strength)
                    elif contradict_strength > support_strength and contradict_strength > neutral_strength:
                        evidence_item["type"] = "contradicting"
                        evidence_item["strength"] = contradict_strength
                        contradicting_evidence_items.append(evidence_item)
                        evidence_quality_scores.append(source_credibility * contradict_strength)
                    else:
                        evidence_item["type"] = "neutral"
                        evidence_item["strength"] = neutral_strength
                        neutral_evidence_items.append(evidence_item)
                        evidence_quality_scores.append(source_credibility * 0.5)  # Neutral evidence has less impact
        
        # Sort evidence by strength and credibility
        supporting_evidence_items.sort(key=lambda x: x.get("strength", 0) * x.get("credibility", 0.5), reverse=True)
        contradicting_evidence_items.sort(key=lambda x: x.get("strength", 0) * x.get("credibility", 0.5), reverse=True)
        
        # Calculate weighted evidence counts
        supporting_evidence_weight = sum(item.get("strength", 0) * item.get("credibility", 0.5) for item in supporting_evidence_items)
        contradicting_evidence_weight = sum(item.get("strength", 0) * item.get("credibility", 0.5) for item in contradicting_evidence_items)
        
        # Calculate evidence quality average
        avg_evidence_quality = sum(evidence_quality_scores) / len(evidence_quality_scores) if evidence_quality_scores else 0.5
        
        # Determine verdict based on weighted evidence, credibility, and confidence
        verdict = "Inconclusive"
        
        # Calculate confidence level with more factors
        confidence_level = (
            credibility * 0.4 +  # Source credibility
            confidence * 0.3 +   # Validation confidence
            avg_evidence_quality * 0.3  # Evidence quality
        )
        
        # Evidence ratio for determining verdict
        evidence_ratio = 0
        if supporting_evidence_weight + contradicting_evidence_weight > 0:
            evidence_ratio = supporting_evidence_weight / (supporting_evidence_weight + contradicting_evidence_weight)
        
        # More nuanced verdict determination
        if evidence_ratio > 0.8 and confidence_level >= 0.7:
            verdict = "True"
        elif evidence_ratio < 0.2 and confidence_level >= 0.7:
            verdict = "False"
        elif evidence_ratio > 0.6 and confidence_level >= 0.6:
            verdict = "Likely True"
        elif evidence_ratio < 0.4 and confidence_level >= 0.6:
            verdict = "Likely False"
        elif 0.4 <= evidence_ratio <= 0.6 and len(supporting_evidence_items) > 0 and len(contradicting_evidence_items) > 0:
            verdict = "Partially True"
        elif len(neutral_evidence_items) > len(supporting_evidence_items) + len(contradicting_evidence_items):
            verdict = "Disputed"
        
        # Check for biases that might affect verdict
        biases = []
        if "biases" in validation_results:
            biases = validation_results["biases"]
        
        # Adjust confidence if significant biases are detected
        if len(biases) > 2:
            confidence_level = max(0.1, confidence_level - 0.1 * len(biases))
        
        return {
            "verdict": verdict,
            "confidence_level": confidence_level,
            "supporting_evidence": supporting_evidence_items,
            "contradicting_evidence": contradicting_evidence_items,
            "neutral_evidence": neutral_evidence_items,
            "evidence_quality": avg_evidence_quality,
            "biases": biases
        }

    def _process_summary(self, state: SummaryState) -> SummaryState:
        """Process the summary for all previous results."""
        try:
            # Validate input
            if not state or "research_results" not in state or "verification_results" not in state or "validation_results" not in state:
                raise ValueError("All previous results are required in state")

            # Prepare content for processing
            content = {
                "claim": state["claim"],
                "research_results": state["research_results"],
                "verification_results": state["verification_results"],
                "validation_results": state["validation_results"]
            }
            
            # Generate citations
            citations = self.generate_citations(content)
            
            # Determine verdict with enhanced evidence analysis
            verdict_data = self.determine_verdict(content)
            
            # Extract key findings with direct quotes from research results
            key_findings = []
            if "scraped_results" in state["research_results"]:
                for result in state["research_results"]["scraped_results"]:
                    if "content" in result and "content" in result["content"] and "url" in result and "error" not in result:
                        content_text = result["content"]["content"]
                        url = result["url"]
                        title = result.get("title", "")
                        
                        # Get source credibility for ranking
                        source_credibility = 0.5  # Default medium credibility
                        if "trust_scores" in state["verification_results"] and url in state["verification_results"]["trust_scores"]:
                            score_data = state["verification_results"]["trust_scores"][url]
                            if isinstance(score_data, dict) and "score" in score_data:
                                source_credibility = score_data["score"]
                        
                        # Extract meaningful quotes (sentences containing claim keywords)
                        claim_keywords = [word.lower() for word in state["claim"].split() if len(word) > 3]
                        sentences = content_text.replace(".", ". ").replace("!", "! ").replace("?", "? ").split(". ")
                        
                        relevant_quotes = []
                        for sentence in sentences:
                            if any(keyword in sentence.lower() for keyword in claim_keywords):
                                clean_sentence = sentence.strip()
                                if clean_sentence and len(clean_sentence) > 10:
                                    relevant_quotes.append(clean_sentence)
                        
                        # Use the most relevant quote or a fallback
                        best_quote = ""
                        if relevant_quotes:
                            # Sort by number of keywords matched
                            relevant_quotes.sort(key=lambda q: sum(1 for kw in claim_keywords if kw in q.lower()), reverse=True)
                            best_quote = relevant_quotes[0][:150] + "..." if len(relevant_quotes[0]) > 150 else relevant_quotes[0]
                        else:
                            # Fallback to first paragraph
                            paragraphs = content_text.split("\n")
                            for p in paragraphs:
                                if len(p.strip()) > 20:  # Reasonable paragraph length
                                    best_quote = p.strip()[:150] + "..." if len(p.strip()) > 150 else p.strip()
                                    break
                        
                        if best_quote:
                            key_findings.append({
                                "finding": title,
                                "source": url,
                                "quote": best_quote,
                                "credibility": source_credibility
                            })
                
                # Sort key findings by source credibility
                key_findings.sort(key=lambda x: x.get("credibility", 0), reverse=True)
                
                # Limit to top findings but ensure diversity of perspectives
                if len(key_findings) > 7:
                    # Keep top 4 findings
                    top_findings = key_findings[:4]
                    # Add some contradicting findings if available
                    contradicting_findings = [f for f in key_findings[4:] if any(url in f["source"] for url in 
                                            [item["source"] for item in verdict_data.get("contradicting_evidence", [])])]
                    if contradicting_findings:
                        top_findings.extend(contradicting_findings[:2])
                    # Fill remaining slots with other findings
                    remaining_slots = 7 - len(top_findings)
                    if remaining_slots > 0:
                        top_findings.extend([f for f in key_findings[4:] if f not in top_findings][:remaining_slots])
                    key_findings = top_findings
            
            # Create comprehensive evidence summary with supporting and contradicting evidence
            supporting_evidence = verdict_data.get("supporting_evidence", [])
            contradicting_evidence = verdict_data.get("contradicting_evidence", [])
            neutral_evidence = verdict_data.get("neutral_evidence", [])
            
            # Build a balanced evidence summary
            evidence_summary_parts = []
            
            if supporting_evidence:
                evidence_summary_parts.append("**Supporting Evidence:**")
                for i, evidence in enumerate(supporting_evidence[:3]):  # Top 3 supporting points
                    evidence_summary_parts.append(f"{i+1}. {evidence.get('snippet', '')}")
                    evidence_summary_parts.append(f"   Source: {evidence.get('title', '')} ({evidence.get('source', '')})")
            
            if contradicting_evidence:
                evidence_summary_parts.append("\n**Contradicting Evidence:**")
                for i, evidence in enumerate(contradicting_evidence[:3]):  # Top 3 contradicting points
                    evidence_summary_parts.append(f"{i+1}. {evidence.get('snippet', '')}")
                    evidence_summary_parts.append(f"   Source: {evidence.get('title', '')} ({evidence.get('source', '')})")
            
            if neutral_evidence:
                evidence_summary_parts.append("\n**Neutral/Contextual Evidence:**")
                for i, evidence in enumerate(neutral_evidence[:2]):  # Top 2 neutral points
                    evidence_summary_parts.append(f"{i+1}. {evidence.get('snippet', '')}")
                    evidence_summary_parts.append(f"   Source: {evidence.get('title', '')} ({evidence.get('source', '')})")
            
            evidence_summary = "\n".join(evidence_summary_parts) if evidence_summary_parts else "No detailed evidence available."
            
            # Create consensus analysis
            consensus_analysis = self._analyze_consensus(supporting_evidence, contradicting_evidence, neutral_evidence, state["validation_results"])
            
            # Create final summary with enhanced data
            summary_data = {
                "verdict": verdict_data["verdict"],
                "confidence_level": verdict_data["confidence_level"],
                "key_findings": key_findings,
                "evidence_summary": evidence_summary,
                "consensus_analysis": consensus_analysis,
                "citations": citations,
                "supporting_evidence": supporting_evidence[:5],  # Limit to top 5
                "contradicting_evidence": contradicting_evidence[:5],  # Limit to top 5
                "timestamp": datetime.now().isoformat()
            }
            
            # Add biases if present
            if "biases" in verdict_data:
                summary_data["biases"] = verdict_data["biases"]
            
            state["final_summary"] = summary_data
            state.setdefault("messages", []).append({
                "agent": "summary",
                "content": f"Summary completed with verdict: {verdict_data['verdict']} (Confidence: {verdict_data['confidence_level']:.2f})",
                "timestamp": datetime.now().isoformat()
            })
            
            return state
            
        except Exception as e:
            logging.error(f"Critical error in summary agent: {str(e)}")
            state.setdefault("messages", []).append({
                "agent": "summary",
                "content": f"Summary failed: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            return state

    def _analyze_consensus(self, supporting_evidence, contradicting_evidence, neutral_evidence, validation_results):
        """Analyze consensus across multiple sources and provide a balanced view."""
        # Count evidence by type
        supporting_count = len(supporting_evidence)
        contradicting_count = len(contradicting_evidence)
        neutral_count = len(neutral_evidence)
        total_sources = supporting_count + contradicting_count + neutral_count
        
        # Calculate consensus percentages
        supporting_percentage = (supporting_count / total_sources * 100) if total_sources > 0 else 0
        contradicting_percentage = (contradicting_count / total_sources * 100) if total_sources > 0 else 0
        neutral_percentage = (neutral_count / total_sources * 100) if total_sources > 0 else 0
        
        # Get credibility-weighted consensus
        supporting_weight = sum(item.get('credibility', 0.5) for item in supporting_evidence)
        contradicting_weight = sum(item.get('credibility', 0.5) for item in contradicting_evidence)
        neutral_weight = sum(item.get('credibility', 0.5) for item in neutral_evidence)
        total_weight = supporting_weight + contradicting_weight + neutral_weight
        
        weighted_supporting = (supporting_weight / total_weight * 100) if total_weight > 0 else 0
        weighted_contradicting = (contradicting_weight / total_weight * 100) if total_weight > 0 else 0
        
        # Check for biases that might affect consensus
        biases = validation_results.get('biases', [])
        bias_factor = len(biases) * 5  # Each bias reduces consensus strength by 5%
        
        # Determine consensus level
        consensus_level = "Strong"
        if weighted_supporting > 80 and supporting_count >= 3:
            consensus_type = "supporting"
            consensus_strength = weighted_supporting - bias_factor
        elif weighted_contradicting > 80 and contradicting_count >= 3:
            consensus_type = "contradicting"
            consensus_strength = weighted_contradicting - bias_factor
        elif weighted_supporting > 60 and supporting_count >= 2:
            consensus_type = "moderately supporting"
            consensus_level = "Moderate"
            consensus_strength = weighted_supporting - bias_factor
        elif weighted_contradicting > 60 and contradicting_count >= 2:
            consensus_type = "moderately contradicting"
            consensus_level = "Moderate"
            consensus_strength = weighted_contradicting - bias_factor
        elif abs(weighted_supporting - weighted_contradicting) < 20:
            consensus_type = "divided"
            consensus_level = "Weak"
            consensus_strength = 100 - abs(weighted_supporting - weighted_contradicting)
        else:
            consensus_type = "insufficient evidence"
            consensus_level = "Insufficient"
            consensus_strength = 0
        
        # Adjust for small sample sizes
        if total_sources < 3:
            consensus_level = "Insufficient"
            consensus_strength = max(0, consensus_strength - 30)  # Penalize small sample sizes
        
        # Format the consensus analysis
        analysis = f"{consensus_level} consensus ({consensus_strength:.1f}% strength): "
        
        if consensus_type == "supporting":
            analysis += f"The majority of sources ({supporting_percentage:.1f}%) support the claim with strong credibility."
        elif consensus_type == "contradicting":
            analysis += f"The majority of sources ({contradicting_percentage:.1f}%) contradict the claim with strong credibility."
        elif consensus_type == "moderately supporting":
            analysis += f"More sources support ({supporting_percentage:.1f}%) than contradict ({contradicting_percentage:.1f}%) the claim."
        elif consensus_type == "moderately contradicting":
            analysis += f"More sources contradict ({contradicting_percentage:.1f}%) than support ({supporting_percentage:.1f}%) the claim."
        elif consensus_type == "divided":
            analysis += f"Sources are divided on this claim: {supporting_percentage:.1f}% support, {contradicting_percentage:.1f}% contradict."
        else:
            analysis += f"There is insufficient evidence to determine a clear consensus on this claim."
        
        # Add information about source diversity
        source_domains = set()
        for evidence in supporting_evidence + contradicting_evidence + neutral_evidence:
            if 'source' in evidence:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(evidence['source']).netloc
                    source_domains.add(domain)
                except:
                    pass
        
        if len(source_domains) > 0:
            analysis += f" Evidence comes from {len(source_domains)} different domains."
        
        # Add bias warning if significant
        if len(biases) > 1:
            bias_types = [bias.get('type', 'unknown') for bias in biases[:2]]
            analysis += f" Note: Potential {' and '.join(bias_types)} bias detected in the evidence."
        
        return analysis
    
    def validate_input(self, state: Dict[str, Any]) -> Optional[str]:
        if not state:
            return "State cannot be empty"
        if "research_results" not in state:
            return "Research results are required in state"
        if "verification_results" not in state:
            return "Verification results are required in state"
        if "validation_results" not in state:
            return "Validation results are required in state"
        return None

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the summary agent on the given state."""
        # Initialize the summary state
        summary_state = {
            "claim": state["claim"],
            "research_results": state["research_results"],
            "verification_results": state["verification_results"],
            "validation_results": state["validation_results"],
            "final_summary": {},
            "messages": state.get("messages", [])
        }
        
        # Run the summary graph
        result = self.graph.invoke(summary_state)
        
        # Update the original state with summary results
        state["final_summary"] = result["final_summary"]
        state["messages"] = result["messages"]
        
        return state