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
import re
from urllib.parse import urlparse, parse_qs
from src.tools.video_transcript_extractor import VideoTranscriptExtractor

logging.basicConfig(level=logging.INFO)

class VideoVerificationOutput(BaseModel):
    verified_claims: List[Dict[str, Any]] = Field(description="List of verified claims from the video")
    unverified_claims: List[Dict[str, Any]] = Field(description="List of claims that couldn't be verified")
    misleading_claims: List[Dict[str, Any]] = Field(description="List of claims that are misleading or false")
    confidence_scores: Dict[str, float] = Field(description="Confidence scores for each verification")
    sources: List[Dict[str, Any]] = Field(description="Sources used for verification")
    verification_summary: str = Field(description="Overall summary of the verification process")

class VideoVerificationState(TypedDict):
    """State for the video verification agent."""
    claim: str
    video_url: str
    video_transcript: Dict[str, Any]
    video_research_results: Dict[str, Any]
    verification_results: Dict[str, Any]
    messages: List[Dict[str, Any]]

class VideoVerificationAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768"
        )
        
        # Initialize the video transcript extractor
        self.transcript_extractor = VideoTranscriptExtractor()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert video fact-checking agent that:
                      1. Analyzes video transcripts to identify factual claims
                      2. Verifies these claims against reliable sources
                      3. Provides evidence-based assessments of claim accuracy
                      4. Maintains objectivity and avoids political bias
                      5. Cites sources for all verifications
                      
                      Verify the factual claims in the video transcript and provide a detailed assessment."""),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        
        # Create tools for the agent
        self.tools = [
            Tool(
                name="extract_claims",
                func=self.extract_claims_from_transcript,
                description="Extract factual claims from the video transcript"
            ),
            Tool(
                name="verify_claim",
                func=self.verify_claim,
                description="Verify a specific claim against reliable sources"
            ),
            Tool(
                name="generate_verification_summary",
                func=self.generate_verification_summary,
                description="Generate a summary of all verification results"
            )
        ]
        
        # Create a LangGraph-compatible agent
        self.agent = create_react_agent(
            self.llm,
            self.tools
        )
        
        # Create the verification graph
        self.graph = self._create_verification_graph()
        
        self.output_parser = PydanticOutputParser(pydantic_object=VideoVerificationOutput)

    def _create_verification_graph(self):
        """Create a LangGraph workflow for the video verification process."""
        workflow = StateGraph(VideoVerificationState)
        
        # Add the verification node that processes the video transcript
        workflow.add_node("video_verification", self._process_video_verification)
        
        # Set the entry point and connect to END
        workflow.add_edge("video_verification", END)
        workflow.set_entry_point("video_verification")
        
        return workflow.compile()

    def extract_claims_from_transcript(self, content):
        """Extract factual claims from the video transcript with enhanced analysis."""
        transcript_data = content.get("video_transcript", {})
        transcript_text = transcript_data.get("transcript", "")
        
        if not transcript_text:
            return {
                "error": "No transcript available for claim extraction",
                "claims": []
            }
        
        # Extract segments if available
        segments = transcript_data.get("segments", [])
        
        # Split transcript into sentences with improved regex for better sentence detection
        sentences = re.split(r'(?<=[.!?])\s+', transcript_text)
        
        # Enhanced list of factual indicators with more nuanced patterns
        factual_indicators = [
            # Basic verbs of being and attribution
            "is", "are", "was", "were", "will be", "has", "have", "had", 
            "said", "claimed", "reported", "according to", "shows", "demonstrates",
            "proves", "confirmed", "verified", "found", "discovered",
            
            # Statistical and research indicators
            "percent", "study", "research", "evidence", "fact", "statistics",
            "data", "survey", "poll", "analysis", "expert", "scientist",
            "researcher", "professor", "doctor", "study shows", "research indicates",
            
            # Quantitative indicators
            "increase", "decrease", "rise", "fall", "grew", "reduced",
            "doubled", "tripled", "quadrupled", "halved",
            
            # Comparative indicators
            "more than", "less than", "greater than", "fewer than",
            "majority", "minority", "most", "many", "few",
            
            # Temporal indicators
            "always", "never", "often", "rarely", "sometimes",
            "frequently", "occasionally", "consistently",
            
            # Certainty indicators
            "definitely", "certainly", "undoubtedly", "clearly",
            "obviously", "without doubt", "absolutely"
        ]
        
        potential_claims = []
        
        # Process each sentence to identify claims
        for i, sentence in enumerate(sentences):
            # Skip very short sentences or empty ones
            if not sentence or len(sentence.split()) < 3:
                continue
                
            # Check if sentence contains factual indicators
            contains_indicator = any(indicator in sentence.lower() for indicator in factual_indicators)
            
            # Check for numerical content (strong indicator of factual claims)
            contains_numbers = bool(re.search(r'\d+', sentence))
            
            # Check for quotations (potential direct quotes)
            contains_quotes = '"' in sentence or "'" in sentence
            
            # Check for named entities (simplified check)
            contains_names = bool(re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', sentence))
            
            # If sentence has indicators of being a factual claim
            if contains_indicator or contains_numbers or (contains_quotes and len(sentence.split()) > 8):
                # Find corresponding segment if available
                timestamp = None
                matching_segment = None
                
                if segments:
                    # Improved segment matching
                    for segment in segments:
                        segment_text = segment.get("text", "")
                        
                        # Check for direct containment
                        if sentence.strip() in segment_text:
                            timestamp = segment.get("start")
                            matching_segment = segment
                            break
                            
                        # Check for significant word overlap
                        sentence_words = set(word.lower() for word in sentence.split() if len(word) > 4)
                        segment_words = set(word.lower() for word in segment_text.split() if len(word) > 4)
                        
                        if sentence_words and segment_words and len(sentence_words.intersection(segment_words)) / len(sentence_words) > 0.6:
                            timestamp = segment.get("start")
                            matching_segment = segment
                            break
                
                # Calculate confidence based on multiple factors
                confidence = 0.3  # Base confidence
                
                # Adjust confidence based on various factors
                if contains_numbers:
                    confidence += 0.2  # Numbers often indicate factual claims
                    
                if contains_quotes:
                    confidence += 0.15  # Direct quotes are often factual claims
                    
                if contains_names:
                    confidence += 0.1  # Named entities often indicate factual claims
                    
                # Specific factual terms increase confidence
                if any(term in sentence.lower() for term in ["percent", "study", "research", "according", "evidence", "fact", "statistics", "data"]):
                    confidence += 0.25
                    
                # Length-based confidence adjustment (longer sentences with factual indicators are more likely to be claims)
                if len(sentence.split()) > 12 and contains_indicator:
                    confidence += 0.1
                    
                # Timestamp presence increases confidence
                if timestamp is not None:
                    confidence += 0.1
                
                # Create claim object with enhanced metadata
                claim_obj = {
                    "claim": sentence.strip(),
                    "timestamp": timestamp,
                    "confidence": min(1.0, confidence),  # Cap at 1.0
                    "context": " ".join(sentences[max(0, i-1):min(len(sentences), i+2)]).strip(),  # Include surrounding context
                    "contains_numbers": contains_numbers,
                    "contains_quotes": contains_quotes,
                    "contains_names": contains_names
                }
                
                # Add segment text if available
                if matching_segment:
                    claim_obj["segment_text"] = matching_segment.get("text")
                    
                potential_claims.append(claim_obj)
        
        # Sort claims by confidence
        potential_claims.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Return claims with metadata about the extraction process
        return {
            "claims": potential_claims[:15],  # Return top 15 claims for more comprehensive analysis
            "total_claims_found": len(potential_claims),
            "transcript_word_count": len(transcript_text.split()),
            "transcript_sentence_count": len(sentences),
            "extraction_quality": "high" if len(potential_claims) > 5 else "medium" if len(potential_claims) > 0 else "low"
        }

    def verify_claim(self, claim_data):
        """Verify a specific claim against reliable sources.
        
        In a production environment, this would use search APIs, knowledge bases, etc.
        For this implementation, we'll simulate the verification process.
        """
        claim_text = claim_data.get("claim", "")
        
        if not claim_text:
            return {
                "error": "No claim provided for verification",
                "verification": {}
            }
        
        # Simulate verification process
        # In a real implementation, this would use external APIs, search engines, etc.
        
        # Simple heuristics for demonstration purposes
        contains_numbers = bool(re.search(r'\d+', claim_text))
        contains_names = bool(re.search(r'[A-Z][a-z]+ [A-Z][a-z]+', claim_text))
        contains_dates = bool(re.search(r'\b\d{4}\b|January|February|March|April|May|June|July|August|September|October|November|December', claim_text))
        contains_statistics = bool(re.search(r'\d+%|percent|proportion|ratio|rate', claim_text.lower()))
        
        # Calculate verifiability score
        verifiability = 0.5  # Base score
        if contains_numbers: verifiability += 0.1
        if contains_names: verifiability += 0.1
        if contains_dates: verifiability += 0.1
        if contains_statistics: verifiability += 0.2
        
        # Simulate verification outcome
        # In a real implementation, this would be based on actual fact-checking
        verification_outcomes = ["verified", "partially_verified", "unverified", "misleading", "false"]
        verification_weights = [0.4, 0.3, 0.1, 0.1, 0.1]  # Probabilities for each outcome
        
        # Simple deterministic approach for demonstration
        outcome_index = int(hash(claim_text) % 100) % len(verification_outcomes)
        verification_status = verification_outcomes[outcome_index]
        
        # Generate simulated sources
        sources = []
        if verification_status != "unverified":
            sources = [
                {
                    "name": "Simulated Source 1",
                    "url": "https://example.com/source1",
                    "credibility": 0.8,
                    "relevance": 0.9
                },
                {
                    "name": "Simulated Source 2",
                    "url": "https://example.com/source2",
                    "credibility": 0.7,
                    "relevance": 0.6
                }
            ]
        
        return {
            "verification": {
                "claim": claim_text,
                "status": verification_status,
                "confidence": verifiability,
                "sources": sources,
                "explanation": f"This is a simulated verification of the claim. In a production environment, this would contain a detailed explanation of the verification process and results."
            }
        }

    def generate_verification_summary(self, verification_results):
        """Generate a summary of all verification results."""
        verified_claims = verification_results.get("verified_claims", [])
        unverified_claims = verification_results.get("unverified_claims", [])
        misleading_claims = verification_results.get("misleading_claims", [])
        
        total_claims = len(verified_claims) + len(unverified_claims) + len(misleading_claims)
        
        if total_claims == 0:
            return {
                "summary": "No claims were verified in this video.",
                "overall_assessment": "insufficient_data"
            }
        
        # Calculate percentages
        verified_percent = len(verified_claims) / total_claims * 100 if total_claims > 0 else 0
        misleading_percent = len(misleading_claims) / total_claims * 100 if total_claims > 0 else 0
        
        # Determine overall assessment
        if verified_percent >= 80:
            overall_assessment = "highly_reliable"
        elif verified_percent >= 60:
            overall_assessment = "mostly_reliable"
        elif verified_percent >= 40:
            overall_assessment = "mixed_reliability"
        elif verified_percent >= 20:
            overall_assessment = "mostly_unreliable"
        else:
            overall_assessment = "highly_unreliable"
        
        # Generate summary text
        summary = f"Analysis of {total_claims} factual claims in the video: "
        summary += f"{len(verified_claims)} verified ({verified_percent:.1f}%), "
        summary += f"{len(unverified_claims)} unverified ({len(unverified_claims) / total_claims * 100:.1f}%), "
        summary += f"{len(misleading_claims)} misleading or false ({misleading_percent:.1f}%). "
        
        # Add assessment
        if overall_assessment == "highly_reliable":
            summary += "Overall, the video content appears to be highly reliable and factually accurate."
        elif overall_assessment == "mostly_reliable":
            summary += "Overall, the video content appears to be mostly reliable with some unverified or misleading claims."
        elif overall_assessment == "mixed_reliability":
            summary += "The video contains a mix of verified and unverified or misleading claims, requiring careful evaluation."
        elif overall_assessment == "mostly_unreliable":
            summary += "The video contains mostly unverified or misleading claims and should be treated with significant skepticism."
        else:
            summary += "The video contains predominantly misleading or false claims and should not be considered a reliable source of information."
        
        return {
            "summary": summary,
            "overall_assessment": overall_assessment
        }

    def _process_video_verification(self, state: VideoVerificationState) -> VideoVerificationState:
        """Process the video verification for a claim and video URL."""
        try:
            # Validate input state
            if not state or "video_url" not in state:
                raise ValueError("Video URL is required in state")

            video_url = state["video_url"]
            # Clean up the video URL if it contains backticks
            if isinstance(video_url, str):
                video_url = video_url.strip('` ')
                
            claim = state.get("claim", "")
            verification_data = {}

            # Extract video transcript if not already present
            if not state.get("video_transcript"):
                transcript_data = self.transcript_extractor.extract_transcript(video_url)
                state["video_transcript"] = transcript_data

                if not transcript_data:
                    error_msg = 'No transcript data available'
                    logging.error(f"Failed to extract transcript: {error_msg}")
                    verification_data["transcript_error"] = error_msg
                    verification_data["status"] = "failed"
                    verification_data.update({
                        "verified_claims": [],
                        "unverified_claims": [],
                        "misleading_claims": [],
                        "confidence_scores": {},
                        "sources": [],
                        "verification_summary": f"Unable to verify: {error_msg}"
                    })
                    state["verification_results"] = verification_data
                    state.setdefault("messages", []).append({
                        "agent": "video_verification",
                        "content": f"Failed to extract transcript: {error_msg}",
                        "error": True,
                        "timestamp": datetime.now().isoformat()
                    })
                    return state
                    
                if transcript_data.get("status") == "failed":
                    error_msg = transcript_data.get('error', 'Failed to extract transcript')
                    logging.error(f"Failed to extract transcript: {error_msg}")
                    verification_data["transcript_error"] = error_msg
                    verification_data["status"] = "failed"
                    verification_data.update({
                        "verified_claims": [],
                        "unverified_claims": [],
                        "misleading_claims": [],
                        "confidence_scores": {},
                        "sources": [],
                        "verification_summary": f"Unable to verify: {error_msg}"
                    })
                    state["verification_results"] = verification_data
                    state.setdefault("messages", []).append({
                        "agent": "video_verification",
                        "content": f"Failed to extract transcript: {error_msg}",
                        "error": True,
                        "timestamp": datetime.now().isoformat()
                    })
                    return state

            # Extract claims from transcript - safely handle potential None values
            try:
                transcript_data = state.get("video_transcript", {})
                transcript_text = transcript_data.get("transcript", "") if transcript_data else ""
                
                # Log the transcript for debugging
                logging.info(f"Analyzing transcript: {transcript_text[:200]}...")
                
                # Extract claims related to the user's claim
                claims_data = self.extract_claims_from_transcript({"video_transcript": state["video_transcript"]})
                
                if claims_data.get("error"):
                    raise ValueError(f"Error extracting claims: {claims_data.get('error')}")
                    
                claims = claims_data.get("claims", [])
                
                if not claims:
                    logging.warning("No claims were extracted from the transcript")
                    verification_data = {
                        "status": "completed",
                        "verified_claims": [],
                        "unverified_claims": [],
                        "misleading_claims": [],
                        "confidence_scores": {},
                        "sources": [],
                        "verification_summary": "No factual claims were identified in the video transcript."
                    }
                    state["verification_results"] = verification_data
                    return state
                
                # Filter claims by relevance to the user's claim
                if claim:
                    # Calculate relevance of each claim to the user's claim
                    relevant_claims = []
                    for claim_item in claims:
                        relevance = self._calculate_claim_relevance(claim_item["claim"], claim)
                        if relevance > 0.3:  # Only include somewhat relevant claims
                            claim_item["relevance_to_user_claim"] = relevance
                            relevant_claims.append(claim_item)
                    
                    # Sort by relevance
                    relevant_claims.sort(key=lambda x: x.get("relevance_to_user_claim", 0), reverse=True)
                    claims = relevant_claims[:10]  # Keep top 10 most relevant claims
                
                # Verify each claim
                verified_claims = []
                unverified_claims = []
                misleading_claims = []
            except Exception as e:
                logging.error(f"Error during claims extraction and processing: {str(e)}")
                verification_data = {
                    "status": "failed",
                    "error": f"Error during claims extraction: {str(e)}",
                    "verified_claims": [],
                    "unverified_claims": [],
                    "misleading_claims": [],
                    "confidence_scores": {},
                    "sources": [],
                    "verification_summary": "Verification process encountered an error during claims extraction."
                }
                state["verification_results"] = verification_data
                return state
            
            for claim_item in claims:
                verification_result = self.verify_claim({"claim": claim_item["claim"]})
                verification = verification_result.get("verification", {})
                
                # Add verification result to appropriate category
                if verification.get("status") == "verified":
                    verified_claims.append({
                        **claim_item,
                        "verification": verification
                    })
                elif verification.get("status") == "partially_verified":
                    verified_claims.append({
                        **claim_item,
                        "verification": verification,
                        "partially_verified": True
                    })
                elif verification.get("status") in ["misleading", "false"]:
                    misleading_claims.append({
                        **claim_item,
                        "verification": verification
                    })
                else:  # unverified
                    unverified_claims.append({
                        **claim_item,
                        "verification": verification
                    })
            
            # Compile verification results
            verification_data = {
                "verified_claims": verified_claims,
                "unverified_claims": unverified_claims,
                "misleading_claims": misleading_claims,
                "total_claims": len(claims),
                "verification_timestamp": datetime.now().isoformat()
            }
            
            # Generate summary
            summary_result = self.generate_verification_summary(verification_data)
            verification_data["summary"] = summary_result.get("summary", "")
            verification_data["overall_assessment"] = summary_result.get("overall_assessment", "")
            
            # Direct analysis of transcript against the user's claim
            if claim and transcript_text:
                direct_analysis = self._analyze_transcript_for_claim(transcript_text, claim)
                verification_data["direct_analysis"] = direct_analysis

            # Update state with verification results
            state["verification_results"] = verification_data
            state.setdefault("messages", []).append({
                "agent": "video_verification",
                "content": "Video verification completed",
                "timestamp": datetime.now().isoformat()
            })

            return state

        except Exception as e:
            logging.error(f"Critical error in video verification agent: {str(e)}")
            state.setdefault("messages", []).append({
                "agent": "video_verification",
                "content": f"Video verification failed: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            return state
            
    def _calculate_claim_relevance(self, sentence: str, original_claim: str) -> float:
        """Calculate relevance of a sentence to the original claim."""
        # Normalize and tokenize
        sentence_words = set(sentence.lower().split())
        claim_words = set(original_claim.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by"}
        sentence_words = sentence_words - stop_words
        claim_words = claim_words - stop_words
        
        # Calculate Jaccard similarity
        if not sentence_words or not claim_words:
            return 0.0
            
        intersection = len(sentence_words.intersection(claim_words))
        union = len(sentence_words.union(claim_words))
        
        return intersection / union
        
    def _analyze_transcript_for_claim(self, transcript_text: str, claim: str) -> Dict[str, Any]:
        """Directly analyze the transcript for evidence supporting or contradicting the claim."""
        # Simple analysis based on keyword matching
        # In a production environment, this would use more sophisticated NLP
        
        # Normalize text and claim
        transcript_lower = transcript_text.lower()
        claim_lower = claim.lower()
        
        # Extract key terms from claim
        claim_words = [w for w in claim_lower.split() if len(w) > 3 and w not in {"the", "and", "that", "this", "with", "from", "what", "where", "when", "which", "who", "whom", "whose", "how", "why"}]
        
        # Count occurrences of claim words in transcript
        word_occurrences = {}
        for word in claim_words:
            word_occurrences[word] = transcript_lower.count(word)
        
        # Calculate a simple relevance score
        total_occurrences = sum(word_occurrences.values())
        coverage = len([w for w, count in word_occurrences.items() if count > 0]) / len(claim_words) if claim_words else 0
        
        # Determine if transcript supports or contradicts claim
        # This is a simplified approach - in production, use more sophisticated analysis
        support_terms = ["yes", "true", "correct", "right", "agree", "confirm", "support", "evidence", "proof", "demonstrate"]
        contradict_terms = ["no", "false", "incorrect", "wrong", "disagree", "deny", "refute", "disprove", "contradict", "against"]
        
        support_count = sum(transcript_lower.count(term) for term in support_terms)
        contradict_count = sum(transcript_lower.count(term) for term in contradict_terms)
        
        # Determine stance based on simple term counting
        stance = "neutral"
        if support_count > contradict_count * 2:
            stance = "supporting"
        elif contradict_count > support_count * 2:
            stance = "contradicting"
        
        # Calculate confidence based on coverage and term counts
        confidence = (coverage * 0.6) + (max(support_count, contradict_count) / (support_count + contradict_count + 1) * 0.4)
        
        return {
            "relevance_score": coverage,
            "key_term_occurrences": word_occurrences,
            "total_occurrences": total_occurrences,
            "stance": stance,
            "supporting_term_count": support_count,
            "contradicting_term_count": contradict_count,
            "confidence": confidence
        }

    def run(self, state: VideoVerificationState) -> VideoVerificationState:
        """Run the video verification agent on the given state."""
        try:
            # Check if we have the necessary data in the state
            if not state.get("video_url"):
                logging.error("Missing video URL in state")
                state["verification_results"] = {
                    "status": "failed",
                    "error": "Missing video URL",
                    "transcript_error": True,
                    "verified_claims": [],
                    "unverified_claims": [],
                    "misleading_claims": [],
                    "confidence_scores": {},
                    "sources": [],
                    "verification_summary": "Verification could not be performed due to missing video URL."
                }
                return state
                
            if not state.get("video_transcript"):
                logging.error("Missing video transcript in state")
                state["verification_results"] = {
                    "status": "failed",
                    "error": "Missing video transcript",
                    "transcript_error": True,
                    "verified_claims": [],
                    "unverified_claims": [],
                    "misleading_claims": [],
                    "confidence_scores": {},
                    "sources": [],
                    "verification_summary": "Verification could not be performed due to missing transcript data."
                }
                return state
                
            # Process the video verification
            updated_state = self._process_video_verification(state)
            
            # Ensure we have a current_agent field set
            if "verification_results" in updated_state and updated_state["verification_results"].get("status") != "failed":
                updated_state["current_agent"] = "video_verification_completed"
                
            return updated_state
        except Exception as e:
            logging.error(f"Error in video verification agent: {str(e)}")
            state["verification_results"] = {
                "status": "failed",
                "error": str(e),
                "transcript_error": False,
                "verified_claims": [],
                "unverified_claims": [],
                "misleading_claims": [],
                "confidence_scores": {},
                "sources": [],
                "verification_summary": f"Verification failed due to an error: {str(e)}"
            }
            return state

    def verify_video(self, video_url: str) -> Dict[str, Any]:
        """Standalone method to verify a video without requiring a specific claim.
        
        This is useful for general video verification without a specific claim to check.
        
        Args:
            video_url: URL of the video to verify
            
        Returns:
            Dictionary containing verification results
        """
        # Initialize state with video URL but no specific claim
        initial_state = {
            "claim": "",  # No specific claim
            "video_url": video_url,
            "video_transcript": {},
            "video_research_results": {},
            "verification_results": {},
            "messages": [{
                "agent": "video_verification",
                "content": f"Starting verification of video: {video_url}",
                "timestamp": datetime.now().isoformat()
            }]
        }
        
        # Run the verification process
        final_state = self.run(initial_state)
        
        # Return the verification results
        return final_state.get("verification_results", {})