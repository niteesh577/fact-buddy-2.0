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

class VideoSummaryOutput(BaseModel):
    verdict: str = Field(description="Final verdict on the claims in the video")
    confidence: float = Field(description="Overall confidence in the conclusion, ranging from 0.0 to 1.0")
    key_findings: List[Dict[str, Any]] = Field(description="Key factual claims or statements identified in the video, with timestamps and relevance scores")
    sources: List[Dict[str, Any]] = Field(description="List of sources used for verification, including URL, reliability, and trust score")
    summary: str = Field(description="A concise summary of the video's content and the verification findings")
    transcript_verification: Dict[str, int] = Field(description="Counts of supporting and contradicting points found during verification")
    video_analysis: Dict[str, Any] = Field(description="Analysis of the video transcript, including quality, word count, and key phrases")

class VideoSummaryState(TypedDict):
    """State for the video summary agent."""
    claim: str
    video_url: str
    video_transcript: Dict[str, Any]
    video_research_results: Dict[str, Any]
    verification_results: Dict[str, Any]
    validation_results: Dict[str, Any]
    final_summary: Dict[str, Any]
    messages: List[Dict[str, Any]]

class VideoSummaryAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert video analysis agent that:
                      1. Synthesizes video transcript and research findings
                      2. Provides evidence-based conclusions about claims in videos
                      3. Includes proper citations with timestamps
                      4. Maintains objectivity in reporting
                      
                      Generate a comprehensive summary of the video content and fact-check the claims."""),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
        
        # Create tools for the agent
        self.tools = [
            Tool(
                name="transcript_analyzer",
                func=self.analyze_transcript,
                description="Analyze the video transcript for factual claims and evidence"
            ),
            Tool(
                name="video_verdict_determiner",
                func=self.determine_video_verdict,
                description="Determine the final verdict on claims in the video based on all available evidence"
            )
        ]
        
        # Create a LangGraph-compatible agent
        self.agent = create_react_agent(
            self.llm,
            self.tools
        )
        
        # Create the summary graph
        self.graph = self._create_summary_graph()
        
        self.output_parser = PydanticOutputParser(pydantic_object=VideoSummaryOutput)

    def _create_summary_graph(self):
        """Create a LangGraph workflow for the video summary process."""
        workflow = StateGraph(VideoSummaryState)
        
        # Add nodes for analysis, verdict determination, and final summary generation
        workflow.add_node("analyze_transcript", self._analyze_transcript_node)
        workflow.add_node("determine_verdict", self._determine_verdict_node)
        workflow.add_node("generate_summary", self._generate_summary_node)
        
        # Define edges
        workflow.add_edge("analyze_transcript", "determine_verdict")
        workflow.add_edge("determine_verdict", "generate_summary")
        workflow.add_edge("generate_summary", END)
        
        # Set the entry point
        workflow.set_entry_point("analyze_transcript")
        
        return workflow.compile()

    def _analyze_transcript_node(self, state: VideoSummaryState) -> VideoSummaryState:
        """Node to analyze the video transcript."""
        logging.info("Analyzing video transcript...")
        transcript_analysis_result = self.analyze_transcript({"video_transcript": state.get("video_transcript")})
        state["video_analysis_results"] = transcript_analysis_result
        state.setdefault("messages", []).append({
            "agent": "video_summary",
            "content": "Transcript analysis complete.",
            "timestamp": datetime.now().isoformat()
        })
        return state

    def _determine_verdict_node(self, state: VideoSummaryState) -> VideoSummaryState:
        """Node to determine the video verdict."""
        logging.info("Determining video verdict...")
        # Ensure all input dictionaries are at least empty dictionaries, not None
        verdict_input = {
            "video_research_results": state.get("video_research_results") or {},
            "verification_results": state.get("verification_results") or {},
            "validation_results": state.get("validation_results") or {},
            "video_transcript": state.get("video_transcript") or {},
            "video_analysis_results": state.get("video_analysis_results") or {}
        }
        verdict_result = self.determine_video_verdict(verdict_input)
        state["verdict_results"] = verdict_result
        state.setdefault("messages", []).append({
            "agent": "video_summary",
            "content": f"Verdict determined: {verdict_result.get('verdict', 'N/A')}",
            "timestamp": datetime.now().isoformat()
        })
        return state

    def _generate_summary_node(self, state: VideoSummaryState) -> VideoSummaryState:
        """Node to generate the final summary."""
        logging.info("Generating final video summary...")
        
        # Consolidate information for the final summary
        verdict_results = state.get("verdict_results", {})
        video_analysis_results = state.get("video_analysis_results", {})
        verification_results = state.get("verification_results", {})
        validation_results = state.get("validation_results", {})
        video_research_results = state.get("video_research_results", {})
        transcript_data = state.get("video_transcript", {})
        
        # Check if we have a transcript error
        transcript_error = verification_results.get("transcript_error")
        
        # Also check if transcript data itself indicates an error
        if not transcript_error and transcript_data.get("status") == "failed":
            transcript_error = transcript_data.get("error")
            logging.warning(f"Transcript extraction failed: {transcript_error}")
        
        # Default empty values for required fields
        key_findings = []
        sources = []
        
        # If we have valid verification data, extract key findings
        if not transcript_error and verification_results:
            # Select key findings based on verification and transcript
            key_findings = self._select_key_findings(
                verification_results.get("verified_claims", []), 
                verification_results.get("misleading_claims", []), 
                transcript_data.get("segments", [])
            )

            # Prepare sources list
            if "sources" in verification_results:
                for source_info in verification_results["sources"]:
                    sources.append({
                        "url": source_info.get("url", "N/A"),
                        "reliability": source_info.get("credibility_assessment", {}).get("level", "Unknown"),
                        "trust_score": source_info.get("credibility_assessment", {}).get("score", 0.0)
                    })

        # Construct the final summary object according to VideoSummaryOutput schema
        final_summary_data = {
            "verdict": verdict_results.get("verdict", "Inconclusive") if not transcript_error else None,
            "confidence": verdict_results.get("confidence_score", 0.5) if not transcript_error else None,
            "key_findings": key_findings,
            "sources": sources,
            "summary": verdict_results.get("evidence_summary", "Summary not available.") if not transcript_error else None,
            "transcript_verification": {
                "supporting_points": len(verification_results.get("verified_claims", [])),
                "contradicting_points": len(verification_results.get("misleading_claims", []))
            },
            "video_analysis": {
                "transcript_quality": "unknown" if transcript_error else video_analysis_results.get("analysis", {}).get("transcript_quality", "unknown"),
                "word_count": 0 if transcript_error else video_analysis_results.get("analysis", {}).get("word_count", 0),
                "key_phrases": [] if transcript_error else video_analysis_results.get("analysis", {}).get("key_phrases", [])
            }
        }

        # Validate with Pydantic model before assigning
        try:
            validated_summary = VideoSummaryOutput(**final_summary_data)
            state["final_summary"] = validated_summary.dict()
            state.setdefault("messages", []).append({
                "agent": "video_summary",
                "content": "Final summary generated successfully.",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logging.error(f"Error validating final summary: {e}")
            state["final_summary"] = {"error": f"Failed to generate valid summary: {e}", **final_summary_data} # Include raw data on error
            state.setdefault("messages", []).append({
                "agent": "video_summary",
                "content": f"Error generating final summary: {e}",
                "timestamp": datetime.now().isoformat()
            })
            
        return state

    def analyze_transcript(self, content):
        """Analyze the video transcript for factual claims and evidence."""
        transcript_data = content.get("video_transcript", {})
        transcript_text = transcript_data.get("transcript", "")
        
        # Extract segments if available
        segments = transcript_data.get("segments", [])
        
        # If no transcript text is available, create a default analysis with empty values
        if not transcript_text:
            return {
                "analysis": {
                    "word_count": 0,
                    "sentence_count": 0,
                    "key_phrases": [],
                    "has_timestamps": len(segments) > 0,
                    "transcript_quality": "unknown"
                },
                "key_segments": segments[:5] if segments else []
            }
        
        # Analyze the transcript content
        # In a production environment, this would use more sophisticated NLP techniques
        
        # Simple analysis of transcript length and structure
        word_count = len(transcript_text.split())
        sentence_count = len([s for s in transcript_text.split(".") if s.strip()])
        
        # Extract potential key phrases
        key_phrases = []
        sentences = transcript_text.split(".")
        for sentence in sentences:
            # Simple heuristic: sentences with numbers or specific keywords might be factual claims
            if any(char.isdigit() for char in sentence) or any(keyword in sentence.lower() for keyword in 
                                                             ["percent", "study", "research", "according", "evidence", "fact"]):
                key_phrases.append(sentence.strip())
        
        # If no key phrases were found but we have sentences, use the first few sentences
        if not key_phrases and sentences:
            for sentence in sentences:
                if len(sentence.strip()) > 20:  # Only use reasonably long sentences
                    key_phrases.append(sentence.strip())
                    if len(key_phrases) >= 3:
                        break
        
        # Limit to top 5 key phrases
        key_phrases = key_phrases[:5]
        
        # Create analysis summary
        analysis = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "key_phrases": key_phrases,
            "has_timestamps": len(segments) > 0,
            "transcript_quality": "high" if word_count > 500 else "medium" if word_count > 100 else "low"
        }
        
        return {
            "analysis": analysis,
            "key_segments": segments[:10] if segments else []  # Return up to 10 segments if available
        }

    def _select_key_findings(self, verified_claims, misleading_claims, segments, max_findings=5):
        """Select diverse and meaningful key findings."""
        potential_findings = []

        # Add verified and misleading claims as potential findings
        for claim in verified_claims + misleading_claims:
            finding_text = claim.get("claim_text", "")
            timestamp = claim.get("timestamp")
            relevance = claim.get("confidence", 0.5) # Use claim confidence as relevance proxy
            
            # Try to format timestamp nicely
            if timestamp:
                try:
                    timestamp_str = f"{int(timestamp // 60)}m{int(timestamp % 60)}s"
                except:
                    timestamp_str = str(timestamp) # Fallback
            else:
                timestamp_str = "N/A"

            potential_findings.append({
                "finding": finding_text,
                "relevance": f"{relevance:.2f}", # Format relevance score
                "timestamp": timestamp_str
            })

        # If not enough findings from claims, supplement with key segments
        if len(potential_findings) < max_findings and segments:
            num_needed = max_findings - len(potential_findings)
            # Prioritize segments with longer text
            sorted_segments = sorted(segments, key=lambda s: len(s.get("text", "")), reverse=True)
            
            for segment in sorted_segments[:num_needed]:
                text = segment.get("text", "")
                start_time = segment.get("start")
                
                if text and start_time is not None:
                    # Avoid adding duplicates if segment text matches an existing finding
                    if any(f["finding"] in text or text in f["finding"] for f in potential_findings):
                        continue
                        
                    try:
                        timestamp_str = f"{int(start_time // 60)}m{int(start_time % 60)}s"
                    except:
                        timestamp_str = str(start_time)
                        
                    potential_findings.append({
                        "finding": text.strip(),
                        "relevance": "Medium", # Assign default relevance for segments
                        "timestamp": timestamp_str
                    })
                    if len(potential_findings) >= max_findings:
                        break

        # Ensure diversity (simple approach: limit findings)
        # A more complex approach could involve topic modeling or clustering
        selected_findings = potential_findings[:max_findings]
        
        # Sort by timestamp if possible
        def get_sort_key(finding):
            ts_str = finding.get("timestamp", "N/A")
            if 'm' in ts_str and 's' in ts_str:
                try:
                    minutes, seconds = ts_str.replace('m', '').replace('s', '').split()
                    return int(minutes) * 60 + int(seconds)
                except:
                    return float('inf') # Put N/A at the end
            elif ts_str != "N/A":
                 try:
                     return float(ts_str)
                 except:
                     return float('inf')
            return float('inf')

        selected_findings.sort(key=get_sort_key)

        return selected_findings

    def determine_video_verdict(self, content):
        """Determine the final verdict on claims in the video based on all available evidence."""
        # Extract all necessary data
        video_research_results = content.get("video_research_results", {})
        verification_results = content.get("verification_results", {})
        validation_results = content.get("validation_results", {})
        transcript_data = content.get("video_transcript", {})
        video_analysis_results = content.get("video_analysis_results", {})
        
        # Get transcript quality
        transcript_quality = video_analysis_results.get("analysis", {}).get("transcript_quality", "unknown")
        transcript_word_count = video_analysis_results.get("analysis", {}).get("word_count", 0)

        # Get verification details
        verified_claims = verification_results.get("verified_claims", [])
        unverified_claims = verification_results.get("unverified_claims", [])
        misleading_claims = verification_results.get("misleading_claims", [])
        overall_credibility = verification_results.get("overall_credibility", 0.5) # From source analysis

        # Get validation details (ensure validation_results is not None)
        validation_results = validation_results or {}
        validation_confidence = validation_results.get("confidence_score", 0.5) # From bias/fallacy analysis
        biases = validation_results.get("biases", [])
        fallacies = validation_results.get("logical_fallacies", [])

        # --- Verdict Logic --- 
        num_verified = len(verified_claims)
        num_misleading = len(misleading_claims)
        num_unverified = len(unverified_claims)
        total_claims = num_verified + num_misleading + num_unverified

        verdict = "Inconclusive"
        if total_claims == 0:
            if transcript_word_count < 50: # Very short or no transcript
                verdict = "No claims found (short transcript)"
            else:
                verdict = "No verifiable claims identified"
        elif num_misleading > num_verified and num_misleading >= total_claims * 0.4:
            verdict = "Contains Misleading Information"
        elif num_verified > num_misleading and num_verified >= total_claims * 0.6:
            verdict = "Largely Accurate"
        elif num_verified > 0 and num_misleading == 0:
             verdict = "Accurate (based on verified claims)"
        elif num_misleading > 0 and num_verified == 0:
             verdict = "Misleading (based on identified claims)"
        elif num_verified > 0 or num_misleading > 0:
            verdict = "Mixed Accuracy"
        elif num_unverified > 0:
            verdict = "Claims Require Verification"

        # --- Confidence Score Logic --- 
        # Start with validation confidence (bias/fallacy check)
        confidence_score = validation_confidence
        
        # Adjust based on verification results
        if total_claims > 0:
            accuracy_ratio = num_verified / total_claims if total_claims > 0 else 0
            misleading_ratio = num_misleading / total_claims if total_claims > 0 else 0
            # Boost confidence for high accuracy, penalize for misleading info
            confidence_score = confidence_score * (1 + accuracy_ratio - misleading_ratio * 1.5)
        
        # Adjust based on transcript quality
        if transcript_quality == "high":
            confidence_score *= 1.1 # Higher confidence with good transcript
        elif transcript_quality == "low":
            confidence_score *= 0.8 # Lower confidence with poor transcript
        
        # Ensure score is within [0, 1]
        confidence_score = max(0.0, min(1.0, confidence_score))

        # --- Evidence Summary Logic --- 
        summary_parts = []
        if num_verified > 0:
            summary_parts.append(f"{num_verified} claim(s) verified as accurate.")
        if num_misleading > 0:
            summary_parts.append(f"{num_misleading} claim(s) identified as misleading or false.")
        if num_unverified > 0:
            summary_parts.append(f"{num_unverified} claim(s) could not be verified.")
        if not summary_parts:
             summary_parts.append("No specific claims were analyzed or verified.")

        if biases:
            summary_parts.append(f"Potential biases identified: {len(biases)}.")
        if fallacies:
            summary_parts.append(f"Potential logical fallacies identified: {len(fallacies)}.")
            
        evidence_summary = " ".join(summary_parts)
        if transcript_quality != 'high' and total_claims > 0:
             evidence_summary += f" Verification based on a transcript of '{transcript_quality}' quality."
        elif transcript_word_count < 100 and total_claims == 0:
             evidence_summary += f" Analysis limited by very short transcript ({transcript_word_count} words)."

        return {
            "verdict": verdict,
            "confidence_score": round(confidence_score, 3),
            "evidence_summary": evidence_summary,
            "supporting_evidence_count": num_verified,
            "contradicting_evidence_count": num_misleading,
            "unverified_evidence_count": num_unverified
        }

    def _process_video_summary(self, state: VideoSummaryState) -> VideoSummaryState:
        """Process all results to generate the final video summary."""
        # This method might become redundant if using the graph structure above
        # Keeping it for potential direct calls or alternative flows
        logging.info("Processing video summary...")
        try:
            # Ensure all required inputs are present
            required_keys = ["video_transcript", "video_research_results", 
                             "verification_results", "validation_results"]
            if not all(key in state for key in required_keys):
                raise ValueError(f"Missing required state keys for video summary: {required_keys}")

            # 1. Analyze Transcript (if not already done)
            if "video_analysis_results" not in state:
                 state = self._analyze_transcript_node(state)
            
            # 2. Determine Verdict (if not already done)
            if "verdict_results" not in state:
                 state = self._determine_verdict_node(state)

            # 3. Generate Final Summary (if not already done)
            if "final_summary" not in state:
                 state = self._generate_summary_node(state)

            state.setdefault("messages", []).append({
                "agent": "video_summary",
                "content": "Video summary processing complete.",
                "timestamp": datetime.now().isoformat()
            })
            return state

        except Exception as e:
            logging.error(f"Error processing video summary: {str(e)}")
            state["final_summary"] = {"error": f"Failed to process video summary: {str(e)}"}
            state.setdefault("messages", []).append({
                "agent": "video_summary",
                "content": f"Error during video summary processing: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
            return state

    def run(self, state: VideoSummaryState) -> VideoSummaryState:
        """Run the video summary agent on the provided state."""
        try:
            # Check if we have the necessary data in the state
            if not state.get("video_url"):
                logging.error("Missing video URL in state")
                state["final_summary"] = {
                    "status": "failed",
                    "error": "Missing video URL",
                    "verdict": "Inconclusive",
                    "confidence": 0.0,
                    "key_findings": [],
                    "sources": [],
                    "summary": "Summary could not be generated due to missing video URL.",
                    "transcript_verification": {"supporting_points": 0, "contradicting_points": 0},
                    "video_analysis": {"transcript_quality": "unknown", "word_count": 0, "key_phrases": []}
                }
                return state
                
            # Check if verification results are available
            if not state.get("verification_results"):
                logging.error("Missing verification results in state")
                state["final_summary"] = {
                    "status": "failed",
                    "error": "Missing verification results",
                    "verdict": "Inconclusive",
                    "confidence": 0.0,
                    "key_findings": [],
                    "sources": [],
                    "summary": "Summary could not be generated due to missing verification results.",
                    "transcript_verification": {"supporting_points": 0, "contradicting_points": 0},
                    "video_analysis": {"transcript_quality": "unknown", "word_count": 0, "key_phrases": []}
                }
                return state
                
            # Check if transcript data is available
            if not state.get("video_transcript"):
                logging.error("Missing transcript data in state")
                state["final_summary"] = {
                    "status": "failed",
                    "error": "Missing transcript data",
                    "verdict": "Inconclusive",
                    "confidence": 0.0,
                    "key_findings": [],
                    "sources": [],
                    "summary": "Summary could not be generated due to missing transcript data.",
                    "transcript_verification": {"supporting_points": 0, "contradicting_points": 0},
                    "video_analysis": {"transcript_quality": "unknown", "word_count": 0, "key_phrases": []}
                }
                return state
                
            # Process through the graph
            updated_state = self.graph.invoke(state)
            
            # Ensure we have a current_agent field set
            if "final_summary" in updated_state:
                updated_state["current_agent"] = "completed"
                
            return updated_state
        except Exception as e:
            logging.error(f"Error in video summary agent: {str(e)}")
            state["final_summary"] = {
                "status": "failed",
                "error": str(e),
                "verdict": "Inconclusive",
                "confidence": 0.0,
                "key_findings": [],
                "sources": [],
                "summary": f"Summary generation failed due to an error: {str(e)}",
                "transcript_verification": {"supporting_points": 0, "contradicting_points": 0},
                "video_analysis": {"transcript_quality": "unknown", "word_count": 0, "key_phrases": []}
            }
            return state

# Example Usage (for testing)
if __name__ == '__main__':
    import asyncio

    # Example input
    test_claim = "Analyze the claims made in this video about climate change."
    test_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ" # Example URL

    # Instantiate the agent
    video_summary_agent = VideoSummaryAgent()

    # Run the agent asynchronously
    async def main():
        try:
            # Simulate previous steps providing necessary state data
            # In a real scenario, these would come from other agents
            simulated_state = {
                "claim": test_claim,
                "video_url": test_video_url,
                "video_transcript": {
                    "status": "success",
                    "transcript": "This is a sample transcript. Claim 1 is true. Claim 2 is false. Claim 3 needs checking.",
                    "segments": [
                        {"start": 0, "end": 5, "text": "This is a sample transcript."},
                        {"start": 6, "end": 10, "text": "Claim 1 is true."},
                        {"start": 11, "end": 15, "text": "Claim 2 is false."},
                        {"start": 16, "end": 20, "text": "Claim 3 needs checking."}
                    ]
                },
                "video_research_results": {
                     "factual_claims": [{"claim_text": "Claim 1 is true.", "timestamp": 8.0}, {"claim_text": "Claim 2 is false.", "timestamp": 13.0}],
                     "video_info": {"info": {"view_count": 1000, "like_count": 100, "comment_count": 10, "published_at": "2023-01-01"}}
                },
                "verification_results": {
                    "verified_claims": [{"claim_text": "Claim 1 is true.", "confidence": 0.9, "timestamp": 8.0}],
                    "misleading_claims": [{"claim_text": "Claim 2 is false.", "confidence": 0.8, "timestamp": 13.0}],
                    "unverified_claims": [{"claim_text": "Claim 3 needs checking.", "confidence": 0.5, "timestamp": 18.0}],
                    "overall_credibility": 0.7,
                    "sources": [
                        {"url": "http://example.com/source1", "credibility_assessment": {"level": "High", "score": 0.9}},
                        {"url": "http://example.com/source2", "credibility_assessment": {"level": "Medium", "score": 0.6}}
                    ]
                },
                "validation_results": {
                    "biases": [],
                    "logical_fallacies": [],
                    "cross_references": ["http://example.com/source1"],
                    "confidence_score": 0.85
                },
                "messages": []
            }
            
            # Invoke the graph with the simulated state
            final_state = await video_summary_agent.graph.ainvoke(simulated_state)
            print("Final Summary:")
            import json
            print(json.dumps(final_state.get("final_summary", {}), indent=2))

        except Exception as e:
            print(f"An error occurred: {e}")

    # Run the async main function
    asyncio.run(main())