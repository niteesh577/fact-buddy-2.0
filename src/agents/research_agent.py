from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from typing import Dict, Any, Optional, List, Tuple, TypedDict
from src.tools.search import SearchTool
from src.tools.web_scraper import WebScraperTool
from src.database.chroma_store import ChromaStore
from datetime import datetime
import logging

from langchain_core.tools import tool
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

logging.basicConfig(level=logging.INFO)


class ResearchState(TypedDict):
    """State for the research agent."""
    claim: str
    source: Optional[str]
    research_results: Dict[str, Any]
    messages: List[Dict[str, Any]]


class ResearchAgent:
    def __init__(self):
        # Initialize the language model engine with fixed temperature
        self.llm = ChatGroq(
            temperature=0,
            model_name="mixtral-8x7b-32768"
        )
        # Set up tools for web search and web scraping
        self.search_tool = SearchTool()
        self.scraper_tool = WebScraperTool()
        self.db = ChromaStore()

        self.tools = [
            Tool(
                name="web_search",
                func=self.search_tool.search,
                description="Search the web for information about a claim"
            ),
            Tool(
                name="web_scraper",
                func=self.scraper_tool.scrape,
                description="Scrape and extract content from a webpage URL"
            )
        ]

        # Create a prompt template for the agent
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research agent tasked with gathering information about claims. "
                      "Use the search tool to find relevant information and the scraper to extract "
                      "detailed content from reliable sources."),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])

        # Create a LangGraph-compatible agent
        self.agent = create_react_agent(
            self.llm,
            self.tools
        )
        
        # Create the research graph
        self.graph = self._create_research_graph()

    def _create_research_graph(self):
        """Create a LangGraph workflow for the research process."""
        workflow = StateGraph(ResearchState)
        
        # Add the research node that processes the claim
        workflow.add_node("research", self._process_research)
        
        # Set the entry point and connect to END
        workflow.add_edge("research", END)
        workflow.set_entry_point("research")
        
        return workflow.compile()

    def _process_research(self, state: ResearchState) -> ResearchState:
        """Process the research for a claim."""
        try:
            # Validate input state
            error = self.validate_input(state)
            if error:
                raise ValueError(error)

            claim = state["claim"]
            source = state.get("source")
            research_data = {}

            if source:
                # If a source URL is provided, scrape that source
                try:
                    source_content = self.scraper_tool.scrape(source)
                    if source_content.get("status") == "failed":
                        logging.error(f"Failed to scrape source: {source_content.get('error')}")
                        research_data["source_error"] = source_content.get("error")
                    else:
                        # Log the content length to help with debugging
                        content_text = source_content.get("content", "")
                        logging.info(f"Scraped source content length: {len(content_text)} characters")
                        if len(content_text) < 10:  # Arbitrary small number to detect failed scrapes
                            logging.warning(f"Very short content scraped from {source}: '{content_text}'")
                        research_data["source_content"] = source_content
                except Exception as e:
                    logging.error(f"Error scraping source {source}: {str(e)}")
                    research_data["source_error"] = str(e)
            else:
                # If no source is provided, perform a single web search and then scrape all URLs
                try:
                    search_query = f"fact check: {claim}"
                    search_results = self.search_tool.search(search_query)
                    if search_results.get("status") == "failed":
                        logging.error(f"Search failed: {search_results.get('error')}")
                        research_data["search_error"] = search_results.get("error")
                    else:
                        research_data["search_results"] = search_results
                        organic_results = search_results.get("organic_results", [])
                        scraped_results = []
                        for result in organic_results:
                            url = result.get("link")
                            if url:
                                try:
                                    logging.info(f"Attempting to scrape URL: {url}")
                                    scraped_content = self.scraper_tool.scrape(url)
                                    
                                    # Check if scraping was successful and content is meaningful
                                    content_text = scraped_content.get("content", "")
                                    logging.info(f"Scraped content length from {url}: {len(content_text)} characters")
                                    
                                    if len(content_text) < 10:  # Arbitrary small number to detect failed scrapes
                                        logging.warning(f"Very short content scraped from {url}: '{content_text}'")
                                        scraped_results.append({
                                            "url": url,
                                            "title": result.get("title", ""),
                                            "error": "Content too short, likely scraping failed",
                                            "content": scraped_content
                                        })
                                    else:
                                        scraped_results.append({
                                            "url": url,
                                            "title": result.get("title", ""),
                                            "content": scraped_content
                                        })
                                        logging.info(f"Successfully scraped URL: {url}")
                                except Exception as e:
                                    logging.error(f"Error scraping URL {url}: {str(e)}")
                                    scraped_results.append({
                                        "url": url,
                                        "error": str(e)
                                    })
                        research_data["scraped_results"] = scraped_results
                except Exception as e:
                    logging.error(f"Error performing search: {str(e)}")
                    research_data["search_error"] = str(e)

            # Store the gathered research data in ChromaDB for future reference
            try:
                research_data["timestamp"] = datetime.now().isoformat()
                self.db.store_research_results(claim, research_data)
            except Exception as e:
                logging.error(f"Error storing research results: {str(e)}")
                research_data["storage_error"] = str(e)

            state["research_results"] = research_data
            state.setdefault("messages", []).append({
                "agent": "research",
                "content": "Research completed" + (
                    " with some errors" if any(k.endswith("_error") for k in research_data.keys()) else ""
                ),
                "timestamp": datetime.now().isoformat()
            })

            return state

        except Exception as e:
            logging.error(f"Critical error in research agent: {str(e)}")
            state.setdefault("messages", []).append({
                "agent": "research",
                "content": f"Research failed: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            })
            return state

    def validate_input(self, state: Dict[str, Any]) -> Optional[str]:
        if not state:
            return "State cannot be empty"
        if "claim" not in state:
            return "Claim is required in state"
        if not isinstance(state["claim"], str) or not state["claim"].strip():
            return "Claim must be a non-empty string"
        return None

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the research agent on the given state."""
        # Initialize the research state
        research_state = {
            "claim": state["claim"],
            "source": state.get("source"),
            "research_results": {},
            "messages": state.get("messages", [])
        }
        
        # Run the research graph
        result = self.graph.invoke(research_state)
        
        # Update the original state with research results
        state["research_results"] = result["research_results"]
        state["messages"] = result["messages"]
        
        return state
