#!/usr/bin/env python3
"""
Test script for the SupervisorAgent workflow

This script tests whether the SupervisorAgent correctly progresses through
all agents in the fact-checking workflow without getting stuck in a loop.
"""

import logging
import sys
from datetime import datetime
from src.agents.supervisor import SupervisorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('supervisor_test.log')
    ]
)

def test_workflow():
    """Test the SupervisorAgent workflow"""
    logging.info("Creating SupervisorAgent")
    agent = SupervisorAgent()
    
    # Mock the agent run methods to avoid actual API calls
    def mock_research(state):
        logging.info("Mock research agent running")
        state["research_results"] = {"mock": "research data", "timestamp": datetime.now().isoformat()}
        return state
    
    def mock_verification(state):
        logging.info("Mock verification agent running")
        state["verification_results"] = {"mock": "verification data", "timestamp": datetime.now().isoformat()}
        return state
    
    def mock_validation(state):
        logging.info("Mock validation agent running")
        state["validation_results"] = {"mock": "validation data", "timestamp": datetime.now().isoformat()}
        return state
    
    def mock_summary(state):
        logging.info("Mock summary agent running")
        state["final_summary"] = {"mock": "summary data", "timestamp": datetime.now().isoformat()}
        return state
    
    # Replace actual agent runs with mocks
    agent.research_agent.run = mock_research
    agent.verification_agent.run = mock_verification
    agent.validation_agent.run = mock_validation
    agent.summary_agent.run = mock_summary
    
    # Run the workflow
    logging.info("Starting test workflow")
    result = agent.run_fact_check("Test claim")
    
    # Check if all agents were executed
    all_completed = all(k in result and result[k] for k in [
        "research_results", "verification_results", "validation_results", "final_summary"
    ])
    
    if all_completed:
        logging.info("SUCCESS: All agents in the workflow were executed")
        print("\n✅ Test passed: All agents in the workflow were executed successfully")
    else:
        missing = [k for k in ["research_results", "verification_results", "validation_results", "final_summary"] 
                  if k not in result or not result[k]]
        logging.error(f"FAILURE: Some agents were not executed: {missing}")
        print(f"\n❌ Test failed: The following agents were not executed: {missing}")
    
    return result

if __name__ == "__main__":
    print("Testing SupervisorAgent workflow...\n")
    result = test_workflow()
    
    # Print messages from the workflow
    print("\nWorkflow messages:")
    for msg in result.get("messages", []):
        agent = msg.get("agent", "unknown")
        content = msg.get("content", "")
        print(f"- {agent}: {content}")