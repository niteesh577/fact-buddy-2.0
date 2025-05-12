import logging
from src.agents.supervisor import SupervisorAgent

# Configure logging to see detailed workflow progression
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create the supervisor agent
agent = SupervisorAgent()

# Run a fact check with a test claim
print("Starting fact-check workflow test...")
result = agent.run_fact_check('Is eating cow dung healthy?')

# Verify that all agents were executed
print("\nWorkflow execution completed!")
print("\nVerifying agent execution:")
print(f"Research results present: {'research_results' in result}")
print(f"Verification results present: {'verification_results' in result}")
print(f"Validation results present: {'validation_results' in result}")
print(f"Final summary present: {'final_summary' in result}")

# Print the final messages from each agent
print("\nAgent messages:")
for message in result.get('messages', []):
    print(f"- {message.get('agent', 'unknown')}: {message.get('content', '')}")