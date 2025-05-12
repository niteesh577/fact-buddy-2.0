import requests
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test server URL
BASE_URL = "http://localhost:5000"

def test_video_endpoint():
    """Test the video fact-checking endpoint with proper headers"""
    
    # Test data
    test_data = {
        "claim": "The Earth is flat",
        "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    }
    
    # Set proper headers
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Make the request with proper headers
        logger.info("Testing video endpoint with proper Content-Type header...")
        response = requests.post(
            f"{BASE_URL}/video", 
            data=json.dumps(test_data),
            headers=headers
        )
        
        # Log response
        logger.info(f"Status code: {response.status_code}")
        logger.info(f"Response: {response.text}")
        
        # Test without proper Content-Type header
        logger.info("Testing video endpoint WITHOUT proper Content-Type header...")
        response_no_header = requests.post(
            f"{BASE_URL}/video", 
            data=json.dumps(test_data)
        )
        
        # Log response
        logger.info(f"Status code: {response_no_header.status_code}")
        logger.info(f"Response: {response_no_header.text}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing video endpoint: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting video endpoint test")
    success = test_video_endpoint()
    if success:
        logger.info("Test completed")
    else:
        logger.error("Test failed")