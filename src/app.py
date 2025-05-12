from flask import Flask, request, jsonify
from src.agents.supervisor import SupervisorAgent
from src.agents.video_supervisor import VideoSupervisorAgent
from dotenv import load_dotenv
import os
import logging
import json
from flask_cors import CORS

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("fact_check.log")
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for required API keys
if not os.getenv("GROQ_API_KEY"):
    logger.warning("GROQ_API_KEY not found in environment variables")

if not os.getenv("SERPAPI_API_KEY"):
    logger.warning("SERPAPI_API_KEY not found in environment variables")

app = Flask(__name__)

# Configure CORS properly with more permissive settings
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})


@app.route('/fact-check', methods=['POST'])
def fact_check():
    try:
        data = request.json
        claim = data.get('claim')
        source = data.get('source', None)

        if not claim:
            return jsonify({"error": "Claim is required"}), 400

        logger.info(f"Processing fact check for claim: {claim}")
        if source:
            logger.info(f"Using provided source: {source}")
        
        # Create supervisor agent and run fact check
        supervisor = SupervisorAgent()
        logger.info("Starting fact-checking workflow")
        result = supervisor.run_fact_check(claim, source)
        
        # Log the completion of each agent's work
        for message in result.get("messages", []):
            agent = message.get("agent", "unknown")
            content = message.get("content", "")
            timestamp = message.get("timestamp", "")
            if message.get("error", False):
                logger.error(f"[{agent.upper()}] {content} at {timestamp}")
            else:
                logger.info(f"[{agent.upper()}] {content} at {timestamp}")
        
        # Log the final verdict
        final_summary = result.get("final_summary", {})
        verdict = final_summary.get("verdict", "Unknown")
        confidence = final_summary.get("confidence_level", 0.0)
        logger.info(f"Final verdict: {verdict} (confidence: {confidence:.2f})")
        
        # Format the response for better readability
        formatted_response = {
            "verdict": final_summary.get("verdict"),
            "confidence": final_summary.get("confidence_level"),
            "summary": final_summary.get("evidence_summary"),
            "key_findings": [
                {
                    "finding": finding.get("finding"),
                    "source": finding.get("source"),
                    "relevance": "High" if i < 2 else "Medium"
                } for i, finding in enumerate(final_summary.get("key_findings", []))
            ],
            "sources": [
                {
                    "url": citation.get("source"),
                    "trust_score": float(citation.get("trust_score", "0")) if citation.get("trust_score", "N/A").replace(".", "", 1).isdigit() else 0.0,
                    "reliability": "High" if float(citation.get("trust_score", "0")) > 0.7 else "Medium" if float(citation.get("trust_score", "0")) > 0.4 else "Low"
                } for citation in final_summary.get("citations", []) if not citation.get("source").endswith("_error") and citation.get("source") != "timestamp"
            ]
        }
        
        # Log the formatted response
        logger.info(f"Returning formatted response with {len(formatted_response['key_findings'])} key findings and {len(formatted_response['sources'])} sources")
        
        return jsonify(formatted_response)

    except Exception as e:
        logger.error(f"Error processing fact check: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/image', methods=['POST'])
def image_check():
    try:
        # Check if image file is in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        
        # Check if a filename is provided
        if image_file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Save the image temporarily
        temp_path = os.path.join(os.getcwd(), "temp_images")
        os.makedirs(temp_path, exist_ok=True)
        image_path = os.path.join(temp_path, image_file.filename)
        image_file.save(image_path)
        
        logger.info(f"Processing image fact check for file: {image_file.filename}")
        
        # Additional text context can be provided
        context = request.form.get('context', '')
        if context:
            logger.info(f"Additional context provided: {context}")
        
        # Create image supervisor agent and run fact check
        from src.agents.image_supervisor import ImageSupervisorAgent
        image_supervisor = ImageSupervisorAgent()
        logger.info("Starting image fact-checking workflow")
        result = image_supervisor.run_image_fact_check(image_path, context)
        
        # Log the completion of each agent's work
        for message in result.get("messages", []):
            agent = message.get("agent", "unknown")
            content = message.get("content", "")
            timestamp = message.get("timestamp", "")
            if message.get("error", False):
                logger.error(f"[{agent.upper()}] {content} at {timestamp}")
            else:
                logger.info(f"[{agent.upper()}] {content} at {timestamp}")
        
        # Log the final verdict
        final_summary = result.get("final_summary", {})
        verdict = final_summary.get("verdict", "Unknown")
        confidence = final_summary.get("confidence_level", 0.0)
        logger.info(f"Final verdict: {verdict} (confidence: {confidence:.2f})")
        
        # Format the response for better readability
        formatted_response = {
            "verdict": final_summary.get("verdict"),
            "confidence": final_summary.get("confidence_level"),
            "summary": final_summary.get("evidence_summary"),
            "deepfake_analysis": final_summary.get("deepfake_analysis", {}),
            "source_verification": final_summary.get("source_verification", {}),
            "text_content": final_summary.get("text_content", {}),
            "key_findings": [
                {
                    "finding": finding.get("finding"),
                    "category": finding.get("category", "General"),
                    "relevance": "High" if i < 2 else "Medium"
                } for i, finding in enumerate(final_summary.get("key_findings", []))
            ],
            "sources": [
                {
                    "url": citation.get("source"),
                    "trust_score": float(citation.get("trust_score", "0")) if citation.get("trust_score", "N/A").replace(".", "", 1).isdigit() else 0.0,
                    "reliability": "High" if float(citation.get("trust_score", "0")) > 0.7 else "Medium" if float(citation.get("trust_score", "0")) > 0.4 else "Low"
                } for citation in final_summary.get("citations", []) if citation.get("source") and not citation.get("source").endswith("_error") and citation.get("source") != "timestamp"
            ]
        }
        
        # Clean up the temporary image file
        try:
            os.remove(image_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary image file: {str(e)}")
        
        # Log the formatted response
        logger.info(f"Returning formatted image analysis response")
        
        return jsonify(formatted_response)

    except Exception as e:
        logger.error(f"Error processing image fact check: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/video', methods=['POST'])
def video_check():
    try:
        # Check if Content-Type header is set correctly
        if request.content_type != 'application/json':
            return jsonify({
                "error": "Unsupported Media Type: Content-Type must be 'application/json'"
            }), 415
            
        # Parse JSON data
        data = request.json
        claim = data.get('claim')
        video_url = data.get('video_url')

        if not claim:
            return jsonify({"error": "Claim is required"}), 400
            
        if not video_url:
            return jsonify({"error": "Video URL is required"}), 400

        logger.info(f"Processing video fact check for claim: {claim}")
        logger.info(f"Using video URL: {video_url}")
        
        # Create video supervisor agent and run fact check
        video_supervisor = VideoSupervisorAgent()
        logger.info("Starting video fact-checking workflow")
        result = video_supervisor.run_video_fact_check(claim, video_url)
        
        # Log the completion of each agent's work
        for message in result.get("messages", []):
            agent = message.get("agent", "unknown")
            content = message.get("content", "")
            timestamp = message.get("timestamp", "")
            if message.get("error", False):
                logger.error(f"[{agent.upper()}] {content} at {timestamp}")
            else:
                logger.info(f"[{agent.upper()}] {content} at {timestamp}")
        
        # Log the final verdict
        final_summary = result.get("final_summary", {})
        verdict = final_summary.get("verdict", "Unknown")
        confidence = final_summary.get("confidence_level", 0.0)
        logger.info(f"Final verdict: {verdict} (confidence: {confidence:.2f})")
        
        # Format the response for better readability
        formatted_response = {
            "verdict": final_summary.get("verdict"),
            "confidence": final_summary.get("confidence_level"),
            "summary": final_summary.get("evidence_summary"),
            "video_analysis": {
                "transcript_quality": final_summary.get("transcript_analysis", "").split("The transcript quality is ")[1].split(".")[0] if "The transcript quality is " in final_summary.get("transcript_analysis", "") else "unknown",
                "word_count": final_summary.get("transcript_analysis", "").split("contains ")[1].split(" words")[0] if "contains " in final_summary.get("transcript_analysis", "") else 0,
                "key_phrases": final_summary.get("transcript_analysis", "").split("Key phrases from the transcript: \n")[1].split("\n") if "Key phrases from the transcript: \n" in final_summary.get("transcript_analysis", "") else []
            },
            "transcript_verification": {
                "supporting_points": len(final_summary.get("supporting_evidence", [])),
                "contradicting_points": len(final_summary.get("contradicting_evidence", []))
            },
            "key_findings": [
                {
                    "finding": finding.get("text"),
                    "timestamp": finding.get("timestamp", "Unknown"),
                    "relevance": "High" if i < 2 else "Medium"
                } for i, finding in enumerate(final_summary.get("key_findings", []))
            ],
            "sources": [
                {
                    "url": citation.get("source"),
                    "trust_score": citation.get("credibility", 0.0),
                    "reliability": "High" if citation.get("credibility", 0.0) > 0.7 else "Medium" if citation.get("credibility", 0.0) > 0.4 else "Low"
                } for citation in final_summary.get("citations", []) if citation.get("source") and not citation.get("source").endswith("_error") and citation.get("source") != "timestamp"
            ]
        }
        
        # Log the formatted response
        logger.info(f"Returning formatted video analysis response")
        
        return jsonify(formatted_response)

    except Exception as e:
        logger.error(f"Error processing video fact check: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/test-image', methods=['POST'])
def test_image_upload():
    try:
        # Log all request information for debugging
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request form data: {dict(request.form)}")
        logger.info(f"Request files: {request.files.keys()}")
        
        # Check if image file is in the request
        if 'image' not in request.files:
            logger.warning("No 'image' field in request.files")
            return jsonify({"error": "No image file provided", "available_fields": list(request.files.keys())}), 400
        
        image_file = request.files['image']
        
        # Check if a filename is provided
        if image_file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Get context if provided
        context = request.form.get('context', 'No context provided')
        
        # Save the file temporarily to verify it works
        temp_path = os.path.join(os.getcwd(), "temp_images")
        os.makedirs(temp_path, exist_ok=True)
        image_path = os.path.join(temp_path, image_file.filename)
        image_file.save(image_path)
        
        # Just return confirmation that we received the file
        return jsonify({
            "status": "success",
            "filename": image_file.filename,
            "file_size": os.path.getsize(image_path),
            "saved_path": image_path,
            "context": context,
            "message": "File received successfully"
        })

    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        return jsonify({"error": str(e), "error_type": type(e).__name__}), 500


if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs("./data/chroma", exist_ok=True)
    # Create temp directory for images if it doesn't exist
    os.makedirs("./temp_images", exist_ok=True)
    app.run(debug=True, port=5000)