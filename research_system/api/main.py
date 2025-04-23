"""
API endpoints for the Fact Checking System.
"""

# --- Start: Add project root to sys.path for direct execution ---
import sys
import os
# Calculate the path to the project root directory (Faraday-Web-Researcher-Agent)
# __file__ is research_system/api/main.py
# os.path.dirname(__file__) is research_system/api
# os.path.dirname(os.path.dirname(__file__)) is research_system
# os.path.dirname(os.path.dirname(os.path.dirname(__file__))) is the project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End: Add project root to sys.path ---

from fastapi import FastAPI, HTTPException, BackgroundTasks, status # Added BackgroundTasks, status
from pydantic import BaseModel
import uvicorn
import logging # Added for logging
from typing import List, Dict, Any, Optional # Added Optional
import uuid # Added for task IDs
import asyncio

# Updated imports to use absolute paths from the project root
# This relies on the sys.path modification above for direct execution
from research_system.agent import run_web_research
from research_system.schemas import ResearchRequest, ResearchReport, ErrorResponse

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Web Research Agent API", # Updated title
    description="API for performing automated web research using an AI agent.", # Updated description
    version="0.1.0",
)

# In-memory storage for task results (Replace with Redis/Celery/DB for production)
task_results: Dict[str, Any] = {}


class ClaimRequest(BaseModel):
    claim: str
    language: str = "en"  # Default language

class ClaimResponse(BaseModel):
    claim: str
    result: str # e.g., "True", "False", "Uncertain"
    confidence_score: float
    explanation: str
    sources: List[Dict[str, Any]] = [] # Make sure this is List[Dict[str, Any]]

# New response model for initiating a task
class TaskResponse(BaseModel):
    task_id: str
    message: str

# New response model for task status/result
class ResultResponse(BaseModel):
    status: str # e.g., "processing", "completed", "error"
    result: Optional[ResearchReport] = None # Use ResearchReport
    error: Optional[ErrorResponse] = None # Include structured error


# --- Helper function to run the research in the background ---
def run_background_research(task_id: str, query: str, language: str): # Renamed function
    """Runs the web research agent and stores the result or error."""
    logger.info(f"Background research task started for task_id: {task_id}, query: '{query}' (lang: {language})")
    try:
        # Run the async agent function using asyncio.run()
        # Pass relevant config if needed, e.g., language preference
        # research_config = {"language": language} # Example config
        research_result = asyncio.run(run_web_research(query=query)) # Call the agent function

        if isinstance(research_result, ResearchReport):
            logger.info(f"Background research task completed for task_id: {task_id}. Storing report.")
            task_results[task_id] = ResultResponse(status="completed", result=research_result)
        elif isinstance(research_result, ErrorResponse):
             logger.error(f"Background research task failed for task_id: {task_id}. Storing error: {research_result.error}")
             task_results[task_id] = ResultResponse(status="error", error=research_result)
        else:
            # Handle unexpected return type
            logger.error(f"Background task for {task_id} returned unexpected type: {type(research_result)}. Storing generic error.")
            task_results[task_id] = ResultResponse(status="error", error=ErrorResponse(error="Agent returned unexpected result type", details=str(research_result)))

    except ImportError as ie:
         logger.error(f"ImportError during background task for task_id {task_id}: {ie}. Agent might not be available.", exc_info=True)
         task_results[task_id] = ResultResponse(status="error", error=ErrorResponse(error="Agent component import failed", details=str(ie)))
    except Exception as e:
        logger.error(f"Exception during background task for task_id {task_id}, query '{query}': {e}", exc_info=True)
        # Store error information
        task_results[task_id] = ResultResponse(status="error", error=ErrorResponse(error="Internal server error during research", details=str(e)))


# --- API Endpoints ---\

@app.post("/research", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_research_endpoint(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Receives a research query, starts the research process in the background,
    and returns a task ID.
    """
    logger.info(f"Received query: '{request.query}' with language '{request.language}'. Starting background task.")
    task_id = str(uuid.uuid4())

    # Add the renamed background task function
    background_tasks.add_task(run_background_research, task_id, request.query, request.language)

    # Store initial processing status
    task_results[task_id] = ResultResponse(status="processing")

    return TaskResponse(task_id=task_id, message="Web research process started.")


@app.get("/results/{task_id}", response_model=ResultResponse)
async def get_results_endpoint(task_id: str):
    """
    Retrieves the status or result of a web research task.
    """
    logger.debug(f"Checking results for task_id: {task_id}")
    result_info = task_results.get(task_id) # Renamed variable

    if not result_info:
        logger.warning(f"Task ID not found: {task_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task ID not found")

    logger.debug(f"Current status for task_id {task_id}: {result_info.status}")
    # Optionally remove completed/error tasks after retrieval? Or implement TTL?
    # Be careful with concurrent requests if modifying shared dict here.

    return result_info


@app.get("/")
async def root():
    """
    Root endpoint providing a welcome message.
    """
    return {"message": "Welcome to the Web Research Agent API!"} # Updated message


if __name__ == "__main__":
    # Note: Reload=True is good for development. Consider turning it off for production.
    # The path here 'research_system.api.main:app' should work if uvicorn is run
    # from the project root directory (parent of research_system).
    # Example: python -m uvicorn research_system.api.main:app --reload
    uvicorn.run("research_system.api.main:app", host="0.0.0.0", port=8000, reload=True) 