import os
import sys
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# Get directories and set up paths to import the agent package
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# Insert parent directory to sys.path to allow importing the backend agent 'app' package
sys.path.insert(0, PARENT_DIR)

# Load environment variables before importing the agent
load_dotenv(os.path.join(PARENT_DIR, ".env"))

# Import the actual research agent runner
from app.graph.agent import run_agent

app = FastAPI(
    title="Research Copilot Frontend Service",
    description="A production-quality standalone frontend server connected to the LangGraph agent pipeline.",
    version="1.0.0"
)

static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Read GCP Project from environment if present
GCP_PROJECT = os.getenv("GCP_PROJECT", "placeholder-project-id")

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface(request: Request):
    """Serves the premium single-page chat application HTML."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"gcp_project": GCP_PROJECT}
    )

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Executes the actual LangGraph research agent pipeline, processes findings,
    and returns a beautifully formatted scientific response including citations.
    """
    # Execute the real LangGraph research agent
    result = await run_agent(request.message)
    
    if not result.get("safety_passed"):
        response_text = f"⚠️ **Request Blocked by Safety Guardrails:** {result.get('safety_reason', 'Query not allowed.')}"
    else:
        response_text = result.get("final_answer", "")
                
    return {
        "response": response_text
    }
