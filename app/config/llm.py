import os
import logging
import subprocess
import httpx
from app.config.settings import settings

logger = logging.getLogger(__name__)

def get_access_token() -> str:
    # Attempt 1: From metadata server (Cloud Run environment)
    try:
        url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
        headers = {"Metadata-Flavor": "Google"}
        r = httpx.get(url, headers=headers, timeout=2.0)
        if r.status_code == 200:
            return r.json()["access_token"]
    except Exception:
        pass

    # Attempt 2: From gcloud (local development)
    try:
        result = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True, check=True, shell=True)
        return result.stdout.strip()
    except Exception:
        pass

    # Attempt 3: Env fallback if set
    return os.getenv("GCP_ACCESS_TOKEN", "")

def get_project_id() -> str:
    # Attempt 1: From metadata server
    try:
        url = "http://metadata.google.internal/computeMetadata/v1/project/project-id"
        headers = {"Metadata-Flavor": "Google"}
        r = httpx.get(url, headers=headers, timeout=2.0)
        if r.status_code == 200:
            return r.text.strip()
    except Exception:
        pass
    
    # Fallback to local default / env
    return os.getenv("GCP_PROJECT_ID", "true-episode-501021-i4")

def generate_llm_content(prompt: str, response_mime_type: str = "text/plain") -> str:
    """Generate content robustly using Vertex AI (Gemini 2.5 Flash) first, falling back to AI Studio."""
    token = get_access_token()
    project_id = get_project_id()
    region = "us-central1"
    
    if token:
        # Use Vertex AI (Gemini 2.5 Flash)
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/gemini-2.5-flash:generateContent"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{
                "role": "user",
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        if response_mime_type == "application/json":
            payload["generationConfig"] = {"responseMimeType": "application/json"}
            
        try:
            # We disable SSL verify bypass to avoid warning logs on secure Cloud Run environment,
            # but allow fallback verify=False if there's any certificate mismatch locally.
            try:
                r = httpx.post(url, json=payload, headers=headers, timeout=30.0)
            except httpx.ConnectError:
                r = httpx.post(url, json=payload, headers=headers, verify=False, timeout=30.0)
                
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                logger.warning(f"Vertex AI call failed with status {r.status_code}: {r.text}. Trying fallback...")
        except Exception as e:
            logger.warning(f"Vertex AI request exception: {e}. Trying fallback...")

    # Ultimate fallback: Use Google AI Studio REST endpoint (Gemini 2.5 Flash)
    api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
    if api_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        if response_mime_type == "application/json":
            payload["generationConfig"] = {"responseMimeType": "application/json"}
            
        try:
            r = httpx.post(url, json=payload, verify=False, timeout=30.0)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                logger.error(f"Google AI Studio fallback failed with status {r.status_code}: {r.text}")
        except Exception as e:
            logger.error(f"Google AI Studio fallback request exception: {e}")

    raise RuntimeError("Both Vertex AI (Gemini 2.5) and Google AI Studio generations failed. Check API configuration or billing status.")
