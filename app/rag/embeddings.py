import logging
import httpx
import google.generativeai as genai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError
from app.config.settings import settings
from app.config.llm import get_access_token, get_project_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure the Gemini API key
genai.configure(api_key=settings.GOOGLE_API_KEY)

class EmbeddingService:
    def __init__(self, model_name: str = None, max_length: int = 10000):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        # Gemini expects the model name to start with 'models/'
        if not self.model_name.startswith("models/"):
            self.model_name = f"models/{self.model_name}"
        self.max_length = max_length

    def _prepare_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
            
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Input text cannot be empty or whitespace only.")
            
        if len(normalized_text) > self.max_length:
            logger.warning(f"Text length ({len(normalized_text)}) exceeds max length ({self.max_length}). Truncating.")
            normalized_text = normalized_text[:self.max_length]
            
        return normalized_text

    def _get_vertex_embeddings_rest(self, texts: list[str]) -> list[list[float]] | None:
        """Call Vertex AI text-embedding-004 REST API to fetch embeddings."""
        token = get_access_token()
        project_id = get_project_id()
        region = "us-central1"
        if not token:
            return None
        
        # Clean up model name for Vertex path if needed
        model_basename = self.model_name.split("/")[-1] # e.g. text-embedding-004
        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/{model_basename}:predict"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        instances = [{"content": text} for text in texts]
        payload = {"instances": instances}
        
        try:
            # Try secure SSL first, fallback to verify=False if needed (primarily for some local proxies)
            try:
                r = httpx.post(url, json=payload, headers=headers, timeout=20.0)
            except httpx.ConnectError:
                r = httpx.post(url, json=payload, headers=headers, verify=False, timeout=20.0)
            
            if r.status_code == 200:
                data = r.json()
                embeddings = [pred["embeddings"]["values"] for pred in data["predictions"]]
                return embeddings
            else:
                logger.warning(f"Vertex embedding generation failed with status {r.status_code}: {r.text}")
        except Exception as e:
            logger.warning(f"Vertex embedding request exception: {e}")
        return None

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable, InternalServerError)),
        reraise=True
    )
    def get_embedding(self, text: str) -> list[float]:
        try:
            prepared_text = self._prepare_text(text)
            
            # Try Vertex AI first
            vertex_embs = self._get_vertex_embeddings_rest([prepared_text])
            if vertex_embs and len(vertex_embs) > 0:
                return vertex_embs[0]
                
            # Fallback to Google AI Studio
            result = genai.embed_content(
                model=self.model_name,
                content=prepared_text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except ValueError:
            raise
        except Exception:
            logger.error("API error while generating embedding. (Details hidden for security)")
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable, InternalServerError)),
        reraise=True
    )
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
            
        prepared_texts = [self._prepare_text(t) for t in texts]
        
        # Batch the requests to stay within token / size limits of Vertex AI and Google AI Studio
        batch_size = 5  # Small batch size to stay well under limits
        all_embeddings = []
        
        for i in range(0, len(prepared_texts), batch_size):
            batch = prepared_texts[i:i + batch_size]
            try:
                # Try Vertex AI first
                vertex_embs = self._get_vertex_embeddings_rest(batch)
                if vertex_embs and len(vertex_embs) == len(batch):
                    all_embeddings.extend(vertex_embs)
                    continue
                    
                # Fallback to Google AI Studio
                result = genai.embed_content(
                    model=self.model_name,
                    content=batch,
                    task_type="retrieval_document"
                )
                all_embeddings.extend(result['embedding'])
            except ValueError:
                raise
            except Exception:
                logger.error(f"API error while generating embeddings batch {i//batch_size}. (Details hidden for security)")
                raise
                
        return all_embeddings

