import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

class Settings:
    def __init__(self):
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL")
        
        # New Storage Variables
        self.STORAGE_TYPE = os.getenv("STORAGE_TYPE", "local").lower()
        self.LOCAL_PAPERS_PATH = os.getenv("LOCAL_PAPERS_PATH")
        self.LOCAL_CHROMA_PATH = os.getenv("LOCAL_CHROMA_PATH")
        self.GCS_PAPERS_PATH = os.getenv("GCS_PAPERS_PATH")
        self.GCS_CHROMA_PATH = os.getenv("GCS_CHROMA_PATH")
        
        self._validate()

    def _validate(self):
        missing = []
        if not self.GOOGLE_API_KEY or self.GOOGLE_API_KEY == "your_google_ai_studio_key":
            missing.append("GOOGLE_API_KEY")
        if not self.EMBEDDING_MODEL:
            missing.append("EMBEDDING_MODEL")
            
        # Validate storage settings
        if self.STORAGE_TYPE not in ["local", "gcs"]:
            raise ValueError(f"Invalid STORAGE_TYPE: '{self.STORAGE_TYPE}'. Must be 'local' or 'gcs'.")
            
        if self.STORAGE_TYPE == "local":
            if not self.LOCAL_PAPERS_PATH:
                missing.append("LOCAL_PAPERS_PATH")
            if not self.LOCAL_CHROMA_PATH:
                missing.append("LOCAL_CHROMA_PATH")
        elif self.STORAGE_TYPE == "gcs":
            if not self.GCS_PAPERS_PATH:
                missing.append("GCS_PAPERS_PATH")
            if not self.GCS_CHROMA_PATH:
                missing.append("GCS_CHROMA_PATH")
            
        if missing:
            raise ValueError(
                f"Missing or invalid required environment variables: {', '.join(missing)}. "
                "Please ensure they are set to valid values in your .env file."
            )

# Global settings instance to be imported across the application
settings = Settings()
