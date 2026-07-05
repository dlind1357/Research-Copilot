import os
from google.cloud import storage
from typing import List, Optional

def get_client() -> storage.Client:
    return storage.Client()

def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parse a GCS URL like 'gs://bucket-name/prefix' into (bucket_name, prefix)."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"GCS path must start with gs://: {gcs_path}")
    parts = gcs_path[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix

def upload_file(bucket_name: str, source_file_content: bytes, destination_blob_name: str, content_type: str = "application/pdf"):
    client = get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(source_file_content, content_type=content_type)

def download_file(bucket_name: str, source_blob_name: str) -> bytes:
    client = get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    return blob.download_as_bytes()

def list_files(bucket_name: str, prefix: Optional[str] = None) -> List[str]:
    client = get_client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    return [blob.name for blob in blobs]

def sync_local_to_gcs(local_dir: str, gcs_path: str):
    """Sync all files from local directory to GCS path."""
    bucket_name, prefix = parse_gcs_path(gcs_path)
    client = get_client()
    bucket = client.bucket(bucket_name)
    
    if not os.path.exists(local_dir):
        return
        
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_file_path, local_dir).replace("\\", "/")
            blob_name = f"{prefix}/{rel_path}" if prefix else rel_path
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file_path)

def sync_gcs_to_local(gcs_path: str, local_dir: str):
    """Sync all blobs from GCS path to local directory."""
    bucket_name, prefix = parse_gcs_path(gcs_path)
    client = get_client()
    bucket = client.bucket(bucket_name)
    
    os.makedirs(local_dir, exist_ok=True)
    
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        blob_name = blob.name
        if prefix:
            if not blob_name.startswith(prefix):
                continue
            # Extract relative path with respect to the prefix
            rel_path = os.path.relpath(blob_name, prefix).replace("\\", "/")
        else:
            rel_path = blob_name
            
        if rel_path == "." or not rel_path or rel_path.startswith(".."):
            continue
            
        local_file_path = os.path.join(local_dir, rel_path).replace("\\", "/")
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)

def save_paper(content: bytes, filename: str):
    """Unified file adapter to save raw papers based on selected STORAGE_TYPE."""
    from app.config.settings import settings
    if settings.STORAGE_TYPE == "local":
        os.makedirs(settings.LOCAL_PAPERS_PATH, exist_ok=True)
        dest = os.path.join(settings.LOCAL_PAPERS_PATH, filename)
        with open(dest, "wb") as f:
            f.write(content)
    else:
        bucket_name, prefix = parse_gcs_path(settings.GCS_PAPERS_PATH)
        blob_name = f"{prefix}/{filename}" if prefix else filename
        upload_file(bucket_name, content, blob_name, content_type="application/pdf")
