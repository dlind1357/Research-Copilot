import os
import re

def main():
    print("==============================================")
    print("      Research Copilot Storage Setup          ")
    print("==============================================\n")
    print("Please select your primary storage backend:\n")
    print("1) Local Storage (Files and database stored on your computer)")
    print("2) GCS Storage   (Files and database backed up to Google Cloud Storage)")
    
    choice = ""
    while choice not in ["1", "2"]:
        choice = input("\nEnter choice (1 or 2): ").strip()
        
    storage_type = "local" if choice == "1" else "gcs"
    
    env_updates = {
        "STORAGE_TYPE": storage_type
    }
    
    default_papers_local = os.path.abspath("papers").replace("\\", "/")
    default_chroma_local = os.path.abspath("chroma_db").replace("\\", "/")
    
    if storage_type == "local":
        print("\n--- Local Storage Configuration ---")
        
        # 1. Full text papers path
        papers_path = input(f"Path for raw papers [{default_papers_local}]: ").strip()
        if not papers_path:
            papers_path = default_papers_local
        env_updates["LOCAL_PAPERS_PATH"] = papers_path.replace("\\", "/")
        
        # 2. ChromaDB path
        chroma_path = input(f"Path for ChromaDB vector store [{default_chroma_local}]: ").strip()
        if not chroma_path:
            chroma_path = default_chroma_local
        env_updates["LOCAL_CHROMA_PATH"] = chroma_path.replace("\\", "/")
        
    else:
        print("\n--- GCS Storage Configuration ---")
        default_bucket = "bio-copilot-data-true-episode-501021-i4"
        
        # 1. GCS full text papers path
        gcs_papers = input(f"GCS path for papers [gs://{default_bucket}/papers]: ").strip()
        if not gcs_papers:
            gcs_papers = f"gs://{default_bucket}/papers"
        if not gcs_papers.startswith("gs://"):
            gcs_papers = "gs://" + gcs_papers.lstrip("/")
        env_updates["GCS_PAPERS_PATH"] = gcs_papers
        
        # 2. GCS ChromaDB path
        gcs_chroma = input(f"GCS path for ChromaDB index [gs://{default_bucket}/chroma_db]: ").strip()
        if not gcs_chroma:
            gcs_chroma = f"gs://{default_bucket}/chroma_db"
        if not gcs_chroma.startswith("gs://"):
            gcs_chroma = "gs://" + gcs_chroma.lstrip("/")
        env_updates["GCS_CHROMA_PATH"] = gcs_chroma

    # Read existing .env file
    env_content = ""
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            env_content = f.read()
            
    # Parse existing lines
    lines = env_content.splitlines()
    new_lines = []
    
    keys_handled = set()
    
    # Update existing lines
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
            
        match = re.match(r"^([A-Z_]+)\s*=\s*(.*)$", stripped)
        if match:
            key, val = match.groups()
            if key in env_updates:
                new_lines.append(f"{key}={env_updates[key]}")
                keys_handled.add(key)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    # Add any new keys that weren't in the original .env
    for key, val in env_updates.items():
        if key not in keys_handled:
            new_lines.append(f"{key}={val}")
            
    # Write back to .env
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
        
    print("\n==============================================")
    print("      Configuration successfully saved!       ")
    print("==============================================")
    print(f"Updated .env settings:")
    for key, val in env_updates.items():
        print(f"  {key} = {val}")
    print("==============================================\n")

if __name__ == "__main__":
    main()
