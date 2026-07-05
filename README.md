# Connected Research Copilot 

A production-ready, highly secure biomedical research assistant using FastAPI, LangGraph agent orchestration, real-time citation retrieval, and a gorgeous, responsive, glassmorphic UI.

---

## Key Features

*   **LangGraph-Driven Pipeline**: An intelligent state-machine router that routes user queries dynamically to specialized tools based on content keywords.
*   **Dynamic PubMed Ingestion**: Real-time querying of the PubMed literature database. The agent fetches XML documents, extracts full text, chunks, embeds with Vertex AI `text-embedding-004`, and indexes them on-the-fly into a local ChromaDB vector store.
*   **Precision Gene API (MyGene.info)**: Direct deep integrations with biological databases to fetch chromosome loci, synonyms, and detailed functional summaries of targeted genes.
*   **Aesthetic Single-Page Chat Interface**: Features premium CSS styling (custom animations, a glassmorphic dashboard, responsive layout) powered by an auto-closing HTML-compliant markdown parser.
*   **Zero Clutter Integrated Inline Citations**:
    *   No annoying reference sections appended at the bottom.
    *   Citations are seamlessly embedded as interactive links inline (e.g. `[PMID: 42393898](https://pubmed.ncbi.nlm.nih.gov/42393898)` or `[Gene Info: BAMBI](https://www.ncbi.nlm.nih.gov/gene/25805)`).
    *   Clicking a citation opens the target NCBI page immediately in a new tab (`target="_blank"`).
*   **Security & Safety Guardrails**: Built-in regex filters and prompt-injection defense engines to neutralize template attacks, off-topic requests, and file-traversal exploits.

---
## Pipeline
<img width="762" height="1162" alt="image" src="https://github.com/user-attachments/assets/55809e27-a32a-4283-97bf-d8fedbd8c3e7" />

---

## Repository Layout

```text
research-copilot/
│
├── app/                       # Core Backend AI Logic
│   ├── config/                # LLM & Vertex Embedding service configurations
│   ├── graph/                 # LangGraph node routing and agent compilations
│   ├── rag/                   # Document loaders, chunkers, and ChromaDB vector store
│   ├── security/              # Security sanitizers and prompt injection guardrails
│   └── tools/                 # PubMed fetchers and MyGene.info API tools
│
├── submission_frontend/       # Standalone Frontend FastAPI Server
│   ├── static/                # Asset files
│   │   ├── css/               # Premium custom CSS stylesheets
│   │   └── js/                # app.js parser and chat controllers
│   ├── templates/             # index.html Jinja2 template file
│   └── server.py              # Standalone web app launcher
│
├── setup_storage.py           # Database initializer script
├── requirements.txt           # Global python dependencies
└── README.md                  # Project documentation (this file)
```

---

## Step-by-Step Installation & Setup

Follow these steps to set up and run the application on your computer:

### 1. Prerequisites
Make sure you have:
*   **Python 3.11 or 3.12** installed on your system.
*   A **Google Cloud Platform (GCP)** account with Vertex AI API enabled.
*   The `gcloud` CLI installed and authenticated.

### 2. Prepare Environment & Dependencies
Clone or download this repository, navigate to the directory, and set up a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment:
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r submission_frontend/requirements.txt
```

### 3. Authenticate with Google Cloud
Authorize your machine to use Vertex AI model endpoints via Application Default Credentials:

```bash
gcloud auth application-default login
```

### 4. Configure Your Env File
Create a `.env` file in the root folder of the project.

#### Option A: Local Storage (Default)
```env
GCP_PROJECT=your-gcp-project-id-here
STORAGE_TYPE=local
LOCAL_PAPERS_PATH=./papers
LOCAL_CHROMA_PATH=./chroma_db
```

#### Option B: Google Cloud Storage (GCS)
```env
GCP_PROJECT=your-gcp-project-id-here
STORAGE_TYPE=gcs
GCS_PAPERS_PATH=gs://your-bucket-name/papers
GCS_CHROMA_PATH=gs://your-bucket-name/chroma_db
```

### 5. Initialize Storage
Run the database initializer script to configure local folders, test embeddings connectivity, and prepare the vector database:

```bash
python setup_storage.py
```

### 6. Run the Application
Navigate to the frontend directory and start the FastAPI web server:

```bash
cd submission_frontend
python -m uvicorn server:app --port 8000 --host 127.0.0.1
```

Once started, open your web browser and go to:
**[http://localhost:8000](http://localhost:8000)**

---

## Example Queries to Try

*   **Literature reviews**: `"Review MS research from 2021-2026. Summarize if and how any discoveries have changed our understanding of the disease."`
*   **Gene functions**: `"Tell me what the BAMBI gene does."`
