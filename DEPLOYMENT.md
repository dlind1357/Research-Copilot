# Google Cloud Run Deployment Guide

This guide provides step-by-step instructions to build, configure, and deploy the Research Copilot application to **Google Cloud Run**.

---

## 1. Architecture Overview
*   **Port**: Listens on port `8080` (dynamically configured via the `$PORT` environment variable).
*   **Statelessness**: The container instance is designed to be stateless. ChromaDB writes to container ephemeral storage, which is perfectly suited for stateless Cloud Run instance lifecycles.
*   **Storage**: Google Cloud Storage (GCS) is used as the durable persistence layer for storing uploaded PDFs.
*   **Configuration**:
    *   **Local Development**: Environment variables are loaded from the local `.env` file.
    *   **Production Deployment**: Environment variables are injected directly into the Cloud Run environment.

---

## 2. Prerequisites
1.  Install the [Google Cloud SDK (gcloud CLI)](https://cloud.google.com/sdk/docs/install).
2.  Authenticate and configure your project:
    ```bash
    gcloud auth login
    gcloud config set project YOUR_PROJECT_ID
    ```
3.  Enable the required Google Cloud API services:
    ```bash
    gcloud services enable run.googleapis.com artifactregistry.googleapis.com
    ```

---

## 3. Deployment Steps

### Step 1: Create an Artifact Registry Repository
Create a Docker repository in Artifact Registry to store your container images:
```bash
gcloud artifacts repositories create research-copilot-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for Research Copilot"
```

### Step 2: Build and Publish the Container Image
Using **Google Cloud Build**, you can compile the image directly in the cloud without needing a local Docker engine:
```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/YOUR_PROJECT_ID/research-copilot-repo/app:latest .
```

### Step 3: Deploy to Cloud Run
Deploy the published image to Cloud Run, injecting production environment variables:
```bash
gcloud run deploy research-copilot \
    --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/research-copilot-repo/app:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080 \
    --set-env-vars GOOGLE_API_KEY="your_production_google_ai_studio_key" \
    --set-env-vars EMBEDDING_MODEL="models/text-embedding-004"
```

---

## 4. Environment Variables Reference

| Variable Name | Description | Source (Local) | Source (Production) |
|---|---|---|---|
| `GOOGLE_API_KEY` | Google Gemini AI Studio API key | `.env` | Cloud Run Env Var |
| `EMBEDDING_MODEL` | Embedding model identifier | `.env` | Cloud Run Env Var |

---

## 5. Local Verification (Dry-Run)
To test the Docker container locally before deploying to the cloud:
1.  Build the container locally:
    ```bash
    docker build -t research-copilot .
    ```
2.  Run the container locally, exposing port `8080` and passing environment variables:
    ```bash
    docker run -p 8080:8080 --env-file .env research-copilot
    ```
3.  Verify the app is running by visiting [http://localhost:8080/health](http://localhost:8080/health).
