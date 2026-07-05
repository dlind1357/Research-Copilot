# Use official slim Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables for Python buffering and path
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Create and set working directory
WORKDIR /app

# Install system dependencies (build-essential, etc. needed for any compiled C dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8080 (default for Cloud Run)
EXPOSE 8080

# Start Uvicorn bound to 0.0.0.0 and the dynamically-set PORT (defaults to 8080)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
