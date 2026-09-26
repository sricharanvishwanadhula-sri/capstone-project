# Zepto Support Assistant — Dockerfile
# Builds a locally runnable FastAPI service for the RAG-based support assistant.
#
# Build:
#   docker build -t zepto-support-assistant .
#
# Run (mock mode — default, graded baseline):
#   docker run -p 7860:7860 zepto-support-assistant
#
# Run (real LLM — optional, requires GROQ_API_KEY):
#   docker run -p 7860:7860 -e MOCK_LLM=0 -e GROQ_API_KEY=your_key zepto-support-assistant
#
# Test:
#   curl -X POST http://localhost:7860/ask \
#        -H "Content-Type: application/json" \
#        -d '{"query": "What is the delivery fee?"}'

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Pre-download the sentence-transformers model at build time
# so the container starts without needing internet access at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Run ingestion to populate ChromaDB at build time
RUN python ingest.py

# Default environment: mock LLM mode (graded baseline)
ENV MOCK_LLM=1

# Expose port
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
