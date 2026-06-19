FROM python:3.11-slim

WORKDIR /app

# Install basic compiler tools if needed, then clean up to keep image slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY src/ ./src/
COPY run_pipeline_test.py .

# Data volume mounts
VOLUME ["/app/data", "/app/output"]

# Default command to run the pipeline on dataset
CMD ["python", "src/pipeline.py", "--input", "/app/data/candidates.jsonl", "--output-dir", "/app/output"]
