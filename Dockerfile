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
COPY Redrobsdataset/ ./Redrobsdataset/
COPY app.py .
COPY run_pipeline_test.py .

# Data volume mounts
VOLUME ["/app/output"]

# Expose Streamlit port
EXPOSE 8501

# Default command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
