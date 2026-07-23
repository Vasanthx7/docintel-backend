FROM python:3.11-slim

# System deps for PyMuPDF / docling / torch runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Big wheels over a flaky connection → generous timeout + retries.
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10

# Install CPU-only torch first (smaller, avoids CUDA wheels), then the rest.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# HF model cache lives on a mounted volume so weights persist across restarts.
ENV HF_HOME=/models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
