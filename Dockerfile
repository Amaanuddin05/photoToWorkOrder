FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    PORT=80

WORKDIR /app

# 1. Install libcap2-bin for binding port 80 as non-root user, plus CPU-only PyTorch
RUN apt-get update && \
    apt-get install -y --no-install-recommends libcap2-bin && \
    setcap 'cap_net_bind_service=+ep' /usr/local/bin/python3.11 && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Copy dependencies and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Create non-root user and persistent models directory
RUN useradd -m -u 1000 appuser && \
    mkdir -p /models && \
    chown -R appuser:appuser /app /models

# 4. Copy application source code
COPY app/ /app/app/

USER appuser

EXPOSE 80

# Healthcheck using Python's standard library urllib (no curl in python:3.11-slim)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, os; port=os.environ.get('PORT', '80'); urllib.request.urlopen(f'http://localhost:{port}/health')"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-80} --workers 1"]

