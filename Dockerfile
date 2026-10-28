FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models
WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY eval ./eval
COPY data/docs ./data/docs

# Bake model weights into the image so cold starts don't hit the network
COPY scripts/download_models.py scripts/download_models.py
RUN python scripts/download_models.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
