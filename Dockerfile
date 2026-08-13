FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake sentence-transformers model at build time (no runtime download).
ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers
# Optional build-time token (public models usually work without it).
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY main.py .
COPY src/ ./src/
COPY data/catalog.json data/catalog.index data/catalog_ids.json data/catalog_index_meta.json ./data/

ENV PYTHONPATH=/app:/app/src
# Render/Railway inject PORT at runtime; default for local docker run only.
EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
