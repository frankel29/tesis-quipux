FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    QUIPUX_LOAD_LLM=false \
    QUIPUX_LLM_PROVIDER=openai \
    QUIPUX_LLM_MODEL=gpt-4o-mini \
    QUIPUX_LLM_USE_AZURE=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download es_core_news_sm

# Pre-descarga BETO en build (sentence-transformers eliminado: sin consumidor desde v5)
RUN python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased'); \
AutoModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')"

COPY main.py quipux_extractor.py ./

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]