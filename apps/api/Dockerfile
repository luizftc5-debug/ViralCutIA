# ViralCut API + worker (one image, two commands).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    VIRALCUT_DATA_DIR=/data

# ffmpeg from Debian ships libx264, libass, libzimg and AAC: everything the pipeline uses
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY viralcut ./viralcut
RUN pip install ".[llm,postgres]"

# uploads are untrusted input: never process them as root
RUN useradd --create-home --uid 10001 viralcut \
    && mkdir -p /data /models \
    && chown viralcut:viralcut /data /models
USER viralcut

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)" || exit 1

CMD ["uvicorn", "viralcut.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
