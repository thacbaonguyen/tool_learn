FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       espeak-ng \
       ffmpeg \
       fonts-dejavu-core \
       tesseract-ocr \
       tesseract-ocr-eng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir ".[dev]"

ENTRYPOINT ["python", "-m"]
CMD ["poc.extract_subtitles", "--help"]
