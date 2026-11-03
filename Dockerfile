# Synap Core server image.
#
# No third-party data or models are inside: the embedding model and the spaCy model are
# downloaded on first start into the `models` volume, and checked against pinned hashes
# (synap/boot/first_start.py). The same image applies the schema (synap/db/runner.py).
#
#   docker build -t synap-core:local .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first, so nothing below pulls the GPU build (about 3.5 GB larger).
# Pinned: PyTorch computes every embedding, so a different version on a rebuild can change what retrieval finds.
# The pin matches the CPU wheel on both amd64 and arm64 ("+cpu" is the local part of the version and need not be written).
# To move it: change the number, rebuild, and run the end-to-end suite (e2e/) before you merge.
RUN pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY synap/ /app/synap/

# Not root. /models is the model volume; owning it here lets a fresh named volume inherit that.
RUN useradd --system --uid 10001 --no-create-home synap \
    && mkdir -p /models /app/originals /app/config \
    && chown synap /models /app/originals

USER synap
ENV SYNAP_EDITION=core \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    PYTHONPATH=/models/spacy

EXPOSE 8000
CMD ["uvicorn", "synap.cloud.application.main:app", "--host", "0.0.0.0", "--port", "8000"]
