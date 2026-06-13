# FLTest container (CPU, Linux). Deliverable for reproducible runs; the locally-verified
# path is the conda env in README.md. Build/run:
#   docker build -t fltest .
#   docker run --rm -it fltest fltest list
#   docker run --rm -it fltest fltest diff examples/configs/differential.yaml
#
# Python is pinned to 3.11 because NVFlare does not yet support 3.12+.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for matplotlib/torch wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching).
COPY pyproject.toml README.md ./
COPY fltest ./fltest
RUN pip install --upgrade pip && pip install -e ".[dev,nvflare]"

COPY examples ./examples
COPY tests ./tests

CMD ["fltest", "list"]
