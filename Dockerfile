# numpy and scipy only: no GPU, no CUDA.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace/src

WORKDIR /workspace
COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --upgrade pip && pip install -e ".[dev]"

COPY . /workspace
CMD ["bash"]
