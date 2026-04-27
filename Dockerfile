FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend ./
RUN npm run build

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml uv.lock README.md ./
COPY app ./app
COPY agents ./agents
COPY preprocessing ./preprocessing
COPY tools ./tools
COPY files ./files
COPY frontend ./frontend
COPY main.py ./
COPY --from=frontend-builder /frontend/dist ./frontend/dist

RUN uv sync --frozen
RUN uv run python tools/build_cache.py

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
