# Prox Challenge: Vulcan OmniPro 220 Assistant

A multimodal support agent for the Vulcan OmniPro 220 welder, built for the Prox founding engineer challenge.

This project turns a dense product manual into an interactive assistant that can:

- answer setup and troubleshooting questions in plain language
- retrieve the most relevant manual pages and excerpts
- surface manual images for visual questions
- generate Claude-style artifacts for diagrams and interactive helpers
- support voice input and spoken answers in the frontend

The goal is not just retrieval. The app is designed to feel like a practical garage-side assistant for someone actively trying to use the machine.

## What The App Does

The assistant handles questions like:

- `What polarity setup do I need for TIG welding? Which socket does the ground clamp go in?`
- `What's the duty cycle for MIG welding at 200A on 240V?`
- `I'm getting porosity in my flux-cored welds. What should I check?`
- `How do I set up for MIG on 1/4 inch mild steel?`

Depending on the question, the app can:

- answer directly from retrieved manual content
- pull in page images when the answer is visual
- generate artifacts when a diagram or richer explanation is more useful than plain text
- read the answer aloud

## How The Agent Works

At a high level:

1. The user asks a question in text or voice.
2. The backend orchestrator classifies the request.
3. Retrieval gathers likely relevant manual pages and structured data.
4. Vision is used when the question depends on diagrams, polarity layouts, weld visuals, or page-level imagery.
5. Diagnostic reasoning is used when the question looks like troubleshooting.
6. Artifact generation is used when a visual or interactive answer would help more than text alone.
7. The frontend streams the answer live, renders images and artifacts, and optionally reads the final answer aloud.

## Retrieval Strategy

The manual is preprocessed into cached page and chunk data. Search uses a hybrid approach:

- BM25 / lexical matching
- semantic embeddings
- lightweight query-type heuristics for settings, troubleshooting, polarity, duty cycle, and visual questions

This helps the app stay practical on narrow product questions where wording can vary a lot but the source material is fixed.

## Voice Support

The frontend supports:

- hold-to-talk voice input
- auto-speak on assistant replies
- manual speak / stop controls
- simple voice commands like:
  - `show me the page`
  - `open the diagram`
  - `read that again`
  - `stop`

Speech output behavior:

- Local development can use backend-generated local TTS when available.
- Hosted deployments should generally use browser speech synthesis instead of local Linux TTS.

## Local Setup

### Requirements

- Python 3.11+
- Node.js 20+
- an Anthropic API key

### 1. Install backend dependencies

This repo uses `uv` for Python dependency management.

```bash
uv sync
```

If you do not have `uv` installed yet:

```bash
pip install uv
```

### 2. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=your_key_here
```

Optional local settings:

```bash
FRONTEND_ORIGIN=http://localhost:5173
STRICT_STARTUP_VALIDATION=false
ENABLE_LOCAL_TTS=true
```

### 4. Run the backend

```bash
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

### 5. Run the frontend

```bash
cd frontend
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

## Fly Deployment

The backend is configured for Fly.io with a 2 GB machine.

Deployment files:

- `fly.toml`
- `Dockerfile`
- `.dockerignore`

Suggested Fly secrets / env vars:

```bash
ANTHROPIC_API_KEY=your_key_here
FRONTEND_ORIGIN=https://your-frontend-domain.example
CORS_ALLOWED_ORIGINS=https://your-frontend-domain.example
DEPLOYMENT_ENV=fly
ENABLE_LOCAL_TTS=false
ENABLE_SEMANTIC_SEARCH=false
STRICT_STARTUP_VALIDATION=false
PAGE_RENDER_DPI=120
```

Typical Fly workflow:

```bash
fly launch --no-deploy
fly secrets set ANTHROPIC_API_KEY=your_key_here
fly secrets set FRONTEND_ORIGIN=https://your-frontend-domain.example
fly secrets set CORS_ALLOWED_ORIGINS=https://your-frontend-domain.example
fly deploy
```

The included `fly.toml` targets:

- VM size: `shared-cpu-2x`
- Memory: `2gb`

## Suggested Demo Questions

- `What polarity setup do I need for TIG welding? Which socket does the ground clamp go in?`
- `I'm getting porosity in my flux-cored welds. What should I check?`
- `Which welding process should I use for thin sheet steel at home?`
- `How do I set up for MIG on 1/4 inch mild steel?`

## Project Structure

```text
.
|-- agents/            # orchestration and specialist agents
|-- app/               # FastAPI app, config, service wiring
|-- cache/             # processed manual artifacts, page images, speech cache
|-- files/             # source manuals
|-- frontend/          # Vite + React client
|-- preprocessing/     # document extraction and indexing
|-- tools/             # retrieval, TTS, page handling, embeddings
|-- main.py            # backend entrypoint
|-- fly.toml           # Fly.io deployment config
|-- Dockerfile         # container image for Fly deployment
`-- challenge_README.md
```
