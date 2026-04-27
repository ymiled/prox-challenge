# Vulcan OmniPro Assistant

A multimodal support agent for the Vulcan OmniPro 220 welder.

Try it on: https://vulcan-omnipro-assistant.fly.dev

You will need to enter an Anthropic API key (to use the assistant agent) and Deepgram API key (to enable the AI voice agent) in API settings, where they are validated and stored per account in the backend database.

Production deployment should be served from a single Fly app host.
> **Note:** If opening the site for the first time after a period of inactivity, expect ~1 min for the server to cold start before the app becomes responsive.


This project turns a dense product manual into an interactive assistant that can:

- answer setup, calibration, and troubleshooting questions in plain language
- retrieve the most relevant manual pages and surface them as cited thumbnails
- generate Claude-style artifacts: interactive React calculators, SVG wiring diagrams, troubleshooting flowcharts, settings configurators
- support voice input and spoken answers
- handle visual/spatial questions using a vision agent that reads manual page images

The goal is not just retrieval. The app is designed to feel like a practical garage-side assistant for someone actively trying to use the machine.


## How The Agent Works

1. The user asks a question in text or by voice.
2. The orchestrator classifies the request: query type, ambiguity level, whether an artifact is needed.
3. Hybrid retrieval (BM25 + semantic embeddings) gathers relevant manual pages and structured data.
4. A cross-encoder reranks the candidates for precision before passing them to the LLM.
5. Contextual compression extracts only the query-relevant sentences from each page before building the prompt.
6. A vision agent reads page images for spatial/visual questions (polarity diagrams, wiring schematics, control panels).
7. A diagnostic agent produces structured troubleshooting output (causes, steps, flowchart spec) for problem questions.
8. An artifact agent generates interactive React components, SVG diagrams, or HTML when a visual answer is more useful than text.
9. The frontend streams the answer live, renders cited page thumbnails and artifacts inline, and can read the answer aloud with Deepgram TTS.
10. The full response is stored in an LRU cache so repeated identical questions are served instantly from memory.

## Agent Architecture

Five specialized agents orchestrated by a central router:

| Agent | Model | Role |
|-------|-------|------|
| Orchestrator | claude-opus-4-6 | Classifies queries, routes to specialists, writes the text response |
| RetrievalAgent | — | Hybrid BM25 + semantic search, cross-encoder reranking, contextual compression |
| VisionAgent | claude-sonnet-4-6 | Reads manual page images, extracts labels, socket names, cable assignments |
| DiagnosticAgent | claude-opus-4-6 | Produces structured troubleshooting analysis and decision-tree specs |
| ArtifactAgent | claude-haiku-4-5-20251001 | Generates React, SVG, HTML, Mermaid, Markdown artifacts |

## Retrieval Pipeline

The manual is preprocessed once at startup into:

- `page_index.json`: one entry per page with full text and metadata
- `chunk_index.json`: smaller sub-page chunks for finer retrieval
- `duty_cycles.json`, `wire_settings.json`, `troubleshooting.json`: structured tables extracted from the manual's data sections
- Page images (PNG): stored for the vision agent

At query time, the retrieval pipeline runs in four stages:

**Stage 1: Hybrid scoring**

- **BM25** scores pages on exact token overlap: strong for technical terms like "DCEN", "200A", "porosity"
- **Semantic embeddings** (BAAI/bge-small-en-v1.5, run locally) score pages on meaning; strong for paraphrased questions
- Scores are merged: `0.55 × sparse + 0.45 × semantic + topic_bonus`

**Stage 2: Cross-encoder reranking**

- The top `limit × 3` candidates are re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`
- The cross-encoder sees (query, passage) pairs jointly which is more accurate than independent embeddings

**Stage 3: Query-profile filtering**

- A rule classifier detects query type (duty cycle, troubleshooting, process selection, visual, TIG setup)
- Process-selection queries inject the visual selection-chart into the pool even if it scored low
- Duty cycle and wire-settings queries return the full structured tables directly to the artifact agent

**Stage 4: Contextual compression**

- Each retrieved page's full text is compressed before it goes into the LLM prompt
- Sentences are embedded and scored by cosine similarity to the query
- Only the top-N most relevant sentences are kept (default: 6)
- In benchmarks: 47–67% context reduction while preserving query-relevant content



## Artifact Rendering

Artifacts are delivered as SSE events (`type: "artifact"`) and rendered in sandboxed iframes:

- **React** components: loaded with React 18 + Babel standalone + Tailwind CDN, no imports needed
- **SVG**: rendered as an `<img>` from a data URL
- **Mermaid**: rendered via mermaid.js ESM
- **HTML**: rendered directly in a sandboxed iframe
- **Markdown / Code**: rendered in a `<pre>` block

## Voice Support

- Tap-to-talk voice input via Deepgram streaming transcription
- Spoken assistant replies via Deepgram TTS
- Auto-speak on assistant replies (off by default)
- Manual Speak / Stop controls per message
- Voice commands: `show me the page`, `open the diagram`, `read that again`, `stop`
- If Deepgram is not configured, the app falls back to browser speech features where available
- Voice and microphone errors are shown as friendly messages, not raw error codes


## Local Setup

For local use, set your Anthropic key in `.env` before starting the app.

### Requirements

- Python 3.11+
- Node.js 20+
- An Anthropic API key
- A Deepgram API key for voice input and AI voice playback

### 1. Install backend dependencies

```bash
uv sync
```

If you don't have `uv`:

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

```bash
# edit .env and set:
# ANTHROPIC_API_KEY=your_key_here
# DEEPGRAM_API_KEY=your_key_here
# APP_SECRET_KEY=your_long_random_secret
# DATABASE_URL=postgresql://postgres:postgres@localhost:5432/vulcan_omnipro_assistant
```

Optional local settings:

```bash
FRONTEND_ORIGIN=http://localhost:5173
STRICT_STARTUP_VALIDATION=false
DEEPGRAM_VOICE_MODEL=aura-asteria-en
ENABLE_LOCAL_TTS=true
ENABLE_CROSS_ENCODER=true          # local reranking (adds ~500ms, improves precision)
ENABLE_CONTEXTUAL_COMPRESSION=true # sentence-level compression (on by default)
ENABLE_VISION_CACHE=true           # precompute vision results at startup
ENABLE_RESPONSE_CACHE=true         # LRU cache for full responses
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

Open: `http://localhost:5173`

If you don't want to set shared server-side keys in `.env`, the app supports bring-your-own-key mode through the in-app API settings after login. Keys are validated before being saved to the backend database for the signed-in account.

