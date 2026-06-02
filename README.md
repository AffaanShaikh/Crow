## Crow: Local Agentic AI Assistant

[Demo. Video](https://youtu.be/svFpWy913pM)

![](/outputs/Crow_ver._0.1.3_demo_GIF.gif)

### *A fully offline (except tool API calls) AI assistant with agentic tool calling, voice I/O, RAG, live reasoning streams, and a packaged Windows installer.*

> Crow is powered by **Elda** - your dramatic, charismatic, slightly peculiar assistant persona built into the system.

#### Core capabilities

    offline-first assistant runtime
    deterministic tool calling
    safe Google Calendar CRUD with event-id recovery
    Spotify search and playback controls
    streaming ASR and TTS
    RAG over local documents
    live reasoning trace visualization
    packaged Windows distribution
    structured logging and runtime diagnostics

## What does it do?

- hold natural conversations with a persistent persona
- call tools safely through a multi-stage agent loop
- manage Google Calendar events end-to-end
- control Spotify through authenticated actions
- search across local documents with RAG
- listen continuously with wake-word + ASR
- respond with streaming TTS
- show live reasoning and token streaming in the UI
- run as a packaged Windows desktop app with no cloud dependency

## How strong is the architecture?

The entire system is built around a clean separation of concerns so that the assistant remains reliable under tool use, streaming, and voice input.

### 1) Two-phase agent loop

Crow does not mix tool execution and final response synthesis in one pass.

**Phase 1: tool orchestration**
- runs with stripped context
- uses deterministic settings
- focuses only on selecting and executing tools correctly
- loops until tool calls are complete

**Phase 2: final synthesis**
- restores full persona context
- injects tool results back into the message chain
- streams the final response token by token

This separation reduces hallucination bleed-through and keeps tool calls stable even when the assistant is highly expressive in the final response.

### 2) Two-tier tool routing

Tool selection is not left to a single model guess.

Crow uses:
- **keyword / regex matching** for fast, high-confidence routing
- **LLM classification** only when the request is ambiguous

That means most requests are classified quickly, while harder cases still get semantic routing. The result is lower token cost, smaller tool context, and better tool selection accuracy.

### 3) Safe, layered MCP-style tool stack

The tool system is intentionally layered:

```text
HTTP route
  ↓
agent_loop.py
  ↓
dispatcher.py
  ↓
registry.py
  ↓
tools/*
  ↓
schemas.py
```

This structure keeps tool definitions, execution, validation, and data shapes separated. It also makes the system easier to extend without turning the agent into a monolith.

### 4) Real-time streaming from model to UI

Crow streams model output end-to-end:
```
Ollama / LLM → FastAPI → SSE → React
```
The stream carries:

    normal assistant tokens
    reasoning tokens when enabled
    live tool progress state

The frontend renders these as streaming chat bubbles, tool pills, and a collapsing reasoning trace. The assistant feels alive, not post-processed.

### 5) Voice pipeline built for concurrency

Voice input is handled without blocking the web app.

    PyAudio callback thread → wake word detection → asyncio queue → SSE/UI

That thread isolation matters because audio callbacks are synchronous at the C level. Crow keeps mic handling separate from the FastAPI event loop so audio capture does not stall HTTP responses or model inference.

The voice stack includes:

    wake-word detection
    streaming ASR
    sentence-level TTS streaming
    real-time transcript events

### 6) RAG is integrated into the same assistant loop

Crow does not treat retrieval as a separate demo feature. It is part of the assistant path.

Supported ingestion:

    PDF
    DOCX
    HTML
    code files

Retrieval pipeline:

    sentence-aware chunking
    embeddings through Ollama
    persistent ChromaDB storage
    thresholded dense retrieval
    prompt injection of retrieved context during synthesis

This lets Crow answer from local knowledge while staying fully offline.

7) Packaging was treated as a product problem, not an afterthought

Crow is distributed as a real desktop app:

PyInstaller onedir build
custom DLL runtime hook for native extensions
Inno Setup installer
VC++ runtime bundling
per-user install without admin privileges
auto-start support
environment seeding
local data persistence for tokens, documents, and vector store

It also solves a subtle packaged-mode networking issue by using relative frontend API URLs, avoiding localhost vs 127.0.0.1 mismatches that can break SSE/CORS in production builds.

### 8) Logging

Crow uses structlog as a structured logging pipeline.

Each log event carries:

timestamp
level
logger/module name
event text
structured extra fields
exception stack traces when present

The logs are rendered through standard logging handlers to both console and logs/app.log, making debugging across agent, audio, and packaging layers much easier.

## Request flow

#### For a normal user message:

    Browser
    ↓
    POST /api/v1/agent/stream
    ↓
    build_messages()
    ↓
    agent_loop.run_streaming()
    ↓
    router.get_tools_for_message()
    ↓
    Path A: no tools → direct persona stream
    Path B: tools → phase 1 tool orchestration → phase 2 synthesis
    ↓
    SSE token stream
    ↓
    React UI updates live

#### For reasoning-enabled responses:

    UI toggle
    ↓
    ChatRequest.thinking = true
    ↓
    agent_loop sets reasoning_effort = low
    ↓
    Ollama emits delta.reasoning
    ↓
    frontend renders a live reasoning bubble

## Summary

Crow-Elda is an end-to-end local AI system that combines agentic tool use, retrieval, audio, and desktop packaging into one cohesive product. The main strength of the project is not any single model or UI feature, but the way the whole stack was engineered to work reliably together: routing, orchestration, streaming, voice isolation, retrieval, packaging, and stateful desktop deployment.