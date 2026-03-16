## Crow: The true AI System
- **Elda** : The charismatic, dramatic, peculiar emo partner 

### working demo video in outputs/ 


## Logging
structlog works as a pipeline of processors.
Every log call creates an event dictionary, and each processor modifies that dictionary before it finally gets rendered to a string.

Conceptually:
structlog logger
      │
      ▼
processors(1, 2, 3..)
      │
      ▼
stdlib logging
      │
      ├── console handler
      │
      └── logs/app.log

Every log contains:-
    timestamp
    level
    logger (module name)
    event
    extra structured fields
    exception stacktrace


## MCP Layered Stack:-
agent route (HTTP boundary)
    ↓
agent_loop.py (reasoning cycle)
    ↓
dispatcher.py (execution + safety)
    ↓
registry.py (lookup)
    ↓
tools/base.py + google_calendar.py (implementation)
    ↓
schemas.py (data shapes shared by all layers)


## agent_loop.py
IMPORTANT - Model requirements:
  The LLM must be fine-tuned for function/tool calling.
  Models confirmed working with Ollama tool calling:
    ✅ llama3.1       (8b, 70b)    - pull: ollama pull llama3.1
    ✅ llama3.2       (3b, 1b)     - pull: ollama pull llama3.2
    ✅ qwen2.5        (7b, 14b)    - pull: ollama pull qwen2.5
    ✅ qwen2.5-coder  (7b)         - pull: ollama pull qwen2.5-coder
    ✅ mistral-nemo                 - pull: ollama pull mistral-nemo
    ✅ firefunction-v2              - pull: ollama pull firefunction-v2
    ❌ mistral        (base 7b)    - ignores tools, hallucinates answers
    ❌ phi3, gemma2, tinyllama     - no tool calling support

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │                    Agent Loop                       │
  │                                                     │
  │  messages -> llm -> has tool_calls?                 │
  │                          │ yes           │ no        │
  │                    [Dispatcher]     final_answer     │
  │                          │                          │
  │               inject tool results                   │
  │               back into messages                    │
  │                          │                          │
  │                    loop back ──────────────────────▶│
  └─────────────────────────────────────────────────────┘


##  ASR:
  Architecture:
  ┌─────────────────┐     raw PCM       ┌──────────────────┐
  │  WebSocket      │ ─────────────────▶ │  ASRService      │
  │  audio handler  │                   │                  │
  │                 │ ◀── TranscriptEvt─ │  faster-whisper  │
  └─────────────────┘                   │  + Silero VAD    │
                                        └──────────────────┘