# Changelog
All notable changes for *Crow* are documented in this file.

## ver. 0.1.3:-
### Added:
    - MCP Agentic support.
        - w/ dispatcher, tool register and data router for llm.
    - Calendar (Google Calendar) CRUD API agentic calls.
        - Google credentials and (cached) auto-tokenization.
    - Prompt curation pipeline.
    - datetime parsing & understanding for time-related tasks.
    - ASR and TTS backend.
    (all agentic behaviour tested with llama3.2 2GB, smallest tool-calling capable ollama model).

### Known issues: 
    - Calendar events agentic behaviour:-
        - imperfect recognition of when to call and use tools (current implementation: keyword detection for using tools)
        - imperfect differentiation between when to update and when to delete or create.
        - incorrect fetching of event_id causing subsequent incorrect updat/deletion of event.
        - incorrect and unauthorized email invitations for created events (perhaps updated ones too).
        - update to next day causes connected days length worth of event not reflecting actual event times.
    - TTS model loading overhead due to dependency incompatibility issues.
