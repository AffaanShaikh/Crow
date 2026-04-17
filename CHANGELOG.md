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
            > Most issues here are solvable by using a "thinking"-capable model: for _llm_classify (+ robustify keywords).
        - imperfect differentiation between when to update and when to delete or create.
            > for _llm_classify, tool selection process.
        - incorrect fetching of event_id causing subsequent incorrect updat/deletion of event.
            > for reading results well from list_calendar_events.
        - incorrect and unauthorized email invitations for created events (perhaps updated ones too).
            > for recognizing if emailaddress w/ @ was given or not.
        - update to next day causes connected event (over the days) not reflecting actual event times.
        - get_calendar_event: 'event_id' is missing.
            > + descriptive docstring and instruction prompt.
    - TTS model loading overhead due to dependency incompatibility issues.


## ver. 0.2.0:-
### Added:
    - Retrieval Augmentation Generation (local w/ chromadb) RAG w/ optional reranking and chunk retrieval tab in frontend.
    - Spotify tools capability. (requires Spotify Premium since Feb '26 - https://developer.spotify.com/blog/2026-02-06-update-on-developer-access-and-platform-security)
    - functional completely local ASR and TTS.
    - wake word for Speech recognition
    - Google and Spotify auth. login and autotokenization.
    - compatibility w/ thinking LLMs. tested & supports: llama3.2, qwen3.5, nemotron-3-nano
    - safety guardrails and prompt protection. 

### Known issues & todo: 
    - when 'thinking': random outta tokens OR finish_length=stop w/o content.
    - recursive unnecessary tool-checking, limit to 1 for each tool?
    - render 'thinking' block and format bold, italic, headings in frontend.
    - google acc. linked not reflected in frontend + should ask to login before asking about tools (already there? sorta).
    - ui for .exe + bigger pic char sheet typa interface.
    - faster tts inference.
    - custom wake word
    - deprecated/unused code cleanup 

### Fixed:
    - streaming of tokens w/&w/o agentic calls
    - compatibility with thinking models (model-dependent and frontend support to show "thinking" seperately from actual reply)
