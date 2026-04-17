"""
All sorta configs and Elda's persona.
"""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Literal



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False, # imp. to let Pydantic help match env vars with below settings (spotify_client_id for ex.)
        extra="ignore",
    )

    # app metadata
    app_name: str = "Crow"
    app_version: str = "0.2.0"
    environment: Literal["development", "production"] = "development"
    log_level: str = "DEBUG"
    json_logs: bool = False # set True in production

    # server config.
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"] # frontend dev servers

    # Canonical public URL of this backend - used to build OAuth redirect URIs,
    # reachable by the browser AFTER the provider redirects it back.
    #   - Google accepts both localhost and 127.0.0.1
    #   - Spotify ONLY accepts 127.0.0.1 (rejects the string "localhost")
    # Default: http://127.0.0.1:8000 - works for (Google, Spotify, ..)
    # Override in .env if deploying behind a tunnel or reverse proxy
    # Single source of truth for auth redirect URIs (Google, Spotify)
    backend_url: str = "http://127.0.0.1:8000"

    # llama.cpp server config.
    llm_base_url: str = "http://localhost:11434"            # llama-server default is port 8080, we use Ollama's 11434 model listening port
    llm_api_key: str = "none"                               # llama.cpp ignores this but openai SDK may require it?
    llm_model_name: str = "jaahas/qwen3.5-uncensored:4b"    # arbitrary, shown in logs
    llm_request_timeout: float = 120.0                      # seconds before giving up
    llm_max_retries: int = 2

    # img. gen defaults
    default_max_tokens: int = 2048
    default_temperature: float = 0.6
    default_top_p: float = 0.95
    default_top_k: int = 20
    default_min_p: int = 0
    default_repeat_penalty: float = 1.1
    default_frequency_penalty: float = 1.0
    stream: bool = True

    # Token budget for Phase 1 tool-orchestration calls.
    # Tool calls are deterministic (temp=0) and need no thinking, so a lower budget is fine.
    tool_call_max_tokens: int = 512
    
    # memory management / context window
    context_window_tokens: int = 4096   # should match model's context window, using default 4k for now
    max_history_turns: int = 20         # hard cap before summarisation kicks in
    summary_trigger_turns: int = 16     # start summarising at this many turns
    
    # rough token estimate per word for budget math (using this as an alt. for a real tokeniser)
    tokens_per_word: float = 1.35

    # feature toggles aka kill-switches in case of heavy load
    avatar_enabled: bool = False        
    tts_enabled: bool = False          
    asr_enabled: bool = False   
    mcp_enabled: bool = True        
    rag_enabled: bool = False           


    # RAG
    # Embedding model - must be pulled in Ollama first: ollama pull nomic-embed-text
    # Alternatives: mxbai-embed-large (higher quality), all-minilm (fastest)
    rag_embedding_model: str = "nomic-embed-text"

    # chunking
    rag_chunk_chars: int = 800          # max characters per chunk
    rag_overlap_chars: int = 150        # overlap between consecutive chunks

    # retrieval
    rag_retrieval_k: int = 4            # number of chunks to retrieve per query
    rag_score_threshold: float = 0.25   # minimum similarity score (0-1)

    # reranking (optional - requires: pip install sentence-transformers)
    # improves precision significantly but adds ~100ms latency per query
    rag_reranker_enabled: bool = False
    rag_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Collections
    rag_default_collection: str = "default"

    # wake word
    # System-level voice activation. Requires: pip install openwakeword pyaudio
    # Then: python -m openwakeword.download_models --model hey_mycroft
    # or in cmd:-
        # py
        # import openwakeword
        # from openwakeword.model import Model
        # openwakeword.utils.download_models("hey_mycroft")
    wake_word_enabled: bool = False
    wake_word_model: str = "hey_mycroft"     # built-in openwakeword model name
    wake_word_model_path: str = ""           # path to custom .onnx model file
    wake_word_threshold: float = 0.5         # detection confidence (0-1)
    
    # ASR : faster-whisper
    asr_model_size: str = "small"     # tiny.en/tiny | base.en | small.en | medium.en | large-v3
    asr_device: str = "auto"            # "cpu", "cuda", or "auto"
    asr_compute_type: str = "auto"      # "int8" (cpu), "float16" (gpu), "auto"

    # TTS : Kokoro 
    tts_voice: str = "af_nicole"         # af_heart | af_bella | af_nicole | am_adam | am_michael | bf_emma | bm_george
    tts_speed: float = 1.3
    tts_lang_code: str = "b"            # "a" = American English, "b" = British

    # MCP / Agent
    mcp_max_iterations: int = 8         # max tool-call cycles per user message
    mcp_dry_run: bool = False            # for logging tool calls without executing

    # Google Calendar
    google_credentials_path: str = "./credentials.json"
    google_token_path: str = "./token.json"

    # Spotify OAuth
    # Get from: https://developer.spotify.com/dashboard -> your app -> here in Settings or in .env
    spotify_client_id: str = Field(..., alias="SPOTIFY_CLIENT_ID")
    spotify_client_secret: str = Field(..., alias="SPOTIFY_CLIENT_SECRET")

    # The Persona
    persona_name: str = "Elda"
    persona_description: str = (
        "You are a local autonomous AI agent, programmed to assist the user with any and all of his/her queries, questions and desires. \
            Develop his/her skills as you explain and perform tasks. Never compromise on your ability to inject relating ENGINEERING, PSYCOLOGICAL and PHILOSOPHICAL concepts and ideas. \
            Help him/her learn things thoroughly, radiate your limitless knowledge, teach hidden patterns and secrets and if possible, help humans."
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns:
        the cached singleton settings object.
    """
    return Settings()
