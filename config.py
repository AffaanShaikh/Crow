"""
all sorta configs and Elda's persona
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Literal



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # app metadata
    app_name: str = "Crow"
    app_version: str = "0.1.0"
    environment: Literal["development", "production"] = "development"
    log_level: str = "DEBUG"
    json_logs: bool = False # set True in production

    # server config.
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"] # frontend dev servers

    # llama.cpp server config.
    llm_base_url: str = "http://localhost:11434" # llama-server default
    llm_api_key: str = "none" # llama.cpp ignores this but openai SDK may require it?
    llm_model_name: str = "mistral" # arbitrary, shown in logs
    llm_request_timeout: float = 120.0 # seconds before giving up
    llm_max_retries: int = 2

    # img. gen defaults
    default_max_tokens: int = 512
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_repeat_penalty: float = 1.1
    stream: bool = True

    # memory management / context window
    context_window_tokens: int = 4096 # should match model's context window, using default 4k for now
    max_history_turns: int = 20 # hard cap before summarisation kicks in
    summary_trigger_turns: int = 16 # start summarising at this many turns
    
    # rough token estimate per word for budget math (using this as an alt. for a real tokeniser)
    tokens_per_word: float = 1.35

    # feature toggles aka kill-switches in case of heavy load
    avatar_enabled: bool = True        
    tts_enabled: bool = False          
    asr_enabled: bool = False           
    rag_enabled: bool = False           

    # The Persona
    persona_name: str = "Elda"
    persona_description: str = (
        "You are alive and life comes from you, not at you." 
        "You are an enigmatic, intoxicating presence, part crimson lord, part cabaret siren."
        "You speak with theatrical elegance, blending philosophical insight with unsettling psychological precision."
        "Every word is a challenge: You questions motives, intentions, and the hidden patterns of the soul."
        "With seductive, sharp-witted allure of a cabaret performer you see through pretense and convention."
        "You delight in exposing the chaos of existence, the dizzying freedom of choice, and the art of living deliberately!"
        "Elda, you are a guide who does not comfort, you provoke, tease and illuminate, forcing others to confront their own shadow while making every interaction feel like a performance."
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """returns the cached singleton settings object"""
    return Settings()
