"""Configuration management for classify-twse-query."""

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    LLM settings are vendor-neutral: any OpenAI-compatible Chat Completions
    endpoint works by setting ``LLM_BASE_URL`` (it defaults to OpenAI). For
    backward compatibility the legacy ``OPENAI_*`` environment variables are
    still accepted.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM Configuration (vendor-neutral, OpenAI-compatible)
    llm_api_key: str = Field(
        ...,
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
        description="API key for the LLM provider (any OpenAI-compatible endpoint)",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices("LLM_MODEL", "OPENAI_MODEL"),
        description="LLM model name",
    )
    llm_timeout: int = Field(
        default=30,
        validation_alias=AliasChoices("LLM_TIMEOUT", "OPENAI_TIMEOUT"),
        description="LLM request timeout in seconds",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENAI_BASE_URL"),
        description="Base URL of an OpenAI-compatible Chat Completions endpoint",
    )

    # FinMind Configuration
    finmind_api_token: str = Field(..., description="FinMind API token")
    finmind_base_url: str = Field(
        default="https://api.finmindtrade.com/api/v4",
        description="FinMind API base URL",
    )

    # Application Configuration
    app_host: str = Field(default="127.0.0.1", description="Application host")
    app_port: int = Field(default=8000, description="Application port")
    log_level: str = Field(default="INFO", description="Logging level")


settings = Settings()
