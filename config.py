"""
config.py – Centralized settings management using Pydantic BaseSettings.
All values read from environment variables or a .env file at project root.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="sk-placeholder", description="OpenAI API key")
    router_model: str = Field(default="gpt-4o-mini", description="Fast LLM used for routing decisions")
    specialist_model: str = Field(default="gpt-4o", description="Heavy LLM for specialist tasks")

    # ── PostgreSQL Checkpointer ───────────────────────────────────────────────
    postgres_uri: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/agent_ops",
        description="psycopg3 connection string for LangGraph checkpointer",
    )
    postgres_pool_size: int = Field(default=20, description="Connection pool size")

    # ── Qdrant Vector Store ────────────────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant REST endpoint")
    qdrant_api_key: str = Field(default="", description="Qdrant API key (empty for local)")
    qdrant_collection: str = Field(default="agent_memory", description="Vector collection name")
    embedding_dim: int = Field(default=1536, description="Embedding vector dimension (text-embedding-3-small)")

    # ── FastAPI Gateway ────────────────────────────────────────────────────────
    jwt_secret: str = Field(default="change-me-super-secret-jwt-key", description="HS256 JWT secret")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=60)

    # Token Bucket Rate Limiter
    rate_limit_capacity: int = Field(default=60, description="Max burst tokens per client")
    rate_limit_refill_rate: float = Field(default=1.0, description="Tokens refilled per second")

    # ── App ───────────────────────────────────────────────────────────────────
    app_title: str = "LLM Orchestrator Gateway"
    app_version: str = "1.0.0"
    log_level: str = "INFO"
    environment: str = Field(default="development", description="development | staging | production")

    # ── Feature Flags ─────────────────────────────────────────────────────────
    use_postgres_checkpointer: bool = Field(
        default=False,
        description="Set True when Postgres is available; False falls back to MemorySaver",
    )
    use_qdrant_memory: bool = Field(
        default=False,
        description="Set True when Qdrant is reachable; False uses in-memory mock",
    )
    use_simulated_llm: bool = Field(
        default=True,
        description="Set True to use intelligent simulated LLM responses for studying & research without needing paid API keys",
    )


settings = Settings()
