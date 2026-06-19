from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # -- LLM --
    deepseek_api_key: str = Field(alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL"
    )
    embedding_model: str = Field(
        default="text-embedding-ada-002", alias="EMBEDDING_MODEL"
    )
    llm_timeout: int = Field(default=120, alias="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")

    # -- Infrastructure --
    postgres_dsn: str = Field(alias="POSTGRES_DSN")
    redis_url: str = Field(default="redis://localhost:16379/0", alias="REDIS_URL")

    # -- MiniCode --
    log_level: str = Field(default="DEBUG", alias="LOG_LEVEL")
    max_context_tokens: int = Field(default=128000, alias="MAX_CONTEXT_TOKENS")
    human_in_loop_enabled: bool = Field(default=True, alias="HUMAN_IN_LOOP_ENABLED")
    similarity_top_k: int = Field(default=10, alias="SIMILARITY_TOP_K")

    # -- Observability (LangSmith) --
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="minicode", alias="LANGSMITH_PROJECT"
    )
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT"
    )


settings = Settings()
