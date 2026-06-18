from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    deepseek_api_key: str = Field(alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")
    deepseek_base_url: str = Field(default="https://api.deepseek.com/v1", alias="DEEPSEEK_BASE_URL")

    postgres_dsn: str = Field(alias="POSTGRES_DSN")
    redis_url: str = Field(default="redis://localhost:16379/0", alias="REDIS_URL")

    log_level: str = Field(default="DEBUG", alias="LOG_LEVEL")
    max_context_tokens: int = Field(default=128000, alias="MAX_CONTEXT_TOKENS")
    human_in_loop_enabled: bool = Field(default=True, alias="HUMAN_IN_LOOP_ENABLED")
    similarity_top_k: int = Field(default=10, alias="SIMILARITY_TOP_K")


settings = Settings()
