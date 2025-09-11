"""Core configuration management for the Advanced Agent System."""

import os
from functools import lru_cache
from typing import Optional, Dict, Any

from pydantic import  Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration with environment variable support."""
    
    # Application Settings
    app_name: str = Field(default="Advanced Agent System", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="GEMINI_API_KEY")
    openai_model: str = Field(default="gpt-4o", env="GEMINI_MODEL")
    openai_temperature: float = Field(default=0.3, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    openai_timeout: int = Field(default=60, env="OPENAI_TIMEOUT")
    
    # Database Configuration
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    redis_ttl: int = Field(default=3600, env="REDIS_TTL")  # 1 hour default
    
    # Security Configuration
    secret_key: str = Field(..., env="SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    cors_origins: list[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Monitoring & Observability
    enable_tracing: bool = Field(default=False, env="ENABLE_TRACING")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Agent Configuration
    max_agent_turns: int = Field(default=50, env="MAX_AGENT_TURNS")
    agent_timeout: int = Field(default=300, env="AGENT_TIMEOUT")  # 5 minutes
    max_concurrent_agents: int = Field(default=100, env="MAX_CONCURRENT_AGENTS")
    
    # Tool Configuration
    tool_timeout: int = Field(default=30, env="TOOL_TIMEOUT")
    max_tool_retries: int = Field(default=3, env="MAX_TOOL_RETRIES")
    tool_retry_delay: float = Field(default=1.0, env="TOOL_RETRY_DELAY")
    
    # Guardrail Configuration
    enable_guardrails: bool = Field(default=True, env="ENABLE_GUARDRAILS")
    strict_mode: bool = Field(default=False, env="STRICT_MODE")
    content_safety_threshold: float = Field(default=0.8, env="CONTENT_SAFETY_THRESHOLD")
    
    # File Processing Configuration
    max_file_size: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    allowed_file_types: list[str] = Field(
        default=["pdf", "docx", "txt", "md", "html"],
        env="ALLOWED_FILE_TYPES"
    )
    upload_directory: str = Field(default="./uploads", env="UPLOAD_DIRECTORY")
    
    # External Services Configuration
    email_service_url: Optional[str] = Field(default=None, env="EMAIL_SERVICE_URL")
    sms_service_url: Optional[str] = Field(default=None, env="SMS_SERVICE_URL")
    storage_service_url: Optional[str] = Field(default=None, env="STORAGE_SERVICE_URL")
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, value):
        """Parse CORS origins from environment variable."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",")]
        return value
    
    @validator("allowed_file_types", pre=True)
    def parse_allowed_file_types(cls, value):
        """Parse allowed file types from environment variable."""
        if isinstance(value, str):
            return [file_type.strip() for file_type in value.split(",")]
        return value
    
    @validator("log_level")
    def validate_log_level(cls, value):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if value.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return value.upper()
    
    @validator("environment")
    def validate_environment(cls, value):
        """Validate environment setting."""
        valid_environments = ["development", "staging", "production"]
        if value not in valid_environments:
            raise ValueError(f"Invalid environment. Must be one of: {valid_environments}")
        return value
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


def get_model_settings(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Get model configuration with optional overrides."""
    settings = get_settings()
    
    return {
        "model": model or settings.openai_model,
        "temperature": temperature or settings.openai_temperature,
        "max_tokens": max_tokens or settings.openai_max_tokens,
        "timeout": settings.openai_timeout,
    }


def is_production() -> bool:
    """Check if running in production environment."""
    return get_settings().environment == "production"


def is_development() -> bool:
    """Check if running in development environment."""
    return get_settings().environment == "development"


def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    settings = get_settings()
    
    return {
        "url": settings.database_url,
        "pool_size": settings.database_pool_size,
        "max_overflow": settings.database_max_overflow,
        "pool_pre_ping": True,
        "pool_recycle": 3600,  # 1 hour
    }


def get_redis_config() -> Dict[str, Any]:
    """Get Redis configuration."""
    settings = get_settings()
    
    return {
        "url": settings.redis_url,
        "max_connections": settings.redis_max_connections,
        "default_ttl": settings.redis_ttl,
    }


# Export commonly used settings
settings = get_settings()
