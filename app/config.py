"""TenderAI configuration — loads all settings from .env via python-dotenv."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    """Return the project root (parent of app/)."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # Transport
    transport: str = "stdio"
    host: str = "0.0.0.0"
    port: int = 8000

    # Auth
    mcp_api_key: str = ""

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-5-20241022"
    llm_max_tokens: int = 4096

    # Embeddings (Voyage AI)
    voyage_api_key: str = ""
    embedding_model: str = "voyage-3-lite"
    embedding_dimensions: int = 512

    # Database
    database_path: str = "db/tenderai.db"

    # Data
    data_dir: str = "data"

    # Company
    company_name: str = "Your Company"
    default_currency: str = "OMR"
    default_margin_pct: float = 15.0

    # Logging
    log_level: str = "INFO"

    # Derived (set in __post_init__)
    project_root: Path = field(default_factory=_project_root)

    def abs_database_path(self) -> Path:
        p = Path(self.database_path)
        return p if p.is_absolute() else self.project_root / p

    def abs_data_dir(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else self.project_root / p


def load_settings() -> Settings:
    """Load settings from .env file and environment variables."""
    env_path = _project_root() / ".env"
    load_dotenv(env_path)

    return Settings(
        transport=os.getenv("TRANSPORT", "stdio"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        mcp_api_key=os.getenv("MCP_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", "claude-sonnet-4-5-20241022"),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "voyage-3-lite"),
        embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "512")),
        database_path=os.getenv("DATABASE_PATH", "db/tenderai.db"),
        data_dir=os.getenv("DATA_DIR", "data"),
        company_name=os.getenv("COMPANY_NAME", "Your Company"),
        default_currency=os.getenv("DEFAULT_CURRENCY", "OMR"),
        default_margin_pct=float(os.getenv("DEFAULT_MARGIN_PCT", "15")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        project_root=_project_root(),
    )
