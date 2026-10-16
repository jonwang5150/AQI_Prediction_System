"""????????????????? .env????????"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


RAG_SOURCES = {'air_quality_hourly': {'kind': '實際觀測',
                        'time_column': 'observed_at',
                        'primary_keys': ['station', 'observed_at'],
                        'content_columns': ['aqi',
                                            'air_quality_indicator',
                                            'pm25',
                                            'pm10',
                                            'o3',
                                            'co',
                                            'so2',
                                            'no2']},
 'air_quality_predictions': {'kind': '逐時預測',
                             'time_column': 'predicted_at',
                             'primary_keys': ['station', 'predicted_at'],
                             'content_columns': ['predicted_aqi', 'actual_aqi']},
 'air_quality_predictions_8h': {'kind': '多步預測',
                                'time_column': 'target_at',
                                'primary_keys': ['station',
                                                 'target_at',
                                                 'horizon_hours',
                                                 'model_version'],
                                'content_columns': ['predicted_aqi',
                                                    'horizon_hours',
                                                    'model_version',
                                                    'created_at']}}


@dataclass(frozen=True)
class Settings:
    database_url: str
    llm_base_url: str
    embedding_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float
    auto_seed: bool
    seed_data_dir: str
    batch_size: int = 1000
    dry_run: bool = False
    rag_sources: dict = field(default_factory=dict)

    # ???????????????????
    @property
    def ollama_base_url(self) -> str:
        return self.llm_base_url

    @property
    def ollama_chat_model(self) -> str:
        return self.llm_model

    @property
    def ollama_embedding_model(self) -> str:
        return self.embedding_model


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    return Settings(
        database_url=os.getenv("DATABASE_URL", "postgresql+psycopg://postgres:abc123@localhost:5432/postgres"),
        llm_base_url=os.getenv("LLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")),
        embedding_model=os.getenv("EMBEDDING_MODEL", os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        top_k=int(os.getenv("TOP_K", "4")),
        score_threshold=float(os.getenv("SCORE_THRESHOLD", "0")),
        auto_seed=_env_bool("AUTO_SEED", True),
        seed_data_dir=os.getenv("SEED_DATA_DIR", str(Path(__file__).with_name("seed_data"))),
        batch_size=int(os.getenv("BATCH_SIZE", "1000")),
        dry_run=_env_bool("DRY_RUN", False),
        rag_sources={name: {key: list(value) if isinstance(value, list) else value
                            for key, value in spec.items()}
                     for name, spec in RAG_SOURCES.items()},
    )


settings = load_settings()
