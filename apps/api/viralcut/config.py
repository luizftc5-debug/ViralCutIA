"""Runtime configuration. Every value can be overridden with a VIRALCUT_* environment variable."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ASSETS_DIR = Path(__file__).parent / "assets"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIRALCUT_", env_file=".env", extra="ignore")

    # --- storage / persistence
    data_dir: Path = Path("./data")
    database_url: str = ""  # default: sqlite file inside data_dir
    retention_hours: float = 24.0  # uploaded sources and rendered clips are deleted after this

    # --- upload limits
    max_upload_mb: int = 2048
    max_duration_s: int = 3 * 3600
    min_duration_s: float = 3.0
    allowed_extensions: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".webm")
    uploads_per_hour_per_ip: int = 30

    # --- http
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    embedded_worker: bool = False  # run the worker loop inside the API process (single-process dev mode)

    # --- transcription
    whisper_model: str = "small"  # tiny | base | small | medium | large-v3 | distil-large-v3 ...
    whisper_device: str = "auto"  # auto | cpu | cuda
    whisper_compute_type: str = "auto"  # auto -> int8 on cpu, float16 on cuda
    whisper_beam_size: int = 5
    models_dir: Path | None = None  # cache dir for downloaded models (default: HF cache)

    # --- narrative analysis (optional LLM)
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    llm_enabled: bool = True  # only takes effect when an API key is present
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"  # low | medium | high | xhigh | max
    llm_max_candidates: int = 24

    # --- selection
    min_score: float = 58.0
    max_clips: int = 20

    # --- rendering
    x264_crf: int = 18
    x264_preset: str = "medium"
    audio_bitrate: str = "192k"
    normalize_audio: bool = True
    max_fps: float = 60.0
    ffmpeg_threads: int = 0  # 0 = let ffmpeg decide
    ffmpeg_timeout_s: int = 3600

    # --- worker
    worker_poll_interval_s: float = 1.0
    max_attempts: int = 2
    stale_job_minutes: int = 30  # a processing job with no heartbeat for this long is re-queued/failed

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'viralcut.db').resolve()}"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def fonts_dir(self) -> Path:
        return ASSETS_DIR / "fonts"

    @property
    def face_model_path(self) -> Path:
        return ASSETS_DIR / "models" / "face_detection_yunet_2023mar.onnx"

    @property
    def llm_active(self) -> bool:
        return bool(self.llm_enabled and self.anthropic_api_key)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.jobs_dir.mkdir(parents=True, exist_ok=True)
    return s
