"""
Configuration for the Wellbeing AI system.

Everything is local-first: data stays in your own SQLite file unless you
explicitly turn on a cloud LLM for the coaching narrative (and even then,
only aggregated numeric features + redacted keywords are sent -- never raw
messages, never contact names, never exact locations).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PKG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_ROOT.parent


# --------------------------------------------------------------------------
# Consent: every data source is OFF until the person turns it on.
# --------------------------------------------------------------------------
@dataclass
class SourceConsent:
    """One toggle per data source. Nothing is ingested without consent=True."""

    sleep: bool = True
    activity: bool = True
    food: bool = True
    work_hours: bool = True
    phone_sensors: bool = True
    screen_time: bool = True
    location_visits: bool = True
    spending: bool = True
    search_history: bool = True
    media_history: bool = True
    messages: bool = True
    social_posts: bool = True
    notes: bool = True
    bookmarks: bool = True

    def enabled(self) -> List[str]:
        return [k for k, v in asdict(self).items() if v]

    def is_on(self, name: str) -> bool:
        return bool(getattr(self, name, False))


@dataclass
class PrivacyConfig:
    """How aggressively to minimise data before it is stored or sent anywhere."""

    # Store the raw text of messages/searches/notes, or only derived features?
    store_raw_text: bool = False
    # Hash contact identifiers instead of storing names/numbers.
    hash_contacts: bool = True
    # Round GPS to a place *category* only; never store lat/lon.
    coarse_location_only: bool = True
    # What may leave the machine when the cloud LLM is used.
    llm_send_raw_text: bool = False
    llm_send_contacts: bool = False
    llm_send_place_names: bool = False
    # Delete raw ingest files after they are parsed into the DB.
    purge_raw_after_ingest: bool = False
    # Auto-forget detailed rows older than this (aggregates are kept).
    retention_days_detail: int = 400
    salt: str = "change-me-this-is-mixed-into-contact-hashes"


@dataclass
class ScoreWeights:
    """Weights for the composite sub-scores. Tunable per person."""

    physical: Dict[str, float] = field(
        default_factory=lambda: {
            "sleep": 0.28,
            "activity": 0.24,
            "nutrition": 0.18,
            "recovery": 0.20,
            "medical_load": 0.10,
        }
    )
    mental: Dict[str, float] = field(
        default_factory=lambda: {
            "mood": 0.24,
            "stress": 0.22,
            "social": 0.18,
            "rumination": 0.14,
            "circadian": 0.12,
            "work_balance": 0.10,
        }
    )


@dataclass
class AlertConfig:
    """When to speak up."""

    # Absolute score thresholds (0-100).
    amber_score: float = 55.0
    red_score: float = 40.0
    # Personal-baseline drop, in robust z units, that counts as a real decline.
    z_drop_amber: float = -1.0
    z_drop_red: float = -1.8
    # How many consecutive days a decline must persist before alerting.
    persistence_days: int = 2
    # Don't nag: minimum hours between alerts of the same kind.
    cooldown_hours: int = 20
    # Celebrate improvements too.
    praise_z_gain: float = 0.9
    quiet_hours: List[int] = field(default_factory=lambda: [23, 0, 1, 2, 3, 4, 5, 6])
    channels: List[str] = field(default_factory=lambda: ["desktop", "html", "json"])
    webhook_url: Optional[str] = None  # ntfy.sh / Pushover / Slack style


@dataclass
class ModelConfig:
    baseline_window_days: int = 28
    min_history_days: int = 14
    anomaly_contamination: float = 0.08
    forecast_horizon_days: int = 3
    cluster_k: int = 4
    random_state: int = 7


@dataclass
class LLMConfig:
    enabled: bool = True
    provider: str = "anthropic"
    model: str = "claude-opus-5"
    max_tokens: int = 2000
    temperature: float = 0.6
    api_key_env: str = "ANTHROPIC_API_KEY"
    # If the API is unreachable or disabled, a local template engine writes
    # the narrative instead, so the system never goes silent.
    offline_fallback: bool = True


@dataclass
class UserProfile:
    name: str = "there"
    timezone: str = "Asia/Calcutta"
    age: Optional[int] = None
    sex: Optional[str] = None
    locale: str = "IN"
    # Personal targets -- used for nutrition/activity scoring.
    target_sleep_hours: float = 7.5
    target_steps: int = 8000
    target_active_min: int = 30
    target_water_ml: int = 2500
    work_start_hour: int = 9
    work_end_hour: int = 18
    # Free-text context the coach can use ("new baby", "training for 10k").
    context_notes: str = ""


@dataclass
class Config:
    profile: UserProfile = field(default_factory=UserProfile)
    consent: SourceConsent = field(default_factory=SourceConsent)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    db_path: str = str(PROJECT_ROOT / "data" / "db" / "wellbeing.sqlite")
    raw_dir: str = str(PROJECT_ROOT / "data" / "raw")
    out_dir: str = str(PROJECT_ROOT / "out")

    # ---------------------------------------------------------------- io
    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            path = PROJECT_ROOT / "config" / "config.yaml"
        path = Path(path)
        if not path.exists():
            cfg = cls()
            cfg.save(path)
            return cfg
        raw = yaml.safe_load(path.read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        cfg = cls()
        for section, klass in [
            ("profile", UserProfile),
            ("consent", SourceConsent),
            ("privacy", PrivacyConfig),
            ("weights", ScoreWeights),
            ("alerts", AlertConfig),
            ("models", ModelConfig),
            ("llm", LLMConfig),
        ]:
            if section in d and isinstance(d[section], dict):
                setattr(cfg, section, klass(**{
                    k: v for k, v in d[section].items()
                    if k in {f.name for f in klass.__dataclass_fields__.values()}
                }))
        for key in ("db_path", "raw_dir", "out_dir"):
            if key in d:
                setattr(cfg, key, d[key])
        return cfg

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False, indent=2))

    def ensure_dirs(self) -> None:
        for p in (Path(self.db_path).parent, Path(self.raw_dir), Path(self.out_dir)):
            Path(p).mkdir(parents=True, exist_ok=True)

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.llm.api_key_env)
