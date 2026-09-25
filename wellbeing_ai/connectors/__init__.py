"""Connector registry: everything that can put data into the store."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Type

from ..config import Config
from ..store import Store
from .base import Connector, IngestResult
from .context import PlacesConnector, SpendingConnector, WorkHoursConnector
from .digital import (BookmarkConnector, GoogleSearchConnector, NotesConnector,
                      SocialConnector, WhatsAppConnector, YouTubeConnector)
from .health import (ActivityConnector, AppleHealthConnector, CheckinConnector,
                     FoodConnector, ScreenTimeConnector, SensorConnector,
                     SleepConnector)

ALL_CONNECTORS: List[Type[Connector]] = [
    SleepConnector,
    ActivityConnector,
    AppleHealthConnector,
    FoodConnector,
    ScreenTimeConnector,
    SensorConnector,
    CheckinConnector,
    WorkHoursConnector,
    PlacesConnector,
    SpendingConnector,
    GoogleSearchConnector,
    YouTubeConnector,
    WhatsAppConnector,
    SocialConnector,
    BookmarkConnector,
    NotesConnector,
]


def ingest_all(cfg: Config, store: Store,
               root: Optional[Path] = None,
               only: Optional[List[str]] = None,
               force: bool = False) -> List[IngestResult]:
    """
    Run every consented connector.

    `force=True` re-reads files that haven't changed since the last run. The
    default is not to, which is what makes running `ingest` twice safe.
    """
    results = []
    for klass in ALL_CONNECTORS:
        if only and klass.name not in only:
            continue
        results.append(klass(cfg, store).ingest(root, force=force))
    return results


__all__ = ["ALL_CONNECTORS", "ingest_all", "Connector", "IngestResult"]
