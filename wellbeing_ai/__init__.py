"""
wellbeing_ai -- a local-first physical and mental wellbeing tracker.

Reads your own data exports, scores each day against your own baseline with
ML, writes a daily/weekly/monthly report, and speaks up when something has
actually changed.

Not a medical device. It does not diagnose and it does not recommend
medication -- see `wellbeing_ai.safety.ClinicalGuard`.
"""
from .config import Config
from .pipeline import Analysis, Report, WellbeingSystem
from .store import Store

__version__ = "1.0.0"
__all__ = ["WellbeingSystem", "Config", "Store", "Analysis", "Report"]
