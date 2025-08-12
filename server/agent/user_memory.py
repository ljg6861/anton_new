"""
User-scoped memory store for preferences and contextual knowledge.
- Persists per-user JSON file under data/user_memory/{user_id}.json
- Thread-safe basic operations.
- Provides a compact context string for prompts.
"""
from __future__ import annotations
import json
import os
import re
import threading
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List, Optional

_DATA_DIR = Path("data/user_memory")
_LOCK = threading.Lock()

@dataclass
class UserProfile:
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    facts: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "UserProfile":
        return UserProfile(
            user_id=d.get("user_id", ""),
            preferences=d.get("preferences", {}) or {},
            facts=d.get("facts", {}) or {},
            notes=d.get("notes", []) or [],
            tags=d.get("tags", []) or [],
        )


def _profile_path(user_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", user_id)
    return _DATA_DIR / f"{safe}.json"


def load_profile(user_id: str) -> UserProfile:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(user_id)
    if not path.exists():
        return UserProfile(user_id=user_id)
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return UserProfile.from_dict(data)
    except Exception:
        return UserProfile(user_id=user_id)


def save_profile(profile: UserProfile) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(profile.user_id)
    tmp = path.with_suffix(".json.tmp")
    with _LOCK:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(profile.to_json())
        os.replace(tmp, path)


# --- High-level helpers ---

def add_preference(user_id: str, key: str, value: Any) -> None:
    prof = load_profile(user_id)
    prof.preferences[key] = value
    save_profile(prof)


def add_fact(user_id: str, key: str, value: Any) -> None:
    prof = load_profile(user_id)
    prof.facts[key] = value
    save_profile(prof)


def add_note(user_id: str, text: str) -> None:
    if not text:
        return
    prof = load_profile(user_id)
    prof.notes.append(text.strip())
    # keep last 200 notes
    if len(prof.notes) > 200:
        prof.notes = prof.notes[-200:]
    save_profile(prof)


def build_context(user_id: str) -> str:
    """Returns a compact context string describing the user profile."""
    prof = load_profile(user_id)
    parts: List[str] = []
    if prof.preferences:
        prefs = ", ".join(f"{k}={v}" for k, v in list(prof.preferences.items())[:12])
        parts.append(f"User Preferences: {prefs}.")
    if prof.facts:
        facts = ", ".join(f"{k}={v}" for k, v in list(prof.facts.items())[:12])
        parts.append(f"User Facts: {facts}.")
    # include a couple of notes as examples
    if prof.notes:
        notes = "; ".join(prof.notes[-3:])
        parts.append(f"Recent Notes: {notes}.")
    return "\n".join(parts)


# --- Lightweight extraction from a user message ---
_PREF_PATTERNS = [
    (re.compile(r"\bmy name is\s+([^.,!\n]+)", re.IGNORECASE), ("name", 1)),
    (re.compile(r"\bcall me\s+([^.,!\n]+)", re.IGNORECASE), ("name", 1)),
    (re.compile(r"\bi prefer\s+([^.,!\n]+)", re.IGNORECASE), ("preference", 1)),
    (re.compile(r"\btimezone is\s+([^.,!\n]+)", re.IGNORECASE), ("timezone", 1)),
    (re.compile(r"\blanguage is\s+([^.,!\n]+)", re.IGNORECASE), ("language", 1)),
]

def extract_and_update_from_message(user_id: str, text: str) -> None:
    """Heuristic extraction of preferences/facts from free-form user text."""
    if not text:
        return
    lowered = text.strip()
    for pat, (key, group_idx) in _PREF_PATTERNS:
        m = pat.search(lowered)
        if not m:
            continue
        value = m.group(group_idx).strip().strip('"\'')
        if key in ("preference",):
            add_note(user_id, f"pref:{value}")
        else:
            add_preference(user_id, key, value)
