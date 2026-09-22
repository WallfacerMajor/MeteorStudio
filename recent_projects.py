"""Small, independent history of named project files (never source images)."""
from __future__ import annotations

import json
import os
from pathlib import Path


MAX_RECENT_PROJECTS = 8


def history_path(autosave_path: Path) -> Path:
    return autosave_path.with_name("recent_projects.json")


def _normalized(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _key(path: Path) -> str:
    return os.path.normcase(str(path))


def recent_projects(index_path: Path, *, prune: bool = True) -> list[Path]:
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    result: list[Path] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str) or not item:
            continue
        path = _normalized(item)
        identity = _key(path)
        if identity in seen or not path.is_file():
            continue
        seen.add(identity)
        result.append(path)
        if len(result) == MAX_RECENT_PROJECTS:
            break
    if prune and [str(path) for path in result] != raw:
        try:
            _write(index_path, result)
        except OSError:
            pass
    return result


def remember_project(index_path: Path, project_path: str | Path) -> list[Path]:
    selected = _normalized(project_path)
    if not selected.is_file():
        return recent_projects(index_path)
    history = [selected]
    history.extend(path for path in recent_projects(index_path, prune=False)
                   if _key(path) != _key(selected))
    history = history[:MAX_RECENT_PROJECTS]
    _write(index_path, history)
    return history


def _write(index_path: Path, paths: list[Path]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(".writing.json")
    temporary.write_text(json.dumps([str(path) for path in paths], ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, index_path)
