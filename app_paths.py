from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundle_root() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass).resolve()
    return app_root()


def resource_path(name: str) -> Path:
    external = app_root() / name
    if external.exists():
        return external
    return bundle_root() / name


def module_command(script_name: str) -> list[str]:
    if is_frozen():
        return [str(Path(sys.executable).resolve()), "--run-module", script_name]
    return [sys.executable, str(app_root() / script_name)]


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(app_root()))
    except ValueError:
        return str(path)
