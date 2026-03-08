"""
DEPRECATED: Gemini CLI settings moved to _admin.gemini_settings.
Re-export for backward compatibility.
"""
import os

from _admin.gemini_settings import (
    build_full_settings,
    get_mcp_servers_for_django,
    write_settings,
)

GITHUB_PERSONAL_ACCESS_TOKEN = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")


def write_gemini_settings(project_root=None, out_path=None, **kwargs):
    """Write settings to .gemini/settings.json. Use _admin.gemini_settings.write_settings for root + .gemini."""
    from pathlib import Path
    root = Path(project_root) if project_root else Path.cwd()
    paths = write_settings(project_root=root, to_root=False, to_gemini_dir=True, **kwargs)
    return paths[0] if paths else root / ".gemini" / "settings.json"


__all__ = [
    "GITHUB_PERSONAL_ACCESS_TOKEN",
    "get_mcp_servers_for_django",
    "write_gemini_settings",
    "build_full_settings",
    "write_settings",
]
