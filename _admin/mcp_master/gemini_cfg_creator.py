"""
Gemini CLI extension config creator.

Creates gemini-extension.json under basedir/mcpmaster from RELAY_CASES_CONFIG,
using Gem to enrich descriptions per Google guidelines.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# -----------------------------------------------------------------------------
# PROMPT (head comment - logic described for gem_cli_cfg_creator):
#
# 1. Resolve basedir: project root (parent of qbrain package) or cwd; any OS.
# 2. Load RELAY_CASES_CONFIG from qbrain.predefined_case.
# 3. For each case: extract case, desc, req_struct, out_struct (omit func).
# 4. Use Gem.ask() to enrich each case's "desc" into a concise, user-facing
#    description (1-2 sentences) following Google extension guidelines:
#    https://geminicli.com/docs/extensions/reference/#mcp-servers
# 5. Build gemini-extension.json per schema:
#    - name: qbrain-relay (lowercase, dashes)
#    - version: 1.0.0
#    - description: short summary of extension (Gem-enriched from cases)
#    - mcpServers: map with qbrain-relay server (command, args, cwd, ${extensionPath})
#    - contextFileName: GEMINI.md
#    - settings: GEMINI_API_KEY (sensitive), optional RELAY_WS_URL
# 6. Generate GEMINI.md context from cases (case names, enriched descs, req_struct keys).
# 7. Write gemini-extension.json and GEMINI.md to basedir/mcpmaster/.
# 8. Use pathlib for cross-OS paths; ${extensionPath} in JSON for portability.
# -----------------------------------------------------------------------------


def _basedir() -> Path:
    """Resolve project base directory (any OS)."""
    # Parent of qbrain package
    p = Path(__file__).resolve().parent.parent.parent
    if (p / "qbrain").is_dir():
        return p
    return Path.cwd()


def _serialize_struct(obj: Any) -> Any:
    """Convert req_struct/out_struct to JSON-serializable form (omit func)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _serialize_struct(v) for k, v in obj.items() if k != "func"}
    if isinstance(obj, (list, tuple)):
        return [_serialize_struct(x) for x in obj]
    if isinstance(obj, type):
        return obj.__name__ if hasattr(obj, "__name__") else str(obj)
    return obj


def _enrich_desc_with_gem(case_name: str, raw_desc: str, req_keys: List[str]) -> str:
    """Use Gem to produce a concise, user-facing description."""
    try:
        from qbrain.gem_core.gem import Gem
        gem = Gem()
        prompt = (
            f"Write a single concise sentence (max 15 words) for a CLI extension tool. "
            f"Case: {case_name}. Raw: '{raw_desc or 'no description'}'. "
            f"Required keys: {req_keys}. Output only the sentence, no quotes."
        )
        out = (gem.ask(content=prompt) or "").strip()
        return out[:120] if out else (raw_desc or case_name)
    except Exception:
        return raw_desc or case_name


def _flatten_req_keys(req_struct: Dict) -> List[str]:
    """Extract required key names from req_struct (auth, data)."""
    keys = []
    for section in (req_struct.get("auth") or {}, req_struct.get("data") or {}):
        if isinstance(section, dict):
            keys.extend(section.keys())
    return keys


class GeminiCfgCreator:
    """
    Creates Gemini CLI extension config from RELAY_CASES_CONFIG.
    Uses Gem to enrich descriptions per Google guidelines.
    """

    def __init__(self, basedir_path: Optional[Path] = None):
        self.basedir = Path(basedir_path) if basedir_path else _basedir()
        self.mcpmaster = self.basedir / "qbrain" / "mcpmaster"

    def gem_cli_cfg_creator(self, use_gem: bool = True, out_dir: Optional[Path] = None) -> Path:
        """Create gemini-extension.json and GEMINI.md. See module head comment for prompt logic."""
        return gem_cli_cfg_creator(
            basedir_path=self.basedir,
            use_gem=use_gem,
            out_dir=out_dir or self.mcpmaster,
        )


def gem_cli_cfg_creator(
    basedir_path: Optional[Path] = None,
    use_gem: bool = True,
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Create gemini-extension.json and GEMINI.md under basedir/mcpmaster.

    Extracts cases from RELAY_CASES_CONFIG, optionally enriches descriptions
    via Gem, and writes config per Google Gemini CLI extension schema.

    Returns:
        Path to the created gemini-extension.json.
    """
    base = Path(basedir_path) if basedir_path else _basedir()
    mcpmaster = Path(out_dir) if out_dir else base / "qbrain" / "mcpmaster"
    mcpmaster.mkdir(parents=True, exist_ok=True)

    from qbrain.predefined_case import RELAY_CASES_CONFIG

    cases_data: List[Dict[str, Any]] = []
    for c in RELAY_CASES_CONFIG:
        if not isinstance(c, dict) or "case" not in c:
            continue
        case_name = c.get("case", "")
        raw_desc = c.get("desc", "")
        req_struct = c.get("req_struct") or {}
        req_keys = _flatten_req_keys(req_struct)
        desc = _enrich_desc_with_gem(case_name, raw_desc, req_keys) if use_gem else (raw_desc or case_name)
        cases_data.append({
            "case": case_name,
            "desc": desc,
            "req_struct": _serialize_struct(req_struct),
            "out_struct": _serialize_struct(c.get("out_struct")),
        })

    # Extension description (summary)
    ext_desc = "QBrain relay cases as Gemini CLI extension: params, envs, modules, fields, methods, files, sessions, control engine, sim analysis, research."
    if use_gem and cases_data:
        try:
            from qbrain.gem_core.gem import Gem
            gem = Gem()
            prompt = (
                f"One sentence for a CLI extension description. "
                f"Tools: {[x['case'] for x in cases_data[:12]]}. "
                f"Output only the sentence."
            )
            ext_desc = (gem.ask(content=prompt) or ext_desc).strip()[:200]
        except Exception:
            pass

    # MCP server: Python stdio MCP if script exists
    mcp_script = mcpmaster / "mcp_server.py"
    mcp_servers: Dict[str, Any] = {}
    if mcp_script.exists():
        mcp_servers["qbrain-relay"] = {
            "command": "py",
            "args": ["${extensionPath}/mcp_server.py"],
            "cwd": "${extensionPath}",
        }

    cfg = {
        "name": "qbrain-relay",
        "version": "1.0.0",
        "description": ext_desc,
        "contextFileName": "GEMINI.md",
        "settings": [
            {
                "name": "Gemini API Key",
                "description": "API key for Gemini (used by QBrain extraction and enrichment).",
                "envVar": "GEMINI_API_KEY",
                "sensitive": True,
            },
            {
                "name": "Relay WebSocket URL",
                "description": "WebSocket URL for QBrain relay (optional).",
                "envVar": "RELAY_WS_URL",
                "sensitive": False,
            },
        ],
    }
    if mcp_servers:
        cfg["mcpServers"] = mcp_servers

    cfg_path = mcpmaster / "gemini-extension.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # GEMINI.md context
    lines = ["# QBrain Relay – Available Cases\n"]
    for d in cases_data:
        lines.append(f"## {d['case']}\n")
        lines.append(f"{d['desc']}\n")
        req_keys = _flatten_req_keys(d.get("req_struct") or {})
        if req_keys:
            lines.append("**Required:** " + ", ".join(req_keys) + "\n")
        lines.append("")
    gemini_md = mcpmaster / "GEMINI.md"
    with open(gemini_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return cfg_path


if __name__ == "__main__":
    out = gem_cli_cfg_creator(use_gem=os.environ.get("GEMINI_API_KEY") is not None)
    print(f"Created: {out}")
