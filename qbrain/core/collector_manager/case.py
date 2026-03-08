"""
COLLECT_INFORMATION relay case.

First-class thalamic event for structured data gathering.
When user intent is to provide or collect information, route here.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from qbrain.core.managers_context import get_orchestrator


# ---- Handler ----

def handle_collect_information(data: Dict[str, Any], auth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle COLLECT_INFORMATION: prompt user for missing keys or return
    structured guidance for data collection.

    Uses orchestrator context to access goal_request_struct and generate
    contextual follow-up when target_case and missing_keys are provided.
    """
    print("handle_collect_information...")
    target_case = (data or {}).get("target_case")
    missing_keys = (data or {}).get("missing_keys") or []
    user_id = (auth or {}).get("user_id", "")

    orch = get_orchestrator()
    follow_up_msg = "What information would you like to provide?"

    if orch and target_case and missing_keys:
        goal_struct = getattr(orch, "goal_request_struct", {}) or {}
        case_goal = goal_struct.get(target_case, {})
        missing_str = ", ".join(missing_keys)
        conversation_history = []
        if hasattr(orch, "_get_conversation_history"):
            session_key = (auth or {}).get("session_id") or f"user_{user_id}"
            conversation_history = orch._get_conversation_history(session_key) or []

        if missing_str and hasattr(orch, "gem"):
            try:
                conv_str = ""
                if conversation_history:
                    recent = conversation_history[-3:]
                    conv_str = "\n\nRecent:\n" + "\n".join(
                        f"{m.get('role','')}: {m.get('content','')}" for m in recent
                    )
                prompt = f"""You are the Orchestrator Assistant. We need: {missing_str}.
                CASE: {target_case}
                Current: {json.dumps(case_goal, ensure_ascii=False)}
                {conv_str}
                Ask a concise follow-up question. Only the question."""
                follow_up_msg = (orch.gem.ask(content=prompt) or "").strip() or follow_up_msg
            except Exception as e:
                print(f"handle_collect_information: gem follow-up error: {e}")
                follow_up_msg = f"Please provide: {missing_str}?"

    result = {
        "type": "CHAT",
        "status": {"state": "success", "code": 200, "msg": ""},
        "data": {"msg": follow_up_msg},
    }
    print("handle_collect_information... done")
    return result


# ---- Case struct ----

COLLECT_INFORMATION_CASE: Dict[str, Any] = {
    "case": "COLLECT_INFORMATION",
    "desc": "Collect Information - gather required data from user for a target case",
    "func": handle_collect_information,
    "req_struct": {
        "auth": {"user_id": str, "session_id": str},
        "data": {"target_case": str, "missing_keys": "list"},
    },
    "out_struct": {"type": "CHAT", "data": {"msg": str}},
}

RELAY_COLLECT_INFORMATION = [COLLECT_INFORMATION_CASE]
