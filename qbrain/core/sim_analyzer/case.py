"""
ANALYZE_SIM_RESULTS relay case.

Analyzes simulation results from env param time-series (user_id, goal_id).
"""
from __future__ import annotations

import json
from typing import Any, Dict

from qbrain.core.managers_context import get_orchestrator
from qbrain.core.sim_analyzer import SimResultAnalyzer


def handle_analyze_sim_results(data: Dict[str, Any], auth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle ANALYZE_SIM_RESULTS: analyze env param series for user_id and goal_id,
    identify events, suggest parameter adjustments.
    """
    print("handle_analyze_sim_results...")
    user_id = (auth or {}).get("user_id", "")
    goal_id = (data or {}).get("goal_id")
    goal_cfg_raw = (data or {}).get("goal_cfg")
    goal_cfg = None
    if goal_cfg_raw:
        try:
            goal_cfg = json.loads(goal_cfg_raw) if isinstance(goal_cfg_raw, str) else goal_cfg_raw
        except Exception:
            pass

    if not user_id:
        print("handle_analyze_sim_results... done")
        return {
            "type": "ANALYZE_SIM_RESULTS",
            "status": {"state": "error", "code": 400, "msg": "user_id required"},
            "data": {"error": "user_id required"},
        }

    try:
        orch = get_orchestrator()
        env_mgr = orch.env_manager if orch else None
        gem = orch.gem if orch else None
        analyzer = SimResultAnalyzer(env_manager=env_mgr, gem=gem)
        results = analyzer.analyze_envs_for_user_goal(user_id=user_id, goal_id=goal_id)

        out_data: Dict[str, Any] = {
            "analyses": [],
            "suggestions": [],
        }
        for ar in results:
            if goal_cfg:
                suggestions = analyzer.suggest_param_adjustments(ar, goal_cfg=goal_cfg)
                out_data["suggestions"].append(
                    {"env_id": ar.env_id, "goal_id": ar.goal_id, **suggestions}
                )
            out_data["analyses"].append(
                {
                    "env_id": ar.env_id,
                    "goal_id": ar.goal_id,
                    "param_events": [
                        {
                            "param_id": e.param_id,
                            "event_type": e.event_type,
                            "description": e.description,
                        }
                        for e in ar.param_events
                    ],
                }
            )

        print("handle_analyze_sim_results... done")
        return {
            "type": "ANALYZE_SIM_RESULTS",
            "status": {"state": "success", "code": 200, "msg": ""},
            "data": out_data,
        }
    except Exception as e:
        print(f"handle_analyze_sim_results: error: {e}")
        print("handle_analyze_sim_results... done")
        return {
            "type": "ANALYZE_SIM_RESULTS",
            "status": {"state": "error", "code": 500, "msg": str(e)},
            "data": {"error": str(e)},
        }


ANALYZE_SIM_RESULTS_CASE: Dict[str, Any] = {
    "case": "ANALYZE_SIM_RESULTS",
    "desc": "Analyze simulation results - learn from param time-series and suggest adaptations",
    "func": handle_analyze_sim_results,
    "req_struct": {
        "auth": {"user_id": str},
        "data": {"goal_id": str, "goal_cfg": str},
    },
    "out_struct": {"type": "ANALYZE_SIM_RESULTS", "data": {"analyses": list, "suggestions": list}},
}

RELAY_ANALYZE_SIM_RESULTS = [ANALYZE_SIM_RESULTS_CASE]
