"""
SimResultAnalyzer: Analyze simulation results from env param time-series.

Queries envs where user_id=X and goal_id=Y, parses per-param JSON columns,
identifies events (convergence, divergence), and suggests parameter adaptations.

Every method follows: print("method_name...") at start, print("method_name... done") at end.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from qbrain.core.env_manager.env_lib import EnvManager
from qbrain.core.qbrain_manager import get_qbrain_table_manager
from qbrain.gem_core.gem import Gem


# ---- Data structures ----

@dataclass
class ParamSeriesEvent:
    """Identified event in a param time-series."""

    param_id: str
    event_type: str  # e.g. "convergence", "divergence", "threshold"
    description: str
    step_range: Optional[tuple[int, int]] = None
    value_range: Optional[tuple[float, float]] = None


@dataclass
class AnalysisResult:
    """Result of simulation result analysis."""

    env_id: str
    user_id: str
    goal_id: Optional[str]
    param_events: List[ParamSeriesEvent] = field(default_factory=list)
    suggested_adjustments: Dict[str, Any] = field(default_factory=dict)
    raw_param_series: Dict[str, Any] = field(default_factory=dict)


# ---- SimResultAnalyzer ----

class SimResultAnalyzer:
    """
    Analyzes env param time-series to recognize events and suggest parameter
    adaptations to reach goal env cfg.
    """

    def __init__(
        self,
        env_manager: Optional[EnvManager] = None,
        gem: Optional[Gem] = None,
    ):
        print("SimResultAnalyzer.__init__...")
        from qbrain.core.qbrain_manager import get_qbrain_table_manager
        self._qb = get_qbrain_table_manager()
        self.env_manager = env_manager or EnvManager(self._qb)
        self.gem = gem or Gem()
        print("SimResultAnalyzer.__init__... done")

    def analyze_envs_for_user_goal(
        self,
        user_id: str,
        goal_id: Optional[str] = None,
    ) -> List[AnalysisResult]:
        """
        Query envs for user_id and goal_id, parse param series, identify events.

        Returns list of AnalysisResult per env row.
        """
        print("analyze_envs_for_user_goal...")
        rows = self.env_manager.retrieve_envs_by_user_goal(
            user_id=user_id,
            goal_id=goal_id,
            select="*",
        )
        results: List[AnalysisResult] = []
        for row in rows:
            env_id = row.get("id", "")
            goal_id_val = row.get("goal_id")
            param_events = self._extract_events_from_row(row)
            raw_series = self._extract_param_series_from_row(row)
            results.append(
                AnalysisResult(
                    env_id=env_id,
                    user_id=user_id,
                    goal_id=goal_id_val,
                    param_events=param_events,
                    raw_param_series=raw_series,
                )
            )
        print("analyze_envs_for_user_goal... done")
        return results

    def _extract_param_series_from_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Parse per-param JSON columns (p_*) from env row."""
        print("_extract_param_series_from_row...")
        out: Dict[str, Any] = {}
        for k, v in row.items():
            if k.startswith("p_") and v:
                try:
                    parsed = json.loads(v) if isinstance(v, str) else v
                    if isinstance(parsed, dict) and ("values" in parsed or "features" in parsed):
                        out[k] = parsed
                except Exception:
                    pass
        print("_extract_param_series_from_row... done")
        return out

    def _extract_events_from_row(self, row: Dict[str, Any]) -> List[ParamSeriesEvent]:
        """Identify events (convergence, divergence) from param time-series."""
        print("_extract_events_from_row...")
        events: List[ParamSeriesEvent] = []
        for k, v in row.items():
            if not k.startswith("p_") or not v:
                continue
            try:
                parsed = json.loads(v) if isinstance(v, str) else v
                if not isinstance(parsed, dict):
                    continue
                vals = parsed.get("values") or []
                if not vals or not isinstance(vals, list):
                    continue
                num_vals = len(vals)
                if num_vals < 2:
                    continue
                first_half = vals[: num_vals // 2]
                second_half = vals[num_vals // 2:]
                mean_first = sum(float(x) for x in first_half) / len(first_half) if first_half else 0
                mean_second = sum(float(x) for x in second_half) / len(second_half) if second_half else 0
                var_first = sum((float(x) - mean_first) ** 2 for x in first_half) / len(first_half) if first_half else 0
                var_second = sum((float(x) - mean_second) ** 2 for x in second_half) / len(second_half) if second_half else 0
                if var_second < var_first * 0.5:
                    events.append(
                        ParamSeriesEvent(
                            param_id=k,
                            event_type="convergence",
                            description="Variance decreased in second half",
                            step_range=(0, num_vals),
                            value_range=(min(float(x) for x in vals), max(float(x) for x in vals)) if vals else None,
                        )
                    )
                elif var_second > var_first * 2:
                    events.append(
                        ParamSeriesEvent(
                            param_id=k,
                            event_type="divergence",
                            description="Variance increased in second half",
                            step_range=(0, num_vals),
                            value_range=(min(float(x) for x in vals), max(float(x) for x in vals)) if vals else None,
                        )
                    )
            except Exception:
                pass
        print("_extract_events_from_row... done")
        return events

    def compute_goal_score(
        self,
        analysis: AnalysisResult,
        goal_cfg: Dict[str, Any],
        tolerance: float = 0.1,
    ) -> float:
        """
        Compare final param values (last step of param evolution) to goal_cfg targets.
        Per-param: score_i = 1 - min(1, |actual - target| / tolerance).
        Returns mean score across params; 1.0 = all params match goal.
        """
        print("compute_goal_score...")
        if not goal_cfg or not analysis.raw_param_series:
            print("compute_goal_score... done")
            return 0.0

        scores: List[float] = []
        for param_col, series in analysis.raw_param_series.items():
            vals = series.get("values") if isinstance(series, dict) else []
            if not vals:
                continue
            actual = float(vals[-1])
            target = goal_cfg.get(param_col)
            if target is None:
                target = goal_cfg.get(param_col.replace("p_", ""))
            if target is None:
                continue
            try:
                target_f = float(target)
            except (TypeError, ValueError):
                continue
            diff = abs(actual - target_f)
            score_i = 1.0 - min(1.0, diff / tolerance)
            scores.append(score_i)

        result = sum(scores) / len(scores) if scores else 0.0
        print("compute_goal_score... done")
        return result

    def suggest_param_adjustments(
        self,
        analysis: AnalysisResult,
        goal_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compare current param values vs goal_cfg; suggest adjustments via Gem.
        """
        print("suggest_param_adjustments...")
        if not goal_cfg:
            print("suggest_param_adjustments... done")
            return {"suggestions": [], "reason": "no goal_cfg provided"}

        try:
            prompt = f"""You are a simulation parameter advisor.
            Env: {analysis.env_id}, Goal: {analysis.goal_id}
            Goal config (target): {json.dumps(goal_cfg, indent=2)}
            Param events observed: {[e.event_type for e in analysis.param_events]}
            Raw param series keys: {list(analysis.raw_param_series.keys())}
            Suggest concrete parameter adjustments (param_id, direction, magnitude hint) to reach the goal config.
            Return a JSON object with key "suggestions" as a list of {{"param_id": "...", "adjustment": "...", "reason": "..."}}.
            Return only valid JSON."""
            raw = self.gem.ask(content=prompt) or "{}"
            text = raw.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            analysis.suggested_adjustments = parsed
            print("suggest_param_adjustments... done")
            return parsed
        except Exception as e:
            print(f"suggest_param_adjustments: error: {e}")
            print("suggest_param_adjustments... done")
            return {"suggestions": [], "error": str(e)}
