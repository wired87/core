from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class BrainNodeType:
    USER = "USER"
    GOAL = "GOAL"
    SUB_GOAL = "SUB_GOAL"
    SHORT_TERM_STORAGE = "SHORT_TERM_STORAGE"
    LONG_TERM_STORAGE = "LONG_TERM_STORAGE"
    CONTENT = "CONTENT"


class BrainEdgeRel:
    DERIVED_FROM = "derived_from"
    REQUIRES = "requires"
    SATISFIES = "satisfies"
    REFERENCES_TABLE_ROW = "references_table_row"
    FOLLOWS = "follows"
    PARENT_OF = "parent_of"


@dataclass
class GoalDecision:
    case_name: str
    confidence: float
    source: str
    reason: str = ""
    req_struct: Dict[str, Any] = field(default_factory=dict)
    out_struct: Dict[str, Any] = field(default_factory=dict)
    case_item: Optional[Dict[str, Any]] = None


@dataclass
class DataCollectionResult:
    resolved: Dict[str, Any]
    missing: List[str]

