"""Modèles de données pour l'exécution fragmentée ODRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class RuntimeMode(str, Enum):
    SIMULATION = "simulation"
    CAMUNDA_7 = "camunda7"
    CAMUNDA_8 = "camunda8"


@dataclass
class ExecutionContext:
    """État runtime : variables métier, règles activées, historique."""

    variables: dict[str, Any] = field(default_factory=dict)
    enabled_rules: set[str] = field(default_factory=set)
    completed_activities: list[str] = field(default_factory=list)
    active_fragment_id: Optional[str] = None
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.variables[key] = value


@dataclass
class ActivityDecision:
    activity: str
    allowed: bool
    reason: str
    fragment_id: Optional[str] = None
    governing_rules: list[str] = field(default_factory=list)


@dataclass
class ExecutionStep:
    step_index: int
    activity: str
    fragment_id: str
    decision: ActivityDecision
    enabled_rules_after: list[str] = field(default_factory=list)
    variables_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    success: bool
    mode: RuntimeMode
    steps: list[ExecutionStep] = field(default_factory=list)
    final_context: Optional[ExecutionContext] = None
    error: Optional[str] = None
    camunda_process_instance_id: Optional[str] = None
    summary: dict[str, Any] = field(default_factory=dict)
