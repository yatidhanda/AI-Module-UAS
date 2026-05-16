from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ThreatModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ThreatAssessment(ThreatModel):
    score: float = Field(ge=0.0, le=100.0)
    level: str
    label: str
    category: str | None = None
    rationale: str
    contributing_factors: list[str]
    recommended_operator_actions: list[str]
    human_review_required: bool = True


class ThreatAlert(ThreatModel):
    alert_id: str
    timestamp_s: float | None = None
    frame_id: int | None = None
    track_id: str
    object_class: str
    threat_level: str
    threat_score: float = Field(ge=0.0, le=100.0)
    summary: str
    recommended_actions: list[str]
    human_review_required: bool = True
