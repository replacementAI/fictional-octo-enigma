from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


ModalityName = Literal["tap", "voice", "spiral"]


@dataclass
class ModelMetadata:
    input_quality: float | None
    sample_count: int | None
    warnings: list[str] = field(default_factory=list)
    feature_schema_version: str = "v1"


@dataclass
class Contributor:
    feature: str
    value: float | int | str | None
    direction: Literal["higher_risk", "lower_risk", "neutral"]


@dataclass
class ModelExplanation:
    top_contributors: list[Contributor] = field(default_factory=list)


@dataclass
class ModelSummary:
    primary_signal: str | None = None
    severity: Literal["low", "moderate", "high"] | None = None


@dataclass
class ModalityResult:
    modality: ModalityName
    model_version: str
    risk_score: float
    raw_features: dict[str, Any]
    metadata: ModelMetadata
    confidence: float | None = None
    summary: ModelSummary | None = None
    explanations: ModelExplanation | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "modality": self.modality,
            "model_version": self.model_version,
            "risk_score": self.risk_score,
            "raw_features": self.raw_features,
            "metadata": asdict(self.metadata),
            "confidence": self.confidence,
            "summary": asdict(self.summary) if self.summary is not None else None,
            "explanations": asdict(self.explanations) if self.explanations is not None else None,
        }
        return payload


def clamp_score(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def build_modality_result(
    modality: ModalityName,
    model_version: str,
    risk_score: float,
    raw_features: dict[str, Any],
    *,
    confidence: float | None = None,
    input_quality: float | None = None,
    sample_count: int | None = None,
    warnings: list[str] | None = None,
    feature_schema_version: str = "v1",
    primary_signal: str | None = None,
    severity: Literal["low", "moderate", "high"] | None = None,
    top_contributors: list[Contributor] | None = None,
) -> dict[str, Any]:
    summary = None
    if primary_signal is not None or severity is not None:
        summary = ModelSummary(primary_signal=primary_signal, severity=severity)

    explanations = None
    if top_contributors:
        explanations = ModelExplanation(top_contributors=top_contributors)

    result = ModalityResult(
        modality=modality,
        model_version=model_version,
        risk_score=clamp_score(risk_score),
        raw_features=raw_features,
        metadata=ModelMetadata(
            input_quality=input_quality,
            sample_count=sample_count,
            warnings=warnings or [],
            feature_schema_version=feature_schema_version,
        ),
        confidence=confidence,
        summary=summary,
        explanations=explanations,
    )
    return result.to_dict()
