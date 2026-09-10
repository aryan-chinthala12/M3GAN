from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class Channel(str, Enum):
    AUDIO = "audio"
    TEXT = "text"


class AcousticIndicators(BaseModel):
    top_emotion: str = Field(..., json_schema_extra={"example": "FEARFUL"})
    pitch_volatility: float = Field(..., description="Pitch std in semitones")
    rms_energy: float = Field(..., description="Mean RMS energy")
    energy_variation: Optional[float] = Field(None, description="RMS variation in dB")
    median_pitch_hz: Optional[float] = Field(None, description="Median F0 in Hz")
    confidence: float = Field(..., json_schema_extra={"example": 88.5})


class AcousticComponent(BaseModel):
    """Per-component SVI contribution with provenance (explainable AI)."""
    name: str = Field(..., description="Component name, e.g. acoustic, emotion, lexical")
    score: float = Field(..., description="0..1 normalized component score")
    weight: float = Field(..., description="Weight applied in final fusion")
    contribution: float = Field(..., description="Weighted 0..100 contribution to SVI")
    detail: Optional[str] = Field(None, description="Human-readable derivation note")


class NLPIndicators(BaseModel):
    threat_level: str = Field(..., json_schema_extra={"example": "HIGH"})
    flagged_keywords: List[str] = Field(default_factory=list)
    distress_categories: List[str] = Field(default_factory=list)
    pii_redactions: int = Field(0, description="Number of PII tokens redacted")


class TextIndicators(BaseModel):
    word_count: int = 0
    distress_density: float = 0.0
    emotion_label: Optional[str] = None
    emotion_confidence: Optional[float] = None


class SVIMetrics(BaseModel):
    final_svi_score: float = Field(..., json_schema_extra={"example": 84.2})
    risk_band: str = Field(..., json_schema_extra={"example": "CRITICAL"})
    risk_color: str = Field(..., json_schema_extra={"example": "#D32F2F"})
    override_triggered: bool = False
    override_reason: Optional[str] = None
    components: List[AcousticComponent] = Field(default_factory=list)


class Explainability(BaseModel):
    summary: str
    top_contributors: List[str]


class AudioAnalysisResponse(BaseModel):
    status: str = "success"
    case_id: Optional[str] = None
    filename: str
    channel: str = "audio"
    language: str = "English"
    duration_seconds: float
    transcript: str
    acoustic_indicators: AcousticIndicators
    nlp_indicators: NLPIndicators
    text_indicators: Optional[TextIndicators] = None
    svi_metrics: SVIMetrics
    explainability: Explainability
    recommended_interventions: List[str]
    consent_recorded: bool = False


class TextAnalysisRequest(BaseModel):
    """Body for POST /api/v1/analyze-text (chat, portal, chatbot narratives)."""
    text: str = Field(..., min_length=1, max_length=20000)
    channel: str = Field("chat", description="chat | portal | chatbot | ivrs | other")
    language: str = "English"
    case_id: Optional[str] = None


class TextAnalysisResponse(BaseModel):
    status: str = "success"
    case_id: Optional[str] = None
    channel: str = "chat"
    language: str = "English"
    redacted_text: str
    text_indicators: TextIndicators
    nlp_indicators: NLPIndicators
    svi_metrics: SVIMetrics
    explainability: Explainability
    recommended_interventions: List[str]
    consent_recorded: bool = False


class InterventionRequest(BaseModel):
    call_id: str
    operator_id: str
    action_taken: str
    assigned_protocol: Optional[str] = None
    notes: Optional[str] = None


class InterventionResponse(BaseModel):
    status: str = "recorded"
    call_id: str
    timestamp: str


class CaseSummary(BaseModel):
    """Compact case row for dashboard / case tables."""
    case_id: str
    created_at: str
    channel: str
    language: str
    svi_score: float
    risk_band: str
    status: str = "logged"
    preview: str = ""
    risk_color: str = "#2E7D32"


class StatsResponse(BaseModel):
    total_cases: int
    by_risk: Dict[str, int]
    by_channel: Dict[str, int]
    avg_svi: float
    critical_cases: int
