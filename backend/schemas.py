from pydantic import BaseModel, Field
from typing import List, Optional

class AcousticIndicators(BaseModel):
    top_emotion: str = Field(..., json_schema_extra={"example": "FEARFUL"})
    pitch_volatility: float = Field(..., json_schema_extra={"example": 68.4})
    rms_energy: float = Field(..., json_schema_extra={"example": 0.042})
    confidence: float = Field(..., json_schema_extra={"example": 88.5})

class NLPIndicators(BaseModel):
    detected_emotion: str = Field(..., json_schema_extra={"example": "SADNESS"})
    threat_level: str = Field(..., json_schema_extra={"example": "HIGH"})
    flagged_keywords: List[str] = Field(
        default_factory=list, 
        json_schema_extra={"example": ["giving up", "no way out"]}
    )

class SVIMetrics(BaseModel):
    final_svi_score: float = Field(..., json_schema_extra={"example": 84.2})
    risk_band: str = Field(..., json_schema_extra={"example": "CRITICAL"})
    risk_color: str = Field(..., json_schema_extra={"example": "#D32F2F"})
    override_triggered: bool = Field(..., json_schema_extra={"example": True})
    override_reason: Optional[str] = Field(
        None, 
        json_schema_extra={"example": "High-risk distress keyword detected in transcript"}
    )

class Explainability(BaseModel):
    summary: str
    top_contributors: List[str]

class AudioAnalysisResponse(BaseModel):
    status: str = Field("success")
    filename: str
    duration_seconds: float
    transcript: str
    acoustic_indicators: AcousticIndicators
    nlp_indicators: NLPIndicators
    svi_metrics: SVIMetrics
    explainability: Explainability
    recommended_interventions: List[str]

class InterventionRequest(BaseModel):
    call_id: str
    operator_id: str
    action_taken: str
    assigned_protocol: str
    notes: Optional[str] = None

class InterventionResponse(BaseModel):
    status: str = "recorded"
    call_id: str
    timestamp: str