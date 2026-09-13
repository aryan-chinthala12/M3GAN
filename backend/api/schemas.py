from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class Channel(str, Enum):
    AUDIO = "audio"
    TEXT = "text"


class SignalQuality(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    LOW_VOLUME = "LOW_VOLUME"
    WHISPER = "WHISPER"
    LOW_SNR = "LOW_SNR"
    UNRELIABLE = "UNRELIABLE"


class SafetyFlagItem(BaseModel):
    """Independent alert signal for human operator review (does NOT corrupt SVI score)."""
    category: str = Field(..., description="Trigger category, e.g. SUICIDAL_IDEATION, SEXUAL_VIOLENCE")
    severity: str = Field(..., description="Flag severity: CRITICAL | HIGH | MODERATE")
    flag: str = Field(..., description="Human action alert code, e.g. URGENT_HUMAN_REVIEW")
    message: str = Field(..., description="Human-readable justification note")
    matched_terms: List[str] = Field(default_factory=list, description="Surface terms detected")


class ConfidenceMetrics(BaseModel):
    """Overall assessment confidence and data-quality breakdown (analysis reliability)."""
    overall_confidence: float = Field(..., description="0..100% confidence in assessment reliability")
    confidence_rating: str = Field(..., description="HIGH | MEDIUM | LOW")
    signal_quality: str = Field(..., description="GOOD | DEGRADED | UNRELIABLE")
    speech_duration_seconds: float = Field(0.0, description="Total seconds of detected voiced speech")
    voiced_ratio: float = Field(0.0, description="Voiced speech ratio (0..1)")
    pause_ratio: float = Field(0.0, description="Silence/pause ratio (0..1)")
    pitch_reliable: bool = Field(True, description="True if pitch feature extraction is reliable")
    snr_state: str = Field("UNAVAILABLE", description="ESTIMATED | UNAVAILABLE")
    snr_db: Optional[float] = Field(None, description="Signal-to-noise ratio in dB if available")
    provenance: Dict[str, str] = Field(default_factory=dict)


class AcousticIndicators(BaseModel):
    top_emotion: str = Field(..., json_schema_extra={"example": "FEARFUL"})
    pitch_volatility: float = Field(..., description="Pitch std in semitones")
    rms_energy: float = Field(..., description="Mean RMS energy")
    energy_variation: Optional[float] = Field(None, description="RMS variation in dB")
    median_pitch_hz: Optional[float] = Field(None, description="Median F0 in Hz")
    confidence: float = Field(..., json_schema_extra={"example": 88.5})
    # Optional metadata
    voiced_ratio: Optional[float] = Field(None, description="Voiced speech ratio (0..1)")
    pause_ratio: Optional[float] = Field(None, description="Silence/pause ratio (0..1)")
    signal_quality: Optional[str] = Field(None, description="GOOD | DEGRADED | UNRELIABLE")
    pitch_reliable: Optional[bool] = Field(True, description="True if >= 5 voiced frames")


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
    safety_flags: List[SafetyFlagItem] = Field(default_factory=list)


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
    confidence_metrics: Optional[ConfidenceMetrics] = None
    safety_flags: List[SafetyFlagItem] = Field(default_factory=list)


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
    confidence_metrics: Optional[ConfidenceMetrics] = None
    safety_flags: List[SafetyFlagItem] = Field(default_factory=list)


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


# =============================================================
# PHASE 2 STREAMING SESSION & WEBSOCKET SCHEMAS
# =============================================================

class SessionStartRequest(BaseModel):
    channel: str = Field("voice", description="voice | phone | ivrs")
    language: str = Field("English", description="Primary audio language")
    consent: bool = Field(True, description="Informed consent flag")
    window_duration_sec: float = Field(3.0, description="Rolling analysis window size in seconds")
    hop_duration_sec: float = Field(0.5, description="Hop interval between analysis steps")


class SessionStartResponse(BaseModel):
    session_id: str
    status: str = "created"
    created_at: str
    websocket_url: str
    config: Dict[str, Any]


class SessionStopResponse(BaseModel):
    session_id: str
    case_id: str
    status: str = "stopped"
    duration_seconds: float
    final_svi_score: float
    final_risk_band: str
    final_transcript: str
    safety_flags: List[SafetyFlagItem] = Field(default_factory=list)
    created_at: str


class SessionSummaryResponse(BaseModel):
    session_id: str
    case_id: Optional[str] = None
    status: str
    duration_seconds: float
    created_at: str
    final_svi_score: float
    final_risk_band: str
    final_transcript: str
    svi_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    risk_history: List[Dict[str, Any]] = Field(default_factory=list)
    safety_flags: List[SafetyFlagItem] = Field(default_factory=list)
    confidence_metrics: Optional[ConfidenceMetrics] = None
    recommended_interventions: List[str] = Field(default_factory=list)


class StreamingSVI(BaseModel):
    raw_score: float = Field(..., description="Unsmoothed raw SVI score (0..100)")
    smoothed_score: float = Field(..., description="EMA smoothed live SVI score (0..100)")
    risk_band: str = Field(..., description="LOW | MODERATE | HIGH | CRITICAL")
    risk_color: str = Field(..., description="Hex color code for UI badge")
    trend: str = Field(..., description="INCREASING | DECREASING | STABLE | INSUFFICIENT_DATA")


class StreamingAssessmentMessage(BaseModel):
    """Structured, versioned JSON payload broadcast on WebSocket updates."""
    type: str = Field("assessment_update", description="Message type identifier")
    version: int = Field(1, description="Schema version number")
    session_id: str
    timestamp: str
    audio_duration: float
    transcript: str
    svi: StreamingSVI
    confidence: ConfidenceMetrics
    signal_quality: str
    emotion: Dict[str, Any]
    acoustics: Dict[str, Any]
    indicators: Dict[str, Any]
    safety_flags: List[SafetyFlagItem] = Field(default_factory=list)
    fast_path_latency_ms: float = Field(0.0, description="VAD & acoustic feature extraction latency in ms")
    heavy_path_latency_ms: float = Field(0.0, description="ASR + SER + NLP model inference latency in ms")
    end_to_end_latency_ms: float = Field(0.0, description="Total window processing latency in ms")
    latency_ms: float = Field(0.0, description="Alias for end_to_end_latency_ms")
    realtime_ratio: float = Field(0.0, description="End-to-end latency / Hop duration ratio (< 1.0 = real-time)")


class StreamingErrorMessage(BaseModel):
    type: str = Field("error", description="Message type")
    version: int = Field(1, description="Schema version")
    session_id: Optional[str] = None
    code: str
    message: str
    timestamp: str
