# backend/config.py
# Single source of truth for SVI fusion weights, risk bands and actions.

# Component weights for AUDIO analysis (sum to 1.0 across primary components)
WEIGHT_ML_EMOTION = 0.45
WEIGHT_ACOUSTIC = 0.30
WEIGHT_LEXICAL = 0.25

# Lexicon contribution cap (bounded, repetition saturates)
LEXICAL_SATURATION_CAP = 1.0

# Risk bands: thresholds, UI colors and protocol actions.
# PS requirement: Low / Moderate / High / Critical.
RISK_RULES = {
    "CRITICAL": {
        "min_score": 75.0,
        "color": "#D50000",
        "actions": [
            "Immediate Priority Transfer to Senior Trauma Counselor",
            "Auto-dispatch Emergency Alert to District Police Desk",
            "Enable line-tracing protocol and emergency logging",
        ],
    },
    "HIGH": {
        "min_score": 50.0,
        "color": "#FF6D00",
        "actions": [
            "Escalate call to Senior Duty Supervisor",
            "Flag profile for Immediate Psychological First Aid (PFA)",
            "Queue parallel dispatch for local legal aid officer",
        ],
    },
    "MODERATE": {
        "min_score": 25.0,
        "color": "#FFD600",
        "actions": [
            "Route to standard counseling intake queue",
            "Schedule follow-up welfare check within 24 hours",
        ],
    },
    "LOW": {
        "min_score": 0.0,
        "color": "#00C853",
        "actions": [
            "Standard grievance intake and administrative logging",
        ],
    },
}

# Legacy compatibility map used by the current frontend
THREAT_LEVEL_MAP = {
    "LOW": "LOW",
    "MODERATE": "MEDIUM",
    "HIGH": "HIGH",
    "CRITICAL": "HIGH",
}

# Component weight -> expansion factor e in [0, 1].
# Component scores are passed through f(x) = x ** g where g = 1 - e/2,
# a monotonic expansion with f(0)=0 and f(1)=1. Higher expansion
# lifts low-intensity signals so meaningful distress is not drowned
# out by the fusion weights, while extremes remain well separated.
EXPANSION_FACTORS = {
    0.20: 0.60,
    0.25: 0.60,
    0.30: 0.50,
    0.35: 0.50,
    0.40: 0.45,
    0.45: 0.45,
    0.50: 0.40,
    0.55: 0.40,
    0.60: 0.35,
    0.65: 0.35,
    0.70: 0.30,
    0.80: 0.30,
    0.85: 0.25,
    0.90: 0.25,
    1.00: 0.20,
}


def expansion_factor(weight: float) -> float:
    """Look up the expansion factor for a component weight."""
    if weight in EXPANSION_FACTORS:
        return EXPANSION_FACTORS[weight]
    # nearest-key fallback
    nearest = min(EXPANSION_FACTORS, key=lambda k: abs(k - weight))
    return EXPANSION_FACTORS[nearest]
