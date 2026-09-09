# backend/config.py

"""
Configuration for the Speech Vulnerability Index (SVI).

SVI is a multimodal distress-indicator score from 0-100.
It is NOT a clinical diagnosis or validated psychological scale.
"""

# =============================================================
# SVI COMPONENT WEIGHTS
# =============================================================

# Speech emotion contributes 45%
WEIGHT_EMOTION = 0.45

# Acoustic characteristics contribute 30%
WEIGHT_ACOUSTIC = 0.30

# Linguistic / lexical distress contributes 25%
WEIGHT_LINGUISTIC = 0.25


# =============================================================
# ACOUSTIC NORMALIZATION
# =============================================================

# Pitch volatility is measured as standard deviation in semitones.
#
# A value around 2.5 semitones is treated as the upper end of
# the normalisation range for this heuristic.
#
# This is NOT a clinical threshold.
PITCH_VOLATILITY_NORMALIZATION = 2.5


# RMS energy depends heavily on microphone distance and recording
# volume, therefore it receives a relatively small contribution.
RMS_NORMALIZATION = 0.10

# Pitch is the more informative acoustic feature; RMS is deliberately
# secondary because it is sensitive to microphone placement and gain.
ACOUSTIC_PITCH_WEIGHT = 0.70
ACOUSTIC_RMS_WEIGHT = 0.30


# Energy variation is another supporting acoustic feature.
ENERGY_VARIATION_NORMALIZATION = 12.0


# =============================================================
# RISK BAND THRESHOLDS
# =============================================================

RISK_RULES = {
    "CRITICAL": {
        "min_score": 75.0,
        "color": "#D50000",
        "actions": [
            "Immediate Priority Transfer to Senior Trauma Counselor",
            "Escalate case for emergency safety assessment",
            "Enable emergency logging and supervisor notification",
        ],
    },

    "HIGH": {
        "min_score": 50.0,
        "color": "#FF6D00",
        "actions": [
            "Escalate call to Senior Duty Supervisor",
            "Flag profile for Immediate Psychological First Aid (PFA)",
            "Prioritize counselor follow-up",
        ],
    },

    "MODERATE": {
        "min_score": 25.0,
        "color": "#FFD600",
        "actions": [
            "Route to standard counseling intake queue",
            "Schedule follow-up welfare check",
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
