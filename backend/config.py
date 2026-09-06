# backend/config.py

# Weights for ML vs Acoustic DSP fusion
WEIGHT_ML_EMOTION = 0.60    # 60% weight on Random Forest emotion probability
WEIGHT_ACOUSTIC = 0.40      # 40% weight on pitch variance / vocal strain

# Feature Normalization Baseline (Pitch variance limit in Hz)
MAX_EXPECTED_PITCH_VAR = 80.0

# Action Rules Mapping for Risk Levels
RISK_RULES = {
    "CRITICAL": {
        "min_score": 75.0,
        "color": "#D50000",
        "actions": [
            "Immediate Priority Transfer to Senior Trauma Counselor",
            "Auto-dispatch Emergency Alert to District Police Desk",
            "Enable line-tracing protocol and emergency logging"
        ]
    },
    "HIGH": {
        "min_score": 50.0,
        "color": "#FF6D00",
        "actions": [
            "Escalate call to Senior Duty Supervisor",
            "Flag profile for Immediate Psychological First Aid (PFA)",
            "Queue parallel dispatch for local legal aid officer"
        ]
    },
    "MODERATE": {
        "min_score": 25.0,
        "color": "#FFD600",
        "actions": [
            "Route to standard counseling intake queue",
            "Schedule follow-up welfare check within 24 hours"
        ]
    },
    "LOW": {
        "min_score": 0.0,
        "color": "#00C853",
        "actions": [
            "Standard grievance intake and administrative logging"
        ]
    }
}