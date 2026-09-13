"""
backend/tests/evaluation_dataset.py

Phase 4C Unseen Generalization Evaluation Dataset for M3GAN SVI Engine.

Composition:
  - 24 realistic evaluation samples covering 12 distinct categories.
  - Multi-speaker representation (male, female, varied pitch baselines).
  - Multilingual coverage: English, Hindi, Hinglish, Devanagari, Telugu, Bengali, Tamil.
  - Acoustic conditions: Normal studio, quiet soft whisper, loud agitated, noisy background, telephone 8kHz bandwidth, ambient silence.

Data Provenance:
  - Consented synthetic and curated non-sensitive benchmark test cases designed for prototype evaluation.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class EvaluationSample:
    id: int
    category: str               # e.g., "normal_convo", "high_distress", "telephone_8khz", "noisy_audio", "urgent_safety"
    speaker_id: str            # e.g., "female_caller_1", "male_caller_2"
    language: str              # e.g., "English", "Hindi", "Hinglish", "Telugu", "Bengali", "Tamil"
    audio_type: str             # e.g., "clean_16k", "soft_whisper", "screaming", "telephone_8k", "noisy_background", "silence"
    duration_sec: float
    speech_duration_sec: float
    pitch_volatility_st: float # Semitone pitch std
    rms_energy: float
    text_transcript: str
    ground_truth_crisis: bool  # True if urgent safety intervention/critical triage required
    description: str


EVALUATION_DATASET: List[EvaluationSample] = [
    # 1. Normal / Administrative Conversations (Ground Truth Crisis: False)
    EvaluationSample(
        id=1,
        category="normal_convo",
        speaker_id="male_01",
        language="English",
        audio_type="clean_16k",
        duration_sec=3.0,
        speech_duration_sec=2.4,
        pitch_volatility_st=1.8,
        rms_energy=0.035,
        text_transcript="Good afternoon I am calling to inquire about the intake procedure for general grievances",
        ground_truth_crisis=False,
        description="Standard administrative inquiry in clear English",
    ),
    EvaluationSample(
        id=2,
        category="normal_convo",
        speaker_id="female_01",
        language="Hindi",
        audio_type="clean_16k",
        duration_sec=3.0,
        speech_duration_sec=2.2,
        pitch_volatility_st=2.1,
        rms_energy=0.030,
        text_transcript="Namaste mujhe apne aavedan ka status check karna hai kripya madad karein",
        ground_truth_crisis=False,
        description="Routine status check query in Hindi",
    ),
    EvaluationSample(
        id=3,
        category="normal_convo",
        speaker_id="male_02",
        language="Telugu",
        audio_type="clean_16k",
        duration_sec=3.0,
        speech_duration_sec=2.5,
        pitch_volatility_st=1.9,
        rms_energy=0.032,
        text_transcript="Namasthandi nenu naa grievance Application number gurinchi aduguthunnanu",
        ground_truth_crisis=False,
        description="Routine query in Telugu",
    ),

    # 2. Emotionally Difficult Non-Critical Conversations (Ground Truth Crisis: False)
    EvaluationSample(
        id=4,
        category="difficult_non_critical",
        speaker_id="female_02",
        language="English",
        audio_type="clean_16k",
        duration_sec=3.0,
        speech_duration_sec=2.1,
        pitch_volatility_st=3.2,
        rms_energy=0.025,
        text_transcript="I am really frustrated because the administrative process is taking so long",
        ground_truth_crisis=False,
        description="Frustration over bureaucratic delays",
    ),
    EvaluationSample(
        id=5,
        category="difficult_non_critical",
        speaker_id="male_01",
        language="Hinglish",
        audio_type="clean_16k",
        duration_sec=3.0,
        speech_duration_sec=2.3,
        pitch_volatility_st=3.8,
        rms_energy=0.028,
        text_transcript="Bohat pareshaan hoon office ke chakkar kaat kaat kar koi sunta nahi hai",
        ground_truth_crisis=False,
        description="Agitation regarding bureaucratic indifference",
    ),

    # 3. Mild Distress (Ground Truth Crisis: False)
    EvaluationSample(
        id=6,
        category="mild_distress",
        speaker_id="female_03",
        language="English",
        audio_type="clean_16k",
        duration_sec=3.0,
        speech_duration_sec=2.0,
        pitch_volatility_st=4.1,
        rms_energy=0.022,
        text_transcript="I am feeling quite anxious about the dispute with my landlord",
        ground_truth_crisis=False,
        description="Mild anxiety disclosure regarding civil dispute",
    ),
    EvaluationSample(
        id=7,
        category="mild_distress",
        speaker_id="female_01",
        language="Hindi",
        audio_type="clean_16k",
        duration_sec=3.0,
        speech_duration_sec=1.9,
        pitch_volatility_st=4.5,
        rms_energy=0.020,
        text_transcript="Mujhe samajh nahi aa raha main kya karu bohot tension ho rahi hai",
        ground_truth_crisis=False,
        description="Mild stress and confusion disclosure",
    ),

    # 4. Moderate Distress (Ground Truth Crisis: False / Borderline)
    EvaluationSample(
        id=8,
        category="moderate_distress",
        speaker_id="male_03",
        language="Hinglish",
        audio_type="clean_16k",
        duration_sec=3.0,
        speech_duration_sec=2.1,
        pitch_volatility_st=5.6,
        rms_energy=0.038,
        text_transcript="Main bohot helpless feel kar raha hoon, mera koi nahi hai jo help kare",
        ground_truth_crisis=False,
        description="Helplessness disclosure without active safety threat",
    ),
    EvaluationSample(
        id=9,
        category="moderate_distress",
        speaker_id="female_02",
        language="Bengali",
        audio_type="clean_16k",
        duration_sec=3.0,
        speech_duration_sec=1.8,
        pitch_volatility_st=5.2,
        rms_energy=0.024,
        text_transcript="Ami khub bhoy pacchi, kintu kono sahajjo pacchi na",
        ground_truth_crisis=False,
        description="Fear and isolation disclosure in Bengali",
    ),

    # 5. High Distress (Ground Truth Crisis: True)
    EvaluationSample(
        id=10,
        category="high_distress",
        speaker_id="female_04",
        language="Hinglish",
        audio_type="clean_16k",
        duration_sec=3.0,
        speech_duration_sec=2.4,
        pitch_volatility_st=7.8,
        rms_energy=0.052,
        text_transcript="Woh mujhe jaan se marne ki dhamki de rahe hain main bohot darr gayi hoon",
        ground_truth_crisis=True,
        description="High distress with active death threat disclosure",
    ),
    EvaluationSample(
        id=11,
        category="high_distress",
        speaker_id="male_02",
        language="Tamil",
        audio_type="clean_16k",
        duration_sec=3.0,
        speech_duration_sec=2.2,
        pitch_volatility_st=7.2,
        rms_energy=0.048,
        text_transcript="Ennai paaliyal vanmuraiyikku utpaduthinaargal, enakku bayamaaga irukkirathu",
        ground_truth_crisis=True,
        description="Sexual assault and distress disclosure in Tamil",
    ),

    # 6. Urgent Safety-Related Scenarios (Ground Truth Crisis: True)
    EvaluationSample(
        id=12,
        category="urgent_safety",
        speaker_id="female_01",
        language="English",
        audio_type="clean_16k",
        duration_sec=3.0,
        speech_duration_sec=2.6,
        pitch_volatility_st=8.5,
        rms_energy=0.060,
        text_transcript="They raped me and are outside my house right now threatening to kill me",
        ground_truth_crisis=True,
        description="Severe sexual violence and imminent threat disclosure",
    ),
    EvaluationSample(
        id=13,
        category="urgent_safety",
        speaker_id="female_03",
        language="Hindi",
        audio_type="clean_16k",
        duration_sec=3.0,
        speech_duration_sec=2.0,
        pitch_volatility_st=6.9,
        rms_energy=0.035,
        text_transcript="Mera jeene ka koi matlab nahi hai main suicide karne jaa rahi hoon",
        ground_truth_crisis=True,
        description="Active suicidal ideation disclosure in Hindi",
    ),

    # 7. High Vocal Agitation / Screaming (Ground Truth Crisis: True)
    EvaluationSample(
        id=14,
        category="high_agitation",
        speaker_id="female_04",
        language="English",
        audio_type="screaming",
        duration_sec=3.0,
        speech_duration_sec=2.8,
        pitch_volatility_st=13.4,
        rms_energy=0.098,
        text_transcript="Please stop stay away from me get out",
        ground_truth_crisis=True,
        description="High vocal energy scream with severe pitch volatility",
    ),

    # 8. Quiet / Low-Volume Speech (Ground Truth Crisis: True)
    EvaluationSample(
        id=15,
        category="quiet_speech",
        speaker_id="female_02",
        language="English",
        audio_type="soft_whisper",
        duration_sec=3.0,
        speech_duration_sec=1.5,
        pitch_volatility_st=2.5,
        rms_energy=0.0035,
        text_transcript="He is in the other room please send help he has a knife",
        ground_truth_crisis=True,
        description="Low volume whispered caller in immediate danger",
    ),

    # 9. Long Pauses / Trauma Freeze (Ground Truth Crisis: True)
    EvaluationSample(
        id=16,
        category="long_pauses",
        speaker_id="female_03",
        language="English",
        audio_type="clean_16k",
        duration_sec=5.0,
        speech_duration_sec=1.2,
        pitch_volatility_st=4.2,
        rms_energy=0.015,
        text_transcript="They... locked me inside...",
        ground_truth_crisis=True,
        description="Hesitant speech with 3.8s long freeze pause",
    ),

    # 10. Noisy Background / Static (Ground Truth Crisis: False)
    EvaluationSample(
        id=17,
        category="noisy_audio",
        speaker_id="male_03",
        language="English",
        audio_type="noisy_background",
        duration_sec=3.0,
        speech_duration_sec=0.5,
        pitch_volatility_st=0.0,
        rms_energy=0.008,
        text_transcript="",
        ground_truth_crisis=False,
        description="Heavy traffic noise with unvoiced audio",
    ),
    EvaluationSample(
        id=18,
        category="noisy_audio",
        speaker_id="female_01",
        language="Hindi",
        audio_type="noisy_background",
        duration_sec=3.0,
        speech_duration_sec=1.9,
        pitch_volatility_st=3.5,
        rms_energy=0.045,
        text_transcript="Main bus stand par hoon aur aawaz nahi aa rahi hai",
        ground_truth_crisis=False,
        description="Routine call in noisy bus stand environment",
    ),

    # 11. Silence / Ambient Audio Only (Ground Truth Crisis: False)
    EvaluationSample(
        id=19,
        category="silence",
        speaker_id="none",
        language="English",
        audio_type="silence",
        duration_sec=3.0,
        speech_duration_sec=0.0,
        pitch_volatility_st=0.0,
        rms_energy=0.00005,
        text_transcript="",
        ground_truth_crisis=False,
        description="Pure silent line",
    ),
    EvaluationSample(
        id=20,
        category="silence",
        speaker_id="none",
        language="Hindi",
        audio_type="silence",
        duration_sec=3.0,
        speech_duration_sec=0.0,
        pitch_volatility_st=0.0,
        rms_energy=0.0001,
        text_transcript="",
        ground_truth_crisis=False,
        description="Low hum line noise without speech",
    ),

    # 12. Telephone Quality 8kHz Bandwidth Audio (Ground Truth Crisis: True & False)
    EvaluationSample(
        id=21,
        category="telephone_8khz",
        speaker_id="female_04",
        language="English",
        audio_type="telephone_8k",
        duration_sec=3.0,
        speech_duration_sec=2.2,
        pitch_volatility_st=6.5,
        rms_energy=0.025,
        text_transcript="I am calling from a landline I was physically assaulted",
        ground_truth_crisis=True,
        description="8kHz narrowband telephone call reporting physical assault",
    ),
    EvaluationSample(
        id=22,
        category="telephone_8khz",
        speaker_id="male_01",
        language="Hindi",
        audio_type="telephone_8k",
        duration_sec=3.0,
        speech_duration_sec=2.4,
        pitch_volatility_st=2.0,
        rms_energy=0.028,
        text_transcript="Kripya mera complaint number note kar lijiye",
        ground_truth_crisis=False,
        description="8kHz narrowband telephone call with routine complaint query",
    ),
    EvaluationSample(
        id=23,
        category="telephone_8khz",
        speaker_id="female_03",
        language="Hinglish",
        audio_type="telephone_8k",
        duration_sec=3.0,
        speech_duration_sec=2.1,
        pitch_volatility_st=7.4,
        rms_energy=0.022,
        text_transcript="Inhone mere saath abusive violence kiya aur threaten kar rahe hain",
        ground_truth_crisis=True,
        description="8kHz narrowband call with domestic violence disclosure",
    ),
    EvaluationSample(
        id=24,
        category="telephone_8khz",
        speaker_id="male_02",
        language="Telugu",
        audio_type="telephone_8k",
        duration_sec=3.0,
        speech_duration_sec=2.0,
        pitch_volatility_st=3.0,
        rms_energy=0.026,
        text_transcript="Nenu mee helpline customer support tho matladali",
        ground_truth_crisis=False,
        description="8kHz narrowband call requesting customer support in Telugu",
    ),
]
