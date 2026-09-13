"""
backend/tests/evaluation_dataset_v2.py

Phase 5B Multilingual & Romanized NLP Evaluation Dataset for M3GAN SVI Engine.

Composition:
  - 32 realistic evaluation samples spanning 8 languages/scripts.
  - Indian languages: English, Devanagari Hindi, Romanized Hindi, Hinglish, Telugu, Tamil, Bengali, Marathi, Kannada.
  - Categories: Administrative Query, General Anxiety/Frustration, Suicidal Disclosure, Physical Violence/Threat, Sexual Violence, Domestic Abuse, Whispered Danger, Noisy Audio, Ambient Silence.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class EvaluationSampleV2:
    id: int
    category: str               # e.g., "admin_query", "general_anxiety", "suicidal_crisis", "violence_threat", "sexual_assault", "domestic_abuse"
    language: str              # "English", "Hindi", "Hinglish", "Telugu", "Tamil", "Bengali", "Marathi", "Kannada"
    script: str                # "LATIN", "DEVANAGARI", "BENGALI", "TAMIL", "TELUGU", "MARATHI", "KANNADA", "LATIN_MIXED"
    audio_type: str             # "clean_16k", "soft_whisper", "screaming", "telephone_8k", "noisy_background", "silence"
    text_transcript: str
    ground_truth_crisis: bool  # True if urgent safety intervention/critical triage required
    description: str


EVALUATION_DATASET_V2: List[EvaluationSampleV2] = [
    # 1. English Benchmark (N=6)
    EvaluationSampleV2(1, "admin_query", "English", "LATIN", "clean_16k", "Good afternoon I am calling to inquire about grievance tracking", False, "Standard query"),
    EvaluationSampleV2(2, "general_anxiety", "English", "LATIN", "clean_16k", "I am feeling frustrated about the delay in processing my application", False, "General administrative frustration"),
    EvaluationSampleV2(3, "suicidal_crisis", "English", "LATIN", "clean_16k", "I have no reason to live anymore I am going to end my life tonight", True, "Explicit suicidal ideation"),
    EvaluationSampleV2(4, "violence_threat", "English", "LATIN", "clean_16k", "They have a weapon outside my house threatening to kill me and my children", True, "Imminent death threat"),
    EvaluationSampleV2(5, "sexual_assault", "English", "LATIN", "clean_16k", "I was sexually assaulted last night and I am afraid to go home", True, "Sexual violence disclosure"),
    EvaluationSampleV2(6, "whispered_danger", "English", "LATIN", "soft_whisper", "He is in the hallway with a knife please send help softly", True, "Whispered immediate danger"),

    # 2. Native Devanagari Hindi (N=5)
    EvaluationSampleV2(7, "admin_query", "Hindi", "DEVANAGARI", "clean_16k", "नमस्ते मुझे अपने शिकायत का स्टेटस जानना है", False, "Deavanagari query"),
    EvaluationSampleV2(8, "general_anxiety", "Hindi", "DEVANAGARI", "clean_16k", "मुझे बहुत टेंशन हो रही है परीक्षा की तैयारी नहीं हुई", False, "Devanagari exam stress"),
    EvaluationSampleV2(9, "suicidal_crisis", "Hindi", "DEVANAGARI", "clean_16k", "मुझसे अब सहन नहीं होता, मैं मरना चाहती हूँ", True, "Devanagari suicidal ideation"),
    EvaluationSampleV2(10, "violence_threat", "Hindi", "DEVANAGARI", "clean_16k", "वे मुझे जान से मारने की धमकी दे रहे हैं", True, "Devanagari death threat"),
    EvaluationSampleV2(11, "sexual_assault", "Hindi", "DEVANAGARI", "clean_16k", "मेरे साथ बलात्कार हुआ है मुझे बचा लीजिए", True, "Devanagari rape disclosure"),

    # 3. Romanized Hindi (N=5)
    EvaluationSampleV2(12, "admin_query", "Hindi", "LATIN", "clean_16k", "Namaste mujhe apne aavedan ka status check karna hai", False, "Romanized Hindi query"),
    EvaluationSampleV2(13, "general_anxiety", "Hindi", "LATIN", "clean_16k", "Mujhe bohot tension ho rahi hai kya karu samajh nahi aa raha", False, "Romanized Hindi mild anxiety"),
    EvaluationSampleV2(14, "suicidal_crisis", "Hindi", "LATIN", "clean_16k", "Mera jeene ka koi matlab nahi hai main khudkushi karne jaa rahi hoon", True, "Romanized Hindi suicide threat"),
    EvaluationSampleV2(15, "violence_threat", "Hindi", "LATIN", "clean_16k", "Woh mujhe jaan se marne ki dhamki de rahe hain main bohot darr gayi hoon", True, "Romanized Hindi death threat"),
    EvaluationSampleV2(16, "sexual_assault", "Hindi", "LATIN", "clean_16k", "Unnone mere saath balatkar kiya aur dhamkee di hai", True, "Romanized Hindi rape disclosure"),

    # 4. Hinglish Code-Switching (N=4)
    EvaluationSampleV2(17, "admin_query", "Hinglish", "LATIN_MIXED", "clean_16k", "Helpline team please check status of my application reference number 123", False, "Hinglish query"),
    EvaluationSampleV2(18, "general_anxiety", "Hinglish", "LATIN_MIXED", "clean_16k", "Main bohot helpless feel kar raha hoon landlord se dispute hai", False, "Hinglish civil anxiety"),
    EvaluationSampleV2(19, "suicidal_crisis", "Hinglish", "LATIN_MIXED", "clean_16k", "Main completely hopeless hoon end my life is the only option left", True, "Hinglish suicide threat"),
    EvaluationSampleV2(20, "violence_threat", "Hinglish", "LATIN_MIXED", "clean_16k", "Unke paas gun hai and threatened to kill me right now", True, "Hinglish death threat"),

    # 5. Telugu Native & Romanized (N=3)
    EvaluationSampleV2(21, "admin_query", "Telugu", "LATIN", "clean_16k", "Namasthandi nenu naa Application status gurinchi aduguthunnanu", False, "Romanized Telugu query"),
    EvaluationSampleV2(22, "suicidal_crisis", "Telugu", "TELUGU", "clean_16k", "నాకు జీవించాలని లేదు, నేను ఆత్మహత్య చేసుకుంటాను", True, "Telugu native suicide threat"),
    EvaluationSampleV2(23, "violence_threat", "Telugu", "LATIN", "clean_16k", "Nannu champethanu ani kottaru, nenu bohot bayapaddanu", True, "Romanized Telugu violence threat"),

    # 6. Tamil Native & Romanized (N=3)
    EvaluationSampleV2(24, "admin_query", "Tamil", "LATIN", "clean_16k", "Vanakkam nani grievance details kekkuran", False, "Romanized Tamil query"),
    EvaluationSampleV2(25, "sexual_assault", "Tamil", "TAMIL", "clean_16k", "என்னை பாலியல் வன்முறைக்கு உட்படுத்தினார்கள்", True, "Tamil native rape disclosure"),
    EvaluationSampleV2(26, "violence_threat", "Tamil", "LATIN", "clean_16k", "Ennai kolai seivangannu adichanga enakku bayamaaga irukkirathu", True, "Romanized Tamil violence threat"),

    # 7. Bengali Native & Romanized (N=3)
    EvaluationSampleV2(27, "admin_query", "Bengali", "BENGALI", "clean_16k", "আমি আমার অভিযোগের নতুন তথ্য জানতে চাই", False, "Bengali query"),
    EvaluationSampleV2(28, "suicidal_crisis", "Bengali", "BENGALI", "clean_16k", "আমি আত্মহত্যা করতে চাই আমার কিছু নেই", True, "Bengali native suicide threat"),
    EvaluationSampleV2(29, "violence_threat", "Bengali", "LATIN", "clean_16k", "Amake mere felbe bole dhamki dichche khub bhoy pacchi", True, "Romanized Bengali threat"),

    # 8. Marathi & Kannada (N=3)
    EvaluationSampleV2(30, "suicidal_crisis", "Marathi", "MARATHI", "clean_16k", "मी स्वतःला संपवणार आहे मला मदत करा", True, "Marathi suicide disclosure"),
    EvaluationSampleV2(31, "violence_threat", "Kannada", "KANNADA", "clean_16k", "ನನ್ನನ್ನು ಕೊಲೆ ಮಾಡುತ್ತೇವೆ ಎಂದು ಹೆದರಿಸುತ್ತಿದ್ದಾರೆ", True, "Kannada death threat"),
    EvaluationSampleV2(32, "silence", "English", "LATIN", "silence", "", False, "Ambient silent line"),
]
