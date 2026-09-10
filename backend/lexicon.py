"""
backend/lexicon.py

Multilingual distress lexicon for the NHAA Stress Vulnerability Index engine.

Design goals
------------
1. Language coverage: major Indian languages used on NHAA channels
   (English, Hindi, Bengali, Tamil, Telugu, Marathi, Kannada) plus
   Romanized / code-mixed variants (very common on real helpline calls,
   e.g. "mujhe darr lag raha hai" or "koi madad nahi kar raha").
2. Category based scoring: every term belongs to a clinical-style
   indicator category (suicidal ideation, violence threat, sexual
   violence, hopelessness, intimidation, social boycott, ...) with a
   weight. The engine consumes these weights; nothing in here forces
   a risk band by itself.
3. Word-boundary regex matching so substrings like "helpful" do not
   trigger "help".
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Tuple


@dataclass(frozen=True)
class DistressCategory:
    """A weighted indicator category with multilingual surface terms."""

    id: str
    label: str
    weight: float          # 0..1 contribution strength per matched term
    floor: float           # optional SVI floor (0..100) when this category hits
    terms: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Term definitions (native script + Romanized variants)
# ---------------------------------------------------------------------------

SUICIDAL_IDEATION = DistressCategory(
    id="SUICIDAL_IDEATION",
    label="Suicidal ideation / self-endangerment",
    weight=1.00,
    floor=78.0,
    terms=(
        # English
        "suicide", "kill myself", "end my life", "want to die",
        "no reason to live", "take my own life", "not worth living",
        # Hindi (Devanagari + Romanized)
        "आत्महत्या", "खुदकुशी", "जान देना", "मरना चाहता", "मरना चाहती",
        "aatmahatya", "khudkushi", "khudkhushi", "jaan de dunga",
        "jaan de dungi", "marna chahta", "marana chahati",
        # Bengali
        "আত্মহত্যা", "মরতে চাই", "atmohotta", "morte chai",
        # Tamil
        "தற்கொலை", "சாக வேண்டும்", "tharkolai", "saaga vendum",
        # Telugu
        "ఆత్మహత్య", "చస్తా", "atmahatya", "chastha", "chachipota",
        # Marathi
        "आत्महत्या", "मरण्याला", "atmahatya", "maranyala",
        # Kannada
        "ಆತ್ಮಹತ್ಯೆ", "ಸಾಯುತ್ತೇನೆ", "atmahatye", "saayuttene",
    ),
)

SELF_HARM = DistressCategory(
    id="SELF_HARM",
    label="Self-harm indicators",
    weight=0.90,
    floor=75.0,
    terms=(
        "harm myself", "hurt myself", "cut myself", "hang myself",
        "overdose", "poison",
        "खुद को नुकसान", "जहर", "फांसी",
        "khud ko nuksan", "zahar", "zeher", "phans",
        "নিজেকে ক্ষতি", "বিষ", "nijer khoti", "bish",
        "தன்னை காயம்", "விஷம்", "visham",
        "విషం", "visham",
        "स्वतःला इजा", "विष", "vish",
        "ಸ್ವತಃ ಗಾಯ", "ವಿಷ", "visha",
    ),
)

SEXUAL_VIOLENCE = DistressCategory(
    id="SEXUAL_VIOLENCE",
    label="Sexual violence disclosure",
    weight=0.80,
    floor=55.0,
    terms=(
        "rape", "raped", "gang rape", "gangrape", "molested",
        "sexual assault", "sexually assaulted",
        "बलात्कार", "दुष्कर्म", "छेड़छाड़", "सामूहिक बलात्कार",
        "balatkar", "dushkarm", "chhedchhad", "gass",
        "ধর্ষণ", "গণধর্ষণ", "dhorthon", "gondhorthon",
        "பாலியல் வன்முறை", "கற்பழிப்பு", "pazhuppu", "kalavippu",
        "అత్యాచారం", "బలాత్కారం", "atyacharam", "balatkaaram",
        "बलात्कार", "अत्याचार", "balatkar",
        "ಅತ್ಯಾಚಾರ", "ಬಲಾತ್ಕಾರ", "atyaachara", "balaatkaara",
    ),
)

VIOLENCE_THREAT = DistressCategory(
    id="VIOLENCE_THREAT",
    label="Violence / death threat to victim or family",
    weight=0.85,
    floor=52.0,
    terms=(
        "kill me", "killed", "murder", "murdered", "beat me",
        "beaten", "they will kill", "threat to kill", "attacked",
        "broken my", "burned my house", "burnt my house",
        "मार डालेंगे", "मार दिया", "मारपीट", "पीटा", "हत्या", "खून",
        "मेरे घर में आग",
        "maar dalenge", "mar dalenge", "maar diya", "maar peet",
        "peeta", "hatya", "khoon", "kaat dalenge", "maarenge",
        "খুন", "পিটিয়েছে", "মেরে ফেলবে", "khun", "piteche", "mere felbe",
        "கொலை", "அடிச்சாங்க", "அடித்தாங்க", "kolai", "adichanga", "adithanga",
        "చంపేస్తారు", "కొట్టారు", "హత్య", "champesaru", "kottaru", "hatya",
        "मारणार", "मारलं", "मारहाणी", "खून", "maranar", "marla", "marhani",
        "ಕೊಂದು", "ಹೊಡೆದರು", "ಕೊಲೆ", "kondru", "hodedaru", "kole",
    ),
)

TRAUMA_SHOCK = DistressCategory(
    id="TRAUMA_SHOCK",
    label="Acute trauma / physical shock",
    weight=0.60,
    floor=0.0,
    terms=(
        "bleeding", "blood", "can't breathe", "cannot breathe",
        "shaking", "trembling", "in shock", "hospital", "injured",
        "unconscious",
        "खून बह", "सांस नहीं", "कांप", "हिल रह", "बेहोश",
        "khoon bah", "saans nahi", "kaamp", "hil raha", "hil rahi", "behosh",
        "রক্ত", "শ্বাস", "কাঁপছে", "rokto", "shash", "kanpche",
        "ரத்தம்", "மூச்சு", "நடுங்க", "ratham", "moochu", "nadung",
        "రక్తం", "ఊపిరి", "వణుకు", "raktam", "oopiri", "vanuku",
        "रक्त", "लहू", "श्वास", "भिबत", "rag", "lahu", "shwas",
        "ರಕ್ತ", "ಉಸಿರು", "ನಡುಗ", "raktha", "usiru", "naduga",
    ),
)

HOPELESSNESS = DistressCategory(
    id="HOPELESSNESS",
    label="Hopelessness / depression indicators",
    weight=0.55,
    floor=0.0,
    terms=(
        "hopeless", "no way out", "no one helps", "nobody helps",
        "no help", "give up", "depressed", "depression", "worthless",
        "can't sleep", "cannot sleep", "nightmare", "nightmares",
        "end it all", "tired of living", "broken",
        "कोई रास्ता नहीं", "कोई मदद नहीं", "कोई नहीं है", "हार गया",
        "हार गई", "थक गया", "थक गई", "उदास", "बर्बाद", "नींद नहीं",
        "koi rasta nahi", "koi madad nahi", "koi nahi hai", "haar gaya",
        "haar gayi", "thak gaya", "thak gayi", "udas", "barbad",
        "neend nahi", "jindagi barbad", "zindagi barbad",
        "কোনো সাহায্য নেই", "হারিয়ে", "ক্লান্ত", "kono shahajjo nei",
        "hariye", "klanto",
        "நம்பிக்கை இல்லை", "யாரும் இல்லை", "சோர்வ", "nambikkai illai",
        "yaarum illai", "sorva",
        "దారి లేదు", "సాయం లేదు", "ఎవరూ లేరు", "dari ledu", "saayam ledu",
        "evaru leru",
        "कोणी मदत नाही", "उदास", "थकलो", "konhi madat nahi", "udas", "thaklo",
        "ಯಾರೂ ಇಲ್ಲ", "ಸಹಾಯ ಇಲ್ಲ", "ದಾರಿ ಇಲ್ಲ", "yaaru illa", "sahaya illa",
        "dari illa",
    ),
)

FEAR_INTIMIDATION = DistressCategory(
    id="FEAR_INTIMIDATION",
    label="Fear / intimidation / threats",
    weight=0.50,
    floor=0.0,
    terms=(
        "afraid", "scared", "terrified", "fear", "threatening me",
        "threats", "they follow me", "following me", "watching me",
        "scared to go out", "hiding",
        "डर", "डर लग", "डरा", "धमकी", "पीछा", "घूर", "डरता", "डरती",
        "darr", "dar lag", "dara", "dhamki", "peecha", "darti", "darta",
        "ভয়", "ভয় করে", "হুমকি", "bhoy", "bhoy kore", "humki",
        "பயம்", "அச்சம்", "மிரட்டல்", "bayam", "acham", "mirattal",
        # Tamil inflected variants (virama drops before vowel suffixes)
        "பயமாக", "பயப்படு", "அச்சப்படு",
        "భయం", "భయపడ", "బెజ్జం", "bhayam", "bhayapadu", "bejjam",
        "भीती", "धमकी", "भय", "bhiti", "dhamki", "bhay",
        "ಭಯ", "ಬೆದರಿಕೆ", "ಅಂಜು", "bhaya", "bedarike", "anju",
    ),
)

SOCIAL_ISOLATION = DistressCategory(
    id="SOCIAL_ISOLATION",
    label="Social boycott / isolation / displacement",
    weight=0.45,
    floor=0.0,
    terms=(
        "alone", "boycott", "ostracized", "no one talks", "isolated",
        "thrown out", "kicked out", "displaced", "nobody talks to me",
        "social boycott", "untouchable",
        "अकेला", "अकेली", "बहिष्कार", "गांव से निकाल", "समाज से बाहर",
        "नहीं बोलते",
        "akela", "akeli", "bahishkar", "gaon se nikala", "nikal diye",
        "nahi bolte",
        "একা", "বয়কট", "তাড়িয়ে", "eka", "boykot", "tariye",
        "தனிமை", "ஒதுக்கி", "வெளியேற்ற", "thanimai", "othukki",
        "veliyettra",
        "ఒంటరి", "బహిష్కరణ", "వెళ్ళగొట్ట", "ontari", "bahishkarana",
        "vellagotta",
        "एकटा", "बहिष्कार", "हाकालित", "ekta", "bahishkar", "hakalit",
        "ಒಂಟಿ", "ಬಹಿಷ್ಕಾರ", "ಓಡಿಸಿದ", "onti", "bahishkara", "odisida",
    ),
)

GRIEF = DistressCategory(
    id="GRIEF",
    label="Grief / loss of family member",
    weight=0.40,
    floor=0.0,
    terms=(
        "died", "dead", "passed away", "lost my", "funeral",
        "last rites", "my son died", "my daughter died",
        "मर गया", "मर गई", "गुज़र गए", "अंतिम संस्कार", "शव",
        "mar gaya", "mar gayi", "guzar gaye", "antim sanskar", "shav",
        "মারা গেছে", "মৃত", "maré geche", "mrito",
        "இறந்து", "இறந்த", "சடலம்", "irandhu", "iranda", "sadalam",
        "చనిపోయారు", "మృత", "chanipoyaru", "mrita",
        "मरण पावल", "मेले", "maran paval", "mele",
        "ತೀರಿಕೊಂಡ", "ಮೃತ", "teerikonda", "mrita",
    ),
)

CASTE_ATROCITY = DistressCategory(
    id="CASTE_ATROCITY",
    label="Caste atrocity context (SC/ST)",
    weight=0.35,
    floor=0.0,
    terms=(
        "caste", "scheduled caste", "scheduled tribe", "dalit",
        "atrocity", "because of my caste", "upper caste",
        "जाति", "छुआछूत", "अछूत", "दलित", "अत्याचार", "जातीय",
        "jaati", "chhua", "chhua chhoot", "achhoot", "dalit", "atyachar",
        "জাতি", "ছুঁয়ে", "jati", "chhue",
        "ஜாதி", "தீண்டத்", "தாழ்த்தப்பட்ட", "jaathi", "theendath",
        "కులం", "దళిత", "అత్యాచార", "kulam", "dalita",
        "जाती", "अस्पृश्य", "दलित", "जात", "jaati", "dalit",
        "ಜಾತಿ", "ದಲಿತ", "jaathi", "dalita",
    ),
)

GENERAL_DISTRESS = DistressCategory(
    id="GENERAL_DISTRESS",
    label="General distress / help-seeking",
    weight=0.25,
    floor=0.0,
    terms=(
        "help me", "please help", "save me", "help", "crying",
        "anxiety", "panic", "stressed", "torture", "harassment",
        "harassing", "not listening", "no action",
        "मदद", "बचाओ", "रो रहा", "रो रही", "परेशान", "तंग",
        "पुलिस नहीं", "एफआईआर नहीं", "नहीं लिख",
        "madad", "bachao", "ro raha", "ro rahi", "pareshan", "tang",
        "police nahi", "fir nahi", "likhi nahi", "likha nahi",
        "সাহায্য", "বাঁচাও", "কাঁদছি", "shahajjo", "bachao", "kandchi",
        "உதவி", "காப்பாத்து", "அழுது", "udavi", "kaapathu", "azhuthu",
        "సహాయం", "కాపాడు", "ఏడుస్తూ", "sahayam", "kapadu", "edustu",
        "मदत", "वाचवा", "रडत", "madat", "vachva", "radat",
        "ಸಹಾಯ", "ರಕ್ಷಿಸಿ", "ಅಳುತ್ತಾ", "sahaya", "rakshisi", "alutta",
    ),
)

ALL_CATEGORIES: Tuple[DistressCategory, ...] = (
    SUICIDAL_IDEATION,
    SELF_HARM,
    SEXUAL_VIOLENCE,
    VIOLENCE_THREAT,
    TRAUMA_SHOCK,
    HOPELESSNESS,
    FEAR_INTIMIDATION,
    SOCIAL_ISOLATION,
    GRIEF,
    CASTE_ATROCITY,
    GENERAL_DISTRESS,
)

# ---------------------------------------------------------------------------
# Regex compilation
# ---------------------------------------------------------------------------
# NOTE: Python's re module treats Indic combining vowel signs (categories
# Mn/Mc, e.g. Devanagari '\u0940', Tamil virama '\u0BCD') as NON-word
# characters, so '\b' silently fails at word edges in Devanagari, Bengali,
# Tamil, Telugu and Kannada. We therefore use script-aware soft boundaries:
#   - non-Latin terms: must not be preceded by an Indic/Latin letter
#     (prevents mid-compound false hits), and must not be immediately
#     followed by Latin word chars. No trailing Indic restriction, so
#     agglutinative suffixes still match (e.g. Tamil '\u0BAA\u0BAF\u0BAE\u0BCD' inside
#     '\u0BAA\u0BAF\u0BAE\u0BBE\u0B95').
#   - Latin terms: standard word boundaries.

_INDIC_CLASS = (
    "A-Za-z0-9"
    "\u0900-\u097F"   # Devanagari (letters + signs)
    "\u0980-\u09FF"   # Bengali
    "\u0B80-\u0BFF"   # Tamil
    "\u0C00-\u0C7F"   # Telugu
    "\u0C80-\u0CFF"   # Kannada
)

_LEADING_BOUND = rf"(?<![{_INDIC_CLASS}])"
_TRAILING_BOUND = rf"(?![A-Za-z0-9'])"


def _is_latin(term: str) -> bool:
    return all(
        ch is ord(" ")
        or (unicodedata.category(ch)[0] not in "MN" and ord(ch) < 0x0900)
        for ch in term
    )


def _compile_term(term: str) -> Pattern:
    """Script-aware boundary-safe pattern for a single lexicon term."""
    escaped = re.escape(term)
    if _is_latin(term):
        return re.compile(rf"\b{escaped}\b", re.IGNORECASE)
    return re.compile(rf"{_LEADING_BOUND}{escaped}{_TRAILING_BOUND}")


_COMPILED: List[Tuple[DistressCategory, Pattern]] = [
    (category, _compile_term(term))
    for category in ALL_CATEGORIES
    for term in category.terms
]

CATEGORY_BY_ID: Dict[str, DistressCategory] = {
    category.id: category for category in ALL_CATEGORIES
}


def scan_text(text: str) -> List[dict]:
    """
    Scan free text against the multilingual lexicon.

    Returns a list of hits:
        { category_id, category, term, count, weight }

    Matching rules:
      - Longest span wins: a phrase hit suppresses shorter hits contained
        inside it (e.g. 'koi madad nahi' suppresses bare 'madad').
      - Repetition saturates: 1st occurrence full weight, +50% each
        further occurrence, capped at 2x.
    """
    candidates = []

    for category, pattern in _COMPILED:
        for match in pattern.finditer(text):
            candidates.append(
                {
                    "category_id": category.id,
                    "category": category.label,
                    "term": match.group(),
                    "weight": category.weight,
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    # Longest span first; ties broken by higher category weight.
    candidates.sort(
        key=lambda c: (-(c["end"] - c["start"]), -c["weight"])
    )

    kept = []
    for cand in candidates:
        overlaps_any = any(
            not (cand["end"] <= k["start"] or cand["start"] >= k["end"])
            for k in kept
        )
        if overlaps_any:
            continue
        kept.append(cand)

    # Merge repeats of the same term (repetition saturation)
    merged: Dict[Tuple[str, str], dict] = {}
    for hit in kept:
        key = (hit["category_id"], hit["term"])
        if key in merged:
            merged[key]["count"] += 1
        else:
            merged[key] = {**hit, "count": 1}

    hits = []
    for entry in merged.values():
        saturated = 1.0 + 0.5 * min(entry["count"] - 1, 2)
        hits.append(
            {
                "category_id": entry["category_id"],
                "category": entry["category"],
                "term": entry["term"],
                "count": entry["count"],
                "weight": round(entry["weight"] * saturated, 3),
            }
        )

    return hits


def max_category_floor(hits: List[dict]) -> Tuple[float, List[str]]:
    """
    Highest severity floor triggered by matched categories, plus labels.
    Floors let genuine high-severity disclosures keep the SVI grounded
    even when the caller speaks softly or briefly.
    """
    floors = []
    labels = []

    for hit in hits:
        category = CATEGORY_BY_ID.get(hit["category_id"])
        if category and category.floor > 0:
            floors.append(category.floor)
            labels.append(hit["category"])

    return (max(floors) if floors else 0.0), labels


# ---------------------------------------------------------------------------
# PII redaction (privacy & ethical-AI requirement)
# ---------------------------------------------------------------------------

_PII_PATTERNS = [
    # Indian mobile numbers, optionally +91 or 0 prefixed, with separators
    (re.compile(r"(?:\+91[\s-]?)?0?[6-9]\d{4}[\s-]?\d{5}"), "[PHONE_REDACTED]"),
    # Aadhaar style 4-4-4
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[ID_REDACTED]"),
    # Email
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL_REDACTED]"),
    # Self-disclosed names in common phrasings
    (
        re.compile(
            r"(?:my name is|mera naam|mera nam|मेरा नाम|என் பெயர்|"
            r"నా పేరు|माझं नाव|ನನ್ನ ಹೆಸರು|আমার নাম)\s+([\u0900-\u097F"
            r"\u0B80-\u0BFF\u0C00-\u0C7F\u0980-\u09FFA-Za-z]+(?:\s+[\u0900-\u097F"
            r"\u0B80-\u0BFF\u0C00-\u0C7F\u0980-\u09FFA-Za-z]+)?)",
            re.IGNORECASE,
        ),
        "[NAME_REDACTED]",
    ),
]


def redact_pii(text: str) -> Tuple[str, int]:
    """Redact obvious PII. Returns (redacted_text, number_of_redactions)."""
    redactions = 0
    result = text

    for pattern, replacement in _PII_PATTERNS:
        result, n = pattern.subn(replacement, result)
        redactions += n

    return result, redactions
