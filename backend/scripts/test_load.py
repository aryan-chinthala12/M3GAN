import sys
import traceback

sys.path.insert(0, ".")

with open("load_err.txt", "w") as f:
    f.write("Testing emotion model loading...\n")
    try:
        from backend.core.svi_engine import SVIEngine
        engine = SVIEngine(use_vad_fallback=True)
        m = engine._load_emotion_model()
        f.write(f"Emotion model loaded successfully: {m}\n")
    except Exception as ex:
        f.write("ERROR DETECTED:\n")
        f.write(traceback.format_exc() + "\n")
