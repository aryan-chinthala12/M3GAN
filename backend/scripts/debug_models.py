import sys
import traceback

sys.path.insert(0, ".")

with open("debug_models.log", "w") as f:
    f.write("Starting debug...\n")
    try:
        f.write("Importing SVIEngine...\n")
        f.flush()
        from backend.core.svi_engine import SVIEngine

        f.write("Creating SVIEngine instance...\n")
        f.flush()
        e = SVIEngine(use_vad_fallback=True)

        f.write("Calling _ensure_models()...\n")
        f.flush()
        e._ensure_models()

        f.write("Models successfully loaded!\n")
        f.flush()
    except Exception as ex:
        f.write("EXCEPTION OCCURRED:\n")
        f.write(traceback.format_exc() + "\n")
        f.flush()
