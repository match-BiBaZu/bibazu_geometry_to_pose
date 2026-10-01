"""Double-click launcher target; works without relying on the working directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
try:
    from chute_pose.gui import main
    raise SystemExit(main())
except Exception:
    import traceback
    import ctypes
    error = traceback.format_exc()
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(None, error, "Pose Roadmap Generator could not start", 0x10)
    else:
        raise
