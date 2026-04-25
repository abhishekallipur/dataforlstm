from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "models" / "residual_hybrid" / "model.py"

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(SCRIPT), run_name="__main__")
