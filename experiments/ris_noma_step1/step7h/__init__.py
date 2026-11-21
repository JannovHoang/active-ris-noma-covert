"""Self-contained 7h benchmark; imports the unmodified, bundled physics runtime."""
from pathlib import Path
import sys

_runtime = str(Path(__file__).resolve().parent / "frozen_runtime")
if "ris_noma" in sys.modules:
    location = Path(sys.modules["ris_noma"].__file__).resolve()
    if not location.is_relative_to(Path(_runtime)):
        raise RuntimeError("Start run_step7h.py in a fresh Python process; another ris_noma runtime is already imported")
if _runtime not in sys.path:
    sys.path.insert(0, _runtime)

FORMAT = "six_method_benchmark_7h_v1"
