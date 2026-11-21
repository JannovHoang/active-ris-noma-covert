"""Read-only-source diagnostic extension for the frozen Windows 7g/32 study."""
from pathlib import Path
import os
import sys

RUNTIME = Path(__file__).resolve().parent / "frozen_runtime"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
if "ris_noma" in sys.modules:
    if Path(sys.modules["ris_noma"].__file__).resolve().parent != RUNTIME / "ris_noma":
        raise RuntimeError("Run the 7g supplement in a separate Python process.")
sys.path.insert(0, str(RUNTIME))
