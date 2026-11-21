"""Check exported common channels and fixed-candidate outputs against Python."""
import argparse
import json
from pathlib import Path
from scipy.io import loadmat
from ris_noma.config import load_config
from ris_noma.legacy_comparison import compare_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="MAT file produced by legacy_octave/export_legacy_reference.m")
    parser.add_argument("--config", default="configs/legacy_physics_k1_v4.json")
    parser.add_argument("--output", default="outputs/octave_comparison.json")
    args = parser.parse_args()
    report = compare_reference(load_config(args.config), loadmat(args.reference))
    report["reference_file"] = str(Path(args.reference).resolve())
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("Common-channel original-convention checks:", "PASS" if report["all_checks_pass"] else "FAIL")
    for row in report["candidates"]:
        changes = row["v4_method_differences"]
        print(f"mb={row['mb']:.2f}, s={row['s']:.2f}: pass={row['all_original_conventions_pass']} | "
              f"public rate change={changes['public_rate_reduction_from_carol_only']:.3e} | "
              f"grid-stationary DEP gap={changes['grid_minus_stationary_dep']:.3e}")
    print("Report:", path.resolve())
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
