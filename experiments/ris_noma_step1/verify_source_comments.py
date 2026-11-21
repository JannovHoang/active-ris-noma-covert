"""Compare readable source with preserved 7h source, then optionally check records.

This tool does not modify result plans, checksums, checkpoints or saved source.
Only Python comments and non-semantic whitespace are allowed in compared files.
The original strict checker runs in a fresh process from the preserved source.
"""
from pathlib import Path, PurePosixPath
import argparse
import ast
import hashlib
import io
import json
import subprocess
import sys
import tokenize


ROOT = Path(__file__).resolve().parent
READABLE_MODULES = (
    "pso.py", "multik_core.py", "metrics.py", "stable_assessment.py",
    "channels.py", "geometry_los.py", "dep.py", "validation.py",
    "continuous_workers.py", "curve_data.py", "multik_graph.py",
    "multik_models.py", "multik_learning_models.py", "curve_fit.py", "curve_models.py",
)


def digest(path):
    """Hash the actual bytes; never substitute a reference hash for changed code."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_path(root, name):
    """Accept relative manifest paths, including the original Windows separators."""
    name = str(name).replace("\\", "/")
    pure = PurePosixPath(name)
    if (not pure.parts or pure.is_absolute() or ".." in pure.parts
            or ":" in pure.parts[0]):
        raise ValueError(f"Unsafe source path: {name}")
    root = Path(root).resolve()
    result = (root / Path(*pure.parts)).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Source path escapes its root: {name}")
    return result


def source_signature(data):
    """Retain code tokens and AST, including literal strings and docstrings.

    COMMENT and blank-line NL tokens can be removed safely for this comparison.
    Logical NEWLINE, indentation, numbers, identifiers and operators are kept.
    AST is a second check; it also preserves type comments and type-ignore tags.
    Source line positions are excluded, so adding explanatory lines is allowed.
    """
    encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    text = data.decode(encoding)
    tree = ast.parse(text, type_comments=True)
    tokens = tuple((token.type, token.string.replace("\r\n", "\n"))
                   for token in tokenize.generate_tokens(io.StringIO(text).readline)
                   if token.type not in (tokenize.COMMENT, tokenize.NL))
    return ast.dump(tree, include_attributes=False), tokens


def compare_source(current, original):
    """Refuse any change to program tokens, constants, indentation or docstrings."""
    try:
        equal = source_signature(current) == source_signature(original)
    except (SyntaxError, UnicodeError, tokenize.TokenError, IndentationError) as error:
        raise ValueError(f"Invalid Python source: {error}") from error
    if not equal:
        raise ValueError("Executable source differs; this is not a comment-only update")


def verify_sources(reference, root=ROOT):
    """Authenticate saved source with the original plan, then compare readable code."""
    reference, root = Path(reference).resolve(), Path(root).resolve()
    plan_path = reference / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    completion = json.loads((reference / "completion.json").read_text(encoding="utf-8-sig"))
    if (plan.get("format") != "six_method_benchmark_7h_v1"
            or completion.get("plan_sha256") != digest(plan_path)):
        raise ValueError("The completed reference does not match its original 7h plan")
    expected = plan.get("code_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("The reference plan has no source manifest")
    source = reference / "source_snapshot"
    # The source used as the reference must still match its original SHA exactly.
    # Comment tolerance applies only to the readable working copy, never evidence.
    for name, value in expected.items():
        original = safe_path(source, name)
        if not original.is_file() or digest(original) != value:
            raise ValueError(f"Original source snapshot changed or missing: {name}")
    current_files = {"run_step7h.py"}
    current_files.update(path.relative_to(root).as_posix()
                         for path in (root / "step7h").rglob("*.py")
                         if "__pycache__" not in path.parts)
    if current_files != set(expected):
        raise ValueError("The 7h source-file set differs from the preserved reference")
    changed, identical = [], 0
    for name, value in expected.items():
        current = safe_path(root, name)
        if not current.is_file():
            raise ValueError(f"Readable source is missing: {name}")
        if digest(current) == value:
            identical += 1
            continue
        try:
            compare_source(current.read_bytes(), safe_path(source, name).read_bytes())
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
        changed.append(name)
    # These top-level modules are the readable counterparts of the unchanged
    # bundled runtime. Verify them too, so comments cannot hide an algorithm change.
    for name in READABLE_MODULES:
        current_name = "ris_noma/" + name
        original_name = "step7h/frozen_runtime/" + current_name
        if original_name not in expected:
            raise ValueError(f"Reference has no original module: {original_name}")
        current = safe_path(root, current_name)
        original = safe_path(source, original_name)
        if not current.is_file():
            raise ValueError(f"Readable module is missing: {current_name}")
        try:
            compare_source(current.read_bytes(), original.read_bytes())
        except ValueError as error:
            raise ValueError(f"{current_name}: {error}") from error
        if current.read_bytes() != original.read_bytes():
            changed.append(current_name)
    return {"source_comparison_passed": True,
            "reference": str(reference), "reference_code_files": len(expected),
            "byte_identical_7h_files": identical,
            "readable_module_copies_checked": len(READABLE_MODULES),
            "comment_only_files": sorted(changed),
            "comparison": "Original snapshot SHA-256; current Python AST and significant tokens",
            "saved_results_modified": False}


def check_saved_results(reference):
    """Run the unchanged checker; it retains every original data/model/hash check."""
    reference = Path(reference).resolve()
    entry = reference / "source_snapshot/run_step7h.py"
    process = subprocess.run(
        [sys.executable, "-X", "utf8", str(entry), "check", "--output", str(reference)],
        capture_output=True, text=True, encoding="utf-8", errors="strict", check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip())
    return json.loads(process.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Complete original 7h result directory")
    parser.add_argument("--check", action="store_true",
                        help="Also run the preserved strict result checker")
    args = parser.parse_args()
    result = {"source_verification": verify_sources(args.reference)}
    if args.check:
        result["saved_experiment"] = check_saved_results(args.reference)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
