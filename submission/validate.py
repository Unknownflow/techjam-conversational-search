"""Validate and optionally archive the allowlisted submission bundle."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "MANIFEST.json"
NETWORK_MODULES = {"boto3", "http", "httpx", "openai", "requests", "socket", "urllib"}
SECRET_VALUE_RE = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*['\"][^'\"]{8,}['\"]"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def validate() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = set(manifest["files"])
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and "dist" not in path.relative_to(ROOT).parts
        and path.suffix != ".pyc"
    }
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(f"Manifest mismatch; missing={missing}, unexpected={unexpected}")

    for relative in sorted(expected):
        path = ROOT / relative
        if path.is_symlink():
            raise RuntimeError(f"Symlinks are not permitted: {relative}")
        lowered = relative.lower()
        if lowered.endswith((".env", ".jsonl")) or "result" in Path(lowered).name:
            raise RuntimeError(f"Disallowed submission content: {relative}")
        if path.suffix in {".py", ".md", ".txt", ".json"}:
            content = path.read_text(encoding="utf-8")
            if SECRET_VALUE_RE.search(content):
                raise RuntimeError(f"Possible embedded secret in {relative}")

    source = ROOT / "src" / "agent.py"
    unexpected_network = import_roots(source) & NETWORK_MODULES
    if unexpected_network:
        raise RuntimeError(
            f"Network-capable imports are not permitted: {sorted(unexpected_network)}"
        )
    expected_source_hash = manifest["sha256"]["src/agent.py"]
    if sha256(source) != expected_source_hash:
        raise RuntimeError("src/agent.py differs from the frozen manifest hash")

    try:
        from .agent import Agent
    except ImportError:
        from agent import Agent
    reset_parameters = list(inspect.signature(Agent.reset).parameters)
    respond_parameters = list(inspect.signature(Agent.respond).parameters)
    if reset_parameters != ["self", "session_id", "user_profile"]:
        raise RuntimeError(f"Unexpected reset signature: {reset_parameters}")
    if respond_parameters != ["self", "session_id", "user_message", "turn", "top_k"]:
        raise RuntimeError(f"Unexpected respond signature: {respond_parameters}")
    return manifest


def build_zip(destination: Path, manifest: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(manifest["files"]):
            archive.write(ROOT / relative, arcname=relative)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, help="Optional destination ZIP path")
    args = parser.parse_args()
    manifest = validate()
    if args.build:
        build_zip(args.build, manifest)
        print(f"Validated and built {args.build}")
    else:
        print("Submission bundle validated")


if __name__ == "__main__":
    main()
