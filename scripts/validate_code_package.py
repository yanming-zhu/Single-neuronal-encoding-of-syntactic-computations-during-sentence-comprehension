#!/usr/bin/env python3
"""Validate this archive statically without importing or running analysis code.

Run from any directory with Python 3.10+: python /path/to/scripts/validate_code_package.py
External dependency availability is reported separately from missing local code.
This checks packaging and syntax, not numerical results or runtime data access.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.machinery
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tokenize


CODE_SUFFIXES = {".py", ".js", ".mjs", ".cjs", ".sh", ".r", ".m"}
IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules"}
LOCAL_PREFIXES = (
    "analyze_", "plot_", "make_", "extract_", "estimate_", "detect_",
    "run_", "summarize_", "update_", "combine_", "decode_", "map_",
    "validate_", "build_", "compute_", "render_", "prepare_", "export_",
)


def validate_package(package_root: Path) -> dict:
    """Return a JSON-serializable validation report; never modify the package."""
    root = Path(package_root).resolve()
    report = {"package_root": str(root), "passed": [], "errors": [], "skips": []}

    def record(category: str, check: str, **details) -> None:
        report[category].append({"check": check, **details})

    def relative(path: Path) -> str:
        return str(path.relative_to(root))

    def included(path: Path) -> bool:
        return not any(part in IGNORED_DIRS for part in path.relative_to(root).parts)

    if not root.is_dir():
        record("errors", "package_root", message="Package directory does not exist")
        report["ok"] = False
        return report
    paths = sorted(path for path in root.rglob("*") if included(path))
    symlinks = [path for path in paths if path.is_symlink()]
    for path in symlinks:
        try:
            target = path.resolve(strict=True)
            if not target.is_relative_to(root):
                record("errors", "symlink", file=relative(path), message="Target is outside package")
        except (OSError, RuntimeError) as exc:
            record("errors", "symlink", file=relative(path), message=str(exc))
    if not any(item["check"] == "symlink" for item in report["errors"]):
        record("passed", "symlinks", count=len(symlinks))
    files = [path for path in paths if path.is_file() and not path.is_symlink()]
    code = [path for path in files if path.suffix.lower() in CODE_SUFFIXES]

    manifest = root / "SCRIPT_MANIFEST.tsv"
    listed = set()
    rows = []
    try:
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or "archive_file" not in reader.fieldnames:
                record("errors", "manifest", message="Missing archive_file column")
            else:
                rows = list(reader)
        seen = set()
        for number, row in enumerate(rows, start=2):
            name = (row.get("archive_file") or "").strip()
            path = root / name
            if not name or Path(name).is_absolute() or not path.resolve().is_relative_to(root):
                record("errors", "manifest", line=number, file=name, message="Invalid archive path")
                continue
            if name in seen:
                record("errors", "manifest", line=number, file=name, message="Duplicate archive path")
            seen.add(name)
            if not path.is_file():
                record("errors", "manifest", line=number, file=name, message="Listed file is missing")
                continue
            listed.add(path.resolve())
            digest = next((value.strip() for key, value in row.items()
                           if key and key.lower() == "sha256" and value), "")
            if digest and hashlib.sha256(path.read_bytes()).hexdigest() != digest.lower():
                record("errors", "manifest", file=name, message="SHA256 mismatch")
        for path in code:
            if path.resolve() not in listed:
                record("errors", "manifest", file=relative(path), message="Code file absent from manifest")
    except (OSError, csv.Error, RuntimeError) as exc:
        record("errors", "manifest", message=str(exc))
    if not any(item["check"] == "manifest" for item in report["errors"]):
        record("passed", "manifest", entries=len(rows), canonical_code_files=len(code))

    trees = {}
    for path in code:
        if path.suffix == ".py":
            try:
                with tokenize.open(path) as handle:
                    tree = ast.parse(handle.read(), filename=relative(path))
                # Compilation catches invalid contexts such as a top-level return;
                # compiling an AST does not execute code or write bytecode files.
                compile(tree, relative(path), "exec")
                trees[path] = tree
            except (OSError, SyntaxError, UnicodeError) as exc:
                record("errors", "python_syntax", file=relative(path), message=str(exc))
    if not any(item["check"] == "python_syntax" for item in report["errors"]):
        record("passed", "python_syntax", canonical_files=len(trees))

    javascript = [path for path in code if path.suffix in {".js", ".mjs", ".cjs"}]
    node = shutil.which("node")
    if javascript and not node:
        record("skips", "javascript_syntax", message="Node.js is unavailable", files=len(javascript))
    else:
        for path in javascript:
            try:
                result = subprocess.run([node, "--check", str(path)], capture_output=True,
                                        text=True, timeout=30, check=False)
                if result.returncode:
                    record("errors", "javascript_syntax", file=relative(path), message=result.stderr.strip())
            except (OSError, subprocess.TimeoutExpired) as exc:
                record("errors", "javascript_syntax", file=relative(path), message=str(exc))
        if not any(item["check"] == "javascript_syntax" for item in report["errors"]):
            record("passed", "javascript_syntax", canonical_files=len(javascript))

    known_modules = {"scripts", "condition_contrast_figures"}
    known_modules.update(path.stem for path in trees)
    known_modules.update(Path(row.get("archive_file") or "").stem for row in rows)
    known_modules.update(path.name for path in root.iterdir() if path.is_dir())
    known_modules.update(path.name for path in (root / "scripts").glob("*") if path.is_dir())
    external = {}
    checked = 0

    def resolve_module(module: str, bases: list[Path]) -> Path | None:
        parts = module.split(".") if module else []
        for base in bases:
            candidate = base.joinpath(*parts)
            if candidate.with_suffix(".py").is_file():
                return candidate.with_suffix(".py")
            if candidate.is_dir():
                return candidate
        return None

    def exported_names(package: Path) -> set[str]:
        tree = trees.get(package / "__init__.py")
        if tree is None:
            return set()
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        names.update(node.name for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        return names

    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for module in modules:
                checked += 1
                bases = [path.parent, root, root / "scripts"]
                is_relative = isinstance(node, ast.ImportFrom) and node.level > 0
                if is_relative:
                    base = path.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    bases = [base] if base.is_relative_to(root) else []
                top = module.split(".")[0]
                target = resolve_module(module, bases)
                local = is_relative or target is not None or top in known_modules or top.startswith(LOCAL_PREFIXES)
                if local:
                    if target is None:
                        record("errors", "local_import", file=relative(path), line=node.lineno,
                               module="." * getattr(node, "level", 0) + module, message="Package-local module is missing")
                    elif isinstance(node, ast.ImportFrom) and target.is_dir():
                        exports = exported_names(target)
                        for alias in node.names:
                            if alias.name != "*" and alias.name not in exports and resolve_module(alias.name, [target]) is None:
                                record("errors", "local_import", file=relative(path), line=node.lineno,
                                       module=f"{module}.{alias.name}".lstrip("."), message="Package-local submodule/export is missing")
                elif top not in sys.stdlib_module_names and top not in sys.builtin_module_names:
                    external.setdefault(top, set()).add(relative(path))
    if not any(item["check"] == "local_import" for item in report["errors"]):
        record("passed", "local_import_closure", static_imports=checked)
    available = []
    for name, users in sorted(external.items()):
        # PathFinder inspects top-level module locations without executing package code.
        if importlib.machinery.PathFinder.find_spec(name) is None:
            record("skips", "external_dependency", module=name, files=sorted(users),
                   message="Not installed in this interpreter; scientific execution was not tested")
        else:
            available.append(name)
    record("passed", "external_dependency_discovery", available=available)
    report["scope"] = "Static syntax, manifest, symlinks, and explicit local imports only; dynamic imports, numerical reproduction, data paths, and dependency versions are not validated."
    report["ok"] = not report["errors"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Optional JSON report path; stdout is always JSON")
    args = parser.parse_args()
    report = validate_package(args.root)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
