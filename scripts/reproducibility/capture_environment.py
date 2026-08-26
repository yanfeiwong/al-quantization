"""Capture a concise, path-sanitized environment report for the paper.

This intentionally does not reproduce ``pip freeze``.  It records the source
revision, accelerator/toolchain, PyTorch runtime, and a curated set of packages
that can affect the reported experiments.  Local wheel paths are reduced to the
wheel basename plus any recorded SHA256 digest.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import locale
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit


REPORT_SCHEMA_VERSION = 1
CORE_DISTRIBUTIONS = (
    "adafactor8bit",
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "came-pytorch",
    "apollo-torch",
    "flash-attn",
    "numpy",
    "tensorboard",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
    "ninja",
    "triton",
)


def decode_command_output(raw: bytes) -> str:
    """Decode command output without corrupting localized Windows toolchains."""
    encodings = ["utf-8", locale.getpreferredencoding(False)]
    if os.name == "nt":
        encodings.extend(["mbcs", "gb18030", "cp1252"])

    attempted: set[str] = set()
    for encoding in encodings:
        normalized = encoding.lower()
        if normalized in attempted:
            continue
        attempted.add(normalized)
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def run_command(
    args: list[str], cwd: Path | None = None, *, allow_nonzero: bool = False
) -> str | None:
    """Return stripped combined output, or None when a command is unavailable."""
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 and not allow_nonzero:
        return None
    output = decode_command_output(completed.stdout).strip()
    return output if output else None


def first_line(value: str | None) -> str | None:
    return value.splitlines()[0].strip() if value else None


def find_repo_root(start: Path) -> Path | None:
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def sanitize_url(url: str) -> str:
    """Remove credentials and local directory components from a provenance URL."""
    parts = urlsplit(url)
    if parts.scheme == "file":
        name = Path(unquote(parts.path)).name
        return f"local wheel: {name}" if name else "local file artifact"

    hostname = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    safe_netloc = f"{hostname}{port}"
    return urlunsplit((parts.scheme, safe_netloc, parts.path, parts.query, ""))


def direct_url_provenance(dist: metadata.Distribution) -> dict[str, Any] | None:
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"kind": "unparsed direct_url metadata"}

    url = str(payload.get("url", ""))
    archive_info = payload.get("archive_info") or {}
    vcs_info = payload.get("vcs_info") or {}
    dir_info = payload.get("dir_info") or {}
    result: dict[str, Any] = {}

    if url:
        result["source"] = sanitize_url(url)
        result["kind"] = "local artifact" if urlsplit(url).scheme == "file" else "direct URL"
    if archive_info.get("hash"):
        result["archive_hash"] = archive_info["hash"]
    elif isinstance(archive_info.get("hashes"), dict):
        result["archive_hashes"] = archive_info["hashes"]
    if vcs_info:
        result["kind"] = vcs_info.get("vcs", "VCS")
        result["commit_id"] = vcs_info.get("commit_id")
        result["requested_revision"] = vcs_info.get("requested_revision")
    if dir_info.get("editable"):
        result["editable"] = True
    return {key: value for key, value in result.items() if value not in (None, "", {})}


def collect_packages(extra: list[str]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    packages: list[dict[str, Any]] = []
    for requested_name in (*CORE_DISTRIBUTIONS, *extra):
        normalized = re.sub(r"[-_.]+", "-", requested_name).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            dist = metadata.distribution(requested_name)
        except metadata.PackageNotFoundError:
            packages.append({"name": requested_name, "installed": False})
            continue
        packages.append(
            {
                "name": dist.metadata.get("Name", requested_name),
                "version": dist.version,
                "installed": True,
                "provenance": direct_url_provenance(dist),
            }
        )
    return packages


def collect_git(repo_root: Path | None) -> dict[str, Any]:
    if repo_root is None:
        return {"available": False}
    head = run_command(["git", "rev-parse", "HEAD"], repo_root)
    if not head:
        return {"available": False}
    status = run_command(["git", "status", "--porcelain"], repo_root)
    remote = run_command(["git", "remote", "get-url", "origin"], repo_root)
    describe = run_command(["git", "describe", "--tags", "--always", "--dirty"], repo_root)
    return {
        "available": True,
        "commit": head,
        "describe": describe,
        "dirty": bool(status),
        "origin": sanitize_url(remote) if remote else None,
    }


def collect_torch_runtime() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on capture host
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    runtime: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "built_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            runtime["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_bytes": props.total_memory,
                }
            )
    return runtime


def collect_nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False}
    query = run_command(
        [
            executable,
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not query:
        return {"available": False}
    rows = []
    for line in query.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3:
            rows.append(
                {
                    "name": fields[0],
                    "driver_version": fields[1],
                    "memory_total_mib": fields[2],
                }
            )
    return {"available": True, "gpus": rows}


def collect_toolchain() -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    nvcc_output = run_command([nvcc, "--version"]) if nvcc else None
    compiler_candidates = (
        ("cl", ["cl"]),
        ("gcc", ["gcc", "--version"]),
        ("clang", ["clang", "--version"]),
    )
    compilers = []
    for name, command in compiler_candidates:
        if shutil.which(command[0]):
            compilers.append(
                {
                    "name": name,
                    "version": first_line(
                        run_command(command, allow_nonzero=(name == "cl"))
                    ),
                }
            )
    return {
        "nvcc": first_line(nvcc_output),
        "nvcc_full": nvcc_output,
        "python_compiler": platform.python_compiler(),
        "compilers": compilers,
    }


def collect_report(repo_root: Path | None, extra_packages: list[str]) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_revision": collect_git(repo_root),
        "host": {
            "os": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "torch_runtime": collect_torch_runtime(),
        "nvidia_smi": collect_nvidia_smi(),
        "toolchain": collect_toolchain(),
        "packages": collect_packages(extra_packages),
        "selected_environment": {
            key: os.environ[key]
            for key in (
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTORCH_CUDA_ALLOC_CONF",
                "CUDA_MODULE_LOADING",
            )
            if key in os.environ
        },
    }


def markdown_value(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    return str(value).replace("|", r"\|").replace("\n", " ")


def provenance_text(provenance: dict[str, Any] | None) -> str:
    if not provenance:
        return "package index / unspecified"
    parts = []
    if provenance.get("source"):
        parts.append(str(provenance["source"]))
    if provenance.get("commit_id"):
        parts.append(f"commit {provenance['commit_id']}")
    if provenance.get("archive_hash"):
        parts.append(str(provenance["archive_hash"]))
    for name, digest in (provenance.get("archive_hashes") or {}).items():
        parts.append(f"{name}={digest}")
    if provenance.get("editable"):
        parts.append("editable install")
    return "; ".join(parts) or markdown_value(provenance.get("kind"))


def render_markdown(report: dict[str, Any]) -> str:
    git = report["source_revision"]
    host = report["host"]
    torch_runtime = report["torch_runtime"]
    toolchain = report["toolchain"]
    lines = [
        "# Reproducibility Environment Report",
        "",
        f"Generated (UTC): `{report['generated_at_utc']}`  ",
        "",
        "## Source Revision",
        "",
    ]
    if git.get("available"):
        lines.extend(
            [
                f"- Commit: `{git['commit']}`",
                f"- Describe: `{markdown_value(git.get('describe'))}`",
                f"- Working tree: `{'dirty' if git.get('dirty') else 'clean'}`",
                f"- Origin: `{markdown_value(git.get('origin'))}`",
            ]
        )
    else:
        lines.append("- Git metadata: unavailable (run inside the repository checkout).")

    lines.extend(
        [
            "",
            "## Host",
            "",
            f"- OS: `{host['os']}`",
            f"- Architecture: `{host['machine']}`",
            f"- Processor: `{markdown_value(host.get('processor'))}`",
            f"- Python: `{host['python']}` ({host['python_implementation']})",
            "",
            "## PyTorch and Accelerator Runtime",
            "",
        ]
    )
    if torch_runtime.get("available"):
        lines.extend(
            [
                f"- PyTorch: `{torch_runtime['version']}`",
                f"- CUDA used to build PyTorch: `{markdown_value(torch_runtime.get('built_cuda'))}`",
                f"- CUDA available: `{torch_runtime['cuda_available']}`",
                f"- cuDNN: `{markdown_value(torch_runtime.get('cudnn_version'))}`",
                f"- Deterministic algorithms at capture: `{torch_runtime['deterministic_algorithms']}`",
                f"- cuDNN deterministic / benchmark: `{torch_runtime['cudnn_deterministic']}` / `{torch_runtime['cudnn_benchmark']}`",
                f"- TF32 matmul / cuDNN: `{torch_runtime['cuda_matmul_allow_tf32']}` / `{torch_runtime['cudnn_allow_tf32']}`",
            ]
        )
        for device in torch_runtime.get("devices", []):
            gib = device["total_memory_bytes"] / (1024**3)
            lines.append(
                f"- GPU {device['index']}: `{device['name']}`; compute capability "
                f"`{device['compute_capability']}`; `{gib:.2f} GiB`"
            )
    else:
        lines.append(f"- PyTorch runtime unavailable: `{markdown_value(torch_runtime.get('error'))}`")

    smi = report["nvidia_smi"]
    if smi.get("available"):
        for index, gpu in enumerate(smi.get("gpus", [])):
            lines.append(
                f"- NVIDIA-SMI GPU {index}: `{gpu['name']}`; driver "
                f"`{gpu['driver_version']}`; `{gpu['memory_total_mib']} MiB`"
            )

    lines.extend(
        [
            "",
            "## CUDA and Compiler Toolchain",
            "",
            f"- nvcc: `{markdown_value(toolchain.get('nvcc'))}`",
            f"- Python compiler: `{markdown_value(toolchain.get('python_compiler'))}`",
        ]
    )
    for compiler in toolchain.get("compilers", []):
        lines.append(f"- {compiler['name']}: `{markdown_value(compiler.get('version'))}`")

    lines.extend(
        [
            "",
            "## Core Python Packages",
            "",
            "| Distribution | Version | Provenance |",
            "|---|---:|---|",
        ]
    )
    for package in report["packages"]:
        if package.get("installed"):
            lines.append(
                f"| {markdown_value(package['name'])} | {markdown_value(package['version'])} | "
                f"{markdown_value(provenance_text(package.get('provenance')))} |"
            )
        else:
            lines.append(f"| {markdown_value(package['name'])} | not installed | -- |")

    lines.extend(["", "## Selected Environment Variables", ""])
    selected_environment = report["selected_environment"]
    if selected_environment:
        for name, value in selected_environment.items():
            lines.append(f"- `{name}={markdown_value(value)}`")
    else:
        lines.append("- None of the selected CUDA/PyTorch variables were set in the capture process.")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Local installation paths are intentionally omitted. Local wheels are identified "
            "by artifact basename and recorded digest when available.",
            "- Runtime flags above describe the capture process. Experiment scripts remain the "
            "source of truth for flags explicitly set during training.",
            "- This curated report records result-relevant dependencies; it is not a complete "
            "dump of every transitive package in the environment.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def default_output(script_path: Path, repo_root: Path | None) -> Path:
    if repo_root is not None and (repo_root / "paper").is_dir():
        return repo_root / "paper" / "reports_md" / "environment.md"
    paper_root = script_path.resolve().parents[2]
    return paper_root / "reports_md" / "environment.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a concise Markdown/JSON report for paper reproducibility."
    )
    parser.add_argument("--repo-root", type=Path, help="Git repository root (auto-detected by default).")
    parser.add_argument("--output", type=Path, help="Markdown output path.")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="JSON output path (defaults to the Markdown path with .json suffix).",
    )
    parser.add_argument(
        "--extra-package",
        action="append",
        default=[],
        help="Additional distribution name to record; may be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_path = Path(__file__)
    repo_root = args.repo_root.resolve() if args.repo_root else find_repo_root(script_path)
    output_path = (args.output or default_output(script_path, repo_root)).resolve()
    json_path = (args.json_output or output_path.with_suffix(".json")).resolve()

    report = collect_report(repo_root, args.extra_package)
    atomic_write(output_path, render_markdown(report))
    atomic_write(json_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote Markdown report: {output_path}")
    print(f"Wrote JSON report: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
