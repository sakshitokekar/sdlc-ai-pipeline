# Docker tools — checking Docker availability, ensuring build prerequisites
# exist, and building tagged images. Dockerfile generation is deterministic
# (no LLM): infrastructure files should be reproducible. requirements.txt
# generation IS dynamic — it scans actual imports so it stays correct as
# Agent 2/Agent 5 add new dependencies over time.
import ast
import subprocess
import sys
from pathlib import Path


#     check_docker_available() -> dict:
#         - verifies the Docker daemon is installed and running
#         - returns {"available": bool, "reason_code": str, "error": str (if failed)}
#         - reason codes: DOCKER_AVAILABLE, DOCKER_NOT_INSTALLED, DOCKER_NOT_RUNNING, DOCKER_TIMEOUT
def check_docker_available() -> dict:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return {"available": True, "reason_code": "DOCKER_AVAILABLE"}
        return {"available": False, "error": result.stderr, "reason_code": "DOCKER_NOT_RUNNING"}
    except FileNotFoundError:
        return {"available": False, "error": "Docker CLI not found on PATH", "reason_code": "DOCKER_NOT_INSTALLED"}
    except subprocess.TimeoutExpired:
        return {"available": False, "error": "docker info timed out", "reason_code": "DOCKER_TIMEOUT"}


#     get_current_commit_hash(repo_path, short=True) -> str:
#         - returns the current HEAD commit hash for the repo
#         - used to tag Docker images deterministically, tying every image
#           back to the exact commit it was built from
#         - returns "unknown" if git command fails (fault-tolerant fallback)
def get_current_commit_hash(repo_path: str, short: bool = True) -> str:
    args = ["git", "rev-parse"] + (["--short"] if short else []) + ["HEAD"]
    result = subprocess.run(args, cwd=repo_path, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _get_stdlib_module_names() -> set:
    """Returns the set of standard library module names for the currently
    running Python. Uses sys.stdlib_module_names (Python 3.10+) — the
    correct, version-accurate way to get this list, rather than a
    hand-maintained list that would drift out of date."""
    if hasattr(sys, "stdlib_module_names"):
        return set(sys.stdlib_module_names)
    return {
        "os", "sys", "json", "re", "math", "datetime", "hashlib", "uuid",
        "pathlib", "subprocess", "typing", "collections", "itertools",
        "functools", "logging", "unittest", "time", "random", "io",
        "abc", "enum", "dataclasses", "contextlib"
    }


# Maps common import names to their actual PyPI package name, for the
# (fairly common) cases where they differ — e.g. `import jwt` installs
# from the PyPI package `pyjwt`, not a package literally called `jwt`.
_IMPORT_TO_PACKAGE_NAME = {
    "flask": "flask",
    "jwt": "pyjwt",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "bcrypt": "bcrypt",
    "requests": "requests",
    "pytest": "pytest",
}


#     _get_local_module_names(repo_path) -> set[str]:
#         - identifies the app's OWN top-level modules/packages, so they are
#           never mistaken for third-party PyPI packages
#         - a folder counts as a local package if it contains ANY .py file
#           anywhere inside it (NOT just if it has __init__.py — Python's
#           "namespace packages" are valid without one, and sample_app's
#           models/, services/, routes/ folders don't have __init__.py files,
#           which caused a real bug: "models" got treated as a third-party
#           import and pip tried to install an unrelated PyPI package
#           literally called "models", which failed to build)
#         - a top-level .py file also counts as a local module by its stem
#         - common non-package folders (tests, venv, .venv, __pycache__,
#           .git) are always excluded
def _get_local_module_names(repo_path: str) -> set:
    repo = Path(repo_path)
    excluded_dirs = {"tests", "venv", ".venv", "__pycache__", ".git", "data"}
    local_modules = set()

    for item in repo.iterdir():
        if item.name in excluded_dirs or item.name.startswith("."):
            continue
        if item.is_file() and item.suffix == ".py":
            local_modules.add(item.stem)
        elif item.is_dir():
            # Counts as a local package if it contains any .py file at all,
            # anywhere inside it — covers both regular packages (with
            # __init__.py) and namespace packages (without one)
            if any(item.rglob("*.py")):
                local_modules.add(item.name)

    return local_modules


#     _scan_third_party_imports(repo_path) -> set[str]:
#         - walks every .py file in repo_path (excluding tests/ and venv-like dirs)
#         - parses each file's AST to find top-level import statements
#         - filters out standard library modules and the app's own local
#           modules (see _get_local_module_names for how those are detected)
#         - maps known import-name-to-package-name mismatches
#         - returns the set of actual third-party package names to install
def _scan_third_party_imports(repo_path: str) -> set:
    stdlib = _get_stdlib_module_names()
    local_modules = _get_local_module_names(repo_path)
    repo = Path(repo_path)

    third_party = set()

    for py_file in repo.rglob("*.py"):
        if "tests" in py_file.parts or "venv" in py_file.parts or ".venv" in py_file.parts:
            continue

        try:
            source = py_file.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    _maybe_add(module_name, stdlib, local_modules, third_party)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:  # level == 0 means absolute import, not relative
                    module_name = node.module.split(".")[0]
                    _maybe_add(module_name, stdlib, local_modules, third_party)

    return third_party


def _maybe_add(module_name: str, stdlib: set, local_modules: set, third_party: set) -> None:
    if module_name in stdlib or module_name in local_modules:
        return
    package_name = _IMPORT_TO_PACKAGE_NAME.get(module_name, module_name)
    third_party.add(package_name)


#     ensure_requirements_file(repo_path) -> dict:
#         - scans the actual codebase for third-party imports (not hardcoded)
#         - if requirements.txt exists, checks whether scanned imports are
#           already covered; adds any missing ones rather than overwriting
#         - this means new imports Agent 2/Agent 5 add in future runs get
#           picked up automatically, instead of silently failing at build time
#         - returns {"created": bool, "updated": bool, "packages": list, "reason_code": str}
def ensure_requirements_file(repo_path: str) -> dict:
    req_path = Path(repo_path) / "requirements.txt"
    scanned_packages = _scan_third_party_imports(repo_path)

    if not req_path.exists():
        sorted_packages = sorted(scanned_packages)
        req_path.write_text("\n".join(sorted_packages) + "\n" if sorted_packages else "")
        return {
            "created": True, "updated": False,
            "packages": sorted_packages, "reason_code": "REQUIREMENTS_CREATED"
        }

    existing_content = req_path.read_text()
    existing_packages = {
        line.strip().split("==")[0].split(">=")[0].split("<=")[0].lower()
        for line in existing_content.splitlines() if line.strip() and not line.startswith("#")
    }
    missing = {p for p in scanned_packages if p.lower() not in existing_packages}

    if not missing:
        return {
            "created": False, "updated": False,
            "packages": sorted(existing_packages), "reason_code": "REQUIREMENTS_UP_TO_DATE"
        }

    with open(req_path, "a") as f:
        for package in sorted(missing):
            f.write(f"{package}\n")

    return {
        "created": False, "updated": True,
        "packages": sorted(missing), "reason_code": "REQUIREMENTS_UPDATED"
    }


#     fix_requirements_file(repo_path) -> dict:
#         - re-scans and REWRITES requirements.txt from scratch, removing any
#           entries that are actually local modules incorrectly listed as
#           packages (fixes the "models" bug retroactively for repos where
#           a broken requirements.txt already exists on disk)
#         - safe to call any time; only removes entries that match detected
#           local module names, never removes genuine third-party packages
def fix_requirements_file(repo_path: str) -> dict:
    req_path = Path(repo_path) / "requirements.txt"
    if not req_path.exists():
        return ensure_requirements_file(repo_path)

    local_modules = _get_local_module_names(repo_path)
    existing_lines = [
        line for line in req_path.read_text().splitlines() if line.strip()
    ]
    cleaned_lines = [
        line for line in existing_lines
        if line.strip().split("==")[0].split(">=")[0].strip().lower() not in local_modules
    ]

    removed = set(existing_lines) - set(cleaned_lines)
    if removed:
        req_path.write_text("\n".join(cleaned_lines) + "\n" if cleaned_lines else "")

    return {
        "removed_entries": sorted(removed),
        "reason_code": "REQUIREMENTS_CLEANED" if removed else "REQUIREMENTS_ALREADY_CLEAN"
    }


# Deterministic Dockerfile template — not LLM-generated, by design.
# Security baseline: non-root user, no debug flags, minimal base image.
_DOCKERFILE_TEMPLATE = """# syntax=docker/dockerfile:1
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 5000

CMD ["python3", "app.py"]
"""


#     ensure_dockerfile(repo_path) -> dict:
#         - creates a Dockerfile from the deterministic template if none exists
#         - never overwrites an existing Dockerfile — respects manual changes
#         - returns {"created": bool, "reason_code": str}
def ensure_dockerfile(repo_path: str) -> dict:
    dockerfile_path = Path(repo_path) / "Dockerfile"
    if dockerfile_path.exists():
        return {"created": False, "reason_code": "DOCKERFILE_EXISTS"}
    dockerfile_path.write_text(_DOCKERFILE_TEMPLATE)
    return {"created": True, "reason_code": "DOCKERFILE_CREATED"}


#     build_image(repo_path, image_name, tag) -> dict:
#         - runs `docker build -t image_name:tag .` in repo_path
#         - tags the image with the git commit hash (passed in as tag) so
#           every image traces back to an exact commit
#         - output truncated to last 2000 chars to keep Jira comments readable
#         - reason_code: BUILD_SUCCESS or BUILD_FAILED (marked recoverable —
#           a failed build doesn't corrupt anything, it's safe to retry)
def build_image(repo_path: str, image_name: str, tag: str) -> dict:
    full_tag = f"{image_name}:{tag}"
    try:
        result = subprocess.run(
            ["docker", "build", "-t", full_tag, "."],
            cwd=repo_path, capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "image_tag": full_tag, "error": "Build timed out after 600s", "reason_code": "BUILD_TIMEOUT", "recoverable": True}

    output = result.stdout + result.stderr
    truncated_output = output[-2000:] if len(output) > 2000 else output

    if result.returncode == 0:
        return {"success": True, "image_tag": full_tag, "output": truncated_output, "reason_code": "BUILD_SUCCESS"}
    return {"success": False, "image_tag": full_tag, "output": truncated_output, "reason_code": "BUILD_FAILED", "recoverable": True}