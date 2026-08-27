# GitHub tools — commit and push code changes made by Agent 2
import subprocess


def commit_and_push_changes(repo_path: str, files: list[str], commit_message: str, branch: str) -> dict:
    """
    Commits and pushes changes to a feature branch on GitHub.
    Returns dict with success status and commit URL.
    """
    try:
        # Create/switch to feature branch
        subprocess.run(
            ["git", "checkout", "-B", branch],
            cwd=repo_path, check=True, capture_output=True, text=True
        )

        # Stage only the modified files
        if not files:
            return {"success": False, "error": "No files to commit"}

        subprocess.run(
            ["git", "add"] + files,
            cwd=repo_path, check=True, capture_output=True, text=True
        )

        # Commit
        subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=repo_path, check=True, capture_output=True, text=True
        )

        # Get commit hash
        commit_hash_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path, check=True, capture_output=True, text=True
        )
        commit_hash = commit_hash_result.stdout.strip()

        # Push to remote
        subprocess.run(
            ["git", "push", "-u", "origin", branch],
            cwd=repo_path, check=True, capture_output=True, text=True
        )

        # Get remote URL to build commit URL
        remote_result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo_path, check=True, capture_output=True, text=True
        )
        remote_url = remote_result.stdout.strip().removesuffix(".git")
        commit_url = f"{remote_url}/commit/{commit_hash}"

        return {
            "success": True,
            "commit_hash": commit_hash,
            "commit_url": commit_url,
            "branch": branch
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": e.stderr,
            "reason_code": "GIT_ERROR",
            "recoverable": True
        }