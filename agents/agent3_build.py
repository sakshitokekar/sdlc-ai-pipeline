# Build Agent — ensures the codebase has the files needed to containerize,
# builds a Docker image tagged with the exact commit it came from, and
# reports the result to Jira. Only runs after Agent 5's tests pass
# (enforced by pipeline.py's conditional routing, not by this agent itself).
from state.pipeline_state import StateSDLC
from tools.jira_tools import add_jira_comment
from tools.github_tools import commit_and_push_changes
from tools.docker_tools import (
    check_docker_available,
    ensure_requirements_file,
    fix_requirements_file,
    ensure_dockerfile,
    get_current_commit_hash,
    build_image,
)
from tools import log_utils as log

REPO_PATH = "../sample_app"
IMAGE_NAME = "sample-app"


def run_build_agent_node(state: StateSDLC) -> dict:
    jira_ticket = state["jira_ticket_details"]
    ticket_key = jira_ticket.get("ticket_key", "UNKNOWN")

    # --- Step 0: Refuse to build a stale image if Agent 2 INTENDED to fix
    # something but the fix silently failed to apply (e.g. exact-match
    # replacement or AST validation rejected every proposed change). This
    # is distinct from "Gemini legitimately found nothing to change" —
    # that case is safe to proceed past. Returning success: False here
    # naturally cascades to Agent 4, which already skips deployment when
    # build_results.success is False. ---
    had_intended_changes = state.get("had_intended_changes", False)
    code_changes_applied = state.get("code_changes_applied", False)

    if had_intended_changes and not code_changes_applied:
        log.error(
            f"Agent 2 proposed code changes for {ticket_key} but none were "
            f"successfully applied. Refusing to build/redeploy the stale image."
        )
        comment = f"""Agent 3 (Build) SKIPPED for {ticket_key}.

WHO: Agent 3 (Build Agent)
WHAT: Agent 2 proposed code changes that were meant to fix this ticket, but every proposed change was rejected (exact-match text not found, or would have produced invalid Python)
WHY: Refusing to rebuild/redeploy the existing image as if the fix had landed — that would silently ship unfixed code
ACTION NEEDED: Review Agent 2's comment on this ticket for the specific rejected changes, and either fix manually or re-run the pipeline to retry
"""
        add_jira_comment(ticket_key, comment)
        return {
            "build_results": {
                "success": False,
                "reason_code": "CHANGES_NOT_APPLIED"
            }
        }

    # --- Step 1: Verify Docker is actually available before doing anything else ---
    docker_status = check_docker_available()
    if not docker_status["available"]:
        error_detail = docker_status.get("error", "Unknown error")
        log.error(f"Docker unavailable: {docker_status['reason_code']} — {error_detail}")

        comment = f"""Agent 3 (Build) could not run for {ticket_key}.

WHO: Agent 3 (Build Agent)
WHAT: Attempted to build a Docker image but Docker is not available on this machine
WHY: {docker_status['reason_code']}: {error_detail}
ACTION NEEDED: Install/start Docker Desktop, then re-run the pipeline from this ticket
"""
        add_jira_comment(ticket_key, comment)
        return {
            "build_results": {
                "success": False,
                "reason_code": docker_status["reason_code"],
                "error": error_detail
            }
        }

    # --- Step 2: Clean up any incorrectly-scanned local modules from
    # requirements.txt first (fixes repos where a previous run mistakenly
    # listed a local package like "models" as an installable dependency) ---
    cleanup_result = fix_requirements_file(REPO_PATH)
    if cleanup_result["reason_code"] == "REQUIREMENTS_CLEANED":
        log.warn(f"Removed incorrect entries from requirements.txt: {cleanup_result['removed_entries']}")

    # --- Step 3: Ensure prerequisite files exist (requirements.txt, Dockerfile) ---
    req_result = ensure_requirements_file(REPO_PATH)
    dockerfile_result = ensure_dockerfile(REPO_PATH)

    new_infra_files = []
    if req_result["created"] or cleanup_result["reason_code"] == "REQUIREMENTS_CLEANED":
        new_infra_files.append("requirements.txt")
        log.step("requirements.txt created or corrected")
    if dockerfile_result["created"]:
        new_infra_files.append("Dockerfile")
        log.step("Created Dockerfile")

    # --- Step 4: Commit new/corrected infra files to the same branch Agent 2 used ---
    if new_infra_files:
        commit_result = commit_and_push_changes(
            repo_path=REPO_PATH,
            files=new_infra_files,
            commit_message=f"build({ticket_key}): add/fix Dockerfile and requirements.txt",
            branch=f"agent2/{ticket_key.lower()}"
        )
        log.step(f"Infra files commit result: {commit_result.get('reason_code', commit_result.get('success'))}")

    # --- Step 5: Build the Docker image, tagged with the exact commit hash ---
    commit_hash = get_current_commit_hash(REPO_PATH)
    log.step(f"Building Docker image tagged with commit {commit_hash}...")
    build_result = build_image(REPO_PATH, IMAGE_NAME, commit_hash)

    log.full(f"Full docker build output for {ticket_key}:\n{build_result.get('output', '')}")
    if build_result["success"]:
        log.success(f"Build result: {build_result['reason_code']}")
    else:
        log.error(f"Build result: {build_result['reason_code']}")
        print(log.truncate(build_result.get("output", ""), 800))

    # --- Step 6: Update Jira with audit trail ---
    if build_result["success"]:
        comment = f"""Agent 3 (Build) completed for {ticket_key}.

WHO: Agent 3 (Build Agent)
WHAT: Built Docker image {build_result['image_tag']}
WHY: Containerize the validated code for {ticket_key} ahead of deployment
COMMIT: {commit_hash}
RESULT: BUILD_SUCCESS
"""
    else:
        comment = f"""Agent 3 (Build) FAILED for {ticket_key}.

WHO: Agent 3 (Build Agent)
WHAT: Attempted to build Docker image {build_result['image_tag']}
WHY: Containerize the validated code for {ticket_key} ahead of deployment
RESULT: {build_result['reason_code']}
OUTPUT (tail): {build_result.get('output', 'No output captured')[-500:]}
"""
    add_jira_comment(ticket_key, comment)

    return {"build_results": build_result}


if __name__ == "__main__":
    test_state = {
        "user_input": "test",
        "jira_ticket_details": {"ticket_key": "SDLC-3", "url": "https://sakshitokekar.atlassian.net/browse/SDLC-3"},
        "code": "",
        "test_results": {},
        "dev_test_retry_count": 0,
        "human_decision": "",
        "build_results": {}
    }
    result = run_build_agent_node(test_state)
    print(result)