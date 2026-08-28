# Test Agent — orchestrates test running and coverage generation via test_tools,
# then reports results and updates Jira via jira_tools
from state.pipeline_state import StateSDLC
from tools.jira_tools import add_jira_comment, update_jira_status
from tools.test_tools import run_pytest, get_existing_test_code, generate_test_coverage, append_and_commit_tests
from tools import log_utils as log

REPO_PATH = "../sample_app"


def run_test_agent_node(state: StateSDLC) -> dict:
    jira_ticket = state["jira_ticket_details"]
    ticket_key = jira_ticket.get("ticket_key", "UNKNOWN")
    code_changes = state.get("code", "")

    # --- Step 1: Run existing tests to establish baseline ---
    log.step("Running existing tests...")
    baseline_result = run_pytest(REPO_PATH)
    log.step(f"Existing tests: {baseline_result['reason_code']}")

    # --- Step 2: Check if new test coverage is needed for the change ---
    existing_tests = get_existing_test_code(REPO_PATH)
    log.step("Checking if new test coverage is needed...")
    coverage_decision = generate_test_coverage(ticket_key, code_changes, existing_tests)
    log.step(f"New tests needed: {coverage_decision.get('needs_new_tests')} — {coverage_decision.get('reasoning', '')[:150]}")

    # --- Step 3: Add new tests if needed ---
    new_tests_added = False
    if coverage_decision.get("needs_new_tests") and coverage_decision.get("new_test_code"):
        commit_result = append_and_commit_tests(REPO_PATH, ticket_key, coverage_decision["new_test_code"])
        new_tests_added = commit_result.get("success", False)
        log.step(f"New tests committed: {new_tests_added}")

    # --- Step 4: Re-run full test suite (includes new tests if added) ---
    log.step("Running final test suite...")
    final_result = run_pytest(REPO_PATH)
    # Full output is written to the log file; only the tail is shown in
    # the terminal. Full output also still goes to Jira and StateSDLC.
    log.full(f"Full pytest output for {ticket_key}:\n{final_result['output']}")
    print(log.truncate(final_result["output"], 800))
    log.step(f"Final result: {final_result['reason_code']}")

    # --- Step 5: Update Jira with audit trail ---
    comment = f"""Agent 5 (Test) completed test run for {ticket_key}.

WHO: Agent 5 (Test Agent)
WHAT: Ran pytest suite. New tests added: {new_tests_added}
WHY: Verify code changes for {ticket_key} don't break existing functionality and have adequate coverage
RESULT: {final_result['reason_code']}
SUMMARY: {final_result['summary']}
QA REASONING: {coverage_decision.get('reasoning', 'N/A')}
"""
    add_jira_comment(ticket_key, comment)
    update_jira_status(ticket_key, "In Review" if final_result["success"] else "In Progress")

    return {
        "test_results": {
            "passed": final_result["success"],
            "summary": final_result["summary"],
            "new_tests_added": new_tests_added,
            "full_output": final_result["output"]
        }
    }


if __name__ == "__main__":
    test_state = {
        "user_input": "test",
        "jira_ticket_details": {"ticket_key": "SDLC-3", "url": "https://sakshitokekar.atlassian.net/browse/SDLC-3"},
        "code": "Modified login_user and register_user to normalize email to lowercase for case-insensitive comparison",
        "test_results": {},
        "dev_test_retry_count": 0,
        "human_decision": "",
        "build_results": {}
    }
    result = run_test_agent_node(test_state)
    print(result)