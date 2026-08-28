from typing import TypedDict

class StateSDLC(TypedDict):
    user_input: str
    jira_ticket_details: dict
    code: str
    code_changes_applied: bool  # set by Agent 2: True only if at least one file was actually modified and committed
    had_intended_changes: bool  # set by Agent 2: True if Gemini proposed any file changes, regardless of whether they were successfully applied
    test_results: dict
    dev_test_retry_count: int  # tracks retries in the Agent 2 (Dev) <-> Agent 5 (Test) loop, prevents infinite loops
    human_decision: str  # set when a human responds to an interrupt() escalation: "approve", "retry", or "abandon"
    build_results: dict  # set by Agent 3 (Build): {"success": bool, "image_tag": str, "reason_code": str}
    deploy_results: dict  # set by Agent 4 (Deploy): {"success": bool, "pod_name": str, "node_port": str, "tunnel_command": str, "reason_code": str}