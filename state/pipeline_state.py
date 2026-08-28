from typing import TypedDict

class StateSDLC(TypedDict):
    user_input: str
    jira_ticket_details: dict
    code: str
    test_results: dict
    dev_test_retry_count: int  # tracks retries in the Agent 2 (Dev) <-> Agent 5 (Test) loop, prevents infinite loops
    human_decision: str  # set when a human responds to an interrupt() escalation: "approve", "retry", or "abandon"
    build_results: dict  # set by Agent 3 (Build): {"success": bool, "image_tag": str, "reason_code": str}