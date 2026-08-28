# Must be the very first import — sets quiet-mode env vars before any
# downstream import (agent2_dev -> Embedder -> sentence_transformers)
# triggers HuggingFace's noisy progress bars.
from tools import log_utils as log

import json
import uuid
from pathlib import Path

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from state.pipeline_state import StateSDLC
from agents.agent1_pm import run_pm_agent_node
from agents.agent2_dev import run_dev_agent_node
from agents.agent5_test import run_test_agent_node
from agents.agent3_build import run_build_agent_node
from agents.agent4_deploy import run_deploy_agent_node

MAX_RETRIES = 3  # guardrail: prevents Agent 2 <-> Agent 5 from looping forever

# Persists which thread_id the last pipeline run used, so a subsequent
# `python3 pipeline.py` execution can detect an INCOMPLETE previous run
# and resume it (skipping already-completed nodes) instead of blindly
# restarting from START every time — which is what checkpointing is
# actually for. Without this file, every invoke() used a fresh full
# input dict on a hardcoded thread_id, which tells LangGraph "start a
# brand new run," making the checkpointer pure overhead with no benefit.
_RUN_STATE_PATH = Path("codebase_search_rag/data/current_run.json")


def increment_retry_node(state: StateSDLC) -> dict:
    """Runs before Agent 2 on a retry — increments the retry counter.
    Separated from run_dev_agent_node so Agent 2 itself doesn't need
    to know about retry logic — single responsibility."""
    current = state.get("dev_test_retry_count", 0)
    return {"dev_test_retry_count": current + 1}


def escalate_to_human_node(state: StateSDLC) -> dict:
    """Runs when max retries are exceeded and tests are still failing.
    Pauses the graph via interrupt() and surfaces the failure details
    to a human, who must decide whether to approve continuing anyway,
    retry with more attempts, or abandon the ticket."""
    ticket_key = state.get("jira_ticket_details", {}).get("ticket_key", "UNKNOWN")
    test_summary = state.get("test_results", {}).get("summary", "No summary available")
    retry_count = state.get("dev_test_retry_count", 0)

    human_decision = interrupt({
        "reason": "MAX_RETRIES_EXCEEDED",
        "ticket_key": ticket_key,
        "retry_count": retry_count,
        "test_summary": test_summary,
        "message": f"Agent 2 <-> Agent 5 loop failed {retry_count} times for {ticket_key}. "
                   f"Tests still failing: {test_summary}. "
                   f"Reply with 'approve' to proceed anyway, 'retry' to allow more attempts, "
                   f"or 'abandon' to stop the pipeline."
    })

    return {"human_decision": human_decision}


def route_after_test(state: StateSDLC) -> str:
    """Conditional edge: if tests pass -> proceed to Agent 3 (Build). If
    tests fail and under the retry limit -> loop back to Agent 2. If retry
    limit exceeded -> escalate to a human via interrupt()."""
    test_results = state.get("test_results", {})
    retry_count = state.get("dev_test_retry_count", 0)

    if test_results.get("passed", False):
        return "build"

    if retry_count >= MAX_RETRIES:
        return "escalate"

    return "retry"


def route_after_human_decision(state: StateSDLC) -> str:
    """After a human responds to the interrupt(), route based on their
    decision. 'approve' proceeds to Build despite failing tests (human
    override) — 'retry' loops back — 'abandon' ends the pipeline."""
    decision = state.get("human_decision", "abandon")
    if decision == "retry":
        return "retry"
    if decision == "approve":
        return "build"
    return "end"  # abandon


def build_graph():
    """Builds and returns the unconfigured StateGraph. Separated from
    compilation so the checkpointer context manager can wrap compile()
    and invoke() together in __main__."""
    graph = StateGraph(StateSDLC)

    graph.add_node("run_pm_agent_node", run_pm_agent_node)
    graph.add_node("run_dev_agent_node", run_dev_agent_node)
    graph.add_node("run_test_agent_node", run_test_agent_node)
    graph.add_node("run_build_agent_node", run_build_agent_node)
    graph.add_node("run_deploy_agent_node", run_deploy_agent_node)
    graph.add_node("increment_retry_node", increment_retry_node)
    graph.add_node("escalate_to_human_node", escalate_to_human_node)

    graph.add_edge(START, "run_pm_agent_node")
    graph.add_edge("run_pm_agent_node", "run_dev_agent_node")
    graph.add_edge("run_dev_agent_node", "run_test_agent_node")

    graph.add_conditional_edges(
        "run_test_agent_node",
        route_after_test,
        {
            "build": "run_build_agent_node",
            "retry": "increment_retry_node",
            "escalate": "escalate_to_human_node"
        }
    )
    graph.add_edge("increment_retry_node", "run_dev_agent_node")
    graph.add_edge("run_build_agent_node", "run_deploy_agent_node")
    graph.add_edge("run_deploy_agent_node", END)

    graph.add_conditional_edges(
        "escalate_to_human_node",
        route_after_human_decision,
        {
            "build": "run_build_agent_node",
            "retry": "increment_retry_node",
            "end": END
        }
    )

    return graph


def _load_saved_thread_id() -> str | None:
    if _RUN_STATE_PATH.exists():
        try:
            return json.loads(_RUN_STATE_PATH.read_text()).get("thread_id")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_thread_id(thread_id: str) -> None:
    _RUN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RUN_STATE_PATH.write_text(json.dumps({"thread_id": thread_id}))


def _print_interrupt_prompt(interrupt_data: dict) -> str:
    print("\n" + "=" * 60)
    print("PIPELINE PAUSED — HUMAN DECISION REQUIRED")
    print("=" * 60)
    print(interrupt_data["message"])
    print("=" * 60)
    return input("Your decision (approve/retry/abandon): ").strip().lower()


if __name__ == "__main__":
    graph = build_graph()

    with SqliteSaver.from_conn_string("codebase_search_rag/data/pipeline_checkpoints.db") as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        # --- Determine whether to RESUME an incomplete previous run or
        # START a fresh one. This is what actually makes the checkpointer
        # useful: we inspect the saved thread's state and only treat it as
        # "still in progress" if the graph genuinely has pending nodes. ---
        saved_thread_id = _load_saved_thread_id()
        resume_run = False
        thread_id = None

        if saved_thread_id:
            candidate_config = {"configurable": {"thread_id": saved_thread_id}}
            try:
                snapshot = app.get_state(candidate_config)
            except Exception:
                snapshot = None

            if snapshot is not None and snapshot.next:
                # .next is a non-empty tuple of pending node names — the
                # previous run crashed or was interrupted before reaching
                # END. Resume it rather than starting over.
                log.step(f"Found an incomplete previous run (thread {saved_thread_id}), pending: {snapshot.next}. Resuming...")
                resume_run = True
                thread_id = saved_thread_id
                config = candidate_config

        if not resume_run:
            thread_id = str(uuid.uuid4())
            _save_thread_id(thread_id)
            config = {"configurable": {"thread_id": thread_id}}

        if resume_run:
            # Passing None as input tells LangGraph "don't inject new
            # input, just continue executing pending work from the last
            # checkpoint" — this is what actually skips already-completed
            # nodes instead of rerunning the whole pipeline from START.
            results = app.invoke(None, config=config)
        else:
            results = app.invoke({
                "user_input": "The Flask app inside the Docker container only binds to host 127.0.0.1 (localhost), so it is unreachable from outside the container when deployed to Kubernetes. app.run() in app.py needs host='0.0.0.0' so it listens on all network interfaces, not just localhost. Additionally, debug=True should be disabled since it exposes an interactive debugger with a PIN, which is a security risk in any deployed environment.",
                "jira_ticket_details": {},
                "code": "",
                "test_results": {},
                "dev_test_retry_count": 0,
                "human_decision": "",
                "build_results": {},
                "deploy_results": {}
            }, config=config)

        # If the graph paused at an interrupt(), results will contain a
        # special "__interrupt__" key instead of finishing normally.
        if "__interrupt__" in results:
            interrupt_data = results["__interrupt__"][0].value
            human_input = _print_interrupt_prompt(interrupt_data)
            results = app.invoke(Command(resume=human_input), config=config)

        # Once a thread reaches a true terminal state (no pending nodes),
        # clear the saved thread_id so the NEXT `python3 pipeline.py` run
        # starts a fresh ticket instead of trying to resume a finished one.
        final_snapshot = app.get_state(config)
        if not final_snapshot.next and _RUN_STATE_PATH.exists():
            _RUN_STATE_PATH.unlink()

        log.success("Pipeline run complete.")
        print(json.dumps({
            k: v for k, v in results.items() if k != "code"
        }, indent=2, default=str))
        if results.get("code"):
            print(f"\ncode (truncated): {log.truncate(results['code'], 400)}")