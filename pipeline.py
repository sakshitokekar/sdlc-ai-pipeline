from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from state.pipeline_state import StateSDLC
from agents.agent1_pm import run_pm_agent_node
from agents.agent2_dev import run_dev_agent_node
from agents.agent5_test import run_test_agent_node

MAX_RETRIES = 3  # guardrail: prevents Agent 2 <-> Agent 5 from looping forever


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
    """Conditional edge: if tests pass -> end. If tests fail and under
    the retry limit -> loop back to Agent 2. If retry limit exceeded ->
    escalate to a human via interrupt() rather than looping forever
    or silently giving up."""
    test_results = state.get("test_results", {})
    retry_count = state.get("dev_test_retry_count", 0)

    if test_results.get("passed", False):
        return "end"

    if retry_count >= MAX_RETRIES:
        return "escalate"

    return "retry"


def route_after_human_decision(state: StateSDLC) -> str:
    """After a human responds to the interrupt(), route based on their decision."""
    decision = state.get("human_decision", "abandon")
    if decision == "retry":
        return "retry"
    return "end"  # approve or abandon both end the automated pipeline here


if __name__ == "__main__":
    graph = StateGraph(StateSDLC)

    graph.add_node("run_pm_agent_node", run_pm_agent_node)
    graph.add_node("run_dev_agent_node", run_dev_agent_node)
    graph.add_node("run_test_agent_node", run_test_agent_node)
    graph.add_node("increment_retry_node", increment_retry_node)
    graph.add_node("escalate_to_human_node", escalate_to_human_node)

    graph.add_edge(START, "run_pm_agent_node")
    graph.add_edge("run_pm_agent_node", "run_dev_agent_node")
    graph.add_edge("run_dev_agent_node", "run_test_agent_node")

    graph.add_conditional_edges(
        "run_test_agent_node",
        route_after_test,
        {
            "end": END,
            "retry": "increment_retry_node",
            "escalate": "escalate_to_human_node"
        }
    )
    graph.add_edge("increment_retry_node", "run_dev_agent_node")

    graph.add_conditional_edges(
        "escalate_to_human_node",
        route_after_human_decision,
        {
            "end": END,
            "retry": "increment_retry_node"
        }
    )

    # MemorySaver checkpointer is REQUIRED for interrupt() to work —
    # it saves the paused state while waiting for the human's response.
    # Dev-only: in-memory, lost on restart. Upgrade to SqliteSaver/PostgresSaver
    # for production persistence (tracked in TODOs).
    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    # A thread_id is required by the checkpointer to know which paused
    # conversation to resume. One thread per pipeline run.
    config = {"configurable": {"thread_id": "sdlc-run-1"}}

    results = app.invoke({
        "user_input": "Create a Jira ticket for a login feature bug where users cannot sign in with valid credentials on the mobile app since yesterday's deployment",
        "jira_ticket_details": {},
        "code": "",
        "test_results": {},
        "dev_test_retry_count": 0
    }, config=config)

    # If the graph paused at an interrupt(), results will contain
    # a special "__interrupt__" key instead of finishing normally.
    if "__interrupt__" in results:
        interrupt_data = results["__interrupt__"][0].value
        print("\n" + "=" * 60)
        print("PIPELINE PAUSED — HUMAN DECISION REQUIRED")
        print("=" * 60)
        print(interrupt_data["message"])
        print("=" * 60)

        human_input = input("Your decision (approve/retry/abandon): ").strip().lower()

        # Resume the graph with the human's decision
        from langgraph.types import Command
        results = app.invoke(Command(resume=human_input), config=config)

    print(results)