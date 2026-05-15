from langgraph.graph import START, END, StateGraph
from state.pipeline_state import StateSDLC
from agents.agent1_pm import run_pm_agent_node

if __name__ == "__main__":
    graph = StateGraph(StateSDLC)
    graph.add_node("run_pm_agent_node", run_pm_agent_node)
    graph.add_edge(START, "run_pm_agent_node")
    graph.add_edge("run_pm_agent_node", END)
    
    app = graph.compile()

    results = app.invoke({
        "user_input": "Create a Jira ticket for a login feature bug where users cannot sign in with valid credentials on the mobile app since yesterday's deployment",
        "jira_ticket_details": {},
        "code": ""
    })
    print(results)    