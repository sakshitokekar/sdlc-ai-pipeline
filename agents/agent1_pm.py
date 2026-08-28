# PM agent
from google import genai
from google.genai import types
from tools.jira_tools import create_jira_ticket, CREATE_JIRA_TICKET_DECLARATION
from tools.gemini_utils import generate_with_retry
from tools import log_utils as log
from state.pipeline_state import StateSDLC
from config.settings import gem_api_key, GEMINI_MODEL

client = genai.Client(api_key=gem_api_key)


def run_pm_agent_node(state: StateSDLC) -> dict:
    tools = types.Tool(function_declarations=[CREATE_JIRA_TICKET_DECLARATION])
    config = types.GenerateContentConfig(
        system_instruction="You are a PM creating a JIRA Ticket. Always use create_jira_ticket tool. Never ask clarifying questions. Make reasonable assumptions for any missing fields",
        tools=[tools]
    )

    contents = [
        types.Content(role="user", parts=[types.Part(text=state["user_input"])])
    ]

    log.step("Asking Gemini to draft a Jira ticket...")
    response = generate_with_retry(client, GEMINI_MODEL, contents, config)
    if response is None:
        log.error("Gemini API unavailable — cannot create ticket.")
        return {"jira_ticket_details": {}}

    tool_call = response.candidates[0].content.parts[0].function_call

    if tool_call is None:
        log.warn("Model did not call a tool. Response was:")
        print(log.truncate(response.text))
        return {"jira_ticket_details": {}}

    if tool_call.name == "create_jira_ticket":
        result = create_jira_ticket(**tool_call.args)
        log.success(f"Jira ticket created: {result.get('ticket_key', 'unknown')} — {result.get('url', '')}")
    else:
        log.warn(f"Unexpected tool call: {tool_call.name}")
        return {"jira_ticket_details": {}}

    # Report the tool result back to Gemini so it can "observe" what
    # happened — completes the Reason -> Act -> Observe loop. The
    # confirmation text itself isn't used anywhere yet; kept for the
    # Phase 5 frontend narration feature.
    function_response_part = types.Part.from_function_response(
        name=tool_call.name,
        response={"result": result}
    )
    contents.append(response.candidates[0].content)
    contents.append(types.Content(role="user", parts=[function_response_part]))
    generate_with_retry(client, GEMINI_MODEL, contents, config)

    return {"jira_ticket_details": result}


if __name__ == "__main__":
    test_state = {
        "user_input": "Create a Jira ticket for a login feature bug where users cannot sign in with valid credentials on the mobile app since yesterday's deployment",
        "jira_ticket_details": {},
        "code": "",
        "test_results": {},
        "dev_test_retry_count": 0,
        "human_decision": "",
        "build_results": {}
    }
    result = run_pm_agent_node(test_state)
    print(result)