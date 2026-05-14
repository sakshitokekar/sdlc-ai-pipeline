# PM agent
from typing import TypedDict
from google import genai
from google.genai import types
from tools.jira_tools import create_jira_ticket, CREATE_JIRA_TICKET_DECLARATION
from state.pipeline_state import StateSDLC
from config.settings import gem_api_key
client = genai.Client(api_key = gem_api_key)

def run_pm_agent_node(state: StateSDLC) -> dict:
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    tools = types.Tool(function_declarations=[CREATE_JIRA_TICKET_DECLARATION])
    config = types.GenerateContentConfig(system_instruction="You are a PM creating a JIRA Ticket. Always use create_jira_ticket tool. Never ask clarifying questions. Make reasonable assumptions for any missing fields",tools=[tools])

    contents = [
        types.Content(
            role="user", parts=[types.Part(text=state["user_input"])]
        )]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config,
    )
    
    print(response.candidates[0].content.parts[0].function_call)
    
    # Extract tool call details, it may not be in the first part.
    tool_call = response.candidates[0].content.parts[0].function_call

    if tool_call is None:
        print("Model did not call a tool. Response was:")
        print(response.text)
        return

    if tool_call.name == "create_jira_ticket":
        result = create_jira_ticket(**tool_call.args)
        print(f"Function execution result: {result}")
        
    # Create a function response part
    function_response_part = types.Part.from_function_response(
        name=tool_call.name,
        response={"result": result}
    )

    # Append function call and result of the function execution to contents
    contents.append(response.candidates[0].content) # Append the content from the model's response.
    contents.append(types.Content(role="user", parts=[function_response_part])) # Append the function response

    final_response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=config,
        contents=contents,
    )

    return {"jira_ticket_details" : result}
    
if __name__ == "__main__":
    run_pm_agent_node({"user_input": "Create a Jira ticket for a login feature bug...", "jira_ticket_details": {}, "code": ""})