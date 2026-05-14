# Jira API functions (called by agent 1)
import requests 
from requests.auth import HTTPBasicAuth
import json
from config.settings import jira_base_url, jira_email, jira_api_token

CREATE_JIRA_TICKET_DECLARATION = {
        "name": "create_jira_ticket",
        "description": "this tool creates a detailed jira ticket, by filling as many details as possible for a jira ticket like a project manager would do",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "the problem in one line eg: The dropdown feature is not showing abc...."
                },
                "description": {
                    "type": "string",
                    "description": "the possible cause, location, expected behaviour, success and failure test cases if available, related documents, previous tickets if available, sme contact, since when its an issue if its a bug. If its a feature, then current behaviour, expected new change, where they think the change could be made if they know, some test cases to check if it works."
                },
                "issue_type": {
                    "type": "string",
                    "description": "bug or feature"
                },
                "priority": {
                    "type": "string",
                    "description": "low, medium, high, critical"
                },
                "acceptance_criteria": {
                    "type": "string",
                    "description": "checklist of conditions for ticket to be DONE"
                },
                "attachments": {
                    "type": "string",
                    "description": "List of attached document filenames e.g. technical_spec.pdf, functional_doc.docx"
                }
            },
            "required": ["summary", "description", "issue_type", "priority"]
        }
    }

def create_jira_ticket(summary: str, description: str, issue_type: str, priority: str, acceptance_criteria: str = "", attachments: str = "") -> dict[str, int|str]:
    url = f"{jira_base_url}/rest/api/3/issue"
    auth = HTTPBasicAuth(jira_email, jira_api_token)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    priority_map = {
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "critical": "Highest"
    }
    issue_type_map = {
        "bug": "Bug",
        "feature": "Story"
    }
    payload = json.dumps({
        "fields": {
            "project": {"key": "SDLC"},
            "summary": summary,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": description}]
                    }
                ]
            },
            "issuetype": {"name": issue_type_map.get(issue_type.lower(), "Bug")},
            "priority": {"name": priority_map.get(priority.lower(), "Medium")},
        }
    })
    
    response = requests.request(
        "POST",
        url,
        data=payload,
        headers=headers,
        auth=auth
    )
    if response.status_code != 201:
        return {"error": response.json()}
    response_data = response.json()
    return {"ticket_key": response_data.get("key"), "url": f"{jira_base_url}/browse/{response_data.get('key')}"}