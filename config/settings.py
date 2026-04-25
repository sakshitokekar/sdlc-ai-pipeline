# API keys, config values (never commit this)

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

gem_api_key = os.getenv("GEMINI_API_KEY")
jira_api_token = os.getenv("JIRA_API_TOKEN")
jira_email = os.getenv("JIRA_EMAIL")
jira_base_url = os.getenv("JIRA_BASE_URL")