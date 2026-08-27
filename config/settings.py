# API keys, config values (never commit this)

import os
from dotenv import load_dotenv, find_dotenv
import chromadb

load_dotenv(find_dotenv())

gem_api_key = os.getenv("GEMINI_API_KEY")
jira_api_token = os.getenv("JIRA_API_TOKEN")
jira_email = os.getenv("JIRA_EMAIL")
jira_base_url = os.getenv("JIRA_BASE_URL")
GEMINI_MODEL = "gemini-3.7-flash"

def get_chroma_collection(collection_name: str = "codebase_sakshitokekar_sample_app"):
    chroma_client = chromadb.PersistentClient(path="codebase_search_rag/data/chroma_db")
    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )