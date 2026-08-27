# Dev Agent — uses Agentic RAG (semantic search + graph traversal) to find
# relevant code, then uses Gemini to generate a code fix, then commits to GitHub
import json
import re
from google import genai
from google.genai import types

from state.pipeline_state import StateSDLC
from config.settings import gem_api_key, get_chroma_collection
from codebase_search_rag.embedder import Embedder
from codebase_search_rag.indexer import Indexer
from codebase_search_rag.retriever import CodeRetriever
from codebase_search_rag.augmentor import Augmentor
from tools.github_tools import commit_and_push_changes
from tools.jira_tools import add_jira_comment, update_jira_status

client = genai.Client(api_key=gem_api_key)

REPO_PATH = "../sample_app"  # adjust to your local sample_app path


def _normalize_relative_path(file_path: str, repo_path: str) -> str:
    """
    Gemini sometimes returns file_path as a full relative path (e.g. '../sample_app/models/user.py')
    and sometimes as just the path within the repo (e.g. 'models/user.py').
    This normalizes it to always be relative to repo_path, so git commands
    (which run with cwd=repo_path) and open() calls work correctly.
    """
    # Extract just the folder name from repo_path (e.g. "sample_app" from "../sample_app")
    repo_folder_name = repo_path.rstrip("/").split("/")[-1]
    normalized_repo = repo_path.rstrip("/")

    # Strip full relative prefix like "../sample_app/"
    if file_path.startswith(normalized_repo):
        return file_path[len(normalized_repo):].lstrip("/")

    # Strip bare folder name prefix like "sample_app/"
    if file_path.startswith(repo_folder_name + "/"):
        return file_path[len(repo_folder_name) + 1:]

    return file_path.lstrip("/")


def run_dev_agent_node(state: StateSDLC) -> dict:
    jira_ticket = state["jira_ticket_details"]
    ticket_key = jira_ticket.get("ticket_key", "UNKNOWN")

    # --- Step 1: Set up ChromaDB, Embedder, Indexer, Retriever, Augmentor ---
    collection = get_chroma_collection()
    embedder = Embedder()
    indexer = Indexer(REPO_PATH, embedder, collection)
    retriever = CodeRetriever(collection, embedder, indexer)
    augmentor = Augmentor()

    # --- Step 2: Index the repository (incremental — only changed files re-indexed) ---
    print(f"Indexing repository at {REPO_PATH}...")
    indexer.index_repository()

    # --- Step 3: Build the search query from the Jira ticket ---
    query = state.get("user_input", "")

    # --- Step 4: Agentic RAG retrieval — semantic search + graph expansion ---
    print(f"Retrieving relevant code for query: {query}")
    retrieval_result = retriever.retrieve(query, top_k=10)
    print(f"Found {len(retrieval_result['chunks'])} relevant chunks")

    # --- Step 5: Build the prompt using Augmentor ---
    prompt = augmentor.build_prompt(jira_ticket, retrieval_result)

    # --- Step 6: Call Gemini to generate the code fix ---
    config = types.GenerateContentConfig(
        system_instruction="You are a senior Python developer. Always respond with valid JSON matching the requested schema. Never include markdown code fences around the JSON itself. In file_path fields, always use paths RELATIVE TO THE REPOSITORY ROOT (e.g. 'models/user.py'), never include the repo folder name itself."
    )
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=config,
    )

    raw_text = response.text.strip()
    print("Gemini response received.")

    # --- Step 7: Parse Gemini's JSON response ---
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    try:
        change_plan = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"Failed to parse Gemini response as JSON: {e}")
        print(f"Raw response: {raw_text}")
        return {"code": ""}

    # --- Step 8: Apply code changes to files ---
    from datetime import datetime
    actual_timestamp = datetime.utcnow().isoformat()
    modified_files = []
    for file_change in change_plan.get("files_to_modify", []):
        raw_file_path = file_change["file_path"]
        relative_path = _normalize_relative_path(raw_file_path, REPO_PATH)
        full_path = f"{REPO_PATH}/{relative_path}"

        try:
            with open(full_path, "r") as f:
                file_content = f.read()
        except FileNotFoundError:
            print(f"File not found: {full_path}, skipping.")
            continue

        for change in file_change.get("changes", []):
            original = change.get("original_code", "")
            modified = change.get("modified_code", "").replace("{TIMESTAMP}", actual_timestamp)
            if original and original in file_content:
                file_content = file_content.replace(original, modified)
            else:
                print(f"Warning: original_code not found exactly in {relative_path}, skipping this change.")

        with open(full_path, "w") as f:
            f.write(file_content)

        modified_files.append(relative_path)
        print(f"Modified: {relative_path}")

    # --- Step 9: Commit and push to GitHub ---
    commit_message = change_plan.get("commit_message", f"fix({ticket_key}): automated change")
    commit_result = commit_and_push_changes(
        repo_path=REPO_PATH,
        files=modified_files,
        commit_message=commit_message,
        branch=f"agent2/{ticket_key.lower()}"
    )
    print(f"Commit result: {commit_result}")

    # --- Step 10: Update Jira with audit trail ---
    comment = f"""Agent 2 (Dev) completed code changes for {ticket_key}.

WHO: Agent 2 (Dev Agent)
WHAT: {change_plan.get('analysis', 'Code changes applied')}
WHY: Addresses requirements in {ticket_key}
WHERE: {', '.join(modified_files) if modified_files else 'No files modified'}
COMMIT: {commit_result.get('commit_url', 'N/A')}
"""
    add_jira_comment(ticket_key, comment)
    update_jira_status(ticket_key, "In Progress")

    return {
        "code": json.dumps(change_plan),
    }


if __name__ == "__main__":
    test_state = {
            "user_input": "Fix login bug where users cannot sign in with valid credentials",
            "jira_ticket_details": {"ticket_key": "SDLC-2", "url": "https://sakshitokekar.atlassian.net/browse/SDLC-2"},
            "code": ""
    }
    result = run_dev_agent_node(test_state)
    print(result)
    
    