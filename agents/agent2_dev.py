# Dev Agent — uses Agentic RAG (semantic search + graph traversal) to find
# relevant code, then uses Gemini to generate a code fix, then commits to GitHub
import json
import re
from datetime import datetime
from google import genai
from google.genai import types

from state.pipeline_state import StateSDLC
from config.settings import gem_api_key, get_chroma_collection, GEMINI_MODEL
from codebase_search_rag.embedder import Embedder
from codebase_search_rag.indexer import Indexer
from codebase_search_rag.retriever import CodeRetriever
from codebase_search_rag.augmentor import Augmentor
from tools.github_tools import commit_and_push_changes
from tools.gemini_utils import generate_with_retry
from tools.jira_tools import add_jira_comment, update_jira_status

client = genai.Client(api_key=gem_api_key)

REPO_PATH = "../sample_app"


def _normalize_relative_path(file_path: str, repo_path: str) -> str:
    """
    Gemini sometimes returns file_path as a full relative path (e.g. '../sample_app/models/user.py'),
    sometimes with just the folder name prefix (e.g. 'sample_app/models/user.py'), and sometimes
    already correct (e.g. 'models/user.py'). This normalizes it to always be relative to repo_path.
    """
    repo_folder_name = repo_path.rstrip("/").split("/")[-1]
    normalized_repo = repo_path.rstrip("/")

    if file_path.startswith(normalized_repo):
        return file_path[len(normalized_repo):].lstrip("/")
    if file_path.startswith(repo_folder_name + "/"):
        return file_path[len(repo_folder_name) + 1:]
    return file_path.lstrip("/")


def run_dev_agent_node(state: StateSDLC) -> dict:
    jira_ticket = state["jira_ticket_details"]
    ticket_key = jira_ticket.get("ticket_key", "UNKNOWN")

    # --- Determine if this is a retry after a test failure ---
    retry_count = state.get("dev_test_retry_count", 0)
    previous_failure = None
    if retry_count > 0:
        previous_test_results = state.get("test_results", {})
        if previous_test_results:
            previous_failure = {
                "retry_count": retry_count,
                "summary": previous_test_results.get("summary", "Unknown failure"),
                "full_output": previous_test_results.get("full_output", "")
            }
            print(f"Retry attempt #{retry_count} — informing Agent 2 of previous test failure: {previous_failure['summary']}")

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

    # --- Step 5: Build the prompt using Augmentor, including retry context if applicable ---
    prompt = augmentor.build_prompt(jira_ticket, retrieval_result, previous_failure=previous_failure)

    # --- Step 6: Call Gemini to generate the code fix (with automatic retry on transient errors) ---
    config = types.GenerateContentConfig(
        system_instruction="You are a senior Python developer. Always respond with valid JSON matching the requested schema. Never include markdown code fences around the JSON itself. In file_path fields, always use paths RELATIVE TO THE REPOSITORY ROOT (e.g. 'models/user.py'), never include the repo folder name itself."
    )
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    response = generate_with_retry(client, GEMINI_MODEL, contents, config)
    if response is None:
        print("Gemini API unavailable after retries — cannot generate code fix.")
        return {"code": ""}

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
    actual_timestamp = datetime.now().isoformat()
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
    if retry_count > 0:
        commit_message = f"{commit_message} (retry #{retry_count})"

    commit_result = commit_and_push_changes(
        repo_path=REPO_PATH,
        files=modified_files,
        commit_message=commit_message,
        branch=f"agent2/{ticket_key.lower()}"
    )
    print(f"Commit result: {commit_result}")

    # --- Step 10: Update Jira with audit trail ---
    # NOTE: this f-string is deliberately flush-left even though it's inside
    # an indented function body — otherwise every line of the Jira comment
    # would carry leading whitespace and render messily in Jira's UI.
    retry_note = f" (retry attempt #{retry_count})" if retry_count > 0 else ""
    comment = f"""Agent 2 (Dev) completed code changes for {ticket_key}{retry_note}.

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
        "user_input": "Fix login bug where passwords are stored in plain text",
        "jira_ticket_details": {"ticket_key": "SDLC-4", "url": "https://sakshitokekar.atlassian.net/browse/SDLC-4"},
        "code": "",
        "test_results": {},
        "dev_test_retry_count": 0,
        "human_decision": ""
    }
    result = run_dev_agent_node(test_state)
    print(result)