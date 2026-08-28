# Dev Agent — uses Agentic RAG (semantic search + graph traversal) to find
# relevant code, then uses Gemini to generate a code fix, then commits to GitHub
import ast
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
from tools import log_utils as log

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


def _is_valid_python(source: str) -> tuple[bool, str]:
    """Parses source with Python's ast module to check it's syntactically
    valid. Returns (is_valid, error_message). This is the guardrail that
    prevents the class of bug where accumulated text-replacements across
    retries silently corrupt a file's indentation/structure — instead of
    writing broken code to disk, we catch it here and reject the change."""
    try:
        ast.parse(source)
        return True, ""
    except SyntaxError as e:
        return False, f"{e.msg} at line {e.lineno}"


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
            log.step(f"Retry attempt #{retry_count} — informing Agent 2 of previous test failure: {previous_failure['summary']}")

    # --- Step 1: Set up ChromaDB, Embedder, Indexer, Retriever, Augmentor ---
    collection = get_chroma_collection()
    embedder = Embedder()
    indexer = Indexer(REPO_PATH, embedder, collection)
    retriever = CodeRetriever(collection, embedder, indexer)
    augmentor = Augmentor()

    # --- Step 2: Index the repository (incremental — only changed files re-indexed) ---
    log.step(f"Indexing repository at {REPO_PATH}...")
    indexer.index_repository()

    # --- Step 3: Build the search query from the Jira ticket ---
    query = state.get("user_input", "")

    # --- Step 4: Agentic RAG retrieval — semantic search + graph expansion ---
    log.step(f"Retrieving relevant code for: {query[:80]}...")
    retrieval_result = retriever.retrieve(query, top_k=10)
    log.success(f"Found {len(retrieval_result['chunks'])} relevant chunks")

    # --- Step 5: Build the prompt using Augmentor, including retry context if applicable ---
    prompt = augmentor.build_prompt(jira_ticket, retrieval_result, previous_failure=previous_failure)

    # --- Step 6: Call Gemini to generate the code fix (with classified retry) ---
    config = types.GenerateContentConfig(
        system_instruction="You are a senior Python developer. Always respond with valid JSON matching the requested schema. Never include markdown code fences around the JSON itself. In file_path fields, always use paths RELATIVE TO THE REPOSITORY ROOT (e.g. 'models/user.py'), never include the repo folder name itself."
    )
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    response = generate_with_retry(client, GEMINI_MODEL, contents, config)
    if response is None:
        log.error("Gemini API unavailable — cannot generate code fix.")
        return {"code": ""}

    raw_text = response.text.strip()
    log.success("Gemini response received.")

    # --- Step 7: Parse Gemini's JSON response ---
    cleaned = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
    try:
        change_plan = json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse Gemini response as JSON: {e}")
        log.full(f"Raw unparseable Gemini response for {ticket_key}:\n{raw_text}")
        print(log.truncate(raw_text))
        return {"code": ""}

    # --- Step 8: Apply code changes to files, validating each with ast.parse ---
    # Every individual replacement is validated BEFORE being kept — if it
    # would produce invalid Python, it's rejected and logged clearly rather
    # than silently written to disk. The full file is validated again at
    # the end as defense in depth.
    actual_timestamp = datetime.now().isoformat()
    modified_files = []
    rejected_changes = []

    for file_change in change_plan.get("files_to_modify", []):
        raw_file_path = file_change["file_path"]
        relative_path = _normalize_relative_path(raw_file_path, REPO_PATH)
        full_path = f"{REPO_PATH}/{relative_path}"

        try:
            with open(full_path, "r") as f:
                file_content = f.read()
        except FileNotFoundError:
            log.warn(f"File not found: {full_path}, skipping.")
            continue

        original_file_content = file_content
        applied_count = 0

        for change in file_change.get("changes", []):
            original = change.get("original_code", "")
            modified = change.get("modified_code", "").replace("{TIMESTAMP}", actual_timestamp)
            func_name = change.get("function_name", "unknown")

            if not (original and original in file_content):
                log.warn(f"original_code not found exactly in {relative_path} ({func_name}), skipping this change.")
                continue

            candidate_content = file_content.replace(original, modified)
            is_valid, error_msg = _is_valid_python(candidate_content)

            if not is_valid:
                log.error(f"Rejected change to {relative_path} ({func_name}): would produce invalid Python — {error_msg}")
                rejected_changes.append({"file": relative_path, "function": func_name, "reason": error_msg})
                continue

            file_content = candidate_content
            applied_count += 1

        # Only write the file if at least one change actually applied AND
        # the final accumulated content is still valid (defense in depth —
        # catches interactions between multiple changes to the same file)
        if applied_count > 0:
            final_valid, final_error = _is_valid_python(file_content)
            if not final_valid:
                log.error(f"Final content for {relative_path} is invalid ({final_error}) — reverting file entirely, no changes written.")
                rejected_changes.append({"file": relative_path, "function": "(whole file)", "reason": final_error})
                continue

            with open(full_path, "w") as f:
                f.write(file_content)
            modified_files.append(relative_path)
            log.success(f"Modified: {relative_path} ({applied_count} change(s) applied)")
        elif file_content != original_file_content:
            # Shouldn't happen given the logic above, but guard against it
            log.warn(f"No valid changes applied to {relative_path}; leaving file untouched.")

    # --- Step 9: Commit and push to GitHub ---
    commit_message = change_plan.get("commit_message", f"fix({ticket_key}): automated change")
    if retry_count > 0:
        commit_message = f"{commit_message} (retry #{retry_count})"

    commit_result = {"success": False, "error": "No files to commit"}
    if modified_files:
        commit_result = commit_and_push_changes(
            repo_path=REPO_PATH,
            files=modified_files,
            commit_message=commit_message,
            branch=f"agent2/{ticket_key.lower()}"
        )
    log.step(f"Commit result: {commit_result.get('reason_code', commit_result.get('success'))}")

    # --- Step 10: Update Jira with audit trail, including any rejected changes ---
    retry_note = f" (retry attempt #{retry_count})" if retry_count > 0 else ""
    rejected_section = ""
    if rejected_changes:
        rejected_lines = "\n".join(
            f"  - {r['file']} ({r['function']}): {r['reason']}" for r in rejected_changes
        )
        rejected_section = f"\nREJECTED CHANGES (would have produced invalid Python):\n{rejected_lines}\n"

    comment = f"""Agent 2 (Dev) completed code changes for {ticket_key}{retry_note}.

WHO: Agent 2 (Dev Agent)
WHAT: {change_plan.get('analysis', 'Code changes applied')}
WHY: Addresses requirements in {ticket_key}
WHERE: {', '.join(modified_files) if modified_files else 'No files modified'}
COMMIT: {commit_result.get('commit_url', 'N/A')}
{rejected_section}"""
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
        "human_decision": "",
        "build_results": {}
    }
    result = run_dev_agent_node(test_state)
    print(result)