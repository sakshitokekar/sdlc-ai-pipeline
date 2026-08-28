# WHO uses it?
#       Called by agent2_dev.py. It sits between retriever.py (which finds relevant code) and Gemini (which writes the fix).
# WHAT is it?
#       A prompt builder. It takes raw data — the Jira ticket and the retrieved code chunks — and formats them into a single,
#       well-structured text prompt that Gemini can understand and act on.
# WHEN does it run?
#       After retriever.retrieve() returns results, and before the Gemini API call.
#       Jira ticket → Retriever.retrieve() → Augmentor.build_prompt() → Gemini
# WHERE does it get its data from?
#       Three sources:
#           - jira_ticket_details from StateSDLC — contains ticket_key, url
#           - The dict returned by retriever.retrieve() — contains seed_symbols, expanded_symbols, dependencies, dependents, chunks
#           - Optional previous_failure dict — passed by agent2_dev.py only on RETRY attempts, contains
#             test failure details so Agent 2 fixes the actual problem instead of blindly repeating itself
# WHY do we need it?
#       Without augmentor.py, you'd have to manually format this messy data every time you call Gemini — repeated code,
#       inconsistent formatting, easy to make mistakes.
#       With augmentor.py, you have ONE place that defines exactly how context gets presented to the LLM.
#       If you want to improve prompt quality later, you only change this one file.
# Think of it as the "translator" between structured data and natural language the LLM can act on.

# INPUT:
#         jira_ticket = {"ticket_key": "SDLC-4", "url": "https://..."}
#         retrieval_result = {
#             "query": "fix login bug",
#             "seed_symbols": ["services.auth_service.login_user"],
#             "expanded_symbols": ["models.user.User.check_password"],
#             "chunks": [{"content": "def login_user(...): ...", "metadata": {...}, "score": 1.0}]
#         }
#         previous_failure (optional, only present on retries) = {
#             "retry_count": 1,
#             "summary": "1 passed, 2 failed in 0.15s",
#             "full_output": "...pytest traceback..."
#         }

# OUTPUT:
#         You are a senior software engineer...
#
#         === THIS IS RETRY ATTEMPT #1 ===          <- only present on retries
#         ...failure details...
#
#         === JIRA TICKET ===
#         Ticket ID: SDLC-4
#         Query: fix login bug
#
#         === PRIMARY TARGETS ===
#         - services.auth_service.login_user
#
#         === RELEVANT CODE CHUNKS ===
#         Symbol: services.auth_service.login_user
#         File: services/auth_service.py (lines 45-67)
#         ```python
#         def login_user(...): ...
#         ```
#
#         === INSTRUCTIONS ===
#         1. Analyse the ticket and code...
#         2. Return JSON in this format: {...}


from codebase_search_rag.models import Chunk


class Augmentor:
#     build_prompt(jira_ticket, retrieval_result, previous_failure=None) -> str:
#         - takes jira ticket dict from StateSDLC and retrieval result from retriever.retrieve()
#         - previous_failure: optional dict with test failure info, only passed on retry attempts
#         - full_file_contents: optional dict {relative_path: full source text} for files
#           that are candidates for modification. RAG only chunks function/class-level
#           code (see indexer.py's parse_and_chunk) — module-level code like imports,
#           top-level constants, and `if __name__ == "__main__":` blocks is NEVER
#           retrieved as a chunk. Without the full file, Gemini has to GUESS the exact
#           bytes of original_code for anything outside a function/class, which fails
#           the exact-match replacement in agent2_dev.py. Passing full files for
#           candidate targets guarantees byte-exact ground truth to copy from.
#         - builds a structured prompt for Agent 2 (Dev Agent)
#         - includes: retry context (if applicable), problem statement, primary targets, relevant code chunks, full file contents, constraints
#         - returns formatted prompt string ready to send to Gemini
    def build_prompt(self, jira_ticket: dict, retrieval_result: dict, previous_failure: dict = None, full_file_contents: dict = None) -> str:
        ticket_key = jira_ticket.get("ticket_key", "UNKNOWN")
        ticket_url = jira_ticket.get("url", "")

        # Format primary target symbols
        seed_symbols = retrieval_result.get("seed_symbols", [])
        expanded_symbols = retrieval_result.get("expanded_symbols", [])
        dependencies = retrieval_result.get("dependencies", [])
        dependents = retrieval_result.get("dependents", [])
        chunks = retrieval_result.get("chunks", [])
        query = retrieval_result.get("query", "")

        # Format code chunks
        # Current chunk looks like:
        # {
        #     "id": "abc123...",
        #     "content": "def login_user(email, password):\n    ...",
        #     "metadata": {
        #         "symbol_id": "services.auth_service.login_user",
        #         "file_path": "services/auth_service.py",
        #         "start_line": 45,
        #         "end_line": 67,
        #         "function_name": "login_user",
        #         "chunk_type": "function_definition"
        #     },
        #     "score": 1.0,
        #     "source": "semantic"
        # }
        #
        # For each chunk, it will additionally add to above
        # Symbol: services.auth_service.login_user
        # File: services/auth_service.py (lines 45-67)
        # Source: semantic | Relevance score: 1.00
        # ```python
        # def login_user(email, password):
        #     ...
        # ```
        chunks_text = ""
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            file_path = meta.get("file_path", "unknown")
            start_line = meta.get("start_line", "?")
            end_line = meta.get("end_line", "?")
            symbol_id = meta.get("symbol_id", "unknown")
            source = chunk.get("source", "unknown")
            score = chunk.get("score", 0)
            content = chunk.get("content", "")
            chunks_text += f"""
---
Symbol: {symbol_id}
File: {file_path} (lines {start_line}-{end_line})
Source: {source} | Relevance score: {score:.2f}
```python
{content}
```
"""

        # Full file contents for candidate target files — this is the fix
        # for exact-match replacement failures on module-level code (like
        # `if __name__ == "__main__":` blocks) that RAG chunking never
        # retrieves, since indexer.py only chunks function_definition and
        # class_definition nodes, not top-level statements.
        full_files_text = ""
        if full_file_contents:
            for file_path, file_text in full_file_contents.items():
                full_files_text += f"""
---
FULL FILE: {file_path}
```python
{file_text}
```
"""

        # Retry context — only present when agent2_dev.py detects this is a
        # retry after a test failure (dev_test_retry_count > 0). Without this,
        # Agent 2 would re-run with identical inputs and likely produce the
        # same flawed fix again.
        retry_section = ""
        if previous_failure:
            retry_count = previous_failure.get("retry_count", 1)
            test_summary = previous_failure.get("summary", "Unknown failure")
            test_output = previous_failure.get("full_output", "No output captured")
            # Truncate very long pytest output — most relevant failure info
            # (assertions, tracebacks) is at the end, so keep the tail.
            truncated_output = test_output[-3000:] if len(test_output) > 3000 else test_output

            retry_section = f"""
=== THIS IS RETRY ATTEMPT #{retry_count} ===
Your previous code change did NOT pass the test suite. You must analyse
the failure below and produce a DIFFERENT, CORRECTED fix — do not repeat
the same change that just failed.

TEST FAILURE SUMMARY: {test_summary}

TEST FAILURE OUTPUT (most recent portion):
```
{truncated_output}
```

Carefully read the failure above. Identify exactly which assertion or
error caused the failure, and make sure your new fix addresses that
specific problem.
"""

        # NOTE: this f-string is written flush-left (no leading indentation
        # on each line) even though it's inside an indented function body.
        # This is deliberate — if the string were indented to match the
        # surrounding code, every line sent to Gemini would carry that
        # leading whitespace, wasting tokens and adding noise to the prompt.
        prompt = f"""You are a senior software engineer working on a Python codebase.
Your task is to fix or implement the change described in the Jira ticket below.
{retry_section}
=== JIRA TICKET ===
Ticket ID: {ticket_key}
URL: {ticket_url}
Query: {query}

=== PRIMARY TARGETS (most likely files/functions to change) ===
{chr(10).join(f"- {s}" for s in seed_symbols) if seed_symbols else "None identified"}

=== RELATED SYMBOLS (context — may or may not need changes) ===
{chr(10).join(f"- {s}" for s in expanded_symbols) if expanded_symbols else "None"}

=== FILE DEPENDENCIES ===
Files this code imports from:
{chr(10).join(f"- {d}" for d in dependencies) if dependencies else "None"}

Files that import this code (changing these could break dependents):
{chr(10).join(f"- {d}" for d in dependents) if dependents else "None"}

=== RELEVANT CODE CHUNKS ===
{chunks_text if chunks_text else "No relevant chunks found."}

=== FULL FILE CONTENTS (use these for byte-exact original_code — RAG chunks above only cover function/class-level code, NOT module-level code like imports or `if __name__ == "__main__":` blocks) ===
{full_files_text if full_files_text else "No full files provided."}

=== INSTRUCTIONS ===
1. Analyse the Jira ticket and the code chunks above carefully.
2. Identify the exact file(s) and function(s) that need to change.
3. Only modify what is necessary — do not refactor unrelated code.
4. For every change you make, add a comment with:
   - WHO: Agent 2 (Dev Agent)
   - WHAT: what was changed
   - WHY: why this change was needed (reference ticket {ticket_key})
   - WHEN: timestamp placeholder {{TIMESTAMP}}
   - WHERE: file path and function name
5. Update the changelog at the top of each modified file if it exists, else create at the top.
6. Return your response in the following JSON format:

{{
  "analysis": "brief explanation of what needs to change and why",
  "files_to_modify": [
    {{
      "file_path": "path/to/file.py",
      "changes": [
        {{
          "function_name": "name of function to modify",
          "original_code": "the exact original code snippet",
          "modified_code": "the complete modified code snippet with 5W comments",
          "reason": "why this change was made"
        }}
      ]
    }}
  ],
  "commit_message": "fix({ticket_key}): brief description of what was fixed",
  "files_not_to_modify": ["list of files reviewed but left unchanged and why"]
}}

IMPORTANT:
- Never modify files outside the scope of this ticket.
- Never delete existing functionality unless explicitly required.
- Never commit to main — changes go to the feature branch only.
- If you are unsure about a change, flag it in the analysis field instead of making it.
- In file_path fields, always use paths RELATIVE TO THE REPOSITORY ROOT (e.g. 'models/user.py'), never include the repo folder name itself.
"""
        return prompt