# WHO uses it?
#       Called by agent2_dev.py. It sits between retriever.py (which finds relevant code) and Gemini (which writes the fix).
# WHAT is it?
#       A prompt builder. It takes raw data — the Jira ticket and the retrieved code chunks — and formats them into a single, 
#       well-structured text prompt that Gemini can understand and act on.
# WHEN does it run?
#       After retriever.retrieve() returns results, and before the Gemini API call.
#       Jira ticket → Retriever.retrieve() → Augmentor.build_prompt() → Gemini
# WHERE does it get its data from?
#       Two sources:
#           - jira_ticket_details from StateSDLC — contains ticket_key, url
#           - The dict returned by retriever.retrieve() — contains seed_symbols, expanded_symbols, dependencies, dependents, chunks
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

# OUTPUT:
#         You are a senior software engineer...

#         === JIRA TICKET ===
#         Ticket ID: SDLC-4
#         Query: fix login bug

#         === PRIMARY TARGETS ===
#         - services.auth_service.login_user

#         === RELEVANT CODE CHUNKS ===
#         Symbol: services.auth_service.login_user
#         File: services/auth_service.py (lines 45-67)
#         ```python
#         def login_user(...): ...
#         ```

#         === INSTRUCTIONS ===
#         1. Analyse the ticket and code...
#         2. Return JSON in this format: {...}


from codebase_search_rag.models import Chunk


class Augmentor:
#     build_prompt(jira_ticket, retrieval_result) -> str:
#         - takes jira ticket dict from StateSDLC and retrieval result from retriever.retrieve()
#         - builds a structured prompt for Agent 2 (Dev Agent)
#         - includes: problem statement, primary targets, relevant code chunks, constraints
#         - returns formatted prompt string ready to send to Gemini
    def build_prompt(self, jira_ticket: dict, retrieval_result: dict) -> str:
        ticket_key = jira_ticket.get("ticket_key", "UNKNOWN")
        ticket_url = jira_ticket.get("url", "")

        # Format primary target symbols
        seed_symbols = retrieval_result.get("seed_symbols", [])
        expanded_symbols = retrieval_result.get("expanded_symbols", [])
        dependencies = retrieval_result.get("dependencies", [])
        dependents = retrieval_result.get("dependents", [])
        related_files = retrieval_result.get("related_files", [])
        chunks = retrieval_result.get("chunks", [])
        query = retrieval_result.get("query", "")

        # Format code chunks
        # Current chunk_text looks like:
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

        prompt = f"""   You are a senior software engineer working on a Python codebase.
                        Your task is to fix or implement the change described in the Jira ticket below.

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
                        """
        return prompt                       