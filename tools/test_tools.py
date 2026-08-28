# Test tools — running pytest and generating new test coverage via LLM
import subprocess
import json
import re
from google import genai
from google.genai import types
from config.settings import gem_api_key, GEMINI_MODEL
from tools.gemini_utils import generate_with_retry

client = genai.Client(api_key=gem_api_key)


#     run_pytest(repo_path) -> dict:
#         - runs pytest against the tests/ folder in repo_path
#         - returns structured result: {"success": bool, "output": str, "reason_code": str}
#         - reason_code: TESTS_PASSED or TESTS_FAILED
def run_pytest(repo_path: str) -> dict:
    result = subprocess.run(
        ["python3", "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=repo_path,
        capture_output=True,
        text=True
    )
    output = result.stdout + result.stderr
    passed = result.returncode == 0

    summary_lines = [line for line in output.split("\n") if "passed" in line or "failed" in line or "error" in line.lower()]
    summary = summary_lines[-1] if summary_lines else ("All tests passed" if passed else "Tests failed")

    return {
        "success": passed,
        "output": output,
        "summary": summary,
        "reason_code": "TESTS_PASSED" if passed else "TESTS_FAILED"
    }


#     get_existing_test_code(repo_path, test_file) -> str:
#         - reads the existing test file content
#         - returns empty string if file not found (fault-tolerant, never crashes)
def get_existing_test_code(repo_path: str, test_file: str = "tests/test_auth.py") -> str:
    try:
        with open(f"{repo_path}/{test_file}", "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""


#     generate_test_coverage(ticket_key, code_changes, existing_tests) -> dict:
#         - calls Gemini (with classified retry via generate_with_retry) to
#           determine if existing tests cover the code changes
#         - if not, generates new pytest test functions matching existing style
#         - returns {"needs_new_tests": bool, "new_test_code": str, "reasoning": str, "reason_code": str}
#         - schema validated via json.loads with fallback on parse failure
def generate_test_coverage(ticket_key: str, code_changes: str, existing_tests: str) -> dict:
    prompt = f"""You are a QA engineer reviewing code changes made for Jira ticket {ticket_key}.

=== CODE CHANGES MADE BY DEV AGENT ===
{code_changes}

=== EXISTING TEST FILE ===
```python
{existing_tests}
```

=== YOUR TASK ===
1. Determine if the existing tests already adequately cover the specific behavior that was changed.
2. If existing tests ALREADY cover it, respond with needs_new_tests: false.
3. If NOT covered, write 1-3 new pytest test functions (following the same style as existing tests: using register_user, login_user, users_db.clear() in setup_function) that specifically test the new behavior.

Respond in this exact JSON format, no markdown fences:
{{
  "needs_new_tests": true or false,
  "reasoning": "brief explanation of your decision",
  "new_test_code": "the new test function(s) as a string, properly indented, ready to append to the file. Empty string if needs_new_tests is false."
}}
"""
    config = types.GenerateContentConfig(
        system_instruction="You are a QA engineer. Always respond with valid JSON, no markdown code fences."
    )
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]

    response = generate_with_retry(client, GEMINI_MODEL, contents, config)
    if response is None:
        return {
            "needs_new_tests": False,
            "reasoning": "Gemini API unavailable after retries",
            "new_test_code": "",
            "reason_code": "COVERAGE_ANALYSIS_API_ERROR"
        }

    try:
        cleaned = re.sub(r"^```json\s*|\s*```$", "", response.text.strip(), flags=re.MULTILINE)
        result = json.loads(cleaned)
        result["reason_code"] = "COVERAGE_ANALYSIS_SUCCESS"
        return result
    except json.JSONDecodeError:
        return {
            "needs_new_tests": False,
            "reasoning": "Failed to parse QA response as JSON",
            "new_test_code": "",
            "reason_code": "COVERAGE_ANALYSIS_PARSE_ERROR"
        }


#     append_and_commit_tests(repo_path, ticket_key, new_test_code, test_file) -> dict:
#         - appends new test code to the test file
#         - commits and pushes the change to GitHub
#         - returns {"success": bool, "reason_code": str}
def append_and_commit_tests(repo_path: str, ticket_key: str, new_test_code: str, test_file: str = "tests/test_auth.py") -> dict:
    try:
        with open(f"{repo_path}/{test_file}", "a") as f:
            f.write("\n\n" + new_test_code)
    except Exception as e:
        return {"success": False, "error": str(e), "reason_code": "FILE_WRITE_ERROR"}

    try:
        subprocess.run(["git", "add", test_file], cwd=repo_path, check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "commit", "-m", f"test({ticket_key}): add test coverage for recent changes"],
            cwd=repo_path, check=True, capture_output=True, text=True
        )
        subprocess.run(["git", "push"], cwd=repo_path, check=True, capture_output=True, text=True)
        return {"success": True, "reason_code": "TESTS_COMMITTED"}
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": e.stderr, "reason_code": "GIT_ERROR", "recoverable": True}