# Shared Gemini API helper — used by all agents to call Gemini with automatic
# retry on transient errors (503 high demand, timeouts etc).
# Single source of truth for retry behavior — avoids duplicating retry logic
# in agent1_pm.py, agent2_dev.py, and test_tools.py separately.
import time


#     generate_with_retry(client, model, contents, config, max_retries=3) -> response or None:
#         - calls client.models.generate_content with automatic retry on failure
#         - uses exponential backoff: 5s, 10s, 15s between attempts
#         - returns None if all retries exhausted (caller must handle this)
#         - reason for retry (503, timeout etc) is printed but not distinguished —
#           all API errors are treated as potentially transient and retried
def generate_with_retry(client, model: str, contents: list, config, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"Gemini API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Gemini API failed after {max_retries} attempts: {e}")
                return None