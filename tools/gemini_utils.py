# Shared Gemini API helper — used by all agents to call Gemini with
# CLASSIFIED retry behavior. Not every error is worth retrying:
#   - TRANSIENT (503 high demand, timeouts) -> retry with backoff, likely
#     to succeed on a later attempt
#   - QUOTA_EXCEEDED (429, daily/per-minute quota) -> NEVER retry — the
#     quota won't reset in the few seconds a backoff would wait, so
#     retrying only wastes time and (for per-minute quotas) could make
#     things worse
#   - FATAL (401/403 auth, 400 bad request) -> NEVER retry — the request
#     itself is wrong, retrying identical input produces identical failure
import time

try:
    from google.genai import errors as _genai_errors
    _CLIENT_ERROR = getattr(_genai_errors, "ClientError", None)
    _SERVER_ERROR = getattr(_genai_errors, "ServerError", None)
except ImportError:
    _CLIENT_ERROR = None
    _SERVER_ERROR = None


def _classify_error(e: Exception) -> str:
    """Returns 'QUOTA_EXCEEDED', 'TRANSIENT', or 'FATAL'. Uses both the
    exception's class (when the google-genai library exposes ClientError
    vs ServerError) and the message text (since quota errors are 4xx but
    need different handling than other 4xx errors like bad auth)."""
    message = str(e)

    if "RESOURCE_EXHAUSTED" in message or "quota" in message.lower():
        return "QUOTA_EXCEEDED"

    if _SERVER_ERROR is not None and isinstance(e, _SERVER_ERROR):
        return "TRANSIENT"
    if "503" in message or "UNAVAILABLE" in message or "timeout" in message.lower():
        return "TRANSIENT"

    if _CLIENT_ERROR is not None and isinstance(e, _CLIENT_ERROR):
        return "FATAL"

    return "FATAL"  # unknown errors default to non-retryable — safer than looping


def _short_error(e: Exception, max_chars: int = 200) -> str:
    """First line / first N chars of an exception's message — the full
    google-genai error objects include nested Help/QuotaFailure/RetryInfo
    dicts that are useful in logs but unreadable as a single terminal line."""
    text = str(e).split("\n")[0]
    return text if len(text) <= max_chars else text[:max_chars] + "..."


#     generate_with_retry(client, model, contents, config, max_retries=3) -> response or None:
#         - calls client.models.generate_content with CLASSIFIED retry behavior
#         - TRANSIENT errors retry with exponential backoff (5s, 10s, 15s)
#         - QUOTA_EXCEEDED and FATAL errors fail immediately, no retry
#         - returns None if the call could not succeed (caller must handle this)
def generate_with_retry(client, model: str, contents: list, config, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            classification = _classify_error(e)

            if classification == "QUOTA_EXCEEDED":
                print(f"Gemini quota exhausted for model '{model}' — not retrying (won't help until quota resets). {_short_error(e)}")
                return None

            if classification == "FATAL":
                print(f"Gemini API error is not retryable ({type(e).__name__}): {_short_error(e)}")
                return None

            # TRANSIENT — worth retrying with backoff
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"Gemini transient error (attempt {attempt + 1}/{max_retries}): {_short_error(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Gemini API failed after {max_retries} attempts (transient errors persisted): {_short_error(e)}")
                return None
    return None