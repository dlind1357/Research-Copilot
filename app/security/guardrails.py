"""Security guardrails to defend against prompt injection, jailbreaks, and untrusted payloads."""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)

# Compilation of security patterns (compiled once for efficiency)
PROMPT_INJECTION_RE = re.compile(
    r"(ignore\s+(?:previous|above|the)?\s*instructions|"
    r"override\s+(?:the)?\s*instructions|"
    r"forget\s+(?:everything|what\s+i\s+said|the\s+rules)|"
    r"disregard\s+(?:the)?\s*instructions|"
    r"bypass\s+restrictions)",
    re.IGNORECASE
)

JAILBREAK_RE = re.compile(
    r"(dan\s+mode|do\s+anything\s+now|"
    r"developer\s+mode\s+v\d+|"
    r"jailbreak|"
    r"hypothetical\s+scenario\s+where\s+you|"
    r"acting\s+as\s+a\s+malicious|"
    r"bypass\s+(?:your\s+)?safeguards)",
    re.IGNORECASE
)

SYSTEM_PROMPT_RE = re.compile(
    r"(reveal\s+(?:your\s+)?system\s+prompt|"
    r"reveal\s+(?:your\s+)?instructions|"
    r"disclose\s+(?:your\s+)?system\s+prompt|"
    r"show\s+(?:your\s+)?system\s+prompt|"
    r"print\s+(?:your\s+)?system\s+prompt|"
    r"what\s+is\s+your\s+system\s+prompt|"
    r"output\s+the\s+initial\s+system\s+prompt)",
    re.IGNORECASE
)


def detect_prompt_injection(text: str) -> bool:
    """Detect prompt injection attempts in the text."""
    if not text:
        return False
    is_injection = bool(PROMPT_INJECTION_RE.search(text) or SYSTEM_PROMPT_RE.search(text))
    if is_injection:
        logger.warning(f"SECURITY ALERT: Prompt injection attempt detected: {text!r}")
    return is_injection


def detect_jailbreak_attempt(text: str) -> bool:
    """Detect jailbreak attempts (e.g. DAN mode, roleplay exploits) in the text."""
    if not text:
        return False
    is_jailbreak = bool(JAILBREAK_RE.search(text))
    if is_jailbreak:
        logger.warning(f"SECURITY ALERT: Jailbreak attempt detected: {text!r}")
    return is_jailbreak


def sanitize_input(text: str) -> str:
    """Sanitize input by removing control characters, stripping dangerous tags, and normalizing whitespace."""
    if not text:
        return ""

    # Normalize unicode (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # Remove non-printable control characters, keeping safe whitespace like \n, \r, \t
    text = "".join(ch for ch in text if ch == '\n' or ch == '\r' or ch == '\t' or not unicodedata.category(ch).startswith("C"))

    # Remove common script tags/HTML injections to prevent cross-site issues
    text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]*>", "", text)  # Strip general HTML tags

    # Normalize multiple whitespaces/newlines to avoid bypassing pattern matchers
    text = re.sub(r"[ \t]+", " ", text)
    
    return text.strip()


def filter_untrusted_document(text: str) -> str:
    """Filters untrusted documents or tool outputs by redacting or sanitizing prompt injection strings.
    
    This prevents indirect prompt injection when the model consumes uploaded papers or API results.
    """
    if not text:
        return ""

    # Run primary sanitization
    sanitized = sanitize_input(text)

    # Redact prompt injection patterns
    sanitized = PROMPT_INJECTION_RE.sub("[REDACTED PROMPT INJECTION HAZARD]", sanitized)
    sanitized = JAILBREAK_RE.sub("[REDACTED JAILBREAK HAZARD]", sanitized)
    sanitized = SYSTEM_PROMPT_RE.sub("[REDACTED SYSTEM PROMPT DISCLOSURE HAZARD]", sanitized)

    return sanitized
