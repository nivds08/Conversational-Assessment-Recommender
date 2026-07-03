"""Measure classifier prompt token usage before/after model switch."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from intent_classifier import ChatMessage, _classifier_prompt  # noqa: E402
from system_prompt import SYSTEM_PROMPT  # noqa: E402

C7_TURN1 = (
    "We're hiring bilingual healthcare admin staff in South Texas — they handle "
    "patient records and need to be assessed in Spanish. HIPAA compliance is critical. "
    "What assessments work?"
)
C9_TURN1 = (
    'Here\'s the JD for an engineer we need to fill. Can you recommend an assessment battery?\n\n'
    '"Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, '
    "Angular, SQL/relational databases, AWS deployment, and Docker. Will own end-to-end "
    "microservice delivery, contribute to architectural decisions, and mentor mid-level "
    'engineers. Strong CI/CD and cloud-native experience required."'
)


def _estimate_tokens(text: str) -> int:
  # Groq/OpenAI-style rough estimate: ~4 chars per token for English prose.
  return max(1, len(text) // 4)


def main() -> None:
    msgs = [ChatMessage(role="user", content=C7_TURN1)]
    prompt = _classifier_prompt(msgs)
    system = SYSTEM_PROMPT
    total_chars = len(system) + len(prompt)
    print("=== Classifier token estimate (chars/4 heuristic) ===")
    print(f"system_prompt chars: {len(system)} (~{_estimate_tokens(system)} tokens)")
    print(f"classifier_user_prompt chars: {len(prompt)} (~{_estimate_tokens(prompt)} tokens)")
    print(f"combined input chars: {total_chars} (~{_estimate_tokens(system + prompt)} tokens)")
    print(f"typical JSON output estimate: ~150 tokens")
    print(f"per /chat call estimate: ~{_estimate_tokens(system + prompt) + 150} tokens")
    print()
    print("Grading run estimate (10 traces × 8 turns × 1 call):")
    per_call = _estimate_tokens(system + prompt) + 150
    print(f"  {per_call} × 80 = {per_call * 80:,} tokens/day")
    print()
    print("Groq free-tier TPD (published):")
    print("  llama-3.3-70b-versatile: 100,000 TPD")
    print("  llama-3.1-8b-instant:     500,000 TPD")
    print()
    print("Ambiguity prompt section (excerpt):")
    start = prompt.find("Ambiguity rule")
    print(prompt[start : start + 420] if start >= 0 else "(not found)")


if __name__ == "__main__":
    main()
