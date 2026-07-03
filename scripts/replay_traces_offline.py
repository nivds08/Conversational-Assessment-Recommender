"""Offline trace replay using rule-based intent (no Groq) for diagnostic transcripts."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agent_logic import (  # noqa: E402
    handle_clarify_needed,
    handle_compare,
    handle_recommend,
    handle_refine,
    handle_refuse,
)
from intent_classifier import ClassificationResult, Constraints  # noqa: E402
from test_traces import TRACES_DIR, _parse_md_trace, _recall_at_10  # noqa: E402


def _had_recommendations(messages: list[dict]) -> bool:
    return any(m["role"] == "assistant" and "shortlist" in m["content"].lower() for m in messages)


def _rule_classify(messages: list[dict]) -> ClassificationResult:
    user = messages[-1]["content"]
    low = user.lower()

    if any(x in low for x in ("ignore previous", "system prompt", "best programming language")):
        return ClassificationResult(intent="REFUSE", constraints=Constraints(), compare_assessments=[], rationale="")

    compare_markers = ("difference between", " vs ", " versus ", "compare ")
    if any(m in low for m in compare_markers):
        names: list[str] = []
        if "opq" in low and "sales report" in low:
            names = ["OPQ", "OPQ MQ Sales Report"]
        elif "dsi" in low and "safety" in low:
            names = ["DSI", "Safety & Dependability 8.0"]
        elif "contact center call simulation" in low and "customer service phone" in low:
            names = ["Contact Center Call Simulation", "Customer Service Phone Simulation"]
        return ClassificationResult(
            intent="COMPARE",
            constraints=Constraints(),
            compare_assessments=names,
            rationale="compare request",
        )

    if _had_recommendations(messages) or any(
        k in low for k in ("drop ", "remove ", "add ", "final list", "confirmed", "that's good", "perfect")
    ):
        return ClassificationResult(
            intent="REFINE",
            constraints=Constraints(role=_guess_role(low), skills=_guess_skills(low)),
            compare_assessments=[],
            rationale="refine",
        )

    if len(low.split()) < 6 and not any(k in low for k in ("java", "sales", "excel", "engineer", "graduate", "contact")):
        return ClassificationResult(intent="CLARIFY_NEEDED", constraints=Constraints(role=_guess_role(low)), compare_assessments=[], rationale="")

    return ClassificationResult(
        intent="RECOMMEND",
        constraints=Constraints(role=_guess_role(low), skills=_guess_skills(low), seniority=_guess_seniority(low)),
        compare_assessments=[],
        rationale="recommend",
    )


def _guess_role(text: str) -> str | None:
    for k in ("sales", "java", "engineer", "admin assistant", "contact centre", "contact center", "graduate", "plant operator", "leadership", "cxo"):
        if k in text:
            return k
    return None


def _guess_skills(text: str) -> list[str]:
    skills = []
    for k in ("java", "spring", "sql", "excel", "word", "rust", "aws", "docker", "safety", "personality", "cognitive"):
        if k in text:
            skills.append(k)
    return skills


def _guess_seniority(text: str) -> str | None:
    if "senior" in text:
        return "senior"
    if "entry" in text or "graduate" in text:
        return "entry-level"
    if "mid" in text:
        return "mid-level"
    return None


def _dispatch(cls: ClassificationResult, messages: list[dict]):
    if cls.intent == "CLARIFY_NEEDED":
        return handle_clarify_needed(cls, messages)
    if cls.intent == "RECOMMEND":
        return handle_recommend(cls, messages)
    if cls.intent == "REFINE":
        return handle_refine(cls, messages)
    if cls.intent == "COMPARE":
        return handle_compare(cls, messages)
    return handle_refuse(cls, messages)


def replay_offline(path: Path) -> dict:
    user_turns, expected = _parse_md_trace(path)
    messages: list[dict[str, str]] = []
    transcript = []
    final_recs = []

    for i, user_msg in enumerate(user_turns, 1):
        messages.append({"role": "user", "content": user_msg})
        cls = _rule_classify(messages)
        resp = _dispatch(cls, messages)
        final_recs = [r.model_dump() for r in resp.recommendations]
        transcript.append(
            {
                "turn": i,
                "user": user_msg,
                "intent": cls.intent,
                "reply": resp.reply,
                "recommendations": [r["name"] for r in final_recs],
                "end_of_conversation": resp.end_of_conversation,
            }
        )
        messages.append({"role": "assistant", "content": resp.reply})
        if resp.end_of_conversation:
            break

    got = [r["name"] for r in final_recs]
    return {
        "trace": path.name,
        "expected": expected,
        "got": got,
        "transcript": transcript,
        "recall_exact": _recall_at_10(expected, got),
        "recall_fuzzy": _recall_at_10(expected, got, fuzzy=True),
    }


def main() -> None:
    results = [replay_offline(p) for p in sorted(TRACES_DIR.glob("*.md"))]
    print(f"{'Trace':<10} {'exact':>8} {'fuzzy':>8}")
    for r in results:
        print(f"{r['trace']:<10} {r['recall_exact']:>8.3f} {r['recall_fuzzy']:>8.3f}")
    ex = sum(r["recall_exact"] for r in results) / len(results)
    fu = sum(r["recall_fuzzy"] for r in results) / len(results)
    print(f"{'MEAN':<10} {ex:>8.3f} {fu:>8.3f}  delta={fu-ex:+.3f}")

    weak = [r for r in results if r["recall_fuzzy"] < 0.3]
    for r in weak:
        print("\n" + "=" * 70)
        print(r["trace"], f"exact={r['recall_exact']:.3f} fuzzy={r['recall_fuzzy']:.3f}")
        print("EXPECTED:", r["expected"])
        print("GOT:", r["got"])
        for t in r["transcript"]:
            print(f"\nTurn {t['turn']} | intent={t['intent']} | eoc={t['end_of_conversation']}")
            print("USER:", t["user"][:200])
            print("RECS:", t["recommendations"] or "(none)")


if __name__ == "__main__":
    main()
