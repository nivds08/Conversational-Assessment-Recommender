"""Diagnostic replay: matching logic audit, C5/C8 side-by-side, fuzzy rerun, weak-trace transcripts."""

from __future__ import annotations

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from intent_classifier import classify_turn  # noqa: E402
from test_traces import (  # noqa: E402
    CHAT_URL,
    MAX_TURNS,
    TRACES_DIR,
    _parse_md_trace,
    _recall_at_10,
)

FUZZY_THRESHOLD = 0.72


def _normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s&\.\-\(\)]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _names_match(a: str, b: str, threshold: float = FUZZY_THRESHOLD) -> tuple[bool, float]:
    target = _normalize_name(a)
    other = _normalize_name(b)
    if target == other:
        return True, 1.0
    if target in other or other in target:
        return True, 0.90
    score = SequenceMatcher(None, target, other).ratio()
    return score >= threshold, score


def _recall_at_10_fuzzy(expected: list[str], got: list[str]) -> float:
    if not expected:
        return 0.0
    matched = sum(1 for e in expected if any(_names_match(e, g)[0] for g in got[:10]))
    return matched / len(expected)


def _replay_trace(path: Path) -> dict:
    user_turns, expected = _parse_md_trace(path)
    messages: list[dict[str, str]] = []
    transcript: list[dict] = []
    final_recs: list[dict] = []
    turns = 0
    user_idx = 0

    with httpx.Client(timeout=30.0) as client:
        while user_idx < len(user_turns) and turns < MAX_TURNS:
            turns += 1
            user_msg = user_turns[user_idx]
            user_idx += 1
            messages.append({"role": "user", "content": user_msg})

            classification = classify_turn(messages)
            resp = client.post(CHAT_URL, json={"messages": messages})
            data = resp.json() if resp.status_code == 200 else {}

            final_recs = data.get("recommendations", [])
            transcript.append(
                {
                    "turn": turns,
                    "user": user_msg,
                    "intent": classification.intent,
                    "constraints": classification.constraints.model_dump(),
                    "reply": data.get("reply", ""),
                    "recommendations": [r.get("name", "") for r in final_recs],
                    "end_of_conversation": data.get("end_of_conversation", False),
                }
            )

            if data.get("end_of_conversation"):
                break
            messages.append({"role": "assistant", "content": data.get("reply", "")})

    got_names = [r.get("name", "") for r in final_recs]
    return {
        "path": path,
        "expected": expected,
        "got": got_names,
        "turns": turns,
        "transcript": transcript,
        "recall_exact": _recall_at_10(expected, got_names),
        "recall_fuzzy": _recall_at_10_fuzzy(expected, got_names),
    }


def _print_side_by_side(label: str, result: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"SIDE-BY-SIDE: {label}")
    print(f"{'=' * 70}")
    print(f"Turns played: {result['turns']}  |  exact recall: {result['recall_exact']:.3f}")
    print("\n(a) AGENT RECOMMENDED (verbatim):")
    for i, name in enumerate(result["got"], 1):
        print(f"  {i}. {name!r}")
    if not result["got"]:
        print("  (none)")
    print("\n(b) EXPECTED SHORTLIST (verbatim from trace):")
    for i, name in enumerate(result["expected"], 1):
        print(f"  {i}. {name!r}")
    print("\n(c) BEST FUZZY SCORE per expected item (threshold=0.72):")
    for exp in result["expected"]:
        best_got = ""
        best_score = 0.0
        for got in result["got"]:
            ok, score = _names_match(exp, got)
            if score > best_score:
                best_score = score
                best_got = got
        flag = "MATCH" if best_score >= FUZZY_THRESHOLD else "MISS"
        print(f"  [{flag}] {exp!r}")
        print(f"         -> best got: {best_got!r}  score={best_score:.3f}")


def _print_full_transcript(result: dict) -> None:
    print(f"\n{'#' * 70}")
    print(f"FULL TRANSCRIPT: {result['path'].name}")
    print(
        f"recall exact={result['recall_exact']:.3f}  "
        f"fuzzy={result['recall_fuzzy']:.3f}  turns={result['turns']}"
    )
    print(f"{'#' * 70}")
    for t in result["transcript"]:
        print(f"\n--- Turn {t['turn']} ---")
        print(f"USER: {t['user']}")
        print(f"INTENT: {t['intent']}")
        c = t["constraints"]
        revealed = {k: v for k, v in c.items() if v not in (None, [], "", False)}
        if revealed:
            print(f"CONSTRAINTS: {revealed}")
        print(f"ASSISTANT: {t['reply'][:500]}{'...' if len(t['reply']) > 500 else ''}")
        if t["recommendations"]:
            print(f"RECS ({len(t['recommendations'])}): {t['recommendations']}")
        print(f"end_of_conversation: {t['end_of_conversation']}")
    print("\nFINAL GOT:", result["got"])
    print("EXPECTED: ", result["expected"])


def main() -> None:
    print("=== MATCHING LOGIC IN test_traces.py ===")
    print(
        "_recall_at_10() uses CASE-INSENSITIVE EXACT string equality after .strip():\n"
        "  e = {x.lower().strip() for x in expected}\n"
        "  g = {x.lower().strip() for x in got[:10]}\n"
        "  recall = len(e & g) / len(e)\n"
        "No fuzzy matching, no normalization of punctuation/suffixes."
    )

    trace_files = sorted(TRACES_DIR.glob("*.md"))
    targets = [TRACES_DIR / "C5.md", TRACES_DIR / "C8.md"]
    c5c8 = {}
    for p in targets:
        print(f"\nReplaying {p.name}...")
        c5c8[p.name] = _replay_trace(p)
        _print_side_by_side(p.name, c5c8[p.name])

    print("\n\n=== RERUN ALL 10 TRACES: EXACT vs FUZZY RECALL ===")
    all_results = []
    for path in trace_files:
        print(f"Replaying {path.name}...")
        r = _replay_trace(path)
        all_results.append(r)

    print(f"\n{'Trace':<12} {'exact':>8} {'fuzzy':>8} {'turns':>6}")
    print("-" * 38)
    exact_mean = 0.0
    fuzzy_mean = 0.0
    for r in all_results:
        print(
            f"{r['path'].name:<12} {r['recall_exact']:>8.3f} "
            f"{r['recall_fuzzy']:>8.3f} {r['turns']:>6}"
        )
        exact_mean += r["recall_exact"]
        fuzzy_mean += r["recall_fuzzy"]
    n = len(all_results)
    exact_mean /= n
    fuzzy_mean /= n
    print("-" * 38)
    print(f"{'MEAN':<12} {exact_mean:>8.3f} {fuzzy_mean:>8.3f}")
    print(f"\nFuzzy matching delta on mean Recall@10: {fuzzy_mean - exact_mean:+.3f}")

    weak = [r for r in all_results if r["recall_fuzzy"] < 0.3]
    print(f"\n=== FULL TRANSCRIPTS FOR TRACES WITH FUZZY RECALL < 0.3 ({len(weak)}) ===")
    for r in weak:
        _print_full_transcript(r)


if __name__ == "__main__":
    main()
