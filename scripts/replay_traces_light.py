"""Lightweight replay: /chat only (no classifier), exact vs fuzzy recall, weak-trace detail."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from test_traces import (  # noqa: E402
    CHAT_URL,
    MAX_TURNS,
    TRACES_DIR,
    _parse_md_trace,
    _recall_at_10,
)


def _replay_chat_only(path: Path) -> dict:
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
            resp = client.post(CHAT_URL, json={"messages": messages})
            if resp.status_code != 200:
                return {"path": path, "error": f"http_{resp.status_code}"}
            data = resp.json()
            final_recs = data.get("recommendations", [])
            transcript.append(
                {
                    "turn": turns,
                    "user": user_msg,
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
        "recall_fuzzy": _recall_at_10(expected, got_names, fuzzy=True),
    }


def main() -> None:
    trace_files = sorted(TRACES_DIR.glob("*.md"))
    results = []
    for path in trace_files:
        print(f"Replaying {path.name}...", flush=True)
        r = _replay_chat_only(path)
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        results.append(r)

    print(f"\n{'Trace':<12} {'exact':>8} {'fuzzy':>8} {'turns':>6}")
    print("-" * 38)
    for r in results:
        print(
            f"{r['path'].name:<12} {r['recall_exact']:>8.3f} "
            f"{r['recall_fuzzy']:>8.3f} {r['turns']:>6}"
        )
    exact_mean = sum(r["recall_exact"] for r in results) / len(results)
    fuzzy_mean = sum(r["recall_fuzzy"] for r in results) / len(results)
    print("-" * 38)
    print(f"{'MEAN':<12} {exact_mean:>8.3f} {fuzzy_mean:>8.3f}")
    print(f"fuzzy delta: {fuzzy_mean - exact_mean:+.3f}")

    weak = [r for r in results if r["recall_fuzzy"] < 0.3]
    out = ROOT / "data" / "trace_diagnostic.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "exact_mean": exact_mean,
                "fuzzy_mean": fuzzy_mean,
                "results": [
                    {
                        "trace": r["path"].name,
                        "recall_exact": r["recall_exact"],
                        "recall_fuzzy": r["recall_fuzzy"],
                        "turns": r["turns"],
                        "expected": r["expected"],
                        "got": r["got"],
                        "transcript": r["transcript"],
                    }
                    for r in results
                ],
            },
            f,
            indent=2,
        )
    print(f"\nSaved full replay data to {out}")
    print(f"Weak traces (fuzzy < 0.3): {[r['path'].name for r in weak]}")


if __name__ == "__main__":
    main()
