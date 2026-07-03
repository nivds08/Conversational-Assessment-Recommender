"""Quick fallback behavior test without live Gemini calls."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent_logic  # noqa: E402


def _boom(messages):  # noqa: ANN001
    raise RuntimeError("synthetic 429-like failure")


def main() -> None:
    original = agent_logic.classify_turn
    agent_logic.classify_turn = _boom
    try:
        resp = agent_logic.generate_agent_response([{"role": "user", "content": "recommend tests"}])
        print("reply:", resp.reply)
        print("recommendations:", resp.recommendations)
        print("end_of_conversation:", resp.end_of_conversation)
    finally:
        agent_logic.classify_turn = original


if __name__ == "__main__":
    main()

