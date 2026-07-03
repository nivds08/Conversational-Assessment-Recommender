"""Construct embedding text from catalog entries for retrieval."""

from __future__ import annotations


def _remote_phrase(remote: str) -> str | None:
    if remote.lower() == "yes":
        return "remote testing available"
    if remote.lower() == "no":
        return "on-site or proctored only"
    return None


def _adaptive_phrase(adaptive: str) -> str | None:
    if adaptive.lower() == "yes":
        return "adaptive/IRT supported"
    if adaptive.lower() == "no":
        return "fixed-form (non-adaptive)"
    return None


def build_embedding_text(entry: dict) -> str:
    """
    Single text blob per catalog entry for embedding.

    Combines name, description, test types, job levels, and delivery flags.
    """
    parts: list[str] = [
        entry["name"],
        entry["description"],
        f"Test types: {', '.join(entry.get('test_type') or [])}",
    ]

    job_levels = entry.get("job_levels") or []
    if job_levels:
        parts.append(f"Job levels: {', '.join(job_levels)}")

    remote = _remote_phrase(entry.get("remote", ""))
    if remote:
        parts.append(remote)

    adaptive = _adaptive_phrase(entry.get("adaptive", ""))
    if adaptive:
        parts.append(adaptive)

    if entry.get("is_interpretive_report"):
        parts.append("interpretive report product (requires prior assessment results)")

    return ". ".join(p.strip() for p in parts if p and p.strip())
