"""Step 4: Intent-specific agent response logic."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Literal

from pydantic import BaseModel, Field

from intent_classifier import ClassificationResult, classify_turn
from retrieval import load_catalog, retrieve


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class AgentResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


SAFE_FALLBACK_RESPONSE = AgentResponse(
    reply="I'm having trouble processing that right now. Please try again in a moment.",
    recommendations=[],
    end_of_conversation=False,
)


def _constraints_to_query(classification: ClassificationResult) -> str:
    c = classification.constraints
    parts: list[str] = []
    if c.role:
        parts.append(f"Role: {c.role}")
    if c.seniority:
        parts.append(f"Seniority: {c.seniority}")
    if c.skills:
        parts.append(f"Skills: {', '.join(c.skills)}")
    if c.test_types:
        parts.append(f"Assessment types: {', '.join(c.test_types)}")
    if c.language:
        parts.append(f"Language: {c.language}")
    if c.max_duration_minutes is not None:
        parts.append(f"Maximum duration: {c.max_duration_minutes} minutes")
    if c.remote_required is True:
        parts.append("Remote testing required")
    elif c.remote_required is False:
        parts.append("On-site testing preferred")
    if c.adaptive_required is True:
        parts.append("Adaptive/IRT assessment required")
    elif c.adaptive_required is False:
        parts.append("Non-adaptive assessment required")
    return ". ".join(parts) if parts else "General SHL assessment recommendation"


def _normalize_bool_flag(value: str) -> bool | None:
    v = (value or "").strip().lower()
    if v == "yes":
        return True
    if v == "no":
        return False
    return None


def _hard_filter_results(results: list[dict], classification: ClassificationResult) -> list[dict]:
    c = classification.constraints
    filtered: list[dict] = []
    for entry in results:
        remote = _normalize_bool_flag(entry.get("remote", ""))
        adaptive = _normalize_bool_flag(entry.get("adaptive", ""))
        if c.remote_required is not None and remote is not c.remote_required:
            continue
        if c.adaptive_required is not None and adaptive is not c.adaptive_required:
            continue
        filtered.append(entry)
    return filtered


def _to_recommendations(entries: list[dict]) -> list[Recommendation]:
    recs: list[Recommendation] = []
    for e in entries:
        test_type = ", ".join(e.get("test_type") or [])
        recs.append(
            Recommendation(
                name=e["name"],
                url=e["url"],
                test_type=test_type,
            )
        )
    return recs


def _catalog_url_map(catalog_entries: list[dict]) -> dict[str, dict]:
    return {e["url"]: e for e in catalog_entries}


def validate_recommendations(recommendations: list[dict], catalog: dict) -> list[dict]:
    """
    Safety-net validation after recommendation generation.

    - URL must exactly exist in catalog map
    - Name must match that URL's catalog entry
    """
    validated: list[dict] = []
    for rec in recommendations:
        url = rec.get("url")
        name = rec.get("name")
        cat = catalog.get(url)
        if not cat:
            print(f"[warn] Dropping recommendation with unknown URL: {url}")
            continue
        if cat.get("name") != name:
            print(
                "[warn] Dropping recommendation due to URL/name mismatch: "
                f"url={url}, got_name={name}, expected_name={cat.get('name')}"
            )
            continue
        validated.append(rec)
    return validated


def _missing_constraint_bucket(classification: ClassificationResult) -> Literal[
    "role", "seniority_skills", "test_type", "duration_delivery", "language"
]:
    c = classification.constraints
    if not c.role:
        return "role"
    if not c.seniority and not c.skills:
        return "seniority_skills"
    if not c.test_types:
        return "test_type"
    if (
        c.max_duration_minutes is None
        and c.remote_required is None
        and c.adaptive_required is None
    ):
        return "duration_delivery"
    return "language"


def handle_clarify_needed(classification: ClassificationResult, messages: list[dict]) -> AgentResponse:
    bucket = _missing_constraint_bucket(classification)
    if bucket == "role":
        question = "What role are you hiring for?"
    elif bucket == "seniority_skills":
        question = "What seniority level and key skills should this assessment focus on?"
    elif bucket == "test_type":
        question = "Do you want a specific test type, like knowledge, personality, or simulations?"
    elif bucket == "duration_delivery":
        question = "Do you have any duration limit or remote/adaptive preference?"
    else:
        question = "Do you need the assessment available in a specific language?"
    return AgentResponse(reply=question, recommendations=[], end_of_conversation=False)


def handle_recommend(classification: ClassificationResult, messages: list[dict]) -> AgentResponse:
    query = _constraints_to_query(classification)
    raw = retrieve(query, k=10)
    filtered = _hard_filter_results(raw, classification)
    recommendations = _to_recommendations(filtered)
    reply = f"I found {len(recommendations)} matching SHL assessments based on your constraints."
    return AgentResponse(reply=reply, recommendations=recommendations, end_of_conversation=False)


def handle_refine(classification: ClassificationResult, messages: list[dict]) -> AgentResponse:
    # classify_turn uses full message history, so constraints are cumulative across turns.
    query = _constraints_to_query(classification)
    raw = retrieve(query, k=10)
    filtered = _hard_filter_results(raw, classification)
    recommendations = _to_recommendations(filtered)
    reply = f"Updated shortlist with your new constraints; I now have {len(recommendations)} matches."
    return AgentResponse(reply=reply, recommendations=recommendations, end_of_conversation=False)


def _normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s&\.\-\(\)]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _find_catalog_match(name: str, catalog: list[dict], threshold: float = 0.72) -> tuple[dict | None, float]:
    target = _normalize_name(name)
    best: dict | None = None
    best_score = 0.0

    for entry in catalog:
        n = _normalize_name(entry["name"])
        if n == target:
            return entry, 1.0
        if target in n or n in target:
            score = 0.90
        else:
            score = SequenceMatcher(None, target, n).ratio()
        if score > best_score:
            best_score = score
            best = entry
    if best_score >= threshold:
        return best, best_score
    return None, best_score


def handle_compare(classification: ClassificationResult, messages: list[dict]) -> AgentResponse:
    catalog = load_catalog()
    names = classification.compare_assessments[:2]
    if not names:
        return AgentResponse(
            reply="Please share the two assessment names you want me to compare.",
            recommendations=[],
            end_of_conversation=False,
        )

    matches: list[tuple[str, dict | None, float]] = []
    for n in names:
        match, score = _find_catalog_match(n, catalog, threshold=0.72)
        matches.append((n, match, score))

    missing = [orig for orig, match, _ in matches if match is None]
    if missing:
        return AgentResponse(
            reply=(
                "I couldn't find these assessment(s) in the catalog: "
                + ", ".join(missing)
                + ". Please check the names and try again."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    a = matches[0][1]
    b = matches[1][1] if len(matches) > 1 else None
    assert a is not None
    if b is None:
        reply = (
            f"{a['name']}: type {', '.join(a.get('test_type') or [])}; "
            f"duration {a.get('duration') or 'not specified'}; "
            f"remote {a.get('remote')}; adaptive {a.get('adaptive')}; "
            f"job levels {', '.join(a.get('job_levels') or ['not specified'])}."
        )
        return AgentResponse(reply=reply, recommendations=[], end_of_conversation=False)
    reply = (
        f"{a['name']} vs {b['name']}: "
        f"types [{', '.join(a.get('test_type') or [])}] vs [{', '.join(b.get('test_type') or [])}], "
        f"duration [{a.get('duration') or 'not specified'}] vs [{b.get('duration') or 'not specified'}], "
        f"remote [{a.get('remote')}] vs [{b.get('remote')}], "
        f"adaptive [{a.get('adaptive')}] vs [{b.get('adaptive')}], "
        f"job levels [{', '.join(a.get('job_levels') or ['not specified'])}] vs "
        f"[{', '.join(b.get('job_levels') or ['not specified'])}]."
    )
    return AgentResponse(reply=reply, recommendations=[], end_of_conversation=False)


def handle_refuse(classification: ClassificationResult, messages: list[dict]) -> AgentResponse:
    text = (messages[-1].get("content", "") if messages else "").lower()
    if "ignore" in text or "instructions" in text:
        reply = "I can only help with SHL assessment recommendations."
    elif any(k in text for k in ("legal", "law", "compliance", "policy advice")):
        reply = "I can't provide legal or policy advice; I can help with SHL assessment selection."
    else:
        reply = "I can help only with SHL assessment recommendations for hiring and talent decisions."
    return AgentResponse(reply=reply, recommendations=[], end_of_conversation=False)


def generate_agent_response(messages: list[dict]) -> AgentResponse:
    try:
        classification = classify_turn(messages)
        if classification.intent == "CLARIFY_NEEDED":
            return handle_clarify_needed(classification, messages)
        if classification.intent == "RECOMMEND":
            response = handle_recommend(classification, messages)
        elif classification.intent == "REFINE":
            response = handle_refine(classification, messages)
        elif classification.intent == "COMPARE":
            return handle_compare(classification, messages)
        else:
            return handle_refuse(classification, messages)

        # Hard safety net always runs for shortlist-producing intents.
        catalog_map = _catalog_url_map(load_catalog())
        rec_dicts = [r.model_dump() for r in response.recommendations]
        valid_dicts = validate_recommendations(rec_dicts, catalog_map)
        response.recommendations = [Recommendation.model_validate(r) for r in valid_dicts]
        return response
    except Exception as exc:
        print(f"[warn] Falling back to safe response due to upstream error: {exc}")
        return SAFE_FALLBACK_RESPONSE

