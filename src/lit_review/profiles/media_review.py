"""Explicit media-taxonomy review planning and section instructions."""

from __future__ import annotations
from typing import Any
from lit_review.core.chat import ChatCompletionClient, LOGGER
from lit_review.collection.sources import dedupe_preserve
import json

TAXONOMY_QUESTIONS = [
    "What persuasion technique taxonomies and propaganda technique labels are used in NLP datasets and media analysis?",
    "What operational definitions, synonyms, and examples are given for persuasion techniques such as loaded language, name calling, fear appeal, scapegoating, false dilemma, appeal to authority, and repetition?",
    "How do papers hierarchically group persuasion techniques, rhetorical devices, discourse strategies, propaganda mechanisms, and media bias features?",
    "How are explicit and implicit persuasion techniques distinguished in annotation schemes?",
    "How is Moral Foundations Theory operationalized for computational text analysis, political communication, and moral framing?",
    "What moral framing categories beyond Moral Foundations Theory appear in ideological, ethical, virtue, justice, purity, national identity, or collective morality research?",
    "What examples of moral signaling language are used for care harm, fairness cheating, loyalty betrayal, authority subversion, sanctity degradation, and liberty oppression?",
    "What emotion categories and affective dimensions are used in emotion detection, affective computing, and emotionally manipulative media analysis?",
    "How do papers distinguish fine grained emotions from positive negative neutral sentiment?",
    "What emotional manipulation strategies, moral emotional language, outrage, fear, disgust, anxiety, sympathy, pride, hope, distrust, cynicism, resentment, patriotism, and collective victimhood categories are described?",
    "Where do persuasion techniques, moral framing, and emotional targeting overlap, and what boundary rules keep them distinct for annotation?",
    "What annotation schemas, benchmark datasets, and validation practices support operationalizing a persuasion attribute taxonomy?",
]


TAXONOMY_OUTLINE = [
    "1. Taxonomy Design Principles",
    "2. Category 1: Persuasion Techniques",
    "3. Category 2: Moral Framing",
    "4. Category 3: Emotional and Affective Targeting",
    "5. Boundary Rules and Co-occurrence Map",
    "6. Copyable JSON Schema and Seed Entries",
    "7. Evidence Gaps",
]


def taxonomy_plan_goal(goal_text: str, max_questions: int) -> dict[str, Any]:
    from lit_review.evidence.review import derive_goal_title

    return {
        "title": derive_goal_title(goal_text),
        "questions": TAXONOMY_QUESTIONS[:max_questions],
        "outline": TAXONOMY_OUTLINE,
        "mode": "taxonomy",
        "question_source": "deterministic",
    }


def llm_plan_taxonomy_goal(
    goal_text: str,
    client: ChatCompletionClient,
    max_questions: int,
    temperature: float = 0.1,
) -> dict[str, Any]:
    from lit_review.evidence.review import derive_goal_title, parse_questions_json, repair_questions_json

    content = client.chat(
        [
            {
                "role": "system",
                "content": (
                    "You plan retrieval queries for an evidence-bound taxonomy builder. "
                    "Return strict JSON only with one key: questions."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Create focused search questions for retrieving evidence to build a "
                    "three-part persuasion attribute taxonomy: persuasion techniques, "
                    "moral framing, and emotional or affective targeting. The questions "
                    "must search for feature names, definitions, synonyms, examples, "
                    "hierarchies, annotation schemas, and boundary rules. Return JSON "
                    f"with at most {max_questions} questions.\n\nGOAL:\n{goal_text}"
                ),
            },
        ],
        temperature=temperature,
        max_tokens=max(2000, min(8000, 800 + max_questions * 160)),
        response_format={"type": "json_object"},
    )
    try:
        questions = parse_questions_json(content)
    except (json.JSONDecodeError, ValueError) as exc:
        LOGGER.warning("Taxonomy question planner returned invalid JSON; repairing: %s", exc)
        questions = repair_questions_json(content, client=client, max_questions=max_questions)

    merged_questions = dedupe_preserve([*questions, *TAXONOMY_QUESTIONS])[:max_questions]
    return {
        "title": derive_goal_title(goal_text),
        "questions": merged_questions,
        "outline": TAXONOMY_OUTLINE,
        "mode": "taxonomy",
        "question_source": "llm",
    }


def section_task(section_title: str) -> str:
    if "persuasion techniques" in section_title:
        section_task = (
            "Build a taxonomy table for persuasion techniques. Include columns: "
            "Parent group, Feature name, Definition, Synonyms or related labels, "
            "Explicit or implicit, Example indicators, Evidence. Include only features "
            "supported by evidence."
        )
    elif "moral framing" in section_title:
        section_task = (
            "Build a taxonomy table for moral framing. Include columns: Moral category, "
            "Definition, Framework or mapping, Example signals, Cross-cultural notes, "
            "Evidence. Include MFT categories when supported, and add non-MFT categories "
            "only when supported by evidence."
        )
    elif "emotional" in section_title or "affective" in section_title:
        section_task = (
            "Build a taxonomy table for emotional or affective targeting. Include columns: "
            "Emotion or affective target, Definition, Valence or dimension, Persuasive use, "
            "Example indicators, Evidence. Do not collapse these into positive/negative/neutral."
        )
    elif "boundary" in section_title or "co-occurrence" in section_title:
        section_task = (
            "Write concise boundary rules that distinguish rhetorical mechanism, moral appeal, "
            "and emotional outcome. Include a small mapping table for common co-occurrences."
        )
    elif "json" in section_title or "schema" in section_title:
        section_task = (
            "Provide a copyable JSON schema and 6-10 seed entries grounded in evidence. "
            "Each seed entry must include category, feature_name, definition, source, "
            "synonyms, parent_category, examples, and notes."
        )
    else:
        section_task = (
            "Summarize only the evidence needed to support the taxonomy design. Focus on "
            "criteria, scope, and operationalization decisions."
        )

    return section_task
