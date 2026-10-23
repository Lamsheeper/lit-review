"""Profiles embedding_cleanup."""

from __future__ import annotations


PERSUASION_HIGH_CONFIDENCE_BOILERPLATE = (
    "A persuasive communication technique",
    "A persuasive communication strategy",
    "A persuasive framing technique",
    "A persuasive linguistic technique",
    "A persuasive technique",
    "A persuasive tactic",
    "A persuasive strategy",
    "A persuasive method",
    "A persuasive approach",
    "A persuasion technique",
    "A persuasion tactic",
    "A persuasion strategy",
    "The strategic use of",
    "The deliberate use of",
    "The practice of",
    "The process of",
    "The use of",
    "The act of",
)


SENTIMENT_AFFECT_HIGH_CONFIDENCE_BOILERPLATE = (
    "An emotional state characterized by",
    "An emotional state of",
    "An emotional state",
    "An affective state of",
    "An affective state",
    "An intense feeling of",
    "An intense feeling",
    "A strong feeling of",
    "A strong feeling",
    "A feeling of",
    "A feeling",
    "The state of",
    "A state of",
    "A state",
)


EMBEDDING_TEXT_CLEANUP_BUILTINS = {
    "persuasion_high_confidence_boilerplate": PERSUASION_HIGH_CONFIDENCE_BOILERPLATE,
    "sentiment_affect_high_confidence_boilerplate": (SENTIMENT_AFFECT_HIGH_CONFIDENCE_BOILERPLATE),
}
