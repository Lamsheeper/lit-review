# Goal: Extract Persuasion Attribute Taxonomy from Literature

## Objective

Conduct a literature review focused on identifying and organizing attributes used to analyze persuasive, propagandistic, ideological, rhetorical, or emotionally manipulative media content.

The goal is to construct a structured taxonomy of persuasion-related features for annotation and computational analysis of text/media.

The taxonomy should be divided into the following three primary categories:

---

# Category 1: Persuasion Techniques

## Definition
Mechanisms, rhetorical devices, discourse strategies, or propaganda techniques used to influence beliefs, attitudes, emotions, or behavior.

These describe *how* persuasion is performed.

## Desired Outputs
Extract:
- Technique names
- Definitions
- Synonyms/alternate terminology
- Example usage
- Hierarchical groupings if present
- Whether techniques are explicit or implicit

## Example Techniques
- Loaded language
- Name-calling
- Fear appeal
- Scapegoating
- Whataboutism
- False dilemma
- Appeal to authority
- Bandwagoning
- Repetition
- Euphemism
- Dehumanization
- Emotional anecdote
- Glittering generalities
- Framing
- Selective omission
- Moral panic induction
- Conspiracy implication

## Relevant Research Areas
Search across:
- Propaganda detection
- Computational rhetoric
- Persuasion analysis
- Argument mining
- Political communication
- Media bias detection
- Information operations
- Disinformation research
- Computational social science
- NLP persuasion datasets

---

# Category 2: Moral Framing

## Definition
Moral values, ethical dimensions, or normative frameworks invoked by persuasive content.

These describe *which moral principles* are activated or appealed to.

## Desired Outputs
Extract:
- Moral categories/frameworks
- Definitions
- Cross-cultural variants
- Moral Foundation Theory mappings
- Examples of framing language
- Explicit vs implicit moral signaling

## Important Frameworks
Prioritize:
- Moral Foundations Theory (MFT)
  - Care/Harm
  - Fairness/Cheating
  - Loyalty/Betrayal
  - Authority/Subversion
  - Sanctity/Degradation
  - Liberty/Oppression

Also search for:
- Moral-emotional framing
- Ethical framing
- Virtue framing
- Collective morality
- National identity morality
- Purity rhetoric
- Justice rhetoric

## Relevant Research Areas
Search across:
- Moral psychology
- Moral NLP
- Political framing
- Computational ethics
- Ideological framing
- Cultural linguistics
- Narrative framing research

---

# Category 3: Emotional / Affective Targeting

## Definition
Emotional states or affective reactions that persuasive media attempts to induce in audiences.

These describe *what the audience is intended to feel*.

## Desired Outputs
Extract:
- Emotion categories
- Affective dimensions
- Persuasion-emotion relationships
- Emotional manipulation strategies
- Emotion taxonomies
- Distinctions between sentiment and emotion

## Example Emotional Targets
- Fear
- Anger
- Disgust
- Anxiety
- Pride
- Hope
- Sympathy
- Distrust
- Cynicism
- Resentment
- Moral outrage
- Patriotism
- Collective victimhood
- Empathy

## Important Notes
Do NOT collapse emotions into only:
- positive
- negative
- neutral

Prioritize fine-grained emotional categories.

## Relevant Research Areas
Search across:
- Emotion detection
- Affective computing
- Emotional framing
- Sentiment analysis
- Political emotion research
- Moral-emotional language
- Computational psychology

---

# Extraction Requirements

For every extracted feature/category:
- Name
- Definition
- Source papers
- Citation/context snippet
- Related/similar terms
- Parent category if hierarchical
- Example phrases if available


Return a copyable JSON within the report markdown file with the format:

```json
{
  "category": "...",
  "feature_name": "...",
  "definition": "...",
  "source": "...",
  "synonyms": [],
  "parent_category": "...",
  "examples": [],
  "notes": "..."
}
```

# Prioritization Criteria

Prioritize:
1. Frequently recurring features across papers
2. Features used in existing NLP datasets/benchmarks
3. Features used in propaganda or persuasion detection
4. Features with operationalizable definitions
5. Fine-grained taxonomies over vague labels
6. Cross-lingual or cross-cultural applicability
7. Explicit distinctions between rhetoric, morality, and emotion

---

# Important Constraints

- Do not merge persuasion techniques, moral framing, and emotional targeting into one category.
- Preserve distinctions between:
  - rhetorical mechanism,
  - moral appeal,
  - emotional outcome.
- Prefer academically grounded terminology over informal internet terminology.
- Include disagreements or taxonomy conflicts across papers when relevant.
- Preserve hierarchical structure when possible.

---

# End Goal

Produce a consolidated persuasion attribute taxonomy suitable for:
- dataset annotation,
- computational persuasion analysis,
- LLM evaluation,
- cross-cultural narrative analysis,
- propaganda detection,
- institutional trust analysis,
- media framing research.