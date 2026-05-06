# Media Attribute Taxonomies for Persuasion, Moral Framing, and Emotion Classification

## Purpose

This literature review should identify classification categories for three observable attributes of media texts:

1. persuasion techniques in writing
2. moral framing categories
3. sentiment and emotion categories

The review is not primarily about stance detection, ideology detection, media bias, misinformation truth status, or actor agenda inference. Those may be downstream applications, but the collection goal is to find taxonomies, annotation schemas, codebooks, corpora, and validated label inventories that can be used to classify pieces of media.

Prioritize papers that provide operational label definitions, annotation guidelines, inter-annotator agreement, construct validity, discriminant validity, multi-label annotation, span-level annotation, sentence-level annotation, document-level annotation, or benchmark datasets for media and political text.

## Core Research Questions

What persuasion technique taxonomies are useful for classifying writing in news, propaganda, political communication, advocacy, public relations, and social media?

Which moral psychology frameworks provide practical moral framing labels for media text classification?

Which sentiment and emotion taxonomies provide enough range to classify intended reader affect, expressed emotion, emotional framing, and target-specific sentiment in media?

How should overlapping labels be handled when a text simultaneously uses persuasion, moral framing, and emotion?

## Attribute Family 1: Persuasion Techniques in Writing

Search for work on propaganda techniques, persuasive message features, rhetorical strategies, social influence principles, argumentation schemes, fallacy detection, and framing techniques. Favor inventories designed for annotating written media, especially news articles, political communication, advocacy text, campaign messages, online propaganda, and social media discourse.

Use fine-grained propaganda detection as the strongest seed because it directly annotates media text spans for persuasive and propagandistic techniques. Also collect communication and psychology taxonomies when they give categories that are easier for human annotators to recognize.

### Candidate Persuasion Labels

Emotional appeals: appeal to fear, threat appeal, appeal to anger, outrage appeal, appeal to pity, empathy appeal, hope appeal, aspiration appeal, guilt appeal, shame appeal, disgust appeal, contamination appeal.

Identity and group appeals: in-group loyalty appeal, out-group threat, scapegoating, demonization, dehumanization, flag-waving, patriotic appeal, nationalism, group solidarity, unity appeal, bandwagon, social proof.

Credibility and source appeals: appeal to authority, expert endorsement, testimonial, reputation appeal, ethos, credibility cue, trusted messenger.

Evidence and information selection: card stacking, cherry picking, selective evidence, omission, anecdotal evidence, misleading statistics, framing by emphasis, quote mining, one-sided evidence.

Argument and fallacy techniques: ad hominem, name calling, labeling, straw man, whataboutism, tu quoque, red herring, false dilemma, black-and-white fallacy, slippery slope, causal oversimplification, hasty generalization, appeal to ignorance, sowing doubt.

Language and repetition: loaded language, emotionally charged language, slogans, repetition, glittering generalities, euphemism, dysphemism, rhetorical question, thought-terminating cliche, exaggeration, minimization.

Urgency and compliance pressure: scarcity, loss framing, deadline pressure, call to action, commitment and consistency appeal, reciprocity appeal, crisis framing, emergency framing.

### Persuasion Search Terms

propaganda technique classification; fine-grained propaganda detection; propaganda span identification; propaganda technique labeling; rhetorical technique detection; persuasive strategy taxonomy; persuasion technique annotation; persuasive message features; rhetorical devices media classification; argumentation scheme annotation; fallacy detection political discourse; social influence principles in writing; Cialdini persuasion principles text analysis; ethos pathos logos classification; fear appeal media persuasion; threat appeal political communication; loaded language detection; name calling labeling propaganda; flag waving propaganda; causal oversimplification; whataboutism detection; straw man detection; card stacking cherry picking; black-and-white fallacy; thought-terminating cliche; scapegoating media discourse; dehumanization media discourse.

### Persuasion Boolean Queries

("propaganda technique" OR "propaganda techniques") AND (classification OR taxonomy OR annotation OR corpus OR codebook) AND ("news articles" OR media OR "political communication")

("fine-grained propaganda detection" OR "propaganda span identification" OR "propaganda technique labeling") AND (news OR media OR propaganda)

("persuasion technique" OR "persuasive strategy" OR "persuasive message features") AND (taxonomy OR annotation OR classification OR detection) AND (writing OR text OR media OR "political communication")

("loaded language" OR "name calling" OR "flag waving" OR "appeal to fear" OR "causal oversimplification") AND (propaganda OR media OR news) AND (classification OR detection OR annotation)

("whataboutism" OR "tu quoque" OR "straw man" OR "red herring" OR "false dilemma") AND ("political discourse" OR media OR propaganda) AND (detection OR annotation OR classification)

("argumentation scheme" OR "fallacy detection" OR "rhetorical device") AND (taxonomy OR annotation OR classification) AND (media OR news OR "political text")

("social proof" OR reciprocity OR scarcity OR authority OR consistency OR unity) AND (persuasion OR influence) AND ("text analysis" OR "message features" OR media)

## Attribute Family 2: Moral Framing Categories

Search for moral framing and moral sentiment work rooted in moral psychology. Moral Foundations Theory is the strongest starting point because it offers a compact, widely used category inventory for moral language in political and social discourse. Collect both foundation-level labels and virtue/vice polarity labels, because media classification may need to distinguish pro-social moral appeals from moral violation accusations.

### Candidate Moral Framing Labels

Care or protection: care, compassion, protection, nurturance, safety, defending the vulnerable.

Harm or victimization: harm, suffering, cruelty, injury, danger, abuse, victimization.

Fairness or justice: fairness, equality, rights, reciprocity, honesty, due process, justice.

Cheating or exploitation: cheating, unfairness, corruption, exploitation, discrimination, hypocrisy.

Loyalty or solidarity: loyalty, patriotism, sacrifice, group solidarity, family, nation, shared fate.

Betrayal or disloyalty: betrayal, treason, abandonment, disloyalty, siding with an out-group.

Authority or order: authority, law, duty, tradition, hierarchy, leadership, social order.

Subversion or disorder: subversion, rebellion against authority, lawlessness, chaos, disrespect for tradition.

Sanctity or purity: purity, sanctity, sacredness, dignity, bodily integrity, spiritual elevation.

Degradation or contamination: degradation, contamination, disgust, desecration, impurity, corruption of the body or sacred things.

Liberty or autonomy: liberty, freedom, autonomy, self-determination, individual rights, freedom from interference.

Oppression or coercion: oppression, tyranny, coercion, domination, censorship, state overreach, forced compliance.

Nonmoral or unclear: no clear moral frame, descriptive claim without moral evaluation, ambiguous moral content.

Optional higher-level grouping: individualizing foundations include care and fairness; binding foundations include loyalty, authority, and sanctity. Liberty or oppression should be treated as an optional extension when the corpus contains freedom, coercion, anti-government, civil liberties, or censorship discourse.

### Moral Framing Search Terms

moral foundations theory; moral framing detection; moral sentiment classification; moral rhetoric; moral language; moral foundations dictionary; extended Moral Foundations Dictionary; Moral Foundations Twitter Corpus; moral foundations annotation; care harm fairness cheating loyalty betrayal authority subversion purity degradation; sanctity degradation; liberty oppression; moral reframing political communication; moral emotions political discourse; moral outrage; moral conviction; moralized language; virtue vice moral sentiment; individualizing binding foundations; group identity moral language; intergroup threat moral framing.

### Moral Framing Boolean Queries

("moral foundations theory" OR "moral foundations") AND (text OR language OR media OR "political communication") AND (classification OR annotation OR detection OR corpus)

("moral framing" OR "moral rhetoric" OR "moral language" OR "moral sentiment") AND (media OR news OR propaganda OR "political discourse") AND (taxonomy OR classification OR annotation)

("Moral Foundations Twitter Corpus" OR "moral foundations dictionary" OR "extended Moral Foundations Dictionary") AND (annotation OR corpus OR classification OR lexicon)

("care/harm" OR "fairness/cheating" OR "loyalty/betrayal" OR "authority/subversion" OR "purity/degradation") AND ("text analysis" OR NLP OR media OR discourse)

("liberty/oppression" OR "freedom" OR "coercion" OR "oppression") AND ("moral foundations" OR "moral framing" OR "moral rhetoric") AND (text OR media OR "political discourse")

("moral outrage" OR "moral conviction" OR "moral emotion") AND (media OR "political communication" OR propaganda) AND (classification OR annotation OR detection)

## Attribute Family 3: Sentiment and Emotion Categories

Search for sentiment analysis, emotion detection, affective computing, emotional framing, intended reader emotion, expressed emotion, target-specific sentiment, and affective dimensions. For media classification, collect both compact emotion taxonomies and fine-grained datasets. The final codebook may use a compact media emotion set, while retaining mappings to larger inventories such as GoEmotions, NRC Emotion Lexicon, Plutchik emotions, Ekman basic emotions, and valence-arousal-dominance models.

### Candidate Sentiment and Emotion Labels

Sentiment polarity: positive, negative, mixed, neutral.

Targeted sentiment: favorable toward target, unfavorable toward target, ambivalent toward target, no target-specific sentiment. Targeted sentiment should be metadata for the emotional target, not a fourth research family.

Compact media emotion labels: anger or outrage, fear or anxiety, disgust or contempt, sadness or grief, joy or amusement, hope or optimism, pride or admiration, shame or guilt or remorse, empathy or care or compassion, surprise or shock, confusion or uncertainty, trust or approval, curiosity or interest, neutral or no clear emotion.

Fine-grained emotion expansion: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise.

Affective dimensions: valence, arousal, dominance; positive affect, negative affect; high-arousal negative emotion, low-arousal negative emotion, high-arousal positive emotion, low-arousal positive emotion.

Moral and political emotions: moral outrage, indignation, contempt, disgust, anger, fear, threat perception, empathy, compassion, guilt, shame, pride, resentment, hope.

### Sentiment and Emotion Search Terms

sentiment analysis; opinion mining; emotion detection; emotion classification; affective computing; emotional framing; intended reader emotion; reader emotion; emotion annotation; emotion taxonomy; GoEmotions; NRC Emotion Lexicon; NRC VAD Lexicon; valence arousal dominance; circumplex model of affect; Plutchik emotions; Ekman basic emotions; political emotion detection; moral emotion classification; outrage detection; fear appeal emotion; anger fear disgust sadness joy hope media; target-specific sentiment; aspect-based sentiment; sentiment toward entity; emotion in political communication; affective polarization emotion; threat perception media.

### Sentiment and Emotion Boolean Queries

("emotion detection" OR "emotion classification" OR "emotion annotation") AND (media OR news OR "political communication" OR propaganda) AND (taxonomy OR corpus OR codebook)

("sentiment analysis" OR "opinion mining" OR "target-specific sentiment" OR "aspect-based sentiment") AND (media OR news OR "political discourse") AND (classification OR annotation)

("GoEmotions" OR "NRC Emotion Lexicon" OR "NRC VAD Lexicon") AND (emotion OR sentiment OR affect) AND (classification OR annotation OR taxonomy)

("valence arousal dominance" OR VAD OR "circumplex model of affect") AND (text OR language OR media OR "political communication")

("fear" OR anger OR disgust OR sadness OR joy OR hope OR pride OR shame OR guilt OR empathy OR contempt OR outrage) AND (media OR news OR propaganda OR "political communication") AND (classification OR detection OR annotation)

("intended emotion" OR "reader emotion" OR "emotional framing" OR "affective framing") AND (media OR news OR persuasive OR propaganda)

## Annotation Schema and Validity

Search for codebook development and annotation validity across all three attribute families. The review should capture how researchers define labels, decide between single-label and multi-label annotation, handle overlapping spans, and evaluate human agreement.

Important design questions: Are labels mutually exclusive or multi-label? Is annotation done at the document, paragraph, sentence, clause, or span level? Are labels assigned to the whole article, the authorial stance, the target entity, or the expected reader response? Does the annotation distinguish expressed emotion from intended reader emotion? Does moral framing include virtue and violation directions? Does persuasion classification distinguish technique, claim content, and truth status?

### Annotation Search Terms

annotation schema; codebook development; taxonomy development; label inventory; construct validity; discriminant validity; convergent validity; inter-annotator agreement; intercoder reliability; Cohen kappa; Krippendorff alpha; multi-label text classification; hierarchical text classification; span-level annotation; sentence-level annotation; document-level annotation; overlapping labels; human-in-the-loop classification; explainable NLP; media annotation; political text annotation.

### Annotation Boolean Queries

("annotation schema" OR "codebook development" OR "taxonomy development" OR "label inventory") AND (persuasion OR propaganda OR "moral framing" OR emotion OR sentiment) AND (media OR "political text")

("inter-annotator agreement" OR "intercoder reliability" OR "Krippendorff alpha" OR "Cohen kappa") AND (propaganda OR persuasion OR "moral sentiment" OR emotion) AND (annotation OR corpus)

("multi-label classification" OR "hierarchical classification" OR "span-level annotation") AND (propaganda OR persuasion OR "moral framing" OR emotion OR sentiment)

## Inclusion Criteria

Include papers, corpora, books, and codebooks that provide reusable categories for persuasion techniques, moral framing, sentiment, or emotion in text.

Include theory papers when they define categories that can become annotation labels, especially moral psychology, persuasion psychology, rhetoric, argumentation, affective science, and communication research.

Include NLP papers when they publish a dataset, benchmark, taxonomy, annotation guide, label definitions, or a model built around an interpretable category inventory.

Include media, news, propaganda, political communication, social media, advocacy, campaign, and public discourse studies when they classify texts using one of the three attribute families.

## Exclusion Criteria

Exclude pure stance detection, ideology prediction, media bias classification, fake news detection, fact checking, bot detection, network diffusion, agenda setting, and public opinion polling unless the paper also defines categories for persuasion techniques, moral framing, sentiment, or emotion.

Exclude general marketing conversion studies unless they provide a transferable taxonomy of persuasive message features or social influence categories for text annotation.

Exclude clinical emotion studies unless they provide a general text emotion taxonomy useful for media classification.

## Seed Literature Anchors

Fine-Grained Analysis of Propaganda in News Articles; SemEval-2020 Task 11: Detection of Propaganda Techniques in News Articles; Findings of the NLP4IF Shared Task on Fine-Grained Propaganda Detection.

Cialdini persuasion principles; social proof, authority, scarcity, reciprocity, consistency, liking, unity.

Argumentation schemes and fallacy annotation; Walton argumentation schemes; argument from authority, argument from popular opinion, argument from cause to effect, ad hominem, slippery slope.

Moral Foundations Theory; Mapping the Moral Domain; Moral Foundations Twitter Corpus; Moral Foundations Dictionary; extended Moral Foundations Dictionary; moral reframing and political communication.

GoEmotions; NRC Emotion Lexicon; NRC VAD Lexicon; Plutchik emotion taxonomy; Ekman basic emotions; circumplex model of affect; sentiment analysis and opinion mining.

## Preferred Output of the Literature Review

The review should produce a practical annotation codebook for media classification with three sections:

1. persuasion technique labels with definitions and examples
2. moral framing labels with virtue and violation directions
3. sentiment and emotion labels with polarity, discrete emotion, and affective dimension mappings

Each section should identify which labels are best supported by prior literature, which labels are redundant, which labels are hard for annotators to distinguish, and whether the label should be applied at document, sentence, or span level.
