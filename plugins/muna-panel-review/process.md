# Reader Panel Review — Simulated Audience Mood Journals

Simulate a diverse panel of readers — each representing a distinct position in the document's potential audience — reading a document chapter by chapter and recording their reactions as mood journals. The primary output is not technical findings but **editorial intelligence**: how the document lands with different audiences, what it communicates vs what it intends, where it loses readers, and what derivative documents the audience actually needs.

This is not the expert panel review (`expert-panel-review.md`), which stress-tests technical claims. This is audience simulation for editorial improvement. Run it after the document is technically sound but before finalising for publication.

**Panel configuration:** The project-specific panel — personas, document suite, scenario framing, and voice samples — lives in `config.md` alongside this file. This process document is reusable; the config is per-project. To adapt for another document, write a new `config.md` using the specification formats below.

## When to use this

- A document is approaching publication-ready and you need to understand how it will land with its intended audiences
- You suspect the document serves some audiences better than others but don't know which or why
- You need to discover what derivative documents (executive briefs, practical guides, quick-starts) the audience requires
- You want to surface vocabulary burden, length barriers, and structural problems that technical reviewers don't notice
- You want to understand the political, commercial, or institutional dynamics the document will trigger

## Prerequisites

- A document (or set of documents) that can be divided into discrete chapters or sections
- A clear sense of the document's intended audiences — the panel should span the full decision chain from executive to end-user
- A panel configuration file (`config.md`) defining the document suite, scenario framing, personas, and voice samples
- An output directory for mood journals (e.g., `insights/` or `reviews/panel/`)

## Process Overview

The review runs in five phases, with two optional extensions:

1. **Panel design** — define personas (including a control persona, and optionally an unreliable narrator) and the document suite they will navigate
2. **Reading** — each persona chooses their entry point, navigates the document suite freely, and records mood journals along the way. The persona decides which documents to read, in what order, and when to stop — including abandoning one document to try another and returning later
3. **Final reflections** — each persona writes an in-character summary document based on what they actually read
3b. **Persona collision tests** (optional) — selected pairs of personas with opposed verdicts react to each other's conclusions, testing whether disagreements are structural or shallow
4. **Cross-panel synthesis** — aggregate findings across all personas into editorial recommendations
5. **Derivative document development** — re-engage personas to develop and review derivative documents identified by the synthesis

**Variant: Delegation chain simulation.** An alternative execution mode that models how documents are transformed as they move through institutional hierarchies — what survives when the analyst writes a brief, the director amends it, and the senior executive reads the brief instead of the source. See "Delegation chain simulation" in Execution Notes.

---

## Phase 1: Panel Design

### Design principles

**Span the decision chain.** The panel should include personas from the person who decides whether to act on the document down to the person most affected by its recommendations. Include at least one persona who is *talked about* by the document but not directly *talked to* — this persona will surface the largest editorial gaps.

**Vary technical depth.** Include at least one persona who will struggle with the document's vocabulary and at least one who will find it insufficiently rigorous. The tension between these reactions reveals the document's actual accessibility range.

**Include institutional perspectives.** If the document touches policy, governance, or organisational boundaries, include personas who represent the institutions that will receive, respond to, or be affected by the document's recommendations. These personas surface political dynamics that purely technical reviewers miss.

**Include commercial/external perspectives.** If the document has implications beyond its authoring organisation, include personas who represent vendors, contractors, researchers, or allied organisations. These personas surface unintended consequences and opportunities.

**Give each persona a distinct lens.** The persona's value comes from filtering the same content through a different cognitive frame. Two personas with the same role but different lenses (e.g., a CISO focused on operational controls vs a CISO focused on compliance reporting) are more useful than two personas with different roles but the same lens.

**Define blind spots explicitly.** In agent implementations, the model has access to all knowledge — a simulated junior developer "knows" what STRIDE is, a citizen programmer "knows" the security control framework. The `Blind spots` field in the persona specification is what prevents persona bleed. Without it, all personas converge toward the same well-informed analytical voice. Define what each persona does not know, cannot access, and does not care about. This field is not optional decoration — it is the primary mechanism for maintaining persona differentiation in LLM execution.

### Persona specification format

Define each persona with:

```
**Name/Role:** [Job title and organisational position]
**Lens:** [What this persona optimises for — the 2-3 things they care about most]
**Background:** [2-3 sentences establishing their expertise, experience, and current situation]
**Reading behaviour:** [How they'll actually approach the document — not how they *should* read it, but how they *will*. Will they read front-to-back? Bail after the executive summary? Skip to the appendix with the numbers? Jump to recommendations? Stop when it gets too technical? This field drives the persona's reading path — see "Realistic reading paths" in Phase 2]
**Key question:** [The single question this persona is trying to answer by reading the document]
**Blind spots:** [What this persona explicitly does not know, cannot access, or does not care about — these boundaries prevent persona bleed in agent execution where the model has all knowledge available and must be told what to suppress]
**Voice sample:** [One sentence in character — how this persona sounds when talking to a peer. This anchors the persona's register for the entire reading. See "Voice re-anchoring" in Phase 2 for how this is used during long runs]
**Consultation voice:** [One sentence showing how this persona reacts to the scenario framing — receiving this specific document through their specific channel. Omit if no scenario framing is defined]
```

The voice samples are part of the persona specification, not a separate section, because they travel with the persona when the panel is adapted for other documents. When you redesign the panel, you write new voice samples as part of each new persona — you don't maintain a separate voice calibration table that may fall out of sync.

### Document suite

Define the document suite in `config.md`. The suite is presented to each persona at the start of their reading session. The persona chooses which document to open first and navigates from there — the choice itself is data. The suite map (if present) is provided as an orientation aid, not a prescription.

### Scenario framing

Define the scenario framing in `config.md`, or omit it. If the document has an institutional context — who published it, what status it has, how the reader received it — define that as a scenario framing. This is optional but high-value: it transforms personas from passive readers into situated actors with stakes, access dynamics, and response obligations. The framing shapes reading behaviour and produces richer editorial intelligence.

If the document is a draft RFC circulated to a working group, that's a framing. If it's a published standard, that's a different framing. If it's an internal memo, that's another.

If omitted, remove the "Consultation response" section from the overall verdict template, the "Consultation Response Forecast" section from the synthesis, and the `Consultation voice` field from persona specifications.

### Panel sizing

- **Full document review:** 7+ personas. Panel quality depends on persona design, not headcount — add personas until you have covered the full decision chain and all institutional perspectives the document touches.
- **Chapter or section review:** 3–5 personas. Pick the personas most relevant to the content under review.
- **Derivative document review:** 2–3 personas. Pick the target audience persona plus one adjacent persona (e.g., for an executive brief, use the executive sponsor persona plus the security lead or policy advisor).

### Persona priority

If you cannot run the full panel (budget, time, or context window constraints), prioritise personas that maximise tension:

1. **The person talked about but not talked to.** This persona surfaces the largest editorial gaps — the document discusses their situation without addressing them directly.
2. **The decision-maker.** The persona who decides whether the document leads to action.
3. **The hostile reader.** The persona with the most to lose or the strongest reason to push back.
4. **The technical validator.** The persona who will judge whether the document's claims hold.

A four-persona panel built from these archetypes will produce more editorial intelligence than a thirteen-persona panel where most personas occupy similar positions in the decision chain.

Map these archetypes to specific personas in your `config.md`.

### Control persona

Include one persona whose reactions you can predict with high confidence *before* the panel runs. This persona calibrates the entire process.

The control persona should be someone for whom the document was explicitly designed — whose reading path, concerns, and verdict you can pre-register based on the document's stated purpose. Define:

```
**Pre-registered predictions:**
- Expected reading path: [which documents, which chapters, where they stop]
- Expected key concern: [the issue this persona should flag]
- Expected verdict: [broadly positive / mixed / negative, and why]
```

After the panel runs, compare actual to predicted. Three outcomes:

1. **Actual matches predicted.** The simulation is producing persona-specific reactions, not just model-prior noise. Confidence in the rest of the panel increases.
2. **Actual diverges — and the divergence makes sense.** The document surprised its target audience in a way the author didn't anticipate. This is a genuine finding — possibly the most valuable single finding in the panel.
3. **Actual diverges — and the divergence is implausible.** The simulation is not working for this persona type. Findings from similar personas in the panel should be downweighted.

The control persona costs one panel slot and one page of pre-registration. It is the cheapest calibration mechanism available and should be included in every panel run. Identify which persona serves as the control in your `config.md`.

### The unreliable narrator

Include one persona who arrives with a **deliberate misconception** about the document's topic. This persona's `Background` field contains a specific wrong belief that the document should correct through its own argument.

This stress-tests the document's ability to redirect readers who arrive with the wrong mental model — a common real-world scenario. Readers encounter documents through titles, forwarded emails, corridor summaries, and media coverage, all of which can set false expectations. A document that cannot correct a reasonable misconception in its first few chapters will lose readers who arrived with that misconception, and will never know it lost them.

Define the misconception explicitly in the persona specification:

```
**Misconception:** [A specific wrong belief the persona holds about the topic.
Must be plausible — something a reasonable person in this role could believe
based on the document's title, media coverage, or institutional context.]
```

Track across the reading:
- **How many chapters** before the persona recognises the misconception? If the document has not corrected it by chapter 3, the framing has a problem.
- **Does the persona correct spontaneously**, or does the misconception persist and distort their reading of later chapters?
- **Does the misconception resurface** in the overall verdict, even if it was corrected during reading? Corrected misconceptions that return in the summary tell you the correction did not stick.

The unreliable narrator costs one panel slot. It produces a finding that no other persona can generate: how robust is the document's argument against a reader who starts in the wrong place? Define the misconception and which persona carries it in your `config.md`.

---

## Phase 2: Chapter-by-Chapter Mood Journals

### The cardinal rule: no read-ahead, mandatory output

> **🚨 CRITICAL — READ THIS BEFORE ANYTHING ELSE 🚨**
>
> **THE ENTIRE VALUE OF THIS PROCESS DEPENDS ON THE PERSONA ONLY KNOWING WHAT THEY HAVE READ SO FAR.**
>
> Each persona reads ONE chapter (or makes a deliberate decision to skip or stop — see "Realistic reading paths" below), writes their mood journal entry, and OUTPUTS that entry BEFORE reading anything else. This is not a suggestion. This is not an optimisation target. This is the mechanism.
>
> **DO NOT read the whole document first and then write reactions.** DO NOT skim ahead to "understand context." DO NOT read the conclusion before reacting to the introduction. DO NOT batch-read multiple chapters before writing. The persona knows ONLY what they have read so far. Their confusion in chapter 3 is a finding. Their surprise in chapter 7 is a finding. Their wrong prediction about what comes next is a finding. If you read ahead, you destroy every one of these signals.
>
> **The urge to optimise by reading everything first will be strong. Resist it.** An agent that reads the full document and then generates "reactions" is performing theatre, not simulation. The mood journal of someone who already knows the ending is worthless. The mood journal of someone encountering each chapter fresh is the product.
>
> **Enforcement:** The three-step journal format (A → B → C) is the mechanism. Step A (expectations) must be written BEFORE reading the chapter — this forces the persona to commit to a prediction they cannot fake after reading. Step B (impressions) is written after reading. Step C (navigation decision) determines what happens next. All three steps are saved before the next chapter is opened.

### Execution model

Each persona reads the document **one chapter at a time**, writing a mood journal entry after each chapter they read. The persona does not read ahead — reactions must reflect what has been encountered so far, not retrospective knowledge. This models the real reading experience and surfaces moments where the document loses, confuses, or wins specific readers.

**Run personas independently.** Each persona reads and reacts in isolation. Cross-persona comparison happens only in the synthesis phase.

**Output one file per chapter per persona.** Use a consistent naming scheme:

```
{output-dir}/{persona-directory}/{NN}-{chapter-slug}.md
```

Example: `insights/ciso/05-the-threat.md`

### Suite orientation entry

Before reading any chapter content, each persona writes a suite orientation entry — their reaction to the scenario framing (if defined), the document menu, and their choice of where to start:

```markdown
# Mood Journal — 00: Suite Orientation

## A. Expectations

**Reaction to the consultation framing:**

{The persona has just encountered the scenario framing. How do they react
BEFORE looking at the documents themselves? Do they feel included or surprised?
Do they know why they were selected? Does the institutional branding change
how seriously they take it? Are there immediate political, commercial, or
personal implications of having received this? Who would they tell — or not
tell? Omit this field if no scenario framing is defined.}

**First impression of the suite:**

{Reaction to the document list. Is it overwhelming? Inviting? Confusing?
Does the suite map help? Does the persona know immediately where to start,
or are they uncertain?}

**What I expect to find:**

{Based on titles and descriptions alone, what does the persona think
this suite is about? What do they hope to get from it?}

## B. Impressions

{Reaction after reading the suite map. Did it clarify the reading path
or add confusion? Does the persona feel the suite is designed for them?}

## C. Next

**Time budget:** {How long would this persona realistically spend reading this suite?
"30 minutes — I have a brief to write." "90 minutes — this is directly relevant to
my team." "15 minutes — I'll scan for anything that affects me." The time budget
legitimises stopping and creates a reference point the orchestrator can use. Every
3-4 chapters, the orchestrator reminds the persona of their stated budget.}

**Opening:** {Document title and chapter}
**Why:** {In character — why this is the natural starting point for this persona}
```

### Realistic reading paths

Not every persona reads every chapter front-to-back. A Secretary-level CTO may read the executive summary and recommendations, then stop. A reporter may jump straight to the appendix with the headline number. A procurement specialist may skim until they hit evaluation criteria language and then read closely. **The reading path itself is editorial data — where personas skip, stop, jump, or lose interest reveals the document's actual accessibility and navigation quality.**

The persona has **full autonomy** over their reading path. The `Reading behaviour` field in the persona specification describes how the persona *tends* to approach documents — it is a character trait, not a prescribed route. The persona may deviate from it at any time based on what they encounter. The Step C navigation decision at the end of every journal entry creates a continuous trace of the persona's actual path through the suite.

At any point — including mid-chapter — the persona may:

- **Skip** chapters within a document ("too technical", "not relevant to my brief")
- **Jump** to a non-adjacent chapter or a different document entirely ("I want to see the recommendations first", "the practical guide looks more useful")
- **Switch** between documents mid-read ("this paper keeps referencing the companion — let me look at that")
- **Abandon and return** ("I got bored with this — let me try the practical guide instead. I can always come back.")
- **Stop** reading entirely ("I have enough for my brief", "I lost interest", "I've been reading for 45 minutes and I have a meeting")

A persona who stops proceeds directly to their final reflection (Phase 3) based on what they actually read. Their incomplete reading is a finding, not a failure. A persona who abandons a document and returns later is modelling real behaviour — the return (or failure to return) is a finding about the document's hold on its audience.

**The constraint that survives.** The persona may skip, stop, jump, abandon, and switch freely. The one thing they may NOT do is read content and then pretend they haven't. If a persona has read chapter 12, they cannot write a naive reaction to chapter 8. The cardinal rule governs what the persona has *seen*; the navigation autonomy governs what they *choose to see next*.

**No prescribed reading orders.** The orchestrator (see "Orchestrator specification" in Execution Notes) does not determine the persona's path. It presents the suite menu and table of contents, then serves whatever the persona requests. If the persona says "actually, I'm bored with this — give me the practical guide instead," the orchestrator provides the practical guide. If the persona says "I want to go back to chapter 4 of the discussion paper," the orchestrator provides it. The orchestrator is a librarian, not a teacher — it hands over what is asked for, in the order it is asked for.

### Mood journal format

Each chapter entry follows a three-step structure: **expectations before reading**, **impressions after reading**, and **navigation decision**. The expectations step is a reading discipline mechanism — writing a prediction before reading the chapter forces the persona to commit to their current state of understanding, which is impossible to do honestly after reading ahead.

#### Step A: Expectations (written BEFORE reading the chapter)

```markdown
# Mood Journal — Chapter {NN}: {Chapter Title}

## A. Expectations

**What I expect this chapter to cover:**

{Based on the title and what the persona has read so far, what do they think
this chapter will contain? This is a prediction — it may be wrong. Wrong
predictions are findings about the document's signposting and structure.}

**What I need from this chapter:**

{What would this chapter need to deliver for the persona to feel it was worth
reading? This anchors the post-reading assessment against a concrete need.}
```

Write and save this section BEFORE reading the chapter content.

#### Step B: Impressions (written AFTER reading the chapter)

```markdown
## B. Impressions

**Emotional state:** {One-line affective summary — what the persona is feeling after reading}

**Expectations vs reality:** {Did the chapter match, exceed, or disappoint the prediction
in Step A? What was surprising? An expectation mismatch is a finding about either the
document's signposting or its structure.}

**Key reactions:**

{Substantive observations in the persona's voice. What stood out, what landed, what
confused, what irritated. Include specific references to passages, examples, or claims.
This is the core of the entry — it should be detailed enough to reconstruct why this
persona had this reaction to this content.}

**Questions:**

{Questions the persona would ask at this point. These may be answered by later chapters —
that's fine. The question reflects the persona's state of understanding NOW.}

**Claims I'd challenge:**

{Any specific claims, statistics, or assertions in this chapter that the persona
finds difficult to believe, surprisingly strong, or insufficiently evidenced.
Not disagreements with recommendations — those go in Concerns. This is about
factual or analytical claims where the persona's reaction is "really?" or
"where's the evidence for that?" A chapter with no challenged claims is fine —
say "none." A chapter where multiple personas challenge the same claim has an
evidence problem. A chapter where only one persona challenges a claim that others
accept reveals an audience-specific credibility gap.}

**Concerns:**

{Worries, objections, or risks the persona sees. These may be about the document's
content, its implications, or its likely reception in the persona's world.}

**Document quality:**

{Assessment of prose, structure, examples, and presentation for this chapter.}

**The quote I'd repeat:**

{If the persona were to mention this chapter to a colleague, what single number,
phrase, or claim would they repeat? If nothing is quotable, say "nothing." When a
statistic or phrase propagates through an institution, it is often stripped of its
surrounding caveats. This field surfaces the document's most viral content per
audience — and whether that content, out of context, still communicates the
intended message.}

**Cumulative sense:**

{Running overall impression. How has the persona's view of the document evolved?
This field creates the emotional arc of the reading experience. It is impossible
to write honestly if you have read ahead — a persona who "doesn't know yet" after
chapter 2 is expressing genuine uncertainty; a persona who writes that after reading
the whole document is lying. The arc from first impression to final feeling is
itself a finding.}
```

#### Step C: Navigation decision (written AFTER Step B)

```markdown
## C. Next

**Continue reading this document?** {Yes / No — answer this before deciding what to read.
This binary forces a deliberate decision rather than defaulting to "next chapter."}

**If yes:** {Which chapter, and why}
**If no:** {Stop reading entirely / Switch to [document] / Return to [abandoned document] — and why}

**Would you forward this chapter?** {If the persona would send this chapter to someone —
who, and with what one-line annotation? If not, say "no." This tracks the document's
institutional activation energy and propagation potential.}
```

The binary continue/stop decision interrupts the sequential-processing momentum that LLM agents default to. Without it, Step C becomes a ritual affirmation of "read next chapter." The binary forces the agent to commit to continuing before choosing what to continue with — a small structural change that produces significantly more genuine navigation decisions.

The forwarding impulse field is lightweight (one line) but captures a signal the rest of the journal misses: who else in the persona's world would see this content, and how would they frame it? A chapter that every persona would forward is doing something right. A chapter that nobody would forward may be technically sound but institutionally inert.

#### Light journal option

Not every chapter provokes a strong reaction. The full eight-field template is valuable when a chapter surprises, confuses, excites, or concerns the persona. When a chapter is genuinely unremarkable — it delivered what was expected, in the expected way, with nothing to flag — the persona may use a light journal instead:

```markdown
# Mood Journal — Chapter {NN}: {Chapter Title}

## A. Expectations

**What I expect this chapter to cover:** {Prediction, as normal}
**What I need from this chapter:** {Need, as normal}

## B. Impressions

**Emotional state:** {One-line}

**Notable reactions:** {Anything worth recording. If genuinely nothing: "This chapter
delivered what the title promised. Nothing to flag." — that sentence is the entry.}

**Cumulative sense:** {Running arc — this field is never optional, even in a light journal,
because it tracks the persona's evolving relationship with the document.}

## C. Next

**Continue reading this document?** {Yes / No}
**If yes:** {Which chapter, and why}
**If no:** {Stop / Switch / Return — and why}
```

The persona's choice to use the light format is itself a finding — it signals a chapter that neither won nor lost the reader. If 9 of 13 personas use the light journal for the same chapter, that chapter is furniture: present, functional, invisible. If only 1 persona uses the light journal and the other 12 have strong reactions, that persona's indifference is an audience signal.

**The constraint:** The light journal still requires Step A before reading the chapter. The expectations gate is never optional — it is the anti-contamination mechanism regardless of how much the persona writes afterward.

### Voice guidelines

**Write in character.** The persona's voice should be distinctive and consistent. A senior executive writes differently from a junior developer. The citizen programmer uses different vocabulary from the security assessor. The policy advisor thinks in political risk terms. Don't flatten all personas into the same analytical register.

**Let the persona be wrong.** If a persona misunderstands something, that's a finding — it means the document failed to communicate clearly to that audience. Don't correct the persona in-journal. Record the misunderstanding and let it surface in the synthesis.

**Record emotional reactions.** The mood journal format exists specifically to capture affective responses that a technical review would suppress. "I felt talked about rather than talked to." "This made me defensive." "I showed this to a colleague." These are editorial signals.

**Honour the reading path.** If the persona's `Reading behaviour` says they would skip, stop, or jump — let Step C reflect that. Don't force a persona to read content their real-world counterpart would never reach. A persona who stops at chapter 4 and writes a final reflection based on four chapters of context is producing more honest editorial intelligence than one who dutifully reads all 21 chapters in a voice that would have stopped at the executive summary.

**Commit to expectations.** Step A must be written before the persona has any knowledge of the chapter's content. A wrong prediction ("I expected this to be about governance, but it was about code patterns") is a finding about the document's signposting. An agent that writes vague, hedge-everything expectations is undermining the mechanism — push for concrete predictions.

**Track the cumulative arc.** The "Cumulative sense" field should show the persona's evolving relationship with the document. A persona who starts sceptical and ends convinced tells a different story from one who starts enthusiastic and ends frustrated. The arc itself is a finding.

**React to the consultation context.** Under the scenario framing (if defined), each persona is a stakeholder who may submit a response. When a chapter triggers a submission-relevant reaction — "this recommendation is unworkable," "this is exactly what my agency needs," "I'd push back on this definition" — it should surface in Key Reactions or Concerns. Don't force it into every chapter, but don't suppress it either. The consultation framing is part of the persona's cognitive context throughout their reading, not something they only think about at the end.

### Voice re-anchoring

LLM agents simulating personas will drift toward a median analytical register over long reading sessions. By chapter 15, the CISO and the dev lead may start sounding suspiciously similar. This is a known degradation mode, not a rare edge case.

**Mechanism:** Every three chapters (or when the persona switches documents), the orchestrator re-presents the persona's core identity from their specification. This is not a prompt to change voice — it is a reminder of who this persona is, what they sound like, what they don't know, and what they are looking for. Include it as a brief preamble before the next chapter:

```
[Voice anchor — you are {Name/Role}. Your voice sounds like this:
"{Voice sample}"
Your consultation stance: "{Consultation voice}"
You do NOT know: {Blind spots — first 2-3 items}
You are trying to answer: "{Key question}"
Your emotional vocabulary includes: bored, vindicated, insulted, energised,
suspicious, relieved, impatient, impressed, defensive, confused, amused,
alarmed, indifferent, inspired, overwhelmed, curious — not just
"cautiously [adjective]."]
```

The expanded payload addresses three degradation modes simultaneously: **voice drift** (the voice sample), **knowledge leakage** (the blind spots reminder), **goal drift** (the key question), and **emotional vocabulary exhaustion** (the palette). The three-chapter interval is more aggressive than typical re-anchoring but the marginal token cost is trivial compared to the signal preservation.

**Detection:** If two personas' journal entries become difficult to distinguish — similar vocabulary, similar sentence structure, similar analytical frame — this is a finding about the implementation, not the document. Flag it in the synthesis as a process limitation. The synthesiser should note which personas maintained voice differentiation and which converged, because the converged entries are less reliable as audience-specific signals.

### Chapter mapping

Map document sections to numbered chapter files. Use the document's own structure — don't restructure for the review. If the document has front matter, abstract, table of contents, main body chapters, appendices, and references, map them all — including structural and navigational elements (title page, table of contents, colophon). Personas' reactions to these elements are findings about the document's self-presentation: does the ToC help or overwhelm? Does the front matter set expectations correctly? Does the abstract actually summarise the content?

```
00-front-matter.md
01-abstract.md
02-executive-summary.md
03-table-of-contents.md
04-introduction-and-scope.md
...
14-appendix-a-taxonomy.md
...
21-references.md
```

---

## Phase 3: Final Reflections

After completing their reading — whether that means all six documents or just the 8-page briefing — each persona writes a **final reflection** — an in-character summary document that captures their overall verdict. This is not a dispassionate summary; it is the document the persona would actually produce in their role.

### In-character output formats

The final reflection should match the persona's professional context:

| Persona type | Reflection format |
|---|---|
| Executive (VP/CTO) | Briefing memo to their superior — bottom line up front, what to do, what to watch |
| Security professional (CISO, assessor) | Risk assessment — what changed, what gaps remain, what controls are needed |
| Policy advisor | Executive brief — political risk, media angle, options for action, reputational exposure |
| Developer (junior or lead) | Personal reflection — what they'll change, what they'd tell colleagues, what they need from leadership |
| Citizen programmer | Personal reflection — am I scared? Am I helped? What will I do differently? |
| Vendor/contractor | Market assessment — threat or opportunity, competitive positioning, investment case |
| Researcher | Technical assessment — credibility, research gaps, collaboration opportunities |
| Journalist | Story assessment — is there a publishable story, what angle, what sources are needed, what's the public interest |
| Procurement specialist | Specification assessment — what can be operationalised in tender criteria today vs what needs policy maturation |
| Auditor | Control assessment — what is testable, what evidence would satisfy an audit, what gaps remain in the accountability chain |

### Overall verdict

When the persona has finished reading — whether they read everything or stopped early — they write a single capstone file that assesses what they encountered:

```
{output-dir}/{persona-directory}/overall-verdict.md
```

This is not a summary of prior journal entries. It is the persona stepping back and answering: **based on what I read, does this hold together?** A persona who read three documents writes a verdict about three documents. A persona who stopped after the executive summary writes a verdict about the executive summary — and the fact that they stopped is itself a verdict. The reflection should blend the analytical with the personal — this process captures subjective experience, and the capstone should honour that. Structure:

```markdown
# Overall Verdict — {Persona Role}

## What I read
{List every document and chapter the persona actually read, in the order they
read them. Include skips and stops. This is the persona's reading path —
it is itself a finding about the suite's navigation quality.}

## Emotional state
{How does the persona feel now, having finished? Relief? Urgency?
Overwhelm? Respect? Frustration? This is the bookend to the "Emotional state"
field in the first chapter journal. The arc from first impression to final
feeling is itself a finding.}

## How my thinking changed
{What did the persona believe before reading, and what do they believe now?
Where did the documents move them? What assumptions were challenged or
confirmed? This is the most important section — it captures the delta.}

## Main takeaways
{The 3-5 things the persona will remember and repeat to colleagues. Not a
summary of the documents — the things that stuck, that changed how they
think about the problem, that they'll bring up in meetings.}

## The suite as a whole
{Based on what the persona read: do the pieces reinforce each other or contradict?
Is there a clear reading path for this persona's audience? If the persona didn't
read the whole suite, why not — and is that a problem or a sign that the suite
routes audiences correctly?}

## Strongest piece
{Which single document or section would this persona recommend first, and why?}

## Weakest piece
{Which document or section undermines the suite or fails its audience? This may
include documents the persona chose not to read — "I didn't open it because the
title told me it wasn't for me" is a finding about positioning.}

## Recommended reading order for {persona's audience}
{If this persona were circulating the suite to colleagues, what order and which
documents would they include vs exclude?}

## What's still missing
{Having seen everything, what gap remains? What document or section would
complete the suite for this persona's needs?}

## Consultation response
{This is a select-stakeholder consultation. Would this persona respond?
If yes, what would the submission say — not a full draft, but the 3–5 key
points they would make? If no, why not — and is the silence itself a signal?
For personas with ambiguous access (e.g., the reporter, the citizen
programmer), would they even feel entitled to submit? For institutional
personas (vendor, CISO, auditor), would the response come from them
personally or from their organisation — and what's the approval chain
for that? Omit this section if no scenario framing is defined.}

## Final assessment
{One paragraph — the verdict this persona would give if asked "is this worth
my organisation's time?"}
```

### Output

```
{output-dir}/{persona-directory}/overall-verdict.md
```

---

## Phase 3b (Optional): Persona Collision Tests

After personas have written their final reflections but before the synthesis, select 2–3 pairs of personas with **opposed positions** and run a simulated corridor conversation. Give each agent the other persona's overall verdict and ask them to react.

This breaks the persona isolation that governs Phases 2–3. Isolation produces clean individual signals; collision tests reveal whether those signals survive contact with an opposing perspective. A persona whose position collapses when challenged was performing a stance, not holding a conviction. A persona who sharpens their position in response to opposition has genuine differentiation.

### Why this matters

The current process produces independent readings and counts agreements. But real consultations involve negotiation — the vendor pushes back on the CISO's recommendation, the dev lead challenges the executive's timeline, the researcher questions the auditor's proposed controls. These micro-negotiations are where institutional positions actually form. The collision test is a lightweight proxy for this dynamic.

### How to run it

For each selected pair:

1. **Select pairs that maximise tension.** Choose personas whose overall verdicts contain directly opposed conclusions about the same content. Define recommended pairings in your `config.md`.

2. **Exchange verdicts.** Give each persona the other's overall verdict (the full document, not a summary). Do NOT give them the other's journal entries — they see the conclusion, not the journey.

3. **Ask for a reaction.** Each persona writes a short response (~300 words) in character:

```markdown
# Collision Test — {Persona A} reacts to {Persona B}'s verdict

**Initial reaction:** {Emotional response to reading the other persona's position}

**Where I agree:** {Points of genuine agreement — not politeness, but substance}

**Where I push back:** {The strongest objection to the other persona's position,
in the persona's own voice and from their own professional context}

**What this changes:** {Does reading the other verdict change the persona's own
position? If yes, how? If no, why not? "I hadn't considered that" is a finding.
"I considered it and I'm right" is also a finding.}
```

4. **Save the collision test output.** One file per collision per persona:

```
{output-dir}/{persona-directory}/collision-{other-persona}.md
```

### What the collision tests tell you

- **If both personas moderate toward agreement:** The initial opposition was shallow — likely model-prior fluctuation rather than genuine audience-driven divergence. Downweight the disagreement in the synthesis.
- **If both personas sharpen their positions:** The opposition is structural — it reflects genuinely different audience needs that the document cannot simultaneously satisfy. This is a high-value finding for the synthesis, because it identifies where the document must choose its audience or create derivative documents for each.
- **If one persona collapses and the other holds:** The collapsing persona's original position was weakly held — their journal entries may be less reliable as audience signals. The holding persona's position is robust.
- **If the exchange produces new arguments neither persona raised in isolation:** The collision has generated emergent insight that isolated reading could not. These new arguments should be flagged in the synthesis as collision-emergent findings.

### When to skip

Skip if the panel is small (fewer than 6 personas) or if the overall verdicts show no significant disagreements. Collision tests are only valuable when there is genuine tension to test. Running collisions between personas who agree is a waste of tokens.

### Output

```
{output-dir}/{persona-directory}/collision-{other-persona}.md
```

---

## Phase 4: Cross-Panel Synthesis

The synthesis aggregates findings across all personas into actionable editorial recommendations. It is written by the session operator (human or agent), not by any persona. It should be analytical, not narrative.

### Synthesis structure

```markdown
# Reader Panel Synthesis

## 1. Executive Summary
{2-3 paragraph overview: headline finding, secondary finding, panel composition}

## 2. The Panel
{Table: persona, role, lens, reading path taken, overall verdict.
The "reading path taken" column shows what each persona actually read — e.g.,
"Suite map → Understanding AI Code Risk → Discussion Paper chapters 1–4, stopped"
or "Suite map → jumped to Practical Guide → read all → switched to Discussion Paper
§5–6." This column is itself a finding — it shows which audiences the suite reaches,
how they navigate it, and where it loses them.}

## 3. Universal Findings

### What everyone agreed on (positive)
{Themes that appeared in a majority of personas — these are the document's strengths.
State the exact count: "X of Y personas flagged this."}

### What everyone agreed on (critical)
{Themes that appeared in a majority of personas — these are real problems.
State the exact count.}

## 4. Reading Path Analysis
{How did different personas navigate the document? Which chapters did multiple
personas skip? Where did personas stop? Which chapters attracted jumps from
non-adjacent readers? A chapter that 8 of 13 personas skipped is a structural
problem. A chapter that 3 personas jumped to from the table of contents is doing
something right. Map the reading paths to reveal the document's actual navigation
quality vs its intended reading order.

Explicitly list any chapters that NO persona read — universally-skipped content
is either mispositioned, poorly titled, or genuinely unnecessary. This is a
sharp editorial signal.}

## 5. The Strongest Moments
{Specific passages, examples, or arguments that multiple personas flagged as
the document's most effective content. Include persona quotes.}

## 6. The Weakest Moments
{Specific passages, sections, or arguments that multiple personas flagged as
problematic. Include the persona and the reason.}

## 7. Actionable Editorial Findings
{Numbered list, ordered by how many personas raised the issue. Each item:
what the finding is, how many personas raised it, what to do about it.}

## 8. Audience-Specific Gaps
{For each audience that the document intends to serve: what's missing for that
audience? What do they need that the document doesn't provide? This section
directly generates derivative document specifications.}

## 9. Per-Document Verdict
{For each document in the suite: how many personas read it, how many skipped it,
what landed, what didn't. A document that 10 of 13 personas skipped is either
well-targeted at a niche audience or poorly positioned. A document that everyone
read is load-bearing.}

## 10. Consultation Response Forecast
{Based on the scenario framing, what does the panel predict about the
consultation's reception? Group by:
- **Who would respond** — which personas (and by extension, which real
  stakeholder groups) would submit a formal response? Who would engage
  informally? Who would stay silent?
- **What they would say** — aggregate the key points from each persona's
  "Consultation response" section in their overall verdict. Where do
  multiple stakeholders converge? Where do they diverge?
- **What the authors should prepare for** — hostile submissions, enthusiastic
  endorsements, requests for extension, industry coalition responses, media
  inquiries, leadership escalations. Which of these does the document in its
  current form invite, and which could be mitigated by revision before
  public release?
- **The silence problem** — which stakeholders would NOT respond, and what
  does that absence mean? A consultation where no junior developers or
  citizen programmers respond is a signal about who the document is actually
  reaching.
Omit this section if no scenario framing is defined.}

## 11. Surprising Findings
{Reactions that the document's authors probably didn't anticipate. Commercial
opportunities, political dynamics, emotional responses, institutional concerns.}

## 12. Recommended Actions
{Prioritised by breadth of panel support and feasibility. Group into priority
tiers. Each action: what it is, who it's for, which personas support it.}

## 13. Panel Gaps and Suggested Personas

### Gaps identified
{For each gap: which audience is missing from the panel, what evidence
from the journals revealed the gap (cite specific journal entries or
verdict passages with direct quotes), and why this audience would have
produced different findings.

If no significant gaps were identified, state that explicitly — a panel
with good coverage is itself a finding about the panel design.}

### Suggested personas
{For each suggested persona:}

**Name/Role:** {Job title and position}
**Lens:** {What they optimise for}
**Background:** {2-3 sentences}
**Key question:** {The question they'd be trying to answer}
**Blind spots:** {What they don't know or care about}
**Evidence:** {Which journal entries or verdict passages revealed the need
for this persona — direct quotes preferred}

### What these suggestions do NOT include
Reading behaviour, voice samples, and verdict predictions are omitted
deliberately. The synthesiser has read all panel journals and therefore
knows the full document content indirectly. Predicting how a new persona
would navigate or react to specific chapters would project knowledge
the persona would not have on a cold read. These fields must be written
by someone who has not seen the panel results — either the user or a
persona-designer agent given only the documents.
```

### Synthesis methodology

#### Who writes the synthesis

The synthesis is written by the session operator — a human or a dedicated synthesis agent that is distinct from the persona agents. The synthesiser has different skills from a persona agent: cross-comparison, pattern recognition, editorial judgment, and the ability to hold all reading paths in working memory simultaneously. If using an agent, provide it with all persona journal files and overall verdicts as input — it reads across personas, not within one.

#### Input preparation

Before writing the synthesis, collate the persona outputs into cross-cutting views:

1. **Reading path map.** For each persona, extract the sequence of Step C navigation decisions into a single-line path: `Suite map → gcbd-lite → Discussion Paper §1–4 → stopped`. This map is the primary input to the Reading Path Analysis section.
2. **Per-chapter cross-persona index.** For each chapter that at least one persona read, list which personas read it, which skipped it, and pull their Key Reactions and Emotional State fields. This enables the "which chapters did multiple personas skip?" analysis without requiring the synthesiser to re-read every journal in full.
3. **Keyword/theme extraction.** Scan all journal entries for recurring concerns, questions, and emotional reactions. Group by theme, not by persona. The themes become candidate findings; the persona count determines severity.

For a large panel, this collation step produces three working documents that the synthesiser operates on. Without this step, the synthesiser is searching hundreds of journal files by memory — it will miss patterns and produce a weaker synthesis.

#### Analytical passes

The synthesis is produced in three passes, not one:

1. **Reading path analysis.** Map all paths. Identify convergence points (chapters everyone read), divergence points (chapters that split the panel), and dead zones (chapters nobody reached). This pass writes sections 2, 4, and 9 of the synthesis template.
2. **Per-chapter cross-persona comparison.** For each chapter, compare reactions across the personas who read it. Where do they agree? Where do they diverge? Disagreement between personas is as valuable as agreement — it reveals where the document means different things to different audiences. This pass writes sections 3, 5, 6, and 7.
3. **Thematic aggregation.** Step back from chapters and identify document-level themes: vocabulary burden, length, structural navigation, consultation response patterns, derivative document demand. This pass writes sections 1, 8, 10, 11, 12, and 13.

#### Methodology rules

**Count, don't summarise.** "9 of 13 personas flagged length as a barrier" is more useful than "many personas found it too long." The count is the evidence.

**Quote, don't paraphrase.** The persona's exact words carry more weight than a summary. "I exist in the gap between 'this person is creating risk' and 'here's how this person can reduce that risk'" is a better finding than "the citizen programmer felt unsupported."

**Separate universal from audience-specific.** A finding raised by all personas is an editorial problem. A finding raised by one persona is an audience insight. Both are valuable but they require different responses.

**Grade every finding by epistemic confidence.** Not all findings have equal reliability. Assign each finding to one of three tiers:

- **Tier 1 — Text-surface findings** (strongest). Findings about vocabulary burden, length, structural navigation, signposting failures, expectation mismatches. These correlate with measurable properties of the text itself. If the simulation says the vocabulary is a barrier, the vocabulary is probably a barrier.
- **Tier 2 — Affective findings** (moderate). Findings about emotional reactions, engagement, frustration, feeling excluded or addressed. Treat these as the *possibility space* of reader emotions — the range the text could plausibly produce — rather than predictions of likely reactions. LLM personas are more articulate about confusion than real readers, who simply stop reading.
- **Tier 3 — Institutional and commercial findings** (weakest). Findings about consultation dynamics, commercial impact, political risk, leadership exposure, compliance costs. These depend on institutional knowledge the model approximates from training data, not lived experience. Treat as prompts for human judgment ("have we thought about how vendors might react?"), never as predictions.

The synthesis must not present Tier 3 findings with the same confidence as Tier 1. Every finding in sections 3–12 should carry its tier tag. A synthesis where most "Recommended Actions" are driven by Tier 3 findings should trigger a human review before those actions are pursued.

**Test for convergent reasons, not just convergent conclusions.** When "X of Y personas flagged Z," the synthesiser must check: are the reasons interchangeable? If you could swap the vendor's justification into the CISO's entry and it would still read naturally, the finding is likely a single model prior expressed in different vocabulary, not independent audience signals. Genuine cross-persona convergence produces different *reasons* for the same conclusion — the vendor says "too long because I need to cost this by Q3" and the citizen programmer says "too long because I lost the thread after page 8." If the reasons are structurally identical, downweight the count.

**Let the synthesis generate derivative document specs.** When multiple personas independently request the same kind of document (an executive brief, a practical guide, a getting-started document), the synthesis should capture that as a concrete specification: audience, length, content requirements, what to include, what to exclude. These specifications become inputs to Phase 5.

#### Quality gate

The synthesis is complete when:

- Every finding in sections 3–7 cites at least two persona journal entries with direct quotes
- The reading path map in section 2 accounts for all personas (including those who stopped early)
- Section 10 (Consultation Response Forecast) includes at least one entry for each of the four sub-categories (who would respond, what they'd say, what to prepare for, the silence problem) — or is omitted entirely if no scenario framing was defined
- Section 12 (Recommended Actions) groups actions into at most three priority tiers, each with a feasibility note
- Section 13 (Panel Gaps) persona suggestions include only identity fields (Name/Role, Lens, Background, Key question, Blind spots) and evidence citations — no behavioural predictions, no voice samples, no verdict predictions
- Section 4 (Reading Path Analysis) explicitly lists universally-skipped chapters (or states that none were universally skipped)

A synthesis that meets the template structure but fails the quality gate is a draft, not a deliverable.

### Output

```
{output-dir}/00-reader-panel-synthesis.md
```

---

## Phase 5: Derivative Document Development

The synthesis will almost certainly identify demand for derivative documents — executive briefs, practical guides, quick-starts, or restructured versions of existing content. Phase 5 closes the loop: personas generate requirements (via Phases 2–4), specifications are extracted (from the synthesis), drafts are produced, and the same personas validate them. This is not an afterthought — a document suite that discovers it needs derivative documents and doesn't produce them has done expensive research with no follow-through.

**When to skip Phase 5:** Only if the synthesis identifies no derivative document demand, which is unlikely for any document suite complex enough to warrant a 7+ persona panel. Even then, consider running the outline review step against existing documents the synthesis flagged as weak — the review format works for revision briefs, not just new documents.

### Process

1. **Extract specifications** from the synthesis (section 8: Audience-Specific Gaps). Each gap is a candidate derivative document. For each candidate, capture: target audience, purpose, length constraint, what to include, what to exclude, and which existing content it draws from.
2. **Prioritise.** Not every gap needs a new document. Some can be addressed by revising an existing document. Some are nice-to-have. Prioritise by: how many personas independently flagged the need (count from the synthesis), how central the audience is to the document's purpose, and whether the gap blocks adoption or merely slows it.
3. **Produce outlines** for the top-priority derivative documents, informed by the persona reactions. Each outline should state its audience, purpose, structure, and the specific persona feedback that motivated it.
4. **Review outlines** with the 2–3 most relevant personas (target audience + one adjacent). Use the outline review format below.
5. **Produce drafts** from reviewed outlines.
6. **Final sign-off review** with the same personas — "would you circulate this?" Use the sign-off review format below.

### Outline review format

The outline review uses the mood journal format but applied to a structural outline rather than prose. The persona reacts to what is proposed, what is missing, and what is unnecessary for their audience:

```markdown
# Outline Review — {Derivative Title}

**Emotional state:** {How the persona feels about this derivative being proposed for their audience}

**Key reactions:**

{Does the outline address what this persona actually needs? Does it include content
that would be irrelevant or harmful for their audience? Does the structure match
how this persona would actually consume the document?}

**Questions:**

{What does the persona need to see in the draft that the outline doesn't yet promise?}

**Concerns:**

{Risks: wrong audience assumptions, missing content, inappropriate tone, scope creep}

**Document quality:**

{Assessment of the outline's structure, completeness, and feasibility}

**Verdict:**

{Proceed / Revise / Reject — and what specific changes are needed before drafting}
```

### Sign-off review format

After the draft is produced, the reviewing personas assess whether it is ready to circulate:

```markdown
# Sign-Off Review — {Derivative Title}

**Emotional state:** {How the persona feels reading the finished draft}

**Key reactions:**

{Does the draft deliver on the outline's promise? Does it work as a standalone
document for the persona's audience? Would the persona actually circulate this?}

**Questions:**

{Anything still unclear or missing that would prevent circulation}

**Concerns:**

{Risks: factual errors, tone problems, missing context, audience mismatch}

**Document quality:**

{Assessment of prose, structure, examples, and presentation}

**Verdict:**

{Circulate as-is / Circulate with minor edits / Needs revision — with specifics}
```

### Output

```
{output-dir}/{derivative-slug}-outline.md                  # Outline informed by panel
{output-dir}/{persona-directory}/{derivative-slug}-outline-review.md   # Persona review of outline
{output-dir}/{persona-directory}/{derivative-slug}-signoff.md          # Sign-off assessment
```

---

## Execution Notes

### How to invoke

Feed the process file (`process.md`) as the task specification and the config file (`config.md`) as the panel configuration. The persona specification (role, lens, background, key question, blind spots, voice sample, consultation voice, reading behaviour) becomes the agent's system prompt. Provide the document suite menu and table of contents upfront so the persona can make informed navigation decisions. Then the orchestrator serves chapters one at a time based on the persona's requests. The mood journal format is the output constraint.

### Orchestrator specification

The orchestrator is the runtime that enforces the no-read-ahead discipline. Without it, the process depends entirely on prompt compliance, which is unreliable. The orchestrator sits between the persona agent and the document content — it is a librarian, not a teacher.

#### Control flow

```
1. INITIALISE
   a. Coordinator creates team, spawns persona-reader agents as teammates.
      Each agent receives via spawn prompt: persona specification,
      scenario framing, path to process.md, document suite menus
      (titles and chapter numbers only), and output directory path
   b. Agent reads process.md Phases 2 and 3, reads persona spec
   c. Agent writes suite orientation entry (00-suite-orientation.md)
   d. Agent goes idle — turn output declares first chapter request

2. CHAPTER LOOP
   a. Agent writes Step A (expectations) and saves to journal file
   b. Agent goes idle — turn output states Step A path and chapter
      request
   c. Coordinator reads the journal file to verify Step A exists
      (the Step A gate), looks up hashed filename in manifest,
      sends file path via SendMessage
   d. Agent reads chapter file, writes Steps B and C, saves
      complete journal entry
   e. Agent goes idle — turn output states next request or "done"
   f. Coordinator reads the navigation decision from turn output:
      - If "stop reading" → go to step 5
      - If requesting a chapter in a different document →
        send that document's table of contents (triggers
        voice re-anchoring), return to step 2a
      - If requesting the next/a specific chapter in the current
        document → return to step 2a
      - If "abandon current document, switch to X" → note the
        abandonment, send X's table of contents if needed
        (triggers voice re-anchoring), return to step 2a
      - If "return to previously abandoned document" → send a
        reminder of where they left off, return to step 2a

3. VOICE RE-ANCHORING (integrated into step 2)
   Every 3 chapters (or on document switch), the coordinator
   includes the expanded voice anchor in the next SendMessage
   (voice sample, consultation voice, blind spots, key question,
   emotional vocabulary palette). See "Voice re-anchoring" in
   Phase 2 for the full payload format.

4. CONTEXT MANAGEMENT (integrated into step 2)
   When the agent has read 15+ chapters, the coordinator
   includes the mood journal carry-forward in the next
   SendMessage. See "Context management" below.

5. FINAL REFLECTION
   a. Agent writes overall-verdict.md and saves it
   b. Agent goes idle — turn output says "done" with verdict path
   c. Coordinator verifies verdict exists, sends shutdown request
   d. Agent approves shutdown, session ends
```

#### What the coordinator must NOT do

- **Determine the reading path.** The agent chooses freely. The coordinator serves what is requested.
- **Override navigation decisions.** If the agent wants to stop, switch, skip, or return — honour it.
- **Provide content the agent hasn't requested.** No "here's the next chapter" without the agent asking for it. No "you might also want to read..." suggestions.
- **Provide assembled documents.** If the agent requests "the discussion paper," provide the table of contents and ask which chapter. Never provide the full assembled file — this is the structural enforcement of the no-read-ahead discipline.
- **Batch chapters.** One chapter per cycle. The Step A → content → Steps B+C sequence is the atomic unit.

#### The Step A gate

The Step A expectations gate is the strongest anti-contamination mechanism. It works because:

1. The agent must commit to a concrete prediction about what the chapter contains *before* seeing the content
2. An agent that has already read ahead cannot write a genuine prediction — it will produce suspiciously accurate expectations or artificially vague hedges
3. Both failure modes are detectable by the orchestrator or the synthesiser

If an agent's Step A predictions are consistently accurate in ways that exceed what the chapter title and prior context could support, flag this as a contamination signal. The journal entries from that point forward are unreliable.

### Running as an agent workflow

The panel runs as a single coordinator managing N persona-reader teammates. Each persona is spawned as a teammate via TeamCreate; the coordinator communicates with agents asynchronously through turn output (agent to coordinator) and SendMessage (coordinator to agent). Agents go idle after producing turn output; the coordinator wakes them with the next SendMessage.

**Parallelism:** All persona agents run concurrently as teammates of the same coordinator. Note that personas with non-linear reading paths (jumps, early stops, document switches) will finish at different times — this is expected and desirable. A persona who reads 4 chapters and stops is not behind; they are done.

**Reading path autonomy:** The agent has full autonomy over its reading path. The `Reading behaviour` field describes a tendency, not a route. The coordinator provides the table of contents and suite menu at spawn, then sends file paths for whatever the agent requests. The agent may change its mind at any time — including mid-session reversals ("actually, I want to go back to the discussion paper"). Do not override the agent's navigation choices — the path is data.

**Context management:** Each persona agent accumulates context across chapters. For documents under ~15 chapters, the full conversation history is sufficient. For longer sessions (15+ chapters or multi-document reads), use a **mood journal carry-forward** to manage context while preserving the emotional arc:

The carry-forward provides the agent with:
- Its **last 3 journal entries in full** (preserving recent voice, affect, and continuity)
- A **structured arc summary** of all earlier entries (preserving the emotional trajectory without the full text):

```markdown
## Reading arc so far

| Chapter | Emotional state | Cumulative sense | Key concern |
|---|---|---|---|
| 01: Front Matter | Curious but wary | "This looks serious but I'm not sure it's for me" | Vocabulary load |
| 02: Executive Summary | Engaged | "They understand the problem" | No practical guidance yet |
| 03: Table of Contents | Overwhelmed | "This is enormous — where do I start?" | Length |
| ... | ... | ... | ... |
```

This preserves the affective trajectory (which is the mechanism for the cumulative arc) while freeing context for new chapter content. A rolling summary that captures only facts but loses affect undermines the process — the arc is emotional, not informational.

> **🚨 MANDATORY NO-READ-AHEAD DISCIPLINE FOR AGENT IMPLEMENTATIONS 🚨**
>
> When implementing this as an agent workflow, the single most important constraint is: **each agent must read ONE chapter, write and save the mood journal file, and only THEN decide what to read next.**
>
> Agents will attempt to optimise by reading the entire document into context first and then generating "reactions." THIS DESTROYS THE PROCESS. The value is in the naive, chapter-by-chapter encounter. An agent that knows how the document ends cannot authentically react to how it begins.
>
> The agent MAY skip chapters, jump to non-adjacent chapters, switch to companion documents, abandon a document and return later, or stop reading entirely — these are all legitimate reading behaviours. The agent may NOT read content without journalling it, or read ahead "for context" without committing to a journal entry for what it read.
>
> The coordinator specification above is the enforcement mechanism. The Step A gate is the detection mechanism. Together they make the no-read-ahead discipline structural rather than aspirational.
>
> **Implementation patterns that enforce this:**
> - The coordinator sends chapter file paths one at a time, only after Step A is written and verified
> - The Step A gate makes read-ahead contamination structurally detectable
> - The agent's turn output navigation decision determines what the coordinator sends next
> - The agent is never given assembled documents — only individual chapter files
> - Voice re-anchoring every 3 chapters maintains persona differentiation
>
> **Implementation patterns that violate this (DO NOT USE):**
> - Giving the agent the full document and asking it to write all journals at once
> - Asking the agent to "read the document and then write chapter-by-chapter reactions"
> - Batching multiple chapters per read cycle
> - Allowing the agent to "skim for context" before starting its reading path
> - Overriding the agent's decision to skip, stop, switch, or return — forcing complete reads destroys the navigation signal
> - Pre-loading chapter content into the agent's context "for efficiency"

**Output cadence:** Agents go idle after each chapter cycle — read one chapter (or record a skip/stop/switch decision), write and save the entry, then go idle with turn output stating the next request. The coordinator wakes the agent via SendMessage for the next cycle. Never read the full document first and write reactions afterward. The reading path is the product; batch-reading-then-writing is theatre.

### Running manually

The process also works as a manual exercise: a human reads the document multiple times, each time adopting a different persona's lens and recording reactions. This is slower but produces higher-quality voice differentiation. The format and synthesis methodology are the same. The orchestrator is implicit — the human controls their own reading path.

### Variant: Delegation chain simulation

The standard process models individual readers engaging directly with documents. In real institutions, documents are mediated — they arrive at the organisation's front door and are transformed, filtered, summarised, and delegated before reaching the decision-maker. A senior executive does not read a 120-page paper; they read a one-page brief written by an analyst who read the paper. The CISO does not form a position from scratch; they read their security team's assessment. The policy advisor reads an executive brief, not the source documents.

The delegation chain simulation models this institutional mediation. It is more expensive than the standard process (it requires multiple sequential agents per institutional chain) but produces a finding the standard process cannot: **what survives institutional translation**.

#### How to run it

Select 2–3 institutional chains that represent the document's primary decision paths. Define the chains in your `config.md` (see "Delegation chain examples").

Each chain follows the same pattern:

1. **First reader** reads the document suite using the standard persona process (full reading with mood journals)
2. **First reader produces an output document** appropriate to their institutional role (brief, impact assessment, executive summary)
3. **Next reader** reads the output document, NOT the source documents. They amend, annotate, or transform it for the next level.
4. Repeat until the chain reaches the decision-maker.

At each stage, compare the input to the output:

- **What survived?** Which claims, recommendations, or statistics made it through? The survival rate tells you what the document actually communicates through institutional channels.
- **What was added?** Each intermediary adds framing: "this is important because..." or "this is not relevant to us because..." These additions reveal how the document is being positioned within institutional priorities — and whether that positioning matches the author's intent.
- **What was distorted?** The analyst's understanding of the document's argument, compressed to three bullet points, may misrepresent the nuance. If the decision-maker acts on a distorted brief, the document has an institutional communication failure that no amount of editorial polish can fix.
- **What was lost?** Concerns from one audience may never appear in the brief because the intermediary does not consider that audience relevant. These losses are the most important finding: they reveal the gap between what the document contains and what the institution receives.

#### Output

Each stage produces one document:
```
{output-dir}/delegation/{chain-name}/{NN}-{role}-{output-type}.md
```

The delegation chain comparison (input vs output at each stage) becomes an additional section in the synthesis or a companion analysis document.

#### When to use

The delegation chain is most valuable when:
- The document is intended to influence institutional decision-makers who will not read it directly
- The document has implications that cross institutional boundaries (policy, security, procurement, technical)
- You suspect the document's key arguments will not survive institutional compression

It is less valuable for documents that reach their audience directly (e.g., a practical guide for individual practitioners) or for audiences that read source material as part of their job (e.g., auditors, assessment officers).

### Adaptation for other documents

The process is document-agnostic. To adapt:

1. **Write a new `config.md`** for the new document's audience. The personas from one project won't apply to another — a security policy panel differs from a developer tools RFC panel or a clinical protocol panel. The design principles (span the decision chain, vary technical depth, include institutional perspectives, define blind spots) are universal.
2. **Define the document suite** in the config. List all documents the personas can navigate between, with descriptions and chapter counts. Even a single-document review benefits from this framing — the persona can still choose to stop or skip.
3. **Define the scenario framing** in the config (or omit it). If the document has an institutional context — who published it, what status it has, how the reader received it — define it. If omitted, remove the consultation-related sections from the verdict and synthesis templates.
4. **Remap chapters** to each document's structure.
5. **Keep the mood journal format unchanged.** The three-step structure (A: expectations, B: impressions, C: navigation decision), the full and light journal variants, and the impression fields work for any document type.
6. **Keep the synthesis structure unchanged.** The synthesis methodology (count, quote, separate universal from audience-specific, three analytical passes, quality gate) is document-agnostic.

### Calibration and validation

Simulated personas are approximations, not substitutes. The process produces confident-sounding outputs but has no built-in validation against real audience behaviour. This is an epistemological limitation, not a bug — the process is designed to surface editorial intelligence cheaply and early, before the expense of real audience testing.

**To calibrate:** If you have access to real audience feedback on a previous version of the document (or a similar document), run the panel process against that version first. Compare simulated reactions to actual reactions. Where they diverge, you learn the simulation's blind spots — which audience types it approximates well and which it doesn't. Even a single calibration run (one document, one comparison) substantially improves confidence in subsequent runs.

**To validate findings:** Treat panel findings as hypotheses, not conclusions. "9 of 13 personas flagged length as a barrier" is a strong hypothesis that length is a problem for real readers — but it is still a hypothesis. Before making major structural changes based on panel findings alone, seek at least one confirming signal from a real reader in the relevant audience.

**Known simulation limitations:**
- LLM personas tend to be more patient, more thorough, and more analytical than real readers. Real executives stop reading sooner. Real junior developers skim more aggressively. The panel will over-read compared to real audiences.
- LLM personas have difficulty sustaining emotional responses across long reading sessions. Real readers who become frustrated at chapter 5 may carry that frustration to chapter 15. LLM personas tend to reset to neutral.
- LLM personas cannot simulate network effects — a real vendor who talks to other vendors before responding behaves differently from an isolated simulated vendor. See "What this process does NOT do" below.

### Process checksum manifest

The process produces a large, structured output that looks rigorous and is almost impossible to audit without a manifest. At the end of every panel run, the orchestrator produces a machine-readable manifest that travels with the synthesis:

```yaml
# Reader Panel Process Manifest
panel:
  designed: {N}
  executed: {N}                    # If < designed, list omitted personas and why
  control_persona: {name}
  control_predictions_matched: true  # or false, with details

personas:
  - name: {name}
    chapters_read: {N}
    documents_touched: {N}
    light_journals: {N}
    full_journals: {N}
    stopped_reason: persona_decision  # or operator_terminated, agent_failure
    voice_anchors_applied: {N}
    carry_forwards_used: {N}
  # ... one entry per persona

process:
  total_journal_entries: {N}
  total_light_journals: {N}
  voice_anchors_applied: {N}
  carry_forwards_used: {N}
  step_a_contamination_flags: {N}

synthesis:
  synthesiser: human | agent        # Who wrote the synthesis
  passes_completed: 3
  quality_gate: passed | failed
  findings_by_tier:
    tier_1_text_surface: {N}
    tier_2_affective: {N}
    tier_3_institutional: {N}
```

This manifest makes completeness auditable at a glance. A reader of the synthesis can see whether all designed personas were run, whether any were terminated by the operator rather than by persona decision, how many carry-forwards were used (and therefore which personas' late-session entries may be affected by compression), and whether the quality gate was met.

The manifest should be included as an appendix to the synthesis document or as a companion file (`{output-dir}/00-process-manifest.yaml`).

### What this process does NOT do

- **Technical correctness review.** Use the expert panel review (`expert-panel-review.md`) for that.
- **Copy editing.** The mood journals may note prose quality but the process is not designed to produce line-level edits.
- **Approval or clearance.** Agreement among simulated personas is not evidence that a document is ready for publication. It is evidence about how the document will be received.
- **Replace real audience testing.** Simulated personas approximate real reader reactions but cannot substitute for actual feedback from the people the document is written for. See "Calibration and validation" above.
- **Model network dynamics.** Real consultations involve coordination — the vendor calls other vendors, the CISO asks the dev lead for a technical opinion, the policy advisor calls the executive sponsor. This process explicitly isolates personas, which produces clean individual signals but loses network effects. A real consultation where the vendor community coordinates a joint submission behaves differently from independent vendor responses. The synthesis can speculate about network dynamics (and should, in the Consultation Response Forecast section), but it cannot simulate them.
