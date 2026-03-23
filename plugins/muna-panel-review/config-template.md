# Reader Panel Configuration — Config File Reference

This document specifies the configuration file format for a reader panel review. The config file defines your document suite, scenario framing, persona specifications, and panel configuration options. It is written once per project and used alongside the reusable methodology in `process.md`.

For a fully worked example with 6 personas, scenario framing, voice samples, collision pairings, and delegation chains, see `config.md` in this directory.

---

## Document Suite

Define the documents your panel will read. Present this as a table with the following columns:

| # | Document | Description | Chapters | Source files |
|---|---|---|---|---|
| 0 | **Document Suite Map** | One-page reading path guide — routes personas to the right starting document | 1 | `suite-map.md` |
| 1 | **Project Architecture Overview** | System design decisions, component relationships, deployment model | 8 | `arch-[0-9][0-9]-*.md` |
| 2 | **API Reference Guide** | Endpoint specifications, authentication, rate limits, error handling | 12 | `api-[0-9][0-9]-*.md` |
| 3 | **Getting Started Guide** | Quick-start for new developers — setup, first integration, common patterns | 1 | `getting-started.md` |
| 4 | **Migration Planning Brief** | Executive summary of migration path, timelines, risk assessment | 1 | `migration-brief.md` |

### Chapter-to-file mapping

Chapter numbers map to source filenames by convention. For multi-chapter documents, use a numeric prefix in the filename that corresponds to the chapter number:

- Chapter 3 of the architecture doc maps to `arch-03-*.md`
- Chapter 11 of the API reference maps to `api-11-*.md`
- Single-chapter documents use a single file (e.g., `getting-started.md`)

For documents split across multiple parts, extend the convention with a part prefix: `ref-01-[0-9][0-9]-*.md` for part 1 chapters, `ref-02-[0-9][0-9]-*.md` for part 2 chapters.

### Suite map (optional)

Document 0 is the suite map — a one-page routing document that helps personas choose where to start. It lists each document with a one-line description and suggests entry points by role type. The suite map is an orientation aid, not a prescription. Its presence is optional, but when included it should be listed as document 0.

### Presentation order

At the start of each persona's session, present content in this order:

1. **Scenario framing** (if defined) — the institutional context
2. **Suite map** (if present) — the routing document
3. **Document table** — the full list of available documents

The persona encounters the institutional context before seeing the documents — the same way a real reader would encounter a covering email or memo before opening the attachments. See `process.md` Phase 2 for the suite orientation journal template.

---

## Scenario Framing (optional)

Scenario framing establishes the institutional context in which personas encounter the document suite. It answers: who published this, what status does it have, and how did the reader receive it?

### Framing statement

Present the framing as a blockquote that is read to each persona before they see the document suite:

> **Personas should be told the following at the start of their reading session, before they see the suite map:**
>
> *The document suite you are about to read has been issued as **[version and status]** by [issuing authority]. [Distribution channel and scope]. You have received it because [reason for access]. [Expected response or engagement].*

The framing statement should establish:

- **Issuing authority** — who published or authored the documents, and what institutional weight that carries
- **Version and status** — draft, final, consultation, internal-only, etc.
- **Distribution channel** — how the reader received it (official channels, forwarded, public, leaked)
- **Expected engagement** — whether feedback is expected, optional, or irrelevant

Follow the framing statement with a prose explanation of *why* each element was chosen and what effect it has on reader behaviour. This explanation is for the config author, not for personas.

### Framing effects table

Document how different persona types receive the documents differently. This table maps persona categories to their access channel and the resulting framing effect:

| Persona type | How they received it | Framing effect |
|---|---|---|
| **Internal team (e.g., lead engineer, product manager)** | Direct distribution — they authored or commissioned the work | They own the document. Their stance is "does this represent our position well?" |
| **Adjacent team (e.g., platform engineer, QA lead)** | Internal channels — forwarded by their manager or a shared channel | They are affected but did not author it. Their question is "what does this mean for my team?" |
| **External stakeholder (e.g., partner engineer, client architect)** | Formal sharing — sent as part of a partnership or integration agreement | They have been given access deliberately. Their question is "what do we need to change?" |
| **End user (e.g., junior developer, new hire)** | Second-hand — found on a wiki, forwarded by a colleague | They may not be sure this is intended for them. Their engagement is personal, not institutional. |
| **Leadership (e.g., VP Engineering, CTO)** | Flagged by a direct report or included in a review package | They will not read the full suite. Their question is "do I need to act on this?" |

Adapt the persona types, channels, and effects to your project. The table should cover every persona in your panel.

### If scenario framing is omitted

If you choose not to define scenario framing:

- Remove the `Consultation voice` field from all persona specifications
- Remove the "Consultation response" section from the overall verdict template (see `process.md` Phase 2)
- Remove the "Consultation Response Forecast" section from the cross-panel synthesis (see `process.md` Phase 4)

---

## Personas

Define each persona as a block with the following fields. See `process.md` Phase 1 for panel design principles (spanning the decision chain, varying technical depth, defining blind spots).

### Required fields

| Field | Description |
|---|---|
| **Name/Role** | Job title and organisational position |
| **Lens** | What this persona optimises for — 2-3 priorities that filter their reading |
| **Background** | 2-3 sentences establishing expertise, experience, and current situation |
| **Reading behaviour** | How they will actually approach the document — not how they *should* read it, but how they *will*. Will they read front-to-back? Skip to examples? Bail after the introduction? Jump to the section that affects their team? |
| **Key question** | The single question this persona is trying to answer by reading the document |
| **Blind spots** | What this persona does NOT know, cannot access, or does not care about. This field is the primary mechanism for preventing persona bleed in LLM execution — without it, all personas converge toward the same well-informed analytical voice |
| **Voice sample** | One sentence in character, how they talk to a peer. This anchors the persona's register for the entire reading |
| **Directory name** | Slug used for the persona's output directory (e.g., `lead-eng`, `junior-dev`, `client-arch`) |

### Optional fields

| Field | Description |
|---|---|
| **Consultation voice** | One sentence showing how this persona reacts to the scenario framing — receiving this document through their specific channel. Only include when scenario framing is defined |
| **Misconception** | A specific wrong belief this persona carries into their reading. Only used for the unreliable narrator persona (see Panel Configuration below) |

### Example persona

```
### Persona 3: Platform Engineer

**Name/Role:** Senior platform engineer, infrastructure team
**Lens:** System reliability, deployment complexity, operational overhead
**Background:** Eight years in infrastructure, mass deployments and monitoring. Joined
this organisation 18 months ago. Responsible for the CI/CD pipeline and production
environment. Has opinions about what belongs in application code vs platform config.
**Reading behaviour:** Will skip any section that does not mention deployment,
infrastructure, or operational concerns. Will read the architecture document first,
looking for the deployment model. If the deployment story is missing or hand-wavy,
will lose trust in the entire suite. Will not read the getting started guide — that
is not for them.
**Key question:** "Does this architecture create operational problems my team will
have to solve, and did the authors even think about that?"
**Blind spots:** Does not know the business rationale for the migration timeline.
Does not care about API design aesthetics. Has never used the client-facing product
and does not understand end-user workflows. Will not notice accessibility or
documentation quality issues.
**Voice sample:** "The architecture diagram shows four services but the deployment
section mentions three containers — someone is going to find out which one is missing
at 2am on a Saturday."
**Consultation voice:** "My tech lead forwarded this with 'FYI — thoughts?' which
means I have about a week before someone asks me if this is feasible."
**Directory name:** `platform-eng`
```

### Persona table

In addition to the detailed persona blocks, include a summary reference table:

| # | Directory name | Role | Lens | Key question |
|---|---|---|---|---|
| 1 | `lead-eng` | Lead engineer, application team | Code quality, team velocity, technical debt | "Does this make my team's life easier or harder?" |
| 2 | `junior-dev` | Junior developer, six months in | Learning, tooling, day-to-day workflow | "What do I actually need to do differently?" |
| 3 | `platform-eng` | Senior platform engineer, infrastructure | Reliability, deployment, operational overhead | "Does this create ops problems we'll have to solve?" |

### Voice samples table

Collect all voice samples and consultation voices into a single reference table:

| Persona | Voice sample | Consultation voice |
|---|---|---|
| Lead engineer | "If we adopt this, I need to know the migration path before I tell my team — not after." | "The VP asked me to review this and give a recommendation by Thursday." |
| Junior dev | "I don't understand half the terminology in chapter 4, but the examples in the guide actually make sense." | "My lead shared this in Slack. I'm not sure if I'm supposed to have opinions about it." |

---

## Panel Configuration

### Control persona

Designate one persona as the control — the persona whose reaction is most predictable given the document's intended audience. If this persona's reading goes badly, either the document has failed its primary audience or the simulation has failed.

Define the control with pre-registered predictions:

```
### Control persona

The [role] is the natural control — the document's [content type] is primarily
addressed to [this role's function], so their reading should be [expected stance].

**Pre-registered predictions:**
- **Expected reading path:** Will start with [document], read chapters [N-M],
  then move to [document]. Will not read [document] — it is below their level
  of concern.
- **Expected key concern:** [The specific issue this persona will focus on]
- **Expected verdict:** [Broadly positive / mixed / negative] because [reasoning]
```

The pre-registration is evaluated after the persona run. Deviation from predictions is a finding, not a failure.

### Unreliable narrator

Designate one persona who carries a specific misconception into their reading. This tests whether the document can correct a plausible misreading:

```
### Unreliable narrator

A natural unreliable narrator for this panel is a **[role] who believes
[specific misconception]**. This is a plausible misreading of [what triggers it]
and a predictable [context] framing. If the document cannot redirect this reader
within [scope], it has a framing problem that will play out at scale.
```

The misconception should be:
- Plausible — a real reader could arrive at this belief from the title, summary, or word-of-mouth
- Specific — not "they misunderstand the document" but "they believe [concrete wrong thing]"
- Testable — the document either corrects it or does not

### Persona priority order

For reduced panel runs (budget, time, or context window constraints), define which personas to run first. Map your panel's personas to the four priority archetypes:

```
### Persona priority for this panel

1. **The person talked about but not talked to:** [persona name]
2. **The decision-maker:** [persona name]
3. **The hostile reader:** [persona name]
4. **The technical validator:** [persona name]
```

A four-persona panel built from these archetypes produces more editorial intelligence than a larger panel where most personas occupy similar positions. See `process.md` Phase 1 for the rationale behind each archetype.

### Collision test pairings (optional)

Define pairs of personas whose perspectives are likely to conflict. These pairs are used in Phase 3b collision tests (see `process.md`):

```
### Collision test pairings

Recommended pairs that maximise tension:
- **[Persona A] + [Persona B]** — [source of tension]
- **[Persona C] + [Persona D]** — [source of tension]
```

Choose pairs where the tension is structural (different institutional positions, opposed incentives) rather than superficial (different opinions on the same topic).

### Delegation chain definitions (optional, deferred)

Define institutional mediation chains that model how the document is transformed as it moves through organisational hierarchies. Each chain shows the sequence: who reads the source, who writes the brief, who reads the brief, and who decides.

```
### Delegation chain: [Chain name]

1. [Role] reads the document suite (standard persona run — full reading)
2. [Role] writes a [deliverable type] for [recipient]:
   - [Content element 1]
   - [Content element 2]
   - [Content element 3]
3. [Recipient role] reads the [deliverable] (NOT the source documents).
   [Action taken on the deliverable].
4. [Final decision-maker] reads the [amended deliverable].
   [Decision options].
```

Delegation chains are an advanced execution mode. See `process.md` "Delegation chain simulation" in Execution Notes for the full methodology.

---

## Notes

The `config.md` file in this directory is a fully worked 6-persona example for a fictional cloud migration project. It includes scenario framing with detailed rationale, voice samples and consultation voices for all 6 personas, collision test pairings, and delegation chain examples across two organisational contexts.
