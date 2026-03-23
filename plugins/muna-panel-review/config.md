# Reader Panel Configuration — Example: Cloud Migration Architecture Suite

This file is a **worked example** of a reader panel configuration. It demonstrates the format, structure, and design thinking documented in `config-template.md`, using a fictional cloud migration project for illustration.

Use this as a reference when building your own config. The scenario, personas, and documents are invented — adapt the patterns to your project's actual documents and stakeholders.

---

## Document Suite

The following document suite is presented to each persona at the start of their reading session. The persona chooses which document to open first and navigates from there — the choice itself is data. The suite map is provided as an orientation aid, not a prescription.

| # | Document | Description | Chapters | Source files |
|---|---|---|---|---|
| 0 | **Document Suite Map** | One-page reading path guide — routes five roles to the right document | 1 | `suite-map.md` |
| 1 | **Cloud Migration Architecture** (Technical Specification) | Full migration design: service decomposition, data migration strategy, rollback procedures, SLA targets. The primary technical document. | 14 | `arch-[0-9][0-9]-*.md` |
| 2 | **API Compatibility and Integration Guide** (Companion Document) | Technical specification: versioning strategy, deprecation policy, client SDK migration, backwards-compatibility guarantees. | 10 | `api-[0-9][0-9]-*.md` |
| 3 | **Executive Migration Brief** | Accessible summary (~6 pp): the migration rationale, timeline, and risk assessment in plain language for leadership and budget holders. | 1 | `exec-brief.md` |
| 4 | **Developer Migration Playbook** | Hands-on guide (~10 pp): step-by-step migration for each service, common pitfalls, local testing setup, rollback procedures. | 1 | `dev-playbook.md` |

**Presentation to personas:** At the start of each persona's session, present the scenario framing (below) FIRST, then the suite map (document 0) and this table. The persona encounters the institutional context before seeing the documents — the same way a real reader would encounter a team email or Slack announcement before opening the linked documents. The persona writes their first journal entry as a **suite orientation** — their reaction to the framing, the menu, and their choice of where to start. See `process.md` Phase 2 for the suite orientation journal template.

The entry point is a finding about audience routing. The time budget is a finding about perceived investment value — a persona who budgets 15 minutes is telling you they do not expect the suite to be central to their work.

---

## Scenario Framing

> **Personas should be told the following at the start of their reading session, before they see the suite map:**
>
> *The document suite you are about to read has been circulated as* ***Version 0.9 — Internal Review Draft*** *by the Platform Engineering team. The migration is scheduled to begin in Q3 and the team is seeking feedback from all affected groups before finalising the architecture. You have received this because your team will be directly affected by the migration. Comments and concerns are expected — the authors want to surface blockers and misunderstandings before implementation begins.*

This framing is not decorative. It changes how every persona relates to the material:

**Why "Internal Review Draft."** The version string signals maturity and intent:
- *0.9* says this is nearly final — the authors are not brainstorming, they are validating. Personas should not expect to reshape the fundamental approach, but they can still influence details, surface blockers, and flag gaps.
- *Internal Review Draft* says this has not been shared with external partners or customers yet. Feedback stays inside the organisation.

**Why "scheduled to begin in Q3."** The timeline creates urgency without panic. Personas know this is happening — the question is not *whether* but *how well*. A persona who raises a blocker has leverage; one who stays silent will be expected to comply.

**Why "all affected groups."** This is not a targeted consultation — everyone who touches the system has been asked. This flattens hierarchy: the junior developer's feedback carries weight because they are the ones who will execute the migration. It also means no one can claim they were not consulted.

### Framing effects by persona type

| Persona type | How they received it | Framing effect |
|---|---|---|
| **Platform team (architect, SRE)** | Direct — they authored or commissioned it | They own the document. Their stance is "does this represent our plan accurately?" |
| **Application teams (lead eng, junior dev)** | Team channels — forwarded by their engineering manager | They are affected but did not author it. Their question is "what does this mean for my sprint?" |
| **Product (PM, designer)** | Cross-functional update — included in a product review email | They care about timeline and customer impact, not technical details |
| **Leadership (VP Eng, CTO)** | Flagged by a direct report as needing exec review | They will not read the full suite. Their question is "is this on track and what are the risks?" |
| **External-facing (support lead, partner eng)** | Forwarded by PM or engineering manager with context | They care about customer impact and communication timing |

---

## Reference Panel

| # | Directory name | Role | Lens | Key question |
|---|---|---|---|---|
| 1 | `lead-eng` | Senior engineer, payments service team | Code ownership, team velocity, technical debt | "How much of my team's roadmap does this eat, and did they think about our edge cases?" |
| 2 | `junior-dev` | Junior developer, six months in, onboarding service | Learning, day-to-day workflow, tooling changes | "What do I actually need to do, and will the playbook get me through it?" |
| 3 | `vp-eng` | VP Engineering, reports to CTO | Budget, timeline, cross-team risk, headcount | "Is this going to land on time, and what's the fallback if it doesn't?" |
| 4 | `product-manager` | Senior PM, customer-facing product | Feature freeze impact, customer communication, launch timing | "When do I need to tell customers, and what can't I ship while this is happening?" |
| 5 | `sre` | Site reliability engineer, on-call rotation lead | Uptime, rollback, monitoring, incident response | "If this goes wrong at 2am, can I fix it without waking the whole platform team?" |
| 6 | `support-lead` | Customer support team lead | Ticket volume, known-issue docs, escalation paths | "What breaks for customers, and what do I tell my team to say?" |

### Voice samples

| Persona | Voice sample | Consultation voice |
|---|---|---|
| Lead engineer | "If the data migration plan doesn't account for our partitioned tables, we're going to find out in production. Show me the rollback." | "I'll review this but I need to loop in Priya — she owns the payment reconciliation pipeline and this will break her daily batch job." |
| Junior dev | "The architecture doc lost me at chapter 6 but the playbook actually makes sense. Can I just follow the playbook?" | "My tech lead posted this in Slack and said 'read it by Friday.' I'm not sure what level of feedback they expect from me." |
| VP Engineering | "What's the confidence level on the Q3 date, and what's the blast radius if we slip?" | "I need a one-pager for the CTO by end of week. If there are budget implications, flag them now." |
| Product manager | "I can live with a two-week feature freeze but four weeks kills our Q3 launch. What's the real number?" | "I need to know the customer-facing impact before I can write the external comms. Right now this reads like an engineering plan — where's the customer story?" |
| SRE | "The monitoring section mentions three new dashboards but doesn't say who builds them or when they go live relative to the migration. That's a gap." | "I'm reviewing the rollback procedures. If they don't work in staging by mid-Q2, I'm going to flag this to the VP as not ready." |
| Support lead | "When the old API starts returning 301s, my team needs a script, not a design document. The playbook is close but it's written for developers, not support." | "I'll need two weeks to update our runbooks and train the team. That needs to be in the timeline, not assumed." |

---

## Panel Configuration Notes

### Persona priority for this panel

The four archetype slots for a reduced panel, mapped to this project's personas:

1. **The person talked about but not talked to:** the junior developer
2. **The decision-maker:** the VP Engineering
3. **The hostile reader:** the lead engineer (payment service — high migration risk)
4. **The technical validator:** the SRE

### Control persona

The lead engineer is the natural control — the migration architecture is primarily addressed to senior engineers who own affected services, so their reading should be critically engaged but constructive. If the lead engineer's reaction is dismissal or confusion, either the document has failed its primary audience or the simulation has failed.

### Unreliable narrator

A natural unreliable narrator for this panel is a **product manager who believes the migration will require a four-week feature freeze across all teams**. This is a plausible misreading of the timeline section (which mentions a two-week freeze for the payments service only) and a predictable cross-functional concern. If the document cannot correct this within the first three chapters, it has a scoping problem that will generate unnecessary pushback from every PM in the organisation.

### Collision test pairings

Recommended pairs that maximise tension:
- **Lead engineer + SRE** — migration velocity vs operational safety
- **VP Engineering + junior developer** — timeline confidence vs ground-level readiness
- **Product manager + support lead** — customer communication timing vs support team preparation

### Delegation chain examples

**Chain 1: Engineering team response**
```
1. Junior developer reads the document suite (standard persona run — full reading)
2. Junior developer summarises to their tech lead in standup:
   - Three things that affect their current work
   - One thing they don't understand
   - Whether the playbook is sufficient for their service
3. Tech lead reads the executive brief and skims the architecture doc.
   Writes a response in the review thread with team-level concerns.
4. Engineering manager aggregates team responses into a one-page summary
   for the VP Engineering.
```

**Chain 2: Product escalation**
```
1. Product manager reads the executive brief and playbook (NOT the architecture doc)
2. Product manager writes a customer impact assessment:
   - Which features are affected during migration
   - Proposed customer communication timeline
   - Feature freeze duration and scope
3. Head of Product reads the impact assessment.
   Decides: accept timeline, negotiate shorter freeze, or escalate to CTO.
```
