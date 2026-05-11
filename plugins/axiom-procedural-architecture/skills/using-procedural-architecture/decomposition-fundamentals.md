---
name: decomposition-fundamentals
description: The five structural properties of a good decomposition — MECE-ish coverage, grain consistency, dependency correctness, reversibility-ordered staging, progressive disclosure — with definitions, signals, and a worked Postgres-setup example showing the bad-and-good versions.
---

# Decomposition Fundamentals

**A good decomposition has a small fixed set of structural properties; they can be stated, taught, and checked.**

This sheet names the five. If a proposed decomposition satisfies all five, it is structurally sound — not necessarily correct in its domain-content, but sound as a procedure. If it violates any of them, there is a specific, nameable defect, not a vibe.

The five properties are not independent. Grain consistency and MECE-ish coverage are both violated by the same god-step smell. Dependency correctness and reversibility-ordered staging both depend on knowing what each stage produces. Progressive disclosure is the integration test: if a decomposition satisfies the first four properties but still feels opaque, progressive disclosure is where to look.

---

## 1. MECE-ish Coverage

**Option sets at every decision point cover the space without overlap; the "-ish" acknowledges that strict MECE is rare in practice.**

At any decision point, the set of selectable options must be:

- **Mutually exclusive** — no audience member should reasonably fall into two branches at once. Overlap means the decomposition is making a choice the audience should have made, or two branches converge identically (see `branching-and-mece-review.md` for the fake-branch smell).
- **Collectively exhaustive** — every audience member fits into exactly one option. Where the space cannot be fully enumerated, an explicit `Other (with discipline)` option is acceptable. "With discipline" means `Other` is not a dead end: it has an exit artifact and a defined next stage, usually escalation or deferred decision. An unenumerated `Other` is an orphan path; the audience falls off the edge of the procedure.

**Signal.** Ask: "Can I construct a plausible audience member who falls into two options, or into none?" If yes, the decision point is defective. Options that are grammatically distinct but functionally identical (both lead to stage N, same inputs required, same exit artifact) are fake branches — exclusive in name only.

**Inline example.** A Postgres access-tier decision with options `{Internal, External, Shared}` may look exhaustive until you meet a read replica used by an external service: it is simultaneously `External` (consumer) and `Internal` (replication topology). That overlap is a design defect; the dimension being decided needs to be split into two separate decision points.

---

## 2. Grain Consistency

**Stages are roughly the same size of work; mixing god-steps with trivial-steps in a single decomposition is a smell.**

Grain is not about absolute size — a curriculum for a junior can have finer grain than one for a senior on the same material. Grain is about *consistency within a single decomposition for a specific audience*. A decomposition where stage 3 takes fifteen minutes and stage 7 takes two weeks is not just inconvenient: it signals that the stages are not being drawn at the same level of abstraction. The god-step hides decision points and dependencies inside itself. The trivial-step (part of the `ladder-of-trivials` cluster smell) is padding that exists to fill space between the real stages.

**Signal.** Estimate effort for each stage against the same audience. If the ratio of max to min effort across stages exceeds ~5:1, the decomposition has a grain problem. Either the large stage needs subdivision or the cluster of small stages needs consolidation into a single stage with a richer exit artifact. Both operations should improve the ratio without hiding dependencies.

A ladder-of-trivials — a long chain of stages that each do one obvious thing — is the complementary smell. The stages are individually consistent, but the entire chain could be one stage with a checklist exit artifact. Grain consistency applies at both ends of the range.

See [granularity-calibration.md](granularity-calibration.md) for the full treatment: working-memory capacity, error cost, and audience competence as the three calibration levers.

---

## 3. Dependency Correctness

**Every precondition of a stage is produced by an earlier stage or present in the audience's prior state before that stage executes; no stage consumes an input that does not yet exist.**

Dependency correctness is the structural property that makes a decomposition *executable*. A stage that silently reads the exit artifact of a later stage is a cycle; a stage that reads an artifact that nothing produces is an orphaned input. Both are bugs in the decomposition's DAG, not in the content of the stages.

Two failure modes:

- **Hidden coupling.** Stage B works correctly in practice because stage A always runs first, but the dependency is not declared. Nobody can audit the order, reorder for a different audience, or safely parallelize. The dependency exists; it is just invisible.
- **Premature commitment.** A stage makes a choice that requires information only produced by a later stage. A wizard that asks "choose your replication topology" before establishing whether the environment supports streaming replication at all is making a decision before its preconditions are met.

**Signal.** For each stage, enumerate its declared inputs. Trace each input to the stage or audience-prior-state that produces it. Any input that cannot be so traced is an undeclared dependency or a missing stage. This is the same exercise as [dependency-and-ordering-audit.md](dependency-and-ordering-audit.md); this sheet names the property, that sheet runs the check.

---

## 4. Reversibility-Ordered Staging

**Cheap-to-revisit decisions come early; expensive-to-revisit decisions are gated behind the scaffolding that makes them informed.**

This is the structural version of the two-way-door / one-way-door framing (Bezos, cited here as useful folklore rather than authority): cheap reversals early, expensive reversals late and scaffolded. It is not about risk aversion — it is about information flow. A staging order where the expensive, hard-to-undo decision comes before the information needed to make it wisely is structurally broken, regardless of how confident the audience feels.

**Signal.** Rank the stages by reversal cost: what does it cost (time, state, money, coordination) to undo the exit artifact of this stage and try a different path? The ranking should be roughly increasing through the decomposition. If a high-reversal-cost stage appears before a low-reversal-cost stage that would provide information relevant to that decision, the ordering is wrong.

The cheaper path is not always the better choice; there are legitimately hard decisions to make early (a data residency choice may be legally imposed regardless of information state). But those cases are exceptions that must be argued, not defaults.

See [decision-flow-design.md](decision-flow-design.md) for the information-readiness reasoning that drives this ordering: when to force a choice, when to defer it, and when deferral is itself a design decision.

---

## 5. Progressive Disclosure

**Each early choice meaningfully narrows the downstream space; if it doesn't, the choice is premature or fake.**

Progressive disclosure is the integration test for the other four properties. A decomposition where the audience makes choices but the remaining procedure looks identical regardless of what was chosen is wasting the audience's time and producing a procedure with fake branches. Each decision point should shrink the space of what remains — fewer stages, different stages, or the same stages with meaningfully different parameters.

**Signal.** After each decision point, ask: what stages or options are now ruled out that were in play before? If the answer is "none," the decision point either belongs later (its information hasn't narrowed the space yet — a dependency correctness issue) or the branches converge so quickly they shouldn't have diverged (a MECE-ish / fake-branch issue).

Progressive disclosure also applies within stages. A stage that asks six questions before it can produce its exit artifact may need to be subdivided so that each question is asked only when the preceding answer is in hand.

---

## Worked Example: Postgres Database Setup

**Procedure:** Set up a new Postgres database for a service in a production environment.

### Bad Version (two stages, multiple violations)

```
Stage 1 — Bootstrap environment
  (Install Postgres, configure OS limits, open firewall, create service account)

Stage 2 — Provision Postgres
  (Create database, set schema, configure users, choose connection-string
   format, set backup schedule, configure replication topology, apply
   security-group rules, tune pg_hba.conf, set connection pool parameters)
```

**Violations visible:**

- **Grain consistency (god-step).** Stage 2 hides at minimum six decision points and a dozen distinct exit artifacts. Its effort ratio against stage 1 is roughly 10:1.
- **Dependency correctness.** "Configure replication topology" requires knowing whether streaming or logical replication is supported by the environment — information that is produced inside stage 2 itself if you happen to order the sub-tasks correctly, but not declared as an input.
- **Reversibility-ordered staging.** "Choose connection-string format" (low reversal cost — a config value) is buried inside the same stage as "configure replication topology" (high reversal cost — infrastructure change). There is no signal about which should be decided first.
- **Progressive disclosure.** The audience cannot see which sub-task they are on or what the current choice eliminates. Stage 2 is opaque.

---

### Good Version (six stages, all five properties satisfied)

**Declared audience parameters (required before decomposition):**

- Prerequisites: shell access to target environment; cloud-provider credentials; schema DDL already authored
- Working-memory capacity: mid-level SRE, can hold ~5 active constraints
- Error cost: high (production data store; rollback is disruptive)
- Reversibility appetite: low (organization prefers to gate expensive decisions)

```
Stage 1 — Confirm environment readiness
  Input:  target environment name, required Postgres version
  Action: verify OS limits, network reachability, IAM permissions
  Exit artifact: environment-readiness-report (pass/fail + blocking items)
  Decision point: none
  Reversal cost: very low (read-only)

Stage 2 — Choose data-topology class
  Input:  environment-readiness-report, service SLA, team backup policy
  Action: select from option set (see below)
  Exit artifact: topology-decision-record (chosen class + rationale)
  Decision point: topology class
    Options (MECE-ish):
      A. Single-instance (dev/staging, RPO > 1hr acceptable)
      B. Streaming-replica (production, RPO < 5min, same-region)
      C. Logical-replica (cross-region or cross-version)
      D. Other (HA configuration not covered above → escalate to DBA)
    Narrows: stages 4–5 differ materially by class; B and C require
             additional network configuration not needed for A
  Reversal cost: low (a document; no infrastructure committed yet)

Stage 3 — Provision instance and storage
  Input:  topology-decision-record, sizing estimate
  Action: create instance, attach volumes, apply OS tuning
  Exit artifact: instance-provisioning-record (endpoint, version, disk config)
  Decision point: none (parameters flow from stage 2)
  Reversal cost: medium (infrastructure exists; destroy-and-recreate is
                 ~20 min but not stateful)

Stage 4 — Bootstrap security baseline
  Input:  instance-provisioning-record, org security policy
  Action: configure pg_hba.conf, create service account, apply
          security-group rules, generate or import TLS certificate
  Exit artifact: security-baseline-record (account name, cert thumbprint,
                 hba rules hash)
  Decision point: credential-delivery method
    Options (MECE-ish):
      A. Secrets manager injection (preferred if available)
      B. Environment variable (acceptable for low-sensitivity)
      C. Other → document rationale in security-baseline-record
  Reversal cost: medium (credentials can be rotated; TLS cert replacement
                 requires coordinated rollout)

Stage 5 — Apply schema and configure replication
  Input:  security-baseline-record, topology-decision-record, DDL
  Action: create database, run DDL, set up replication per topology class
          (skipped for class A)
  Exit artifact: schema-deployment-record (migration hash, replica lag
                 baseline if applicable)
  Decision point: none (topology already decided in stage 2)
  Reversal cost: high (schema changes are costly to roll back in production;
                 this is why topology and security gate here)

Stage 6 — Configure operations (backup, alerting, connection pool)
  Input:  schema-deployment-record, ops policy
  Action: set backup schedule, configure WAL archiving, wire alerts,
          set pool parameters in application config
  Exit artifact: ops-readiness-record (backup job ID, alert channel,
                 connection string)
  Decision point: none (parameters follow from ops policy and topology)
  Reversal cost: low (all operational configuration; no data committed)
```

**Properties demonstrated:**

- **MECE-ish coverage.** Stage 2 topology option set covers single-instance, two replication modes, and an explicit `Other` with escalation path. Stage 4 credential option set covers the two standard delivery methods and an explicit `Other` with required documentation.
- **Grain consistency.** No stage exceeds ~2 hours of work for the target audience; no stage is purely mechanical and trivial. Ratio of max to min effort is roughly 4:1.
- **Dependency correctness.** Stage 5 (replication) receives topology-decision-record from stage 2 — the topology choice is an explicit declared input, not a hidden assumption. Stage 4 (security) gates on instance existence from stage 3.
- **Reversibility-ordered staging.** Stage 2 (document only, very low reversal cost) precedes stage 3 (infrastructure, medium reversal cost), which precedes stage 5 (schema, high reversal cost). The expensive-to-undo decision is never asked before the information that makes it tractable is in hand.
- **Progressive disclosure.** After stage 2, class-A audiences skip the replication branch entirely. After stage 4, the credential-delivery decision eliminates one integration path. Each choice narrows.

---

## Cross-references

- [granularity-calibration.md](granularity-calibration.md) — the grain-size question in full: working-memory capacity, error cost, and audience competence as the three calibration levers; when to subdivide vs consolidate.
- [decision-flow-design.md](decision-flow-design.md) — the information-readiness reasoning that drives reversibility ordering: when to force a choice, when to defer it, and how to design a decision point so it fires only when its inputs exist.
- [dependency-and-ordering-audit.md](dependency-and-ordering-audit.md) — the critic-side mirror: how to audit a proposed decomposition for the same five properties; the same preconditions-met-before-use discipline run adversarially.
- [procedural-boundary-and-handoffs.md](procedural-boundary-and-handoffs.md) — when the content judgement inside a stage (which Postgres extension to use, which backup tool, which replication mode is technically superior) is not this pack's job; where to hand off to a domain pack.
