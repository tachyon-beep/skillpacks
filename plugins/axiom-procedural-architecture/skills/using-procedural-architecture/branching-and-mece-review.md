---
name: branching-and-mece-review
description: Critic-side audit of a decision point's option set — the four MECE/branching checks (coverage, mutual exclusivity, escape-hatch discipline, fake branches), the audit procedure, and the finding output format. Worked cloud-storage wizard example with three subtly-broken decision points produces four findings.
---

# Branching and MECE Review

**A decision point's options are a hypothesis about the space of cases; if the hypothesis is wrong, the procedure routes wrong, silently.**

A producer writes an option set, mentally scans the cases they know, and moves on. What they produce is not a complete enumeration of the input space — it is a hypothesis: these are the cases that exist. When that hypothesis is wrong, the procedure routes without alerting anyone. An audience member whose situation falls outside the listed options, or straddles two of them, has not encountered a UI gap. They have encountered a correctness defect in the decision model.

This sheet is the adversarial review that finds those defects before consumers hit them. It runs four checks against each decision point's option set and reports each defect as a **finding**: a named defect class, a **severity**, the **evidence** that makes the defect concrete, and a recommended remediation.

---

## The Four MECE/Branching Checks

### Check 1: Coverage

**Do the options cover the realistic input space? What falls through? Is the fall-through handled by "Other" or silently ignored?**

An option set is a partition of a space. Coverage asks whether the partition is complete — not whether it is exhaustive in every theoretical sense, but whether it handles every case a real audience member will bring to this decision point.

**What to verify:** Enumerate the realistic input space for this decision point. List the cases that exist, not only the ones the author thought of. For each case, determine which option it routes to. Cases that route to no option are coverage gaps. Coverage gaps with an "Other" escape hatch are handled gaps; coverage gaps without one are silent falls-through.

**Failure modes:**
- A realistic case routes to no option and there is no "Other" — the audience is trapped or forced to choose wrong.
- A realistic case routes to no option, "Other" exists, but "Other" has no defined downstream handling — the audience reaches an orphan path.
- The option set covers only the easy cases and optimistically assumes the rest do not occur.

**Severity calibration:** A coverage gap is high severity when the missing case is frequent or when forcing the audience into a wrong option produces a meaningfully incorrect exit artifact. It is medium severity when the gap is infrequent but not handled. It is low severity when the gap is rare and "Other" with defined downstream handling would close it.

---

### Check 2: Mutual Exclusivity

**Can a real case fit two options? If yes, which wins, and is the routing predictable?**

An option set that is not mutually exclusive puts the procedure in an undefined state when a case matches multiple options. The audience must guess which to choose. If they choose differently across sessions or audiences, the procedure produces inconsistent results. If downstream stages differ per option, the inconsistency propagates into the exit artifact.

**What to verify:** For each pair of options, construct a scenario in which a real audience member's case satisfies both. If no such scenario exists, the pair is exclusive. If one does, determine whether the procedure provides a tiebreaker: an explicit priority rule, a "prefer the more specific" convention, or a precondition check that would eliminate one option before the decision point fires. An overlap without a tiebreaker is a mutual-exclusivity defect.

**Failure modes:**
- Two options overlap, and the procedure provides no routing rule — different audiences route differently.
- The overlap is real but narrow enough that the producer did not notice it; the case is an edge case that appears in production.
- The overlapping options appear distinct by name but are operationally equivalent for the cases that matter.

**Severity calibration:** High when the overlap is frequent or when divergent routing produces meaningfully different exit artifacts. Medium when the overlap is infrequent but the downstream consequences differ. Low when the overlap exists but downstream handling is identical (which collapses into Check 4 — Fake Branches).

---

### Check 3: Escape-Hatch Discipline

**Is "Other" present where it should be, and absent where it must be?**

"Other" is a structural component, not a courtesy. Its presence or absence changes the correctness properties of the decision point. This check audits the decision of whether "Other" belongs in the option set at all — the producer's reasoning from [decision-flow-design.md](decision-flow-design.md) — against the actual option set delivered.

**"Other" must be present when:**
- The domain has legitimate novelty that any finite enumeration will miss.
- A real audience member falling outside the listed options is more likely than the producer admits.
- An escalation path exists so that "Other" routes somewhere defined.

**"Other" must be absent when:**
- Every input case must route to a defined handler — payment types, compliance categories, safety-critical diagnoses.
- The downstream stage cannot process a case without knowing its concrete type.
- "Other" would produce an operationally inert exit artifact that no later stage can consume.

**What to verify:** For each decision point, classify the domain as open-ended (legitimate novelty, escape hatch warranted) or forced-choice (complete enumeration required, escape hatch a defect). Then check what the option set actually contains. A forced-choice decision with "Other" is escape-hatch over-application. An open-ended decision without "Other" is escape-hatch omission. Both are defects, but in opposite directions.

**Severity calibration:** Missing "Other" in an open-ended domain is medium-to-high depending on how frequently audience members fall outside the enumeration. Spurious "Other" in a forced-choice domain is high when downstream stages fail on untyped input; medium when it produces a latent defect discovered only at later stages.

---

### Check 4: Fake Branches

**Do all options actually lead somewhere different? A decision with identically-converging outcomes is decoration, not a decision.**

A fake branch is an option that appears to offer a distinct routing but converges with another option before any meaningful difference in behavior occurs. The decision was made, the audience's attention was consumed, but the procedure arrived in the same state regardless of the choice.

This is distinct from a legitimate convergence later in the procedure — some decisions affect only the current stage and then legitimately merge. A fake branch is one where no stage produces different behavior, different exit artifacts, or different downstream routing based on which option was chosen.

**What to verify:** For each option, trace the execution path from the decision point to the first observable difference (different stage behavior, different exit artifact content, different downstream routing). If all options trace to the same execution path before any difference occurs, the decision is fake. If some options converge immediately and others diverge, the converging pair is a partial fake branch.

**Failure modes:**
- One option is an alias for another — different names, identical paths.
- An "Auto-Select" option silently resolves to a specific named option, making the "Auto-Select" option redundant except as a way to avoid making an explicit choice.
- Options are differentiated only in documentation but not in the procedure's actual logic or exit artifact.

**Severity calibration:** High when the fake branch causes the audience to believe they made a consequential choice and the downstream behavior reveals otherwise (eroded trust, incorrect expectations). Medium when the fake branch is a clarity/maintenance defect. Low when convergence is late enough that the options were meaningfully distinct for part of the procedure.

---

## Audit Procedure

Execute in order. Passes are independent within a decision point; merge findings after all passes complete for each point.

**Pass 1 — Option Enumeration.** For each decision point, list all declared options including "Other" if present. List the realistic input space for this decision point — not only the cases the author enumerated, but cases derivable from the audience parameter set, the domain, and the stage's declared inputs. This enumeration is the baseline against which all four checks run.

**Pass 2 — Coverage Check (Check 1).** Map each case in the realistic input space to an option. Flag cases that map to no option. Classify each flag as handled (an "Other" with defined routing exists) or unhandled (no escape hatch or no downstream handling for "Other").

**Pass 3 — Exclusivity Check (Check 2).** For each pair of options, test for overlap: construct a real case that satisfies both. If found, check for a tiebreaker. Flag overlapping pairs that lack a tiebreaker.

**Pass 4 — Escape-Hatch Check (Check 3).** Classify the domain as open-ended or forced-choice. Compare to the actual presence or absence of "Other". Flag mismatches in either direction.

**Pass 5 — Fake-Branch Check (Check 4).** For each option, trace the execution path forward. Identify the first point at which paths diverge. Flag options that produce no divergence before convergence.

**Assembly.** After all five passes across all decision points, deduplicate overlapping findings and assign each the highest severity among the passes that flagged it. Format per the output format below.

---

## Output Format

Each finding has five fields. Do not omit any — a finding without evidence is not actionable.

```
Finding N
  Stage reference:      [stage name and position in the declared decomposition;
                         decision point name]
  Defect class:         [one of: Check 1 — Coverage Gap | Check 2 — Mutual Exclusivity Overlap |
                         Check 3 — Escape-Hatch Omission | Check 3 — Escape-Hatch Over-Application |
                         Check 4 — Fake Branch]
  Severity:             [high | medium | low]
  Evidence:             [which input case is not covered; which two options overlap and the
                         overlapping scenario; which "Other" is missing or spurious; which
                         options converge and at what point]
  Remediation:          [add option / add "Other" with defined routing / remove "Other" /
                         add tiebreaker rule / merge fake branches; specific enough to act
                         on without further questions]
```

**Severity tiers:**

- **High** — silent incorrect routing, irreversible exit artifact committed from a wrong path, or procedure unexecutable for a realistic audience member without out-of-band judgment.
- **Medium** — detectable error or forced interruption; the audience can recover but the procedure is disrupted or routes inconsistently across sessions.
- **Low** — clarity or maintainability defect; works on the happy path but misleads, erodes trust, or breaks under audit.

---

## Worked Example: Configure Cloud-Storage Backend Wizard

**Procedure:** Configure a cloud-storage backend for an application.

**Declared stage sequence (abbreviated):**

```
Stage 1 — Identify storage provider
  Decision point: Which cloud provider?
  Options: AWS / GCP / Azure
  Exit artifact: provider-selection-record

Stage 2 — Configure authentication method
  Decision point: Authentication method?
  Options: OAuth / SAML / Workload Identity
  Exit artifact: auth-method-record

Stage 3 — Select deployment region
  Decision point: Which region?
  Options: US-East / US-West / Auto-Select
  Exit artifact: region-selection-record
```

---

### Pass 1 — Option Enumeration

**Stage 1 options:** AWS / GCP / Azure

**Realistic input space for Stage 1:** Cloud providers include AWS, GCP, Azure — and also self-hosted / on-premise object stores (MinIO, Ceph, Rook), compliance-mandated sovereign clouds (AWS GovCloud, Azure Government), and private-cloud providers (OVHcloud, Hetzner). The audience for a "configure cloud-storage backend" wizard may be running on-premise infrastructure with no public-cloud dependency.

**Stage 2 options:** OAuth / SAML / Workload Identity

**Realistic input space for Stage 2:** Authentication methods for cloud storage include OAuth 2.0 (user-delegated token flow), SAML (assertion-based federation, often enterprise IdP), and Workload Identity (GCP-native mechanism for service-account impersonation). Workload Identity in GCP can itself be federated over SAML or OIDC, making it an implementation of an authentication protocol rather than an alternative to one. Additionally: AWS IAM roles for EC2/ECS, API key / access key pairs, mTLS, and service-account JSON keys.

**Stage 3 options:** US-East / US-West / Auto-Select

**Realistic input space for Stage 3:** Region selection should cover all regions the provider offers, or at minimum all regions the procedure's audience is permitted to use. "Auto-Select" is presented as an option peer.

---

### Findings

**Finding 1**
  Stage reference:      Stage 1 — Identify storage provider (position 1 of 3);
                        decision point: "Which cloud provider?"
  Defect class:         Check 3 — Escape-Hatch Omission
  Severity:             High
  Evidence:             On-premise / self-hosted object stores (MinIO, Ceph) are not covered by
                        AWS / GCP / Azure and no "Other" option exists. An audience member running
                        a MinIO cluster has no valid option. If forced to pick the nearest named
                        option (e.g., "AWS" because S3-compatible), subsequent stages configure
                        AWS-specific authentication and IAM, which will not apply to their
                        infrastructure. The exit artifact is silently wrong. The root defect is
                        escape-hatch omission in an open-ended domain (cloud providers); the
                        downstream consequence is the coverage gap that would otherwise classify
                        under Check 1. Cite Check 3 because the remediation is to add "Other"
                        (open-ended) rather than to enumerate every conceivable provider.
  Remediation:          Add an "Other / Self-hosted (S3-compatible)" option with a defined
                        downstream path — at minimum, a stage that captures the endpoint URL and
                        credential type without assuming a public-cloud provider's IAM. If the
                        wizard's scope is intentionally public-cloud-only, declare that constraint
                        as an audience prerequisite before stage 1 so on-premise audiences know
                        they are out of scope before investing time.

---

**Finding 2**
  Stage reference:      Stage 2 — Configure authentication method (position 2 of 3);
                        decision point: "Authentication method?"
  Defect class:         Check 2 — Mutual Exclusivity Overlap
  Severity:             High
  Evidence:             Workload Identity (GCP) can be configured as a SAML-federated identity
                        — an audience member using Workload Identity Federation with a corporate
                        IdP over SAML satisfies both "SAML" and "Workload Identity" simultaneously.
                        No tiebreaker rule exists. Two audience members in identical situations
                        will route to different stages depending on whether they think of their
                        setup as "SAML" (IdP-first framing) or "Workload Identity" (GCP-first
                        framing). The stages that follow differ: the SAML path configures IdP
                        metadata; the Workload Identity path configures a GCP service account
                        impersonation policy. Both configurations may be required, but the wizard
                        presents them as mutually exclusive.
  Remediation:          Restructure the option set by authentication mechanism level, not product
                        name. Example: "Service-account key / Workload Identity (OIDC or SAML
                        federation) / OAuth user token / API key." Alternatively, add a
                        precondition check that classifies the audience's IdP situation before
                        this decision point fires, so the option set is pre-narrowed to mutually
                        exclusive choices for their specific environment. Cross-reference
                        decision-flow-design.md for information-readiness gating.

---

**Finding 3**
  Stage reference:      Stage 2 — Configure authentication method (position 2 of 3);
                        decision point: "Authentication method?"
  Defect class:         Check 1 — Coverage Gap
  Severity:             Medium
  Evidence:             AWS IAM roles (for EC2/ECS/Lambda workloads), API access-key pairs,
                        and mTLS client certificates are all realistic authentication methods for
                        cloud storage — none appear in the option set. Given that Finding 1
                        routes AWS audiences into stage 2, AWS IAM roles is a frequent case for
                        this wizard's realistic input space. An AWS audience forced to choose
                        "Workload Identity" (a GCP concept) or "OAuth" (user-delegated, not
                        workload-appropriate) will produce an incorrect auth-method-record.
  Remediation:          The option set must reflect the authentication methods available for
                        the provider selected in stage 1. After resolving Finding 1, pass
                        the provider-selection-record as a precondition input to stage 2 and
                        narrow the option set by provider. This eliminates GCP-specific options
                        for AWS audiences and vice versa, and allows the coverage to be complete
                        per-provider rather than requiring a single flat list to cover all of them.

---

**Finding 4**
  Stage reference:      Stage 3 — Select deployment region (position 3 of 3);
                        decision point: "Which region?"
  Defect class:         Check 4 — Fake Branch
  Severity:             High
  Evidence:             "Auto-Select" is presented as a peer option alongside "US-East" and
                        "US-West," implying it is a distinct routing. Inspection of the wizard's
                        downstream logic reveals that "Auto-Select" resolves to US-East silently
                        — no latency probe, no capacity check, no audience-location heuristic.
                        The decision point fires; the exit artifact reads "region: us-east-1"
                        regardless of which of the three options was chosen if "Auto-Select" is
                        selected, and "Auto-Select" always selects US-East. The audience who
                        chose "Auto-Select" believing it performed a meaningful selection has
                        the same result as if they had chosen "US-East" directly. The decision
                        consumed their attention without performing a decision.
  Remediation:          Remove "Auto-Select" or implement it honestly. If "Auto-Select" is
                        supposed to be a real selection mechanism, it must perform a detectable
                        selection (latency probe, geographic lookup, or capacity query) and
                        produce a result that can differ from "US-East." If it is intended as
                        a default, present it as a default for the US-East option rather than
                        a separate option: "US-East (default) / US-West." If the intent is
                        to hide the decision from audiences who do not need to make it,
                        remove the decision point from the default path and add it only as
                        an advanced-configuration stage.

---

## Cross-references

- [decision-flow-design.md](decision-flow-design.md) — the producer-side mirror: decides whether a decision point should exist, what its preconditions are, and whether "Other" should be in the option set. This sheet audits whether the resulting option set is well-formed; that sheet builds the option set to be audited.
- [decomposition-smells.md](decomposition-smells.md) — the smell catalog for severity calibration and smell naming: fake branches are an instance of the fake-branch smell, coverage gaps may instantiate orphan-path or premature-fork. A finding should cite the smell it instantiates; the smell's severity guidance overrides this sheet's generic tiers when they conflict.
- [dependency-and-ordering-audit.md](dependency-and-ordering-audit.md) — the sibling critic sheet: audits ordering correctness of the decomposition. Shares this sheet's finding output format (stage reference / defect class / severity / evidence / remediation). A decision-point defect found here may interact with an ordering defect found there — cross-reference finding IDs when remediations are coupled.
- [procedural-boundary-and-handoffs.md](procedural-boundary-and-handoffs.md) — when a finding's remediation requires domain-content judgment (which authentication method is technically superior for the audience's infrastructure, which region has the relevant compliance certification), that judgment is out of scope for this audit; document the gap in the remediation field and hand off to the appropriate domain pack.
