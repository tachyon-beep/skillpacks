---
name: dependency-and-ordering-audit
description: Critic-side audit of a proposed decomposition's ordering correctness — the four ordering checks (preconditions met before use, no premature commitment, cheap-decisions-early/expensive-decisions-gated, no hidden coupling), the audit procedure, and the finding output format (stage / defect class / severity / evidence / remediation). Worked broken-SSH-wizard example produces five findings.
---

# Dependency and Ordering Audit

**Ordering bugs in procedures are silent until execution; an audit must surface them before a consumer hits them.**

A producer can deliver a structurally tidy-looking decomposition — stages named, exit artifacts declared, decision points enumerated — and still ship a procedure that cannot be executed in the declared order. Ordering bugs are invisible to the author because they already know the correct order from domain knowledge they have not encoded. The consumer hits the gap at runtime: an input that does not exist yet, a commitment demanded before the information that justifies it is in hand, a side effect silently depended on, an expensive decision that should have been scaffolded but was not.

This sheet is the adversarial pass that finds those bugs before they reach a consumer. It runs four checks against a proposed decomposition and reports each defect as a **finding**: a named defect class, a **severity**, the **evidence** that makes the defect concrete, and a recommended remediation. The critic's job is to produce findings specific enough that the producer can act without guessing.

---

## The Four Ordering Checks

### Check 1: Preconditions Met Before Use

**Every stage's required inputs are produced by an earlier stage or present in the audience's declared prior state before that stage executes.**

A stage that consumes an input that does not yet exist cannot run correctly. The failure mode is either a detectable error, a silent bad result (the stage runs against a default or empty value and produces a plausible-looking but wrong exit artifact), or a runtime stall.

**What to verify:** For each declared input of each stage, trace a producer — either a named earlier stage whose exit artifact contains or implies that input, or a named audience parameter declared as a prerequisite. An input with no traceable producer is an undeclared dependency or a missing stage.

---

### Check 2: No Premature Commitment

**No decision point is forced before the inputs it depends on exist.**

A decision point whose preconditions are not yet satisfied is asking the audience to guess. They may not know they are guessing — they make a choice, the procedure advances, and the error manifests stages later when a committed artifact contradicts information that finally arrived.

**What to verify:** For each decision point, enumerate its declared preconditions. Confirm every precondition is satisfied before the stage containing that decision point executes — either produced by an earlier stage or present in the audience's prior state.

Distinguish from Check 1: Check 1 is about a stage that cannot run; Check 2 is about a choice that cannot be made meaningfully. A stage may technically execute (the decision point fires) while still being premature (the audience lacks the inputs needed to choose rationally).

---

### Check 3: Cheap-Decisions-Early, Expensive-Decisions-Gated

**Cheap-to-revisit choices appear early; expensive-to-revisit choices are scaffolded with the information-gathering stages that make them informed.**

This is the adversarial version of reversibility-ordered staging from `decomposition-fundamentals.md`. The producer is supposed to place high-reversal-cost decisions after the stages that produce the information needed to make them safely; the audit checks that this was actually done.

**What to verify:** Rank stages and decision points by reversal cost. The ranking should be roughly increasing through the decomposition. For each high-reversal-cost decision point, verify that preceding stages produce the information required to make that decision with appropriate confidence. A high-reversal-cost decision appearing before the stages that would scaffold it is a gating failure.

---

### Check 4: No Hidden Coupling

**Stage B should not silently depend on a side effect of stage A that is not declared as an exit artifact of stage A.**

Hidden coupling means the procedure works in practice because the stages always run in a certain order, and one happens to leave behind a side effect the other consumes — but the dependency is not declared. Nobody can safely reorder, skip, or parallelize stages without discovering the coupling the hard way.

**What to verify:** For each stage, list all side effects it produces beyond its declared exit artifact: environment state changes, implicit file writes, service-state transitions, cached credentials, queue mutations. Check whether any later stage consumes such a side effect without declaring it as an input. The test: could a consumer execute stage B without running stage A and get the same result? If not, and the dependency is undeclared, the coupling is hidden.

---

## Audit Procedure

Execute in order. Each pass is independent; results are merged at the end.

**Pass 1 — Input-Producer Trace (Check 1).** List every stage with its declared inputs and exit artifact. For each input, identify its producer (earlier stage or audience parameter). Flag every input whose producer cannot be identified, or whose producer stage is ordered after the consumer stage.

**Pass 2 — Decision-Point Precondition Walk (Check 2).** List every decision point with its declared preconditions. Trace each precondition to its producer. Verify the producer stage precedes the decision point in the declared flow. Flag every decision point whose preconditions are not satisfied before control reaches it.

**Pass 3 — Reversal-Cost Ranking (Check 3).** Assign each stage and decision point a reversal-cost tier: very-low / low / medium / high. Walk the decomposition in declared order. Flag each position where a higher-tier decision point precedes a lower-tier stage that would have produced scaffolding information for it.

**Pass 4 — Side-Effect Inventory (Check 4).** For each stage, enumerate side effects beyond the declared exit artifact. For each side effect, check whether any later stage consumes it without a declared input dependency. Flag each such consumption.

**Assembly.** After all four passes, deduplicate overlapping findings and assign each the highest severity among the passes that flagged it. Format per the output format below.

---

## Output Format

Each finding has five fields. Do not omit any — a finding without evidence is not actionable.

```
Finding N
  Stage reference:  [stage name and position in the declared decomposition]
  Defect class:     [one of: Check 1 — Preconditions Not Met | Check 2 — Premature Commitment |
                     Check 3 — Gating Failure | Check 4 — Hidden Coupling]
  Severity:         [high | medium | low]
  Evidence:         [which stage is or isn't producing the required input; what specific input is
                     missing, what side effect is undeclared, or what information is absent from
                     the decision point's precondition set]
  Remediation:      [reorder / add precondition stage / promote side effect to declared exit
                     artifact / restructure; specific enough to act on without further questions]
```

**Severity tiers:**

- **High** — silent incorrect result, irreversible artifact committed before supporting information exists, or procedure unexecutable without out-of-band knowledge.
- **Medium** — detectable error or forced interruption; consumer can recover but the procedure is disrupted.
- **Low** — clarity or maintainability problem; works on the happy path but breaks under reordering or for a different audience.

---

## Worked Example: Broken SSH-Access Wizard

**Procedure:** Set up SSH access to a new server.

**Declared stage sequence (as proposed — deliberately broken):**

```
Stage 1 — Deploy key to server
  Declared input:   public key
  Exit artifact:    key deployed to server authorized_keys

Stage 2 — Choose key type (RSA / Ed25519 / ECDSA)
  Declared input:   none
  Exit artifact:    key-type-decision-record
  Decision point:   key type
  Preconditions:    (none declared)

Stage 3 — Generate keypair
  Declared input:   key-type-decision-record
  Exit artifact:    keypair (public key + private key)

Stage 4 — Configure passphrase policy
  Declared input:   keypair
  Exit artifact:    passphrase-policy-record
  Decision point:   use passphrase or not?
  Preconditions:    (none declared)

Stage 5 — Test SSH connection
  Declared input:   server address, username
  Exit artifact:    connection-test-result
```

---

**Finding 1**
  Stage reference:  Stage 1 — Deploy key to server (position 1 of 5)
  Defect class:     Check 1 — Preconditions Not Met
  Severity:         High
  Evidence:         Stage 1 requires "public key" as input. The public key is the exit artifact
                    of Stage 3 (Generate keypair), which appears at position 3 — after Stage 1.
                    No audience prior state covers this; the audience arrives without a keypair.
  Remediation:      Reorder: move Stage 2 (Choose key type) and Stage 3 (Generate keypair) to
                    positions 1 and 2. Corrected order: Choose key type → Generate keypair →
                    Deploy key → Configure passphrase policy → Test connection.

---

**Finding 2**
  Stage reference:  Stage 2 — Choose key type (position 2 of 5)
  Defect class:     Check 2 — Premature Commitment
  Severity:         High
  Evidence:         The key-type decision fires after Stage 1 has already deployed a key —
                    making the choice at position 2 either redundant or contradictory. Even after
                    the reorder from Finding 1, Stage 2 has no declared preconditions: it does
                    not state whether the audience needs the server's supported-algorithm list
                    or the org's key-type policy. The choice fires without scaffolding.
  Remediation:      Declare explicit preconditions: org key-type policy, or a server-capability
                    check if the server restricts algorithms. If no policy exists, supply a
                    safe default (Ed25519) with low reversal cost and allow revision.

---

**Finding 3**
  Stage reference:  Stage 4 — Configure passphrase policy (position 4 of 5)
  Defect class:     Check 2 — Premature Commitment
  Severity:         Medium
  Evidence:         The decision "use passphrase or not?" has no declared preconditions.
                    Whether to use a passphrase depends on whether an SSH agent or password
                    manager is available: without one, a passphrase on an automated-use key
                    makes it unusable for non-interactive sessions. No earlier stage produces a
                    "passphrase-support-check" artifact, and no audience parameter covers it.
  Remediation:      Add a precondition: "environment supports passphrase-protected keys for
                    the intended use (SSH agent available, or key is for interactive use only)."
                    Either add a check stage before Stage 4 or add this to the audience
                    prerequisites declared before the procedure begins.

---

**Finding 4**
  Stage reference:  Stage 1 — Deploy key to server (position 1 of 5)
  Defect class:     Check 3 — Gating Failure
  Severity:         High
  Evidence:         Deploying a key to a server's authorized_keys is high-reversal-cost:
                    removing it requires server access, coordination with configuration
                    management, and potential revocation propagation. Stage 2 (Choose key type)
                    and Stage 3 (Generate keypair) have very-low reversal cost — local files
                    and a decision record only. The high-reversal-cost action executes at
                    position 1 before any of the low-cost scaffolding that would inform it.
  Remediation:      After the reorder from Finding 1, verify the reversal-cost profile reads:
                    choose key type (very-low) → generate keypair (very-low) → configure
                    passphrase (low) → deploy key (high) → test connection (read-only). No
                    Check 3 inversions should remain in the corrected order.

---

**Finding 5**
  Stage reference:  Stage 5 — Test SSH connection (position 5 of 5)
  Defect class:     Check 4 — Hidden Coupling
  Severity:         Medium
  Evidence:         Stage 5 tests the SSH connection, which succeeds only if Stage 1 deployed
                    the correct public key. Stage 5 declares "server address" and "username" as
                    its only inputs — the deployed key is not declared. If Stage 1 ran with a
                    wrong key (e.g., a stale key from audience prior state, as would happen
                    under the original broken order), Stage 5 reports against the wrong key
                    without detecting the mismatch.
  Remediation:      Stage 5 should declare the keypair's public key (Stage 3 exit artifact) as
                    an explicit input and verify the deployed key matches it — not merely that
                    a connection succeeds. This promotes the hidden coupling to a declared
                    dependency and makes the test meaningful.

---

## Cross-references

- [decomposition-fundamentals.md](decomposition-fundamentals.md) — the five structural properties this audit checks adversarially: dependency correctness (Checks 1, 4), reversibility-ordered staging (Check 3), and the precondition discipline behind Check 2.
- [decomposition-smells.md](decomposition-smells.md) — the smell catalog for severity calibration: god-step, mystery-step, orphan-path, premature-fork, and related patterns. A finding should cite the smell it instantiates; the smell's severity guidance overrides this sheet's generic tiers when they conflict.
- [decision-flow-design.md](decision-flow-design.md) — the producer's information-readiness gating discipline that Checks 2 and 3 audit. A decision point that violates Check 2 or 3 has a correspondingly missing or misplaced gate from that sheet's perspective.
- [procedural-boundary-and-handoffs.md](procedural-boundary-and-handoffs.md) — when a finding's remediation requires domain-content judgment (which key type is technically superior, whether the org's tooling supports passphrases), that judgment is out of scope for this audit; document the gap in the remediation field and hand off to the appropriate domain pack.
