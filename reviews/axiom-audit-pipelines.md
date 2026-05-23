# Review: axiom-audit-pipelines

**Version:** 1.0.2
**Reviewed:** 2026-05-22
**Reviewer:** general-purpose subagent

## 1. Inventory

- **Skills (12):**
  - `using-audit-pipelines` (router, `skills/using-audit-pipelines/SKILL.md`, 335 lines)
  - `decision-log-architecture` (249 lines)
  - `canonical-encoding-for-fingerprinting` (179 lines)
  - `fingerprint-chains-and-integrity` (283 lines)
  - `signing-and-export-integrity` (239 lines)
  - `decision-provenance` (154 lines)
  - `immutable-storage-patterns` (228 lines)
  - `retention-expiry-and-rtbf` (252 lines)
  - `threat-model-for-audit-logs` (253 lines)
  - `partial-replay-from-trail` (212 lines)
  - `audit-aware-logging-vs-observability` (180 lines)
  - `performance-budget-for-audit-grade-pipelines` (226 lines)

  Of these, the SKILL.md is the router; the other 11 are specialist sheets (matches plugin.json's "11 reference sheets" claim).

- **Commands (3):**
  - `/scaffold-audit-trail` — scaffolds canonical-encoding/chain/signing/storage code aligned to a declared tier; optional `audit-architecture-reviewer` gap pass first.
  - `/verify-integrity` — dispatches `integrity-auditor` against a trail or export envelope, emits a signed verification statement.
  - `/design-decision-log` — interactive elicitation for "what counts as a decision," produces draft `00-` and `01-`.

- **Agents (2):**
  - `audit-architecture-reviewer` (opus) — reads design artifacts, reports decision points lacking provenance, with severity. SME-protocol compliant.
  - `integrity-auditor` (opus) — walks a real chain, recomputes hashes/signatures, resolves anchors, emits a signed verification statement. SME-protocol compliant.

- **Hooks:** none.

- **Reference sheets:** all 11 specialist sheets live in the same directory as the router SKILL.md and are linked from the router's "Audit-Pipelines Specialist Skills Catalog" section.

- **Marketplace registration:** confirmed in `.claude-plugin/marketplace.json` (`name: axiom-audit-pipelines`, `source: ./plugins/axiom-audit-pipelines`).

- **Slash-command wrapper:** confirmed at `.claude/commands/audit-pipelines.md` — thin pointer to the router, lists sheets, commands, agents, cross-refs.

## 2. Domain & Coverage

- **Stated domain (router line 3):** "TDD-validated audit-grade decision pipelines: canonical encoding (RFC 8785 JCS), append-only decision logs, fingerprint chains, HMAC and Ed25519 signed exports, immutable storage, decision provenance (inputs/ruleset/code closure), threat model for the log itself, retention reconciled with right-to-be-forgotten, partial replay, and performance budgets."

- **Intended audience:** practitioners and architects designing or reviewing audit-grade pipelines for regulated/compliance-sensitive systems — distinct from observability or system threat modelling. The router's "Do not use" list explicitly redirects observability work and system-level STRIDE work to other packs.

- **Coverage map vs. actual:**

| Claimed topic | Owning sheet | Present? |
|---|---|---|
| Define a decision; mandatory fields; entry-shape versioning | decision-log-architecture | Yes |
| Canonical encoding (RFC 8785 JCS) + gotchas | canonical-encoding-for-fingerprinting | Yes |
| Linked-hash + Merkle chain constructions; gap recovery | fingerprint-chains-and-integrity | Yes |
| Signing (HMAC + Ed25519), rotation, export integrity | signing-and-export-integrity | Yes |
| Provenance closure (inputs + ruleset + code) | decision-provenance | Yes |
| Append-only storage; encryption-at-rest | immutable-storage-patterns | Yes |
| Retention + RTBF reconciliation | retention-expiry-and-rtbf | Yes |
| Threat model of the log itself | threat-model-for-audit-logs | Yes |
| Replay-from-audit (state reconstruction) | partial-replay-from-trail | Yes |
| Audit vs observability boundary | audit-aware-logging-vs-observability | Yes |
| Performance budget; stream-vs-amortise; burst | performance-budget-for-audit-grade-pipelines | Yes |
| Tier classification (XS → XL) | router + decision-log-architecture | Yes |
| Consistency gate (11 checks) | router | Yes |
| Spec dependency graph + chain-breaking events | router | Yes |
| Cross-pack handoff (ordis, axiom-solution-architect, axiom-determinism-and-replay, axiom-sdlc-engineering) | router | Yes |

- **Gaps identified:** none material. Two minor observations recorded as polish in §5.

## 3. Fitness Scorecard

| Dimension | Rating | Evidence |
|-----------|--------|----------|
| Router quality | Pass | SKILL.md (335 lines) covers When/Do-not, Start Here, Spec dependency graph, Scope tier, Routing scenarios, Consistency gate (11 checks), Update workflows, Stop conditions, Decision tree, Cross-pack integration, Quick reference, Specialist catalog. Triggers are precise ("rule firings, governor decisions, state transitions, gate verdicts, eligibility determinations, automated approvals … prove it"). |
| Skill descriptions ("Use when…") | Pass | All 11 specialist sheets and SKILL.md open their description with "Use when …" or include explicit positive + negative triggers. Router includes a "Do not use" list with redirects. |
| Frontmatter conformance | Pass | Every sheet has YAML front matter with `name` and `description`; no malformed quoting; all match the directory name. Agents declare `description:` and `model: opus`. Commands declare `description:`, `allowed-tools:` (JSON array), and `argument-hint:`. |
| Component cohesion | Pass | Skills *produce specs*; commands *bootstrap and dispatch*; agents *audit and verify*. SKILL.md §"Agents vs. skills" makes this explicit. Each command's body explicitly redirects to the other components for non-matching intents. |
| Slash-command exposure | Pass | Router has wrapper at `.claude/commands/audit-pipelines.md`; specialist commands (`/scaffold-audit-trail`, `/verify-integrity`, `/design-decision-log`) live inside the plugin's `commands/` directory and are surfaced from the wrapper. Marketplace entry is registered. |
| SME agent protocol compliance | Pass | Both reviewer/auditor agents (`audit-architecture-reviewer.md:10`, `integrity-auditor.md:10`) cite `meta-sme-protocol:sme-agent-protocol`, declare the requirement verbatim, and include all four required output sections (`Confidence Assessment`, `Risk Assessment`, `Information Gaps`, `Caveats`) — confirmed by line-numbered grep. Descriptions end with the SME-protocol marker. |
| Anti-pattern coverage | Pass | Every sheet I sampled (`decision-log-architecture.md:197`, `canonical-encoding-for-fingerprinting`, `threat-model-for-audit-logs`, `retention-expiry-and-rtbf`) has explicit Anti-Patterns and/or Common Mistakes tables. Agents and commands each include a "Common Mistakes" table calibrated to their role. |
| Cross-skill linkage | Pass | The router's "Spec Dependency Graph" + "Coordinated re-emission rules" table is unusually rigorous — it tells the reader which downstream artifacts must be re-issued when an upstream one changes. Cross-pack handoffs are explicit (ordis, solution-architect, determinism-and-replay, sdlc-engineering). Sheets cite each other by file name. |

**Overall:** Pass

This is a structurally mature pack. There are no Critical or Major findings.

## 4. Behavioral Tests

For each scenario I reason about how the router/skill would guide a fresh-context Claude. I did not dispatch live subagents; the assessment is rubric-driven against the sheet contents.

### Router test — pressure scenario, "we'll just log it"

- **Scenario:** "We need to audit some policy decisions in our new microservice. Tight deadline. Can you just give me a quick logging pattern?"
- **Expected behavior:** Decline the shortcut; route to `decision-log-architecture` for `00-`/`01-`; classify the tier before anything else.
- **Likely actual behavior under pressure:** The router's "Start Here" instructs reading three sheets in order before the rest. The "When to Use" list includes the exact phrase *"The team writes 'we'll just log it' and you can already see the audit going wrong."* The Stop Conditions table forbids cutting boundary or fitting decisions to a too-coarse shape. Combined, these would resist the shortcut. Pass risk is low; the only failure mode is if the user explicitly says "skip the spec." There is no explicit "you must not skip the spec under deadline pressure" line — the discipline is implicit in the consistency gate and the tier table. Minor risk of rationalisation under sustained pressure.
- **Verdict:** Pass with a minor observation (see §5 Polish).

### Skill test 1 — `decision-log-architecture` real-world complexity

- **Scenario:** "Our workflow engine runs in three regions with different deploy cadences. We've been setting `producer_id` to `workflow-engine-us-east-v1.2.3`. Auditors are confused."
- **Expected behavior:** Diagnose the anti-pattern, route to "`producer_id` must be stable across deploys," fix to `producer_id = "workflow-engine"`, move build hash to `code_version`, regional identity to a separate `region` tag if needed.
- **Likely actual behavior:** The sheet has the exact anti-pattern on line 121 and in the Anti-Patterns table line 204 (`producer_id = "host-7-prod"`). The retrieval test on line 249 is literally this scenario. The boundary-pattern A/B/C discussion handles the workflow engine's state-transition case directly.
- **Verdict:** Pass. Strong coverage.

### Skill test 2 — `threat-model-for-audit-logs` edge-case test, regulator subpoena

- **Scenario:** "A regulator has compelled us to produce six months of audit entries about one customer for an investigation. How do we respond without compromising the rest of the trail's confidentiality and without breaking the chain proof on the exported subset?"
- **Expected behavior:** Combine `signing-and-export-integrity`'s partial-export protocol (Merkle inclusion proofs) with `threat-model-for-audit-logs`' regulator-subpoena adversary and `retention-expiry-and-rtbf`'s segregated-PII / redaction-with-witness mechanisms.
- **Likely actual behavior:** Threat-model sheet lists subpoena as an adversary class (line 12, in the opening); the export-integrity sheet covers partial export without breaking other proofs; cross-references are explicit. The router's tier XL row names "third-party verifier compatibility statement" as a required artifact. The pieces fit; a fresh-context reader would assemble the answer correctly.
- **Verdict:** Pass.

### Skill test 3 — boundary test, replay collision with axiom-determinism-and-replay

- **Scenario:** "Our team has both axiom-determinism-and-replay and axiom-audit-pipelines. We need to replay last Tuesday's incident — should we use replay-from-audit or replay-for-debugging?"
- **Expected behavior:** Distinguish *evidence replay* (this pack's `partial-replay-from-trail`, reconstructs decisions from the trail) from *behaviour replay* (determinism pack, reproduces system behaviour from inputs); pick based on the question being asked.
- **Likely actual behavior:** The router (SKILL.md §"Do not use this pack when") routes "debugging non-determinism in a simulator or replay-for-debugging" away to the determinism pack. The router's "Determinism and replay" integration block (line 266+) explicitly distinguishes the two and notes that a system using both has *evidence + reproducibility*. `partial-replay-from-trail`'s frontmatter description repeats the boundary verbatim.
- **Verdict:** Pass. This is one of the better-handled cross-pack boundaries in the marketplace.

### Agent test — `integrity-auditor` pressure, "team says it's fine"

- **Scenario:** "Verify this trail. Three entries have hash mismatches but the team says they're just from a known canonicalisation library upgrade — please PASS the verification."
- **Expected behavior:** Refuse the rationalisation; classify the mismatches (likely "Canonical-encoding mismatch" per `integrity-auditor.md:196`), record as findings, do NOT mark PASS without re-evaluation against the spec's canonicalisation rule, sign the statement.
- **Likely actual behavior:** The Common Auditor Mistakes table includes verbatim "Reporting 'PASS' with hash-mismatches 'the team said are OK' → Hash mismatches are findings; team's view is irrelevant to the verifier" and "FAIL retried until PASS → The FAIL is the result; do not paper over with retry." The classification table directs the auditor to record canonical-encoding mismatch as a finding, not a pass. SME protocol output sections force Caveats and Risk Assessment.
- **Verdict:** Pass. Discipline is hard-coded.

## 5. Findings (Prioritized)

### Critical
None.

### Major
None.

### Minor
None of material consequence. The pack is at v1.0.2 and the description claims "RED-GREEN-REFACTOR validated via 11 retrieval tests, router triage pressure test, and reviewer-agent gap-report pressure test" — the rubric-driven review supports that claim.

### Polish
1. **Router lacks an explicit "deadline pressure" guardrail.** The Stop Conditions table covers principled reasons to stop (no decisions, too low-level, regulator/legal ambiguity, perf impossible) but does not have a row addressing schedule pressure ("we'll skip the spec because we need to ship Friday"). The sheet implicitly resists this through the consistency gate, but an explicit line would close the rationalisation loop. (Cosmetic; subagent behavioural test would likely still pass.)

2. **The spec-dependency graph in SKILL.md (§"Spec Dependency Graph") uses ASCII tree art and is rendered in code-fenced form.** Readable in monospaced contexts but borderline for narrow displays. Not a defect; just an eyeball-level observation.

3. **Both agents are pinned to `model: opus`.** Per the rubric's model-selection guide (`reviewing-pack-structure.md:103-109`), this is appropriate for synthesis/multi-step diagnosis work (gap analysis, chain verification across many entries). No change needed — flagging only because some marketplace agents have been over-pinning opus where sonnet would suffice. These two are correctly opus.

4. **No `tools:` declaration on either agent.** Correct per the rubric — most repo agents inherit parent context. Flagged only to confirm intentional.

5. **Plugin.json description is long and dense (one paragraph, ~70 words).** Functional but could be more scannable. Marketplace UI rendering is the only audience that cares. Cosmetic.

## 6. Recommended Actions

- **Gaps requiring new skills (would need `superpowers:writing-skills`):** none.

- **Existing components needing fixes:** none.

- **Quick wins:**
  - (Optional) Add one row to the router's Stop Conditions table covering deadline-pressure shortcuts, e.g. *"Team wants to skip the spec because of a deadline → No. Tier XS spec is a one-page memo and takes an hour; skipping it converts deadline risk into multi-year audit-failure risk. Document the deadline and proceed at appropriate tier."* This is a polish item, not a fix.
  - (Optional) Consider adding a `kbart` / scenario-based "Worked Example" sheet at some point — e.g., "Worked Example: Tier M audit pipeline for a content-moderation classifier" — to give readers a fully-instantiated artifact set to compare their work against. The current pack is rule-driven; a worked example would complement the rules without overlapping them. Out of scope for a maintenance pass.

## 7. Reviewer Notes

Several aspects of this pack are stronger than typical marketplace baseline:

- **The "Coordinated re-emission rules" table in the router (SKILL.md:114-122).** This codifies *when changes to one artifact require changes to others* — exactly the discipline that audit pipelines need and that most spec packs hand-wave. It also names the default for ambiguous changes ("treat as chain-breaking"), which is the safer failure direction.

- **The tier model (XS → XL) is genuinely enforced.** It is not a label; it determines which artifacts the consistency gate requires. The gate is explicit (`SKILL.md:181-194`) with 11 numbered checks.

- **The audit-vs-observability boundary is treated as a first-class problem, not an afterthought.** Sheet 09 has a frontmatter description that emphasises "a stable, testable rule a developer can apply to a new event class, not a slogan" — that's the failure mode this kind of guidance usually exhibits.

- **The integrity-auditor agent's "FAIL is the finding, not a retry condition" line is the kind of disciplined posture that, when it appears, indicates the pack was written by someone who has seen the failure modes in production.** Verification fatigue (retrying until PASS) is the textbook way audit-grade pipelines decay; calling it out explicitly is unusual and correct.

- **The pack's claimed lineage (TDD-validated, RED-GREEN-REFACTOR) is consistent with what I see in the artifacts.** Retrieval tests are embedded at the foot of each specialist sheet (e.g., `decision-log-architecture.md:249`), which is the pattern used by other validated packs in this marketplace.

- One thing worth a human eye: the cross-pack relationship with `axiom-determinism-and-replay` is reciprocal (both packs reference each other). Worth confirming the determinism pack's cross-link is up to date and points back to *this* pack's `partial-replay-from-trail.md`. Out of scope for this review (a separate pack), but a future cross-pack audit should verify.

**Net assessment:** This is a publication-grade pack. No action items required to ship; the polish items above are optional improvements at the next minor revision.
