---
description: Bootstrap an initial rule set and manifest for a static analyzer. Walks through priority rules to ship in v1, drafts each rule's metadata (id, severity, category, CWE, examples_violation, examples_clean), proposes the manifest's initial enable/severity overrides, and seeds the test corpus. Aligns each rule against the lattice tiers from `02-` and the inference pipeline from `03-`. Emits drafts for `04-rule-plugin-spec.md` rule entries plus a starter manifest. Optionally dispatches the rule-designer agent for narrative drafting per rule.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[analyzer_name_or_path]"
---

# Design Rule Set Command

You are bootstrapping the initial rule set for a static analyzer. The output is a set of rule metadata blocks ready to be added to `04-rule-plugin-spec.md`, a starter `manifest-default.yaml`, and seeded test corpus fixtures (one TP and one TN per rule). This command does NOT implement rule logic; it produces design content and the surrounding infrastructure.

## Invocation Path

`/design-rule-set` is a Claude Code slash command. It runs after `02-` and `03-` are settled (the lattice and the inference pipeline) and after `04-rule-plugin-spec.md` exists in skeleton form (the rule lifecycle, metadata schema, conflict policy). Use this command when:

- The engine is scaffolded (`/scaffold-analyzer`) and you need a starting set of rules.
- An existing analyzer is migrating to this pack's discipline and you are formalising rules that historically lived as ad-hoc functions.
- A new domain (security, performance, internal-policy) is being added and you are bootstrapping its rules.

For the design discipline behind a single rule, use `using-static-analysis-engineering`'s `plugin-architecture-for-analyzer-rules.md`. For interactive lattice design, use `/design-tier-model` first.

## Preconditions

```bash
INPUT="${ARGUMENTS}"
TARGET_DIR=""

if [ -z "${INPUT}" ]; then
  # Use AskUserQuestion to collect:
  # "Which analyzer is this rule set for? Provide a name or a path containing analyzer-engineering/."
  :
fi

ls "${TARGET_DIR}/analyzer-engineering/02-abstract-domain-spec.md" \
   "${TARGET_DIR}/analyzer-engineering/03-inference-pipeline-spec.md" \
   "${TARGET_DIR}/analyzer-engineering/04-rule-plugin-spec.md" 2>/dev/null
```

Required inputs:

- `02-` settled (the lattice; rule severity and consumes-tier specifications need it).
- `03-` settled (the phasing; whether rules consume Phase-1 envs, Phase-2 summaries, or Phase-3 callsite envs).
- `04-` skeleton settled (the rule metadata schema, lifecycle, conflict policy).

If any is missing or stale (older than `99-`), halt and instruct the user to settle the prerequisite via `using-static-analysis-engineering` or `/design-tier-model` before running this command.

If a rule set is partially in place, ask via AskUserQuestion:

1. **Augment** — propose new rules; leave existing untouched.
2. **Refine** — walk through existing rules; propose metadata fixes (missing CWE, no examples, severity mismatch).
3. **Seed-fresh** — start from blank; produce a starter set.

## Workflow

### Round 1 — Rule sourcing

Decide where the initial rules come from. Sources to surface:

- **Domain rule list** (the user has a list of CWEs, OWASP categories, or in-house policy items they want enforced).
- **Existing-analyzer port** (the user is migrating from `pylint`/`semgrep`/`flake8` and wants equivalent rules formalised).
- **Threat-model derivation** (`/security-architect`'s threat model lists invariants to enforce; each becomes a candidate rule).
- **First-principles** (the lattice from `02-` has tiers; ask "what would be a violation of each tier transition?").

Most v1 rule sets have 5–25 rules. More than 25 is suspicious — push back and ask whether some are best deferred.

### Round 2 — Rule sketches

For each candidate rule, elicit four fields up front (the rest can be inferred or deferred):

1. **Plain-English description** — what flow / pattern triggers it?
2. **Phase consumed** — Phase 1 (intra-procedural), Phase 2 (function summary), Phase 3 (callsite env after callgraph propagation)?
3. **Tier (`02-`) consumed** — which tier or join of tiers is the trigger?
4. **Sink shape** — which AST shape is the rule's anchor (a call, an assignment, a return, a class definition)?

Bucket candidates that lack any of these — they are not rules yet, they are wishes. Move them to `04-` deferred section.

### Round 3 — Metadata draft per rule

For each candidate that passed Round 2, draft the metadata block:

```python
RuleMetadata(
    id="STA-XXX",
    name="<short human-readable>",
    severity="error|warning|info|hint",       # default; per-project overridable
    category="taint|capability|style|security|policy",
    description="<paragraph>",
    rationale="<why this rule exists; reference to threat model, CWE, or policy>",
    examples_violation=["<source snippet that should trigger>"],
    examples_clean=["<source snippet that should NOT trigger>"],
    cwe=<int or None>,
    owasp="<string or None>",
    introduced_version="1.0.0",
    deprecated_in=None,
    replaced_by=None,
    tags=frozenset({...}),
)
```

Defaults the command suggests but the user must confirm:

- **Severity** — start at `warning`; promote to `error` only if the rule is high-confidence and security-bearing. New rules at `error` produce immediate adoption friction; warnings let the team grow into them.
- **Category** — drawn from the analyzer's category enum (in `04-`'s taxonomy).
- **CWE / OWASP** — assigned from the threat model. If the user does not have a threat model, leave blank and surface as an open question.
- **Examples** — minimum one TP and one TN. The TP triggers the rule; the TN does not. Both are valid syntactically.

Validate against the `02-` lattice: a rule that consumes a tier the lattice does not have is a contradiction. Surface and halt.

### Round 4 — Optional narrative drafting via agent

For rules that need richer rationale or example pools, dispatch the `rule-designer` agent:

```
Task(subagent_type="rule-designer",
     description="Draft rule metadata for STA-XXX",
     prompt="Given the lattice in analyzer-engineering/02-, the inference pipeline in 03-, and the rule sketch [<sketch>], draft a complete RuleMetadata block plus 3 TP and 3 TN example fixtures. Use the conventions in 04-. Under 600 words.")
```

The agent's output is a draft; the user reviews and adjusts before commit.

### Round 5 — Manifest seeding

Produce `manifest-default.yaml` with:

```yaml
schema_version: "1.0"
analyzer_version_compat: ">=1.0,<2.0"

rules:
  enabled:
    # Every rule from Round 3, with the default severity.
    - id: STA-XXX
      severity: warning
    ...
  # Rules deferred to v1.1+ (per Round 2 bucketing) listed under disabled with rationale:
  disabled:
    - id: STA-DEFERRED-001
      author: <user>
      justification: "Deferred to v1.1; lattice extension required."
      expires: 2026-12-01

ci:
  block_on: [error]
  emit: [sarif, console]
  resolution_rate_threshold: 0.85   # if 07- present
  stub_coverage_threshold: 0.70      # if 08- present
```

The manifest is a starting point. The user adapts it per project; it is the engine's effective configuration when no overlay is present.

### Round 6 — Test corpus seeding

For each rule in Round 3, write fixture files:

```
tests/fixtures/tp/<rule_id>.<ext>
tests/fixtures/tn/<rule_id>.<ext>
```

Fixture content comes from the rule's `examples_violation` (TP) and `examples_clean` (TN). Each fixture is a minimal compilable / parseable file; the test runner exercises the rule against each fixture and verifies expected behaviour.

For taint rules with non-trivial paths, also generate a `tests/fixtures/path/<rule_id>.<ext>` that demonstrates the multi-step flow — the path is the kind of evidence the LLM enrichment in `13-` (if present) consumes.

This is consistency gate check 11. Without it, the rule is unfalsifiable.

### Round 7 — Conflict and subsumption review

Walk the rule set for likely conflicts:

- Two rules that fire on the same node shape with overlapping conditions (e.g., `STA-SQL-INJECT` and `STA-CMD-INJECT` could both fire on `subprocess.run` if the call shape is unusual).
- Rules where one subsumes another (`STA-WIDE-CWE-89` subsumes `STA-NARROW-MYSQL-89`).

Apply the conflict policy from `04-` — typically "highest severity wins" with explicit subsumption markers. Add `subsumes: [...]` fields to rule metadata as needed.

## Output

Emit:

- `analyzer-engineering/04-rule-plugin-spec.md` — appended (or replaced if user chose seed-fresh) with the rule metadata blocks from Round 3.
- `manifest-default.yaml` — at the project root from Round 5.
- `tests/fixtures/tp/*` and `tests/fixtures/tn/*` — from Round 6.
- `analyzer-engineering/04.deferred-rules.md` — bucketed candidates from Round 2 that didn't ship in v1.

Plus a brief report:

- **Rules drafted** (count by category, by severity).
- **Rules deferred** (count, with reasons).
- **Test corpus** (count of TP + TN fixtures generated).
- **Manifest** (path + summary of enabled rules).
- **Open questions** — any field the user left blank (CWE assignments, deferred severity decisions).
- **Next steps** — typically: implement rule logic in `src/<analyzer>/rules/builtin/` against the metadata; run the test corpus; iterate on FPs.

## Failure modes

| Symptom | Cause | Response |
|---------|-------|----------|
| Round 2 produces 50+ candidates | Scope is too broad for v1 | Push back; bucket aggressively; ship v1 with the 5–10 most load-bearing |
| Round 3 reveals a candidate consumes a tier not in `02-` | Lattice is incomplete or candidate is misaligned | Halt; either extend `02-` (and re-gate) or revise the candidate |
| Round 6 cannot construct a TN fixture for a rule | The rule fires on every shape that matches its anchor; the TN doesn't exist | Halt; the rule is over-broad; refine scope before shipping |
| Round 7 surfaces unresolvable conflicts | Two rules genuinely overlap and subsumption isn't natural | Halt; either merge into one rule or split the lattice tier they share |
| User wants to ship a rule with no examples | Bypassing consistency gate check 11 | Refuse; "no examples" = "unfalsifiable" |

## Cross-References

- Router: `using-static-analysis-engineering` — the design pack
- `plugin-architecture-for-analyzer-rules.md` — rule metadata schema, lifecycle, conflict policy
- `taint-lattice-design.md` — the lattice this command's rules consume
- `false-positive-economics.md` — once rules ship, the suppression lifecycle and FP-rate budget begin; cross-link for the operational counterpart
- `/scaffold-analyzer` — must run before this command (the engine must exist)
- `/design-tier-model` — must run before this command if `02-` is not yet settled
- Agent: `rule-designer` — narrative drafting per rule
- Agent: `false-positive-analyst` — once the rule set is shipping, this agent reviews suppressions
- Cross-pack: `ordis-security-architect:design-controls` — when rules enforce control families, align rule taxonomy with control taxonomy
- Cross-pack: `axiom-sdlc-engineering:requirements-lifecycle` — rules are governance-visible artifacts
