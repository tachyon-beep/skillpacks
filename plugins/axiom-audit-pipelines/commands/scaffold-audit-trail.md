---
description: Scaffold an audit-trail implementation aligned to a declared tier. Drops in canonical-encoding, chain construction, signing, and storage scaffolding consistent with `axiom-audit-pipelines` specs. Optionally runs a gap-analysis pass via the audit-architecture-reviewer agent before scaffolding.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[component_name_or_path]"
---

# Scaffold Audit Trail Command

You are scaffolding audit-trail implementation for a component. The output is *implementation scaffolding* (code files, configuration, schema definitions) that align with a previously-produced or about-to-be-produced `audit-pipeline/` specification. This command does NOT replace the design specs in the `using-audit-pipelines` skill; it implements them.

## Invocation Path

`/scaffold-audit-trail` is a Claude Code slash command. It dispatches the specialist sheets in `axiom-audit-pipelines` to determine the right scaffolding shape, optionally calls the `audit-architecture-reviewer` agent to find decision points lacking provenance, and emits skeleton code + configuration.

For a clean design pass without code, use the `using-audit-pipelines` skill directly. For verification of an existing trail, use `/verify-integrity`. For an interactive walkthrough of "what counts as a decision," use `/design-decision-log`.

## Preconditions

The command takes a single argument: a component name (string) or a path to a component directory.

### Resolve the argument

```bash
INPUT="${ARGUMENTS}"

if [ -z "${INPUT}" ]; then
  # Use AskUserQuestion to collect:
  # "Which component or service is this audit pipeline for?
  #  Provide a name (e.g., 'policy-engine') or a path (e.g., 'services/policy/')."
  :
fi

# Path → verify directory
if [ -d "${INPUT}" ]; then
  echo "Scaffolding into: ${INPUT}"
elif [ -f "${INPUT}" ]; then
  echo "ERROR: ${INPUT} is a file. Provide a directory or a component name."
  exit 1
else
  echo "Treating as component name: ${INPUT}"
  echo "Will create audit-pipeline/${INPUT}/ for design specs."
fi
```

### Check for existing audit-pipeline workspace

```bash
ls audit-pipeline/ 2>/dev/null
```

If `audit-pipeline/` exists with the 12-artifact set (00–10, 99) for this component, this command consumes those specs and emits scaffolding consistent with them. If specs are absent, the command runs the design pass first via the `using-audit-pipelines` skill.

**Resume vs fresh protocol:**

If `audit-pipeline/<component>/` already has scaffolded files (e.g., source code, schema definitions, migration scripts), ask the user via AskUserQuestion:

1. **Augment** — fill in missing scaffolding pieces, leave existing alone.
2. **Replace** — archive existing `audit-pipeline/<component>/` to `audit-pipeline/<component>.bak.YYYY-MM-DD/`, scaffold fresh.
3. **Targeted** — scaffold a single specific layer (chain, signing, storage), leave others.

If no scaffolding exists, proceed without asking.

## Workflow

### Step 1 — Confirm or run the design pass

Check for the 12 artifacts in `audit-pipeline/<component>/`:

```
00-scope-and-decisions.md
01-decision-log-schema.md
02-canonical-encoding-spec.md
03-chain-and-integrity-spec.md
04-signing-and-export-spec.md
05-provenance-bindings.md
06-storage-and-retention.md
07-threat-model.md
08-replay-capability.md
09-audit-vs-observability-boundary.md
10-performance-budget.md
99-audit-pipeline-specification.md
```

If any are missing:

- For tier XS or S, some artifacts are not required — check the Scope Tier table in `using-audit-pipelines/SKILL.md`.
- For required-but-missing artifacts, run the corresponding skill from the catalog before scaffolding.
- Do NOT scaffold against an incomplete spec. The scaffold's choices come from the spec; absent specs make those choices arbitrarily.

### Step 2 — Optional gap analysis

Before scaffolding, dispatch the `audit-architecture-reviewer` agent against any existing system design (HLD, SAD, code map) to find decision points that should be in the trail but aren't yet:

```
Use the Task tool with subagent_type: "audit-architecture-reviewer"
Provide: the component's design artifacts, the 00- and 01- specs
Receive: gap report — decision points without provenance, severity-rated
```

Incorporate gap-report findings into `00-scope-and-decisions.md` and `01-decision-log-schema.md` before scaffolding.

### Step 3 — Scaffold per spec

Generate scaffolding consistent with the spec:

| Spec section | Scaffolds into |
|--------------|----------------|
| `02-canonical-encoding-spec.md` | Canonical-encoding wrapper module; pinned library; CI test vectors |
| `01-decision-log-schema.md` | Schema definition (JSON Schema, protobuf, Avro per spec); validator; entry-construction helper |
| `03-chain-and-integrity-spec.md` | Chain-link computation; `prev_hash`/`entry_hash` helpers; gap-marker scaffold |
| `04-signing-and-export-spec.md` | Signing client (KMS, HSM, application-managed); `key_id` handling; export-envelope template |
| `06-storage-and-retention.md` (storage) | Database schema or object-store client config; INSERT-only credential setup; backup config stub |
| `06-storage-and-retention.md` (retention) | Retention timer scaffold; cryptographic-erasure key registry stub if applicable |
| `08-replay-capability.md` | Replay tool skeleton with deterministic mode; coverage-statement schema |

### Step 4 — Wire integrations

Wire the scaffolding into the component:

- The decision producer's main entry point routes decisions through the audit-entry-construction helper before acknowledging.
- The producer's deployment configuration includes audit-storage credentials (INSERT-only), KMS / signing service identity, and external-anchor configuration if periodic anchoring is in scope.
- The component's existing observability emission is reviewed against `09-audit-vs-observability-boundary.md` — gray-zone events are dual-written.

### Step 5 — CI hooks

Generate CI configuration:

- RFC 8785 test vectors (or the canonical-encoding equivalent) run on every commit.
- Application-user INSERT-only assertion: a test that attempts UPDATE/DELETE on audit storage and asserts failure.
- Schema-validation test: every test that produces a decision validates the resulting entry.
- Signature-verification test: a written entry round-trips through the verifier.

### Step 6 — Run the consistency gate

Invoke the consistency-gate procedure from `using-audit-pipelines/SKILL.md`. Each check produces a pass/fail in the gate report. Failures are addressed before declaring scaffolding complete; this command does NOT bypass the gate.

## Output Location

Specs land in `audit-pipeline/<component>/`. Code scaffolding lands in the component's existing source tree at the appropriate path (`src/audit/`, `lib/audit/`, or per the component's convention). Configuration lands wherever the component holds configuration (a `audit-config.toml`, environment-variable schema, etc.).

## Downstream Handoffs (suggest after completion)

- Verification — `/verify-integrity` against the scaffolded pipeline once it has produced entries.
- Threat-model alignment — coordinate `07-threat-model.md` with `ordis-security-architect`'s system threat model.
- SDLC governance — register `99-audit-pipeline-specification.md` with `axiom-sdlc-engineering` for lifecycle ownership.
- Observability boundary — `09-audit-vs-observability-boundary.md` reviewed by the observability owner.

## Scope Boundaries

Covered: design pass through the spec, optional gap analysis, code/config scaffolding, CI hooks, consistency gate.

Not covered: production deployment, key-custody operational provisioning (KMS / HSM bootstrap), operator-account separation (typically owned by ops/security), choice of storage provider beyond what the spec declares.

## Common Mistakes (in scaffolding)

| Mistake | Fix |
|---------|-----|
| Scaffolding generated before specs are complete | Halt; complete the spec; come back |
| Scaffolding hard-codes choices not in spec | Choices come from spec; if undocumented, the spec is incomplete |
| Application user has UPDATE permission on audit tables | INSERT-only enforced by DB; CI test asserts failure |
| Canonical-encoding library not pinned | Pin the version; CI tests RFC vectors |
| Signing key created locally, no KMS wiring | KMS-issued at tier M+; runbook required |
| Storage scaffolding writes plaintext at L+ | Application-side encryption with KMS-managed keys |
