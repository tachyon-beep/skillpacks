---
name: manifest-driven-configuration-with-coherence-validation
description: Use when an analyzer needs *configuration* — which rules are enabled, at what severity, with what suppressions, with what stubs, with which framework recognitions, and which monkey-patches and decorator declarations are in play — and you need that configuration to be auditable, layered (workspace → project → package), and validated for coherence (a manifest cannot disable a rule it depends on, override a stub for a library not installed, declare a tier the lattice doesn't have). Covers the manifest schema, layering and overlays, validation passes, drift detection between manifest and codebase reality, and the manifest-as-decision discipline. Produces `10-manifest-and-coherence.md`.
---

# Manifest-Driven Configuration with Coherence Validation

## Why a Manifest, Not Code

Most analyzers start with configuration in code: `enabled_rules = ["STA001", "STA002"]` somewhere, severity overrides scattered through CI scripts, suppressions inline in source. This is fine for one project; it is rapidly catastrophic at scale. Three failures recur:

- **Configuration drift** — the rule is disabled in CI but enabled in pre-commit; the suppression is in `.flake8` *and* `# noqa: STA001` *and* `pyproject.toml`. Nobody knows which is authoritative.
- **No audit trail** — who disabled rule STA014 in this repo, and why? Buried in a six-month-old commit message, if it is documented at all.
- **No coherence checking** — the manifest disables STA005, which depends on STA001 being enabled; both are now silently ineffective. Nobody is told.

A manifest moves configuration to a single, declarative, versioned, machine-readable artifact. It is the analyzer's `02-` for configuration: an explicit lattice of decisions, validatable, auditable.

`10-manifest-and-coherence.md` defines that manifest's schema and the validation discipline that keeps the manifest aligned with the analyzer engine and the codebase reality.

## What Goes in the Manifest

The manifest is the union of every per-project decision the analyzer respects:

| Section | Contents | Example |
|---------|----------|---------|
| **Rule enablement** | Which rules run; per-rule severity overrides | `STA001: severity=error, enabled=true` |
| **Suppressions** | Per-location waivers (machine-readable form of `# noqa`) | `app/db.py:14 STA001 expires=2026-09-12 by=alice@team` |
| **Stub overrides** | Project-specific stubs replacing library defaults | `requests.get → custom-stubs/requests_get.py` |
| **Framework recognitions** | Project-specific framework idioms | `entrypoints: app.handlers.* → request handler entry points` |
| **Monkey-patch declarations** | `Class.method = …` patches the analyzer should know about | `MyClass.execute → safe_module.safe_execute` |
| **Decorator registry** | Decorators-as-assertion the analyzer recognises | `app.security.requires_capability → STA-CAP-001` |
| **Lattice tier overrides** | Per-project additions to the standard lattice (rare; in tier L+) | Adds tier `tenant_isolated` between `clean` and `top` |
| **CI gate behaviour** | Exit code semantics, severity threshold for blocking, SARIF emission options | `block_on=error,warning; emit=sarif` |
| **Resolution rung** | Override the default callgraph rung for this project | `callgraph.rung=2` (RTA) |
| **Stub-coverage budget** | Project-specific target for `08-` stub coverage | `stub_coverage_target=0.80` |

The manifest is **declarative**: it says *what* the configuration is. It does not contain logic; logic stays in code. A manifest with code-equivalent expressions (string-templated rules, conditional enables) is a sign that more of it should be code, or that the engine needs new declarative knobs.

## Layering

In any non-trivial codebase, configuration comes from multiple sources. The manifest's layering rules state how they compose.

A standard four-layer model:

```
1. ENGINE-DEFAULT      (shipped with the analyzer)
       ▼ overlay
2. WORKSPACE-MANIFEST  (organisation-wide policy)
       ▼ overlay
3. PROJECT-MANIFEST    (per-repository configuration)
       ▼ overlay
4. PACKAGE-MANIFEST    (per-subpackage local override; rare)
       ▼ overlay
5. INLINE              (per-line `# noqa` style)
```

Each layer **overlays** the layer above. Overlay semantics:

- **Disable beats enable** — once a rule is disabled, lower layers cannot re-enable it (an overlay can disable but not un-disable). This is conservative: a security-bearing rule disabled at workspace level cannot be silently re-enabled per project.
- **Severity narrows** — a higher layer's severity is the floor; a lower layer can raise but not lower (otherwise, an audit-disabled-error could become a warning by overlay).
- **Stubs and recognitions accumulate** — lower layers add; they do not remove. Removal requires explicit `unstub` / `unrecognise` directives, which fail validation if the target wasn't present.
- **Suppressions are local** — they apply only at their declared layer; they are never inherited.

These rules are conservative by default. State the policy in `10-`; some teams deliberately invert "disable beats enable" for engineering teams that need to opt-out, but the inversion has security implications and must be deliberate.

## Schema Discipline

A manifest without a schema is a YAML file. The schema is the contract.

### Schema declaration

The manifest declares its schema version explicitly:

```yaml
schema_version: "1.0"
analyzer_version_compat: ">=2.0,<3.0"
```

The engine:

- Refuses to load a manifest with a `schema_version` it does not understand.
- Refuses to load a manifest whose `analyzer_version_compat` does not match.
- Reports the schema version of every manifest in its diagnostic output.

### Validation passes

When the engine loads the manifest stack, it runs validation passes in order. Each pass either passes or produces structured errors.

| Pass | Checks |
|------|--------|
| **Syntax** | YAML parses; schema_version present; required fields present |
| **Schema** | Every key validates against the schema; values are well-typed; enums are in range |
| **Reference** | Every rule ID, stub target, framework name, decorator name, lattice tier referenced exists in the engine or in a lower layer |
| **Coherence** | No contradictions: rule is not both `enabled=false` and overridden to `severity=error`; no two stubs target the same library function in the same layer; no monkey-patch declaration with no matching codebase site |
| **Drift** | Manifest entries match codebase reality: declared monkey-patches exist; declared framework entry points are present; stub overrides match installed library versions |
| **Lifecycle** | Suppressions have valid expiries; expired entries are flagged; entries with deprecated rule IDs are flagged |

Each pass's output is structured (rule ID, location in manifest, error class). The engine emits a manifest-validation report alongside the analyzer's findings; CI gates on it.

### Coherence checks in detail

Coherence is the pass most teams skip and most need.

- **No silent dependency breaks.** If rule `STA005` declares `depends_on: [STA001]` and the manifest disables `STA001`, `STA005` either errors or auto-disables (with a warning). Silent ineffectiveness is the failure mode.
- **No tier reference to non-existent tier.** A suppression refers to a tier the lattice no longer has (after a `02-` change) — the suppression is orphaned and must be flagged.
- **No stub for uninstalled library.** A stub override targets `kafka.producer.send`, but the project's `pyproject.toml` does not depend on `kafka`. Either the dependency is missing or the stub is dead. Either way, surface it.
- **No decorator declaration with no decorator.** A decorator-as-assertion entry names `app.security.requires_capability` but no such decorator exists in the codebase. Flag.
- **No mutually exclusive overlays.** A child manifest enables a rule the parent disabled (and "disable beats enable" is the policy). Flag, refuse to apply.

These are not all errors at all severities; the team picks which are blocking. The discipline is the *checking*, not the *severity*.

## Drift Detection

Coherence checks the manifest internally; drift checks the manifest against codebase reality. The two are complementary.

| Drift | Detected by | Action |
|-------|-------------|--------|
| Declared monkey-patch site doesn't exist | Scan codebase for the declared assignment; not found → flag | Investigate; either the code changed or the manifest is stale |
| Declared framework entry point doesn't exist | Scan for the decorator/pattern; not found → flag | Same |
| Stubbed library not installed | Read `pyproject.toml` / `package.json`; absent → flag | Either install or remove stub |
| Library version outside stub range | Read installed version; mismatch → flag | Update stub or pin library |
| Suppression at non-existent location | Resolve suppression's file:line; missing → flag | Suppression is stale; remove |
| Rule ID renamed | Engine recognises old ID maps to new; not in current rule set → flag | Update manifest |

Drift findings are emitted in the same channel as analyzer findings. CI behaviour: drift in suppressions is high-priority (they're rotted waivers); drift in stubs is medium (they may be silently ineffective); drift in monkey-patch declarations is high (silent unsoundness).

## Manifest as Audit Artifact

A manifest is an audit-grade artifact. Each entry is a procedural decision: someone decided STA014 should not run in this repo, with stated rationale, at a stated time, on a stated authority. This is the same evidence regime as suppressions (see `false-positive-economics.md` and `axiom-audit-pipelines`).

Implications:

- **Manifest in version control** — yes, with full diff history. Manifest changes are reviewed like code.
- **Manifest signed at deploy** — for high-assurance environments, the manifest hash is signed and embedded in the analyzer run's evidence; the analyzer refuses to run with an unsigned manifest in those environments.
- **Manifest expiry on entries** — consistent with suppression lifecycle; a rule disablement may expire after 12 months and require re-justification.
- **Manifest provenance fields** — author, review date, justification — on every disablement, severity override, stub override.

For tier L/XL analyzers, treat the manifest as the audit pack treats decision logs: canonical encoding, append-only history, signed exports. For tier S/M, version control + reviewed diffs are typically enough.

## Tooling: Diff, Effect, Dry-Run

The manifest is layered; the *effective* configuration is what the engine computes after applying overlays. Three tools the engine should ship:

### `analyzer config diff`

Shows what changed between two effective configurations. Used to review a manifest PR: "this PR disables STA014 and adds a stub for `kafka.producer.send`" rather than line-noise YAML diffs.

### `analyzer config show --effective`

Computes and prints the effective configuration: the post-overlay state. Resolves disagreements about "is this rule on or off?"

### `analyzer config dry-run`

Runs the analyzer with the proposed manifest, reports what would change in findings (count, which rules, which files). Used to triage a proposed manifest change: "if we disable STA014, we lose 47 findings; here are five of them — are any real?"

These are the manifest's debugging aids. Without them, "the manifest is correct" is a hope; with them, it is verifiable.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Manifest without schema | Cannot tell what fields are valid; engine accepts garbage | Schema declaration; validation pass on load |
| No layering rules | Conflict resolution is whoever-wrote-the-last-overlay | Explicit overlay semantics; "disable beats enable" or stated alternative |
| No coherence checking | Silently ineffective rules; orphaned suppressions; stubs for absent libraries | Coherence pass on every load; CI gate |
| No drift detection | Manifest rots away from codebase reality | Drift pass on every load; emit findings |
| Manifest in code, not declarative | Configuration changes require code review of analyzer's own logic | Move to YAML/TOML/JSON; treat as data |
| `# noqa` overriding manifest invisibly | Inline suppressions hide manifest-level decisions | Inline is layer 5; record in manifest stack so `config show --effective` includes it |
| Stub overrides without version pinning | Stubs misalign with installed library versions | Stub entry includes library version range; drift pass enforces |
| No audit metadata on manifest entries | Cannot tell who disabled what or why | Mandatory `author`, `justification`, `expires` per entry that disables or overrides defaults |
| `analyzer config show --effective` not implemented | Team disagrees about what the configuration is; no resolution mechanism | Implement; gate on it in CI |
| Manifest changes don't trigger drift re-check | Drift detected once and never again | Run drift on every analyzer invocation, not just on manifest change |

## The Manifest File Layout

A canonical example structure (one section per concern; engine-specific names will vary):

```yaml
schema_version: "1.0"
analyzer_version_compat: ">=2.0,<3.0"

rules:
  enabled:
    - id: STA001
      severity: error
    - id: STA042
      severity: warning
  disabled:
    - id: STA014
      author: alice@team
      justification: "Tier model does not include `cache` tier; rule is noise."
      expires: 2026-12-01

suppressions:
  - location: app/db.py:14
    rule: STA001
    author: bob@team
    justification: "html_escape applied at line 12; rule cannot see call due to dynamic dispatch."
    expires: 2026-09-12

stubs:
  overrides:
    - target: requests.get
      source: custom-stubs/requests_get.py
      author: charlie@team
      library_version: ">=2.27,<3.0"
      review_date: 2026-04-01

framework_entry_points:
  - name: app.handlers.*
    pattern: "^app\\.handlers\\.[A-Z][A-Za-z]+$"
    input_shape: request_handler

monkey_patches:
  - target: legacy.SafeCursor.execute
    replacement: safe_module.safe_execute
    author: dana@team
    justification: "Legacy code path; replacement vetted for safety."
    expires: 2027-01-01

decorators:
  - name: app.security.requires_capability
    aliases: [requires_capability]
    metadata_attr: __required_caps__
    arg_extraction: positional[0]
    rule_trigger: STA-CAP-001

callgraph:
  rung: 2
  refinement: iterated

ci:
  block_on: [error, warning]
  emit: [sarif, json]
  resolution_rate_threshold: 0.85
```

Yours will differ; the *shape* — sections, schema version, audit fields per entry, validation passes — is what matters.

## The Decision Output (`10-manifest-and-coherence.md`)

A complete `10-` answers:

1. **Manifest format** — YAML / TOML / JSON; rationale.
2. **Schema** — explicit; versioned; published.
3. **Layering** — number of layers; precedence; overlay semantics; "disable beats enable" or alternative.
4. **Required fields** — every section's required and optional fields, with audit metadata where applicable.
5. **Validation passes** — syntax, schema, reference, coherence, drift, lifecycle; what each pass checks; severity policy.
6. **Coherence rules** — explicit list of "X is not coherent with Y"; not derived from intuition.
7. **Drift rules** — what counts as drift; how detected; CI behaviour.
8. **Tooling** — `config diff`, `config show --effective`, `config dry-run`; expected interface.
9. **Audit metadata** — required fields per entry that disables or overrides defaults; expiry policy; author identity.
10. **Cross-link to suppressions** — the manifest's suppressions section follows `false-positive-economics.md`'s lifecycle; cross-link rather than duplicate.
11. **High-assurance variant** — for tier L/XL, signed manifests, append-only history, audit-pipeline integration.

## Cross-References

- `plugin-architecture-for-analyzer-rules.md` — rules' metadata schema; the manifest references rule IDs declared there
- `false-positive-economics.md` — suppression lifecycle; the manifest's suppressions section
- `taint-lattice-design.md` — lattice tiers; the manifest may extend tiers per project
- `callgraph-construction.md` — resolution rung; decorator registry; monkey-patch declarations
- `cross-module-flow-analysis.md` — stub overrides; framework recognition entries
- `decorator-as-assertion.md` — decorator registry's recognition entries
- `sarif-emission-and-ci-integration.md` — CI gate behaviour declared in the manifest
- `scaling-to-large-codebases.md` — incremental analysis configuration in the manifest
- Cross-pack: `axiom-audit-pipelines:decision-log-architecture` — the manifest as an audit-grade decision log in tier L/XL
- Cross-pack: `axiom-sdlc-engineering:requirements-lifecycle` — manifest changes are governance-visible artifacts
