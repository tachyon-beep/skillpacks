---
name: sarif-emission-and-ci-integration
description: Use when the analyzer must integrate with CI / code-review tooling — GitHub Code Scanning, GitLab Code Quality, Azure DevOps, Jenkins, IDE extensions — and the integration discipline matters: SARIF (Static Analysis Results Interchange Format) as the lingua franca, exit-code semantics that distinguish "error", "infrastructure failure", and "config error", suppression round-tripping (manifest waivers ↔ SARIF suppressions ↔ inline comments), severity mapping, fingerprint stability across runs, baseline comparison, and the GitHub Code Scanning idioms that the format does and does not specify. Produces `11-sarif-and-ci.md`.
---

# SARIF Emission and CI Integration

## Why a Sheet for the Output Format

The analyzer produces findings; CI consumes findings. The pipe between them is mostly invisible until something goes wrong, at which point it becomes the entire problem. Three failure modes recur:

- **Findings render but don't gate** — CI reports findings as a comment, not as a blocking status. Developers learn to ignore them.
- **Findings gate but don't render** — CI fails the build with no actionable output. Developers re-run, eventually find the log, eventually find the rule.
- **Suppressions don't round-trip** — a `# noqa` in source is invisible to GitHub Code Scanning, which re-flags every suppressed finding on every PR.

`11-sarif-and-ci.md` is where you commit to the output format, the exit-code contract, and the integration discipline that keeps CI honest.

SARIF is the standard target; the discipline is broader than SARIF.

## SARIF in One Page

SARIF is a JSON schema (OASIS, current major version 2.1.0) that every serious code-scanning consumer understands: GitHub Code Scanning, GitLab, Azure DevOps, Sonatype, Snyk, JetBrains IDE, VSCode extensions. Emitting SARIF is the cheapest way to be a citizen of the modern CI ecosystem.

The structure (compressed):

```json
{
  "version": "2.1.0",
  "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json",
  "runs": [{
    "tool": {
      "driver": {
        "name": "wardline",
        "version": "2.3.1",
        "informationUri": "https://example.com/wardline",
        "rules": [
          {
            "id": "STA001",
            "name": "SqlInjection",
            "shortDescription": {"text": "Untrusted value flows to SQL execute"},
            "fullDescription": {"text": "..."},
            "helpUri": "https://example.com/wardline/rules/STA001",
            "defaultConfiguration": {"level": "error"},
            "properties": {
              "category": "taint",
              "tags": ["security", "cwe"],
              "cwe": ["CWE-89"]
            }
          }
        ]
      }
    },
    "results": [
      {
        "ruleId": "STA001",
        "level": "error",
        "message": {"text": "Untrusted value flows to SQL execute at line 14"},
        "locations": [{
          "physicalLocation": {
            "artifactLocation": {"uri": "app/db.py"},
            "region": {"startLine": 14, "startColumn": 8, "endLine": 14, "endColumn": 27}
          }
        }],
        "partialFingerprints": {
          "primaryLocationLineHash": "deadbeef..."
        },
        "codeFlows": [...],         // optional: the flow path from source to sink
        "suppressions": [
          {"kind": "external", "justification": "..."}
        ]
      }
    ],
    "invocations": [{"executionSuccessful": true, "exitCode": 0}],
    "originalUriBaseIds": {"%SRCROOT%": {"uri": "file:///path/to/repo"}}
  }]
}
```

Key consumer expectations:

- **Schema version present** — consumers refuse to ingest SARIF without `"version": "2.1.0"`.
- **Tool driver fully specified** — name, version, rules. Consumers use this for de-duplication and tooling-aware presentation.
- **Rule metadata present** — once per rule, not per finding. Consumers index by rule.
- **`partialFingerprints`** — load-bearing for suppression tracking and baseline comparison; without it, every formatting change re-flags every finding.
- **`originalUriBaseIds`** — relativises file URIs; without it, SARIF embeds machine-local paths.
- **`codeFlows`** — optional but powerful for taint findings; renders as a path in GitHub Code Scanning.

## What SARIF Does Not Specify (That You Still Need)

SARIF is interchange format, not a complete contract. The following must come from `11-`, not the spec:

| Concern | What SARIF says | What you must decide |
|---------|-----------------|------------------------|
| **Severity → CI behaviour** | Defines `level` enum (`none`, `note`, `warning`, `error`) | Which `level` blocks the build; whether `note` and `warning` block on first occurrence |
| **Suppression source of truth** | Allows `suppressions` array on results | Is the source of truth manifest waivers, inline comments, GitHub Code Scanning UI dismissals, or all three? Conflict resolution? |
| **Baseline** | Allows comparison runs but doesn't specify the algorithm | What counts as a "new" finding; how a finding is matched across runs (fingerprint? rule + location? both?) |
| **Multi-run aggregation** | Allows multiple runs in one file | Whether your CI emits one run per analyzer or aggregates |
| **Helper output formats** | Doesn't say not to emit JSON, plain text, GitHub PR annotation API | Ship SARIF + a developer-facing format; the SARIF is for consumers, the developer format is for humans |

`11-` answers each; the answer is the contract.

## Exit-Code Semantics

CI gates on the analyzer's exit code. Three classes the engine must distinguish:

| Exit code | Meaning | CI behaviour |
|-----------|---------|--------------|
| **0** | Analysis ran; no findings at or above the gate threshold | Pass |
| **1** | Analysis ran; findings at or above the gate threshold | Fail (security gate) |
| **2** | Configuration error: bad manifest, schema mismatch, drift | Fail (alert; this is fixable but not by re-running) |
| **3** | Infrastructure error: unreachable file, parser crash, OOM | Fail (alert; this is a CI/analyzer bug) |
| **4** *(optional)* | Resolution rate below threshold | Fail or warn (decide per project) |

Conflating these is the source of the "we re-ran CI four times and the build went green" pattern: a parser crash on one PR, fixed by chance on the next, while the analyzer's verdict on the *findings* was never recomputed.

The **exit-code matrix** lives in `11-`; CI scripts cite it. SARIF is informational; exit code is operational.

## Suppression Round-Tripping

A finding can be suppressed in multiple places:

1. **Inline `# noqa: STA001`** — closest to the code; oldest convention.
2. **Manifest waivers** (`10-manifest-and-coherence.md`) — declarative; auditable.
3. **SARIF `suppressions` array on the result** — set by the engine when it reads (1) or (2).
4. **GitHub Code Scanning UI dismissal** — set by a reviewer after the fact.

These four must round-trip coherently. The discipline:

- The engine reads inline + manifest, emits SARIF with `suppressions` populated, and tracks each suppression's source.
- CI uploads SARIF; GitHub Code Scanning honours the `suppressions` array; UI dismissals are stored in GitHub's database.
- On the next run, the engine re-emits SARIF; UI dismissals already present in GitHub Code Scanning are *not* duplicated. The engine's emission is authoritative for engine-known suppressions; GitHub is authoritative for UI dismissals.

The **conflict policy:**

- **Engine says suppressed, GitHub UI says active** — engine wins; the suppression is in source / manifest and was deliberate. (UI re-activation is a sign that the dismissal was wrong; track it.)
- **Engine says active, GitHub UI says dismissed** — GitHub UI wins for *that finding*, but the dismissal does not propagate back to source / manifest. The finding stays "active per engine" and "dismissed per UI" — surface the disagreement on a dashboard, don't paper over it.
- **Both suppressed via different mechanisms** — fine; record both sources in the SARIF for traceability.

The point is: **suppressions are decisions** (`05-`); they belong in the audit trail; conflicts are signal, not noise.

## Fingerprint Stability

`04-rule-plugin-spec.md` mandates a fingerprint per finding. SARIF stores it as `partialFingerprints`. Stability rules:

- **Stable across cosmetic changes** — formatting, comments, blank lines, line shifts.
- **Stable across rename of unrelated symbols** — moving an unrelated function does not refingerprint findings in another function.
- **Unstable when the defect changes** — the same finding's fingerprint changes when the underlying expression changes substantively.

The recommended composition: `sha256(rule_id || file_relpath || token_byte_range_at_anchor || lattice_value_at_sink_if_present || canonical_AST_neighbourhood_hash)` where the AST neighbourhood is the *N* nearest non-trivial AST nodes. This is stable under formatting; unstable under structural change.

A weaker fingerprint (`rule_id + file + line_number`) re-flags every finding on every shifted line. State the composition explicitly in `11-`.

## Baseline Comparison

GitHub Code Scanning supports **baseline runs** — compare current findings against a previous run on `main`, report only the new ones on the PR.

Two issues:

- **Matching findings** — a finding is "the same" across runs if its fingerprint matches. Reliance on line numbers fails on any whitespace change.
- **Baseline drift** — the baseline run is sometimes weeks old; a finding "fixed" between baseline and now is reported as "fixed" only if the comparison runs match it correctly.

`11-` states:

- The baseline source (typically `main` HEAD or last successful CI run on `main`).
- The matching algorithm (fingerprint-based; algorithm specified).
- The "new finding" definition for PR gating.
- The fallback when fingerprints don't match (re-flag, don't silently drop).

## CI Gate Configuration

A typical gate matrix in `11-`:

```
GATE ON PR:
   level=error                → block merge (cannot be overridden)
   level=warning              → block merge unless dismissed in PR
   level=note                 → annotate, don't block
   resolution_rate < 0.85     → block merge (callgraph too noisy)
   stub_coverage < 0.70       → annotate, don't block (advisory)
   manifest_drift             → block merge (must fix manifest)
   parser_error               → block merge (analyzer bug; alert team)

GATE ON main / release branch:
   level=error                → fail build
   level=warning              → fail build (stricter than PR gate)
   level=note                 → annotate
   suppression_expiry_breach  → fail build (rotted waivers)
   stub_drift                 → fail build (library upgraded; stubs stale)
```

The matrix is the contract with the team. State it; cite it; gate against it.

## Developer Output (Not SARIF)

SARIF is for consumers. Developers want:

- **Console output** — colorised, with file:line:column, rule ID, message, optional code-flow path.
- **Terminal-friendly summary** — `47 findings (3 errors, 12 warnings, 32 notes); see SARIF for details`.
- **IDE format** — most IDEs read SARIF natively; some read LSP diagnostics; the engine should emit at least one of them.
- **Machine-readable JSON for scripts** — same content as SARIF but flatter; for `jq`-style filtering and report generation.

`11-` declares which developer formats are emitted and when (always? on `--format=tty`? on stderr only?). The principle: SARIF is what *machines* see; the developer format is what *humans* see; both ship.

## Engine Self-Describing for Consumers

For SARIF consumers (GitHub, IDE) to render findings well, the engine must publish rule metadata in a form they can index:

- `helpUri` per rule — links to documentation. Without it, "STA001" is opaque.
- `defaultConfiguration.level` per rule — the consumer's default rendering when the result has no explicit level.
- `properties.tags` — used for filtering ("security", "performance", "style").
- `properties.cwe` — array of CWE IDs; GitHub Code Scanning uses these for compliance dashboards.

This metadata comes from `04-rule-plugin-spec.md`'s rule metadata schema. SARIF is its serialisation. Drift between the analyzer's internal metadata and SARIF emission is a bug — gate against it (the CI runs `analyzer rules --format=sarif` and diffs; mismatch = fail).

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Exit code 1 conflated with infra failure | Re-running CI yields green; build verdict unstable | Distinct exit codes per failure class; document the matrix |
| `partialFingerprints` omitted | Every formatting change re-flags every finding | Mandatory fingerprint with stable composition |
| Suppressions only inline, not in SARIF | UI shows finding, source has `# noqa` | Engine reads inline, emits SARIF with `suppressions` array |
| Suppression conflict silently resolved | Engine and UI disagree; nobody notices | Surface conflicts; treat as signal |
| No baseline comparison configured | PRs flag findings already on `main`; team learns to ignore | Configure baseline; align on fingerprint matching |
| `helpUri` absent | Rules opaque to consumers | Mandatory `helpUri` per rule |
| Engine version not in SARIF | Consumers can't tell which analyzer version produced the run; baseline matching breaks | Always include `tool.driver.version` |
| Multi-run aggregation wrong | Findings de-duplicated across analyzers, or duplicated within one | Decide policy: one run per analyzer or aggregated; document |
| SARIF emitted without schema version | Consumers refuse to ingest | `version: "2.1.0"` always |
| Resolution-rate / stub-coverage metrics absent from CI | Internal honesty metrics are invisible | Emit as part of the run's `properties`; gate matrix references them |

## Compatibility With Specific Consumers

A short matrix in `11-`:

| Consumer | Quirks |
|----------|--------|
| **GitHub Code Scanning** | Honours `partialFingerprints`; UI dismissal via `suppressions`; max 25k findings per run; will refuse runs with no `originalUriBaseIds`; max file size 10MB |
| **GitLab Code Quality** | Different schema (Code Quality format); SARIF supported via plug-in; severity mapped differently |
| **Azure DevOps** | SARIF supported; per-pipeline retention; result fingerprints required for incremental display |
| **JetBrains IDE** | Reads SARIF and LSP diagnostics; prefers `helpUri` per rule |
| **VSCode** | LSP-style diagnostics native; SARIF Viewer extension for richer rendering |
| **Sonatype / Snyk / etc.** | Often re-derive their own structure from SARIF; ensure CWE metadata present |

State the consumer set explicitly. "We support SARIF" without naming the consumers is a hope.

## The Decision Output (`11-sarif-and-ci.md`)

A complete `11-` answers:

1. **Output formats** — SARIF (always); developer console; IDE-specific (LSP / native); machine JSON; SARIF schema version.
2. **Exit-code matrix** — per failure class; documented; cited in CI scripts.
3. **Severity mapping** — analyzer's internal severity to SARIF `level`; rationale.
4. **Fingerprint composition** — exact recipe; stability claims; cross-link to `04-`.
5. **Suppression round-trip** — sources of truth, conflict policy, audit trail.
6. **Baseline comparison** — source, matching algorithm, "new finding" definition, fallback policy.
7. **CI gate matrix** — per-trigger thresholds; PR vs main vs release.
8. **Developer output** — formats, when emitted, content vs SARIF.
9. **Consumer compatibility** — explicit list of supported consumers; consumer-specific notes.
10. **Self-description discipline** — engine emits rule metadata to SARIF; CI gates on engine-vs-SARIF drift.
11. **Properties for honesty metrics** — resolution rate, stub coverage, manifest drift, suppression-rate; emitted in SARIF run properties.

## Cross-References

- `plugin-architecture-for-analyzer-rules.md` — rule metadata that becomes SARIF rule descriptors; fingerprint composition
- `false-positive-economics.md` — suppression lifecycle; the `suppressions` array's audit trail
- `manifest-driven-configuration-with-coherence-validation.md` — CI gate configuration; declared in manifest
- `callgraph-construction.md` — resolution-rate metric emitted in SARIF properties
- `cross-module-flow-analysis.md` — stub-coverage metric emitted in SARIF properties
- `scaling-to-large-codebases.md` — incremental analysis flags; partial-run SARIF handling
- Cross-pack: `axiom-audit-pipelines:fingerprint-chains` — the suppression-as-decision trail; SARIF is one serialisation, the audit pipeline is another
- Cross-pack: `axiom-sdlc-engineering:platform-integration` — GitHub-specific Code Scanning idioms and platform constraints
