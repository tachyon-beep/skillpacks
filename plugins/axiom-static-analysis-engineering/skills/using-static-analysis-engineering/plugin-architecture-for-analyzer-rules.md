---
name: plugin-architecture-for-analyzer-rules
description: Use when designing the extension surface of an analyzer — how third parties (or your own future team) add rules without forking the engine. Covers rule discovery (entry points, decorator registry, manifest enumeration), the rule lifecycle (load → validate → enable → fire → emit → unload), the metadata schema (id, severity, category, taxonomy alignment with CWE/CWE), conflict resolution when multiple rules fire on the same node, output schema versioning, and the deprecation lifecycle. Produces `04-rule-plugin-spec.md`.
---

# Plugin Architecture for Analyzer Rules

## What "Plugin" Means Here

A "plugin" in an analyzer is the unit of extension: a piece of code that registers a rule, declares what it consumes from the inference output, decides when to fire, and emits findings in the engine's output format. The rule does **not** re-implement traversal, lattice computation, or callgraph propagation — those are the engine's job.

If your "rules" are functions that walk the AST themselves, you do not have a plugin architecture; you have a script. This sheet is about the architecture that makes rules first-class.

## Why a Plugin Architecture Matters

Three problems a real plugin model solves:

- **Composition** — many rules fire on the same nodes. Without a shared traversal, every rule pays for its own walk; with a shared traversal, the engine walks once and fans out.
- **Versioning** — rules deprecate, get renamed, change severity. Without metadata, none of that is tractable; users have a sea of rule IDs and no idea which are still load-bearing.
- **Coverage and conflict** — two rules can both fire on the same finding. Without a conflict policy, downstream tooling sees duplicates or, worse, the second rule silently drops because the first beat it to the line.

`04-rule-plugin-spec.md` exists so the rule contract is in writing before there are dozens of rules to migrate.

## Rule Discovery

How does the engine find rules at startup? Three common patterns, in order of growing rigour:

### Decorator registry (intra-process)

```python
RULES: dict[RuleId, Rule] = {}

def rule(id: str, severity: Severity, category: Category, *, cwe: int | None = None):
    def deco(fn: RuleFn) -> RuleFn:
        RULES[id] = Rule(id=id, fn=fn, severity=severity, category=category, cwe=cwe)
        return fn
    return deco

@rule(id="STA001", severity="error", category="taint", cwe=89)
def sql_injection(ctx: AnalysisCtx, node: ast.Call) -> Iterator[Finding]:
    if ctx.taint(node.args[0]) >= TAINT_UNTRUSTED and is_sql_execute(node):
        yield Finding(node=node, message="Untrusted value reaches SQL execute")
```

**Strengths:** trivial to write, no manifest, reads naturally.
**Weaknesses:** rule discovery requires importing every rule module (no lazy loading); rules from third parties require they ship a Python package; cannot easily disable a rule without touching code.

### Entry-point registry (cross-package)

Use the platform's plugin mechanism: Python entry points (`pyproject.toml` `[project.entry-points."your_analyzer.rules"]`), Node `package.json` `peerDependencies` + a discovery convention, Rust feature-gated `inventory!` macros.

**Strengths:** third parties ship rules in independent packages; engine discovers them via the platform; no import-everything cost.
**Weaknesses:** debugging "why is rule X not loaded" requires understanding the platform's plugin discovery, which is invariably worse-documented than the analyzer.

### Manifest enumeration (declarative)

A manifest file (YAML, TOML, JSON) explicitly lists every rule, its module path, and its enabled state. The engine reads the manifest at startup; nothing is implicit.

**Strengths:** explicit; auditable; supports per-project enables/disables; allows dry-run ("which rules would fire if I enabled this set?"). `manifest-driven-configuration-with-coherence-validation.md` is dedicated to this.
**Weaknesses:** more ceremony per rule; out-of-tree rules need a manifest entry plus the package install.

**Recommendation:** decorator registry inside the analyzer's own ruleset; entry points or manifest for third-party rules. Decide in `04-`; do not mix unintentionally.

## Rule Lifecycle

Every rule passes through six states. The engine guarantees the order; the rule guarantees its behaviour at each step.

```
load → validate → enable → fire → emit → unload
```

| Stage | Engine responsibility | Rule responsibility |
|-------|------------------------|----------------------|
| **load** | Import the rule module, register the rule object | Side-effect-free import |
| **validate** | Check metadata (id uniqueness, severity in enum, category known, CWE valid) | Provide complete metadata |
| **enable** | Decide whether the rule runs this invocation (config-driven) | Tolerate being disabled |
| **fire** | Call the rule per matched node with the analysis context | Inspect context, decide if a finding applies |
| **emit** | Collect findings, deduplicate by `(rule_id, location, fingerprint)`, emit in output format | Produce findings with location, message, severity |
| **unload** | Release any rule-held caches; clean up | Have nothing to clean up (rules should be stateless) |

**The rule should be stateless across invocations.** State that persists across `fire` calls is the standard source of "the rule fires inconsistently when files are analysed in a different order" bugs. Cache only via the engine's facilities (which are aware of incremental invalidation; see `three-phase-inference.md`'s incremental section).

## Rule Metadata Schema

`04-` must define a metadata schema. A minimum-viable one:

```python
@dataclass(frozen=True)
class RuleMetadata:
    id: str                       # stable, machine-readable: "STA001", "TAINT-SQL-INJECT"
    name: str                     # human-readable: "SQL Injection via Untrusted Input"
    severity: Literal["error", "warning", "info", "hint"]
    category: str                 # "taint", "style", "type", "security"
    description: str              # one paragraph
    rationale: str                # why this rule exists
    examples_violation: list[str] # source snippets that should trigger
    examples_clean: list[str]     # source snippets that should NOT trigger
    cwe: int | None = None        # CWE-89 etc., for security rules
    owasp: str | None = None      # "A03:2021", for OWASP-relevant rules
    introduced_version: str       # analyzer version this rule shipped in
    deprecated_in: str | None = None
    replaced_by: RuleId | None = None
    tags: frozenset[str] = frozenset()
```

**The fields exist for a reason:**

- `id` — used in suppressions (`# noqa: STA001`), in reports, in cross-references. It must never change after first ship; rename via deprecation, not edit.
- `severity` — drives CI behaviour (`error` blocks merge); user-overridable per project but rule defines a default.
- `category` — used by `/system-archaeologist` and downstream tooling to group findings; must be drawn from a stable taxonomy.
- `cwe` / `owasp` — taxonomy alignment for security rules; lets `/security-architect` cross-reference findings to threat models.
- `examples_*` — drive the test corpus (see consistency gate check 11). A rule without examples is unfalsifiable.
- `introduced_version`, `deprecated_in`, `replaced_by` — lifecycle metadata; consumed by the deprecation flow.

## Conflict Resolution

What happens when two rules fire on the same AST node?

Three policies in increasing rigour:

- **Both emit, downstream deduplicates** — engine emits both findings; downstream (CI, report) decides. Simplest; pollutes reports with duplicates.
- **Highest severity wins, others suppressed** — engine emits only the highest-severity finding; others suppressed with a "subsumed by RULE-ID" annotation. Cleaner reports; loses information.
- **Explicit subsumption rules** — rules declare which other rules they subsume. `STA042 (subsumes: STA001, STA017)` means when 042 fires on a node where 001 and 017 also fired, 042 wins. Most rigorous; requires authoring discipline.

**Recommendation:** "highest severity wins" with subsumption as an opt-in refinement. Document the policy in `04-`. The "both emit" policy should only be used if the downstream actively wants duplicates (rare).

## Output Schema Versioning

The output of the analyzer (the finding format) is a contract with downstream consumers (`/system-archaeologist`, IDE plug-ins, CI gates, dashboards). It must be versioned.

Recommended fields per finding:

```json
{
  "schema_version": "1.0.0",
  "analyzer": {"name": "wardline", "version": "2.3.1"},
  "rule": {"id": "STA001", "name": "SQL Injection", "severity": "error", "category": "taint", "cwe": 89},
  "location": {"file": "app/db.py", "start_line": 14, "start_col": 8, "end_line": 14, "end_col": 27},
  "message": "Untrusted value flows to SQL execute at line 14",
  "fingerprint": "sha256:...",        // stable across runs; key for dedup and waiver matching
  "context": {                          // optional: rule-specific structured detail
    "source_var": "request.GET['q']",
    "sink_call": "cursor.execute(...)",
    "sanitisers_in_path": []
  },
  "suggested_fix": null                 // optional: structured replacement
}
```

**The fingerprint is load-bearing.** Without a stable fingerprint, suppressions cannot match findings across runs (a comment shift renumbers lines and old `# noqa`s become stale). Fingerprint should compose: `(rule_id, file_relpath, byte_range_of_meaningful_token, lattice_value_at_sink_if_present)` or similar — stable under formatting changes, unstable when the underlying defect changes.

**SARIF emission** (`sarif-emission-and-ci-integration.md`) is the standardised export format for GitHub Code Scanning and similar; until then, ship a stable native format and document it.

## Deprecation Flow

Rules deprecate. The flow:

1. **Mark deprecated** — set `deprecated_in: "2.4.0"` and (usually) `replaced_by: "STA042"`. The rule continues to fire; output now includes a `deprecation` field.
2. **Grace period** — at least one minor version. Downstream consumers see deprecation warnings; suppressions on the deprecated rule continue to suppress.
3. **Remove** — at the next major version, delete the rule. Suppressions targeting the removed rule become orphans; emit them in the engine's diagnostic output ("STA001 is no longer a rule; this suppression has no effect").
4. **Renaming** is not a thing. A "renamed rule" is a deprecate-old + add-new. Renames break every suppression that referenced the old ID.

This flow is the rule-level analogue of API deprecation. It is non-optional once the analyzer has any third-party consumer.

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Rules walk the AST themselves | No composition; every rule pays for traversal | Rules consume the engine's analysis context; engine walks once |
| Rule IDs change | Every suppression and every dashboard breaks | Treat rule IDs as immutable; deprecate-and-replace, never rename |
| Rules hold state across fires | Order-dependent findings | Stateless rules; use engine cache for legitimate caching |
| No conflict policy | Duplicate findings or silent suppression | Pick "highest severity wins" + optional subsumption; document |
| Output schema unversioned | Downstream consumers break on every analyzer release | Version the schema; semver it; document the contract |
| No fingerprint | Suppressions go stale on every formatting change | Fingerprint is `(rule_id, file, byte-range-of-meaningful-token, ...)`; stable under cosmetic edits |
| Rules without `examples_violation` and `examples_clean` | Unfalsifiable; no test corpus | Mandatory in metadata; consistency gate check 11 enforces |
| No deprecation lifecycle | Rules accrete; users can't tell which are still load-bearing | Mark, grace, remove; orphan-suppression diagnostic |
| Decorator registry mixed with entry-point discovery without rationale | Mysterious "rule X is not loading" debugging sessions | Pick one mechanism per rule source (own-tree vs third-party) and document |

## The Decision Output (`04-rule-plugin-spec.md`)

A complete `04-` answers:

1. **Discovery mechanism** — decorator registry, entry points, manifest, or hybrid.
2. **Rule lifecycle** — the six states above; what the engine guarantees, what the rule guarantees at each.
3. **Metadata schema** — exact fields, types, validation rules.
4. **Taxonomy** — the category enum; CWE alignment; OWASP alignment for security rules.
5. **Conflict policy** — highest-severity-wins, subsumption rules, or both.
6. **Output schema** — full JSON shape; SARIF or native; semver of the schema.
7. **Fingerprint composition** — exact recipe; stability properties claimed.
8. **Deprecation flow** — mark → grace → remove; orphan-suppression behaviour.
9. **Caching contract** — engine-provided caches available to rules; invalidation triggers.
10. **Testing contract** — `examples_*` drive a per-rule test corpus; CI must run it.
11. **Versioning** — analyzer semver, rule-ID-set semver, output-schema semver. Three independent versions.

## Cross-References

- `ast-visitation-patterns.md` — what the engine walks; rules consume the visitor's output
- `taint-lattice-design.md` — what rules read from the analysis context (lattice values at nodes)
- `three-phase-inference.md` — what summaries / callsite envs rules consume
- `false-positive-economics.md` — suppression lifecycle is the operational counterpart to this sheet's deprecation lifecycle
- `static-vs-runtime-tradeoffs.md` — when the right answer is "no static rule for this; runtime check"
- `manifest-driven-configuration-with-coherence-validation.md` — the manifest variant in depth
- `sarif-emission-and-ci-integration.md` — SARIF output schema and CI exit-code semantics
- `decorator-as-assertion.md` — runtime + static dual rules; a single source of truth driving both
- Cross-pack: `axiom-system-archaeologist:analyze-codebase` — consumes this sheet's output schema; needs a stable contract
- Cross-pack: `ordis-security-architect:design-controls` — the control taxonomy that aligns with the rule category enum
