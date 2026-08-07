---
description: Mechanical sweep of a codebase for contract-drift signals — schema defaults on measurement fields, lenient .get() readers, dual-key fallbacks, fail-open version comparisons, defaulted version fields, deny-list projections and asdict-based redaction, decision-shaping constants outside policy records, dual schema definitions, dangling schema-doc references, and constructor-round-trip contract tests. Grep-driven, cheap, CI-friendly; produces structured JSON findings with severity and the sheet that closes each gap. Optionally feeds the contract-reviewer agent for narrative synthesis.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task"]
argument-hint: "[project_path]"
---

# Audit Contract Drift Command

You are sweeping a project for mechanical signatures of contract drift. This is the cheap, grep-driven complement to `/review-contracts`: every hit below is a *lead* that you confirm by reading context before reporting, not a finding by pattern-match alone. No code edits, no redesign.

Target: the project path argument (default current directory). First locate the contract surface (`contracts/`, `schemas/`, IDL files, cross-boundary dataclass modules, their parsers and tests); if none exists, say so and stop.

## Sweep dimensions

For each: pattern, heuristic, severity, closing sheet.

1. **Defaults on measurement-shaped fields**

   ```bash
   grep -rn -E "(util|rate|latency|depth|count|score|temp|pressure|_ms|_s\b)[a-z_]*\s*:\s*(float|int)\s*=\s*[0-9-]" --include="*.py" "${P}"
   grep -rn -E "\.get\(\s*[\"'][a-z_]+[\"']\s*,\s*(0(\.0)?|-1|\"\"|None)\s*\)" --include="*.py" "${P}"
   ```
   Heuristic: a numeric default on a field describing a measurement, or a defaulted `.get()` on a wire payload. Confirm the field crosses a boundary and trace one consumer before flagging. Severity: **high** (consumer actuates on it), **med** (stored but unconsumed). Sheet: `silent-default-elimination.md`.

2. **Bare `.get()` absence acceptance with no version gate**

   ```bash
   grep -rn -B5 "\.get(" --include="*.py" "${P}" | grep -v "version"
   ```
   Heuristic: parser functions using bare `.get(key)` where no `schema_version`/`wire_version` check appears in the enclosing function. Severity: **med**. Sheet: `silent-default-elimination.md` §4.

3. **Dual-key fallbacks (compat shims)**

   ```bash
   grep -rn -E "\.get\([^)]+,\s*[a-z_]*\.get\(" --include="*.py" "${P}"
   ```
   Heuristic: nested `.get(new, .get(old))` — two names for one meaning reconciled inline. Severity: **high**. Sheet: `schema-versioning-and-evolution.md`.

4. **Fail-open version gates**

   ```bash
   grep -rn -E "version\s*(>=|<=|>|<)\s*[0-9]" --include="*.py" "${P}"
   grep -rn -E "(schema_version|wire_version)[\"']?\s*,\s*[\"']?[0-9]" --include="*.py" "${P}"
   ```
   Heuristic: range comparison in version dispatch (first pattern) or a defaulted version field (second — `.get("schema_version", <literal>)`). Severity: **high** (range dispatch), **high** (defaulted version). Sheet: `schema-versioning-and-evolution.md`.

5. **Version-in-name-only candidates**

   ```bash
   git -C "${P}" log --oneline -20 -- '*contract*' '*schema*' 2>/dev/null
   grep -rn -E "(SCHEMA_VERSION|WIRE_VERSION|_VERSION)\s*=" --include="*.py" "${P}"
   ```
   Heuristic: recent commits touching contract/schema files without touching a `*_VERSION` constant; comments like "as of", "now in", "changed to" near fields. This dimension needs human/reviewer confirmation — report as **med** leads. Sheet: `schema-versioning-and-evolution.md`.

6. **Deny-list projections / redaction at serialization**

   ```bash
   grep -rn -E "(__dict__|asdict)\s*(\(|\.copy)" --include="*.py" "${P}"
   grep -rn -E "(redact|pop|del )[^\n]*(provider|source|vendor|pii|secret)" --include="*.py" "${P}"
   ```
   Heuristic: a projection that copies everything then removes/overwrites sensitive keys. Severity: **critical** if the output feeds an external/blinded consumer, else **high**. Sheet: `blinding-by-construction.md`.

7. **Free-text fields crossing boundaries**

   ```bash
   grep -rn -E "(notes|comment|detail|context|rationale|message)\s*:\s*str" --include="*.py" "${P}"
   ```
   Heuristic: free-text fields on cross-boundary record classes; confirm the record crosses an authority or blinded boundary. Severity: **med** (unconsumed), **high** (feeds a resolver or blinded view). Sheet: `blinding-by-construction.md`, `deterministic-resolution.md`.

8. **Decision-shaping constants outside policy records**

   ```bash
   grep -rn -E "^[A-Z_]+(THRESHOLD|_HIGH|_LOW|_LIMIT|_BUDGET|WEIGHT|_BAND)[A-Z_]*\s*=\s*[0-9]" --include="*.py" "${P}"
   ```
   Heuristic: module-level threshold/weight constants read by decision code; confirm decision records don't bind a policy version. Severity: **high**. Sheet: `versioned-policy-parameters.md`.

9. **Hidden resolver state**

   ```bash
   grep -rn -E "(self\._[a-z_]*(cache|last|seen|prev|history))" --include="*.py" "${P}"
   grep -rn -E "(datetime\.now|time\.time|random\.)" --include="*.py" "${P}"
   ```
   Heuristic: mutable memory or clock/RNG reads inside resolver/decider classes. Confirm the value influences output and is not recorded as an explicit input. Severity: **high**. Sheet: `deterministic-resolution.md`.

10. **Dual schema definitions / dangling schema docs**

    ```bash
    grep -rn -E "(schema.*\.json|\.schema\.|documented in|see docs/)" --include="*.py" "${P}"
    ```
    Heuristic: comments citing an external schema doc — verify the file exists and (spot-check) agrees with the code. Severity: **med** (exists, drifted), **low** (dangling reference). Sheet: `contract-first-boundaries.md`.

11. **Vacuous contract tests**

    ```bash
    grep -rn -E "(__dict__|asdict)\s*(\(\)|\.copy\(\))" --include="test_*.py" --include="*_test.py" "${P}"
    grep -rn -E "def test_.*(future|unknown).*(version|schema)" --include="test_*.py" "${P}"
    ```
    Heuristic: round-trip tests built from the record's own dict (first), or tests asserting unknown/future versions parse (second — these pin fail-open behavior). Severity: **high** (future-version-parses test), **med** (constructor round-trip as the only coverage). Sheet: `contract-testing.md`.

## Output format

```json
{
  "summary": {"critical": 0, "high": 0, "med": 0, "low": 0},
  "findings": [
    {
      "severity": "high",
      "sheet": "silent-default-elimination.md",
      "signal": "defaulted .get() on wire payload",
      "location": "contracts/metrics.py:31",
      "evidence": "cpu_util=payload.get(\"cpu_util\", 0.0)",
      "consumption": "decider.py:12 reads 0.0 as idle -> scale_down",
      "remediation": "Encode absence explicitly (Measured|Absent); version-gate the absence semantics."
    }
  ]
}
```

Order critical → high → med → low; within a band, by path then line. Every finding carries `evidence` (the confirmed line, not just the grep hit) and, for severities above med, a `consumption` trace. After the JSON, a plain-language triage of 3–5 sentences.

## Optional: dispatch reviewer agent

With `--review` (or on request), pass the findings JSON to `contract-reviewer` for narrative synthesis and coverage of the non-mechanical catalogue entries (resolver preferences, silent definition edits, blinding-policy completeness):

```
Task(subagent_type="contract-reviewer",
     description="Synthesize contract-drift sweep findings",
     prompt="The following JSON is an /audit-contract-drift sweep of ${P}.
     Group the findings, add the non-mechanical catalogue entries the sweep
     cannot check, produce a prioritized remediation sequence citing sheets,
     and your SME protocol sections.\n\n${FINDINGS_JSON}")
```

Present the JSON first, the narrative second.

## Verification

Critical/high findings block merge of the surface they sit on; med findings are scheduled to the next touch; low findings are acknowledged or fixed opportunistically. Re-run after each band — the sweep is cheap. For depth beyond mechanical signatures, run `/review-contracts`.

## Cross-references

- `using-contract-engineering` — router; failure-mode catalogue
- `/review-contracts` — full catalogue audit via `contract-reviewer`
- `/design-contract-suite` — when drift signals mean the suite needs redesign
- Sheets cited per dimension above
