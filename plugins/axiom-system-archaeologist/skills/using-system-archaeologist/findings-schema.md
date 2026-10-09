# Optional Findings Schema

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Use a structured schema when multiple reviewers or a downstream tool need mechanical merging. A single reviewer may use concise prose with equivalent evidence.

```yaml
schema_version: 1
baseline: <revision and relevant dirty state>
scope: [<paths or subsystem>]
coverage: [{path: <path>, method: <source/tool/runtime>, status: <inspected/partial/unread>}]
findings:
  - id: <stable id>
    location: <path:symbol or line>
    kind: <observation/defect/risk/debt/question>
    claim: <specific statement>
    evidence: [<source or observed result>]
    consequence: <affected behavior/consumer>
    confidence: <high/medium/low plus reason>
    action: <correction or discriminating check>
    status: <open/resolved/accepted/unverified>
unknowns: [<scope and consequence>]
```

Merge by stable identity/location, preserve contradictory evidence and resolve against source. No four partials, fixed focus assignments or mandatory scribe is required. Do not collapse “unread” into “clean”.
