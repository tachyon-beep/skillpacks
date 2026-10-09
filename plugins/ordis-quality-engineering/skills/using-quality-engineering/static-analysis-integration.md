# Integrate Static Analysis into Quality Evidence

Language/framework engineering and `axiom-static-analysis-engineering` own analyzer
implementation and configuration. Use native lint/type/security tools and the
repository's existing scripts before introducing another scanning platform.

For a quality gate, record tool/version/configuration, scanned scope, findings,
suppression policy and required disposition. Verify the check actually runs on the
changed paths and returns a failing status for a known material violation.
Distinguish new findings from baselines without silently accepting new defects.

Investigate false positives with source evidence; narrow suppressions need an
explicit reason and review trigger. A clean scan establishes only the configured
rules on scanned inputs, not runtime correctness or complete security.
Report execution failures separately from zero findings. Check unscanned generated
code, dependencies or languages when those are material to the release claim.
