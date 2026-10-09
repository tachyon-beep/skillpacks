# Dependency Scan Evidence

`ordis-security-architect` owns supply-chain threat/control design;
`axiom-devops-engineering` owns delivery integration. Use this reference to verify
that dependency checks support an actual release decision.

Inspect manifests, resolved lockfiles, build/runtime/transitive scope and existing
scanner commands. Record scanner/database version or update time, scanned artifact,
findings and applicability. Check reachability/exposure and remediation options;
a version match alone does not establish exploitability, and no known advisory
does not establish safety.

Verify signatures/provenance against expected policy where required. Track fixes,
exceptions and accountable owners. Confirm dependency changes still build and
satisfy affected integration/contract checks. Report unsupported ecosystems,
private components, unavailable advisory sources and tool failures as coverage
limits rather than a clean result.
