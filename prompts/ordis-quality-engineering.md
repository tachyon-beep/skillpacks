# Refresh: ordis-quality-engineering

**Verdict:** MEDIUM / S effort. Mostly mechanical — deprecated CI versions + supply-chain gap.

## Context

- Pack path: `/home/john/skillpacks/plugins/ordis-quality-engineering/`
- Full review: `/tmp/skillpack-refresh-review/ordis-quality-engineering.md`
- Purpose: testing, coverage, flakiness, CI/CD test pipelines.

## Why refresh

- **Deprecated GitHub Actions versions:**
  - `actions/checkout@v3` → `@v4`+
  - `actions/setup-python@v4` → `@v5`
  - `actions/upload-artifact@v3` → `@v4`
- **Supply-chain gap** in dependency-scanning skill: missing Trivy, OSV-Scanner, Sigstore (cosign), SLSA, SBOM generation (Syft).

## Scope — DO

1. **Bump all GH Actions** to current major versions across every example.
2. **Dependency-scanning sheet refresh.** Add Trivy (filesystem + container), OSV-Scanner, Syft (SBOM), cosign (signing), SLSA provenance.
3. **Cross-pack pointer.** Add cross-ref to `ordis-security-architect` supply-chain skill (after that pack is refreshed).

## Scope — DO NOT

- Do not change the test-pyramid / coverage-gap analysis content — it's solid.
- Do not duplicate `ordis-security-architect` content.

## Acceptance criteria

1. Zero `@v3` GH Actions references that should be `@v4`+.
2. Trivy + OSV-Scanner + cosign + SBOM (Syft or CycloneDX) all covered.
3. SLSA framework referenced at least once.
4. `plugin.json` version bumped (minor).

## Process

1. Read `/tmp/skillpack-refresh-review/ordis-quality-engineering.md`.
2. `grep -rn "@v3\|@v4 # setup-python" plugins/ordis-quality-engineering/` — find every action version.
3. Edit. Verify each action version is current.
4. Coordinate cross-ref with `ordis-security-architect`.
5. Bump version.

## Constraints

- Every action version must be the current major.
- Every supply-chain tool named must exist and be actively maintained.
- No fabrication of CLI flags.
