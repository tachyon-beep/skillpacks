# Refresh: ordis-security-architect

**Verdict:** HIGH / M effort. Solid core, conspicuous AI/supply-chain gap.

## Context

- Pack path: `/home/john/skillpacks/plugins/ordis-security-architect/`
- Full review: `/tmp/skillpack-refresh-review/ordis-security-architect.md`
- Purpose: threat modeling, defense-in-depth, security controls, compliance.

## Why refresh

- **Zero LLM/AI security coverage** — no prompt injection, no agentic exfil, no model-supply-chain (poisoned weights), no RAG poisoning, no jailbreak taxonomy.
- **No SLSA / SBOM / supply-chain skill** — large gap given current threat landscape.
- **Unversioned standards refs** — OWASP, ISO 27001, NIST CSF, PCI-DSS without version tags. NIST CSF 2.0 (2024) is materially different from 1.1.
- **No CWE / CVSS / ATT&CK** mappings — table-stakes for modern threat modeling.
- **Hard-coded 2025 dates** make the pack look dated.
- **Phase-1 stub artefacts** still visible in the pack.

## Scope — DO

1. **New skill: LLM/AI threat modeling.** Cover OWASP LLM Top 10 (latest), prompt injection (direct + indirect), training-data poisoning, model-supply-chain (HuggingFace risks, pickle files, weight tampering), agentic risks (tool exfil, lateral movement via tool-use), RAG poisoning, jailbreak taxonomy.
2. **New skill: Supply-chain security.** SLSA levels, SBOM (SPDX, CycloneDX), Sigstore / cosign / fulcio, in-toto attestations, dependency confusion, typosquatting, package-namespace defense.
3. **Standards refresh.** Tag every standards reference with version: NIST CSF 2.0, ISO 27001:2022, PCI-DSS 4.0.1, OWASP Top 10:2021 (and watch for 2025).
4. **Threat modeling sheet.** Add CWE / CVSS / ATT&CK mappings as standard fields in the threat-model output.
5. **Strip Phase-1 stub artefacts.**
6. **Unhardcode dates.** Use "as of {{review-date}}" or remove.

## Scope — DO NOT

- Do not duplicate `yzmir-llm-specialist` content — security focuses on threats and controls, not LLM development.
- Do not replace the STRIDE / defense-in-depth core — it's solid.

## Acceptance criteria

1. Dedicated LLM/AI threat-modeling skill exists.
2. Dedicated supply-chain skill exists with SLSA, SBOM, Sigstore.
3. Every standard reference has a version tag.
4. CWE / CVSS / ATT&CK referenced in at least the threat-model and controls skills.
5. No Phase-1 stub language remaining.
6. No hard-coded year dates outside changelog.
7. `plugin.json` version bumped (minor or major).

## Process

1. Read `/tmp/skillpack-refresh-review/ordis-security-architect.md`.
2. Read every SKILL.md.
3. Coordinate with `yzmir-llm-specialist` (cross-ref but don't duplicate).
4. Coordinate with `axiom-devops-engineering` and `ordis-quality-engineering` for SLSA/SBOM convergence.
5. Draft new skills, verify references against current OWASP/NIST/MITRE docs.
6. Bump version.

## Constraints

- Every CWE/CVE/ATT&CK ID cited must be a real ID (verify).
- Every standard version tag must be current as of refresh date.
- No fabrication of attack patterns — every named attack must have a public reference.
