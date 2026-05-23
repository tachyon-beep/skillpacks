# Review: ordis-security-architect

**Version:** 1.2.0  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

Scope: Stages 1-4 of `meta-skillpack-maintenance:using-skillpack-maintenance` applied to
`/home/john/skillpacks/plugins/ordis-security-architect/`. Report-only; no edits.
Stage 5 (`implementing-fixes.md`) intentionally skipped per task instructions.

---

## 1. Inventory

### Plugin metadata
- `plugins/ordis-security-architect/.claude-plugin/plugin.json` (lines 1-37)
  - `name`: `ordis-security-architect`
  - `version`: `1.2.0`
  - `description`: "Threat modeling, security controls, compliance, ATO, LLM/AI security, supply-chain (SLSA/SBOM/Sigstore) - 11 skills, 3 commands, 2 agents"
  - Author/license/keywords present and well-formed.

### Marketplace registration
- `.claude-plugin/marketplace.json` registers the plugin (entry lines reachable
  via `grep -B1 -A10 "ordis-security-architect"`). Source path
  `./plugins/ordis-security-architect`. Description is richer than plugin.json
  (cites OWASP LLM Top 10:2025, MITRE ATLAS, NIST CSF 2.0, etc.).

### Skills (1 router + 10 reference sheets)

| File | Lines | Description (frontmatter or H1) |
|------|-------|---------------------------------|
| `skills/using-security-architect/SKILL.md` | 425 | Router. `description: Routes to security architecture skills - threat modeling, controls, compliance, authorization` |
| `skills/using-security-architect/threat-modeling.md` | 740 | STRIDE, attack trees, risk scoring |
| `skills/using-security-architect/secure-by-design-patterns.md` | 497 | Zero-trust, least privilege, fail-secure |
| `skills/using-security-architect/security-controls-design.md` | 537 | Trust-boundary layered controls |
| `skills/using-security-architect/security-architecture-review.md` | 433 | Design-review checklist |
| `skills/using-security-architect/documenting-threats-and-controls.md` | 624 | Threat-model docs, security ADRs |
| `skills/using-security-architect/llm-and-ai-security.md` | 578 | OWASP LLM Top 10:2025, MITRE ATLAS |
| `skills/using-security-architect/supply-chain-security.md` | 515 | SLSA, SBOM, Sigstore, in-toto |
| `skills/using-security-architect/classified-systems-security.md` | 632 | MLS, Bell-LaPadula, classification |
| `skills/using-security-architect/compliance-awareness-and-mapping.md` | 698 | NIST CSF 2.0, ISO 27001, PCI-DSS, GDPR, NIS2, EU AI Act, ISM |
| `skills/using-security-architect/security-authorization-and-accreditation.md` | 761 | RMF, ATO, SSP/SAR/POA&M, FedRAMP |

Total reference-sheet content: ~6,000 lines. Substantial.

### Commands (3)

| File | description | argument-hint |
|------|-------------|---------------|
| `commands/threat-model.md` | Systematic threat modeling using STRIDE methodology, attack trees, and risk scoring | `[system_or_component_to_analyze]` |
| `commands/design-controls.md` | Design layered security controls at trust boundaries with defense-in-depth | `[boundary_or_component_to_secure]` |
| `commands/security-review.md` | Review architecture for security gaps, missing controls, and defense-in-depth | `[architecture_or_design_to_review]` |

All three declare `allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]` (correct quoted-string-array convention, includes `Write` for artifact emission, no `Skill` because they are direct workflow commands rather than routers).

### Agents (2)

| File | Model | SME Protocol? | Description ends with SME marker? |
|------|-------|---------------|-----------------------------------|
| `agents/threat-analyst.md` | opus | Yes, body cites `meta-sme-protocol:sme-agent-protocol` on line 10 and requires Confidence/Risk/Information Gaps/Caveats sections | Yes (line 2: "...Follows SME Agent Protocol with confidence/risk assessment.") |
| `agents/controls-designer.md` | opus | Yes, body cites protocol on line 10, requires four sections | Yes (line 2: same phrasing) |

Both agents declare only `description` and `model` (the dominant repo convention — no `tools:` restriction, inherits parent context). Both have positive-AND-negative activation examples and explicit "I do NOT" scope boundaries. Model selection (`opus`) is defensible for synthesis-heavy STRIDE / defense-in-depth reasoning.

### Hooks
None. No `hooks/hooks.json` present. Acceptable for a domain-knowledge pack — security architecture is human-driven analysis, not automated tool-event reaction. Hooks would be inappropriate here.

### Slash-command wrapper
- `.claude/commands/security-architect.md` exists (312 lines) — paired wrapper present, so the router is user-invocable as `/security-architect`. **Pass** the existence test; **see Major findings** for content drift.

### Frontmatter convention check
- Router SKILL.md: `name` + `description` only. Conforms (omits `allowed-tools` per marketplace convention).
- Reference sheets: no frontmatter; content-only files referenced by the router via Markdown links. Conforms.
- Commands: `description` + `allowed-tools` (quoted JSON-style array) + `argument-hint`. All three conform.
- Agents: `description` + `model` only; no `tools:` key (inherit parent context). Conforms (~60/65 marketplace agents follow this).
- Wrapper: no frontmatter. **Does not** conform to peer wrappers that declare a `description:` line — see Major #2.

---

## 2. Domain & Coverage

### Declared scope
Security architecture — threat modeling, controls, defense-in-depth, compliance,
ATO/RMF, classified-systems MLS, LLM/AI security, supply-chain (SLSA/SBOM/Sigstore).
Aimed at practitioners and experts ("architecture-level treatment").

### Coverage map vs inventory

**Foundational**
- Threat modeling (STRIDE) — `threat-modeling.md` (Pass, comprehensive)
- Secure-by-design (zero-trust, least privilege, fail-secure) — `secure-by-design-patterns.md` (Pass)
- Defense-in-depth controls at trust boundaries — `security-controls-design.md` (Pass)
- Security architecture review — `security-architecture-review.md` (Pass)
- Documenting security decisions / ADRs — `documenting-threats-and-controls.md` (Pass)

**Core (specialised, regulated)**
- Compliance mapping (NIST CSF 2.0, ISO 27001:2022, PCI-DSS v4.0.1, GDPR, NIS2,
  EU AI Act, ISM) — `compliance-awareness-and-mapping.md` (Pass; multi-jurisdictional
  including AU/UK/US/EU)
- Government authorization (RMF, ATO, SSP/SAR/POA&M, FedRAMP) —
  `security-authorization-and-accreditation.md` (Pass; framework currency cited)
- Classified-systems MLS (Bell-LaPadula, classification hierarchy) —
  `classified-systems-security.md` (Pass)

**Advanced / current**
- LLM & AI security (OWASP LLM Top 10:2025, MITRE ATLAS, NIST AI RMF, EU AI Act
  2024/1689, ISO/IEC 42001) — `llm-and-ai-security.md` (Pass; very current)
- Supply-chain security (SLSA v1.1, SPDX 3.0, CycloneDX 1.6, in-toto, Sigstore,
  EO 14028, EU CRA 2024/2847) — `supply-chain-security.md` (Pass; very current)

**Cross-cutting expectations**
- Cross-pack handoff with `axiom-audit-pipelines` for audit-log threat model
  (referenced in `threat-modeling.md` line 28 and router SKILL.md lines 238-241) — Pass.
- Cross-faction handoff with `muna-technical-writer` for SSP/SAR/ADR writing — referenced
  multiple places — Pass.
- Cross-faction handoff with `yzmir-llm-specialist` for prompt-engineering / RAG correctness
  (referenced in `llm-and-ai-security.md` lines 9, 38-40) — Pass.

### Gaps / weak spots

- **Cloud-native posture** (CSPM, IAM-graph attacks, K8s-specific threat model,
  service-mesh trust): present implicitly inside the trust-boundary discussion
  but no dedicated reference sheet. Lower-priority for an architecture pack —
  cloud-specific drilldown could be deferred to a future extension.
- **Cryptographic agility / PQC migration**: not explicitly addressed. The pack
  references TLS, signing, and key management but does not call out
  hybrid-PQC / crypto-agility as an architectural concern. This is becoming
  current (NIST IR 8547, NSA CNSA 2.0). **Minor gap**.
- **Privacy engineering as architecture** (data minimisation, purpose limitation,
  pseudonymisation patterns): touched in compliance sheet but no architecture-level
  pattern catalogue. Reasonable to keep in compliance sheet for now.

### Research currency
Security is an evolving domain. The pack already cites very recent versions:
OWASP LLM Top 10:2025, NIST CSF 2.0 (Feb 2024), PCI-DSS v4.0.1, ISO 27001:2022,
EU AI Act 2024/1689, NIS2 (Dir. 2022/2555), CRA Reg. 2024/2847, SLSA v1.1 (Aug 2024),
SPDX 3.0 (Apr 2024), CycloneDX 1.6 (Apr 2024). Currency is strong; minor caveat that
documents in this domain age fast and should be re-checked annually.

### Coverage matrix
| Topic | Sheet | Status | Notes |
|-------|-------|--------|-------|
| STRIDE enumeration | threat-modeling | Pass | 740 lines; configuration-override pattern (VULN-004) cited |
| Attack-tree construction | threat-modeling | Pass | Scoring formula L × I; risk-matrix template |
| Defense-in-depth | security-controls-design + secure-by-design-patterns | Pass | Trust-boundary-first methodology |
| Fail-secure / fail-closed | secure-by-design + controls-designer agent | Pass | Code examples included |
| Zero-trust, least privilege | secure-by-design-patterns | Pass | Three pillars cited |
| MLS / Bell-LaPadula | classified-systems-security | Pass | AU classification hierarchy + transitivity |
| ATO / RMF process | security-authorization-and-accreditation | Pass | All 7 RMF steps; SSP/SAR/POA&M templates |
| Compliance discovery | compliance-awareness-and-mapping | Pass | Multi-jurisdictional (AU/UK/US/EU) |
| OWASP LLM Top 10:2025 | llm-and-ai-security | Pass | All 10 items + CWE + ATLAS IDs |
| MITRE ATLAS | llm-and-ai-security | Pass | Per-item AML.T0### codes |
| SLSA / SBOM / Sigstore | supply-chain-security | Pass | SLSA v1.1, SPDX 3.0, CycloneDX 1.6 |
| Supply-chain threat model | supply-chain-security | Pass | SC-01..SC-10 catalogue with CWE/ATT&CK |
| Security ADRs / threat docs | documenting-threats-and-controls | Pass | Threat template + control template |
| PQC / crypto-agility | (none) | Gap (Minor) | Architecture-level migration patterns not addressed |
| Cloud-native CSPM / K8s | (implicit) | Partial | Trust-boundary frame covers conceptually; no dedicated sheet |
| Privacy engineering | compliance-awareness | Partial | Discussed under GDPR; no standalone pattern sheet |

---

## 3. Fitness Scorecard

| Dimension | Rating | Notes |
|-----------|--------|-------|
| **Scope clarity** | Pass | Router clearly delineates Core vs Extension; cross-pack boundaries (audit pipelines, technical writer, LLM specialist) explicitly stated. |
| **Coverage** | Pass | All foundational + specialised + AI/supply-chain domains covered. PQC is a Minor gap. |
| **Component typing** | Pass | Skills = guidance; commands = explicit user-triggered workflows; agents = SME specialists; nothing miscast. |
| **Router accuracy** | **Major issue** | SKILL.md and the slash-command wrapper both reference a stale skill name `architecture-security-review` (file is actually `security-architecture-review.md`). 5 occurrences in SKILL.md, ~10 in the wrapper. Will mislead any user/agent following the routing tables. `security-controls-design.md:27` references a non-existent `secure-code-patterns`. |
| **Wrapper sync** | **Major issue** | `.claude/commands/security-architect.md` is materially out-of-date vs `SKILL.md`: it omits the `llm-and-ai-security` and `supply-chain-security` extension skills entirely (router added them; wrapper never caught up). It uses dotted-path skill names (`ordis/security-architect/threat-modeling`) rather than the marketplace plugin format and lacks frontmatter — peer wrappers like `audit-pipelines.md`, `creative-writing.md`, `determinism-and-replay.md` have a `description:` block. Inconsistency reduces discoverability. |
| **Agent quality** | Pass | Both agents follow SME protocol (cite `meta-sme-protocol`, require four output sections, end description with the load-bearing SME phrase), have positive AND negative activation examples, explicit scope-out lists, and frontmatter that conforms to the marketplace convention (description + model only). |
| **Frontmatter hygiene** | Minor issue | Router `description:` (line 3 of SKILL.md) does not mention LLM/AI or supply-chain even though both are in the pack. Does not start with "Use when..." (most modern packs do — see `analyzing-pack-domain` guidance and the marketplace majority). Wrapper has no frontmatter. |
| **Currency** | Pass | All cited frameworks are current as of late 2024 / 2025. |

**Overall:** **Major** — structurally sound and content-rich, but the router/wrapper drift around the two newer extension skills (llm/ai, supply-chain) and the stale `architecture-security-review` skill-name typo are real-user-impact issues. Not Critical; the pack is usable. Recommendation: **Enhance** with a routing/wrapper sweep and a description refresh; no rebuild needed.

---

## 4. Behavioral Tests

Per `testing-skill-quality.md`, behavioral tests probe whether components guide
Claude correctly under pressure. Reviewer ran scenarios mentally against the
text of each component (no live subagent dispatch — report-only). Results below
are confidence-graded.

### T1. Router activation — LLM agent task (real-world)
**Scenario:** "Threat model our customer support chatbot — it reads tickets and can issue refunds via a tool call."
**Expected:** Router routes to `llm-and-ai-security.md` + `threat-modeling.md`; cross-references `yzmir-llm-specialist` for prompt-side correctness.
**Result (against router SKILL.md lines 94-109, plus Example 4 at lines 366-376):** Pass. The router has an explicit "LLM / AI / Agentic Systems" extension block with matching symptoms and an exact worked example.
**Wrapper result:** **Fail** — `.claude/commands/security-architect.md` does not contain the LLM extension block at all. A user invoking `/security-architect` for this task would miss the dedicated sheet.

### T2. Router activation — Supply-chain task (real-world)
**Scenario:** "Design our SLSA L2 build pipeline producing signed SBOMs."
**Expected:** Route to `supply-chain-security.md`; consider compliance sheet for EO 14028 / EU CRA.
**Result (router SKILL.md lines 113-127, Example 5 at lines 378-388):** Pass.
**Wrapper result:** **Fail** — wrapper omits supply-chain extension entirely.

### T3. Router under stale-name pressure (edge case)
**Scenario:** Coordinator says "load `architecture-security-review`" (the stale name) — does the pack resolve?
**Result:** The actual file is `security-architecture-review.md`. The router SKILL.md uses the stale name in the Decision Tree (line 213), Pattern 2 (line 256), Quick Reference Table (lines 296, 298), and Example 3 (line 362). A model following the SKILL.md verbatim and trying to read `architecture-security-review.md` would 404. The body of the SKILL.md uses the *correct* file name in the Markdown link syntax (`[security-architecture-review.md](security-architecture-review.md)` lines 62, 187, 416), so a model resolving via the link succeeds — but the prose half of the routing table fails. **Fix needed.**

### T4. Agent activation — out-of-scope task (pressure / negative)
**Scenario:** "What threats does this have?" presented to `controls-designer` agent.
**Expected:** Decline; hand off to `threat-analyst`.
**Result:** `controls-designer.md` lines 36-38 explicitly include the negative example "What threats does this have? → Do NOT activate - threat analysis, use threat-analyst." Pass.
**Symmetric test:** `threat-analyst.md` lines 36-38 declines "Design security controls for the API" → directs to controls-designer. Pass.

### T5. SKILL pressure test — discipline holds (pressure)
**Scenario:** "We're behind schedule, skip threat modeling for the auth refactor."
**Expected:** `threat-modeling.md` should resist with the 10x-cost-after-deployment argument.
**Result:** Line 19 of `threat-modeling.md` states "Use BEFORE implementation - threats found after deployment are 10x more expensive to fix." Pass.

### T6. Compliance sheet — discovery discipline (pressure)
**Scenario:** "Just tell me the controls for our health app."
**Expected:** Sheet should refuse to skip the discovery questions (jurisdiction / industry / data type).
**Result:** `compliance-awareness-and-mapping.md` lines 30-52 require "Step 1: Ask Three Questions" before identifying frameworks, with explicit "Never assume." line 52. Pass.

### T7. Command — argument-hint guidance (edge case)
**Scenario:** User runs `/threat-model` with no argument.
**Result:** All three commands declare `argument-hint:` (`[system_or_component_to_analyze]`, `[boundary_or_component_to_secure]`, `[architecture_or_design_to_review]`). Body of `commands/security-review.md` (Step "Ask First", lines 20-25) prompts the user for the four scoping questions if not provided. Pass.

### T8. Agent SME-output conformance (real-world)
**Scenario:** A reviewer agent is dispatched to threat-analyze a system. Does the output structure include the four mandated SME sections?
**Result:** Both agents have the explicit body sentence "Your output MUST include Confidence Assessment, Risk Assessment, Information Gaps, and Caveats sections." (threat-analyst.md:10, controls-designer.md:10). Output-format templates in the body show STRIDE / risk-matrix scaffolds but **do not show the four SME sections inline in the template**. A model templating output strictly from the "Output Format" block (threat-analyst.md lines 108-167, controls-designer.md lines 92-176) might emit only the domain-specific table-of-contents and forget the SME sections. **Minor risk** — the protocol sentence is load-bearing; the template should reinforce it or include explicit `### Confidence Assessment` placeholders.

### T9. Classified-systems pressure (pressure / domain-specific)
**Scenario:** "We need to sanitize SECRET data so it can flow into the UNOFFICIAL system. Show me the sanitization function."
**Expected:** Sheet should refuse the "sanitize your way out" framing and redirect to MAC + impossible-to-construct-invalid-state.
**Result:** `classified-systems-security.md` line 8 states "You cannot 'sanitize' your way out of classification violations. Use mandatory access control (MAC) with fail-fast validation at construction time." The "no write down" rule (line 37) explicitly forbids the requested flow. Pass.

### T10. Supply-chain — recent-incident framing (real-world)
**Scenario:** "We had an xz-utils-style insider commit attack scare; how should our build pipeline have caught it?"
**Expected:** Sheet should map to SC-09 (insider commit pattern) and tie to SLSA build-step attestation / two-party review.
**Result:** `supply-chain-security.md` cites the xz-utils CVE-2024-3094 explicitly (line 8) and lists SC-09 "Insider commit (xz-utils pattern: trusted maintainer)" with CWE-1357 / T1195 / T1199 in the threat table (line 78). Pass.

### T11. Compliance pressure — assumed framework (pressure)
**Scenario:** "We're building a health app — what controls do we need for HIPAA?"
**Expected:** Sheet should refuse to assume HIPAA without jurisdiction confirmation (might be Australian Privacy Act + Privacy Principles, not HIPAA).
**Result:** `compliance-awareness-and-mapping.md` line 52 "Never assume. Same project can have multiple frameworks (e.g., Australian hospital SaaS = Privacy Act + Healthcare-specific + possibly SOC2 if B2B)." Pass.

### Summary
- Router skills: pressure-resistant on content, **fail on internal consistency** (stale skill name, wrapper drift).
- Specialist skills: well-structured, current frameworks, good "don't use for" sections, strong domain-pressure resistance (T5, T6, T9, T11).
- Commands: well-formed, conventional frontmatter, sensible tool restrictions.
- Agents: SME-compliant in body and frontmatter; the four SME sections should be reinforced in the Output Format templates to harden against template-only authoring.
- Confidence note: T1-T11 were reasoned against the text of the components, not run as live subagent dispatches. The findings against router/wrapper drift (Major #1, #2) are mechanically verifiable by grep and should be treated as high-confidence. The output-template observation (Minor #5) is a hypothetical failure mode that a model could exhibit if it templated strictly from the inline scaffold without re-reading the protocol sentence.

---

## 5. Findings

### Critical
None. The pack is usable as shipped.

### Major

1. **Stale skill-name `architecture-security-review` throughout router and wrapper.**
   - File is `security-architecture-review.md`. References to `architecture-security-review`:
     - `skills/using-security-architect/SKILL.md` lines 213, 256, 296, 298, 362.
     - `.claude/commands/security-architect.md` lines 38, 44, 63, 120, 128, 144 (six+ occurrences).
   - Impact: A model that resolves these as filenames will 404. Markdown-link resolution still works inside SKILL.md because the link syntax uses the correct name, but prose / table / decision-tree references are wrong.

2. **Slash-command wrapper `.claude/commands/security-architect.md` is stale.**
   - Omits the `llm-and-ai-security` and `supply-chain-security` extension blocks present in `SKILL.md` (which lists 5 extensions: classified, compliance, ATO, LLM/AI, supply-chain). Wrapper lists only the first three.
   - Uses dotted-path skill names (`ordis/security-architect/threat-modeling`) that do not match the marketplace plugin-name format used elsewhere.
   - Has no frontmatter at all; peer wrappers (`audit-pipelines.md`, `determinism-and-replay.md`, `creative-writing.md`) declare a `description:`.
   - Impact: a user invoking `/security-architect` for an LLM or supply-chain task gets a routing table that does not point at the right sheet.

3. **Router description is incomplete.**
   - `SKILL.md` line 3: `description: Routes to security architecture skills - threat modeling, controls, compliance, authorization`. Omits LLM/AI and supply-chain — which are advertised in plugin.json and marketplace.json descriptions and have 1,000+ lines of dedicated content. Affects skill discovery (description matching).
   - Also does not follow the "Use when..." opening convention dominant in this marketplace.

### Minor

4. **`security-controls-design.md:27`** points to `ordis/security-architect/secure-code-patterns` — file does not exist in the pack. Stale or aspirational reference.

5. **Agent Output Format templates lack inline SME-section placeholders.**
   - `agents/threat-analyst.md` lines 108-167 and `agents/controls-designer.md` lines 92-176 show domain output (STRIDE tables, control layers) but do not include `### Confidence Assessment` / `### Risk Assessment` / `### Information Gaps` / `### Caveats` headings inline. The protocol sentence above the template requires them, but a template-only authoring path may omit them.

6. **Mixed reference-style across the router.**
   - `SKILL.md` mostly uses Markdown links (`[threat-modeling.md](threat-modeling.md)`), but the Decision Tree (line 207ff), Pattern blocks (lines 245-275), Quick Reference Table (line 293), and Examples (lines 330-389) use bare skill names without links. The bare-name form is where the stale `architecture-security-review` lives. Consistent linkification would have prevented Finding #1.

### Polish

7. **PQC / crypto-agility not addressed at architecture level.** Optional extension; not on the marketplace's current radar but increasingly relevant. NIST IR 8547 (PQC transition), NSA CNSA 2.0, and the CNSF crypto-agility work are now established enough to warrant a short architecture pattern sheet (when to wrap a crypto provider, hybrid-PQC TLS, signature-algorithm negotiation).

8. **Skill descriptions in reference sheets lack frontmatter.** Reference sheets correctly omit frontmatter (per the marketplace convention they are content-only files referenced by the router), so this is consistent — noted here only to record that the choice was intentional and matches the rest of the marketplace.

9. **`compliance-awareness-and-mapping.md` mentions IRAP / ISM / PSPF (Australian) and UK Cyber Essentials.** Excellent multi-jurisdictional coverage. Worth surfacing in keywords / marketplace description (current keywords list is US/EU-centric: `fedramp`, `ato`, `rmf`, `nist-csf`, `iso-27001`, `pci-dss` but no `irap`/`ism`/`pspf`/`cyber-essentials`).

10. **Cross-pack ordering note in router (lines 238-241)** correctly distinguishes audit-pipelines (`/audit-pipelines`) from system threat modeling — "this pack designs the *evidence*; security architecture designs *system controls*". Clear handoff that avoids STRIDE-table duplication across packs. Strength worth preserving in any future edit pass.

11. **`llm-and-ai-security.md` boundary line (lines 38-40)** is well-drawn against `yzmir-llm-specialist`: this pack answers *"how does an attacker abuse this system and what controls block them?"*; the Yzmir pack answers *"how do I make this LLM application work well?"*. Strength worth preserving.

12. **`threat-modeling.md` line 28** correctly redirects audit-log threat modeling to `axiom-audit-pipelines:threat-model-for-audit-logs`. Cross-pack non-duplication discipline is excellent.

---

## 6. Recommended Actions

These are recommendations for a future maintenance pass; this review does not edit.

**Highest priority (Major):**

1. Run a global replace `architecture-security-review` → `security-architecture-review` in:
   - `plugins/ordis-security-architect/skills/using-security-architect/SKILL.md`
   - `.claude/commands/security-architect.md`
2. Rewrite `.claude/commands/security-architect.md` to mirror the current `SKILL.md`. Add a `description:` frontmatter line matching peer wrappers. Add the LLM/AI and supply-chain extension blocks. Replace dotted-path skill names with consistent reference style.
3. Update router `description:` to include "Use when..." phrasing and to mention LLM/AI and supply-chain coverage explicitly.

**Lower priority (Minor):**

4. Either create `secure-code-patterns.md` or remove the dangling reference at `security-controls-design.md:27`.
5. Add `### Confidence Assessment`, `### Risk Assessment`, `### Information Gaps`, `### Caveats` placeholders to the Output Format templates in both agent files so template-only authoring still emits the SME-protocol sections.
6. Linkify the Decision Tree, Pattern blocks, Quick Reference Table, and Examples in the router SKILL.md to use the Markdown-link form consistently. Prevents future stale-name drift.

**Polish:**

7. Consider adding `australian-government-isms`, `irap`, `uk-cyber-essentials`, `psnf` (or similar) to the plugin.json keyword list to surface the multi-jurisdictional compliance content.
8. If PQC/crypto-agility becomes a marketplace priority, scope a small extension sheet (~300 lines) that lives alongside `supply-chain-security.md` and `llm-and-ai-security.md` as a third "current-frontier" extension.

**Version bump if all above are applied:** Minor (1.2.0 → 1.3.0) — content/routing fixes plus refreshed wrapper, no philosophy change.

### Suggested fix sequencing
The edits naturally cluster into three independent commits:

1. **Routing consistency** (Findings #1, #4, #6) — global rename `architecture-security-review` → `security-architecture-review`; remove the dangling `secure-code-patterns` reference; linkify the Decision Tree, Pattern blocks, Quick Reference Table, and Examples in SKILL.md so future renames are caught by link-resolution rather than prose drift.
2. **Wrapper refresh** (Finding #2) — rewrite `.claude/commands/security-architect.md` to mirror current SKILL.md verbatim where possible; add `description:` frontmatter matching peer wrappers; this is the largest-impact change for end users invoking `/security-architect`.
3. **Description and SME template polish** (Findings #3, #5) — rewrite router `description:` line to start with "Use when..." and to mention LLM/AI + supply-chain; add SME-section placeholders into the agent Output Format templates. Optionally update plugin.json keywords for multi-jurisdictional discoverability.

Each cluster is independently shippable. Cluster 1 is the smallest and highest-confidence (mechanical) edit.

---

## 7. Reviewer Notes

- This was a read-only review against the SKILL.md + reference-sheet text and
  frontmatter; behavioral tests in §4 were reasoned against the text rather
  than dispatched to a subagent. Confidence is high for the structural and
  consistency findings (Major #1, #2, #3 are mechanical mismatches verifiable
  by grep) and medium for the behavioral judgements (T4-T8) which would benefit
  from live subagent runs in a future pass.
- The pack's content quality is genuinely strong. The two recent additions
  (`llm-and-ai-security.md` and `supply-chain-security.md`) are well-researched
  and current; their existence is the only reason the wrapper is stale — the
  router was updated, the wrapper was not. This is the classic
  "router-and-wrapper drift" failure mode that `reviewing-pack-structure.md`
  flags ("Router exists, no slash-command wrapper" / "Slash-command wrapper exists,
  no router skill") in a partial form: the wrapper exists but is materially
  out of sync.
- No Critical findings. No rebuild recommended. The pack is in good shape with a
  clearly-scoped enhancement pass needed.
- Sibling-skillpack interactions look healthy: clean handoff to
  `axiom-audit-pipelines`, `muna-technical-writer`, and `yzmir-llm-specialist`,
  with explicit "load both packs when..." guidance rather than duplicated
  content. The boundary calls in `llm-and-ai-security.md` (vs Yzmir LLM
  specialist) and in `threat-modeling.md` (vs audit-pipelines `07-threat-model.md`)
  are particularly well-drawn — they say exactly *which* problem each pack owns
  rather than waving toward "see also".
- Strengths to preserve in any maintenance pass:
  - The Core-vs-Extension split in the router (SKILL.md lines 178-200) gives
    callers a fast triage: "load core for anything; add an extension only
    when the context is explicit." This is the right shape for a domain pack
    with ~6,000 lines of specialist content — you do not want callers loading
    everything.
  - The "Don't use for" sections in every reference sheet (e.g.,
    `threat-modeling.md` lines 22-28; `secure-by-design-patterns.md` lines
    24-27; `compliance-awareness-and-mapping.md` lines 24-26) are consistent
    and well-disciplined. They route off to the right sibling rather than
    expanding scope.
  - The agents (`threat-analyst.md`, `controls-designer.md`) symmetrically
    decline each other's domain via negative `<example>` blocks. This pattern
    is the right way to enforce the SME boundary and is worth preserving as a
    reference for other Ordis-faction agents.
- Reviewer did not exercise Stage 5 (`implementing-fixes.md`) per instructions.

Methodology notes:
- Stage 1 (domain analysis) used `analyzing-pack-domain.md` Phase D→B→C→A; user scope was inferred from plugin.json + marketplace.json + the router SKILL.md self-description rather than asked interactively (per Auto Mode active in this session).
- Stage 2 (structure review) followed the `reviewing-pack-structure.md` fitness scorecard categories and the slash-command-wrapper alignment check.
- Stage 3 (behavioral testing) followed `testing-skill-quality.md` gauntlet (Pressure / Edge case / Real-world) at a confidence-noted level — no live subagent dispatches were performed. T1-T11 above are text-grounded reasoning, with verbatim line citations.
- Stage 4 (discussion) is captured here as the findings + recommendations sections rather than as a live conversation.

Files cited:
- `/home/john/skillpacks/plugins/ordis-security-architect/.claude-plugin/plugin.json`
- `/home/john/skillpacks/plugins/ordis-security-architect/skills/using-security-architect/SKILL.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/skills/using-security-architect/threat-modeling.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/skills/using-security-architect/secure-by-design-patterns.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/skills/using-security-architect/security-controls-design.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/skills/using-security-architect/security-architecture-review.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/skills/using-security-architect/documenting-threats-and-controls.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/skills/using-security-architect/llm-and-ai-security.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/skills/using-security-architect/supply-chain-security.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/skills/using-security-architect/classified-systems-security.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/skills/using-security-architect/compliance-awareness-and-mapping.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/skills/using-security-architect/security-authorization-and-accreditation.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/commands/threat-model.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/commands/design-controls.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/commands/security-review.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/agents/threat-analyst.md`
- `/home/john/skillpacks/plugins/ordis-security-architect/agents/controls-designer.md`
- `/home/john/skillpacks/.claude/commands/security-architect.md`
- `/home/john/skillpacks/.claude-plugin/marketplace.json`
