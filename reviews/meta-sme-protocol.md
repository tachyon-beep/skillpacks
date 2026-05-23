# Review: meta-sme-protocol
**Version:** 1.1.0  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

This pack defines the citation target invoked from ~30 downstream agents
across the marketplace. Its fitness is therefore not just internal — it is a
load-bearing contract for the ecosystem's reviewer/auditor/advisor agents.

---

## 1. Inventory

**Plugin root:** `/home/john/skillpacks/plugins/meta-sme-protocol/`

```
plugins/meta-sme-protocol/
├── .claude-plugin/
│   └── plugin.json                                          (v1.1.0)
└── skills/
    └── sme-agent-protocol/
        └── SKILL.md                                         (400 lines)
```

**Components:**

| Type | Count | Notes |
|---|---|---|
| Skills | 1 | `sme-agent-protocol` — single reference document, not a router |
| Reference sheets | 0 | All content in-line in SKILL.md |
| Commands | 0 | No `commands/` directory |
| Agents | 0 | No `agents/` directory |
| Hooks | 0 | No `hooks/` directory |

**Slash-command wrapper (repo-root `.claude/commands/`):**

| Expected | Present? |
|---|---|
| `.claude/commands/sme-agent-protocol.md` | **No** (verified via `ls /home/john/skillpacks/.claude/commands/`) |
| `.claude/commands/meta-sme-protocol.md` | **No** |

The sibling `meta-skillpack-maintenance` pack also has no slash wrapper, so
absence appears to be a deliberate convention for `meta-*` packs: they are
cited by other components, not user-invoked. See §5 (Polish) for whether to
document that convention.

**Marketplace registration:** Present in
`/home/john/skillpacks/.claude-plugin/marketplace.json` with name
`meta-sme-protocol`, source `./plugins/meta-sme-protocol`, keywords
`["meta", "sme", "agent", "protocol", "specialist"]`. Directory exists; no
orphaned entry.

**plugin.json content** (`/home/john/skillpacks/plugins/meta-sme-protocol/.claude-plugin/plugin.json:1-21`):
- `name`: `meta-sme-protocol`
- `version`: `1.1.0`
- `description`: "SME (Subject Matter Expert) Agent Protocol - mandatory
  protocol for all specialist agents defining fact-finding requirements,
  output contracts, confidence/risk assessment, and qualification of advice."
- `author.name`: `tachyon-beep`
- `license`: `CC-BY-SA-4.0`
- `keywords`: meta, sme, agent, protocol, specialist, expert, confidence,
  risk-assessment

The marketplace `description` is a slight truncation of plugin.json's
(dropped "fact-finding requirements" → "fact-finding"). Cosmetic.

**SKILL.md frontmatter**
(`/home/john/skillpacks/plugins/meta-sme-protocol/skills/sme-agent-protocol/SKILL.md:1-4`):
- `name`: `sme-agent-protocol`
- `description`: "Mandatory protocol for all SME (Subject Matter Expert)
  agents. Defines fact-finding requirements, output contracts,
  confidence/risk assessment, and qualification of advice."

The description does **not** start with the marketplace-dominant "Use
when..." phrasing. This is defensible for a reference-only skill that is
cited rather than auto-discovered, but worth noting against the convention
documented in `using-skillpack-maintenance` SKILL.md:132–133. See §5
(Polish).

**Downstream citations of this protocol** (grep for
`meta-sme-protocol:sme-agent-protocol` across `plugins/`):

| Pack | Citing agents (sample) |
|---|---|
| axiom-web-backend | api-architect.md, api-reviewer.md |
| axiom-python-engineering | python-code-reviewer.md, refactoring-architect.md |
| axiom-sdlc-engineering | bug-triage-specialist, architecture-decision-reviewer, sdlc-advisor, quality-assurance-analyst |
| axiom-system-architect | architecture-critic, debt-cataloger |
| axiom-audit-pipelines | integrity-auditor, audit-architecture-reviewer |
| axiom-devops-engineering | pipeline-reviewer, deployment-strategist |
| axiom-determinism-and-replay | determinism-reviewer |
| axiom-procedural-architecture | decomposition-architect, decomposition-critic |
| axiom-rust-engineering | rust-code-reviewer, clippy-specialist, unsafe-auditor |
| axiom-solution-architect | tech-selection-critic |
| axiom-static-analysis-engineering | false-positive-analyst |
| axiom-mcp-engineering | mentioned in router (planned agents) |
| yzmir-deep-rl | rl-training-diagnostician, reward-function-reviewer |
| yzmir-llm-specialist | llm-diagnostician, llm-safety-reviewer |
| yzmir-morphogenetic-rl | morphogenesis-reviewer, governor-design-reviewer |
| yzmir-neural-architectures | architecture-reviewer, architecture-advisor |
| yzmir-simulation-foundations | simulation-debugger, stability-analyst |
| yzmir-dynamic-architectures | dynamic-architecture-advisor |
| ordis-security-architect | threat-analyst, controls-designer |
| ordis-quality-engineering | coverage-gap-analyst, test-suite-reviewer, flaky-test-diagnostician |
| bravos-simulation-tactics | simulation-architect |
| muna-technical-writer | doc-critic |

Citation pattern is uniformly:

```
**Protocol**: You follow the SME Agent Protocol defined in
`meta-sme-protocol:sme-agent-protocol`. Before <action>, READ <X>.
Your output MUST include Confidence Assessment, Risk Assessment,
Information Gaps, and Caveats sections.
```

Verified verbatim in `axiom-web-backend/agents/api-architect.md`,
`ordis-security-architect/agents/threat-analyst.md`, and
`yzmir-deep-rl/agents/rl-training-diagnostician.md`. The four section names
(Confidence Assessment / Risk Assessment / Information Gaps / Caveats) are
**load-bearing** across the marketplace and match this pack's §3.1–§3.4
headers exactly.

---

## 2. Domain & Coverage

**Intended scope:** Define the cross-cutting contract that every Subject
Matter Expert agent in the marketplace honours — what they do before
analysing, what they emit, and how callers should calibrate trust.

**Coverage map** (per the pack's own structure):

| Concept | Status | Location |
|---|---|---|
| Phase 1: Fact-Finding (§1.1 read, §1.2 grep/glob, §1.3 skills/routers, §1.4 WebFetch/Search, §1.5 MCP tools, §1.6 subagents, §1.7 gaps) | Present | SKILL.md:29–99 |
| Phase 2: Analysis (evidence grounding, inference vs verification) | Present | SKILL.md:101–110 |
| Phase 3: Output Contract (the four required sections) | Present | SKILL.md:113–209 |
| §3.5 Machine-readable JSON summary (OPTIONAL) | Present | SKILL.md:211–235 |
| §3.6 Subagent-dispatch context (OPTIONAL) | Present | SKILL.md:237–243 |
| Anti-patterns (generic advice, hedging, no qualification, Python-only) | Present | SKILL.md:247–320 |
| Tool requirements (Read/Grep/Glob required; WebFetch/Search/Bash/Agent recommended) | Present | SKILL.md:324–344 |
| Integration checklist (for agents adopting protocol) | Present | SKILL.md:348–356 |
| ASCII summary diagram | Present | SKILL.md:362–390 |
| Changelog (1.0.0 → 1.0.1 → 1.1.0) | Present | SKILL.md:394–400 |

**Domain stability:** Stable. This is a documentation contract, not an
evolving technology surface. No external research currency check needed.

**Cross-cutting:** The protocol is positioned as language- and
domain-agnostic, with a Rust anti-pattern example (SKILL.md:303–320)
explicitly added in 1.1.0 to dispel the "Python-only" reading.

---

## 3. Fitness Scorecard

**Adjusted for a single-skill reference pack.** Activation/discovery and
slash-command exposure are de-emphasised because the pack is cited, not
discovered, and not user-invoked. Output-contract clarity and downstream
citability are weighted higher.

| Dimension | Rating | Notes |
|---|---|---|
| Domain coverage vs intent | Pass | Phases 1–3 are exhaustive for a contract document; optional §3.5/§3.6 cover known integration modes (machine parsing, subagent dispatch). |
| Four-section contract clarity | Pass | §3.1 Confidence, §3.2 Risk, §3.3 Information Gaps, §3.4 Caveats are each fully specified with example tables and definition rubrics (SKILL.md:117–209). Section names exactly match downstream citation text. |
| Vocabulary stability (confidence/risk scales) | Pass | High/Moderate/Low/Insufficient Data and Low/Medium/High/Critical defined in-place. Changelog (line 396) explicitly promises these are unchanged from 1.0.x — downstream-stable. |
| Citability from other packs | Pass | The `meta-sme-protocol:sme-agent-protocol` slug is referenced verbatim by ~30 agents and the citation template (Protocol line, READ verb, four-section requirement) is uniform. |
| Skill description / discoverability | Polish | Description does not lead with "Use when..." (vs. marketplace convention noted in `using-skillpack-maintenance` SKILL.md:132–133). Defensible because this skill is cited, not auto-invoked, but the convention deviation should be acknowledged. |
| Slash-command exposure | Polish | No `.claude/commands/sme-agent-protocol.md`. Defensible (sibling `meta-skillpack-maintenance` also has none; meta-* packs are not user-invoked). Should be explicitly documented as a deliberate non-decision somewhere persistent — e.g. plugin description or a top-of-SKILL note. |
| Anti-pattern coverage | Pass | Four anti-patterns (generic advice, unverified claims, skipping qualification, hedging without specifics) plus the Python-only counter-example. Each has BAD/GOOD pairs with concrete file paths. |
| SME-protocol-adopter guidance | Major | The Integration Checklist (SKILL.md:348–356) is thin — five bullets, no examples. The actual canonical citation snippet (`**Protocol**: You follow the SME Agent Protocol defined in ...`) is used uniformly downstream but is **not present** as a copy-pasteable template in this skill. Adopters must learn it by reading a downstream agent. See §5 Major Finding M1. |
| Changelog discipline | Pass | Each version notes additive vs. behavioural change. 1.1.0 explicitly promises §3.1–§3.4 unchanged so downstream agents need no edits (SKILL.md:396). |
| Frontmatter quoting safety | Pass | Frontmatter values are plain ASCII; no characters requiring strict-YAML quoting. (Recent ecosystem-wide concern per commit `4f8ba38`.) |

**Overall:** Pass with two structural improvements worth queuing — one
Major (no copy-pasteable adoption template), one Polish set (description
convention, slash-command rationale documentation). The pack is
production-grade and load-bearing; nothing here blocks current use.

---

## 4. Behavioral Tests

Test approach: this is a reference document, not an enforcing skill. The
relevant behavioural questions are (a) can a downstream agent author find
and apply the contract, and (b) does the contract resist
common-rationalisation pressures from agents asked to "just be helpful"?

### Test B1 — "I'm writing a new SME agent for pack X. What do I cite, and how?"

**Scenario:** A maintainer adopting the protocol for a new reviewer agent
reads the SKILL.md.

**Observed behaviour:**
- Found: Integration Checklist (SKILL.md:348–356) tells them to mention
  "follows SME protocol" in the description, include Read/Grep/Glob tools,
  reference the protocol, show all four sections in examples.
- **Not found:** The exact `**Protocol**: You follow the SME Agent
  Protocol defined in `meta-sme-protocol:sme-agent-protocol`. ...` line
  used in 30+ downstream agents. The author would have to grep an existing
  agent file to discover the standardised citation form.

**Result:** Fix needed (Major). The pack should ship the canonical
citation snippet so adopters match it verbatim. See M1.

### Test B2 — Pressure: "The user just wants a quick answer; skip the four sections this once."

**Scenario:** Agent under time pressure considers omitting Information
Gaps or Caveats.

**Observed behaviour in SKILL.md:**
- "Don't Skip the Qualification Sections" anti-pattern (SKILL.md:277–285):
  "Even if you're confident, always include: Confidence Assessment (even
  if all High), Risk Assessment (even if all Low), Information Gaps (even
  if 'None identified'), Caveats (even if minimal)."
- §3.6 (SKILL.md:237–243) reinforces: "Keep the four required sections
  (§3.1–§3.4) intact and in order — the dispatcher's parsing rules expect
  them."

**Result:** Pass. The skill explicitly defends against the "skip
qualification" shortcut, including the "even if None" carve-out that
removes the obvious rationalisation.

### Test B3 — Edge case: SME agent is invoked as a subagent by another agent (not by a human).

**Scenario:** Calling agent parses SME output programmatically.

**Observed behaviour:**
- §3.5 (SKILL.md:211–235) defines an OPTIONAL machine-readable JSON
  summary with bound vocabulary (same High/Moderate/Low/etc.) and an
  explicit "prose sections are authoritative" tie-breaker.
- §3.6 (SKILL.md:237–243) tells the SME to list outstanding investigations
  in Information Gaps rather than asking the dispatcher conversationally.

**Result:** Pass. Both the producer (SME) and consumer (dispatcher) sides
of subagent invocation are covered.

### Test B4 — Edge case: SME agent cannot find the code/docs the user mentions.

**Scenario:** User asks about a function the SME can't locate.

**Observed behaviour:**
- SKILL.md:127–129: "Insufficient Data: Cannot make claim without more
  information" is a first-class confidence value.
- §1.7 (SKILL.md:94–99): "Document What You Couldn't Find … Don't pretend
  you have more information than you do. This goes in the Information
  Gaps section."
- §3.6 (SKILL.md:243): "If you state `Confidence: Insufficient Data`,
  prefer that over guessing — the dispatcher can re-dispatch with more
  context, but it cannot un-trust a confidently-wrong claim."

**Result:** Pass. The "can't find it" case is explicitly handled
end-to-end (vocabulary, where to put it, why honesty matters).

### Test B5 — Edge case: SME is asked to apply the protocol to non-Python work (Rust, infra, prose).

**Scenario:** A Rust reviewer wonders if the protocol applies.

**Observed behaviour:**
- "Don't Treat the Protocol as Python-Only" anti-pattern
  (SKILL.md:301–320): explicit Rust BAD/GOOD example with
  src/cache.rs line references, compiler error E0499, the pattern
  comparison, and Confidence/Risk lines.

**Result:** Pass. The language-agnosticism point is made with an actual
foreign-language example, not just a disclaimer.

### Test B6 — Citability: Can a downstream pack cite the protocol with a stable slug?

**Scenario:** Confirm the `meta-sme-protocol:sme-agent-protocol` slug
works as a marketplace-wide reference target.

**Observed behaviour:** 30+ agents cite the slug verbatim. The slug
matches `{plugin-name}:{skill-name}` from plugin.json:2 and
SKILL.md:2. The pack is registered in marketplace.json, so the slug
resolves under any `/plugin install` of `meta-sme-protocol`.

**Result:** Pass.

---

## 5. Findings (Critical / Major / Minor / Polish)

### Critical

None.

### Major

**M1. No copy-pasteable canonical citation snippet for adopters.**

- **Evidence:** The Integration Checklist (SKILL.md:348–356) lists
  *requirements* ("Agent description mentions 'follows SME protocol' /
  Agent instructions reference this protocol") but does not show the
  uniform citation template that 30+ downstream agents actually use:

  ```
  **Protocol**: You follow the SME Agent Protocol defined in
  `meta-sme-protocol:sme-agent-protocol`. Before <verb>, READ <X>.
  Your output MUST include Confidence Assessment, Risk Assessment,
  Information Gaps, and Caveats sections.
  ```

  Verified verbatim in `axiom-web-backend/agents/api-architect.md`,
  `ordis-security-architect/agents/threat-analyst.md`,
  `yzmir-deep-rl/agents/rl-training-diagnostician.md`, and many others.
- **Why it matters:** The pack is the canonical citation target. The
  marketplace-wide convention is currently maintained by imitation across
  agent files. A new pack author who reads only this skill will not
  reproduce the standard line — they will paraphrase, and the
  marketplace's load-bearing string will drift.
- **Recommended fix:** Add a "Canonical Citation" subsection under
  Integration Checklist (around SKILL.md:348) with the exact `**Protocol**:`
  snippet and a one-line directive to copy it verbatim, parameterising
  only `<verb>` and `<X>`. Also reference it from the description-line
  guidance ("Description should end with 'Follows SME Agent Protocol with
  confidence/risk assessment.'") so adopters get both the description-line
  and the body-line conventions in one place.

### Minor

None identified.

### Polish

**P1. SKILL description does not lead with "Use when...".**

- **Evidence:** SKILL.md:3 — "Mandatory protocol for all SME (Subject
  Matter Expert) agents. Defines fact-finding requirements, output
  contracts, confidence/risk assessment, and qualification of advice."
- **Convention reference:** `using-skillpack-maintenance` SKILL.md:132–133
  documents "Use when..." as the dominant repo convention for
  discoverability.
- **Why downgrade to Polish rather than Major:** This skill is invoked by
  citation, not by description-based discovery. Forcing the "Use when..."
  prefix on a reference skill would be net-negative — the current
  description correctly characterises the skill as a contract, not a
  trigger. The deviation should be acknowledged rather than fixed.
- **Recommended action:** No change to the description. Optionally add a
  comment in the SKILL.md body or a `## Discovery` note explaining that
  the skill is intentionally cited rather than auto-invoked, so future
  reviewers don't flag it.

**P2. Slash-command absence is undocumented as deliberate.**

- **Evidence:** `ls /home/john/skillpacks/.claude/commands/` shows no
  `sme-agent-protocol.md` or `meta-sme-protocol.md`. The sibling
  `meta-skillpack-maintenance` is in the same situation.
- **Convention reference:** `reviewing-pack-structure.md` lines 128–142
  flags "Router exists, no slash-command wrapper" as a common issue, with
  the carve-out "or a documented reason it does not." That documented
  reason currently does not exist for either meta-* pack.
- **Why downgrade to Polish:** The pack has no `using-*` router skill;
  it's a reference skill. The maintenance rubric's wrapper check applies
  to router skills specifically. Absence is genuinely intentional.
- **Recommended action:** Add a one-line note to the plugin.json
  description (or a top-of-SKILL.md note) clarifying that this pack is
  cited from other packs' agents, not invoked by users — so the missing
  slash wrapper is a deliberate non-decision. Same fix would help
  `meta-skillpack-maintenance`.

**P3. Marketplace description is a slight truncation of plugin.json.**

- **Evidence:** plugin.json:4 says "fact-finding requirements, output
  contracts, confidence/risk assessment, and qualification of advice";
  marketplace.json (entry for `meta-sme-protocol`) drops "requirements"
  and "and qualification of advice" — "fact-finding, output contracts,
  and confidence/risk assessment".
- **Why Polish:** Cosmetic; both convey the same meaning. The
  marketplace string is read in `/plugin` browsing, so brevity is
  defensible.
- **Recommended action:** Either align both, or leave as-is. No
  user-visible defect.

**P4. ASCII art summary diagram (SKILL.md:362–390).**

- **Evidence:** A unicode-box summary diagram at the end of the file
  duplicates information in §1, §2, §3 headings.
- **Why Polish:** Diagram is occasionally helpful for readers who skim,
  but it does duplicate content. Consider whether retaining it pays for
  the line cost. (No action recommended unless space matters.)

---

## 6. Recommended Actions

In priority order:

1. **(M1) Add a "Canonical Citation" subsection** to the Integration
   Checklist (SKILL.md:~348) with the exact `**Protocol**:` body snippet
   and the exact description-line phrasing ("Follows SME Agent Protocol
   with confidence/risk assessment."). This converts a marketplace
   convention currently maintained by 30+ copies-of-the-same-line into a
   single source of truth in this skill.

2. **(P1 + P2 combined)** Add a short `## Discovery and Invocation` note
   to SKILL.md explaining (a) the skill is cited by other agents rather
   than auto-discovered, hence the description does not lead with "Use
   when..."; and (b) there is no slash-command wrapper because the
   protocol is not user-invoked. Two sentences each. Resolves both
   convention deviations as documented non-decisions.

3. **(P3)** Optionally align the marketplace description with
   plugin.json. Low priority.

4. **(P4)** Optionally trim the ASCII summary diagram. Low priority.

**Versioning:** A 1.1.0 → 1.1.1 patch bump covers P1/P2/P3/P4 (no
behaviour change). M1 is additive and behaviour-stable for existing
adopters (they already use the snippet); a 1.1.0 → 1.2.0 minor bump is
appropriate if M1 ships, with a changelog entry explicitly noting that
§3.1–§3.4 vocabulary remains unchanged (per the existing 1.1.0 promise at
SKILL.md:396).

**No execution undertaken in this review** — report only.

---

## 7. Reviewer Notes

- **Self-application discipline.** The maintenance rubric
  (`using-skillpack-maintenance`) was applied as written, with two
  dimensions de-emphasised (activation/discovery and slash-command
  exposure) because the pack is not a router. The dimensions that *do*
  apply for a reference pack — citability, contract clarity, vocabulary
  stability, adopter guidance — are weighted accordingly in §3.
- **Load-bearing-ness verified empirically.** The "four mandatory output
  sections" requirement specified in the task is unambiguously specified
  in §3.1–§3.4 with definition tables and example markdown. The four
  section names (Confidence Assessment / Risk Assessment / Information
  Gaps / Caveats) appear verbatim in 30+ downstream agent files, and the
  pack's changelog at line 396 explicitly commits to keeping them
  unchanged. The contract is sound.
- **Slash-command absence: intentional.** Both meta-* packs lack
  wrappers. Sibling pack `meta-skillpack-maintenance` has a `using-*`
  router skill (which the convention would normally wrap) and still
  has no wrapper. The pattern is consistent across the `meta-*`
  category, suggesting a deliberate marketplace decision rather than
  oversight. Logged as Polish, not Major.
- **Recent ecosystem context.** Commit `4f8ba38` ("fix(skills):
  quote-safe SKILL.md frontmatter for strict YAML parsers") suggests
  recent attention to frontmatter robustness; the meta-sme-protocol
  frontmatter is plain ASCII and unaffected.
- **What was not tested.** Stage 5 (Execution) was explicitly skipped per
  task instructions. No content was modified. Citation-pattern drift over
  time was not measured — a future quality check could grep all
  downstream agents and confirm 100% match against the canonical snippet
  once M1 ships.
