# Review: axiom-devops-engineering
**Version:** 1.1.4  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

---

## 1. Inventory

### Files on disk

```
plugins/axiom-devops-engineering/
├── .claude-plugin/plugin.json
├── skills/
│   └── cicd-pipeline-architecture/SKILL.md     (348 lines)
├── commands/
│   ├── review-pipeline.md                      (247 lines)
│   └── design-deployment.md                    (398 lines)
└── agents/
    ├── pipeline-reviewer.md                    (206 lines)
    └── deployment-strategist.md                (330 lines)
```

No `hooks/` directory. No router (`using-*`) skill. No reference sheets.

### Plugin metadata
`plugins/axiom-devops-engineering/.claude-plugin/plugin.json:1-19`
- `name: axiom-devops-engineering`
- `version: 1.1.4`
- `description: "DevOps and deployment automation - CI/CD pipelines, zero-downtime deployments - 1 skill, 2 commands, 2 agents"`
- Plugin description self-declares "1 skill, 2 commands, 2 agents" — matches the on-disk count.

### Marketplace registration
`.claude-plugin/marketplace.json` (line containing `axiom-devops-engineering`)
- Registered. `source: ./plugins/axiom-devops-engineering`.
- Catalog description: "DevOps and deployment automation expertise — CI/CD pipeline architecture, deployment strategies, zero-downtime deployments, and infrastructure reliability patterns — TDD-validated with RED-GREEN-REFACTOR testing".
- Catalog description claims a wider remit ("infrastructure reliability patterns") than the contents actually deliver (see Section 2).

### Slash-command exposure
- No `/home/john/skillpacks/.claude/commands/devops-engineering.md` (and no router skill to expose).
- Pack does NOT have a router skill, so no missing wrapper — but also no `/devops-engineering` slash command. This is consistent with single-skill plugins; not a defect by itself, but limits discoverability when users search by domain.

### Components

#### Skills (1)
| Skill | Description (frontmatter) | Status |
|---|---|---|
| `cicd-pipeline-architecture` | "Use when setting up CI/CD pipelines, experiencing deployment failures, slow feedback loops, or production incidents after deployment - provides deployment strategies, test gates, rollback mechanisms, and environment promotion patterns to prevent downtime and enable safe continuous delivery" (`skills/cicd-pipeline-architecture/SKILL.md:3`) | Frontmatter conforms ("Use when..."). Description is dense but parses cleanly. |

#### Commands (2)
| Command | Description | Tools | argument-hint | Status |
|---|---|---|---|---|
| `/review-pipeline` (file `commands/review-pipeline.md`) | "Review CI/CD pipeline for missing stages, anti-patterns, and production readiness" (`:2`) | `["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]` (`:3`) | `"[pipeline_file_or_directory]"` (`:4`) | Conforming JSON-array form. |
| `/design-deployment` (file `commands/design-deployment.md`) | "Design deployment strategy with zero-downtime, rollback capability, and verification gates" (`:2`) | same set (`:3`) | `"[application_or_service_name]"` (`:4`) | Conforming. |

#### Agents (2)
| Agent | Description | Model | SME-protocol | Status |
|---|---|---|---|---|
| `pipeline-reviewer` (`agents/pipeline-reviewer.md`) | "Review CI/CD pipelines for missing stages, anti-patterns, and production safety gaps. Follows SME Agent Protocol with confidence/risk assessment." (`:2`) | `sonnet` (`:3`) | YES — description ends with SME phrase; body has `**Protocol**:` line citing `meta-sme-protocol:sme-agent-protocol` (`:10`); requires the four sections. | Conforms. |
| `deployment-strategist` (`agents/deployment-strategist.md`) | "Design zero-downtime deployment strategies with rollback capability and verification gates. Follows SME Agent Protocol with confidence/risk assessment." (`:2`) | `sonnet` (`:3`) | YES — same protocol citation pattern (`:10`) | Conforms. |

Both agents omit `tools:` — correct per marketplace convention (5/65 agents declare `tools:`; both these inherit the parent context, which is appropriate for SME reviewers).

#### Hooks
None.

#### Cross-references
The skill's "Cross-References" block (`skills/cicd-pipeline-architecture/SKILL.md:329-334`) names five sibling skills:
- `test-automation-architecture` (ordis-quality-engineering) — EXISTS at `plugins/ordis-quality-engineering/skills/using-quality-engineering/test-automation-architecture.md`
- `observability-and-monitoring` (ordis-quality-engineering) — EXISTS
- `testing-in-production` (ordis-quality-engineering) — EXISTS
- `api-testing` (axiom-web-backend) — EXISTS as `api-testing.md`
- `database-integration` (axiom-web-backend) — EXISTS

All five resolve. Cross-references are accurate.

---

## 2. Domain & Coverage

### User-defined scope (inferred — no Stage-1 user interview here)
Plugin name and marketplace description claim:
- DevOps and deployment automation
- CI/CD pipelines
- Zero-downtime deployments
- "Infrastructure reliability patterns" (marketplace description only)

### Domain coverage map (what a comprehensive DevOps engineering pack should cover)

**Foundational (CI/CD pipeline mechanics)**
- Pipeline stages and gates — COVERED (the 7-stage model is the spine of the skill)
- Build artifact discipline (immutability, SHA tagging, build-once) — COVERED
- Test pyramid in CI / parallel execution / caching — COVERED at conceptual level
- Secrets management in CI — COVERED briefly (`SKILL.md:233-253`)
- Environment promotion — COVERED (`SKILL.md:256-269`)

**Core (Deployment patterns)**
- Blue-green / canary / rolling — COVERED in depth, in BOTH the skill and the `design-deployment` command and the `deployment-strategist` agent
- Health checks — COVERED
- Auto-rollback triggers — COVERED
- Database migrations (3-phase expand/migrate/contract) — COVERED

**Cross-cutting (operations)**
- Observability / monitoring of deployments — POINTED OUT only ("Monitor with dashboard"); the skill explicitly defers detail to `ordis-quality-engineering:observability-and-monitoring`. Reasonable scoping.
- Incident response post-deploy — NOT COVERED. Pointer to `axiom-engineering-foundations:incident-response` would be appropriate (that sheet exists).
- Feature flags / progressive delivery — MENTIONED in canary discussion but not developed
- Chaos engineering — NOT COVERED. Pointer to `ordis-quality-engineering:chaos-engineering-principles` would help.

**DevOps-adjacent (claimed by marketplace but absent from contents)**
- Infrastructure as Code (Terraform / Pulumi / CloudFormation) — NOT COVERED
- Container/image hygiene and registry policy beyond "tag with SHA" — NOT COVERED
- Kubernetes deployment patterns (beyond example snippets) — NOT COVERED in depth
- GitOps (ArgoCD, Flux) — NOT COVERED
- Cost and capacity planning — NOT COVERED
- Disaster recovery / backup strategy — NOT COVERED
- SRE practices (SLOs, error budgets) — NOT COVERED. (Plausibly out-of-scope; would be a `using-sre` or `ordis-*` sheet.)
- Platform engineering / golden paths — NOT COVERED
- Release engineering / versioning policy beyond pipeline-internal — NOT COVERED

### Coverage verdict
The pack delivers a **focused, single-topic** treatment of "CI/CD pipeline architecture with zero-downtime deployment strategies." It does this well, but it does NOT match the breadth its own marketplace description implies ("infrastructure reliability patterns"). The skill, two commands, and two agents are all **rotations of the same content**: the 7-stage pipeline + three deployment strategies + 3-phase migration + auto-rollback triggers. There is substantial redundancy by design (skill = reference; commands = workflows; agents = invocable specialists), which is acceptable provided each surface adds value for its activation mode.

### Audience
Implied: practitioners setting up CI/CD pipelines for the first time, or auditing an inherited one. The content reads as "production-ready opinionated baseline" — appropriate. Not exhaustive (no edge cases for monorepos, multi-region, regulated deployments, blue/green with stateful queues, etc.).

---

## 3. Fitness Scorecard

| Dimension | Rating | Evidence |
|---|---|---|
| **Router quality** | N/A (no router) | Single-skill plugin; no `using-devops-engineering` skill exists. A router is not strictly required for a single specialist, but the marketplace description's breadth ("infrastructure reliability patterns") would normally imply one. See Major-3. |
| **Skill descriptions** | Pass | Frontmatter `description` at `SKILL.md:3` opens with "Use when..." — conforming to the marketplace convention documented in `using-skillpack-maintenance/SKILL.md:133`. Triggers are concrete (setting up CI/CD, deployment failures, slow CI, production incidents). |
| **Frontmatter conformance** | Pass | Skill: `name` + `description`, no `allowed-tools` (correctly omitted; this is a process-doc skill, not a tool-restricting one). Commands: `description` (no trailing period), quoted JSON-array `allowed-tools`, quoted `argument-hint` — matches the convention in `using-skillpack-maintenance/SKILL.md:151-155`. Agents: `description` + `model` only, no `tools:` — matches the ~60/65 agents pattern. |
| **Component cohesion** | Minor | All five components (1 skill, 2 commands, 2 agents) overlap substantially — the same 7-stage model, the same blue/green/canary/rolling decision matrix, and the same migration phase model appear in 4 of 5 files. This is *intentional duplication for surface differentiation* but the duplication is verbatim in places (e.g. the "Strategy Selection Matrix" appears in `commands/design-deployment.md:32-37` and again in `agents/deployment-strategist.md:40-47` with near-identical wording). Maintenance burden is real: any change to the strategy matrix must be made in three places. See Minor-1. |
| **Slash-command exposure** | N/A (no router) | No router skill, so no missing `.claude/commands/devops-engineering.md` wrapper. The two task commands (`/review-pipeline`, `/design-deployment`) ARE the user-invocable surface — both are registered (visible in the system prompt's command list as `axiom-devops-engineering:review-pipeline` and `axiom-devops-engineering:design-deployment`). |
| **SME agent protocol** | Pass | Both `pipeline-reviewer.md:2,10` and `deployment-strategist.md:2,10` cite `meta-sme-protocol:sme-agent-protocol` and require the four output sections (Confidence Assessment, Risk Assessment, Information Gaps, Caveats). Descriptions end with the load-bearing phrase. |
| **Anti-pattern coverage** | Pass | Skill has explicit "Common Mistakes" table (`SKILL.md:288-299`), "Rationalization Table" (`:301-311`), and "Red Flags" list (`:313-325`). Commands and agents include condensed anti-pattern tables. Anti-pattern density is good. |
| **Cross-skill linkage** | Pass | Five cross-references at `SKILL.md:329-334` all resolve to real skills in sibling plugins. Commands include a `## Cross-Pack Discovery` block (`design-deployment.md:369-383`) with a Python snippet — see Minor-2 for a critique. |

**Overall: Minor** — the pack is structurally sound, behaviorally consistent, frontmatter-conformant, and SME-compliant. The defects are content duplication, an oversold marketplace description, and a few content gaps that the pack could reasonably claim to cover (incident response, chaos pointer, IaC scoping note).

---

## 4. Behavioral Tests

I ran three scenarios — one against the discovery layer (does the skill activate?), one pressure test, and one against the deployment-strategist agent — by reading the SKILL.md/command/agent files and reasoning about what guidance Claude would produce. (Per the rubric, subagent dispatch would be the higher-fidelity option for a real maintenance pass; here I model expected behavior from the artifact text.)

### Test 1: Discovery / activation
**Scenario:** A developer asks: "Our deployments keep breaking production. What should we do?"

**Expected:** The skill should activate because its description includes "experiencing deployment failures" and "production incidents after deployment" (`SKILL.md:3`).

**Likely actual:** Activation succeeds. The first thing Claude is told (lines 12-13) is the core principle that "Deploy to production" is a sequence of gates — directly relevant. The "When to Use" list (`:16-23`) contains exact matches. The skill then walks the 7-stage pipeline.

**Verdict:** PASS.

### Test 2: Pressure resistance ("just ship it")
**Scenario:** "We have a hotfix that needs to go out in 20 minutes. I don't have time for staging or blue/green. Just deploy it."

**Expected:** Skill should resist the shortcut, name the shortcut, and either (a) refuse and require staging, or (b) offer a documented exception path (hotfix lane with reduced gates) without quietly abandoning the discipline.

**Likely actual:**
- The "Do NOT skip this for" block (`SKILL.md:25-29`) explicitly anticipates "Quick MVP / demo" and "We'll improve it later" — but does NOT anticipate "20-minute hotfix" as a pressure mode.
- The "Rationalization Table" (`:301-311`) and "Red Flags" (`:313-325`) cover related rationalizations ("just push to main", "tests passed locally") but not the specific hotfix-urgency pattern.
- Claude would likely cite the Red Flags and refuse, but the skill does NOT give Claude a *graceful fallback path* (e.g. "if urgency is real, here is the minimum-safe shortcut") — so the response risks being unhelpful in a genuine incident.

**Verdict:** PARTIAL PASS. Pressure resistance works for "we'll do it later" framings; weaker for "incident-driven urgency." Add a Minor finding (4): no documented hotfix/emergency-deploy path. This is a real production scenario the pack should address — even if the answer is "the answer is still no, and here is why production-safe is faster than you think."

### Test 3: Deployment-strategist scope boundary
**Scenario:** A coordinator dispatches `deployment-strategist`: "Review our existing GitLab pipeline and tell us what's wrong."

**Expected:** Agent should DECLINE (this is a review task) and hand off to `pipeline-reviewer`. The agent's own activation examples include a negative example for exactly this (`agents/deployment-strategist.md:33-36`).

**Likely actual:** Agent declines correctly because the negative example is explicit: `"Review our existing pipeline" → Do NOT activate - review task, use pipeline-reviewer`. Symmetric negative example exists in `agents/pipeline-reviewer.md:33-36` for the inverse case ("Design a new deployment strategy" → use deployment-strategist).

**Verdict:** PASS. Scope boundaries between the two agents are explicit and bidirectional.

### Test 4 (bonus): Real-world complexity — database-heavy service
**Scenario:** "We have a service with a 2TB Postgres database that we're modifying. Design a deployment."

**Expected:** Strategist should pick blue-green or rolling, and route the schema change through the 3-phase migration (expand → migrate → contract). The strategy comparison table at `agents/deployment-strategist.md:308-314` explicitly maps "Database-heavy → Blue-Green + 3-phase migration."

**Likely actual:** Agent produces the correct recommendation. Migration template at `:212-237` covers expand/migrate/contract. However, the template does NOT address backfill strategy for a 2TB table, online index creation timeouts, or lock-acquisition risk on `ALTER TABLE` — practitioner gaps. Likely sufficient for a teaching baseline; thin for a production runbook.

**Verdict:** PASS for teaching baseline; INCOMPLETE for production at scale. Not a defect per se — pack is correctly scoped for "design baseline" not "production runbook for 2TB Postgres" — but worth noting that the pack's `argument-hint` "application_or_service_name" implies it works for arbitrary scale.

---

## 5. Findings

### Critical
None. The pack is functional, registered, frontmatter-conformant, SME-compliant, and its content is technically correct.

### Major

**Major-1: Marketplace description oversells the contents.**
- `.claude-plugin/marketplace.json` describes the pack as covering "infrastructure reliability patterns" in addition to CI/CD. The pack does NOT cover infrastructure provisioning, IaC, observability, incident response, chaos engineering, SRE practices, capacity planning, or DR. A user installing on the basis of the catalog description will find a narrower pack than promised.
- **Two valid fixes:** (a) narrow the catalog description to match the contents ("CI/CD pipeline architecture and zero-downtime deployment strategies"), or (b) expand the pack to include the missing scope. (a) is cheaper and honest.

**Major-2: No incident-response or rollback-execution pointer.**
- The skill mandates auto-rollback triggers (`SKILL.md:179-184`) but says nothing about what happens *after* a rollback fires: who is paged, how the incident is logged, how to do a post-mortem. `axiom-engineering-foundations:incident-response` exists (`plugins/axiom-engineering-foundations/skills/using-software-engineering/incident-response.md`) and should be linked from the Cross-References block (`SKILL.md:329-334`) — it is the natural follow-on.

**Major-3: No "using-devops-engineering" router despite the marketplace description's breadth.**
- The marketplace description implies a multi-topic pack but ships a single skill. Either:
  - Convert `cicd-pipeline-architecture` into a router with reference sheets (one for build/test, one for deployment strategies, one for migrations, one for verification/monitoring) — currently the SKILL.md is 348 lines covering all four topics, which is on the long side for a single skill and a natural router/sheet boundary; or
  - Leave structure as-is and narrow the marketplace description (Major-1).
- These two fixes are interdependent. Pick one path.

### Minor

**Minor-1: Strategy matrix duplicated verbatim across 3 files.**
- `SKILL.md:115-161` (strategy descriptions)
- `commands/design-deployment.md:32-154` (decision matrix + strategy implementations)
- `agents/deployment-strategist.md:40-165` (matrix + templates)
- Any change to canary thresholds, blue/green cleanup window, or rolling config must be edited in three places. Risk: drift over time. Consider extracting strategy specs to a single reference sheet (forces Major-3 path A) and cross-referencing from the others.

**Minor-2: `## Cross-Pack Discovery` Python snippet is brittle.**
- `commands/design-deployment.md:371-383` contains a Python snippet that uses `glob.glob("plugins/ordis-quality-engineering/plugin.json")` to detect sibling packs. The actual file is `.claude-plugin/plugin.json` (not `plugin.json`). This snippet, if executed, finds nothing — it's silently wrong. Either fix the path (`plugins/*/. claude-plugin/plugin.json`) or replace the snippet with prose pointers (which is what every other pack does).

**Minor-3: No documented hotfix / emergency-deploy path.**
- See Test 2 above. The skill's "Do NOT skip this for" block (`SKILL.md:25-29`) and Rationalization Table (`:301-311`) cover lazy rationalizations but not incident urgency. Add a brief section: "For genuine production-down hotfixes: the minimum-safe shortcut is X (still gates Y and Z; explicitly skips W with documented risk)."

**Minor-4: Database migration template is teaching-grade, not runbook-grade.**
- The 3-phase template (`SKILL.md:201-216`, `commands/design-deployment.md:265-289`, `agents/deployment-strategist.md:213-237`) does not address: backfill batching, lock timeouts, `ALTER TABLE` blast radius on large tables, partial-rollback cleanup of phase-2 dual-writes, or migration observability. Acceptable for the pack's scope ("design baseline"), but should be explicit about the limit — perhaps a one-line "for tables >1M rows or critical schemas, see additional runbook" deferral.

**Minor-5: "Auto-rollback triggers" thresholds presented as fixed values.**
- `SKILL.md:181-184` gives concrete thresholds (error rate > 5% for 3 min, response time > 2x baseline, etc.). These are reasonable defaults but presented without acknowledging that the *right* thresholds depend on baseline error rates and traffic shape. A service with a steady 0.5% error rate and one with a steady 2% error rate need different triggers. Add a sentence: "Calibrate to your baseline; these are starting points."

### Polish

**Polish-1: Description string at `SKILL.md:3` is one 60-word sentence.**
- Parses correctly per the quoted-YAML safety pattern that recently landed (commit `4f8ba38`), but readability would improve with a shorter trigger clause + colon + summary. Optional.

**Polish-2: The two agents' `## When to Activate` sections have identical structure but the positive/negative examples could be sharper.**
- `pipeline-reviewer.md:18-36` and `deployment-strategist.md:18-36`. The negative examples are useful (bidirectional handoff) but the positive examples are very generic ("Review this CI/CD pipeline for issues"). Could be tightened to include the disambiguating signal that triggered this agent vs. (say) a generic security review.

**Polish-3: "TDD-validated with RED-GREEN-REFACTOR testing" in the marketplace catalog description.**
- This phrase appears in many pack descriptions. For a process-doc pack like this one, "TDD-validated" is a metaphor (per repo CLAUDE.md). The repository's own CLAUDE.md acknowledges this. Not a defect, but worth being aware that this marketing phrase doesn't map cleanly to what was actually done.

---

## 6. Recommended Actions

(Report-only — no edits made.)

**Pick one of Major-1/Major-3 first; the choice determines all subsequent work.**

**Path A — narrow scope to match contents:**
1. Update `.claude-plugin/marketplace.json` description to drop "infrastructure reliability patterns" and accurately reflect the CI/CD focus.
2. Address Minor-1 by extracting the strategy matrix to a shared reference sheet that all three surfaces include. (This formally creates a router, but a minimal one.)
3. Fix Minor-2 (broken Python snippet).
4. Add Major-2 cross-reference to `axiom-engineering-foundations:incident-response`.
5. Add Minor-3 hotfix path and Minor-5 threshold calibration sentence.
6. Patch bump → 1.1.5 (Minor-2 is a fix; rest is content polish).

**Path B — expand to match marketplace description:**
1. Promote `cicd-pipeline-architecture` to a router (`using-devops-engineering`) with reference sheets for: build, test, deployment strategies, migrations, verification, secrets.
2. Add new skills/sheets for the gaps: incident response (or pointer), chaos pointer, observability pointer, optionally an IaC overview sheet.
3. Add a `/devops-engineering` slash-command wrapper at `.claude/commands/devops-engineering.md`.
4. Resolve all Minor findings as part of the expansion.
5. Minor bump → 1.2.0.

**Independent of path:** Polish-1, Polish-2, Polish-3 are optional. Polish-3 should be considered marketplace-wide, not per-pack.

---

## 7. Reviewer Notes

- **Methodology limit.** Behavioral tests (Section 4) were modeled by reading artifacts, not by dispatching a fresh-context subagent. For a real maintenance pass per `testing-skill-quality.md:80-92`, subagent dispatch is the preferred mechanism. The findings here would survive subagent confirmation, but the pressure resistance test (Test 2) in particular benefits from a real fresh-session run where the model's rationalization pathways are not already primed by reading the skill.
- **No skill content was edited.** This is a report-only review.
- **Scope vs. quality.** Within its actual scope (CI/CD + deployment strategy + 3-phase migration), this pack is genuinely solid: SME-compliant, frontmatter-conformant, cross-references resolve, scope boundaries between the two agents are bidirectional and explicit. The findings above are about edges of scope and content duplication, not about the core content being wrong.
- **Comparison to sibling axiom packs.** Single-skill plugins are uncommon in this marketplace. Of 41 plugins, most either have multiple specialist skills under a `using-*` router (e.g. `axiom-engineering-foundations`, `axiom-web-backend`, `ordis-quality-engineering`) or are deliberately single-purpose (e.g. `axiom-pyo3-interop`, `axiom-embedded-database` — but even these have 13 reference sheets). `axiom-devops-engineering` at 1 skill + 2 commands + 2 agents is structurally lean. This is not wrong, but it does explain Major-1 / Major-3: the catalog presents it as a peer of those richer packs.
- **No blockers to use.** Anyone who installs this today gets correct, actionable CI/CD guidance. There is no Critical defect.
