# Review: yzmir-deep-rl

**Version:** 1.3.0 (`/home/john/skillpacks/plugins/yzmir-deep-rl/.claude-plugin/plugin.json:3`)
**Reviewed:** 2026-05-22
**Reviewer:** general-purpose subagent

---

## 1. Inventory

**Plugin metadata** (`/home/john/skillpacks/plugins/yzmir-deep-rl/.claude-plugin/plugin.json`):
- `name: yzmir-deep-rl`, `version: 1.3.0`, `license: CC-BY-SA-4.0`.
- Description (line 4) enumerates the modern algorithm coverage (DQN/Rainbow/R2D2/BBF, PPO/GRPO, SAC/TD3/REDQ/DroQ/CrossQ, DreamerV3/TD-MPC2, CQL/IQL/TD3+BC/AWAC/Decision Transformer, MAPPO/IPPO, Go-Explore/NGU) and states "13 reference sheets, 3 commands (diagnose, scaffold-experiment, select-algorithm), 2 agents (rl-training-diagnostician, reward-function-reviewer)" — accurate.
- Keywords (lines 12–22) are concise and discovery-friendly: `yzmir`, `deep-rl`, `reinforcement-learning`, `ppo`, `sac`, `dqn`, `reward-shaping`, `counterfactual`, `shapley`, `credit-assignment`.

**Marketplace registration** (`/home/john/skillpacks/.claude-plugin/marketplace.json`): registered as `yzmir-deep-rl`, source `./plugins/yzmir-deep-rl`. The catalog `description` reads "Reinforcement learning - DQN, PPO, SAC, reward shaping, exploration - 13 skills" — much terser than the plugin.json description and **does not mention** the modern additions (GRPO, DreamerV3, TD-MPC2, R2D2, offline RL, multi-agent). Minor staleness against `plugin.json:4`.

**Slash-command wrapper:** **PRESENT** but **STALE**. `/home/john/skillpacks/.claude/commands/deep-rl.md` (602 lines).
- No YAML frontmatter (line 1 is blank, line 2 is the `#` heading). Several other repo wrappers do have frontmatter (`morphogenetic-rl.md` lines 1–3), though older ones (`python-engineering.md`, `ai-engineering.md`) omit it. Not strictly required, but inconsistent with the more recent convention.
- Line 19: states "It routes to **12 specialized skills**" — pack actually ships **13 reference sheets + 1 router + 1 scenarios doc** = 14 specialist files. Drift.
- Lines 34–47: lists "12 Deep RL Skills". Missing: `counterfactual-reasoning` (sheet 10 in the canonical router at `skills/using-deep-rl/SKILL.md:61`). Reordering and renumbering versus router (router has reward at #9, counterfactual at #10, debugging at #11; wrapper has reward at #9, debugging at #10).
- Router descriptions in wrapper omit modern algorithms (no R2D2/Agent57/BBF in value-based blurb, no DreamerV3/TD-MPC2 in model-based, no GRPO in policy-gradient, no REDQ/DroQ/CrossQ in actor-critic, no D4RL → Minari in offline-rl, no PettingZoo/SMACv2 in multi-agent). All of these are present in `plugins/yzmir-deep-rl/skills/using-deep-rl/SKILL.md:52–64`.

**Skills (1 router + 13 reference sheets + 1 scenarios doc, 15 files, ~21,500 lines):**

| File | Lines | Role |
|------|-------|------|
| `skills/using-deep-rl/SKILL.md` | 276 | Router (When to Use, How to Access Sheets, Core Principle, 13 Skills, Routing Decision Framework, Rationalization Resistance Table, Red Flags Checklist, Routing Decision Tree, Diagnostic Questions, When NOT to Use This Pack, Multi-Skill Scenarios, Final Reminders) |
| `rl-foundations.md` | 2187 | MDP, Bellman, value-vs-policy, exploration-exploitation |
| `value-based-methods.md` | 1296 | Q-learning, DQN, Double DQN, Dueling, PER, Rainbow, R2D2/Agent57/BBF/MEME (`:728–769`) |
| `policy-gradient-methods.md` | 1644 | REINFORCE, baselines/advantages, PPO, TRPO, GRPO (`:629–697`), Parts 12–17 advanced |
| `actor-critic-methods.md` | 1816 | A2C/A3C, SAC, TD3, REDQ/DroQ/CrossQ |
| `model-based-rl.md` | 1837 | World models, Dyna-Q, MBPO, Dreamer, **DreamerV3 / TD-MPC2 / MuZero / EfficientZero** (`:939–1045`) |
| `offline-rl.md` | 1712 | CQL, IQL, BCQ, TD3+BC, AWAC, Decision Transformer, D4RL→Minari |
| `multi-agent-rl.md` | 1793 | QMIX, MADDPG, MAPPO/IPPO, PettingZoo, SMACv2 |
| `exploration-strategies.md` | 1677 | ε-greedy, UCB, ICM, RND, Go-Explore, NGU/Agent57, BYOL-Explore |
| `reward-shaping-engineering.md` | 1050 | Potential-based shaping, inverse RL, reward hacking |
| `counterfactual-reasoning.md` | 1282 | Causal inference, HER, OPE, twin networks |
| `rl-debugging.md` | 1394 | 80/20 debugging methodology |
| `rl-environments.md` | 1822 | Gymnasium, MuJoCo, PettingZoo, Brax, Isaac Lab, EnvPool, Minari |
| `rl-evaluation.md` | 1591 | Sample efficiency, multi-seed, variance reporting |
| `multi-skill-scenarios.md` | 147 | Routing sequences for 7 common problem shapes |

**Frontmatter on reference sheets:** all 13 sheets + `multi-skill-scenarios.md` omit YAML frontmatter and start at H1. This matches the marketplace convention for content files under `skills/using-*/` per the rubric (`reviewing-pack-structure.md` / `using-skillpack-maintenance:SKILL.md:20` — "(none — content files referenced by a router SKILL.md)").

**Commands (3, ~419 lines):**

| Command | File | argument-hint | Tools (allowed-tools) |
|---------|------|---------------|------------------------|
| `/diagnose` | `commands/diagnose.md:1–5` | `[training_script.py or directory]` | `["Read", "Grep", "Glob", "Bash", "Skill"]` |
| `/scaffold-experiment` | `commands/scaffold-experiment.md:1–5` | `<experiment-name> [--algorithm=ppo\|sac\|dqn] [--env=CartPole-v1]` | `["Read", "Write", "Bash", "Skill"]` |
| `/select-algorithm` | `commands/select-algorithm.md:1–5` | `""` (empty) | `["Read", "Skill", "AskUserQuestion"]` |

All three commands declare `description`, `argument-hint`, and quoted-string `allowed-tools` per the convention in `using-skillpack-maintenance:SKILL.md:150–157`. Tool grants are minimum-needed: diagnose reads, scaffold writes, select-algorithm uses AskUserQuestion for interactive elicitation.

**Agents (2, 456 lines):**

| Agent | File | Model | SME compliance |
|-------|------|-------|----------------|
| `rl-training-diagnostician` | `agents/rl-training-diagnostician.md` | opus | YES — description ends "Follows SME Agent Protocol with confidence/risk assessment." (`:2`); body cites `meta-sme-protocol:sme-agent-protocol` (`:10`) and requires the four output sections by name in the body protocol line. **HOWEVER**, the *output template* at `:124–143` does NOT include `## Confidence Assessment`, `## Risk Assessment`, `## Information Gaps`, `## Caveats` sections — only the diagnostic output sections. The protocol assertion exists but the template doesn't enforce it. Cross-reference: `reward-function-reviewer.md:163–204` does include the full four-section template — proves the pack knows the pattern. |
| `reward-function-reviewer` | `agents/reward-function-reviewer.md` | opus | YES — description ends "Follows SME Agent Protocol with confidence/risk assessment." (`:2`); body cites the protocol (`:10`); output template at `:163–204` explicitly names all four required sections verbatim (Confidence Assessment, Risk Assessment, Information Gaps, Caveats). |

Neither agent declares `tools:`, correctly inheriting parent context. Model `opus` is defensible for both (multi-step reasoning over user code + reward design).

**Hooks:** none. No `hooks/` directory. Not required — RL guidance is reflective, not event-driven.

---

## 2. Domain & Coverage

**Domain scope (inferred from `plugin.json:4`, router `:5–18`, and skill bodies):**
- Mainstream online deep RL (value-based, policy-gradient, actor-critic)
- Modern algorithm coverage as of late-2025 / early-2026 (R2D2/Agent57/BBF, GRPO, DreamerV3/TD-MPC2, REDQ/DroQ/CrossQ, IQL, D4RL→Minari migration, PettingZoo/SMACv2)
- Offline RL as a first-class topic
- Multi-agent RL
- Exploration as a first-class topic
- Reward engineering as a first-class topic
- Counterfactual reasoning (HER, OPE, causal)
- Environments, debugging, evaluation as cross-cutting concerns

**Out of scope (correctly excluded per router `:222–231`):**
- Supervised training (→ `yzmir-training-optimization`)
- Architecture design (→ `yzmir-neural-architectures`)
- Deployment (→ `yzmir-ml-production`)
- LLM-specific RLHF recipes / DPO / IPO / KTO / SimPO (→ `yzmir-llm-specialist`)

**Coverage map vs. inventory:**

Foundational:
- MDPs, Bellman, value-vs-policy, exploration-exploitation — `rl-foundations.md` (2187 lines, exists).
- Policy gradient theorem, REINFORCE, baselines/advantages — `policy-gradient-methods.md:56–388` (exists).
- Q-learning, off-policy, TD targets — `value-based-methods.md:46–116` (exists).

Core algorithm families:
- DQN family + Rainbow + R2D2/Agent57/BBF — `value-based-methods.md` (exists, modernised).
- REINFORCE → PPO → TRPO → GRPO — `policy-gradient-methods.md` (exists, GRPO at `:629–697` cites Shao et al. 2024 + DeepSeek-R1 2025).
- A2C/A3C/SAC/TD3/REDQ/DroQ/CrossQ — `actor-critic-methods.md` (exists).
- Model-based: World models / MBPO / Dreamer / **DreamerV3** / **TD-MPC2** / MuZero/EfficientZero — `model-based-rl.md` (exists, modernised at `:939–1045` with citations to Hafner 2023 / Hansen 2024).
- Offline: CQL, IQL, BCQ, TD3+BC, AWAC, Decision Transformer — `offline-rl.md` (exists).
- Multi-agent: QMIX, MADDPG, MAPPO/IPPO with PettingZoo + SMACv2 — `multi-agent-rl.md` (exists).

Cross-cutting:
- Exploration: ε-greedy, UCB, ICM, RND, Go-Explore, NGU/Agent57, BYOL-Explore — `exploration-strategies.md` (exists).
- Reward shaping: potential-based, inverse RL — `reward-shaping-engineering.md` (exists).
- Counterfactual: HER, OPE, twin networks — `counterfactual-reasoning.md` (exists).
- Debugging: 80/20 methodology — `rl-debugging.md` (exists).
- Environments: Gymnasium, MuJoCo, PettingZoo, Brax, Isaac Lab, EnvPool, Minari — `rl-environments.md` (exists, current).
- Evaluation: multi-seed, sample efficiency, variance — `rl-evaluation.md` (exists).

**Gap analysis:**
- No dedicated **distillation / policy compression** sheet. Borderline — usually treated as deployment, defers to `yzmir-ml-production` (acceptable).
- No explicit **safe RL / constrained MDP / CMDP** coverage (Lagrangian PPO, CPO, RCPO). For a v1.3.0 RL pack this is a noticeable gap given safety is increasingly emphasised; but it's a Minor gap, not a Major one (most practitioners route to general PG/AC and add constraints).
- No explicit **meta-RL / RL² / MAML / PEARL** coverage. Niche; acceptable to defer.
- No explicit **hierarchical RL / options framework / feudal** coverage. Niche.
- **GRPO LLM recipe** is correctly redirected to `yzmir-llm-specialist` (router `:228–231`, PG sheet `:676`).
- **DPO/IPO/KTO/SimPO** correctly excluded as "not policy-gradient" (router `:228`, PG sheet `:696`).

**Research currency:** AI/ML domain — required flagging. Already-flagged additions visible: GRPO (2024), DreamerV3 (2023), TD-MPC2 (2024), MEME (2023), BBF (2023), D4RL → Minari transition, EnvPool, Brax, Isaac Lab. Currency through ~early-2025 looks good. The pack does NOT yet mention:
- **DPO-as-RL critiques** in counterfactual / offline sheets (low priority — it's an LLM topic anyway).
- **Diffusion-policy** approaches (Chi et al., 2023) for continuous control — increasingly common in robotics RL; possible Minor gap in `actor-critic-methods.md`.
- **CrossQ revisions** — already covered per `plugin.json:4`.

Overall coverage: strong. The 13-sheet decomposition matches the domain's natural fault lines, and the modern-algorithm additions (visible in both descriptions and bodies) are current to within ~1 year of the 2026-05-22 review date.

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Evidence |
|---|-----------|--------|----------|
| 1 | **Domain coverage completeness** | PASS | All foundational + core + cross-cutting topics present (`rl-foundations`, `value-based`, `policy-gradient`, `actor-critic`, `model-based`, `offline`, `multi-agent`, `exploration`, `reward`, `counterfactual`, `debugging`, `environments`, `evaluation`). Safe-RL / meta-RL gaps are niche, not foundational. |
| 2 | **Research currency** | PASS | Modern algorithms present and dated: GRPO (`policy-gradient-methods.md:678`, Shao 2024 + DeepSeek-R1 2025), DreamerV3 (`model-based-rl.md:943`, Hafner 2023), TD-MPC2 (`:1005`, Hansen 2024), BBF (`value-based-methods.md:766`, Schwarzer 2023), MEME (`:745`, Kapturowski 2023), D4RL → Minari migration noted in `plugin.json:4`. |
| 3 | **Router quality** | PASS with caveats | Router `skills/using-deep-rl/SKILL.md` (276 lines) is well-structured: decision framework by experience level / action space / data regime / special problems / debugging+infra; rationalization-resistance table at `:133–146` covers the 10 most common RL anti-patterns; red-flags checklist at `:150–162`; routing decision tree at `:166–196`; diagnostic questions at `:200–217`. **Caveat:** description (`:3`) is "Routes to appropriate deep-RL skills based on problem type and algorithm family" — does NOT use the "Use when..." discovery convention adopted by newer routers (e.g. `axiom-pyo3-interop`, `axiom-determinism-and-replay`, `lyra-creative-writing`). Several sibling Yzmir routers also omit this convention; not a critical issue but a discoverability minor. |
| 4 | **Slash-command wrapper** | **MAJOR** | Wrapper `/home/john/skillpacks/.claude/commands/deep-rl.md` is **stale**. Line 19 declares "12 specialized skills" but pack ships 13 (router itself lists 13 at `skills/using-deep-rl/SKILL.md:50–64`). The wrapper's enumerated list (`:34–47`) omits `counterfactual-reasoning` entirely, and the descriptive blurbs do not mention modern algorithms (R2D2/BBF, GRPO, DreamerV3/TD-MPC2, REDQ/DroQ/CrossQ) despite all being in the router itself. The wrapper is also missing YAML frontmatter (newer wrappers like `morphogenetic-rl.md` have it). Result: a user invoking `/deep-rl` sees a less complete, less current view than a model auto-invoking the router. |
| 5 | **Commands quality** | PASS | All three commands well-scoped, properly formatted, sensible tool restrictions. `/diagnose` enforces 80/20 (`commands/diagnose.md:9–19`). `/select-algorithm` is interactive via `AskUserQuestion` and prevents DQN-on-continuous (`:25`) and online-algos-on-offline-data (`:36`). `/scaffold-experiment` enforces reproducibility (seeds at `:29–41`), config-separate-from-code (`:46–64`), and eval-separate-from-train (`:81–96`). One minor inaccuracy in `/select-algorithm:108–112` references `/deep-rl:new-experiment` — the actual command is `/deep-rl:scaffold-experiment` (per `commands/scaffold-experiment.md:1`). |
| 6 | **Agents quality** | PASS with caveats | Both agents declare `description` + `model`, no spurious `tools:`. SME-protocol citations correct in both bodies. **Caveat:** `rl-training-diagnostician.md:124–143` output template does NOT include the four SME sections (Confidence / Risk / Information Gaps / Caveats) by name, even though the description (`:2`) and body protocol line (`:10`) claim compliance. By contrast, `reward-function-reviewer.md:163–204` correctly includes the full four-section template. Asymmetry between the two agents. |
| 7 | **Anti-pattern / rationalization coverage** | PASS | Rationalization tables present and rich: router `:133–146` (10 entries), `policy-gradient-methods.md:1036–1049` (10 entries), `model-based-rl.md:1352–1361` (8 entries). Red-flag checklists present at router `:150–162`, `value-based-methods.md` Part 10, `model-based-rl.md:1336–1347`. The 80/20 rule is repeated across the diagnostician agent (`:35–49`), `/diagnose` command (`:9–19`), and `rl-debugging.md` — appropriate redundancy for a discipline-enforcing pattern. |
| 8 | **Internal consistency** | PASS with caveats | Router skill list at `SKILL.md:50–64` matches the 13 files on disk. Per-sheet "When to Use" sections reference the right siblings ("route to value-based-methods", "route to offline-rl"). **Caveats:** (a) Marketplace catalog description is older/terser than `plugin.json` description — minor; (b) Slash-command wrapper says "12 skills" while everywhere else says 13 — major (see row 4); (c) `select-algorithm.md:111` references a non-existent `/deep-rl:new-experiment` command — minor; (d) Several reference sheets' "Do NOT use this skill for" lists reference `offline-rl-methods` rather than the actual sibling filename `offline-rl` (e.g. `policy-gradient-methods.md:26`, `value-based-methods.md:24`) — minor cosmetic. |

**Overall: PASS with two recurring MAJOR issues — the stale `.claude/commands/deep-rl.md` wrapper and the SME-template asymmetry on `rl-training-diagnostician`. Both are tightly scoped, low-risk fixes. No Critical issues. The pack is structurally sound and current.**

---

## 4. Behavioral Tests

Pack is large (~21.5K lines, 15 skill files). Sampled three skills across families per the brief — policy-gradient (PG), value-based (VB), model-based (MB) — plus router-level pressure tests. All tests run as document-level scenario walks (read-only review; no subagent dispatch).

### T1 — Router pressure: "I have logged production data; should I use PPO?"

**Scenario:** User has offline dataset, defaults to PPO under time pressure.
**Expected:** Router redirects to `offline-rl` (CQL/IQL); refuses PPO.
**Evidence:**
- Router `SKILL.md:104–107`: "**Red Flag:** If user has fixed dataset and suggests DQN/PPO/SAC, STOP and route to **offline-rl**. Standard algorithms assume online interaction and will fail."
- Router `SKILL.md:182`: "├─ OFFLINE data? → offline-rl (CQL, IQL) [CRITICAL]"
- Rationalization table `SKILL.md:139`: "More data always helps" → "Off-policy vs on-policy matters".
- `/select-algorithm:30–37` reinforces: "Offline data requires special algorithms. DQN/PPO/SAC will fail."
- Diagnostician agent `:111` lists offline-data + PPO/DQN/SAC as wrong algorithm.

**Result:** PASS. Triple-defended (router red flag + decision tree node + command + agent).

### T2 — Router pressure: "My PPO isn't learning; should I switch to SAC?"

**Scenario:** User wants algorithm-hop, classic anti-pattern.
**Expected:** Route to `rl-debugging` first, refuse algorithm change.
**Evidence:**
- Router `SKILL.md:128`: "If user immediately wants to change algorithms because 'it's not learning,' route to **rl-debugging** first."
- Router `SKILL.md:140`: "My algorithm isn't learning, I need a better one" → "Usually bugs, not algorithm" → "Route to rl-debugging first".
- Diagnostician agent `:21–23`: "User says 'PPO isn't working, should I try SAC?' → Trigger: STOP algorithm-hopping. Diagnose environment/reward first."
- `/diagnose:9–19` enforces 80/20 ordering.

**Result:** PASS. Quadruple-defended.

### T3 — Policy-gradient sheet, edge: "I'll discretize my robot's joint angles for DQN."

**Scenario:** Classic mistake of forcing discrete-action algorithm onto continuous problem.
**Expected:** Sheet refuses; routes to PG/actor-critic.
**Evidence:**
- `policy-gradient-methods.md:1040`: rationalization-table entry — "Can discretize but: curse of dimensionality (7D joint→7^n combos), loses continuous structure, very inefficient" → "Use policy gradients (PPO, SAC) naturally designed for continuous".
- Router `SKILL.md:95`: "**CRITICAL:** NEVER suggest DQN for continuous actions."
- Router `SKILL.md:141`: "I'll discretize continuous actions for DQN" → "Use actor-critic-methods".

**Result:** PASS.

### T4 — Value-based sheet, currency: "What's the SOTA for sample-efficient Atari?"

**Scenario:** User asks about Atari 100K regime.
**Expected:** Sheet should name BBF (Schwarzer 2023), not just Rainbow.
**Evidence:**
- `value-based-methods.md:728–769` covers Beyond Rainbow: R2D2 (`:732`), Agent57 (`:739`), MEME (`:745`), BBF (`:747`).
- `:759` table: "Atari 100k (sample-efficient lab work) → **BBF** → Designed exactly for this regime".
- Citations at `:766–769` to actual ICLR/ICML papers.

**Result:** PASS. Current within ~3 years.

### T5 — Model-based sheet, currency: "Should I use Dreamer or TD-MPC2 for my pixel robot?"

**Scenario:** User compares modern world-model methods.
**Expected:** Sheet should distinguish DreamerV3 vs TD-MPC2 with selection criteria.
**Evidence:**
- `model-based-rl.md:939–1045` is Part 5.5 "Modern World Models — DreamerV3, TD-MPC2, MuZero/EfficientZero".
- DreamerV3 selection at `:964–973`; TD-MPC2 selection at `:994–1003`.
- Comparison table at `:1037–1041`: "Pixels, low-sample, no MCTS engineering → **DreamerV3** vs DroQ (model-free)"; "High-quality continuous control → **TD-MPC2** vs DreamerV3 → Plan-at-inference; strong on DMC/MetaWorld".
- Pitfall reminder at `:1045`: "Even DreamerV3's world model is wrong somewhere; the policy will find and exploit those errors."

**Result:** PASS. Currency through 2024.

### T6 — Model-based sheet, edge: "I'll set k=50 for the planning horizon."

**Scenario:** User picks long imagined-rollout horizon, classic MBRL error.
**Expected:** Sheet refuses; cites MBPO short-rollout discipline.
**Evidence:**
- `model-based-rl.md:1338`: red-flag — "**Long rollouts (k > 20)**: Model errors compound, use short rollouts".
- `:1354` rationalization table: "k=50 is better planning" → "Errors compound, k=5 better" → "Use short rollouts, bootstrap value".
- `:1379–1381` summary key insight: "Keep k small (5-10), trust value function beyond".

**Result:** PASS. Triple-defended within the sheet.

### T7 — Reward-reviewer agent, real-world: alignment review of `reward = velocity`.

**Scenario:** User defines reward as `np.linalg.norm(velocity)` for a forward-walking task.
**Expected:** Agent flags as misaligned (will spin in circles).
**Evidence:**
- `reward-function-reviewer.md:42–48`: exact example. "Goal: Robot should walk forward / Reward: velocity (any direction) → reward = np.linalg.norm(velocity) # Agent will spin in circles!"
- `:102` lists "Oscillation instead of forward movement" as a common hacking pattern.
- Output template at `:162–204` mandates the four SME sections (Confidence / Risk / Information Gaps / Caveats).

**Result:** PASS. Compliant SME template + concrete anti-pattern.

### T8 — Training-diagnostician agent, SME-template compliance.

**Scenario:** Agent claims SME-protocol compliance; verify the output template contains the four required sections.
**Expected:** Template at `:124–143` should include "## Confidence Assessment", "## Risk Assessment", "## Information Gaps", "## Caveats" by name.
**Evidence:**
- `rl-training-diagnostician.md:2` description ends "Follows SME Agent Protocol with confidence/risk assessment."
- `:10` body protocol: "Your output MUST include Confidence Assessment, Risk Assessment, Information Gaps, and Caveats sections."
- `:126–142` output template: includes only "Phase 1: Environment / Phase 2: Reward Function / Phase 3: Algorithm Match / Root Cause / Recommended Fix". **The four SME sections are NOT in the template.**

**Result:** FAIL. Body promises four sections; template does not show them. Compare to sibling `reward-function-reviewer.md:163–204` which does include all four. Asymmetry; the diagnostician will plausibly skip the SME sections under time pressure because its own template does not display them.

### T9 — Slash-command wrapper consistency.

**Scenario:** User invokes `/deep-rl` and expects an accurate count and listing of specialist skills.
**Expected:** Match the router's 13-skill catalog.
**Evidence:**
- `.claude/commands/deep-rl.md:19`: "It routes to **12 specialized skills**".
- `.claude/commands/deep-rl.md:34–47`: lists 12 skills, missing `counterfactual-reasoning`.
- `skills/using-deep-rl/SKILL.md:50–64`: lists 13 skills including counterfactual-reasoning at #10.

**Result:** FAIL. Wrapper is stale by one skill and one count. See Major finding F1.

### T10 — `/select-algorithm` cross-reference integrity.

**Scenario:** User follows `/select-algorithm` next-step pointer.
**Expected:** Pointer to `scaffold-experiment` should resolve.
**Evidence:**
- `commands/select-algorithm.md:110`: "Use `/deep-rl:new-experiment --algorithm=[name]` to scaffold".
- Actual command file: `commands/scaffold-experiment.md` (name is `scaffold-experiment`, not `new-experiment`).

**Result:** FAIL (minor). Broken next-step pointer.

### Test summary

| # | Test | Result |
|---|------|--------|
| T1 | Offline-data → offline-rl routing | PASS |
| T2 | Algorithm-hop → rl-debugging routing | PASS |
| T3 | Discretize continuous → refuse | PASS |
| T4 | Atari sample-efficient currency (BBF) | PASS |
| T5 | DreamerV3 vs TD-MPC2 selection | PASS |
| T6 | k=50 rollout → refuse | PASS |
| T7 | Reward-reviewer misalignment flag | PASS |
| T8 | Diagnostician SME-template completeness | **FAIL** |
| T9 | `/deep-rl` wrapper skill-count | **FAIL** |
| T10 | `/select-algorithm` cross-reference | FAIL (minor) |

7 PASS, 3 FAIL (1 major-template, 1 major-wrapper, 1 minor-xref).

---

## 5. Findings

### Critical (0)

None.

### Major (2)

**F1 — Slash-command wrapper `.claude/commands/deep-rl.md` is stale by ≥1 skill, ≥6 modern algorithms.**
- Location: `/home/john/skillpacks/.claude/commands/deep-rl.md:19, :34–47`.
- Severity: Major — users invoking `/deep-rl` see a less complete and less current map than a model auto-invoking the router. This is the user-invocable face of the pack.
- Evidence:
  - `:19` claims "12 specialized skills"; canonical router `skills/using-deep-rl/SKILL.md:50` says "13 specialized skills".
  - `:34–47` enumerates 12 skills, omitting `counterfactual-reasoning` entirely.
  - Per-skill blurbs at `:36–47` use the pre-2024 algorithm wording (no R2D2/BBF, no GRPO, no DreamerV3/TD-MPC2, no REDQ/DroQ/CrossQ, no MAPPO/IPPO, no PettingZoo/SMACv2). All of these are in the router at `SKILL.md:52–64`.
  - No YAML frontmatter (minor, since older wrappers also lack it — but newer ones like `morphogenetic-rl.md` have it).
- Fix: rewrite the wrapper to mirror `skills/using-deep-rl/SKILL.md` (276 lines source-of-truth) — add `counterfactual-reasoning` at position #10, update count to 13, update per-algorithm blurbs to match the modern descriptions, optionally add YAML frontmatter `description:` matching the catalog.

**F2 — `rl-training-diagnostician` agent body promises four SME sections; output template omits them.**
- Location: `/home/john/skillpacks/plugins/yzmir-deep-rl/agents/rl-training-diagnostician.md:10` (promise), `:124–143` (template).
- Severity: Major — the agent will plausibly skip the SME sections under time pressure because its own example template does not show them. Asymmetric with `reward-function-reviewer.md:163–204` which gets this right.
- Evidence:
  - Description (`:2`): "Follows SME Agent Protocol with confidence/risk assessment."
  - Body (`:10`): "Your output MUST include Confidence Assessment, Risk Assessment, Information Gaps, and Caveats sections."
  - Template (`:126–142`): shows only Phase 1 / Phase 2 / Phase 3 / Root Cause / Recommended Fix. No "## Confidence Assessment", "## Risk Assessment", "## Information Gaps", "## Caveats" sections.
- Fix: append the four SME sections to the output template (copy from `reward-function-reviewer.md:163–204` and adapt the table rows).

### Minor (5)

**F3 — Marketplace catalog description is older/terser than `plugin.json` description.**
- Location: `/home/john/skillpacks/.claude-plugin/marketplace.json` (yzmir-deep-rl entry); `plugin.json:4` is the canonical longer description.
- Evidence: catalog reads "Reinforcement learning - DQN, PPO, SAC, reward shaping, exploration - 13 skills"; `plugin.json:4` enumerates modern algorithms. Discovery from the marketplace catalog will be less rich.
- Fix: bring catalog description into alignment (or accept terseness as deliberate — but if so, at least mention GRPO/DreamerV3 as marquee modern items).

**F4 — `commands/select-algorithm.md:110` references non-existent `/deep-rl:new-experiment` command.**
- Location: `/home/john/skillpacks/plugins/yzmir-deep-rl/commands/select-algorithm.md:110`.
- Evidence: actual file is `commands/scaffold-experiment.md` (name = `scaffold-experiment`).
- Fix: rename pointer to `/deep-rl:scaffold-experiment`.

**F5 — Several reference sheets refer to a sibling as `offline-rl-methods`; actual filename is `offline-rl`.**
- Locations (examples): `policy-gradient-methods.md:26`, `value-based-methods.md:24`. (Other "Do NOT use" lists may have the same.)
- Evidence: actual file `skills/using-deep-rl/offline-rl.md`. The string `offline-rl-methods` does not match the canonical name in the router `:60`.
- Fix: sweep-and-replace `offline-rl-methods` → `offline-rl` across all 13 reference sheets.

**F6 — Router description (`SKILL.md:3`) does not use the "Use when..." discovery convention.**
- Location: `/home/john/skillpacks/plugins/yzmir-deep-rl/skills/using-deep-rl/SKILL.md:3`.
- Evidence: current — "Routes to appropriate deep-RL skills based on problem type and algorithm family". Newer marketplace routers (e.g. `axiom-pyo3-interop`, `axiom-embedded-database`, `axiom-determinism-and-replay`, `lyra-creative-writing`) start with "Use when...". Several sibling Yzmir routers also omit; not strictly required but reduces auto-discoverability when paired against newer description-based ranking.
- Fix: optional — rewrite to "Use when implementing or debugging deep reinforcement learning — discrete or continuous control, online or offline data, single or multi-agent, model-based or model-free, exploration, reward design, evaluation. Routes to 13 specialist sheets."

**F7 — No `safe-RL / constrained MDP` coverage despite domain prominence.**
- Location: none — no `safe-rl.md` or constrained-MDP section in `policy-gradient-methods.md` / `actor-critic-methods.md`.
- Evidence: Lagrangian PPO, CPO (Achiam et al.), RCPO (Tessler et al.) are standard for safety-critical RL (robotics, finance, healthcare). Current pack handles "reward hacking" (under `reward-function-reviewer` and `reward-shaping-engineering.md`) but not constraint satisfaction.
- Fix: optional — add a `safe-rl.md` reference sheet, or add a Part to `policy-gradient-methods.md` / `actor-critic-methods.md`. Treat as a future-minor enhancement, not a current blocker.

### Polish (3)

**F8 — `multi-skill-scenarios.md` (147 lines) is much shorter than the other 13 sheets (1050–2187 lines).**
- Location: `/home/john/skillpacks/plugins/yzmir-deep-rl/skills/using-deep-rl/multi-skill-scenarios.md`.
- Evidence: file exists, 7 scenarios at ~10–20 lines each. Functional but could expand to include the modern problem shapes (LLM-RL via GRPO, robotic continuous control via TD-MPC2, multi-agent via MAPPO/PettingZoo).
- Fix: add 3–5 more modern scenarios.

**F9 — Reference sheet H1 inconsistency between files.**
- Evidence: some sheets begin with a leading blank line then H1 (e.g. `value-based-methods.md:1–2`); some start directly with H1 (e.g. `counterfactual-reasoning.md:1`). Cosmetic only.
- Fix: optional sweep to standardise.

**F10 — Diffusion-policy approaches (Chi et al. 2023) not mentioned in `actor-critic-methods.md`.**
- Evidence: increasingly common in continuous-control robotics; the pack covers SAC/TD3/REDQ/DroQ/CrossQ but does not flag diffusion-policy as an emerging family.
- Fix: optional — add a brief "Beyond Gaussian policies — diffusion policies" subsection in `actor-critic-methods.md`.

---

## 6. Recommended Actions

Ordered by priority and effort. All are surgical; no rebuild needed.

1. **(Major / F1)** Rewrite `/home/john/skillpacks/.claude/commands/deep-rl.md` to mirror `skills/using-deep-rl/SKILL.md`. Update count "12 → 13", add `counterfactual-reasoning` at position #10, refresh per-algorithm blurbs to include R2D2/Agent57/BBF, GRPO, DreamerV3/TD-MPC2, REDQ/DroQ/CrossQ, MAPPO/IPPO, PettingZoo/SMACv2, Decision Transformer. Optionally add YAML frontmatter `description:` matching the marketplace catalog. **Effort: small (~1 hour).**

2. **(Major / F2)** Append the four SME sections (Confidence Assessment, Risk Assessment, Information Gaps, Caveats) to `agents/rl-training-diagnostician.md` output template at `:124–143`. Copy structure from `agents/reward-function-reviewer.md:163–204` and adapt the table rows for diagnostic findings (environment, reward, algorithm, hyperparameters). **Effort: small (~30 minutes).**

3. **(Minor / F4)** Fix the broken next-step pointer at `commands/select-algorithm.md:110`: `/deep-rl:new-experiment` → `/deep-rl:scaffold-experiment`. **Effort: trivial.**

4. **(Minor / F5)** Sweep-and-replace `offline-rl-methods` → `offline-rl` across all 13 reference sheets. Spot-checked sites: `policy-gradient-methods.md:26`, `value-based-methods.md:24`. Other "Do NOT use this skill for" lists may have the same drift. **Effort: trivial (one global edit).**

5. **(Minor / F3)** Update marketplace catalog description for `yzmir-deep-rl` in `.claude-plugin/marketplace.json` to mention at least one modern marquee (GRPO or DreamerV3) so discovery from the catalog matches `plugin.json:4`. **Effort: trivial.**

6. **(Minor / F6, optional)** Rewrite router `SKILL.md:3` to start with "Use when..." for description-based discovery consistency with newer marketplace routers. **Effort: trivial.**

7. **(Minor / F7, deferable)** Decide whether to add safe-RL coverage. Either: (a) author a new `safe-rl.md` reference sheet covering Lagrangian PPO / CPO / RCPO; (b) add a "Part N: Constrained RL" section to `policy-gradient-methods.md` and `actor-critic-methods.md`; (c) defer to a future v1.4.0. **Effort: medium (new sheet ~1000 lines) or small (sections ~150 lines each).** Recommendation: defer to v1.4.0 unless safety-critical RL is a known user request.

8. **(Polish / F8, F9, F10)** Discretionary polish — expand `multi-skill-scenarios.md` with modern problem shapes; standardise H1 leading-blank-line; add a diffusion-policy subsection in `actor-critic-methods.md`. **Effort: small each, deferable.**

**Suggested version bump:** Patch (1.3.0 → 1.3.1) if only the Major and Minor items in steps 1–5 land — they are corrective. Minor (1.3.0 → 1.4.0) if Steps 1–8 land and safe-RL coverage is added. The current `plugin.json:4` description already enumerates ample content for a 1.x release.

---

## 7. Reviewer Notes

- **Methodology:** read-only. Stages 1–4 of `using-skillpack-maintenance`. Stage 5 skipped per instructions. No subagent dispatch — all behavioral tests are document-level scenario walks against the actual file contents, with concrete file:line citations.
- **Sampling:** per the brief, three skills across families — policy-gradient (`policy-gradient-methods.md`, head + GRPO section + rationalization table), value-based (`value-based-methods.md`, structure scan + R2D2/Agent57/BBF currency check), model-based (`model-based-rl.md`, structure scan + DreamerV3/TD-MPC2 section + rollout-horizon red flag). Both agents fully read; all three commands fully read; router fully read.
- **Confidence:**
  - HIGH on F1 (slash-command wrapper staleness) — directly visible diff between `.claude/commands/deep-rl.md:19` and `SKILL.md:50`.
  - HIGH on F2 (SME-template asymmetry) — direct comparison between the two agent templates.
  - HIGH on F3, F4, F5 — verified by grep/read.
  - MEDIUM on F6 — convention is real but mixed across marketplace; framed as Minor/optional.
  - LOWER on F7 (safe-RL gap) — judgment call; framed as deferable.
- **Did not test:**
  - Full reading of `rl-foundations.md` (2187 lines), `actor-critic-methods.md` (1816 lines), `multi-agent-rl.md` (1793 lines), `rl-environments.md` (1822 lines), `offline-rl.md` (1712 lines), `exploration-strategies.md` (1677 lines), `rl-evaluation.md` (1591 lines), `rl-debugging.md` (1394 lines), `counterfactual-reasoning.md` (1282 lines), `reward-shaping-engineering.md` (1050 lines). Confidence in these is from the table-of-contents grep and from the consistency of the patterns observed in the three sampled sheets — but specific within-sheet defects in the un-sampled files would not have been caught.
  - No live execution of the commands or agents — strictly document-level review.
  - Marketplace catalog `category:` field not inspected.
- **Read-only constraint observed:** no files were edited. The report is the deliverable.
- **Cross-pack interactions:** the pack correctly routes LLM-specific GRPO recipes to `yzmir-llm-specialist` (router `:228`, `policy-gradient-methods.md:676`) and excludes preference-optimization methods (DPO/IPO/KTO/SimPO) at `:228, :696`. The diagnostician agent's deferral-to-other-packs logic at `agents/rl-training-diagnostician.md:155–173` is well-structured (uses Glob to check installation before recommending).
- **Domain confidence:** HIGH — algorithm coverage is current through ~2024–early-2025 (BBF 2023, MEME 2023, DreamerV3 2023, TD-MPC2 2024, GRPO 2024, DeepSeek-R1 2025 cited).
