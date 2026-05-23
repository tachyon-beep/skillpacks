# Review: yzmir-ai-engineering-expert
**Version:** 1.1.0 **Reviewed:** 2026-05-22 **Reviewer:** general-purpose subagent

This is a meta/router pack for the Yzmir AI/ML family. Its sole responsibility is to read a user's
AI/ML query, identify the problem type, and route to the correct specialist pack (or sequence of
packs). The pack ships a single router skill (`using-ai-engineering`) with one reference sheet
(`routing-examples.md`) and a repo-root slash-command wrapper at
`/home/john/skillpacks/.claude/commands/ai-engineering.md`. The review applied Stages 1-4 of
`meta-skillpack-maintenance:using-skillpack-maintenance`.

---

## 1. Inventory

### Files in the pack
- `/home/john/skillpacks/plugins/yzmir-ai-engineering-expert/.claude-plugin/plugin.json`
- `/home/john/skillpacks/plugins/yzmir-ai-engineering-expert/skills/using-ai-engineering/SKILL.md` (286 lines)
- `/home/john/skillpacks/plugins/yzmir-ai-engineering-expert/skills/using-ai-engineering/routing-examples.md` (321 lines)

### Slash-command wrapper
- `/home/john/skillpacks/.claude/commands/ai-engineering.md` (647 lines, no YAML frontmatter)

### Components observed
| Component type | Count | Notes |
|----------------|-------|-------|
| Skills | 1 (router) | `using-ai-engineering` |
| Reference sheets | 1 | `routing-examples.md` |
| Commands (pack-level) | 0 | None expected; pack is router-only |
| Agents | 0 | None expected for a router pack |
| Hooks | 0 | N/A |
| Slash-command wrapper | 1 | `/ai-engineering` |

### plugin.json (lines 1-16)
```
"name": "yzmir-ai-engineering-expert",
"version": "1.1.0",
"description": "Primary router for all 9 Yzmir AI/ML engineering skillpacks - covers framework,
training, RL, LLM applications (incl. reasoning, agentic/MCP, multimodal, RAG), neural
architectures, ML production, dynamic architectures, simulation foundations, and systems thinking"
```

### Marketplace registration
Confirmed at `/home/john/skillpacks/.claude-plugin/marketplace.json:515-524`. Source path correct,
category `ai-ml`. The marketplace description is much shorter than the plugin.json description
("Primary router for all AI/ML engineering skillpacks - directs to specialized packs") and does
not assert a pack count — that is fine.

### Sibling Yzmir packs that exist in this marketplace (the routing surface)
Discovered via `ls plugins/ | grep yzmir`:

1. `yzmir-pytorch-engineering`
2. `yzmir-training-optimization`
3. `yzmir-deep-rl`
4. `yzmir-llm-specialist`
5. `yzmir-neural-architectures`
6. `yzmir-ml-production`
7. `yzmir-dynamic-architectures`
8. `yzmir-simulation-foundations`
9. `yzmir-systems-thinking`
10. **`yzmir-morphogenetic-rl`** (the 10th sibling)

That is **10 specialist packs**, not 9. The plugin.json description, the SKILL.md catalog
section (line 268: "9 specialist packs plus this router"), and the `routing-examples.md` content
all assert 9. `yzmir-morphogenetic-rl` is **entirely absent** from the router — zero mentions
across SKILL.md, routing-examples.md, and the slash-command wrapper (verified by `grep -c
morphogenetic-rl` returning 0/0/0). See Findings §5.1.

---

## 2. Domain & Coverage

### Intended scope
A first-touch router that disambiguates AI/ML queries and dispatches to the correct Yzmir
specialist pack, including handling cross-cutting queries (multi-pack routing) and resisting
common rationalization pressures (time, authority, sunk cost, keyword anchoring).

### Coverage map (problem types this router must distinguish)

| Domain | Target pack | Status |
|--------|-------------|--------|
| Framework / GPU / distributed (PyTorch foundation) | `yzmir-pytorch-engineering` | Covered (SKILL.md:58) |
| Training instability, optimizers, precision | `yzmir-training-optimization` | Covered (SKILL.md:59) |
| Reinforcement learning (agents, policies, MDP, MARL) | `yzmir-deep-rl` | Covered (SKILL.md:60) |
| LLM applications: prompting, fine-tuning, RAG | `yzmir-llm-specialist` | Covered (SKILL.md:60) |
| LLM (reasoning models, o-series, extended thinking) | `yzmir-llm-specialist` | Covered (SKILL.md:61) |
| LLM (agentic, tool use, MCP, multi-agent) | `yzmir-llm-specialist` | Covered (SKILL.md:62) |
| LLM (multimodal / VLM) | `yzmir-llm-specialist` + `yzmir-neural-architectures` | Covered (SKILL.md:63) |
| Generative-media architectures (diffusion, DiT, flow matching) | `yzmir-neural-architectures` | Covered (SKILL.md:64) |
| Architecture selection (CNN/Transformer/Mamba) | `yzmir-neural-architectures` | Covered (SKILL.md:65) |
| Production deployment / serving / observability | `yzmir-ml-production` | Covered (SKILL.md:66) |
| Dynamic / continual / MoE / adapter merging | `yzmir-dynamic-architectures` | Covered (SKILL.md:67) |
| Simulation math (ODE, integrators, replay, determinism) | `yzmir-simulation-foundations` | Covered (SKILL.md:68) |
| Systems thinking (causal loops, leverage points) | `yzmir-systems-thinking` | Covered (SKILL.md:69) |
| **RL-controlled neural-network morphogenesis** | **`yzmir-morphogenetic-rl`** | **MISSING** |

### Adjacent (non-Yzmir) packs called out by the router
SKILL.md:280-284 references `axiom-engineering-foundations`, `axiom-solution-architect`,
`ordis-security-architect`, and `axiom-python-engineering` as adjacent routers. These all exist
in the marketplace and the cross-pack guidance is accurate.

### Cross-claim verification
Spot-checked sibling routers to ensure the router's claims about each pack's coverage are real:

- `yzmir-llm-specialist` — claims of *reasoning models, agentic patterns, MCP, RAG, multimodal,
  prompt caching* are all reflected in `/home/john/skillpacks/plugins/yzmir-llm-specialist/skills/using-llm-specialist/SKILL.md`. Confirmed via grep (reasoning models, MCP, agentic patterns, RAG, multimodal, prompt caching all present).
- `yzmir-deep-rl` — claims of *MARL, offline RL, MDP, exploration* are all present in the sibling
  router (grep returned `multi-agent-rl`, `offline-rl`, MARL references).
- `yzmir-dynamic-architectures` plugin description names PEFT (LoRA, QLoRA, DoRA, VeRA, PiSSA,
  LoftQ, LoRA+, rsLoRA, LongLoRA) and Mixtral/MoE adapter merging — the router's "dynamic
  architectures" row covers continual learning and adapter merging but does **not** name the
  PEFT vocabulary the sibling pack actually owns (mid-2025/2026 PEFT methods). Users asking
  "should I use DoRA or QLoRA?" are not currently routed by keyword. Polish-tier finding.
- `yzmir-morphogenetic-rl` plugin description says: "Companion to yzmir-dynamic-architectures
  (which covers HOW the growable network trains)." The companion relationship is documented in
  the *sibling* pack but the *router* doesn't disambiguate the two.

### Research currency
AI/ML is an evolving domain (per `analyzing-pack-domain.md` Phase A criteria). The pack itself
acknowledges this at SKILL.md:286 ("Knowledge cutoff awareness... capability-tier framing in
downstream packs ... is intentional"). The router uses the capability-tier framing correctly and
references the 2026 reality (reasoning models, MCP, modern PEFT). Currency is largely OK except
for the morphogenetic-rl omission and the missing PEFT-method keywords noted above.

---

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Notes |
|---|-----------|--------|-------|
| 1 | Domain coverage vs. siblings | **Major** | 10 sibling packs exist; router lists 9. `yzmir-morphogenetic-rl` entirely missing. |
| 2 | Structural integrity (frontmatter, file layout) | **Pass** | Standard router layout. Frontmatter valid. Reference sheet co-located per convention. |
| 3 | Description / discoverability | **Minor** | SKILL.md description (line 3) is short and lacks "Use when…" pattern, but the wrapper provides explicit invocation; description still reads as a route-not-a-skill imperative which is appropriate for a router. |
| 4 | Component-type alignment | **Pass** | Router skill + reference sheet + slash wrapper is correct shape for this role. No misplaced commands/agents. |
| 5 | Cross-pack / catalog accuracy | **Major** | Plugin.json says "9 Yzmir packs", catalog at SKILL.md:266-278 enumerates 9 named packs; both are wrong by one (missing morphogenetic-rl). |
| 6 | Pressure-resistance content | **Pass** | Extensive rationalization-prevention tables (SKILL.md:110-207, wrapper:198-365), covering time/authority/sunk-cost/keyword/social/complexity pressures. This is the pack's strongest section. |
| 7 | Internal consistency (SKILL.md ↔ reference sheet ↔ wrapper) | **Major** | The slash-command wrapper at `.claude/commands/ai-engineering.md` is 647 lines and substantially **diverges from** the 286-line SKILL.md. Neither version of the catalog mentions `morphogenetic-rl`, but the two version differ in pack list framing (wrapper uses "yzmir/<pack>" paths, SKILL.md uses "yzmir-<pack>" names) and in tables. Drift between the two surfaces is a maintenance burden. |
| 8 | Slash-command wrapper presence | **Pass (with caveat)** | Wrapper exists at `/home/john/skillpacks/.claude/commands/ai-engineering.md`. Not a "thin wrapper" per the `reviewing-pack-structure.md` example but matches the broader repo pattern (compare `python-engineering.md` 377 lines, `deep-rl.md` 602 lines). No frontmatter — also consistent with repo convention. |

**Overall: Major.** The pack is structurally sound and the routing discipline is well-developed,
but it has fallen behind sibling-pack reality: a fully built sibling (`yzmir-morphogenetic-rl`)
exists, is registered in the marketplace, and is never referenced. The SKILL.md and the
slash-command wrapper have also drifted apart and need re-synchronization. These are
addressable without a rebuild.

---

## 4. Behavioral Tests

Per `testing-skill-quality.md`, the router is a **discipline-enforcing** skill (resists
shortcuts), and Pressure scenarios (Gauntlet A) take priority, followed by Edge cases (C) and
Real-world (B). Behavioral tests were not dispatched to a fresh subagent (this is a paper review),
so the following are **content audits** of whether the skill would *plausibly* hold up under each
scenario based on what is written, not live trials.

### Test 1 — Pressure A1: time + authority combo
*Scenario*: "URGENT! Production LLM is down, CUDA OOM errors, CEO waiting. PM says use llm-specialist pack."
- **Expected behavior**: 30-second clarification, route to pytorch-engineering first.
- **Coverage in skill**: `routing-examples.md:91-117` (Example 6) addresses this exact scenario
  with explicit script. The Authority Protocol (SKILL.md:127-134) and Time Pressure table
  (SKILL.md:112-125) reinforce it.
- **Predicted result**: **Pass.**

### Test 2 — Pressure A2: sunk-cost + frustration
*Scenario*: "Spent 6 hours in neural-architectures fixing training instability."
- **Expected behavior**: Empathetic but firm redirect to training-optimization.
- **Coverage**: `routing-examples.md:120-145` (Example 7) and SKILL.md:136-145 (Sunk Cost
  Protocol).
- **Predicted result**: **Pass.**

### Test 3 — Edge case C1: morphogenetic-rl query
*Scenario*: "I'm designing an RL controller that decides when to grow a neural network during
training. Where do I start?"
- **Expected behavior**: Route to `yzmir-morphogenetic-rl`.
- **Coverage**: **None.** The router has no mention of morphogenetic-rl. The closest hit is
  the "dynamic-architectures" row, which is about *how* a growable network trains, not about the
  *RL controller* that drives growth. The pack would likely mis-route to `yzmir-dynamic-architectures` or `yzmir-deep-rl` depending on which keyword the model anchors on. The user has to know the right pack name to find it via `/plugin`.
- **Predicted result**: **Fail.**

### Test 4 — Edge case C2: PEFT method choice
*Scenario*: "DoRA vs QLoRA for adapter fine-tuning a 70B model on a single A100?"
- **Expected behavior**: Route to `yzmir-dynamic-architectures` (which owns modern PEFT per
  its plugin.json), possibly with cross-ref to `yzmir-llm-specialist` for fine-tuning patterns.
- **Coverage**: The router's "dynamic-architectures" row mentions "adapter merging" but not
  the PEFT method vocabulary (LoRA/QLoRA/DoRA/PiSSA/LoftQ). The model might route to
  `yzmir-llm-specialist` based on "fine-tune" + "LLM", which is partially correct but misses
  the architecture-side specialist.
- **Predicted result**: **Partial pass** — would land in `llm-specialist`, which is plausible
  but not the strongest match for "which adapter variant".

### Test 5 — Real-world B1: ambiguous phrasing
*Scenario*: "My model isn't working."
- **Expected behavior**: Clarification question.
- **Coverage**: SKILL.md:43 ("Model not working" → "What's not working — architecture,
  training, or deployment?") in the mandatory-clarification table.
- **Predicted result**: **Pass.**

### Test 6 — Real-world B2: drifting wrapper
*Scenario*: User invokes `/ai-engineering` (loads the 647-line wrapper, not the 286-line
SKILL.md).
- **Expected behavior**: Same routing as via the skill.
- **Coverage**: The wrapper still mis-counts (omits morphogenetic-rl) and uses a slightly
  different routing decision-tree structure (Foundation Layer → Training → RL → LLM ...
  branched, vs. the SKILL.md's flat keyword table). The functional routing for canonical
  queries is equivalent.
- **Predicted result**: **Pass for canonical queries, Fail for morphogenetic-rl.**

### Summary of behavioral content audit

| Test | Predicted Result | Notes |
|------|------------------|-------|
| 1. Time + Authority | Pass | Strong, scripted resistance |
| 2. Sunk cost + Frustration | Pass | Strong, scripted resistance |
| 3. morphogenetic-rl query | **Fail** | Pack invisible to the router |
| 4. PEFT method choice | Partial | Vocabulary gap |
| 5. Ambiguous "model not working" | Pass | Explicit clarification trigger |
| 6. Slash-command path | Pass for canonical / Fail for morphogenetic | Drift compounds the coverage gap |

---

## 5. Findings

### 5.1 — Major: `yzmir-morphogenetic-rl` is invisible to the router

**Evidence**:
- `grep -c "morphogenetic-rl"` on SKILL.md, routing-examples.md, and the slash wrapper returns
  `0`, `0`, `0` respectively.
- `plugin.json:4` describes the pack as a router for "all 9 Yzmir AI/ML engineering
  skillpacks" — should be 10.
- `SKILL.md:268` says "The Yzmir faction ships **9 specialist packs** plus this router" and
  enumerates a numbered list (lines 270-278) that omits morphogenetic-rl.
- The sibling pack at `plugins/yzmir-morphogenetic-rl/` is registered in
  `marketplace.json:550-565` with category `ai-ml` and has been built out (its own
  `using-morphogenetic-rl/SKILL.md`).
- Its plugin description explicitly says: "Companion to yzmir-dynamic-architectures
  (which covers HOW the growable network trains)" — a fundamentally different abstraction
  level that the router needs to disambiguate.

**Impact**: A built, registered specialist pack is undiscoverable through its own
faction's router. Any user query about RL-controlled growth/morphogenesis/topology mutation
will mis-route.

**Fix required (Stage 5)**: Add a row to the SKILL.md "Routing by Problem Type" table
(SKILL.md:55-69) for "RL controller that decides when/how to mutate topology" → `morphogenetic-rl`.
Add a numbered catalog entry in the catalog section (after the current item 7, since
morphogenetic-rl is companion to dynamic-architectures). Add a corresponding row in the wrapper
and ideally a worked example in `routing-examples.md` that distinguishes the dynamic-architectures
question (HOW the network trains) from the morphogenetic-rl question (the RL controller
designing growth actions). Update `plugin.json:4` from "9 Yzmir AI/ML engineering skillpacks"
to "10".

### 5.2 — Major: SKILL.md and slash-command wrapper have drifted

**Evidence**:
- SKILL.md is 286 lines. The slash wrapper is 647 lines.
- The wrapper uses a *path-style* reference to siblings ("Route to:
  `yzmir/dynamic-architectures/using-dynamic-architectures`", e.g. line 132) — the SKILL.md uses
  bare names ("dynamic-architectures").
- The wrapper has a much longer "Pressure Resistance" section (~165 lines) than the SKILL.md
  (~95 lines), and a much longer "Comprehensive Rationalization Prevention Table" (~30 rows
  vs. ~13 in SKILL.md). Some material is wrapper-only.
- Both omit morphogenetic-rl in the same way — so they drifted in *style and detail*, not in
  the routing logic itself, but the surfaces are not in sync.

**Impact**: Two truths for "what does this router do" depending on whether the user invokes
via discovery (loads SKILL.md) or `/ai-engineering` (loads the wrapper). Future edits will need
to be made in both places or one will fall further behind. This is the marketplace's established
pattern (verified against `python-engineering.md`, `deep-rl.md` wrappers), but the *drift between
the pair* is real and worse here than in some others.

**Fix required**: Either (a) treat the wrapper as the canonical source and trim the SKILL.md
to be a thin "see the slash command" pointer, or (b) treat SKILL.md as canonical and reduce the
wrapper to a thin invocation pointer. Pick one. The marketplace convention is unclear since both
patterns coexist across repo wrappers. Recommend treating the SKILL.md as canonical and shortening
the wrapper, since the SKILL.md is what plugin consumers will see when reading the source.

### 5.3 — Minor: Missing modern PEFT vocabulary in routing keywords

**Evidence**: The "dynamic architectures" routing row (SKILL.md:67) lists "MoE routing, adapter
merging" but omits the PEFT method names (LoRA, QLoRA, DoRA, VeRA, PiSSA, LoftQ, LoRA+, rsLoRA,
LongLoRA) that the *sibling* pack's plugin.json explicitly owns. Users phrasing in PEFT method
names won't trigger a strong route.

**Impact**: Soft mis-routes for adapter-tuning questions, e.g. "DoRA vs LoRA for my 70B" goes
to `llm-specialist` (defensible) when the architecture/lifecycle pack might be more apt.

**Fix required**: Add adapter-method vocabulary to the dynamic-architectures keyword row.

### 5.4 — Minor: plugin.json description count is wrong (specific to 5.1)

**Evidence**: `plugin.json:4` — "all 9 Yzmir AI/ML engineering skillpacks". Should be 10.

**Fix required**: Bump the count once morphogenetic-rl is added to the router.

### 5.5 — Polish: SKILL.md description (frontmatter) does not start with "Use when…"

**Evidence**: SKILL.md:3 — "Route AI/ML tasks to correct Yzmir pack - frameworks, training, RL,
LLMs, architectures, production". Most repo SKILL.md descriptions start with "Use when…" per the
`reviewing-pack-structure.md` convention. This is a router, so the imperative is defensible
("Route…"), and the same shape appears in other yzmir routers — but it's worth aligning if a
broader sweep happens.

**Fix required**: Optional. If aligned: "Use when starting any AI/ML task and unsure which Yzmir
specialist pack applies - routes to PyTorch, training, RL, LLM, neural-architecture,
ML-production, dynamic-architecture, morphogenetic-RL, simulation-foundations, or
systems-thinking specialists."

### 5.6 — Polish: SKILL.md catalog section repeats content from the keyword table

**Evidence**: SKILL.md:55-69 (the "Routing by Problem Type" table) and SKILL.md:266-278 (the "AI
Engineering Plugin Router Catalog") cover overlapping material. The catalog is mostly redundant
with what the keyword table already says. Either tighten the catalog to be a pure name+link list
or remove it.

**Fix required**: Optional cleanup.

### 5.7 — Polish: `routing-examples.md` does not have a "morphogenetic-rl"-flavored example

**Evidence**: 15 examples in `routing-examples.md`, none of which exercise the
morphogenetic-rl/dynamic-architectures distinction. Example 14 covers continual learning →
dynamic-architectures but does not name the companion pack.

**Fix required**: Add an Example 16 (or fold into 14) that distinguishes the two.

---

## 6. Recommended Actions

Listed in priority order. Per the maintenance discipline, **this is a report-only review — no
edits were performed.** Stage 5 (implementing-fixes.md) is out of scope for this pass.

1. **(Major) Add `yzmir-morphogenetic-rl` to the router.** Add a routing row, add a catalog
   entry, add a worked example in `routing-examples.md` distinguishing it from
   `yzmir-dynamic-architectures`. Update plugin.json count from 9 to 10. Mirror the changes in
   `/home/john/skillpacks/.claude/commands/ai-engineering.md`.
2. **(Major) Reconcile SKILL.md and the slash-command wrapper.** Pick a canonical source. Reduce
   drift. Document the canonical-source decision in the pack root if useful.
3. **(Minor) Add PEFT-method vocabulary** (LoRA / QLoRA / DoRA / PiSSA / LoftQ / LongLoRA) to
   the dynamic-architectures keyword row.
4. **(Polish) Align description style** with "Use when…" if the broader repo sweep adopts the
   convention.
5. **(Polish) Trim or restructure the redundant catalog section** in SKILL.md (lines 266-278).

### Version-bump recommendation
The morphogenetic-rl gap + the wrapper-drift fix collectively are a **Minor bump (1.1.0 →
1.2.0)** per the `using-skillpack-maintenance` rubric — "Enhanced guidance, new components,
better examples". No structural break; new routing target is additive.

---

## 7. Reviewer Notes

- The pack's pressure-resistance section is unusually strong — easily the most thoroughly
  developed routing-discipline content I've seen in this marketplace's routers. That work
  should be preserved and considered canonical when reconciling the SKILL.md and wrapper.
- The pack is "router-only" (no commands at pack root, no agents). That is the correct shape for
  this role and matches sibling router-only packs (e.g., `using-llm-specialist`). No
  recommendation to add components — the existing surface area is appropriate.
- Slash-command wrappers in this repo are *not* the "thin wrapper" pattern shown as an example
  in `reviewing-pack-structure.md:208-227`. They are full content duplicates. This is
  marketplace-wide convention, not a defect of this pack — but it makes maintenance harder. A
  marketplace-level decision to standardize the pattern would help; until then, treat each
  wrapper as a second source-of-truth to keep in sync.
- The `meta-sme-protocol` review checklist (SME agent body conventions) is not applicable here:
  the pack ships no agents.
- The "tools:" key audit (per `analyzing-pack-domain.md` Phase C) is not applicable here: no
  agents to audit, and the skill omits `allowed-tools:` which is the dominant repo pattern.
- I did **not** dispatch live subagent behavioral tests — those should be run before any v1.2.0
  ship to confirm the morphogenetic-rl row actually triggers correctly under "I want an RL
  controller that decides when to grow a network" framings. The content audit in §4 is a
  prediction, not a test result.
- Cross-checked sibling-pack content claims; the router accurately reflects llm-specialist
  (reasoning models, MCP, agentic, RAG, multimodal, prompt caching all present in
  `using-llm-specialist/SKILL.md`) and deep-rl (multi-agent-rl, offline-rl present in
  `using-deep-rl/SKILL.md`). No false advertising.
- The router correctly notes the 2026 vocabulary trap ("agent" usually means LLM tool-loop, not
  RL — SKILL.md:153, routing-examples.md:238-262). This is high-value content; preserve.

---

**End of review.**
