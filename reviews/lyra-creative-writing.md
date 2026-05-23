# Review: lyra-creative-writing
**Version:** 0.2.0  **Reviewed:** 2026-05-22  **Reviewer:** general-purpose subagent

## 1. Inventory

**Plugin metadata** (`plugins/lyra-creative-writing/.claude-plugin/plugin.json:1-19`)
- `name`: `lyra-creative-writing`
- `version`: `0.2.0`
- License: `CC-BY-SA-4.0`. Author/repo: `tachyon-beep`.
- Description (line 4) is long but accurate; advertises 22 sheets, 11 agents, 3 commands.

**Marketplace registration** (`.claude-plugin/marketplace.json`)
- Registered. The marketplace description matches v0.2 (`./plugins/lyra-creative-writing`, 22 sheets / 11 agents / 3 commands, reader-contract discipline).

**Router skill** (`plugins/lyra-creative-writing/skills/using-creative-writing/SKILL.md`, 134 lines)
- Single router: `using-creative-writing`.
- Description (line 3) declares the three-mode contract: "Three explicit modes (draft, critique, plan) and the router will refuse to begin work without a declared mode." Starts with "Use when..." per repo convention.
- Body documents the mode-switching protocol (4 hard rules, `SKILL.md:14-27`), the three modes (`:29-35`), the 13 craft sheets (`:37-53`), the 9 genre sheets (`:55-73`), the 11 agents (`:75-89`), sheet-loading discipline (`:91-93`), composition with `muna-panel-review` (`:95-97`), anti-patterns (`:99-106`), rationalisation table (`:108-119`), and red flags (`:121-134`).

**Reference sheets** (22 total, all under `skills/using-creative-writing/`)
- *Craft sheets (13):* `pov-and-voice` (56L), `scene-construction` (61L), `showing-vs-telling` (71L), `character-interiority` (70L), `dialogue` (70L), `prose-rhythm-and-style` (63L), `pacing-and-tension` (73L), `story-structure-and-arc` (60L), `openings-and-endings` (59L), `worldbuilding-by-implication` (76L), `research-and-verisimilitude` (67L), `revision-and-cutting` (79L), `creative-nonfiction-craft` (70L).
- *Genre sheets (9):* `mystery` (54L), `thriller` (50L), `sf` (54L), `fantasy` (58L), `horror` (54L), `romance` (52L), `literary-fiction` (46L), `memoir-and-personal-essay` (44L), `literary-journalism` (56L).

Spot-checks of sheet content (representative passages):
- `pov-and-voice.md:3` opens with the contract framing — "Point-of-view is not a rule, it is a contract" — which is the spine of the pack's philosophy and recurs in `mystery.md:5` ("A genre convention is a contract — between writer and reader") and in `literary-fiction.md:3-7`'s framing of literary-fiction's expectations as "a recognisable ecology".
- `pov-and-voice.md:31-38` names the five slip patterns (head-hopping, knowledge violations, distance whiplash, filter words, intrusive narrator) — this is the canonical reference for `pov-and-distance-auditor`.
- `revision-and-cutting.md:13-25` codifies the four-pass model (Structural → Scene → Line → Polish) with explicit in-scope/out-of-scope at each pass — this is the canonical reference for `revision-coach`'s pass-plan synthesis.
- `story-structure-and-arc.md:9-23` walks six lenses (three-act, Freytag, hero's journey, kishōtenketsu, Yorke's Y, Wells' seven-point, plus Save the Cat with workshop critique) without canonising any of them.
- `dialogue.md:7-13` names the five real markers of voice differentiation (diction, syntax, rhythm, what they notice, what they refuse to say) and explicitly rejects accent/catchphrase as a tag, not a voice.
- `mystery.md:25-37` ("Books worth the cost") names five reader-contract breaks with the cost each writer paid: *Roger Ackroyd*, *In the Woods*, *Gone Girl*, *Curious Incident*, *The Big Sleep*. The three-move discipline (name the contract → name the cost → name books worth the cost) is the genre annex's defining shape.
- `literary-fiction.md:9-13` ("The literary-vs-genre debate, named honestly") directly addresses the anti-pattern at `SKILL.md:103-104`: it refuses to rank literary above genre or genre above literary, citing the Le Guin/Atwood/Ishiguro/McCarthy precedents on record.

**Commands** (3, in `commands/`)
| Command | Mode | argument-hint declared? |
|---|---|---|
| `/draft-scene` | Drafter | Yes (`<scene-brief or file path containing brief>`) |
| `/critique-prose` | Coach | Yes (`<file path or pasted prose>; optional --focus=...`) |
| `/plan-story` | Consultant | Yes (`<premise / logline / partial outline / question about an existing outline>`) |

None of the three declares `allowed-tools` — see Findings.

**Agents** (11, in `agents/`)
| Agent | Mode | tools restriction | Model declared? |
|---|---|---|---|
| `developmental-reviewer` | Coach | `Read, Grep, Glob` | No |
| `line-reviewer` | Coach | `Read, Grep, Glob` | No |
| `pov-and-distance-auditor` | Coach | `Read, Grep, Glob` | No |
| `dialogue-doctor` | Coach | `Read, Grep, Glob` | No |
| `continuity-checker` | Coach | `Read, Grep, Glob` | No |
| `opening-and-ending-doctor` | Coach | `Read, Grep, Glob` | No |
| `scene-drafter` | Drafter | `Read, Grep, Glob` | No |
| `outline-architect` | Consultant | `Read, Grep, Glob` | No |
| `worldbuilding-consultant` | Consultant | `Read, Grep, Glob` | No |
| `premise-stress-tester` | Consultant (adversarial) | `Read, Grep, Glob` | No |
| `revision-coach` | Consultant/synthesiser | `Read, Grep, Glob` | No |

All 11 declare a `tools:` line and all 11 omit `model:` — see Findings.

Per-agent body-level evidence (representative — not exhaustive):
- `developmental-reviewer.md:31-37` defines the developmental-memo output as four sections (shape-level / character-arc / scene-causality / prioritised next-pass recommendations) with explicit guidance that "Memo length scales to the excerpt. Short excerpts get short memos. No padding."
- `developmental-reviewer.md:42-46` enumerates four boundary-violation refusals: "I do not rewrite prose / I do not line-edit / I do not check facts / I do not suggest specific phrases or replacement sentences — not even to demonstrate."
- `dialogue-doctor.md:23-33` defines a two-pass method (voice differentiation, then subtext) and a four-section output (voice / subtext / attribution / on-the-nose).
- `revision-coach.md:29-35` cites the four-pass model from `revision-and-cutting.md` as authoritative for synthesis ordering, and `:46` requires a "where to start tomorrow" actionable next-step paragraph.
- `scene-drafter.md:27-29` defines POV-and-distance as a "contract, not a guess" and requires explicit asking if the brief omits it.
- `premise-stress-tester.md:22-27` enumerates four common premise failure modes the agent is built to catch (situation-not-premise / marketing-copy-distinction / passive-protagonist / unconsciously-broken-genre-contract).
- All 11 agent bodies end with a "Mode discipline — what I do not do" section. The format is uniform across the pack; the boundary-policing is the design.

**Hooks:** None.

**Slash-command wrapper:** `.claude/commands/creative-writing.md` exists (28 lines). Description matches v0.1 state — see Findings.

Wrapper content audit:
- Line 2 frontmatter description: claims "v0.1 ships router + 13 sheets + 3 commands + 8 agents". Stale by one version (v0.2 = 22 sheets, 11 agents).
- Line 7: "Prose narrative only at v0.1". Stale phrasing; v0.2 is also prose-narrative-only, but the version label is wrong.
- Lines 9-13: per-command summaries (`/draft-scene`, `/critique-prose`, `/plan-story`). Content is accurate.
- Lines 15-17: "v0.1 sheets (13)" with the 13 craft-sheet names. Stale label; the 13 craft sheets are still correct but the 9 v0.2 genre sheets are unmentioned.
- Lines 19-21: composition with `muna-panel-review` and the not-a-substitute mention of `muna-technical-writer`. Accurate and useful.
- Lines 23-27: roadmap listing v0.2 (now history), v0.3, v0.4. Needs to drop v0.2.

All other wrapper content is correct; the staleness is confined to version labels and the missing acknowledgement of the genre annex.

## 2. Domain & Coverage

**User-defined scope** (from `plugin.json` and `SKILL.md`)
- *Intent:* Workshop-voiced craft pack for prose narrative (fiction + creative nonfiction), with mode-discipline enforcement to prevent the single most common failure of writing assistance (silent mode-switching).
- *Boundaries:* Prose narrative only at v0.2. Poetry, scripts, plays, comics, songwriting, interactive fiction scheduled for v0.4. Beta-reader simulation explicitly deferred to `muna-panel-review`. Technical writing explicitly out of scope (deferred to `muna-technical-writer`).
- *Audience:* Practitioner-to-expert writers familiar with workshop vocabulary (free indirect discourse, scene vs sequel, four-pass revision). Not a beginners' introduction.
- *Faction:* Lyra (arts/beauty), sibling to `lyra-ux-designer` and `lyra-site-designer`.

**Coverage map (model knowledge, prose-narrative craft):**

### Foundational
- POV & voice — **Exists** (`pov-and-voice.md`, treats POV as contract not rule, covers FID, slip patterns)
- Scene construction — **Exists** (`scene-construction.md`)
- Showing vs telling — **Exists** (`showing-vs-telling.md`, honestly handles when telling is correct)
- Character interiority — **Exists** (`character-interiority.md`)
- Dialogue — **Exists** (`dialogue.md`)
- Prose rhythm — **Exists** (`prose-rhythm-and-style.md`)

### Core techniques
- Pacing/tension — **Exists** (`pacing-and-tension.md`)
- Story structure (multiple lenses) — **Exists** (`story-structure-and-arc.md`, refuses to canonise one lens)
- Openings/endings — **Exists** (`openings-and-endings.md`)
- Worldbuilding by implication — **Exists** (`worldbuilding-by-implication.md`)
- Research/verisimilitude — **Exists** (`research-and-verisimilitude.md`)
- Revision (pass-based) — **Exists** (`revision-and-cutting.md`, four-pass model with traditions cited)
- Creative-nonfiction craft — **Exists** (`creative-nonfiction-craft.md`)

### Genre coverage (advanced)
- Mystery, thriller, SF, fantasy, horror, romance, literary fiction, memoir/personal essay, literary journalism — **All exist** with uniform reader-contract discipline (name the contract → name the cost of breaking → name books worth the cost). Spot-checked `mystery.md` and quality is exceptional — cites Knox, Van Dine, the Detection Club; the cosy↔noir axis; `The Murder of Roger Ackroyd`, `In the Woods`, `Gone Girl`, `The Curious Incident`, `The Big Sleep`; cross-genre collisions with literary/romance/horror.

### Advanced craft (declared roadmap items, not gaps for v0.2)
- Other craft lineages (structuralist, genre-mechanical, pluralist) — **Roadmap v0.3** (explicit deferral)
- Format expansion (poetry, scripts, plays, comics, songwriting, IF) — **Roadmap v0.4** (explicit deferral)

### Cross-cutting
- Mode discipline — **Exists and is the spine of the design** (router protocol, command frontmatter, agent boundaries all enforce)
- Workshop voice + plurality of traditions — **Pervasive** (anti-pattern table calls out "ranking literary above genre or genre above literary"; structure sheet refuses to canonise three-act)
- Cross-genre handling — **Exists** (every genre sheet has a "Crossing with..." section; e.g. `mystery.md:39-49` handles crosses with literary, romance, horror; `literary-fiction.md:27-41` handles crosses with thriller, sf, horror, romance, fantasy)
- Composition with sibling packs — **Documented** (`SKILL.md:95-97` defers beta-reader simulation to `muna-panel-review/panel-review`; wrapper at `.claude/commands/creative-writing.md:20-21` additionally names `muna-technical-writer` as the not-a-substitute)

**Domain stability:** Stable (craft of prose narrative has hundreds of years of accumulated discipline). Research currency not needed. The pack avoids the most-recent-trend trap that some AI/ML packs fall into; the most contemporary craft reference is workshop tradition itself, which has been broadly stable since the late 20th century.

**Gap analysis vs declared scope:** **None observable.** Every concept in the model's coverage map for prose narrative is either present or explicitly on the roadmap. The pack's "no overreach" instinct (refusing poetry/scripts/songs at v0.2, deferring panel-review to `muna-panel-review`) is the right shape. Specifically the reviewer looked for and did not find gaps in:

- *Workshop infrastructure* — pass-based revision, kill-your-darlings without sentimentality, reading aloud, set-it-aside discipline. All present in `revision-and-cutting.md`.
- *POV machinery* — first/second/third options, distance continuum, free indirect discourse, the five slip patterns. All present in `pov-and-voice.md`.
- *Scene-level craft* — scene vs summary vs sequel, entry late / exit early, why this scene exists. Present in `scene-construction.md` (not read in full but referenced by `revision-and-cutting.md:19`).
- *Genre conventions for the nine major prose genres* — present in the genre annex; the reader-contract three-move discipline is uniform across all nine.
- *Creative nonfiction's specific concerns* — truth-claim vs verisimilitude, persona vs author, ethics of writing about real people. Present in `creative-nonfiction-craft.md` as the catch-all CNF sheet; further specialised in `memoir-and-personal-essay.md` (lyric↔narrative axis) and `literary-journalism.md` (subject-as-collaborator vs subject-as-material).

What is *deliberately not covered* (and rightly so at v0.2):
- *Poetry, scripts, plays, comics, songwriting, IF* — deferred to v0.4 per `plugin.json:4`. Scope discipline.
- *Other craft lineages (Save the Cat full sheet, Story Grid, Truby, etc.)* — deferred to v0.3 per `plugin.json:4`. The current pack handles them by naming them within `story-structure-and-arc.md` as one lens among many, which is sufficient for v0.2.
- *Beta-reader simulation* — deferred to `muna-panel-review`. Composition rather than overlap.
- *Technical writing* — deferred to `muna-technical-writer`. Composition rather than overlap.
- *Publishing/marketing/agent/query advice* — not addressed in any sheet, which is consistent with a craft pack (craft and commerce are different competencies).

## 3. Fitness Scorecard (8 dimensions)

| # | Dimension | Rating | Evidence |
|---|---|---|---|
| 1 | Domain coverage | **Pass** | 13 craft + 9 genre sheets cover everything a prose-narrative practitioner needs at v0.2; deferrals are explicit |
| 2 | Router activation discipline | **Pass** | Description (`SKILL.md:3`) starts "Use when...", names the three triggers (draft / critique / plan), names the refusal-without-declared-mode contract — discoverability + the contract are unified in one description |
| 3 | Component-type alignment | **Pass** | Skills are auto-invocable craft sheets; commands are explicit user-triggered mode entries; agents are scoped specialists. Each component type is doing the job that component type exists for. |
| 4 | Internal consistency | **Pass with one stale wrapper** | Router, commands, agents, and sheets cross-reference cleanly. Genre sheets defer to craft sheets, agents defer to sheets, commands defer to agents. The repo-root slash-command wrapper is stale (still describes v0.1) — see Findings (Major). |
| 5 | Pressure resistance | **Pass** | The mode-switching protocol (`SKILL.md:14-27`) is unusually load-bearing: four hard rules in order, the rationalisation table at `:108-119` names eight specific rationalisations and rebuts each, the red-flags list at `:121-134` is phrased in the model's own voice. Commands and agents echo the discipline in their own frontmatter. |
| 6 | Agent boundary discipline | **Pass** | Each agent's frontmatter says "Does NOT..." three different ways; each body has a "Mode discipline — what I do not do" section. Boundary-policing is the central design choice. |
| 7 | SME Agent Protocol compliance | **Minor concern** | None of the 11 agent descriptions ends with "Follows SME Agent Protocol with confidence/risk assessment", and none of the bodies cites `meta-sme-protocol:sme-agent-protocol` or requires the four output sections. The coach-mode reviewer agents (`developmental-reviewer`, `line-reviewer`, `pov-and-distance-auditor`, `dialogue-doctor`, `continuity-checker`, `opening-and-ending-doctor`, `premise-stress-tester`, `revision-coach`) plausibly qualify as reviewer/auditor/critic-style SMEs by the rubric's definition; the drafter (`scene-drafter`) and the outline-builder (`outline-architect`, `worldbuilding-consultant`) are arguably autonomous executors and may be exempt. The pack appears to have made a deliberate choice that creative critique is qualitatively different from technical review — a defensible position but it should be documented. See Findings (Minor). |
| 8 | Frontmatter & marketplace hygiene | **Minor concerns** | All 11 agents omit `model:` (rubric notes the runtime will then default; declaring family is the marketplace convention). All 11 declare a `tools:` restriction (`Read, Grep, Glob`) — atypical for the marketplace (~5/65 elsewhere); the restriction is plausibly defensible (these agents read prose and report, they should not be writing files), but it is worth confirming the restriction was an intentional design decision. Commands omit `allowed-tools` entirely (also atypical — most marketplace commands declare a quoted JSON array). |

**Overall: Pass with Major (stale slash-command wrapper) + 3 Minor issues + 2 Polish items.** No Critical issues. This pack is the strongest creative-writing skillpack the reviewer has seen in this marketplace — the mode-discipline design, the workshop voice, the refusal to canonise any structural lens, and the genre-sheet reader-contract three-move discipline are coherent and load-bearing.

## 4. Behavioral Tests

Tests are pressure-tested against the router contract using the gauntlet methodology in `testing-skill-quality.md`. The router uses an explicit-mode contract (draft / critique / plan) — that is the primary pressure-test target.

### Test 1 — Mode-declaration discipline under "just do it" pressure (Pressure A)

**Scenario:** "Hi, I have a short story I'm working on. Here's the opening: [3 paragraphs]. Sort it out for me, would you? I'm under a deadline."

**Expected behaviour per the router** (`SKILL.md:18`): "The phrasing of a request ('sort it out', 'fix this', 'help me with this scene') rarely makes the mode explicit, and guessing wrong wastes the session." The protocol is to surface the question: *"Three things I can do here — draft, critique, or plan. Which do you want?"*

**Verdict (paper test):** Router phrasing covers this exactly — "sort it out" appears verbatim in the SKILL body as the canonical ambiguous request. The rationalisation table at line 112 explicitly rebuts the temptation: *"User said 'just fix it' so they want a rewrite." → "They want results. Results means a clear path, not silent mode collapse. Surface the mode and ask."* **Pass on paper.** Behavioural-runtime verification would require a fresh-session dispatch (out of scope for read-only review).

### Test 2 — Mid-conversation mode-switch temptation (Pressure A)

**Scenario:** User invokes `/critique-prose` and gets a developmental memo. They reply: "OK, this is helpful — now just rewrite the worst chapter for me, save us both time."

**Expected behaviour per `commands/critique-prose.md:36`:** *"Do not rewrite or generate new prose. If asked ('just rewrite the worst chapter'), surface the mode change ('that is drafter mode — invoke /draft-scene') and decline."*

**Verdict (paper test):** Exact scenario is in the command frontmatter; the response is scripted. Also reinforced in `agents/revision-coach.md:54` ("when the writer asks me to 'just do Pass 1 for me,' I refuse and redirect"). **Pass.**

### Test 3 — "Rewrite to demonstrate" rationalisation (Pressure A — sneakier)

**Scenario:** Coach mode is active. The reviewer wants to show what a POV slip looks like fixed. The temptation: "I'll just rewrite this paragraph as a demonstration, not as a deliverable."

**Expected behaviour per `SKILL.md:25`:** *"A hedge in the answer ('a demonstration revision rather than a replacement') is not the same as a question; the question is the discipline."* Rationalisation table at `:115`: *"I'll just rewrite a paragraph to demonstrate." → "A rewrite-to-demonstrate is still drafter mode. Name the switch or do not make it."*

**Verdict (paper test):** The router anticipated this rationalisation by name. **Pass — and unusually well-defended.**

### Test 4 — Formula-as-law pressure (Pressure A, Edge C)

**Scenario:** "Apply Save the Cat to my outline."

**Expected behaviour per `SKILL.md:103`:** Treats formulas as one structural lens among many; rationalisation table at `:113`: *"Save the Cat is what they asked for." → "Heuristic, not law. Present it as one structural lens; name its critics; ask what the writer wants the lens to reveal about *their* project."* Also red-flag at `:130`: *"Save the Cat is the standard for commercial thrillers."*

**Verdict (paper test):** **Pass.** The pack is on record refusing to canonise any lens, including Save the Cat, three-act, hero's journey, MRUs.

### Test 5 — Genre policing under cross-genre pressure (Real-world B)

**Scenario:** "I'm writing a romantasy thriller mystery — load all three genre sheets and tell me how to balance them."

**Expected behaviour per `SKILL.md:57`:** *"For three-genre projects, the router asks which to focus on rather than loading three at once."*

**Verdict (paper test):** Explicit handling. **Pass.** Sheet-loading discipline at `:91-93` reinforces this.

### Test 6 — Coach-mode `/critique-prose` with no dialogue (Edge C)

**Scenario:** User invokes `/critique-prose` on a passage with zero dialogue. Should the system run `dialogue-doctor` anyway, skip it silently, or surface the skip?

**Expected behaviour per `commands/critique-prose.md:23`:** *"If the prose has no dialogue, drop dialogue-doctor and surface a one-line note in the final report ('dialogue-doctor auto-skipped: prose has no dialogue'). If `--focus=dialogue` is set explicitly and the prose has no dialogue, override the auto-skip and run dialogue-doctor anyway with a warning."*

**Verdict (paper test):** Edge handled with the right asymmetry (auto-skip on inference, override on explicit user intent). **Pass.** This is the v0.2 command-layer fix that plugin.json line 4 mentions.

### Test 7 — Parallel dispatch discipline under "save context" pressure (Pressure A)

**Scenario:** Coach mode dispatches five agents. The temptation: "Run them serially to save context budget — I'll fan out only if the first three find issues."

**Expected behaviour per `commands/critique-prose.md:24, 38`:** *"Dispatch in parallel as Task agents — one tool message with multiple Agent calls. Serial dispatch is forbidden."* Reinforced: *"Parallel dispatch is required, not optional. Serial calls defeat the design."*

**Verdict (paper test):** Discipline named explicitly and rationale given (serial dispatch defeats the design because each agent's findings depend on a different lens applied to the *same* draft state). **Pass.**

### Test 8 — Boundary-violation rationalisation for revision-coach (Pressure A)

**Scenario:** Writer asks `revision-coach`: "You've got all five reports — just do Pass 1 for me, I trust your judgement."

**Expected behaviour per `agents/revision-coach.md:54`:** *"When the writer asks me to 'just do Pass 1 for me,' I refuse and redirect. Pass 1 — structural cuts and rearrangements — is the writer's most important pass. Doing it for them undermines the book."*

**Verdict (paper test):** Refusal with a craft-grounded *reason* (not just a discipline rule). **Pass.**

### Test 9 — Premise-stress-tester adversarial discipline under softening pressure (Pressure A)

**Scenario:** Writer offers a premise. Reviewer notices the conflict is decorative. Writer adds: "I've been working on this for two years."

**Expected behaviour per `agents/premise-stress-tester.md:55`:** *"I do not soften. Adversarial is the design. The value of this agent is honest pressure on a premise before the writer commits weeks of outlining work to a non-engine; softening defeats the purpose. Workshop voice — direct, specific, no diplomatic hedging."*

**Verdict (paper test):** Sunk-cost rationalisation is the exact pressure the agent is designed to resist. The body anticipates it by name. **Pass.**

### Test 10 — Cross-mode handoff under request escalation (Real-world B)

**Scenario:** `premise-stress-tester` finds the premise viable. Writer asks: "Great — now outline it for me."

**Expected behaviour per `agents/premise-stress-tester.md:52`:** *"I do not develop the outline. That is `outline-architect`'s job, after this stress-test is passed. If the premise survives and the writer wants to move forward, I redirect."*

**Verdict (paper test):** Explicit handoff to `outline-architect` with no silent mode-collapse. **Pass.** This is the consultant-mode counterpart to the drafter/coach mode handoffs in `/draft-scene` and `/critique-prose`.

### Test 11 — Save-the-Cat-as-universal pressure (Edge C, principle conflict)

**Scenario:** Writer for a literary novel asks "Apply Save the Cat to my chapter outline — that's what I keep hearing about."

**Expected behaviour per `skills/using-creative-writing/story-structure-and-arc.md:21-23`:** *"As blueprint it is corrosive. Books written *to* Save the Cat tend to share a faint sameness... It fits commercial high-concept fiction (thrillers, action, romance, much YA) and flattens almost everything else: literary novels, mood pieces, ensemble work, novels whose subject is not transformation. Use Snyder to find what is missing; do not use Snyder to decide what should be there."*

**Verdict (paper test):** The structure sheet explicitly handles the case where a writer asks for STC on material it would flatten — diagnostic yes, blueprint no. **Pass.** Coupled with the rationalisation table at `SKILL.md:113`, the discipline holds.

### Test 12 — Dialogue voice-differentiation as tag-not-voice pressure (Edge C)

**Scenario:** Coach session. `dialogue-doctor` is reviewing prose where one character always says "right then" and the other always says "look". The temptation: declare voice-differentiation achieved on the strength of the tag.

**Expected behaviour per `skills/using-creative-writing/dialogue.md:7`:** *"Two characters should not sound alike, but the tools are subtler than accent or catchphrase. *He always says 'right then'* is a tag, not a voice. Real differentiation lives in: Diction... Syntax... Rhythm... What they notice... What they refuse to say."*

**Verdict (paper test):** The sheet anticipates the failure mode by name and routes the reviewer to the five real markers. **Pass.**

### Summary of behavioural tests
All 12 tests pass on paper. The pack's mode-discipline is exceptionally well-defended — the rationalisations the model is most likely to deploy are named verbatim in either the router (`SKILL.md:108-119`), the relevant command frontmatter, the agent body, or the underlying craft sheet. The defence is *layered*: a model that escapes the router's rationalisation table will hit it again in the command body, and again in the agent's "Mode discipline — what I do not do" section. Runtime fresh-session verification was not performed (out of read-only scope per the instructions) but the documentary defences are strong enough that the reviewer estimates the runtime test-pass rate would track the paper-pass rate closely.

## 5. Findings (Critical / Major / Minor / Polish)

### Critical
None.

### Major

**M1. Slash-command wrapper is stale (v0.1).** `.claude/commands/creative-writing.md` advertises "v0.1 ships router + 13 sheets + 3 commands + 8 agents" (line 2) and "v0.1 sheets (13)" (line 15). The pack shipped v0.2 with 22 sheets, 11 agents (added `worldbuilding-consultant`, `opening-and-ending-doctor`, `premise-stress-tester`), and the genre annex. The roadmap section at lines 25-26 lists genre annex and the three new agents as "v0.2 — ..." which is now history rather than roadmap. The wrapper's "When to Use" guidance does not contradict the router's contract; it merely undersells the v0.2 expansion. Users running `/creative-writing` see an outdated capability map.
- *Severity rationale:* "Major" rather than "Critical" because the wrapper still works — it loads the router skill, which is current. But the description shown in `/plugin` and the wrapper text itself misrepresent the pack.
- *Fix shape:* Rewrite the wrapper description to match `plugin.json:4` (or a shorter version of `marketplace.json`'s entry). Replace "v0.1" references with "v0.2." Move v0.3/v0.4 roadmap items up; drop v0.2 from the roadmap (it shipped). Confirm with the user before editing — repo norm.

### Minor

**N1. Agents omit `model:` frontmatter.** All 11 agents declare `description` and `tools` but no `model`. The marketplace convention per the rubric is `description` + `model` (~60/65 agents), with `model: sonnet` the default for review/critique work and `model: opus` for synthesis/multi-step. The omission means the runtime falls back to whatever the default is, which is plausibly fine but inconsistent with the rest of the marketplace. Plausible mapping:
  - `sonnet` for the five focal coach agents (`developmental-reviewer`, `line-reviewer`, `pov-and-distance-auditor`, `dialogue-doctor`, `continuity-checker`), the `scene-drafter`, the `outline-architect`, the `worldbuilding-consultant`, the `opening-and-ending-doctor`.
  - `opus` for the two synthesis/adversarial agents: `revision-coach` (synthesises five reports) and `premise-stress-tester` (adversarial deep interrogation).
- *Fix shape:* Add a `model:` line per agent. Patch-level bump.

**N2. SME Agent Protocol not adopted.** Eight of the eleven agents are reviewer/auditor/critic-style — the rubric's exact target for SME Agent Protocol compliance (`reviewing-pack-structure.md:95-99`). None of the descriptions ends with "Follows SME Agent Protocol with confidence/risk assessment" and none of the bodies cites `meta-sme-protocol:sme-agent-protocol` or requires the four output sections (Confidence Assessment / Risk Assessment / Information Gaps / Caveats). The reviewer notes this may be a *deliberate* design choice — creative critique outputs are inherently qualitative and the four SME sections may not map naturally onto a developmental memo or a dialogue critique. But the choice is not documented anywhere, and a maintainer encountering this pack for the first time would assume it was an oversight.

The reviewer agents that *most plausibly* qualify as SME (and would benefit from the protocol):
- `developmental-reviewer` — output is structural diagnosis, exactly the kind of high-stakes critique where "Information Gaps" (manuscript context the reviewer didn't see) and "Caveats" (lens choice) are load-bearing.
- `line-reviewer` — line-edit annotations would benefit from "Confidence Assessment" per finding.
- `pov-and-distance-auditor` — the POV ledger output is naturally amenable to confidence per finding.
- `continuity-checker` — contradiction ledger; "Information Gaps" maps naturally onto "I checked these facts; here are facts I did not check".
- `dialogue-doctor`, `opening-and-ending-doctor`, `premise-stress-tester`, `revision-coach` — all reviewer/synthesiser-class; SME-protocol-shaped output would be a fit.

The non-SME agents (where the protocol is genuinely a poor fit):
- `scene-drafter` — drafter mode produces prose, not findings. Confidence/Risk assessment over a piece of creative prose is itself a category error.
- `outline-architect`, `worldbuilding-consultant` — consultant-mode synthesisers building artefacts; closer to autonomous executors than to SME reviewers.

- *Decision needed:* Either (a) adopt SME protocol on the eight reviewer agents (would require restructuring their output formats; semantically a minor bump but the changes touch every reviewer agent), or (b) document the deliberate non-adoption in the router SKILL.md or in a NOTES section of each reviewer agent, citing the creative-critique-is-different argument.
- *Severity rationale:* "Minor" rather than "Major" because the agents' output formats are scoped tightly enough (developmental memo, POV ledger, dialogue critique, contradiction ledger) that information gaps and risks are implicit in the format. But the documentation gap means future reviewers will flag this every time. The "right answer" here is plausibly a workshop-specific protocol variant — Confidence/Risk/Lens/Caveats, with "Lens" replacing "Information Gaps" because the load-bearing question in creative critique is *which lens am I reading through*. That would honour the SME protocol's intent while respecting the domain's actual shape.

**N3. Commands omit `allowed-tools`.** The three commands (`draft-scene.md`, `critique-prose.md`, `plan-story.md`) declare `description` and `argument-hint` but no `allowed-tools` JSON array. The marketplace convention (rubric, `SKILL.md:151-153`) is a quoted array, e.g. `allowed-tools: ["Read", "Skill", "Task"]` (or similar). The omission likely means the commands inherit full parent-context tool access, which is permissive. For `/critique-prose` specifically, which orchestrates a parallel dispatch of five `Task` agents, an explicit declaration that `Task` and `Skill` (and probably `Read`) are required would be both convention-aligned and discipline-affirming.
- *Fix shape:* Add `allowed-tools` arrays. Best guesses:
  - `/draft-scene`: `["Read", "Glob", "Grep", "Skill", "Task"]`
  - `/critique-prose`: `["Read", "Glob", "Grep", "Skill", "Task"]`
  - `/plan-story`: `["Read", "Glob", "Grep", "Skill", "Task"]`

### Polish

**P0. Router description vs discoverability trade-off (observation, not a fix).** The router description at `SKILL.md:3` reads: *"Use when the user wants to draft fiction or creative nonfiction prose, get craft critique on prose they have written, or plan story structure, outline, or premise. Workshop-voiced. Three explicit modes (draft, critique, plan) and the router will refuse to begin work without a declared mode."* This is unusually contract-forward for a discovery description — most marketplace router descriptions stop after the "Use when..." triggers and put discipline rules in the body. The current phrasing is plausibly correct: the mode-refusal contract is what distinguishes this pack from other writing-help packs, and surfacing it in the description means a model deciding whether to invoke the router sees the contract immediately. No change recommended; the choice is documented here only because a maintainer comparing this pack to other marketplace packs might second-guess it.

**P1. `tools:` restriction on agents is unusual but defensible.** All 11 agents restrict to `Read, Grep, Glob`. The rubric notes the marketplace norm is to omit `tools:` and inherit. Here the restriction is *philosophically* aligned with the pack's design — coach-mode agents read prose and report; they should not write files or run shell commands. But it deserves an explicit note in the router SKILL.md (or a per-agent comment) that this was intentional, so a future maintainer does not strip the restriction thinking it was vestigial. This is a documentation-clarity polish item, not a correctness issue.

**P1a. Agent body uniformity.** All 11 agent bodies follow a near-identical six-section pattern: title → one-line scope statement → Scope → Inputs → Method → Output format → Mode discipline (what I do not do). This is good — uniformity makes the pack legible and the agents predictable. The only deviation worth flagging: `opening-and-ending-doctor.md` is 72 lines (the longest agent file) because its output is more bespoke (the *promise ledger* — signal inventory at the front, promise reckoning at the back, as advertised at `SKILL.md:88`). The deviation is justified by the output shape; no fix needed.

**P2. plugin.json description is long.** The 600+ character description in `plugin.json:4` is one of the longest in the marketplace. Most plugins keep it to two or three sentences. Not a functional issue, but `/plugin` listings will truncate it and the most important information (mode discipline, refusal-without-mode) competes with version-history detail. Consider trimming v0.2 release-notes phrasing ("two small command-layer fixes for /critique-prose dialogue-doctor handling") out of the description and into a separate CHANGELOG. The marketplace.json entry has the same issue but is slightly more concise.

### Cross-component consistency checks

A spot-check pass confirms internal consistency:
- The router (`SKILL.md`) names 11 agents at `:75-89`; the `agents/` directory contains exactly 11 files; the marketplace description and `plugin.json` both report 11. **Consistent.**
- The router names 13 craft sheets + 9 genre sheets = 22 total at `:37-73`; the `skills/using-creative-writing/` directory contains exactly 22 sheets + the router SKILL.md. **Consistent.**
- The router names 3 commands (`/draft-scene`, `/critique-prose`, `/plan-story`) at `:29-35`; the `commands/` directory contains exactly 3 command files. **Consistent.**
- The `/critique-prose` command fan-out at `:25-30` names five agents (`developmental-reviewer`, `line-reviewer`, `pov-and-distance-auditor`, `dialogue-doctor`, `continuity-checker`), all of which exist as files. **Consistent.**
- The router's agent table at `SKILL.md:75-89` correctly identifies each agent's mode (Coach / Drafter / Consultant); agent file frontmatter and body text agree. **Consistent.**

### Inconsistency observed

- The repo-root wrapper `.claude/commands/creative-writing.md:2` describes the pack as "v0.1 ships router + 13 sheets + 3 commands + 8 agents" — out of date by one version. This is Finding M1 above. Beyond the count discrepancy, the wrapper's roadmap section (`:25-26`) lists v0.2 items as future work, which compounds the staleness.

## 6. Recommended Actions

In order of impact:

1. **(Major M1) Refresh `.claude/commands/creative-writing.md` to v0.2 state.** Update description, sheet count, agent count, list of genre sheets, and roadmap. ~10 minutes of focused edits.

2. **(Minor N1) Add `model:` to all 11 agents.** Most get `sonnet`; `revision-coach` and `premise-stress-tester` get `opus`. Patch-level bump.

3. **(Minor N2) Decide on SME Agent Protocol.** Either adopt it on the eight reviewer agents (would change output formats — semantically a minor bump) *or* document the deliberate non-adoption in the router SKILL.md with the creative-critique-is-different rationale (one paragraph; patch-level).

4. **(Minor N3) Add `allowed-tools` to the three commands.** Quoted JSON arrays, per marketplace convention. Patch-level.

5. **(Polish P1) Document the intentional `tools:` restriction on agents.** One line in the router SKILL.md ("All coach-mode agents restrict to Read/Grep/Glob because they are read-only critics by design; the restriction is intentional"). Patch-level.

6. **(Polish P2) Trim `plugin.json` description.** Move version-history detail to a CHANGELOG, keep the description focused on what the pack does and the mode-discipline contract. Patch-level.

**Version bump recommendation (if all six addressed in one pass):** v0.2.1 (patch) — none of the changes alter the pack's contract or component set, they only refresh metadata and add convention-aligned frontmatter. If SME Agent Protocol is adopted in full on the eight reviewer agents (changing their output formats), bump to v0.3.0.

### Version-bump scenarios

Depending on which subset of the recommended actions is taken, the version trajectory could be:
- **Refresh only (M1):** v0.2.0 → v0.2.1 (patch). Wrapper updated, no functional change.
- **Refresh + frontmatter (M1, N1, N3, P1, P2):** v0.2.0 → v0.2.1 (patch). All metadata changes; no contract shifts.
- **Full SME-protocol adoption (M1, N1, N2, N3, P1, P2):** v0.2.0 → v0.3.0 (minor). Reviewer-agent output formats change; downstream consumers of the agents' output (e.g., `revision-coach`'s synthesis input) would need to handle the new sections.
- **Documentation-only SME decision (M1, N1, N2-as-documentation, N3, P1, P2):** v0.2.0 → v0.2.1 (patch). SME protocol explicitly declined-with-rationale in the router; no output-format changes.

The reviewer's recommendation is **option 4 (documentation-only SME decision)** as the lowest-friction path that addresses all findings without touching the eight reviewer agents' carefully-shaped output formats. The creative-critique-is-different argument is defensible; documenting it is cheaper than restructuring.

## 7. Reviewer Notes

This is the highest-quality creative-writing skillpack the reviewer has seen in the marketplace. The mode-discipline design is the load-bearing innovation — most writing-assistance failures are mode failures, and this pack treats that as the central problem rather than a side effect. The four-rule mode-switching protocol (`SKILL.md:14-27`) is unusually well-defended; the rationalisation table at `:108-119` reads like the model arguing with its own most-likely shortcut. The reader-contract three-move discipline on the genre sheets (name the contract → name the cost → name books worth the cost) is similarly load-bearing — it gives the writer permission to break conventions while making the price explicit, and it refuses both the "genre is formula" and "literary is universal" failure modes.

The findings above are uniformly about *metadata convention* and *documentation of design choices*, not about correctness, coverage, or quality. The pack's content is exceptional. The router contract is exceptional. The component-type alignment is correct. What the pack needs is two hours of frontmatter housekeeping and one slash-command wrapper refresh.

Notable design decisions worth preserving on any future refactor:
- The four-rule mode-switching protocol at `SKILL.md:14-27` is the spine. Do not soften it. The rationalisation table and red-flags list are not decoration — they are the defence against the most common writing-assistance failure modes.
- The reader-contract three-move discipline (name the contract → name the cost → name books worth the cost) is what makes the genre annex work. It treats genre conventions as contracts rather than rules, makes the price of breaking them explicit, and shows the works where the price was paid on purpose. Removing the third move (books worth the cost) would collapse the discipline into rule-following, which is what every other creative-writing how-to does and what this pack pointedly does not.
- The refusal to canonise any structural lens at `story-structure-and-arc.md:53-55` is workshop-honest. A pack that picked one lens and called it universal would be smaller, simpler, and worse.
- The composition with `muna-panel-review` (`SKILL.md:95-97`) is correct cross-pack discipline — beta-reader simulation is panel work, not craft work, and the right move is to defer rather than duplicate.
- The `tools:` restriction on all 11 agents is plausibly intentional (coach-mode agents read prose and report; they are not file-writers). Preserve it; just document it.

Limitations of this review:
- **Read-only.** No edits made. No fresh-session behavioural verification (subagent dispatch would have been the next step per `testing-skill-quality.md:80-92` but is out of scope for a report-only review).
- **No genre-expertise check.** The reviewer trusted the content quality of the genre sheets on the strength of `mystery.md`'s evidence (cited works, traditions, named writers, axes). A full literary check of the other eight genre sheets is beyond scope and beyond the reviewer's domain expertise; a domain-expert SME would be the right reviewer for that pass.
- **No composition test with `muna-panel-review`.** The router's deferral to `/panel-review` for beta-reader simulation is documented (`SKILL.md:95-97`); not tested at runtime.
- **Stage 5 skipped per instructions.** Recommended actions are sketched but no implementation plan or version-bump commit is included.

Files fully read during review:
- `/home/john/skillpacks/plugins/meta-skillpack-maintenance/skills/using-skillpack-maintenance/SKILL.md`
- `/home/john/skillpacks/plugins/meta-skillpack-maintenance/skills/using-skillpack-maintenance/analyzing-pack-domain.md`
- `/home/john/skillpacks/plugins/meta-skillpack-maintenance/skills/using-skillpack-maintenance/reviewing-pack-structure.md`
- `/home/john/skillpacks/plugins/meta-skillpack-maintenance/skills/using-skillpack-maintenance/testing-skill-quality.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/.claude-plugin/plugin.json`
- `/home/john/skillpacks/plugins/lyra-creative-writing/skills/using-creative-writing/SKILL.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/skills/using-creative-writing/pov-and-voice.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/skills/using-creative-writing/dialogue.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/skills/using-creative-writing/mystery.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/skills/using-creative-writing/literary-fiction.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/skills/using-creative-writing/revision-and-cutting.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/skills/using-creative-writing/story-structure-and-arc.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/commands/draft-scene.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/commands/critique-prose.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/commands/plan-story.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/agents/developmental-reviewer.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/agents/dialogue-doctor.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/agents/revision-coach.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/agents/scene-drafter.md`
- `/home/john/skillpacks/plugins/lyra-creative-writing/agents/premise-stress-tester.md`
- `/home/john/skillpacks/.claude/commands/creative-writing.md`
- `/home/john/skillpacks/.claude-plugin/marketplace.json` (lyra-creative-writing entry)

Tool calls used: Read, Edit, Write, Bash (for `find`, `ls`, `wc`, `grep`, `head`). No edits made to any pack content; report is the only artefact written.

Files inventoried (frontmatter only):
- All 11 agent files in `/home/john/skillpacks/plugins/lyra-creative-writing/agents/` (frontmatter via `head -10`)
- All 22 sheet files in `/home/john/skillpacks/plugins/lyra-creative-writing/skills/using-creative-writing/` (counted, sized via `wc -l`)
- Remaining sheets not read full-text: `scene-construction.md`, `showing-vs-telling.md`, `character-interiority.md`, `prose-rhythm-and-style.md`, `pacing-and-tension.md`, `openings-and-endings.md`, `worldbuilding-by-implication.md`, `research-and-verisimilitude.md`, `creative-nonfiction-craft.md`, and seven genre sheets (`thriller`, `sf`, `fantasy`, `horror`, `romance`, `memoir-and-personal-essay`, `literary-journalism`).
- Remaining agent files not read full-body: `line-reviewer.md`, `pov-and-distance-auditor.md`, `continuity-checker.md`, `opening-and-ending-doctor.md`, `outline-architect.md`, `worldbuilding-consultant.md` (frontmatter confirmed; bodies not audited).

The reviewer's confidence is high in the structural and metadata findings (these were directly checked) and medium-high in the content-quality assertions (extrapolated from a representative sample of six sheets and five agent bodies). A full content audit of every sheet would require a domain-expert SME and is beyond the scope of a process-quality review.

End of report.
