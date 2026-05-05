# Refresh: yzmir-deep-rl

**Verdict:** MEDIUM / M effort. Solid PPO/SAC fundamentals; algorithmic coverage ~2021.

## Context

- Pack path: `/home/john/skillpacks/plugins/yzmir-deep-rl/`
- Full review: `/tmp/skillpack-refresh-review/yzmir-deep-rl.md`
- Purpose: deep RL algorithm selection, scaffolding, diagnosis.

## Why refresh

Solid PPO / SAC / CQL / IQL fundamentals but algorithmic coverage froze ~2021. Add:

- **Modern offline RL.** Decision Transformer, TD3+BC, AWAC.
- **Preference / RLHF crossover.** GRPO (DeepSeek), with cross-ref to `yzmir-llm-specialist` for DPO/IPO/SimPO.
- **Model-based.** DreamerV3, TD-MPC2.
- **Sample-efficient model-free.** REDQ, CrossQ, DroQ.
- **Multi-agent.** MAPPO, IPPO basics.
- **Agentic RL pointer.** Briefly note RL-for-LLMs and how it differs from classic RL — link out.
- **Framework awareness.** Minari (offline data), cleanrl (single-file reference), sb3 vs Pearl vs Tianshou current state, gymnasium (replacing gym).

## Scope — DO

1. **Algorithm selection wizard.** Add modern algos to decision tree.
2. **Offline RL skill.** New or expanded — DT, TD3+BC, AWAC, IQL.
3. **Model-based skill.** DreamerV3, TD-MPC2.
4. **Sample efficiency.** REDQ, CrossQ, DroQ.
5. **Multi-agent.** MAPPO basics.
6. **Framework primer.** Update gym → gymnasium, mention Minari for offline.
7. **Agentic-RL pointer.** Cross-ref to `yzmir-llm-specialist`.

## Scope — DO NOT

- Do not duplicate LLM-RLHF content (lives in `yzmir-llm-specialist`).
- Do not name a single "best" algorithm.

## Acceptance criteria

1. DT, TD3+BC, AWAC, IQL covered in offline-RL.
2. DreamerV3 or TD-MPC2 covered in model-based.
3. MAPPO covered.
4. gym → gymnasium migration noted.
5. Cross-pack reference to `yzmir-llm-specialist` for RLHF.
6. `plugin.json` version bumped (minor).

## Process

1. Read `/tmp/skillpack-refresh-review/yzmir-deep-rl.md`.
2. Read every SKILL.md.
3. Each algorithm cites a paper.
4. Bump version.

## Constraints

- Every algorithm has a paper citation.
- No "this beats X by Y%" claims — leaderboards drift.
