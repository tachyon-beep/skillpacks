# Report Card — yzmir-pytorch-engineering

**Version:** 1.2.1 (plugin.json) · **Track:** H — Hard / Technical
**Graded:** 2026-06-22 · **Prior review:** 2026-05-22 (claimed v1.2.0 — version has since bumped to 1.2.1; fresh reading weighted over prior)

Structure verified: 1 router + 8 reference sheets (~14k LOC of sheet content), 3 commands, 2 agents. Counts match plugin.json declaration exactly.

---

## Subject grades

| Subject | Grade | Evidence |
|---------|-------|----------|
| **A — Substance** | A− | Deep, correct, expert-level coverage of the modern PyTorch surface. Router currency gate (`SKILL.md:14,18-27`) is exemplary. `mixed-precision-and-optimization.md` is a clean "API truth layer": correct `torch.amp.GradScaler("cuda")` usage, correct unscale-before-clip ordering (`:166-209`), accurate FP16/BF16 range table (`:289-300`), explicit migration note citing PyTorch docs (`:45-57`). `distributed-training-strategies.md` correctly stages DDP→FSDP1→FSDP2 (`:48-57`) with accurate `nn.DataParallel` "not-recommended-not-deprecated" framing (`:64`). Held off S by a real currency leak (below) — sheets that swore off `torch.cuda.amp` still echo it. |
| **B — Usefulness** | A | Router is reference-grade for routing: symptom tables, cross-cutting sequences (`:246-271`), common-routing-mistakes table (`:297-310`), diagnosis-first matrix (`:400-409`). Sheets are concrete (runnable snippets, decision tables). Commands have argument hints and phased frameworks. Reading it changes what you do. |
| **C — Discipline** | B+ | Strong: rationalizations table names "User is rushed", "Authority says Y", "Just a quick question" verbatim (`SKILL.md:333-346`); red-flags self-check (`:350-379`); both agents declare SME Agent Protocol with Confidence/Risk/Information Gaps/Caveats and carry `model: sonnet` (`agents/*.md:1-10`). Docked because the pack's own anti-pattern — "Never echo deprecated APIs" (`SKILL.md:310,324`) — is violated by its own sheets (see Top finding). Discipline that the pack fails to apply to itself. |
| **D — Form** | B | Conformant, fully wired: slash wrapper `.claude/commands/pytorch-engineering.md` is present, current, and detailed (mirrors router). Registered in marketplace (`marketplace.json:635`). One Minor drift: marketplace description reads "9 skills" with no 2.9 feature detail (`:637`) vs plugin.json's richer "8 reference sheets + 1 router, 3 commands, 2 agents" — stale counting convention. Internal version labels mix "PyTorch 2.9 (2025 release)" (agent) and "2.11" (distributed sheet); not contradictory but loose against the declared "2.9+" baseline. |

---

## Gate analysis

1. **Discoverability gate:** PASS. Installs, router loads, slash wrapper present + current, registered. No cap.
2. **Substance-dominates gate:** Substance = A−, so overall ≤ A. Not binding below A.
3. **Honor-roll (S) gate:** FAILS for S — Substance is A− (not S) and there is a Major-adjacent self-consistency defect. No S.
4. **Honesty override:** N/A — pack is complete, not a scaffold; marketing slightly overstates currency ("migrated to torch.amp") vs reality, which is the finding below, not a vapor case.

---

## Layered — worst-offending components

| Component | Grade | Note |
|-----------|-------|------|
| `tensor-operations-and-memory.md` | B− | Three live `from torch.cuda.amp import autocast, GradScaler` example snippets (`:359,861,954`) — the exact deprecated API the router forbids echoing. Not in migration-note context. |
| `debugging-techniques.md` | B− | Same leak: live `from torch.cuda.amp import ...` at `:1396`. |
| `performance-profiling.md` | B | Echoes deprecated API in a printed remediation string `"Use mixed precision (torch.cuda.amp)"` (`:1165`) and example import (`:1190`). |
| `marketplace.json` entry | B | "9 skills" description drift vs plugin.json's modern framing (`:637`). |
| `pytorch-code-reviewer.md` (agent) | B+ | Solid SME-compliant reviewer; "PyTorch 2.9 (2025 release)" section (`:90-124`) is slightly thin/marketing-flavored vs the sheets' rigor. |

**Exemplar worth copying:** `mixed-precision-and-optimization.md` (the AMP authority sheet) and the router `SKILL.md` — both near-S in their lane (correct modern API, cited sources, ordered workflows, rich routing discipline).

---

## Overall: **A−**

The Substance/Usefulness ceiling is high (A/A) and the wiring is clean, but the pack is dragged off a clean A by a self-consistency defect: it loudly markets full migration off `torch.cuda.amp` and forbids echoing it, then echoes it in live example code across three secondary sheets. That is a Minor-bordering-Major drift between marketing and content — exactly what Subject C/D penalize — so the blend lands A−, not A. Reconciles with existing-system **Minor**.

**Verdict:** Reference-grade router and AMP sheet undercut by a handful of deprecated-API snippets in secondary sheets that contradict the pack's own currency promise.

**Top finding:** The currency gate (`SKILL.md:14`, plugin desc) and red-flag rules ("Never echo deprecated APIs", `SKILL.md:310,324`) claim `torch.cuda.amp` has been migrated to `torch.amp` throughout — but ~5 live example snippets still import `from torch.cuda.amp import autocast, GradScaler` outside any migration note (`tensor-operations-and-memory.md:359,861,954`; `debugging-techniques.md:1396`; `performance-profiling.md:1165,1190`). The pack violates its own headline discipline.

**Top fix:** Sweep the three secondary sheets and replace every live `torch.cuda.amp.*` import/call with the `torch.amp.autocast("cuda", ...)` / `torch.amp.GradScaler("cuda")` form already canonical in `mixed-precision-and-optimization.md`; leave deprecated mentions only inside explicitly-labeled migration notes. Also refresh the marketplace description ("9 skills" → match plugin.json's "8 sheets + 1 router, 3 commands, 2 agents"). Both are mechanical and would lift the pack to a clean A.
