# Report Card — meta-sme-protocol

**Version:** 1.1.0 (plugin.json) · **Track:** P (Process / Hybrid — explicitly listed in rubric §3)
**Graded:** 2026-06-22 · **Prior review:** `reviews/meta-sme-protocol.md` (2026-05-22, also v1.1.0 — no version divergence; fresh read agrees)

This is a deliberately minimal pack: a single reference document
(`skills/sme-agent-protocol/SKILL.md`, 400 lines) with no router, sheets,
commands, or agents. It is not user-invoked — it is the **citation target** for
SME agents across the marketplace. **83 downstream agent files** reference it
(`grep -rl` over `plugins/*/agents/`), so it is a load-bearing ecosystem
contract, not a standalone capability. It must be graded as the protocol-spec
it is, not penalized for lacking router scaffolding it intentionally omits.

---

## Subjects

| Subject | Grade | Load-bearing evidence |
|---|---|---|
| **A — Substance** (Track P) | **A** | Methodology is valid and complete for its declared scope: a 3-phase contract (Fact-Find → Analyze → Output) with a 4-section mandatory output contract (Confidence/Risk/Gaps/Caveats), `SKILL.md:113–209`. Currency is good for a P-track meta-spec: 1.1.0 changelog (`:396`) explicitly modernized the toolset (dropped `firecrawl`/generic `LSP`, added `WebSearch`, `Agent`-tool subagent dispatch, MCP examples §1.5). Confidence/risk vocabularies are crisply defined (`:132–136`, `:152–157`). It is not domain-deep (it can't be — it's a cross-domain protocol), but within its lane it is authoritative and teaches the *why* ("it cannot un-trust a confidently-wrong claim", `:243`). Held off S because depth is necessarily shallow per-domain and it asserts rather than catalogs failure modes exhaustively. |
| **B — Usefulness** | **A** | Highly actionable. The output contract is copy-paste-ready markdown templates (`:119–209`), the Integration Checklist (`:348–356`) tells an agent author exactly how to adopt it, and the Tool Requirements block (`:324–344`) is concrete (Required vs Recommended vs Domain-specific). The WRONG/RIGHT and BAD/GOOD pairs (`:42–59`, `:251–320`) are specific, not platitudes. Proof of usefulness: 83 agents actually consume it. Not S only because there is no router/decision-tree to grade (single doc). |
| **C — Discipline** | **A** | This pack *is* the discipline-signature for the whole marketplace. It names the rationalizations it exists to defeat ("You are NOT providing value if you give generic advice", `:31`; "Don't Pretend to Know What You Haven't Verified", `:264`; "Don't Hedge Without Specifics", `:287`) and operationalizes calibration via mandatory per-finding confidence + risk + gaps. The cross-domain anti-pattern (Rust example, `:303–320`) pre-empts "treat the protocol as Python-only". Self-honest about OPTIONAL vs MUST sections (§3.5/§3.6). Not S because it defines the protocol but cannot itself emit calibrated confidence (it's a spec, not an analysis), so C3 is only partially demonstrable in-pack. |
| **D — Form / Integrity** | **A** | Conformant frontmatter (`:1–4`); registered in `marketplace.json:457–468`; plugin.json/SKILL/marketplace descriptions consistent. **No slash wrapper, correctly** — `meta-*` packs are cited, not user-invoked (sibling `meta-skillpack-maintenance` follows the same convention), so D2 does not penalize absence. Zero drift across surfaces. Changelog + "Last reviewed" footer present (`:394–400`). Trivial nit only: the ASCII summary box has minor right-border misalignment (`:363–365`). |

---

## Gate analysis

1. **Discoverability gate** — Installs fine; not a user-invoked router so the
   missing slash wrapper is *correct convention*, not a broken wiring surface.
   Marketplace-registered. Gate does **not** fire; no cap.
2. **Substance-dominates gate** — Overall ≤ Substance(A) + 1 = S. Not binding.
3. **Honor-roll (S) gate** — Substance is A, not S; so S is unreachable. Correct
   — a single 400-line cross-domain spec is excellent but not "reference-grade
   authoritative across a whole declared domain at expert depth".
4. **Honesty override** — N/A; this is a complete deliverable, not a scaffold.

Blended 40/25/20/15 of A/A/A/A → **A**.

---

## Layered per-component grades

Single component; no weak tail to surface.

| Component | Grade | Note |
|---|---|---|
| `skills/sme-agent-protocol/SKILL.md` | **A** | The pack. Complete, current, exemplary discipline-spec; consumed by 83 agents. Sub-S only because per-domain depth is intentionally shallow and the ASCII box has a cosmetic border nit. |

**S-grade exemplar to copy:** the **mandatory 4-section output contract** with
defined confidence/risk vocabularies (`SKILL.md:113–209`) is the single most
imitable artifact in the marketplace — it is exactly what makes the 83 downstream
reviewer agents calibrated rather than confidently wrong.

---

## Overall: **A**

**Verdict:** A tight, current, load-bearing meta-protocol — the marketplace's
calibration backbone, complete and disciplined within its intentionally narrow lane.

**Top finding:** The pack does what it claims and is genuinely depended upon (83
agent files cite it); v1.1.0 currency refresh kept the tool examples honest while
preserving the §3.1–§3.4 contract so downstream agents needed no edits.

**Top fix:** Cosmetic only — realign the ASCII summary box right border
(`SKILL.md:363–390`); optionally add a one-line note in SKILL.md stating the
deliberate "no slash wrapper, cited-not-invoked" convention for `meta-*` packs
so future graders/maintainers don't flag it as a gap.
