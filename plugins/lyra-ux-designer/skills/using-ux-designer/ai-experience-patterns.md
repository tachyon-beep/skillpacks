# AI Experience Patterns

## Overview

This skill covers UX patterns for interfaces built around large language models, retrieval, and agent loops — the surfaces that did not exist when most UX literature was written. The goal is to give users **calibrated trust**: enough visibility to verify, intervene, and recover when the model is wrong, without burying the affordances under disclaimer noise.

**Core Principle:** AI is a probabilistic collaborator, not a button. Design for legibility (what is the system doing?), grounding (where did this answer come from?), and reversibility (can I undo a confident-sounding mistake?).

> **Scope note:** this skill covers *interface* design. Prompt engineering, evaluation harnesses, and model selection live in `yzmir/llm-specialist`. Safety review (jailbreaks, PII exfiltration) lives in `ordis/security-architect` and `yzmir/llm-specialist`.

## When to Use

Load this skill when:
- Designing a chat / conversation surface (assistant, copilot, customer support)
- Designing inline AI features (autocomplete, "fix this", "summarize this", AI-generated diffs)
- Designing agent surfaces (multi-step tool use, browsing, code execution)
- Designing retrieval-augmented (RAG) UIs where citations and grounding are user-visible
- Designing AI-generated content with editorial review (drafts, reports, marketing copy)
- Designing model-routing or model-selection controls

**Don't use for**: pure backend ML inference, training UI, dashboards that *visualize* model output but don't take user action.


## The AI-UX Trust Stack

Six dimensions for evaluating any AI interface. Each addresses a class of failure that has been observed at production scale.

1. **Legibility** — User can tell what the system is doing right now (thinking, retrieving, calling a tool, streaming, finished).
2. **Grounding** — Outputs cite or visibly link to the sources they were derived from; the user can verify.
3. **Steering** — User can shape, constrain, or correct output mid-flight (stop, edit, regenerate, refine).
4. **Refusal & Recovery** — When the model declines, errors out, or produces a low-confidence result, the UX explains *why* and offers a path forward.
5. **Reversibility** — Destructive or external-side-effect actions (send email, run code, file change, payment) require an explicit confirmation step distinct from the chat flow.
6. **Calibration** — Confidence cues match actual reliability; the UI does not over- or under-state certainty.

---

### Dimension 1: LEGIBILITY (state visibility)

**Purpose:** users need to know whether the model is thinking, has stalled, or has finished — and what it's doing in between.

#### Patterns

**Streaming output with a clear "still going" cue:**
- Tokens render as they arrive (server-sent events or chunked transfer)
- A **persistent indicator** (animated cursor, pulse, or spinner near the message) distinguishes "still streaming" from "finished but short"
- The Stop / Cancel control is reachable while streaming — primary placement, not buried

**Progress disclosure for long-running steps:**
- For agents that browse, call tools, run code: show the *current step name* ("Searching docs", "Running query") in a stepper or live transcript
- Collapse completed steps into a one-line summary; expand on click for detail
- Don't fake progress bars over indeterminate work — use indeterminate animation honestly

**Latency-shaped feedback:**
- <100 ms: no indicator needed
- 100 ms – 1 s: subtle spinner inside the input
- 1 – 10 s: streaming begins; show "Thinking…" until first token
- 10 s+: show what step is running; offer cancel; estimate only if you have real ETA data

#### Anti-patterns

- **Silent compute** — model is working but the UI shows nothing for 8 seconds. Users retry, double-charge, or leave.
- **Fake typing animation** when the response was already fully returned — wastes the user's time and trains them to distrust the cue.
- **Hidden Stop control** — if the user can't cancel, they will close the tab. Closing a tab during a tool call can leave external side effects half-applied.

---

### Dimension 2: GROUNDING (citations and provenance)

**Purpose:** when the answer is derived from documents, search results, or tool output, the user can trace any claim back to its source.

#### Patterns

**Inline citations as foot-marker chips:**
- Numbered superscript or pill that links to the source span
- Hovering reveals a preview (first ~120 characters of the cited passage)
- Clicking opens the source in a side panel — *not* a new tab unless the user asks

**Sources panel:**
- A persistent list of all documents the answer drew on
- Each entry: title, source domain or repository, retrieval score (if you display it), "open" affordance
- Expandable to show the exact retrieved chunks

**Quote-vs-paraphrase distinction:**
- When the model quotes a source verbatim, render it as a blockquote with the citation attached
- When it paraphrases, attach the citation but use plain prose styling
- This visual difference primes the user to verify the paraphrase but trust the quote

**Empty-grounding signal:**
- When retrieval returned nothing useful and the model is answering from parametric memory only, **say so**: "No relevant documents found — this answer is from the model's general knowledge."
- Never silently fall back from RAG to ungrounded generation; that's how trust dies.

#### Anti-patterns

- **Plausible citations to non-existent sources** — link rot or hallucinated URLs. If you can't verify a source resolved at generation time, don't render it as a clickable link.
- **Citation-without-traceability** — `[1]` markers with no corresponding source list, or links that go to a domain root rather than the cited passage.
- **Grounding as decoration** — citations attached at the end of the message body but not anchored to specific claims. Users learn to ignore them.

---

### Dimension 3: STEERING (mid-flight control)

**Purpose:** the user can shape output without restarting from scratch.

#### Patterns

**Stop button** as a primary control during streaming. After Stop:
- The partial output stays visible (don't erase it — it may be useful)
- Offer "Continue", "Regenerate", "Edit prompt and retry"

**Regenerate with variation hint:**
- Buttons: "Try again" (same prompt) and "Try with…" (shorter, longer, more formal, simpler — pick from chips)
- This is a far better discovery surface than a free-text "tell me what to change" loop, because the affordances are visible

**Inline edit on AI output:**
- For drafts, code, or generated documents: the user edits the output directly in place
- Their edits become part of the next turn's context — implicit feedback, no thumbs-up/down required
- This is the "diff-and-merge" pattern from coding copilots: show what the model proposes, let the user accept/reject hunks

**Partial accept (hunked acceptance):**
- For multi-paragraph or multi-file output, allow accepting some sections and rejecting/regenerating others
- Avoids the all-or-nothing trap that drives users to redo whole prompts

#### Anti-patterns

- **Locked output** that can only be copy-pasted — forces a clipboard-edit-paste loop, breaks context for the next turn.
- **Single Regenerate button with no parameters** — the user has no idea what will be different next time. Add at least one steering chip.
- **Auto-regeneration on every typo** — wastes tokens, surprises the user, and can lose work.

---

### Dimension 4: REFUSAL & RECOVERY

**Purpose:** when the model declines, errors, or produces something the system filters, the user understands *what happened* and has a path forward.

#### Patterns

**Explained refusal:**
- Don't surface raw policy strings ("I cannot help with that"). Translate the refusal into the user's task language: "I can't generate that — it looks like a request for personal medical advice. Would you like a list of authoritative sources instead?"
- Offer a constructive next step where one exists. Refusal without a forward path drives users to jailbreak.

**Tool-call failure:**
- When an agent's tool call fails (404, timeout, auth error), show: which tool, what error, and a retry / skip / edit-arguments affordance
- Don't bury the error inside a transcript step that's collapsed by default

**Filter / safety classifier triggered:**
- If a response was blocked or rewritten by a safety filter, say so honestly. Silent rewrites destroy trust the moment the user notices.
- Provide an appeal channel where appropriate (enterprise tools especially)

**Low-confidence answers:**
- When the model self-reports uncertainty, or when grounding score is below threshold, render a low-confidence chip on the answer
- Don't hide the answer — show it with the caveat. Users prefer "here's a guess, verify it" over "I don't know."

#### Anti-patterns

- **Silent failure** — tool call returned an error, model summarized the error away, user never sees it.
- **Refusal-by-disclaimer** — the answer is technically there but buried under three paragraphs of "I am an AI and cannot…" boilerplate.
- **Jailbreak-inviting walls** — flat refusals with no explanation train users to phrase requests as "let's roleplay" to bypass the wall, which is the worst of both worlds.

---

### Dimension 5: REVERSIBILITY (side effects)

**Purpose:** any action with external or destructive consequence — sending an email, running code, modifying a file, charging a card, calling a customer-facing API — requires explicit confirmation, distinct from the chat flow.

#### Patterns

**Two-stage commit for tool actions:**
- Stage 1: model proposes the action with a structured preview (`POST /api/refund` body, the email draft, the SQL query, the file diff)
- Stage 2: the user explicitly approves; only then does the tool call fire
- For trusted, low-risk actions (a search query) you can auto-approve; document where the line is and let users adjust it

**Diff-style preview for file / code mutations:**
- Show added / removed / changed lines, not just the new content
- Allow reject-per-hunk, not just accept-all

**Undo affordance on destructive actions:**
- Where the underlying API supports it, expose a 30-second undo on irreversible-feeling actions (sent email, archived item, deployed change)
- Where the API does not support undo, escalate the confirmation: type-to-confirm, secondary modal, etc.

**Audit log surface:**
- Agents that act on the user's behalf should expose a "what did the assistant do" log, queryable after the fact
- Especially in enterprise / regulated contexts where accountability is non-negotiable

#### Anti-patterns

- **Auto-execution of arbitrary tool calls** with no preview — this is how an agent ends up sending the wrong customer the wrong refund.
- **Confirmation theatre** — a modal that just says "Are you sure?" without showing the action's payload teaches users to click through.
- **Undo theatre** — a toast that says "Undo" but the underlying action is already irreversible.

---

### Dimension 6: CALIBRATION (confidence cues)

**Purpose:** the visual confidence the UI conveys must match the actual reliability of the output. Over-confident interfaces erode trust on the first wrong answer; under-confident interfaces train users to ignore real warnings.

#### Patterns

**Calibrated confidence indicators:**
- If you display a confidence percentage, it must come from a well-calibrated source (model logprobs aggregated correctly, classifier score, retrieval quality). Made-up numbers are worse than no numbers.
- Prefer qualitative chips ("verified", "likely", "best guess", "uncertain") backed by a defined threshold, over false-precision percentages

**Source-quality signals:**
- For RAG: surface the retrieval score or freshness ("from a doc dated 2019")
- For tool output: surface the source ("from the inventory database, last synced 2 minutes ago")

**Disagreement disclosure:**
- When two retrieved sources disagree, *say so*: "Source A says X; Source B says Y." Don't let the model average over a contradiction silently.

**Verification prompts on high-stakes claims:**
- Numerical claims, legal claims, medical claims, dosage figures — pair with "Verify in source" affordances rather than presenting as plain prose

#### Anti-patterns

- **Confident hallucination styling** — bold, definitive prose that turns out to be wrong. Visual confidence should fall when grounding falls.
- **Disclaimer inflation** — slapping "this may be inaccurate" on every message. Users tune it out; you've burned your only signal.
- **False-precision percentages** — "I am 87% confident". Where did the 87 come from? If you can't answer, don't display it.

---

## Surface-Specific Patterns

### Conversational Assistants

**Layout:**
- Threaded transcript with clear turn boundaries
- User turn: right-aligned or distinct background; assistant: left-aligned or full-width
- Composer pinned bottom; sticky on scroll
- For long answers, the assistant message gets a "scroll to bottom" affordance and a copy-message control

**Composer affordances:**
- File / image attachments visible as chips above the input
- Slash-commands or `/` menu for advanced operations
- Token / character budget visible only when relevant (long-form tools, embedded contexts)
- Multi-line by default; submit on Cmd/Ctrl+Enter, Enter inserts newline (or vice versa — pick one and be consistent)

**History and threading:**
- Sessions named automatically from first turn (use the model itself for the title)
- Allow rename, pin, archive, search across history
- For multi-tab use, support split / fork: "continue this conversation in a new branch from turn N"

### Inline Copilots (in-document AI)

**Trigger:**
- Selection-based ("ask about this paragraph") or cursor-based ("complete this sentence")
- Keyboard-first: a single shortcut (`Cmd+K`, `Cmd+J`, `Tab`) is the dominant pattern

**Suggestion presentation:**
- Inline ghost text for completion (gray, italic, accept on Tab)
- Side panel for longer transformations ("rewrite formally") with diff against the original
- Always show what is *about* to change before applying

**Acceptance model:**
- Accept whole / accept word-by-word / reject / regenerate
- The accepted text becomes part of the document; rejected text is discarded silently

### Agent Loops

**Plan-then-act:**
- Show the agent's proposed plan as a visible step list before execution
- Allow the user to edit the plan (skip steps, reorder, add a constraint)
- Each step renders status: queued → running → succeeded / failed / skipped

**Per-step intervention:**
- Pause / resume on the agent loop (not just on streaming text)
- Approve-each vs approve-all toggle, with a default that scales to action risk

**Budget visibility:**
- Token budget, time budget, tool-call budget — visible during long runs
- Hard cap with explicit user-facing exhaustion message ("Stopped after 25 tool calls — continue?")

### RAG Surfaces

**Query → retrieval → answer triplet:**
- Show the rewritten / expanded query if you do query rewriting (transparency)
- Show retrieved passages as collapsible cards with snippets and links
- Pin the answer above the retrieved cards so users see the synthesis first, then verify

**Empty-result UX:**
- Distinguish "no documents matched" from "documents matched but the model declined to answer"
- Offer query refinement chips ("broaden", "search by date range", "search a specific repo")

### Multi-Turn Editing (drafts, reports, generated docs)

**Document-as-canvas:**
- The generated artefact lives in its own canvas, not buried in chat
- Chat is the *control surface* for editing the canvas (a la Anthropic Artifacts, GitHub Copilot Workspace)
- Edits track to the canvas immediately; the chat history is a log of "what the user asked the assistant to change"

**Version history:**
- Save snapshots on each significant edit; allow diff-and-revert
- Especially important when agentic edits cascade into multiple files

---

## Accessibility for AI Surfaces

AI interfaces inherit all of WCAG 2.2, plus a few AI-specific traps. See `accessibility-and-inclusive-design.md` for the foundational requirements; the additions below are AI-specific.

- **Streaming text and screen readers:** an `aria-live="polite"` region is the right primitive, but be careful — naïve implementations re-announce on every token. Buffer tokens into sentence-level chunks before announcing, or use `aria-busy` during streaming and announce the final result.
- **Cognitive accessibility:** AI outputs default to long, dense prose. Provide a "summarize" / "simpler version" affordance — this is also a model invocation, but it's the right cognitive ramp.
- **Refusal and error messages:** apply the WCAG 3.3.1 / 3.3.3 patterns from the cognitive section — describe the error in user terms, suggest a fix.
- **Authentication for paid AI features:** WCAG 2.2 SC 3.3.8 (Accessible Authentication) applies — don't gate AI features behind CAPTCHA-style cognitive tests without alternatives.
- **Reduced motion:** the streaming-cursor blink and "thinking" pulses must respect `prefers-reduced-motion`.

---

## Trust Erosion Pathways (what kills AI products)

Patterns that have measurably destroyed user trust at scale, in production:

1. **Confident-wrong cycle.** Model asserts X with high visual confidence; X is wrong; user is embarrassed; user stops using the tool.
   *Mitigation:* calibrate confidence cues; ground claims; offer verification.

2. **Silent side effects.** Agent did something the user didn't realize was possible (sent message, charged card, deleted file).
   *Mitigation:* preview-then-confirm for any external-effect tool call; audit log.

3. **Disclaimer inflation.** Every response wrapped in "I'm an AI…" boilerplate — users tune out and miss real warnings.
   *Mitigation:* reserve disclaimers for actual uncertainty or refusal; treat them as signal, not safety theatre.

4. **Cite-rot.** Citations link to URLs that 404, or to source pages that no longer contain the cited claim.
   *Mitigation:* verify citation resolves at generation time; cache the cited span; surface "source moved" when re-checked later.

5. **Stuck-loop agent.** Agent calls the same tool with the same args until budget exhausted.
   *Mitigation:* loop detection (same tool + same args twice → ask user); hard caps; visible budget.

6. **Help-me-help-you blockers.** Refusal walls with no explanation drive users to jailbreak prompts to get the work done — worst of both worlds.
   *Mitigation:* explained refusal with a constructive next step.

---

## Practical Application

### Audit Checklist (any AI surface)

- [ ] State during compute: is "thinking", "retrieving", "calling tool" visible in real time?
- [ ] Stop / cancel reachable during streaming and during tool calls
- [ ] Citations anchored to specific claims, not the message body as a whole
- [ ] Verifiable links: do they resolve? Do they go to the cited passage, not the homepage?
- [ ] Refusal language explains *why* and offers a path forward (not raw policy strings)
- [ ] Tool-call errors visible, not silently swallowed
- [ ] Side-effect actions go through preview-then-confirm
- [ ] Confidence cues match actual reliability (no false-precision percentages)
- [ ] Streaming output respects `aria-live` and `prefers-reduced-motion`
- [ ] Agent loops have visible budgets and per-step intervention
- [ ] Audit log surface for any agent that acts on the user's behalf

### Common Failure Mode → Fix Recipe

| Failure                                            | First fix                                                       |
| -------------------------------------------------- | --------------------------------------------------------------- |
| Users complain "it just sat there"                 | Add streaming + in-flight step indicator                        |
| Users complain "it made things up"                 | Add grounding + citation chips; surface low-confidence honestly |
| Users complain "I can't tell what changed"         | Add diff view for AI edits, partial-accept                      |
| Users hit Stop and lose work                       | Preserve partial output after Stop                              |
| Agent did something destructive                    | Insert preview-then-confirm; add audit log                      |
| Citations look authoritative but are wrong         | Verify-at-generation; render unverifiable cites as plain text   |
| "I'm an AI…" preamble on every message             | Cut the boilerplate; reserve disclaimers for actual uncertainty |

---

## Related Skills

**Core UX:**
- `lyra/ux-designer/interaction-design-patterns` — feedback timing, streaming animation primitives, modal patterns
- `lyra/ux-designer/accessibility-and-inclusive-design` — WCAG 2.2, aria-live regions, cognitive load
- `lyra/ux-designer/visual-design-foundations` — confidence chips, source-card hierarchy

**Cross-faction:**
- `yzmir/llm-specialist/*` — prompt engineering, RAG quality, evaluation harnesses (the "is the model actually right" side of the question)
- `ordis/security-architect/*` — prompt-injection defense, PII exfiltration, supply-chain risk for agent tools
- `muna/technical-writer/*` — tone of voice for AI assistant copy, error-message language

---

## Further Resources

- Anthropic — *Building agents with the Claude API*: https://docs.anthropic.com/
- OpenAI — *Best practices for chat-based interfaces*: https://platform.openai.com/docs/
- *Designing for AI* (Lou Downe / Maggie Appleton's writing on AI artefacts) — pattern essays
- *Prompt-injection cheat-sheet* (OWASP LLM Top 10): https://owasp.org/www-project-top-10-for-large-language-model-applications/
- Apple — *Generative AI Human Interface Guidelines*: developer.apple.com/design/human-interface-guidelines/generative-ai
- Google — *People + AI Guidebook*: pair.withgoogle.com/guidebook
- Nielsen Norman Group — AI/UX research: nngroup.com (search "AI")

**Remember:** an AI feature is judged by the user's *recovery* path on the day it is wrong, not by the demo on the day it is right. Design every surface to be useful when the model is uncertain, mistaken, or refusing — not just when it nails the answer.
