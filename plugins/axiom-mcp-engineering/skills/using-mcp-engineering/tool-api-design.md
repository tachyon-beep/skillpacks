---
name: tool-api-design
description: Use when an agent mis-calls tools, picks the wrong one of two similar tools, leaves required parameters blank or guesses them, ignores a tool it should have used, or treats your server like a database CRUD panel — and when you are naming a new tool, writing its description, deciding its granularity, or shaping its parameters for a model that has never seen your source code. Covers workflow-verb naming, agent-voice intent statements, tool-description-as-prompt-fragment, parameter shapes the agent can actually fill, granularity (when to split vs merge tools), and the CRUD-mirror anti-pattern.
---

# Tool API Design

*Current as of MCP revision 2025-11-25 (prior: 2025-06-18); JSON Schema dialect 2020-12.*

**An LLM reads your tool surface the way it reads a prompt: literally, with no source code, no schema docs you did not ship, and no human to ask. The tool name, the description, and the parameter schema are the *entire* interface. Everything you know that the model cannot read is a defect.**

A human integrating a REST API reads the OpenAPI doc once, writes a client, debugs it against real responses, and internalizes the quirks. An LLM integrating your MCP server gets the `tools/list` payload — name, description, `inputSchema` (JSON Schema 2020-12), and optional annotations and `outputSchema` — re-injected into its context on (potentially) every turn, and has to decide, in one forward pass, *which* tool to call and *how* to fill it, with no memory of last time except what is in the conversation. The tool surface is a prompt fragment competing for attention with the user's actual request. Designing it is prompt engineering wearing an API's clothes.

This sheet is read by two roles over the same surface. The **architect** is naming tools, writing descriptions, and shaping parameters for a workflow. The **critic** is re-reading a deployed catalog as a prompt and asking where an LLM with no source code will misfire. They share the catalog; they do not share epistemics. The architect's tools always *feel* clean — they mirror the code the architect just wrote. The critic's job is to read them as the model will, which is to say without the code.

---

## The Core Inversion: Name For Agent-Meaning, Not For The Codebase

The single most common tool-API defect is naming the tool after **what it does in your code** instead of **what it means to an agent's workflow**.

Your codebase has an `issues` table. The obvious tools are `get_issue`, `create_issue`, `update_issue`, `delete_issue`. This is a CRUD mirror, and it is almost always wrong, because the agent does not have a workflow called "update a row." The agent has workflows called "find something to work on," "claim it so no other agent takes it," "record what I did," "hand it off." The agent has to *translate* its intent into your data-model verbs, and that translation is exactly where it goes wrong: it calls `update_issue` with `{status: "in_progress"}` when it meant "claim this," not realizing that nothing stops a second agent from doing the same `update_issue` a millisecond later.

Name tools for the **effect an agent cares about**, in the **agent's own voice**:

| CRUD-mirror name (codebase voice) | Workflow-verb name (agent voice) |
| --- | --- |
| `update_issue` (status field) | `claim_issue`, `close_issue`, `reopen_issue` |
| `create_record` (type=deployment) | `record_deployment` |
| `get_rows` (table=alerts, filter) | `list_active_alerts` |
| `patch_config` (key, value) | `set_feature_flag`, `rotate_api_key` |
| `delete_lock` | `release_claim` |
| `post_message` (channel, body) | `notify_oncall`, `reply_in_thread` |

The test: read the tool name aloud as a sentence the agent would say to itself — *"I should `claim_issue`"* vs *"I should `update_issue`"*. The first names a decision the agent is actually making. The second names a side effect of a decision the agent has not yet articulated. A tool whose name only makes sense if you know the schema is a tool the agent will mis-fill.

This is **not** a style preference. The CRUD mirror has three concrete failure modes that workflow verbs avoid:

1. **Indistinguishable tools.** `update_issue` does five different conceptual things depending on which field you set. The model cannot tell from `tools/list` that setting `status` is a claim race and setting `title` is harmless. Splitting into `claim_issue` / `rename_issue` / `close_issue` makes each tool *one* decision with *one* idempotency story.
2. **Parameter shapes the agent cannot fill.** `update_issue` needs a partial-update payload — which fields? all of them? just the changed ones? The model guesses, often sending fields it did not mean to change. `claim_issue(issue_id)` has exactly one fillable parameter.
3. **Idempotency lies.** `update_issue` cannot declare a single idempotency guarantee because it depends on the field. `claim_issue` declares `fails-second-call` (someone already holds it) or `requires-claim-lease`; `close_issue` declares `no-op-after-first`. Per the [Consistency Gate](SKILL.md), every side-effecting tool must declare *one* idempotency guarantee — a CRUD-mirror tool structurally cannot.

---

## The Tool Description IS A Prompt Fragment

The `description` field is not documentation. It is a sentence injected into the model's context that competes with the user's request for the model's attention and decides whether the tool gets called at all, and called correctly. Treat it as prompt engineering.

A description does four jobs, in this order of importance:

1. **States the agent-voice intent** — the effect an agent cares about, not the implementation. This is the Consistency Gate's first hard requirement. *"Marks an issue as in-progress and held by the calling agent so no other agent claims it"* — not *"updates the status column of the issues table to 'in_progress'."*
2. **Disambiguates from sibling tools.** If `list_active_alerts` and `list_all_alerts` both exist, each description must say *when to pick this one over the other*. The model reads both descriptions side by side; the disambiguation has to live in the text, because the names alone rarely carry it.
3. **States preconditions and the recovery path.** *"Call `claim_issue` first; this tool fails if you do not hold the claim."* The model has no other way to learn ordering constraints.
4. **States what NOT to use it for.** A negative scope line prevents the most common misuse. *"Does not create the issue — use `create_issue` for that. This only transitions an existing issue."*

A description that reads like a docstring auto-generated from the function signature (*"Update issue. Args: issue_id, fields."*) fails all four jobs. The model can see the signature already; the description has to add the *meaning the signature cannot carry*.

Worked contrast — the same tool, badly and well described:

```jsonc
// BAD: docstring voice. Implementation-described, no disambiguation,
// no precondition, no negative scope. The model will call this whenever
// it sees an issue, including to create one.
{
  "name": "update_issue",
  "description": "Updates an issue.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "id": { "type": "integer" },
      "fields": { "type": "object" }   // the agent cannot fill this
    },
    "required": ["id", "fields"]
  }
}
```

```jsonc
// GOOD: agent voice. Intent first, disambiguation, precondition,
// negative scope. One decision, one fillable parameter.
{
  "name": "claim_issue",
  "title": "Claim issue for work",
  "description": "Marks an open issue as in-progress and held by you, so no other agent claims the same work. Call this BEFORE you start working an issue you found via list_open_issues. Fails with retry-with-changes if another agent already holds the claim (pick a different issue). Does NOT create or close issues — use create_issue / close_issue for those. The claim is released by close_issue or release_claim, or it expires after the lease TTL.",
  "inputSchema": {
    "type": "object",
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "properties": {
      "issue_id": {
        "type": "integer",
        "description": "The id field from a list_open_issues result. Do not invent ids."
      }
    },
    "required": ["issue_id"],
    "additionalProperties": false
  },
  "annotations": {
    "title": "Claim issue for work",
    "idempotentHint": false,        // claiming twice is meaningful (race)
    "destructiveHint": false,
    "readOnlyHint": false
  }
}
```

Note the per-parameter `description` (*"The id field from a list_open_issues result. Do not invent ids."*) — this closes the most common parameter-fill failure, where the model fabricates an id rather than carrying one forward from a prior result. Annotations are advisory metadata the host *may* show to the user or use to gate confirmation; per the spec they are **untrusted unless from a trusted server**, so never treat `readOnlyHint: true` as a security control — it is a UX hint, not a sandbox.

---

## Parameters The Agent Can Actually Fill

Every parameter is a question you are asking the model mid-reasoning. The model can only answer from: the conversation so far, prior tool results, the system prompt, and its training. If a parameter requires information from *none* of those sources, the model will fabricate it, omit it, or pick a placeholder. This is the **parameter-the-agent-cannot-fill** smell.

Parameters fall into fillable and unfillable categories:

**Fillable** — the model has a path to the value:
- Carried forward from a prior tool result (`issue_id` from `list_open_issues`). Make this explicit in the parameter description.
- Derivable from the user's request (a search query, a free-text title, a boolean choice).
- A constrained choice the model can reason about (an `enum`).

**Unfillable** — the model has no path; redesign the parameter or supply the value server-side:
- Internal identifiers the model never saw (a `tenant_uuid`, a `shard_key`, an `etag` the model was never handed). Inject these server-side from session/auth context; do not ask the model.
- Values that require running your code to compute (a content hash, a computed `expected_version` the model cannot derive without reading the row — supply it in the prior tool's result, or have the tool read it and use optimistic concurrency internally).
- Opaque tokens (a pagination `cursor` is fine *only if* you returned it in the previous page and the description says "pass the cursor from the previous result verbatim").
- Free-form objects (`fields: object`, `options: object`) — the model cannot know the shape. Replace with named, typed properties.

Concrete parameter discipline, expressed in JSON Schema 2020-12:

```jsonc
{
  "name": "list_open_issues",
  "description": "Returns a bounded page of open, unclaimed issues you could work on next, most recently updated first. Use this to FIND work before claim_issue. Returns at most page_size issues plus a next_cursor when more exist; pass that cursor back to get the next page.",
  "inputSchema": {
    "type": "object",
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "properties": {
      "label": {
        "type": "string",
        "description": "Optional. Filter to issues with this label, e.g. from the user's request ('bug', 'docs'). Omit to see all open issues."
      },
      "priority": {
        "type": "string",
        "enum": ["low", "medium", "high", "critical"],
        "description": "Optional. Filter to a single priority. Omit for all priorities."
      },
      "page_size": {
        "type": "integer",
        "minimum": 1,
        "maximum": 50,
        "default": 20,
        "description": "How many issues to return (1-50). Defaults to 20; leave default unless the user asked for more."
      },
      "cursor": {
        "type": "string",
        "description": "Opaque pagination token. Pass the next_cursor from the previous result VERBATIM to get the next page. Omit for the first page; do not invent a value."
      }
    },
    "required": [],
    "additionalProperties": false
  }
}
```

What this schema does right, and why it matters to an LLM client:
- **`enum` over free-text** for `priority`: the model picks from a closed set instead of guessing the server's vocabulary (`"high"` vs `"High"` vs `"P1"`).
- **`default` + `minimum`/`maximum`** for `page_size`: the model does not have to invent a number, and cannot request a page that blows the context budget. (Per the Consistency Gate, this tool's return shape is paginated — the schema enforces the bound.)
- **`additionalProperties: false`**: the server rejects fabricated extra fields as a **Tool Execution Error** (input-validation failures are returned as tool errors in the result, not protocol errors, since the 2025-06-18 revision), so the model gets a recoverable signal and can self-correct rather than silently having junk ignored.
- **Per-parameter `description` with provenance**: every parameter says *where the value comes from*, especially `cursor` ("pass verbatim, do not invent").
- **Minimal `required`**: a tool with many required parameters is a tool the model often cannot call at all. Required only the parameters with no sensible default and a clear source.

---

## Granularity: When To Split, When To Merge

The granularity question — *one tool or several?* — has a decision rule, not a vibe.

**Split a tool when** any of these hold:
- It does conceptually distinct things depending on which parameter is set (the `update_issue`-by-field problem). Each distinct effect deserves its own name, intent, and idempotency guarantee.
- The distinct paths have **different idempotency guarantees**. `claim` (`fails-second-call`) and `close` (`no-op-after-first`) cannot live in one tool that declares one guarantee.
- The distinct paths have **different concurrency contracts** or **different permission requirements**.
- The parameter schema has mutually-exclusive groups ("if `mode=A` then `x` required, if `mode=B` then `y` required"). JSON Schema can express this with `oneOf`, but the model fills `oneOf` schemas poorly. Split instead.

**Merge tools when** any of these hold:
- They are the *same decision* with a parameter difference the model already reasons about naturally (`list_open_issues` with an optional `label` filter, not `list_open_issues` + `list_open_issues_by_label`).
- They are so similar that the model cannot reliably pick between them (the **overlapping-tools** smell). If two descriptions have to work hard to disambiguate, the catalog is probably carrying one tool too many.
- One is only ever called immediately after the other with no agent decision in between (a forced two-step that should be one tool, or a workflow the server can fuse).

The granularity sweet spot: **one tool = one decision the agent makes = one idempotency guarantee = one concurrency contract.** A surface where every tool maps to a distinct workflow decision is one the model navigates in a single forward pass. A surface with twelve overlapping variants of "get stuff" forces the model to disambiguate at call time, which it does badly.

A heuristic for the critic: count the tools whose descriptions begin with the same verb (`get_`, `list_`, `fetch_`). Three `list_*` tools that differ only by filter are usually one tool with parameters. Three `get_*` tools that return different *shapes* of the same entity (`get_issue_summary`, `get_issue_full`, `get_issue_history`) may be justified by context-budget profiles — but verify, do not assume.

---

## Common Mistakes

- **CRUD-mirror surface.** Tools named `get_/create_/update_/delete_<table>`. The agent has workflows, not tables. Symptom: the critic cannot write an agent-voice intent for a tool without using the word "the record." Fix: rename to workflow verbs; split `update_*` by the distinct effects it actually performs.
- **Implementation-voice description.** *"Updates the status column."* The model reads this and cannot tell what *workflow* it serves. Fix: lead with the effect-an-agent-cares-about, per the Consistency Gate's first requirement.
- **Unfillable parameters.** `expected_version`, `tenant_uuid`, `etag`, free-form `options: object`. The model fabricates or omits them. Fix: inject server-side from session/auth, return the value in the prior tool's result, or replace free-form objects with named typed properties.
- **Overlapping indistinguishable tools.** `search_issues` and `find_issues` and `query_issues`. The model picks one at random per turn. Fix: merge, or make each description state explicitly when to pick it over the others.
- **Free-text where an enum belongs.** `priority: string` invites `"P1"`, `"high"`, `"urgent"`, `"Critical"`. Fix: `enum`, so the model picks from your vocabulary.
- **Everything required.** A tool with eight required parameters is rarely callable. Fix: defaults and optionality; require only what has no sensible default and a clear source.
- **No negative scope.** A description that does not say what the tool is *not* for invites the adjacent misuse (`claim_issue` used to create issues). Fix: one negative-scope sentence.
- **Description grows stale silently.** The code changed; the description did not. The model now calls the tool based on a contract the server no longer honors. Fix: a golden-conversation regression test per tool (Consistency Gate) that fails when behavior drifts from description.
- **Parameter names from the database, not the conversation.** `usr_id`, `iss_pk`, `fk_proj`. The model carries forward whatever you returned; abbreviations and FK names add friction and fabrication risk. Fix: clear names that match what the tool *returns* elsewhere (`issue_id` everywhere, not `id` here and `issue_pk` there).

---

## Critic Checklist For This Sheet

When auditing a tool surface for API-design defects, produce **finding / severity / evidence** triples (per the router's Consistency Gate). Severity is one of *blocker / major / minor / nit*; evidence is the specific tool name, parameter, or description excerpt.

- [ ] **Every tool name is a workflow verb, not a CRUD/codebase verb.** A `get_/update_/delete_<table>` name is at least *major* — it predicts mis-fill and idempotency ambiguity. Evidence: the name plus the table it mirrors.
- [ ] **Every tool has an agent-voice intent in its description** (effect-an-agent-cares-about, not implementation). Implementation-voice is *major*. Evidence: the description's first sentence.
- [ ] **Every parameter is fillable from conversation / prior result / user request / training.** An unfillable parameter is a *blocker* (the tool cannot be reliably called). Evidence: the parameter name and why the model has no source for it.
- [ ] **No two tools are indistinguishable from their names + descriptions.** Overlapping pairs are *major*. Evidence: the two names and the missing disambiguation.
- [ ] **Closed vocabularies use `enum`; free-form `object` parameters are justified or eliminated.** *minor* to *major* depending on misuse blast radius.
- [ ] **`required` is minimal and `additionalProperties: false` is set** where fabrication risk exists. *minor* to *major*.
- [ ] **Each tool maps to one decision / one idempotency guarantee / one concurrency contract.** A multi-effect tool is *major* — it cannot satisfy the Consistency Gate's idempotency requirement. Evidence: the distinct effects bundled under one name.
- [ ] **Descriptions state preconditions, ordering, and negative scope** where they exist. Missing precondition on an ordered tool (`close` before `claim`) is *major*.

If the audit produces a short, clean findings list on a surface with more than a handful of tools, re-read the catalog as a prompt with no source code in hand — that is how the model reads it. Per the router's Anti-Overconfidence note: a critic that agrees with the architect is usually reading the surface the way it was written, not the way it will be called.

---

## Cross-References

- **Idempotency and concurrency contracts** for the tools you split here: [idempotency-and-atomicity.md](idempotency-and-atomicity.md). Naming is the front of a decision; the idempotency guarantee is the back of it — design them together.
- **Return-shape and context-budget discipline** (the other half of "the agent can use this"): [output-shape-and-pagination.md](output-shape-and-pagination.md). A perfectly named tool that returns 40k tokens still breaks the conversation.
- **Tool vs resource vs prompt vs sampling** — before naming a tool, confirm it should be a tool at all: [mcp-primitive-selection.md](mcp-primitive-selection.md).
- **Error envelopes** the tools here return on the recovery paths the descriptions promise: [error-envelopes-and-recovery.md](error-envelopes-and-recovery.md).
- **The full smell catalog** (overlapping-tools, tool-as-CRUD-mirror, parameter-the-agent-cannot-fill): [mcp-server-smells.md](mcp-server-smells.md).
- **Golden-conversation regression** that pins each tool's name/description/shape against drift: [testing-mcp-servers.md](testing-mcp-servers.md).
