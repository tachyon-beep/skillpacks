---
name: resources-prompts-sampling
description: Use when an MCP server exposes everything as a tool and the tool list is bloated with read-only getters, when a model burns a tool call to fetch context the host could have attached as a resource, when you need a user-invoked entry point but only have model-invoked tools, when a server needs the host to run inference (summarize, classify, draft) but you are tempted to embed your own LLM, when a tool blocks waiting for a value only the human can supply, when you cannot decide between a resource link and a tool result, or when you are auditing whether the four MCP primitives (tools, resources, prompts, sampling, elicitation) are each carrying the load they should — closing the "everything is a tool" failure from the resource/prompt/sampling/elicitation side.
---

# Resources, Prompts, Sampling, Elicitation — Beyond Tools

> Currency: written to MCP protocol revision **2025-11-25** (prior widely-deployed: 2025-06-18). JSON Schema dialect **2020-12**. Sampling supports tool calling (`tools`/`toolChoice`). Elicitation supports enum/multi-select/default-value/URL-mode. Resources support resource links and templates. Tools/resources/prompts/templates may carry `icons` metadata. Experimental **Tasks** utility exists for durable/deferred requests. Verify against `schema/2025-11-25/schema.ts` in the spec repo before shipping.

## The failure this sheet closes

[tool-api-design.md](tool-api-design.md) closes "everything is a tool" from the *tool* side: it teaches you to name tools for agent-visible effects and keep the surface small. This sheet closes the same failure from the *other* side. Even a perfectly-named tool surface is wrong if half those tools should not have been tools at all.

The MCP primitive set is **not** "tools, plus three things you can ignore." It is a deliberate split along two axes that the protocol enforces:

- **Who initiates the interaction** — the *model* (tools), the *user* (prompts), or the *server* (sampling, elicitation).
- **What the interaction costs the agent** — a tool call consumes a reasoning turn and a slot in the model's tool-selection attention; a resource is attached as context without a turn; a prompt is a user gesture; sampling and elicitation invert the call direction entirely.

```
                model-initiated        user-initiated        server-initiated
              ┌────────────────────┬───────────────────┬──────────────────────┐
 server→client│  TOOLS             │  PROMPTS          │  (resources are       │
 (S exposes)  │  model decides     │  user picks from  │   model-readable but  │
              │  to act            │  a menu, fills    │   NOT model-invoked:  │
              │                    │  args, sends      │   host attaches them) │
              ├────────────────────┴───────────────────┴──────────────────────┤
              │  RESOURCES — addressable context (URI), read not called        │
              ├────────────────────────────────────────────────────────────────┤
 client→server│  SAMPLING     server asks host to run inference (now with     │
 (S requests) │               tool-calling via tools/toolChoice)              │
              │  ELICITATION  server asks host to collect structured user input│
              │  ROOTS        server asks host which filesystem roots it may   │
              │               see (out of scope here; see transport sheet)    │
              └────────────────────────────────────────────────────────────────┘
```

The "everything is a tool" anti-pattern collapses this two-axis space onto a single point. Symptoms, all of which this sheet teaches you to recognize and fix:

- A `get_*` / `read_*` / `list_*` tool that takes no decision-relevant parameter and only fetches context the host could have attached. **That is a resource.**
- A multi-step workflow the *user* wants to trigger ("summarize this PR", "scaffold a migration") wired as a tool the *model* has to discover and decide to call. **That is a prompt.**
- A tool that calls `openai`/`anthropic` directly inside the server to summarize/classify/extract, embedding a model the host already has and the user already pays for. **That is sampling.**
- A tool that returns `{"error": "missing approval, ask the user"}` and relies on the agent to remember to relay it. **That is elicitation.**

The architect uses this sheet to place each interaction on the right primitive. The critic uses it to find interactions placed on the wrong one. They read the same four primitives; the architect asks "where does this belong?" and the critic asks "why is this a tool?"

---

## Resources — addressable context, not a getter tool

A **resource** is data the model can read, identified by a URI, exposed by the server and **attached to context by the host** — not invoked by the model. This is the single most under-used primitive and the source of the most common tool-bloat.

### The decision rule

Expose state as a **resource** when *all* of these hold:

1. It is **read-only** from the agent's perspective (reading it has no side effect worth a tool's idempotency declaration).
2. The agent's value comes from *having it in context*, not from *deciding to fetch it*. Project config, a schema, a style guide, the current sprint board, a file's contents — the agent rarely benefits from reasoning about *whether* to read these; it benefits from them being present.
3. It is **addressable** — there is a stable, namespaceable identifier (a URI) for it. `config://project/settings`, `file:///repo/README.md`, `db://issues/1234`.

Expose it as a **tool** instead when reading it *is a decision* — the agent must choose parameters that aren't known up front, the read is expensive enough that you want the model to gate it, or the "read" actually performs a query the agent composes (a search). A search is a tool; a *document* is a resource.

> Litmus test (from [mcp-server-smells.md](mcp-server-smells.md), `resource-that-should-be-a-tool` and its inverse): if you can name a *stable URI* for the thing and the agent never needs to *choose its arguments*, it is a resource that you have mis-modeled as a tool. If the agent must compose a query whose answer is unbounded, it is a tool, even if it "reads."

### Resource templates and resource links (2025-11-25)

- **Resource templates** parameterize a URI: `db://issues/{id}` declares a *family* of resources. The host can resolve `db://issues/1234` without the server enumerating every issue. Use templates when the resource space is large or open-ended and you do not want a `resources/list` that returns ten thousand entries.
- **Resource links** let a *tool result* reference a resource by URI instead of inlining its content. This is the bridge between the two primitives and directly serves the context-budget discipline of [output-shape-and-pagination.md](output-shape-and-pagination.md): a `search_issues` tool returns the matching URIs as resource links; the host fetches the bodies only for the ones the agent actually opens. The 50-result search no longer blows the budget by inlining 50 full issue bodies.

### Concrete example — config as a resource, search as a tool

A project-management MCP server. The wrong design exposes `get_project_config` as a tool. The right design splits along the decision-rule.

```python
# server.py — using the official Python SDK (mcp). Illustrative, not a full server.
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("project-mcp")

# --- RESOURCE: project config. Read-only, stable URI, agent benefits from
#     having it in context, never needs to "decide" to fetch it. ---
@mcp.resource("config://project/settings")
def project_settings() -> str:
    """Project conventions the agent should respect: label taxonomy,
    priority scale, definition-of-done. Attached to context by the host;
    the model does not spend a tool call to read it."""
    return json.dumps(load_settings(), indent=2)  # bounded by construction

# --- RESOURCE TEMPLATE: one issue, addressable by id. The agent reaches a
#     specific issue by URI rather than by a tool round-trip. ---
@mcp.resource("db://issues/{issue_id}")
def issue_resource(issue_id: str) -> str:
    """Full body of a single issue, by id. Addressable; read-only.
    Reached via a resource link returned by search_issues, or directly
    when the agent already knows the id."""
    return render_issue(get_issue(issue_id))  # bounded: one issue

# --- TOOL: search. The agent COMPOSES the query; the result set is
#     unbounded; this is a decision, not a document. Returns resource
#     LINKS, not inlined bodies — context-budget discipline. ---
@mcp.tool()
def search_issues(query: str, status: str | None = None, limit: int = 20):
    """Find issues matching a free-text query and optional status. Returns
    a bounded list of matches as resource links; open an issue's full body
    by reading its db://issues/{id} resource. Use this to LOCATE issues;
    do not use it to read an issue you already have the id for.

    Intent (agent-voice): locate the issues relevant to what I am working on,
      cheaply, without pulling every body into my context.
    Idempotency: read-only, no side effect; safe to call repeatedly.
    Context-budget: bounded — returns <= `limit` links (default 20), each a
      one-line title + URI, never the full body. May-be-truncated signaled
      via `truncated: true` + a refinement hint.
    Concurrency: safe to call concurrently with itself and all other tools.
    """
    rows = run_search(query, status=status, limit=min(limit, 50))
    return {
        "matches": [
            {"type": "resource_link", "uri": f"db://issues/{r.id}",
             "name": r.title, "description": f"{r.status} · {r.priority}"}
            for r in rows
        ],
        "truncated": len(rows) >= min(limit, 50),
        "refine_hint": "narrow with status= or a more specific query"
        if len(rows) >= min(limit, 50) else None,
    }
```

What this buys you: the tool list shrinks (no `get_config`, no `get_issue`), the agent never spends a turn re-reading config it already has, and a wide search returns 20 links (~a few hundred tokens) instead of 20 full bodies (~tens of thousands). Every Consistency-Gate clause is satisfiable on the tool because the *tool* is genuinely a tool; the resources are not forced through tool-shaped checks they do not need.

---

## Prompts — user-invoked entry points

A **prompt** is a templated message or workflow that the **user** selects and parameterizes. In a host UI it typically appears as a slash command, a menu item, or a button. The distinction that matters: a tool is *the model deciding to act*; a prompt is *the user deciding to start something* and handing the model a pre-built, server-authored opening.

### When a prompt beats a tool

Use a prompt when the trigger is a **human gesture**, not a model decision:

- "Summarize this pull request" — the user wants it now; the model should not have to *discover* a `summarize_pr` tool and decide to call it.
- "Scaffold a migration for table X" — a user-initiated workflow with a couple of arguments the user fills.
- "Run the incident-review template against this outage" — the server owns the *shape* of the review (the questions, the structure); the user supplies the outage.

The benefit is twofold. First, **discoverability for the human**: prompts surface in the host UI as things the user can do, where tools live in the model's head. Second, **server-owned quality**: the prompt template is authored by the server developer who knows the domain, so the user gets a good prompt without writing one, and the server can evolve the template without retraining anyone's habits.

### What a prompt is not

A prompt is **not** a tool the model calls. If your "prompt" only ever fires because the model decided to, you wanted a tool. A prompt is also not a place to hide side effects — a prompt *produces messages*; it does not mutate state. If selecting the prompt should write to a database, the prompt should produce a message that leads the model to call a (properly idempotent, properly-enveloped) tool.

### Concrete example — a user-invoked PR-review prompt

```python
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("project-mcp")

@mcp.prompt(title="Review a pull request")
def review_pr(pr_url: str, focus: str = "correctness") -> list[base.Message]:
    """User-invoked. Builds a structured PR-review opening the user
    triggers from the host UI — the user supplies the PR and a focus;
    the server owns the review shape (the checklist, the ordering, the
    output contract). Produces messages only; performs no side effect.
    `focus` is a server-defined enum surfaced to the user, not a free
    string the model has to guess."""
    return [
        base.UserMessage(
            f"Review the pull request at {pr_url}. Focus: {focus}.\n\n"
            "Proceed in this order, and use the project's tools to gather "
            "evidence rather than guessing:\n"
            "1. Read the diff (search_issues / file resources as needed).\n"
            "2. Check against config://project/settings definition-of-done.\n"
            "3. Report findings as: finding / severity (blocker|major|minor|nit)"
            " / evidence (file:line). No finding without evidence.\n"
            "4. State what you could NOT verify."
        )
    ]
```

The `focus` argument is a server-defined choice (the host can render it as a dropdown). The template encodes the *house review discipline* — severity + evidence, name what you couldn't verify — so every invocation inherits it. None of this would happen reliably if "review a PR" were a tool the model occasionally remembered to call its own way.

> Note: prompts can carry `icons` metadata in 2025-11-25 for nicer host rendering; this is cosmetic and never load-bearing for correctness.

---

## Sampling — the server asks the host to run inference

**Sampling** inverts the call direction: the *server* sends a `sampling/createMessage` request and the *host* runs an LLM and returns the completion. This exists so a server can use language-model capability **without shipping, hosting, or paying for its own model** — it borrows the host's, under the host's consent and budget.

### When to reach for sampling instead of a tool that calls an LLM

Use sampling when your server needs a model in the loop for a server-internal step:

- A `triage_issue` tool that wants the model to classify severity from free text.
- A resource renderer that wants to summarize a 50-page document down to a context-budget-safe abstract before returning it.
- An agentic sub-step inside the server that itself needs to call tools — 2025-11-25 sampling supports `tools` + `toolChoice`, so the server-requested inference can itself be a small tool-using loop.

The alternative — `import openai` (or `import anthropic`) inside the server and call a model directly — is almost always wrong for a distributable MCP server. It hardcodes a provider, requires the *server operator* to hold credentials, double-pays (the host already runs a model the user is paying for), and bypasses the host's consent and observability. Sampling routes the inference through the host so the user sees and approves it, the host's model and budget are used, and the server stays provider-agnostic.

> Trust boundary (Security Best Practices, 2025-11-25): sampling requires **explicit user consent** at the host. The protocol deliberately limits server visibility into the user's broader prompt context, and the host may refuse, redact, or modify a sampling request. **Design sampling as best-effort, never as a guaranteed function call.** A server that cannot tolerate the host declining a sampling request has an architecture bug, not a protocol limitation.

### Concrete example — triage via host sampling, not an embedded model

```python
import mcp.types as types
from mcp.server.lowlevel import Server

server = Server("project-mcp")

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.ContentBlock]:
    if name != "triage_issue":
        raise ValueError(f"unknown tool: {name}")

    body = arguments["body"]

    # Ask the HOST to run inference. We do not host a model; we borrow one.
    # toolChoice="none" because this is a pure classification — no tools.
    try:
        result = await server.request_context.session.create_message(
            messages=[
                types.SamplingMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=(
                            "Classify the severity of this issue as exactly one "
                            "of: blocker, major, minor, nit. Reply with the single "
                            f"word only.\n\nISSUE:\n{body}"
                        ),
                    ),
                )
            ],
            max_tokens=8,
            # tools=[...], tool_choice=... available in 2025-11-25 if the
            # sampled step itself needs to call tools. Here it does not.
        )
        severity = result.content.text.strip().lower()
        if severity not in {"blocker", "major", "minor", "nit"}:
            severity = "unknown"  # degrade; never trust the sampled output blindly
    except Exception:
        # Host may DECLINE or be unavailable. Sampling is best-effort.
        # Return a recoverable envelope, not a crash — see error sheet.
        return [types.TextContent(type="text", text=json.dumps({
            "ok": False,
            "error_class": "retry-with-changes",
            "message": "automatic triage unavailable; set severity explicitly "
                       "via the `severity` argument and call again",
        }))]

    return [types.TextContent(type="text", text=json.dumps({
        "ok": True, "severity": severity,
    }))]
```

Note three disciplines: the server **declares no model dependency**; the sampled output is **validated** before use (the host's model can return anything); and host refusal is handled with a **`retry-with-changes` envelope** (per [error-envelopes-and-recovery.md](error-envelopes-and-recovery.md)) rather than an exception, so the agent can recover by supplying severity itself.

---

## Elicitation — the server asks the host to collect structured user input

**Elicitation** lets the server pause and request **structured input from the user** mid-interaction, via `elicitation/create`. This is the right tool for "I need a value only the human can give, and I need it in a typed shape" — an approval, a missing required field, a choice among options, a confirmation before a destructive action.

The anti-pattern it replaces: a tool that returns `{"error": "need approval — please ask the user"}` and *hopes* the agent relays it, gets a free-text answer back, and reparses it. That is fragile (the model may paraphrase, forget, or hallucinate the answer) and untyped. Elicitation gives the host a schema to render as a real form and returns a typed result.

### 2025-11-25 elicitation features

- **Enum schemas**, titled and untitled, single- and multi-select — the user picks from server-defined options rather than free-typing.
- **Default values** for primitive fields.
- **URL-mode elicitation** — the server can direct the user to complete a step at a URL (e.g. an external authorization or a long form) and resume.

### When to elicit vs. when to make it a tool parameter

Elicit when the value is **genuinely the human's to supply and not inferable from context**: an irreversible-action confirmation, a credential the agent must not see, a business choice the agent has no authority to make. Do *not* elicit for values the agent already has or can derive — that just adds a round-trip and a consent fatigue tax. Per the Security Best Practices, elicitation (like sampling) is a consent surface; over-eliciting trains users to click through, which defeats the consent it exists to provide.

### Concrete example — confirm-before-destroy via elicitation

```python
import mcp.types as types
from mcp.server.lowlevel import Server

server = Server("project-mcp")

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.ContentBlock]:
    if name != "bulk_close_issues":
        raise ValueError(f"unknown tool: {name}")

    ids = arguments["issue_ids"]

    # Destructive + irreversible + the call is the agent's but the AUTHORITY
    # is the human's. Elicit a typed confirmation; do not infer consent.
    elicited = await server.request_context.session.elicit(
        message=f"Close {len(ids)} issues? This cannot be undone.",
        requestedSchema={
            "type": "object",
            "properties": {
                "confirm": {
                    "type": "string",
                    "title": "Confirm bulk close",
                    "enum": ["cancel", "close_all"],   # enum, not free text
                    "default": "cancel",                # safe default
                },
            },
            "required": ["confirm"],
        },
    )

    if elicited.action != "accept" or elicited.content.get("confirm") != "close_all":
        return [types.TextContent(type="text", text=json.dumps({
            "ok": False, "error_class": "fatal",
            "message": "bulk close cancelled by user; no issues were closed",
        }))]

    closed = bulk_close(ids)  # idempotent: closing a closed issue is a no-op
    return [types.TextContent(type="text", text=json.dumps({
        "ok": True, "closed": closed,
        "note": "re-running closes none already closed (no-op-after-first)",
    }))]
```

The confirmation is a typed enum with a safe default, not a parsed free-text "yes". The user can decline (`action != "accept"`), which the server handles as a clean `fatal` (no agent-side recovery — the human said no). The underlying close is idempotent so an agent retry after the confirmation does not double-act.

---

## Choosing the primitive — one decision table

Run a candidate interaction down this table; the first matching row wins.

| Question about the interaction | If yes → primitive |
| --- | --- |
| Is it read-only, addressable by a stable URI, and valuable just by being in context? | **Resource** (template if the space is large/open) |
| Does the *user* initiate it as a gesture, with the server owning the workflow shape? | **Prompt** |
| Does the *server* need a model to run a step (summarize/classify/draft/sub-agent)? | **Sampling** (validate output; tolerate host refusal) |
| Does the *server* need a typed value only the human can supply (approval/choice/credential)? | **Elicitation** (enum + safe default; never infer consent) |
| Does the *model* decide to take an action, compose a query, or cause an effect? | **Tool** (then run the full Consistency Gate) |

If two rows seem to match, you have probably conflated a *document* with a *query* (resource vs. tool) or a *human gesture* with a *model decision* (prompt vs. tool). Re-read the litmus tests above before forcing it onto a tool.

---

## Common mistakes

1. **Read-only getters shipped as tools.** `get_config`, `read_schema`, `list_labels` with no decision-relevant parameters. They bloat the tool list (every tool competes for the model's selection attention), and they waste a turn re-fetching context that could have been attached as a resource. *Fix:* make them resources; reserve tools for decisions and effects.

2. **Search results inlined instead of linked.** A `search_*` tool that returns 50 full bodies blows the context budget on the median query. *Fix:* return **resource links** and let the host fetch only opened items — this is the canonical resource/tool collaboration and the context-budget profile the gate requires.

3. **User workflows wired as tools.** "Summarize this PR" as a model-discovered tool means the user cannot reliably trigger it and the model summarizes in an ad-hoc shape each time. *Fix:* make it a **prompt** with a server-owned template and typed (enum) arguments.

4. **Embedding an LLM in the server (`import openai` / `import anthropic`).** Hardcodes a provider, forces the operator to hold credentials, double-pays, and bypasses host consent and observability. *Fix:* use **sampling** so the host's model runs the inference under the host's consent and budget; keep the server provider-agnostic.

5. **Treating sampling as a guaranteed function call.** The host may decline, redact, or modify a sampling request, and limits the server's view of the user's context by design. A server that crashes or hangs on refusal is broken. *Fix:* design sampling as **best-effort**, **validate** the returned content, and return a `retry-with-changes` or `fatal` envelope when the host declines.

6. **Asking for human input via tool-error-strings.** `{"error": "ask the user for approval"}` is untyped, unreliable, and reparses free text. *Fix:* use **elicitation** with a `requestedSchema` (enum + default for choices/confirmations) so the host renders a real form and returns a typed result.

7. **Over-eliciting / over-sampling.** Both are consent surfaces. Eliciting values the agent already has, or sampling for things a deterministic rule could decide, trains users to click through and defeats the consent the primitives exist to provide. *Fix:* elicit only the genuinely-human, not-inferable value; sample only when you genuinely need model capability.

8. **Prompts with side effects.** A "prompt" that writes to a database when selected has overloaded a message-producing primitive with mutation. *Fix:* prompts produce **messages**; route any state change through a properly idempotent, properly-enveloped **tool** that the resulting messages lead the model to call.

9. **Confusing a document with a query.** Modeling a free-text search as a resource (it has no stable URI; its result set is unbounded) or a single addressable record as a tool (it has a perfectly good URI and needs no composed query). *Fix:* document → resource (template if parameterized); composed/unbounded query → tool.

10. **Forgetting the primitive choice is itself a versioned surface.** Promoting a tool to a resource, or splitting a tool into a prompt + tool, changes what the agent sees. *Fix:* treat it as a non-backward-compatible change and **bump the server capability** (see [schema-versioning-and-drift.md](schema-versioning-and-drift.md)); do not silently re-shape the surface under a model that learned the old one.

---

## For the critic — how to audit this corner of the surface

Findings carry **severity** (`blocker` / `major` / `minor` / `nit`) and **evidence** (the tool/resource name, the schema excerpt, the return-shape sample, the conversation fragment).

- **Enumerate the tool list and flag every read-only getter.** For each `get_*` / `read_*` / `list_*` tool, ask: stable URI? no decision-relevant parameter? If both, it is a `resource-that-should-be-a-tool` smell. *Evidence:* the tool name + its (empty or trivial) parameter schema. *Severity:* usually `minor` per tool, escalating to `major` when getters dominate the list and crowd out real tools from the model's selection attention.
- **Check searches for inlining.** Does any `search_*` / `query_*` / `find_*` tool return full bodies? *Evidence:* a return-shape sample on a realistic result count, with an estimated token size. *Severity:* `major` (it is a latent context-budget incident), `blocker` if the median input already exceeds a typical budget.
- **Look for user workflows trapped as tools.** A tool whose natural trigger is a human gesture, not a model decision, and which is rarely or never called because the model does not discover it. *Evidence:* the tool description (does it read as "the user wants…"?) + call-frequency telemetry if available. *Severity:* `minor`–`major`.
- **Grep the server for embedded model SDKs.** `import openai`, `import anthropic`, `genai`, etc. inside tool handlers. *Evidence:* the import + the call site. *Severity:* `major` for a distributable server (provider lock-in, operator-credential burden, double-pay, consent bypass); name **sampling** as the fix.
- **Audit sampling call sites for refusal handling and output validation.** Does the code assume the completion is well-formed? Does it crash/hang on host refusal? *Evidence:* the `create_message` call site + the absence of validation/refusal branches. *Severity:* `major` (a host that declines is in-spec and will happen).
- **Audit human-input paths.** Any tool returning an "ask the user" error string, or reparsing free-text answers, instead of eliciting a typed value. *Evidence:* the error envelope + the reparse logic. *Severity:* `major` for confirmations on **irreversible** actions (untyped consent on a destructive op is a `blocker`).
- **Check for over-eliciting / over-sampling.** Consent surfaces fired for values the agent already has or a rule could decide. *Evidence:* the elicit/sample call + the available context that made it unnecessary. *Severity:* `minor` (consent fatigue), `major` if it gates a hot path.
- **Verify primitive re-shapes bumped the capability.** Did a recent change move an interaction between primitives without a capability bump? *Evidence:* the schema diff across server versions. *Severity:* `major` (silent surface drift; see [schema-versioning-and-drift.md](schema-versioning-and-drift.md)).

Record any **architect↔critic disagreement** with both positions and the resolution. A pass over this corner that finds zero misplaced primitives on a non-trivial surface is, per the router's Anti-Overconfidence, evidence the critic is reading the surface the way the architect wrote it — re-run with a fresh frame before concluding the primitives are well-placed.

---

## Cross-references

- [tool-api-design.md](tool-api-design.md) — the *tool* side of "everything is a tool"; agent-voice intent statements, granularity. Read with this sheet, not instead of it.
- [output-shape-and-pagination.md](output-shape-and-pagination.md) — resource links are the primary mechanism for keeping search/list results inside the context budget.
- [error-envelopes-and-recovery.md](error-envelopes-and-recovery.md) — sampling refusal and elicitation decline must be returned as `retry-with-changes` / `fatal` envelopes, not exceptions.
- [idempotency-and-atomicity.md](idempotency-and-atomicity.md) — the tool that an elicited confirmation gates must still be idempotent under retry.
- [schema-versioning-and-drift.md](schema-versioning-and-drift.md) — moving an interaction between primitives is a non-backward-compatible surface change and bumps the capability.
- [authentication-and-trust.md](authentication-and-trust.md) — sampling and elicitation are consent surfaces; trust and scoping interact with both.
- [mcp-server-smells.md](mcp-server-smells.md) — `resource-that-should-be-a-tool`, `tool-that-should-be-a-resource`, and the embedded-model and tool-error-as-prompt smells.
