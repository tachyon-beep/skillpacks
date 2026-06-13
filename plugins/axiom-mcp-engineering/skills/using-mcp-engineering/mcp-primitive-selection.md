---
name: mcp-primitive-selection
description: Use when you cannot decide whether a thing should be a tool, a resource, a prompt, or a sampling request; when every capability on your MCP server has become a tool and the tool list is bloating the agent's context; when you are about to wrap a read-only lookup as a tool, expose a config file as a tool, hard-code a workflow the user should trigger, or call an external LLM from inside a tool handler instead of asking the host to sample; when an agent burns turns calling a "get_X" tool that returns context it could have been handed; when resource subscriptions, resource links, prompt templates, elicitation, or server-initiated sampling are on the table and you need the decision rule rather than the vibe.
---

# MCP Primitive Selection

**Every capability you add to an MCP server is one of four kinds of thing, and the kind is not a stylistic choice — it is determined by *who initiates the interaction* and *what the interaction is for*. Get the kind wrong and the agent pays for it on every turn.**

The Model Context Protocol gives a server four ways to participate in an agent's context. Three flow server→client (**tools**, **resources**, **prompts**); one flows client→server (**sampling**), with two more control-plane client→server primitives (**roots**, **elicitation**) that shape how the others behave. The architect's job is to put each piece of state and each interaction on the *right* primitive. The critic's job is to find the pieces that are on the wrong one — overwhelmingly, the wrong one is "tool", because "everything is a tool" is the path of least resistance and the single most common structural defect in deployed MCP servers.

This sheet is the decision rule. It serves the **architect** (which primitive should this be?) and the **critic** (this is a tool; should it be?) over the same four-primitive corpus, with different epistemics.

> Currency note: written against MCP protocol revision **2025-11-25** (prior widely-deployed revision 2025-06-18). JSON Schema dialect is **2020-12**. Tools now carry `structuredContent` + declared `outputSchema`; resources support **resource links** (a tool result can reference a resource by URI); sampling now supports **tool calling** (`tools` / `toolChoice`); elicitation supports enum schemas, default values, and URL-mode; tools/resources/templates/prompts can expose **icons**. There is an experimental **Tasks** utility for durable long-running requests. Where this sheet says "tool result," assume the structured-content path unless noted.

---

## The four primitives, by initiator and purpose

The whole decision collapses to two axes: **who initiates** and **what it is for**.

| Primitive | Direction | Initiated by | It is for… | One-line test |
| --- | --- | --- | --- | --- |
| **Tool** | server→client | the **model** (autonomously, mid-reasoning) | a **model-invoked action** — usually with a side effect, or a computation the model chooses to run | "The agent *decides* to do this as a step in its reasoning." |
| **Resource** | server→client | the **application/user** (attached to context) | **model-readable context** — data the host puts in front of the model | "The agent should be able to *read* this; it does not *act* by reading it." |
| **Prompt** | server→client | the **user** (explicitly, e.g. a slash command) | a **user-invoked template** — a parameterized message/workflow the user triggers | "A *human* picks this from a menu to start something." |
| **Sampling** | client→server→client | the **server** (asks the host to infer) | **server-asks-host-to-infer** — the server needs an LLM completion and asks the host's model to produce it | "*My server* needs to think, and I want to borrow the host's model to do it." |

Two control-plane primitives shape the above and are almost never the answer to "what kind of thing is this capability," but you must know they exist so you do not reach for a tool to do their job:

- **Roots** (client→server): the host tells the server which filesystem/URI boundaries it may operate within. Not a place to put your data; a place to *learn the agent's scope*.
- **Elicitation** (client→server, host-mediated): the server asks the host to **collect a structured value from the user mid-interaction** (titled/untitled enum schemas, single/multi-select, default values, URL-mode). This is how you ask the user a question *without* turning it into a tool parameter the agent has to hallucinate.

The decision rule, stated once: **if the model autonomously decides to invoke it, it is a tool; if the model should merely read it, it is a resource; if a human triggers it from a menu, it is a prompt; if your server needs an inference, it is sampling; if your server needs a value from the user, it is elicitation.** Everything below is this rule applied to the cases where it is non-obvious.

---

## Decision procedure (run top to bottom; stop at the first match)

For each piece of state or each interaction you are about to expose, walk this in order. The ordering matters: the early checks divert the things that should *not* be tools, leaving tools as the residue, not the default.

```
1.  Is your SERVER the thing that needs an LLM completion (to summarize,
    classify, draft, or reason over something it holds)?
        → SAMPLING.  Do NOT call an external LLM from inside a tool handler.
          Ask the host to infer; the user stays in control of which model,
          cost, and consent.

2.  Do you need a structured VALUE FROM THE USER mid-interaction
    (a choice, a confirmation, a missing field the agent cannot know)?
        → ELICITATION.  Do NOT add it as a tool parameter and hope the
          agent fills it; do NOT spawn a "ask_user" tool.

3.  Is this a workflow a HUMAN would deliberately start from a menu
    (a slash command, a "/triage", a "/write-postmortem")?
        → PROMPT.  The user picks it; it expands into a templated message
          (optionally with arguments and embedded resources).

4.  Is this DATA the model should be able to READ as context, where
    reading it has no side effect and the model is not "taking an action"
    by reading it (a file, a record, a schema, a config, a doc set)?
        → RESOURCE  (or a resource template for parameterized URIs).
          If the model frequently must FETCH it by calling something, that
          is a smell that it wants to be context, not a tool round-trip.

5.  Does the model autonomously DECIDE to invoke this as a step in its
    reasoning — to cause an effect, mutate state, or run a computation
    it chose to run?
        → TOOL.  (Now go uphold the Consistency Gate: intent statement,
          idempotency, context-budget profile, error envelope,
          concurrency contract, golden test.)

If two branches both seem to fit, you have probably conflated "the model
can read it" (resource) with "the model acts by causing it" (tool). Split
them: expose the data as a resource AND the action as a tool, and let a
tool RESULT carry a resource LINK to the data.
```

The deliberate consequence of running steps 1–4 before step 5 is that **tools are what is left over** after you have removed the inferences (sampling), the user values (elicitation), the user-triggered workflows (prompts), and the readable context (resources). If you start at step 5, everything is a tool. That is the failure mode this sheet exists to close.

---

## The hardest call: tool vs resource

Steps 1–3 are usually unambiguous. The 80% of real disagreement is **tool vs resource**, because both are server→client and both can "give the model some data." The distinguishing question is **who initiates and whether reading is an action**:

- A **resource** is *handed to the model* by the host/user. It sits in context. Reading it is not a decision the model narrates ("I will now read the file"); it is just *present*. Good resources: the contents of a file the user opened, a database schema, a project's README, a config the agent needs to reason about, a log file, an API spec. Resources have **URIs**, can be **subscribed** to for change notifications, can be **templated** (`db://{table}/schema`), and crucially **do not cost a tool-call round-trip** — the host can attach them directly.
- A **tool** is *invoked by the model* as an action. `get_weather(city)` is a tool because the model decides, mid-reasoning, that it needs the weather and chooses to fetch it; the result is computed on demand and is parameterized by the model's choice. The boundary case: a read-only lookup. `read_file(path)` *can* be a tool — but if the host already knows which files matter (via roots), those files should be **resources** the host attaches, and the tool, if it exists at all, is for the long tail the host could not pre-attach.

Operational rule of thumb:

> **If the host can know it should be in context without the model asking, it is a resource. If the model must *decide* it needs the thing, parameterized by something only the model knows at that moment, it is a tool.**

Two further signals:

- **Resource link, not resource dump in a tool result.** Since 2025-11-25 a tool result can return a *resource link* (a URI) rather than inlining a 50 KB payload. So "the agent calls `search_docs` and gets back the matching documents" is correctly modeled as: a **tool** (`search_docs`, the model's decision to search) whose **result carries resource links** to the matching docs (resources, attached on demand, with their own context-budget profile). This is the canonical fix for the "tool that should be a resource" smell when the access is genuinely model-initiated but the payload is genuinely context.
- **Subscriptions belong to resources.** If consumers want to be *notified when data changes* (a file changed on disk, a build finished), that is `resources/subscribe`, not a polling tool the agent has to remember to call.

---

## "Everything is a tool" — the anti-pattern this sheet exists to close

This is the dominant structural defect in shipped MCP servers. It arises because tools are the most general primitive (the model can call them, they can do anything) and because every SDK's "hello world" is a tool. Symptoms, each with its correct primitive:

| You wrote a tool for… | It should be a… | Why the tool form hurts the agent |
| --- | --- | --- |
| `get_config()` returning a static config file | **resource** (`config://app`) | Costs a tool-call round-trip every turn the agent needs it; pollutes the tool list; the host could have attached it once. |
| `read_schema(table)` over a known DB | **resource template** (`db://{table}/schema`) | The host knows the tables (or can list them); the agent should not spend a turn fetching schema it could be handed. |
| `summarize(text)` that calls OpenAI inside the handler | **sampling** | Steals model choice, cost, and consent from the user; double-bills; the host's model is right there. (See worked example 2.) |
| `ask_user(question)` | **elicitation** | The agent has to *invent* the question and *parse* a free-text answer; elicitation gives a typed schema and host-mediated consent. |
| `run_triage_workflow()` the human always starts | **prompt** | A workflow a human deliberately initiates should be a slash-command-style prompt, not a tool the model might fire on its own. |
| `list_files()` + `read_file()` over the project root | **resources via roots** | The host already knows the roots; pre-attach the relevant files; keep a `read_file` tool only for the long tail. |

The cost of "everything is a tool" is concrete and compounding: **every tool is a prompt fragment in the agent's context on every turn.** A 40-tool server where 25 of the tools are disguised resources or prompts spends thousands of tokens per turn describing capabilities that did not need to be tools, crowds the model's attention, and multiplies the surface the critic must audit. The fix is not "fewer tools for aesthetics" — it is *putting each thing on the primitive that matches its initiator.*

---

## When a tool, a resource, AND a prompt all touch the same domain

A mature server frequently exposes the same domain object three ways, correctly:

- The **data** as a **resource** (`issue://PROJ-123`) — readable context, subscribable, URI-addressable.
- The **actions** as **tools** (`claim_issue`, `close_issue`) — model-decided effects, each with the full Consistency Gate apparatus.
- The **human-started workflow** as a **prompt** (`/triage-issues`) — a templated message the user picks that may *embed the resource* and *suggest the tools*.

This is not duplication; it is correct factoring. The smell is the opposite: a single `issue(action, id, ...)` mega-tool that does reads, writes, and workflow dispatch through a mode flag — that collapses three primitives into one and forces the agent to encode the primitive choice as a parameter.

---

## Sampling and elicitation: the client→server primitives people forget

These two are the most under-used and the most mis-substituted-by-tools.

**Sampling** is "my server wants to think." Your tool handler is parsing a 2,000-row CSV and wants a natural-language summary, or classifying a support ticket, or drafting a commit message. The wrong move is to embed an LLM client in the handler. The right move is `sampling/createMessage`: you hand the host a message list and (since 2025-11-25) optionally `tools`/`toolChoice`, and the host runs *its* model with *the user's* consent, model selection, and budget. The host can decline. Your server gets the completion as a result. This keeps the human in the loop on cost and model choice, which is a core security principle of the protocol — the user consents to inference, not just to tool calls.

**Elicitation** is "my server needs a value from the human." Mid-tool-execution you discover you need a choice the agent cannot know (which of three matching records did the user mean? confirm the destructive delete? supply an API region?). The wrong move is a free-text tool parameter the agent hallucinates, or a separate `ask_user` tool. The right move is `elicitation/create` with a typed schema — enum (titled or untitled, single or multi-select), defaults for primitives, or URL-mode for "go authorize here and come back." The host renders it, the user answers, you get a validated value. This is also the correct home for **consent on side effects** that the protocol says must be explicit.

Both are **server-initiated**, which is exactly why people miss them: the mental model "the server only responds" makes you reach for a tool. The server can ask. Let it.

---

## Common mistakes

- **Defaulting to tool.** Treating "tool" as the base case and only reaching for resource/prompt/sampling when forced. Run the decision procedure top-down; tools are the residue, not the default. *(Critic: severity `major` when a read-only, side-effect-free lookup of host-knowable data is a tool; `blocker` when an LLM call lives inside a tool handler instead of sampling.)*
- **Resource dumped inline in a tool result.** Returning a 40 KB document body as the tool's text content instead of a resource link. Blows the context budget (see [output-shape-and-pagination.md](output-shape-and-pagination.md)) and re-fetches every call. Return a **resource link**; let the host attach the body on demand. *(Critic: `major`; evidence is the return-shape excerpt and a token estimate on a realistic input.)*
- **`ask_user`/`confirm` as a tool.** Forcing the agent to author the question and parse a free-text reply, with no typed schema and no host-mediated consent. Use **elicitation**. *(Critic: `major`; evidence is the tool and the absence of an elicitation schema for a value only the human can supply.)*
- **Calling an external LLM inside a tool handler.** Bypasses the host's model selection, cost accounting, and the user's consent-to-inference; double-bills; often violates the "explicit consent for sampling" security principle. Use **sampling**. *(Critic: `blocker`; evidence is the LLM SDK import in the handler.)*
- **Mega-tool with a `mode`/`action` parameter** (`issue(action="claim"|"close"|"read")`). Collapses distinct primitives and distinct effects into one tool, forcing the agent to encode the primitive choice as a parameter and defeating per-action intent statements, idempotency guarantees, and concurrency contracts. Split into discrete tools (and a resource for the read). *(Critic: `major`.)*
- **Static config / schema / README as a tool.** A per-turn round-trip for data the host could attach once as a **resource** (or **resource template** for parameterized URIs). *(Critic: `minor`–`major` depending on call frequency and payload size.)*
- **Workflow the human always starts modeled as a tool.** The model may then fire it autonomously when the user did not mean to. If a human picks it from a menu, it is a **prompt**. *(Critic: `minor`.)*
- **Forgetting subscriptions exist.** Building a polling tool (`check_if_build_done`) the agent must remember to call, instead of a subscribable **resource**. *(Critic: `minor`.)*
- **Ignoring roots.** Re-implementing "what files/dirs am I allowed to touch" as tool parameters the agent supplies, instead of reading the host-provided **roots** and pre-attaching the relevant files as resources. *(Critic: `minor`.)*
- **Primitive choice with no stated rationale.** Shipping a capability without recording *why* it is the primitive it is. The architect-critic disagreement that catches a mis-classified primitive cannot happen if the classification was never made explicit. *(Critic: `nit`, but it is the nit that hides the `blocker`.)*

---

## Worked example 1 — issue tracker: tool + resource + prompt over one domain

A server fronting an issue tracker. The naive build makes everything a tool: `get_issue`, `list_issues`, `update_issue`, `get_issue_schema`, `run_triage`. Below is the correct factoring across primitives, in a Python MCP SDK style.

```python
from mcp.server.fastmcp import FastMCP
from mcp.types import ResourceLink

mcp = FastMCP("issue-tracker")

# --- RESOURCE: the data the model should be able to READ as context.
#     URI-addressable, attachable by the host without a tool round-trip,
#     subscribable for change notifications. NOT a tool.
@mcp.resource("issue://{issue_id}")
def issue_resource(issue_id: str) -> str:
    """The full record for one issue, as readable context."""
    return render_issue_markdown(load_issue(issue_id))   # bounded by construction

# A resource TEMPLATE for the schema — host-knowable, never a per-turn tool call.
@mcp.resource("issue-tracker://schema")
def schema_resource() -> str:
    """Field definitions and allowed status transitions."""
    return SCHEMA_MARKDOWN

# --- TOOL: a model-INITIATED action with a side effect.
#     Intent in AGENT VOICE (effect, not implementation). Idempotency declared.
#     Result carries a resource LINK rather than inlining the record.
@mcp.tool()
def claim_issue(issue_id: str, agent_id: str) -> dict:
    """Take ownership of an issue so no other agent works it in parallel.

    Idempotency: requires-claim-lease. Calling again as the SAME agent_id is a
    no-op-after-first (re-confirms your lease). Calling as a DIFFERENT agent_id
    fails-second-call with a retry-with-changes error naming the current holder.
    Concurrency contract: serialised on issue_id via an atomic conditional claim.
    Context budget: bounded — returns a status line plus one resource link.
    """
    holder = try_claim(issue_id, agent_id)          # atomic compare-and-set
    if holder != agent_id:
        return {
            "isError": True,
            "error": {
                "class": "retry-with-changes",
                "message": f"issue {issue_id} is already claimed by {holder}",
                "recovery": "claim a different issue, or wait for release",
            },
        }
    return {
        "structuredContent": {"issue_id": issue_id, "claimed_by": agent_id},
        "content": [ResourceLink(uri=f"issue://{issue_id}").model_dump()],
    }

# --- PROMPT: a HUMAN-initiated workflow, picked from a menu (slash command).
#     Templated message; may embed a resource and suggest the tools above.
@mcp.prompt()
def triage_issues(label: str = "needs-triage") -> str:
    """User-invoked: walk open issues with a label and propose actions."""
    return (
        f"Triage open issues labelled '{label}'. For each, read its "
        f"issue:// resource, then propose claim_issue / close_issue actions. "
        f"Do not claim anything without confirming with me first."
    )
```

What the factoring buys: the issue *record* costs zero tool calls to read (resource), the *action* carries the full Consistency Gate apparatus (intent, idempotency, concurrency, budget, error envelope), and the *workflow* fires only when the human picks it (prompt) — the model will not autonomously run a triage sweep. The single `update_issue` mega-tool of the naive build is gone.

---

## Worked example 2 — summarization: sampling, not an LLM call in the handler

A server exposes `summarize_logs(path)`. The naive build calls a third-party LLM inside the handler. That steals model choice, cost control, and consent from the user, double-bills, and breaks the protocol's "explicit consent for sampling" principle. The correct build asks the **host** to infer.

```python
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("log-tools")

@mcp.tool()
async def summarize_logs(path: str, ctx: Context) -> dict:
    """Summarize a log file into a short incident-relevant digest.

    Intent (agent voice): turn a long log into a digest I can reason over
    without reading every line.
    Idempotency: no-op semantics — read-only, repeatable, safe to retry.
    Concurrency: safe to call concurrently with itself and all other tools.
    Context budget: bounded — the digest is capped at ~400 tokens by the
    sampling maxTokens; the raw log is NEVER returned inline.
    Errors: retry-safe on transient read failure; fatal if path is outside
    the host-provided roots (the server has no permission and cannot recover).
    """
    log_text = read_within_roots(path)          # respects host-provided roots

    # SAMPLING: borrow the host's model. The host chooses which model, applies
    # the user's consent and budget, and may decline. We do NOT import an LLM.
    result = await ctx.session.create_message(
        messages=[{
            "role": "user",
            "content": {
                "type": "text",
                "text": f"Summarize these logs into <=8 bullets, "
                        f"incidents first:\n\n{log_text}",
            },
        }],
        max_tokens=400,
    )
    return {"structuredContent": {"summary": result.content.text}}
```

Why this is the right primitive: the inference happens on the **host's** model under the **user's** consent and budget; the server never holds an API key or makes a model choice on the user's behalf; and if the user has not granted sampling, the host declines cleanly rather than the server silently billing a third party. The raw log stays out of the context budget — only the bounded digest crosses the boundary.

---

## Architect vs critic over this sheet

- **Architect** asks, per capability: *which primitive does the initiator and purpose dictate?* — and records the answer so it can be challenged. The architect's failure mode is the tool default: shipping resources, prompts, and sampling as tools because the SDK made tools easy.
- **Critic** asks, per tool: *should this be a tool at all?* — and runs the "everything is a tool" table against the whole surface. The critic's failure mode is reading the surface the way the architect built it and waving through a `summarize()` that calls an external LLM, or a `get_config()` that should be a resource. Every finding carries **severity** (`blocker`/`major`/`minor`/`nit`) and **evidence** (the tool name, the handler import, the return-shape excerpt, the token estimate).

If the architect and critic agree that every capability is correctly classified on the first pass of a non-trivial server, the critic did not run the table. Expect at least one reclassification on any surface with more than a handful of tools.

---

## See also

- [tool-api-design.md](tool-api-design.md) — once you have decided a thing IS a tool, this is how to name, scope, and shape it.
- [resources-prompts-sampling.md](resources-prompts-sampling.md) — the deep dive on the three non-tool primitives this sheet routes to.
- [output-shape-and-pagination.md](output-shape-and-pagination.md) — context-budget profiles, resource links vs inline payloads, truncation policy.
- [idempotency-and-atomicity.md](idempotency-and-atomicity.md) — once a thing is a tool with a side effect, this is the idempotency/claim-lease/concurrency apparatus the Consistency Gate demands.
- [mcp-server-smells.md](mcp-server-smells.md) — `resource-that-should-be-a-tool`, `tool-that-should-be-a-resource`, and the rest of the catalog.
