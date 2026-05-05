
# Agentic Patterns and MCP

## Context

You're building an LLM application that calls tools, runs agentically, or orchestrates multiple model invocations. Common mistakes:
- **Tool sprawl** — exposing 30+ tools to the model and watching it pick wrong ones
- **No structured output discipline** — parsing prose into JSON with regex
- **Prompt-injection-via-tool-results** — treating retrieved content as trusted instructions
- **Infinite loops or runaway iteration** — no max-iteration safeguard
- **Multi-agent over-engineering** — ten coordinated agents where one prompt would do
- **Treating MCP as "just another protocol"** — ignoring tool-use discipline because the transport changed
- **Hardcoding model IDs** in agent definitions that go stale every quarter

**This skill provides patterns for tool use, structured outputs, the Model Context Protocol (MCP), multi-agent orchestration, and the security/reliability discipline that keeps agents from going off the rails.**

This sheet uses the **capability-tier vocabulary** from [reasoning-models.md](reasoning-models.md) (frontier-reasoning / frontier-general / fast-cheap / on-device). Read that sheet first if you haven't.


## Tool-Use Loop Anatomy

The core loop is the same across providers:

```
┌─────────────┐
│   Model     │ ◀──── system + user + history
└──────┬──────┘
       │ tool_call(name, args)
       ▼
┌─────────────┐
│ Tool runner │ executes, returns result (or error)
└──────┬──────┘
       │ tool_result(content)
       ▼
┌─────────────┐
│   Model     │ ◀──── ... + tool_result
└──────┬──────┘
       │ tool_call OR final_message
       ▼
   continue or finish
```

Three things make this work:

1. **A tool schema** the model can read (name, description, JSON-schema arguments).
2. **A runner** that calls the actual function with the model-supplied arguments.
3. **A loop** that feeds results back until the model emits a final message or hits a stop condition.

### Provider differences

- **OpenAI** — tools defined in `tools` array, calls returned as `tool_calls` on the message; each call has `id`, `function.name`, `function.arguments` (a JSON string). Results returned as `role: "tool"` messages keyed by `tool_call_id` ([OpenAI function-calling guide](https://platform.openai.com/docs/guides/function-calling)).
- **Anthropic** — tools defined in `tools` array, calls returned as `tool_use` content blocks; results returned as `tool_result` content blocks in a user message keyed by `tool_use_id` ([Anthropic tool-use docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)).
- **Google Gemini** — function declarations in `tools.functionDeclarations`; calls in `functionCall`, results in `functionResponse` ([Gemini function-calling guide](https://ai.google.dev/gemini-api/docs/function-calling)).

The shapes differ, the discipline doesn't. Treat the wire format as plumbing; spend your design budget on the loop semantics.

### Parallel tool calls

OpenAI, Anthropic, and Gemini all support parallel tool calls — the model can request several tools in one turn and you execute them concurrently before returning results. This is a real latency win for independent calls (e.g., three lookups in parallel) but introduces three trap doors:

- **Order-dependent tools.** If tool B depends on tool A's result, parallel execution breaks. Document tool dependencies; consider returning a "must-be-sequential" hint in the schema description.
- **Partial failure.** If one of three parallel calls errors, you must still return *all three* results (one as an error) or the model loses track. Don't drop failures.
- **Cost compounding.** Parallel != cheaper; it's faster, not less spent.

### Max-iteration safeguards

**Always** cap loop iterations. A reasonable default is 10–20 turns for production; lower for cheap-tier models, higher for frontier-reasoning planners. On the cap:

```python
for turn in range(MAX_TURNS):
    response = model.invoke(messages)
    if not response.tool_calls:
        return response
    messages.append(response)
    for call in response.tool_calls:
        result = run_tool(call)
        messages.append(tool_result_message(call.id, result))
else:
    raise MaxIterationsExceeded(turn)
```

Without the cap, a model in a confused state can rack up hundreds of turns before someone notices. (See *infinite loops* under Anti-Patterns.)

### Error recovery

When a tool errors, return the error *to the model* with structure: error type, message, and (when safe) a hint. Reasoning-tier models recover well from this; chat-tier models may need a "retry with corrected arguments" hint in the prompt. Don't silently swallow errors and re-prompt — the model needs the signal.

> **Cross-ref:** for which tier to put at the planner vs worker layer in tool-using agents, see [reasoning-models.md](reasoning-models.md) "Reasoning Models in Agent Loops."


## Structured Outputs

Tool calls are one form of structured output. There are also direct structured-output APIs for "give me JSON matching this schema" without a tool round-trip.

### OpenAI Structured Outputs

Pass `response_format={"type": "json_schema", "json_schema": {...}, "strict": true}` ([OpenAI structured-outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)). With `strict: true`, the API guarantees output conforms to the schema (constrained decoding under the hood). The legacy `{"type": "json_object"}` mode only guarantees valid JSON syntax, not schema adherence — prefer `json_schema` strict mode for new code.

Schema constraints under strict mode (verify in current docs): all properties required, `additionalProperties: false` typically required, limited recursion, no `oneOf` / `anyOf` in some positions. The error from passing an invalid schema is loud and at API time — fix it once, then trust it.

### Anthropic — tool_use as structured output

Anthropic's path to structured output is to define a tool whose `input_schema` *is* the desired output schema, then force the model to call that tool. The model's `tool_use` block carries the structured payload. Use `tool_choice: {"type": "tool", "name": "<your_tool>"}` to force the call. This is the canonical Anthropic pattern; it works because tool-use blocks are constrained-decoded against the tool's input schema ([Anthropic tool-use docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)).

### Gemini — `responseSchema` and function-call mode

Gemini supports `responseMimeType: "application/json"` plus `responseSchema` for structured output, and function calling for tool-use-as-structured-output ([Gemini structured output docs](https://ai.google.dev/gemini-api/docs/structured-output)).

### Open-source / framework-level

- **[Outlines](https://github.com/dottxt-ai/outlines)** — constrained decoding library; supports JSON schema, regex, context-free grammars; works with local models (vLLM, Hugging Face) and some hosted APIs.
- **[Instructor](https://github.com/jxnl/instructor)** — Python library that wraps OpenAI / Anthropic / Gemini SDKs and validates outputs against Pydantic models, with retry-on-validation-failure built in.
- **JSON-schema-mode in vLLM / SGLang** — server-side constrained decoding for local deployment.

### When to use which

| Situation | Choice |
|---|---|
| OpenAI hosted, want guaranteed schema | `response_format=json_schema, strict=true` |
| Anthropic hosted, want structured payload | Single tool with `tool_choice` forced |
| Gemini hosted, want JSON | `responseMimeType=application/json` + `responseSchema` |
| Local model (vLLM, Llama.cpp) | Outlines or server-side JSON-schema mode |
| Need Pydantic ergonomics, multi-provider | Instructor |
| Schema is complex, hosted strict mode rejects it | Simplify schema (flatten unions); fallback to tool_use; or do post-validation with retries |

> **Anti-pattern:** parsing JSON out of prose with regex. If you need structured output, use a structured-output API. If you can't, use a constrained-decoding library. "Just regex it" produces silent corruption on edge cases.


## MCP — Model Context Protocol

MCP standardizes how LLM applications expose tools, prompts, and resources to model clients. It's a client-server protocol, not a model API.

### Spec source

The canonical spec lives at **[modelcontextprotocol.io](https://modelcontextprotocol.io)**. The current spec revision (verify at [modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification/)) defines the message format, capability negotiation, and transports.

### Architecture

- **MCP server** — exposes tools, prompts, and/or resources. Runs as a process (local or remote).
- **MCP client** — embedded in the host application (Claude Desktop, Cursor, Zed, custom agent). Connects to one or more servers, surfaces capabilities to the model.
- **Host application** — the LLM-using app that bridges between the model and the MCP client(s).

Capabilities are discovered at connection time; the model sees only what the host chooses to expose. This separation is the security primitive — see *prompt-injection-via-tool-results* under Anti-Patterns.

### Transports

Verify the current state at [modelcontextprotocol.io](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports). As of the **2025-03-26 spec revision**, MCP introduced **Streamable HTTP** and **deprecated the older HTTP+SSE transport**. As of 2026-05, current spec transports are:

- **stdio** — local subprocess, simplest case. The MCP server runs as a child process; client and server communicate via stdin/stdout. Used for desktop integrations.
- **Streamable HTTP** — single HTTP endpoint that supports POST and GET, with optional SSE streaming for server-to-client multi-message responses. This is the current remote transport ([Why MCP deprecated SSE — fka.dev, 2025-06](https://blog.fka.dev/blog/2025-06-06-why-mcp-deprecated-sse-and-go-with-streamable-http/); [Auth0 blog: streamable HTTP simplifies security, 2025](https://auth0.com/blog/mcp-streamable-http/)).
- **Legacy HTTP+SSE** — deprecated; servers should still host both endpoints during the transition. Removal deadlines from major hosts (Atlassian, Keboola) land mid-2026.

If a sheet, doc, or library tells you "MCP uses SSE," it is referring to the legacy transport. New code should target Streamable HTTP.

### When to expose tools as MCP server vs in-process

Use an MCP server when:
- The tool is shared across multiple LLM applications/clients.
- The tool runs in a different security domain (sandboxed, separate user, separate machine).
- You want the user (not just the developer) to wire up tools (e.g., end-user-installable Claude Desktop tools).
- The tool integrates with an existing service that already has a server.

Use in-process tool calls when:
- The tool is application-specific.
- You need tight latency (no IPC overhead).
- You want full control over auth, rate-limiting, and error handling without protocol mediation.
- The tool requires application-private state.

> **Anti-pattern:** wrapping every internal function in an MCP server "to be modern." MCP is a protocol for *cross-application* tool exposure. In-process function calls are still correct for most agents.

### Security implications

MCP servers run in the security context the host gave them. If a host launches an MCP server with broad filesystem access, the server has it. The protocol does not constrain what a server can do — the *host* must decide what to expose and what to forbid. The user-consent UX (install-time approval, per-tool approval prompts) is a host concern, not a protocol concern.

This pushes the threat-modeling burden onto host implementers. See [ordis-security-architect](../../ordis-security-architect/) for sandboxing, capability-based authorization, and confused-deputy patterns. Cross-ref also: prompt-injection-via-tool-results below.


## Multi-Agent Orchestration

When the task exceeds what one model invocation can handle gracefully, you can structure the workload as multiple coordinating model invocations. Honest framing: **multi-agent is often over-engineering.** Most "agentic" problems are well-served by a single tool-using loop with good prompting and structured outputs. Reach for multi-agent when single-agent has hit a real wall.

### Pattern 1: Orchestrator–Worker

A planner/orchestrator (typically frontier-reasoning) decomposes a task into subtasks; one or more workers (typically frontier-general or fast-cheap) execute subtasks in parallel; the orchestrator synthesizes the results.

**When it works:** subtasks are independent (parallelizable), the synthesis is mechanical, the orchestrator's plan is stable.

**When it fails:** subtask boundaries leak (worker A needs worker B's output mid-flight); orchestrator over-decomposes; cost balloons because every subtask pays the prompt-context tax.

This is the pattern Anthropic published as the basis for their multi-agent research system ([How we built our multi-agent research system, Anthropic engineering blog 2025](https://www.anthropic.com/engineering/built-multi-agent-research-system)).

### Pattern 2: Hierarchical

Tree of agents, each level decomposing further. Used in long research/coding tasks where the top-level is "ship a feature" and leaves are "edit one file." Risk: depth multiplies cost and latency. Use sparingly and cap depth.

### Pattern 3: Swarm / peer-to-peer

Agents communicate horizontally; no central orchestrator. Used in simulation and exploratory tasks. **Almost always over-engineering for production agents** — emergent coordination is a research problem, not a feature. If you find yourself reaching for swarm, ask whether orchestrator-worker would do.

### Pattern 4: Blackboard

Shared workspace (file, database, scratchpad) where agents read and write. Used to coordinate long-running asynchronous agents. Practical incarnation: file-based memory in the agent's working directory; agents leave todo lists, partial results, and notes for each other (or for their future selves on resume). See [context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md) "Context engineering as a discipline."

### Sub-agent / context-isolation pattern

A common, well-justified multi-agent pattern: spawn a sub-agent with a clean context to perform a focused subtask (e.g., "search the codebase for X"), capture its summarized result, and return to the parent context with only the summary. This isolates the parent's context window from the sub-agent's intermediate clutter — a context-engineering tactic, not a coordination tactic. The Claude Code agent SDK and Anthropic's research system both use this pattern.


## Computer Use as a Special Case

Computer use is a specialization of tool use where the "tool" is a virtual computer (screenshot in, click/keystroke out).

- **Anthropic computer use** — public beta as of 2024-10, currently in beta as of 2026-05 ([Anthropic computer use docs](https://docs.anthropic.com/en/docs/build-with-claude/computer-use); [Anthropic announcement, 2024-10](https://www.anthropic.com/news/3-5-models-and-computer-use)). The "tool" exposes screenshot, mouse, keyboard, bash, and text-editor actions. The model picks an action, you execute it on a sandboxed VM, return the new screenshot, repeat.
- **OpenAI Computer-Using Agent (CUA) / `computer-use` tool** — research preview in the Responses API for usage tiers 3–5 ([OpenAI computer-use guide](https://developers.openai.com/api/docs/guides/tools-computer-use); [OpenAI: Introducing CUA, 2025-01](https://openai.com/index/computer-using-agent/)). Operator (the consumer-facing product) folded into ChatGPT agent mode.
- **Google** — verify current state in Gemini docs; this sheet does not assert a Google computer-use API exists at parity.

**Computer use rules of engagement:**
- Run in a sandbox (VM or container) with no production credentials.
- Cap turns aggressively — agents loop on stuck UIs.
- Treat every screen as untrusted (prompt-injection-by-screenshot is a real attack).
- Cross-ref [ordis-security-architect](../../ordis-security-architect/) for sandbox design and threat modeling.


## Anti-Patterns

### Anti-pattern 1: Tool sprawl

**Wrong:** Expose 30 tools to one model invocation. "More options for the model."

**Right:** Expose the smallest set of tools that solves the task. If you need many tools, partition them into specialist sub-agents (each with 5–10 tools) and route via an orchestrator.

**Principle:** Tool-selection accuracy degrades with set size. Smaller tool sets = better selection, smaller prompts, lower cost.

### Anti-pattern 2: Prompt-injection-via-tool-results (confused deputy)

**Wrong:** Treating retrieved content (web pages, emails, files) as if it were trusted instructions in the prompt.

**Right:** Tag retrieved content explicitly ("untrusted content follows"), strip or escape instruction-like markers, never let tool results trigger sensitive actions without explicit user re-confirmation, sandbox tool effects.

**Principle:** Anything the model can read can try to instruct it. Tool results are *user input*, not *developer prompts*. Cross-ref [ordis-security-architect](../../ordis-security-architect/) for full threat modeling and the confused-deputy pattern.

### Anti-pattern 3: Infinite loops

**Wrong:** No max-iteration cap. "We'll just monitor."

**Right:** Hard cap on loop iterations (10–20 typical), surfaced in logs and metrics, with a graceful "max-turns reached" final message back to the user/caller.

**Principle:** A confused model loops cheaply for itself and expensively for you. Caps are non-negotiable in production.

### Anti-pattern 4: Context bloat from tool results

**Wrong:** Append every tool result verbatim to the conversation history.

**Right:** Summarize or trim tool results before appending; for long results, store full text in the blackboard / file system and append a reference. Cross-ref [context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md) for compaction strategies.

**Principle:** Context is finite and expensive. Tool results that "might be useful later" eat both budget and attention.

### Anti-pattern 5: Treating MCP as "just another protocol"

**Wrong:** Switching tool plumbing to MCP and assuming the agent's behavior will improve.

**Right:** MCP is transport-level standardization. The agent's quality depends on tool-use discipline (good schemas, good descriptions, error recovery, max-iteration cap, prompt-injection guards) — exactly the same discipline as in-process tool use. MCP doesn't fix bad tool design; it lets you reuse the bad design across more clients.

**Principle:** Protocol changes don't fix design problems; they propagate them.

### Anti-pattern 6: Multi-agent for everything

**Wrong:** "Let's architect this as a swarm of seven specialized agents."

**Right:** Start with a single tool-using loop. Add sub-agents only when context isolation is needed; add orchestrator-worker only when subtasks are genuinely parallel. Treat multi-agent as a *cost*, not a feature.

**Principle:** Complexity is liability. Coordination overhead, prompt-tax-per-agent, and debugging cost compound fast.

### Anti-pattern 7: Parsing prose into JSON

**Wrong:** Model emits "The answer is John, age 30" → regex extraction.

**Right:** Use structured-outputs API (OpenAI strict json_schema, Anthropic forced tool_use, Gemini responseSchema) or a constrained-decoding library (Outlines, Instructor).

**Principle:** Specify format up front; don't reverse-engineer it after.

### Anti-pattern 8: Hardcoded model IDs in agent definitions

**Wrong:** `agent = Agent(model="o1-mini-2024-09-12")` baked into agent code.

**Right:** Reference by capability tier in config (`tier=frontier-reasoning`), resolve to a current ID at startup from env or registry. Cross-ref [reasoning-models.md](reasoning-models.md) "Capability Tiers."

**Principle:** Provider lineups change quarterly; tier-based references survive churn.


## Summary

**Tool-use loop:** model → tool_call → tool_result → model. Cap iterations. Recover from errors with structure. Parallel calls when independent.

**Structured outputs:** prefer provider-native (OpenAI strict json_schema, Anthropic forced tool_use, Gemini responseSchema). Fall back to Outlines / Instructor for local or multi-provider. Don't regex prose.

**MCP:** spec at modelcontextprotocol.io. Transports are stdio and Streamable HTTP (SSE deprecated 2025-03-26). Use MCP for cross-application tool exposure; use in-process calls for application-specific tools. Security is the host's problem.

**Multi-agent:** orchestrator-worker is the workhorse pattern. Sub-agent context isolation is the most-justified pattern. Swarm and deep hierarchies are usually over-engineering.

**Computer use:** Anthropic in beta, OpenAI in research preview. Always sandboxed. Always capped. Treat the screen as untrusted.

**Cross-refs:**
- [reasoning-models.md](reasoning-models.md) — capability tiers; reasoning models in agent loops
- [context-engineering-and-prompt-caching.md](context-engineering-and-prompt-caching.md) — sub-agent context isolation; tool-result compaction
- [ordis-security-architect](../../ordis-security-architect/) — tool authorization, sandboxing, confused-deputy threat modeling, prompt-injection defense
- [axiom-engineering-foundations](../../axiom-engineering-foundations/) — surrounding system design (queues, retries, observability)

---

*Model lineup and provider feature set current as of 2026-05; revisit quarterly. Verify provider docs for current model IDs and pricing.*
