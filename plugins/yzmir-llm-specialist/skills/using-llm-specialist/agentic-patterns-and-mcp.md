# Tool execution and agent boundaries

Use when an application delegates work to tools, runs an agent loop or integrates an MCP client/server. A transport does not confer instruction authority or permission to act.

## Tool contract

- Specify input schema, output/error shape, authentication/authorization, side effects, idempotency and retry semantics.
- Treat user/retrieved/tool text as data at its actual trust level. A tool result cannot elevate itself to developer policy.
- Enforce permissions and consequential-action checks in the runtime boundary. Prompt warnings are not an isolation mechanism.
- Validate structured outputs before execution. Distinguish unsupported arguments, schema failures, tool exceptions, timeouts and refusals; avoid retrying non-idempotent effects blindly.
- Bound task cost, iterations, time and parallelism. Preserve cancellation and partial-result handling; define how the user can recover or intervene.
- Log task/tool/artifact identity and outcomes while protecting secrets. Evaluate end-to-end completion, incorrect effects and recovery, not just valid tool-call syntax.

## Orchestration choice

Use a direct tool call or one agent when sufficient. Planner/executor separation, concurrent specialists and durable workflows are choices justified by independent work, authority boundaries or context isolation. Shared prompts and models can share blind spots; extra agents are not independent empirical votes.

## MCP integration

Choose MCP when reusable tool discovery/integration or a client/server boundary helps the product. Inspect the negotiated protocol version, capability/tool schemas, transport/auth and cancellation/session behavior in current official docs. For implementation checks use MCP engineering; this sheet defines application use and trust.

Computer-use actions need observed UI state, stable target identification and recovery from partial effects. Do not infer successful external writes from a proposed action or model narration.

## Deliverable

Tool/agent contract or focused repair, with permission/data boundaries, effect/retry tests, representative failure traces and measured task outcome. See [safety](llm-safety-alignment.md), [evaluation](llm-evaluation-metrics.md) and [context](context-engineering-and-prompt-caching.md) as needed.

## Existing source pointers

Check current applicability/version before using a recipe. These pointers are optional supporting sources.

- <https://platform.openai.com/docs/guides/function-calling>
- <https://docs.anthropic.com/en/docs/build-with-claude/tool-use>
- <https://developers.openai.com/api/docs/guides/structured-outputs>
- <https://modelcontextprotocol.io>
- <https://modelcontextprotocol.io/specification/>
- <https://modelcontextprotocol.io/specification/2025-11-25/basic/transports>
- <https://docs.anthropic.com/en/docs/build-with-claude/computer-use>
- <https://developers.openai.com/api/docs/guides/tools-computer-use>
