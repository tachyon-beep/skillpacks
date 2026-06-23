---
description: Routes MCP (Model Context Protocol) server engineering questions to specialist reference sheets (tool API design, idempotency, error envelopes, schema drift, transport, primitives, composition, observability, testing)
---

# MCP Engineering Routing

**This router directs you to specialized MCP (Model Context Protocol) server-engineering reference sheets. An MCP server is a contract with an unreliable, non-deterministic, retrying client — each sheet provides deep, production-grade expertise in one slice of building it.**

Use the `using-mcp-engineering` skill from the `axiom-mcp-engineering` plugin to route MCP questions to the appropriate specialist:

- **tool-api-design** - Tool surface as a prompt fragment, naming, parameter shape, descriptions an LLM reads
- **mcp-primitive-selection** - Choosing between tools / resources / prompts / sampling
- **resources-prompts-sampling** - The non-tool primitives and when to reach for them
- **idempotency-and-atomicity** - Safe retry, atomic claim/lease semantics under concurrent agents
- **error-envelopes-and-recovery** - Structured errors an agent can parse and recover from
- **schema-versioning-and-drift** - Versioning tool schemas across model drift
- **transport-reliability** - stdio framing, HTTP, reconnect and state across sessions
- **output-shape-and-pagination** - Pagination, truncation, summary-vs-detail within the context budget
- **composition-and-namespaces** - Composing multiple servers, namespacing, collision avoidance
- **authentication-and-trust** - Auth, trust boundaries, scoping tool access
- **observability-for-tool-calls** - Tracing, logging, and metrics for tool invocations
- **testing-mcp-servers** - Golden-conversation testing and agent-context validation
- **mcp-server-smells** - Anti-patterns and failure modes in MCP server design

**Commands:** `/design-mcp-server`, `/review-mcp-server`, `/audit-mcp-tools`
**Agents:** mcp-server-architect, mcp-server-critic

Boundary concerns (siblings, contrasts, downstream consumers) are absorbed into the router's own Pipeline Position section rather than a separate sheet.

**Cross-references to other packs:**
- General REST/GraphQL API design → `/web-backend`
- Client-side prompt engineering & tool loops → `/llm-specialist`
- General in-process plugin architecture → `/system-architect` or `/procedural-architecture`
- Multi-stage agent workflows expressed as tool sequences → `/procedural-architecture`
- Cryptographic-provenance audit trails → `/audit-pipelines`
- Whole-system deterministic replay (vs golden-conversation replay) → `/determinism-and-replay`
