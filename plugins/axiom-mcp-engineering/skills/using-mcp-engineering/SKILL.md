---
name: using-mcp-engineering
description: "Use when designing or reviewing MCP server surfaces: model-visible capability contracts, host negotiation, retries, authorization, bounded output and real task evaluation."
---

# MCP Engineering

Design a capability surface that a host/model can discover and use correctly, and a server can enforce under retries, races and reconnects. Ordinary API correctness still applies; MCP adds negotiated host capabilities and model interpretation.

## Workflow

1. Identify the real caller/host, supported protocol/SDK versions, transport, negotiated capabilities and authority boundary. Test support rather than assuming sampling/resources/prompts are exposed uniformly.
2. Classify capabilities by use: tools for model-selected operations (including useful reads); resources for addressable context where hosts support discovery/attachment; prompts for reusable workflows; host sampling only when negotiated and suitable.
3. Write schemas/descriptions and bounded outputs around user intent. Name required permissions, side effects, error/recovery classes, pagination and compatibility.
4. State durable idempotency/concurrency/atomicity guarantees for mutations and observable external effects. Trace logical operation separately from attempts.
5. Implement and verify protocol/contract behavior deterministically; replay captured calls with explicit volatile-field binding/comparison. Frozen calls do not evaluate how a model interprets descriptions.
6. Use real model/host tasks where tool selection or interpretation is uncertain. Record task outcomes, model/host configuration, failures and denominator; no universal pass quota.
7. Review evidence, supported limitations and recovery. Clean results are allowed after a scoped sweep; do not require critic disagreement.

An external inference SDK in a server is a product/consent/data/credential choice, not automatically a blocker. Distinguish server-owned inference from borrowing host inference; disclose the trust/cost boundary and evaluate negotiated alternatives.

## Optional references

[Primitive selection](mcp-primitive-selection.md), [tool contracts](tool-api-design.md), [errors](error-envelopes-and-recovery.md), [idempotency](idempotency-and-atomicity.md), [bounded output](output-shape-and-pagination.md), [trust](authentication-and-trust.md), [transport](transport-reliability.md), [resources/prompts/sampling](resources-prompts-sampling.md), [composition](composition-and-namespaces.md), [versioning](schema-versioning-and-drift.md), [telemetry](observability-for-tool-calls.md), [tests](testing-mcp-servers.md), [smells](mcp-server-smells.md).

## Working contract

Use the smallest deliverable that makes the decision or change reviewable. Existing project records can satisfy these fields; do not create duplicate documents. Read only references needed for the unresolved question. Ordinary work does not require loading a specialist, delegating, or asking a routing question.

Use tools and current project evidence where available. Distinguish observed facts, inferences and unknowns. Report material findings with a source, consequence and proposed action; report a supported clean result when appropriate. State checks run, checks omitted and residual uncertainty without mandatory report sections. Honor current user authorization and applicable project policy; ask only when a missing decision materially blocks progress.
