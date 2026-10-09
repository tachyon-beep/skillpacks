---
name: using-web-backend
description: "Use when implementing or reviewing production API behavior: authorization, compatibility, transaction boundaries, retries, bounded results and contract tests."
---

# Web Backend Engineering

Specify and verify API behavior against the project's supported framework/runtime and deployment constraints. Framework syntax alone does not need this skill.

## Workflow

1. Inspect routes, schemas, identity/permission model, storage/message boundaries and existing tests. Identify supported versions from manifests/lockfiles and deployment images.
2. Define request/response/error contracts, authorization per resource/action, compatibility and pagination/size limits.
3. Trace writes through transaction boundaries. State idempotency/retry behavior, optimistic concurrency, external side effects and recovery from partial failure.
4. For asynchronous work, define delivery/deduplication/ordering and how status reaches the caller. For GraphQL, bound complexity and check resolver authorization/N+1 behavior.
5. Implement with repository conventions and current official framework documentation. Avoid copying old scaffold versions or adding a service/framework without a real requirement.
6. Check observable contracts: success, denied/invalid/missing cases, race/retry behavior, persistence and failure responses. Measure performance when a claim depends on it.
7. Document externally relevant behavior and report checks/limitations. Use security, distributed-system or operational guidance selectively when the boundary requires it.

## Optional recipes

[REST compatibility](rest-api-design.md), [authentication/authorization](api-authentication.md), [transactions](database-integration.md), [messages](message-queues.md), [GraphQL](graphql-api-design.md), [API tests](api-testing.md), [documentation](api-documentation.md), [service boundaries](microservices-architecture.md), [FastAPI](fastapi-development.md), [Django](django-development.md), [Express](express-development.md).

Recipes are checklists and primary-documentation pointers, not pinned tutorial implementations. Verify version-specific semantics in the installed/project version before adopting a recipe.

## Working contract

Use the smallest deliverable that makes the decision or change reviewable. Existing project records can satisfy these fields; do not create duplicate documents. Read only references needed for the unresolved question. Ordinary work does not require loading a specialist, delegating, or asking a routing question.

Use tools and current project evidence where available. Distinguish observed facts, inferences and unknowns. Report material findings with a source, consequence and proposed action; report a supported clean result when appropriate. State checks run, checks omitted and residual uncertainty without mandatory report sections. Honor current user authorization and applicable project policy; ask only when a missing decision materially blocks progress.
