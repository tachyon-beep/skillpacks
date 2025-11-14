---
name: using-web-backend
description: Use when building web APIs, backend services, or encountering FastAPI/Django/Express/GraphQL questions, microservices architecture, authentication, or message queues - routes to 12 specialist skills rather than giving surface-level generic advice
---

# Using Web Backend Skills

## Overview

**This router directs you to specialized web backend skills. Each specialist provides deep expertise in their domain.**

**Core principle:** Different backend challenges require different specialist knowledge. Routing to the right skill gives better results than generic advice.

## When to Use

Use this router when encountering:

- **Framework-specific questions**: FastAPI, Django, Express implementation details
- **API design**: REST or GraphQL architecture, versioning, schema design
- **Architecture patterns**: Microservices, message queues, event-driven systems
- **Backend infrastructure**: Authentication, database integration, deployment
- **Testing & documentation**: API testing strategies, documentation approaches

## Quick Reference - Routing Table

| User Question Contains | Route To | Why |
|------------------------|----------|-----|
| FastAPI, Pydantic, async Python APIs | `fastapi-development` | FastAPI-specific patterns, dependency injection, async |
| Django, ORM, views, middleware | `django-development` | Django conventions, ORM optimization, settings |
| Express, Node.js backend, middleware | `express-development` | Express patterns, error handling, async flow |
| REST API, endpoints, versioning, pagination | `rest-api-design` | REST principles, resource design, hypermedia |
| GraphQL, schema, resolvers, N+1 | `graphql-api-design` | Schema design, query optimization, federation |
| Microservices, service mesh, boundaries | `microservices-architecture` | Service design, communication, consistency |
| Message queues, RabbitMQ, Kafka, events | `message-queues` | Queue patterns, reliability, event-driven |
| JWT, OAuth2, API keys, auth | `api-authentication` | Auth patterns, token management, security |
| Database connections, ORM, migrations | `database-integration` | Connection pooling, query optimization, migrations |
| API testing, integration tests, mocking | `api-testing` | Testing strategies, contract testing, mocking |
| OpenAPI, Swagger, API docs | `api-documentation` | API docs (also see: muna-technical-writer) |
| Docker, deployment, health checks | `backend-deployment` | Containerization, config, monitoring |

## Cross-References to Other Packs

**Before routing, check if these packs are more appropriate:**

- **Security concerns** → `ordis-security-architect` (threat modeling, OWASP, security patterns)
- **API usability/UX** → `lyra-ux-designer` (error messages, API ergonomics)
- **Python code patterns** → `axiom-python-engineering` (Python-specific engineering)
- **Documentation writing** → `muna-technical-writer` (technical writing, clarity)

## How to Route

**STOP: Do not attempt to answer web backend questions yourself.**

**Instead:**

1. **Identify the specialist domain** from the routing table above
2. **State which specialist skill you're using**: "I'll use the `[skill-name]` skill for this"
3. **Apply the specialist skill** to provide deep, expert guidance

## Red Flags - Using Generic Knowledge Instead of Specialists

If you catch yourself doing any of these, STOP and route to a specialist:

- ❌ "I'll answer all these diverse backend questions myself"
- ❌ Giving code examples from "general backend knowledge"
- ❌ Providing "competent but broad" answers across multiple domains
- ❌ Not mentioning specialized skills would give better guidance
- ❌ "The specialist skill isn't available right now, so I'll answer myself"

**All of these mean: You're being a jack-of-all-trades. Route to the specialist.**

## Rationalization Table

| Excuse | Reality |
|--------|---------|
| "I can handle this backend question myself" | Specialists provide deeper expertise than generic knowledge |
| "These are all backend topics" | Backend spans 12+ distinct specialist domains |
| "I'll just give a quick answer" | Quick answers miss edge cases and production patterns |
| "The user didn't ask for specialized help" | Better results come from routing to specialists proactively |
| "Specialist skill not available, I'll answer instead" | State which skill would help, explain routing, guide user to install if needed |

## Example Routing

**User**: "I need to build a REST API with FastAPI. How should I structure my dependency injection?"

**Your response**:
"I'll use the `fastapi-development` skill to help you structure dependency injection properly. This specialist skill covers FastAPI patterns in depth."

[Then apply fastapi-development skill]

**User**: "My GraphQL API has N+1 problems and we're splitting our monolith into microservices."

**Your response**:
"You have two distinct specialist areas:
1. GraphQL optimization → I'll use `graphql-api-design`
2. Microservices architecture → I'll use `microservices-architecture`

Let me address the GraphQL N+1 problem first with the graphql-api-design skill..."

[Apply each specialist skill to its domain]

## Why This Matters

**Without routing**: Surface-level answers covering multiple domains broadly
**With routing**: Deep expertise addressing edge cases, production patterns, and domain-specific best practices

Specialist skills = better results.
