---
name: rest-api-design
description: Use when designing REST APIs, choosing HTTP methods/status codes, implementing versioning/pagination/filtering, or applying REST constraints - covers resource modeling, HATEOAS, API evolution patterns
---

# REST API Design

## Overview

**REST API design specialist covering resource modeling, HTTP semantics, versioning, pagination, and API evolution.**

**Core principle**: REST is an architectural style based on resources, HTTP semantics, and stateless communication. Good REST API design makes resources discoverable, operations predictable, and evolution manageable.

## When to Use This Skill

Use when encountering:

- **Resource modeling**: Designing URL structures, choosing singular vs plural, handling relationships
- **HTTP methods**: GET, POST, PUT, PATCH, DELETE semantics and idempotency
- **Status codes**: Choosing correct 2xx, 4xx, 5xx codes
- **Versioning**: URI vs header versioning, managing API evolution
- **Pagination**: Offset, cursor, or page-based pagination strategies
- **Filtering/sorting**: Query parameter design for collections
- **Error responses**: Standardized error formats
- **HATEOAS**: Hypermedia-driven APIs and discoverability

**Do NOT use for**:
- GraphQL API design → `graphql-api-design`
- Framework-specific implementation → `fastapi-development`, `django-development`, `express-development`
- Authentication patterns → `api-authentication`

## Quick Reference - HTTP Methods

| Method | Semantics | Idempotent? | Safe? | Request Body | Response Body |
|--------|-----------|-------------|-------|--------------|---------------|
| GET | Retrieve resource | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| POST | Create resource | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| PUT | Replace resource | ✅ Yes | ❌ No | ✅ Yes | ✅ Optional |
| PATCH | Partial update | ❌ No* | ❌ No | ✅ Yes | ✅ Optional |
| DELETE | Remove resource | ✅ Yes | ❌ No | ❌ Optional | ✅ Optional |
| HEAD | Retrieve headers | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| OPTIONS | Supported methods | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |

*PATCH can be designed to be idempotent but often isn't

## Quick Reference - Status Codes

| Code | Meaning | Use When |
|------|---------|----------|
| 200 OK | Success | GET, PUT, PATCH succeeded with response body |
| 201 Created | Resource created | POST created new resource |
| 202 Accepted | Async processing | Request accepted, processing continues async |
| 204 No Content | Success, no body | DELETE succeeded, PUT/PATCH succeeded without response |
| 400 Bad Request | Invalid input | Validation failed, malformed request |
| 401 Unauthorized | Authentication failed | Missing or invalid credentials |
| 403 Forbidden | Authorization failed | User authenticated but lacks permission |
| 404 Not Found | Resource missing | Resource doesn't exist |
| 409 Conflict | State conflict | Resource already exists, version conflict |
| 422 Unprocessable Entity | Semantic error | Valid syntax but business logic failed |
| 429 Too Many Requests | Rate limited | User exceeded rate limit |
| 500 Internal Server Error | Server error | Unexpected server failure |
| 503 Service Unavailable | Temporary outage | Maintenance, overload |

## Resource Modeling Patterns

### 1. URL Structure

**✅ Good patterns**:

```
GET    /users                    # List users
POST   /users                    # Create user
GET    /users/{id}               # Get specific user
PUT    /users/{id}               # Replace user
PATCH  /users/{id}               # Update user
DELETE /users/{id}               # Delete user

GET    /users/{id}/orders        # User's orders (nested resource)
POST   /users/{id}/orders        # Create order for user
GET    /orders/{id}              # Get specific order (top-level for direct access)

GET    /search/users?q=john      # Search endpoint
```

**❌ Anti-patterns**:

```
GET    /getUsers                 # Verb in URL (use HTTP method instead)
POST   /users/create             # Redundant verb
GET    /users/123/delete         # DELETE operation via GET
POST   /api?action=createUser    # RPC-style, not REST
GET    /users/{id}/orders/{id}   # Ambiguous - which {id}?
```

### 2. Singular vs Plural

**Convention: Use plural for collections, even for single-item endpoints**

```
✅ /users/{id}         # Consistent plural
✅ /orders/{id}        # Consistent plural

❌ /user/{id}          # Inconsistent singular
❌ /users/{id}/order/{id}  # Mixed singular/plural
```

**Exception**: Non-countable resources can be singular

```
✅ /me                 # Current user context
✅ /config             # Application config (single resource)
✅ /health             # Health check endpoint
```

### 3. Nested Resources vs Top-Level

**Nested when showing relationship**:

```
GET /users/{userId}/orders          # "Orders belonging to this user"
POST /users/{userId}/orders         # "Create order for this user"
```

**Top-level when resource has independent identity**:

```
GET /orders/{orderId}               # Direct access to order
DELETE /orders/{orderId}            # Delete order directly
```

**Guidelines**:
- Nest ≤ 2 levels deep (`/users/{id}/orders/{id}` is max)
- Provide top-level access for resources that exist independently
- Use query parameters for filtering instead of deep nesting

```
✅ GET /orders?userId=123           # Better than /users/123/orders/{id}
❌ GET /users/{id}/orders/{id}/items/{id}  # Too deep
```

## Pagination Patterns

### Offset Pagination

**Good for**: Small datasets, page numbers, SQL databases

```
GET /users?limit=20&offset=40

Response:
{
  "data": [...],
  "pagination": {
    "limit": 20,
    "offset": 40,
    "total": 1000,
    "hasMore": true
  }
}
```

**Pros**: Simple, allows jumping to any page
**Cons**: Performance degrades with large offsets, inconsistent with concurrent modifications

### Cursor Pagination

**Good for**: Large datasets, real-time data, NoSQL databases

```
GET /users?limit=20&after=eyJpZCI6MTIzfQ

Response:
{
  "data": [...],
  "pagination": {
    "nextCursor": "eyJpZCI6MTQzfQ",
    "hasMore": true
  }
}
```

**Pros**: Consistent results, efficient for large datasets
**Cons**: Can't jump to arbitrary page, cursors are opaque

### Page-Based Pagination

**Good for**: UIs with page numbers

```
GET /users?page=3&pageSize=20

Response:
{
  "data": [...],
  "pagination": {
    "page": 3,
    "pageSize": 20,
    "totalPages": 50,
    "totalCount": 1000
  }
}
```

**Choice matrix**:

| Use Case | Pattern |
|----------|---------|
| Admin dashboards, small datasets | Offset or Page |
| Infinite scroll feeds | Cursor |
| Real-time data (chat, notifications) | Cursor |
| Need page numbers in UI | Page |
| Large datasets (millions of rows) | Cursor |

## Filtering and Sorting

### Query Parameter Conventions

```
GET /users?status=active&role=admin           # Simple filtering
GET /users?createdAfter=2024-01-01            # Date filtering
GET /users?search=john                        # Full-text search
GET /users?sort=createdAt&order=desc          # Sorting
GET /users?sort=-createdAt                    # Alternative: prefix for descending
GET /users?fields=id,name,email               # Sparse fieldsets
GET /users?include=orders,profile             # Relationship inclusion
```

### Advanced Filtering Patterns

**LHS Brackets (Rails-style)**:

```
GET /users?filter[status]=active&filter[role]=admin
```

**RHS Colon (JSON API style)**:

```
GET /users?filter=status:active,role:admin
```

**Comparison operators**:

```
GET /products?price[gte]=100&price[lte]=500   # Price between 100-500
GET /users?createdAt[gt]=2024-01-01           # Created after date
```

## API Versioning Strategies

### 1. URI Versioning

```
GET /v1/users
GET /v2/users
```

**Pros**: Explicit, easy to route, clear in logs
**Cons**: Violates REST principles (resource identity changes), URL proliferation

**Best for**: Public APIs, major breaking changes

### 2. Header Versioning

```
GET /users
Accept: application/vnd.myapi.v2+json
```

**Pros**: Clean URLs, follows REST principles
**Cons**: Less visible, harder to test in browser

**Best for**: Internal APIs, clients with header control

### 3. Query Parameter Versioning

```
GET /users?version=2
```

**Pros**: Easy to test, optional (can default to latest)
**Cons**: Pollutes query parameters, not semantic

**Best for**: Minor version variants, opt-in features

### Version Deprecation Process

1. **Announce**: Document deprecation timeline (6-12 months recommended)
2. **Warn**: Add `Deprecated` header to responses
3. **Sunset**: Add `Sunset` header with end date (RFC 8594)
4. **Migrate**: Provide migration guides and tooling
5. **Remove**: After sunset date, return 410 Gone

```
HTTP/1.1 200 OK
Deprecated: true
Sunset: Sat, 31 Dec 2024 23:59:59 GMT
Link: </v2/users>; rel="successor-version"
```

## Error Response Format

**Standard JSON error format**:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "One or more fields failed validation",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format",
        "code": "INVALID_FORMAT"
      },
      {
        "field": "age",
        "message": "Must be at least 18",
        "code": "OUT_OF_RANGE"
      }
    ],
    "requestId": "req_abc123",
    "timestamp": "2024-11-14T10:30:00Z"
  }
}
```

**Problem Details (RFC 7807)**:

```json
{
  "type": "https://api.example.com/errors/validation-error",
  "title": "Validation Error",
  "status": 400,
  "detail": "The request body contains invalid data",
  "instance": "/users",
  "invalid-params": [
    {
      "name": "email",
      "reason": "Invalid email format"
    }
  ]
}
```

## HATEOAS (Hypermedia)

**Level 3 REST includes hypermedia links**:

```json
{
  "id": 123,
  "name": "John Doe",
  "status": "active",
  "_links": {
    "self": { "href": "/users/123" },
    "orders": { "href": "/users/123/orders" },
    "deactivate": {
      "href": "/users/123/deactivate",
      "method": "POST"
    }
  }
}
```

**Benefits**:
- Self-documenting API
- Clients discover available actions
- Server controls workflow
- Reduces client-server coupling

**Tradeoffs**:
- Increased response size
- Complexity for simple APIs
- Limited client library support

**When to use**: Complex workflows, long-lived APIs, discoverability requirements

## Idempotency Keys

**For POST operations that should be safely retryable**:

```
POST /orders
Idempotency-Key: key_abc123xyz

{
  "items": [...],
  "total": 99.99
}
```

**Server behavior**:
1. First request with key → Process and store result
2. Duplicate request with same key → Return stored result (do not reprocess)
3. Different request with same key → Return 409 Conflict

**Implementation**:

```python
@app.post("/orders")
def create_order(order: Order, idempotency_key: str = Header(None)):
    if idempotency_key:
        # Check if key was used before
        cached = redis.get(f"idempotency:{idempotency_key}")
        if cached:
            return JSONResponse(content=cached, status_code=200)

    # Process order
    result = process_order(order)

    if idempotency_key:
        # Cache result for 24 hours
        redis.setex(f"idempotency:{idempotency_key}", 86400, result)

    return result
```

## API Evolution Patterns

### Adding Fields (Non-Breaking)

**✅ Safe changes**:
- Add optional request fields
- Add response fields
- Add new endpoints
- Add new query parameters

**Client requirements**: Ignore unknown fields

### Removing Fields (Breaking)

**Strategies**:
1. **Deprecation period**: Mark field as deprecated, remove in next major version
2. **Versioning**: Create v2 without field
3. **Optional → Required**: Never safe, always breaking

### Changing Field Types (Breaking)

**❌ Breaking**:
- String → Number
- Number → String
- Boolean → String
- Flat → Nested object

**✅ Non-breaking**:
- Number → String (if client coerces)
- Adding nullability (required → optional)

**Strategy**: Add new field with correct type, deprecate old field

## Richardson Maturity Model

| Level | Description | Example |
|-------|-------------|---------|
| 0 | POX (Plain Old XML) | Single endpoint, all operations via POST |
| 1 | Resources | Multiple endpoints, still using POST for everything |
| 2 | HTTP Verbs | Proper HTTP methods (GET, POST, PUT, DELETE) |
| 3 | Hypermedia (HATEOAS) | Responses include links to related resources |

**Most APIs target Level 2** (HTTP verbs + status codes).
**Level 3 is optional** but valuable for complex domains.

## Common Anti-Patterns

| Anti-Pattern | Why Bad | Fix |
|--------------|---------|-----|
| Verbs in URLs (`/createUser`) | Not RESTful, redundant with HTTP methods | Use POST /users |
| GET with side effects | Violates HTTP semantics, not safe | Use POST/PUT/DELETE |
| POST for everything | Loses HTTP semantics, not idempotent | Use appropriate method |
| 200 for errors | Breaks HTTP contract | Use correct 4xx/5xx codes |
| Deeply nested URLs | Hard to navigate, brittle | Max 2 levels, use query params |
| Binary response flags | Unclear semantics | Use proper HTTP status codes |
| Timestamps without timezone | Ambiguous | Use ISO 8601 with timezone |
| Pagination without total | Can't show "Page X of Y" | Include total count or hasMore |

## Best Practices Checklist

**Resource Design**:
- [ ] Resources are nouns, not verbs
- [ ] Plural names for collections
- [ ] Max 2 levels of nesting
- [ ] Consistent naming conventions (snake_case or camelCase)

**HTTP Semantics**:
- [ ] Correct HTTP methods for operations
- [ ] Proper status codes (not just 200/500)
- [ ] Idempotent operations are actually idempotent
- [ ] GET/HEAD have no side effects

**API Evolution**:
- [ ] Versioning strategy defined
- [ ] Backward compatibility maintained within version
- [ ] Deprecation headers for sunset features
- [ ] Migration guides for breaking changes

**Error Handling**:
- [ ] Consistent error response format
- [ ] Detailed field-level validation errors
- [ ] Request IDs for tracing
- [ ] Human-readable error messages

**Performance**:
- [ ] Pagination for large collections
- [ ] ETags for caching
- [ ] Gzip compression enabled
- [ ] Rate limiting implemented

## Cross-References

**Related skills**:
- **GraphQL alternative** → `graphql-api-design`
- **FastAPI implementation** → `fastapi-development`
- **Django implementation** → `django-development`
- **Express implementation** → `express-development`
- **Authentication** → `api-authentication`
- **API testing** → `api-testing`
- **API documentation** → `api-documentation` or `muna-technical-writer`
- **Security** → `ordis-security-architect` (OWASP API Security)

## Further Reading

- **REST Dissertation**: Roy Fielding's original thesis
- **RFC 7807**: Problem Details for HTTP APIs
- **RFC 8594**: Sunset HTTP Header
- **JSON:API**: Opinionated REST specification
- **OpenAPI 3.0**: API documentation standard
