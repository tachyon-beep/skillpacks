---
description: Review API design for REST/GraphQL best practices, security, and production readiness
allowed-tools: ["Read", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "[api_directory_or_file]"
---

# Review API Command

Review an API implementation for design quality, security, and production readiness.

## Core Principle

**Good APIs are predictable, consistent, and hard to misuse. Review for developer experience, not just functionality.**

## Review Checklist

### 1. REST API Design

| Check | Good | Bad |
|-------|------|-----|
| **Resource naming** | `/users`, `/orders` | `/getUsers`, `/createOrder` |
| **HTTP methods** | GET reads, POST creates | POST for everything |
| **Status codes** | 201 Created, 404 Not Found | 200 for everything |
| **Versioning** | `/v1/users` or header | No versioning |
| **Pagination** | `?page=1&limit=20` | Return all records |
| **Filtering** | `?status=active` | Separate endpoints |

### 2. GraphQL API Design

| Check | Good | Bad |
|-------|------|-----|
| **Schema design** | Types match domain | Generic catch-all types |
| **N+1 queries** | DataLoader batching | Query per item |
| **Depth limiting** | Max depth = 5-7 | Unlimited nesting |
| **Complexity limits** | Cost analysis | No limits |
| **Error handling** | Structured errors | Generic messages |

### 3. Security

| Check | Required | Anti-Pattern |
|-------|----------|--------------|
| **Authentication** | JWT/OAuth2 with expiry | API keys in URLs |
| **Authorization** | Per-resource checks | Global admin check only |
| **Input validation** | Schema validation | Trust client input |
| **Rate limiting** | Per-user limits | No limits |
| **CORS** | Specific origins | `*` wildcard |
| **SQL injection** | Parameterized queries | String concatenation |

### 4. Error Handling

| Check | Good | Bad |
|-------|------|-----|
| **Structure** | `{"error": {"code": "USER_NOT_FOUND", "message": "..."}}` | `{"error": "Something went wrong"}` |
| **Status codes** | Match error type | 500 for everything |
| **Sensitive data** | No stack traces in prod | Full stack traces |
| **Validation errors** | Field-level details | Generic "invalid input" |

### 5. Documentation

| Check | Required | Missing Impact |
|-------|----------|----------------|
| **OpenAPI/Swagger** | Auto-generated | Manual sync issues |
| **Examples** | Request/response samples | Guessing game |
| **Error codes** | Documented | Trial and error |
| **Authentication** | Clear instructions | Support tickets |

## Review Process

### Step 1: Find API Endpoints

```bash
API_DIR="${ARGUMENTS:-src/}"

# FastAPI routes
grep -r "@app\.\|@router\." "$API_DIR" --include="*.py" -A 3

# Express routes
grep -r "router\.\(get\|post\|put\|delete\)" "$API_DIR" --include="*.ts" -A 3

# Django URLs
grep -r "path\|re_path" "$API_DIR" --include="urls.py"
```

### Step 2: Check Resource Design

```bash
# Look for verb-based URLs (anti-pattern)
grep -rE "/(get|create|update|delete|fetch)" "$API_DIR" --include="*.py" --include="*.ts"

# Look for proper resource URLs
grep -rE '"/[a-z]+s?"' "$API_DIR" --include="*.py" --include="*.ts"
```

### Step 3: Check Security

```bash
# Missing auth decorators
grep -r "def \|async def " "$API_DIR" --include="*.py" | grep -v "Depends\|authenticate"

# Hardcoded secrets
grep -rE "(password|secret|api_key)\s*=\s*['\"]" "$API_DIR"

# SQL injection risks
grep -r "f\".*SELECT\|\.format.*SELECT" "$API_DIR" --include="*.py"
```

### Step 4: Check Error Handling

```bash
# Generic exception handlers
grep -r "except Exception:" "$API_DIR" --include="*.py"

# Missing error responses
grep -r "raise HTTPException" "$API_DIR" --include="*.py" | grep -v "detail="
```

## Output Format

```markdown
## API Review: [API Name/Path]

### Summary

| Aspect | Score | Issues |
|--------|-------|--------|
| REST/GraphQL Design | ✓/⚠️/✗ | [Count] |
| Security | ✓/⚠️/✗ | [Count] |
| Error Handling | ✓/⚠️/✗ | [Count] |
| Documentation | ✓/⚠️/✗ | [Count] |

### Critical Issues (Fix Immediately)

| Location | Issue | Risk | Fix |
|----------|-------|------|-----|
| [file:line] | [Issue] | [Security/Data] | [Action] |

### Design Issues

| Location | Issue | Best Practice |
|----------|-------|--------------|
| [file:line] | [Issue] | [Pattern] |

### Security Findings

| Location | Vulnerability | Severity | Fix |
|----------|--------------|----------|-----|
| [file:line] | [Issue] | High/Medium/Low | [Action] |

### Recommendations

**Immediate:**
1. [Critical fix]

**This Sprint:**
1. [Important improvement]

**Future:**
1. [Nice to have]

### Positive Findings

- [Good patterns observed]
```

## Common Anti-Patterns

### REST Anti-Patterns

| Anti-Pattern | Example | Fix |
|--------------|---------|-----|
| Verb in URL | `POST /createUser` | `POST /users` |
| Wrong method | `GET /users/delete/1` | `DELETE /users/1` |
| Flat structure | `/user-orders` | `/users/{id}/orders` |
| No pagination | Returns 10,000 records | Add `?page=&limit=` |
| Inconsistent naming | `/Users`, `/get_orders` | Consistent lowercase |

### Security Anti-Patterns

| Anti-Pattern | Example | Fix |
|--------------|---------|-----|
| Token in URL | `/api?token=abc` | Authorization header |
| No rate limit | Unlimited requests | Add per-user limits |
| CORS wildcard | `Access-Control-Allow-Origin: *` | Specific origins |
| Sensitive in logs | Log full request body | Redact sensitive fields |

## Cross-Pack Discovery

```python
import glob

# For security review
security_pack = glob.glob("plugins/ordis-security-architect/plugin.json")
if security_pack:
    print("For deeper security review: use ordis-security-architect")

# For API testing
quality_pack = glob.glob("plugins/ordis-quality-engineering/plugin.json")
if quality_pack:
    print("For API testing patterns: use ordis-quality-engineering")
```

## Load Detailed Guidance

For REST design principles:
```
Load skill: axiom-web-backend:using-web-backend
Then read: rest-api-design.md
```

For authentication patterns:
```
Then read: api-authentication.md
```
