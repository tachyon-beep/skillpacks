---
description: Scaffold a new API project with framework selection, structure, and best practices
allowed-tools: ["Read", "Bash", "Glob", "Grep", "Write", "AskUserQuestion"]
argument-hint: "[project_name]"
---

# Scaffold API Command

Create a new API project with proper structure, framework selection, and production-ready patterns.

## Core Principle

**Start with structure, not code. Framework choice follows requirements, not preference.**

## Information Gathering

Before scaffolding, determine:

1. **API style**: REST or GraphQL?
2. **Language/Framework**: Python (FastAPI/Django) or Node.js (Express)?
3. **Database**: PostgreSQL, MongoDB, or none initially?
4. **Auth pattern**: JWT, OAuth2, API keys, or none?
5. **Scale expectations**: Single service or microservices?

## Framework Selection Guide

| Requirement | Recommended | Why |
|-------------|-------------|-----|
| Async-heavy, high performance | FastAPI | Native async, fast |
| Full-featured, batteries included | Django | Admin, ORM, auth built-in |
| JavaScript ecosystem, flexibility | Express | Middleware-based, lightweight |
| Complex nested data, flexible queries | GraphQL | Client-driven queries |
| Simple CRUD, many clients | REST | Universal, cacheable |

## Project Structure by Framework

### FastAPI (Python)

```
project_name/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, routers
│   ├── config.py            # Settings, environment
│   ├── dependencies.py      # Dependency injection
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── users.py
│   │   └── items.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── user.py          # SQLAlchemy models
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── user.py          # Pydantic schemas
│   ├── services/
│   │   ├── __init__.py
│   │   └── user_service.py  # Business logic
│   └── db/
│       ├── __init__.py
│       └── session.py       # Database session
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Fixtures
│   └── test_users.py
├── alembic/                  # Migrations
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

### Django (Python)

```
project_name/
├── config/
│   ├── __init__.py
│   ├── settings/
│   │   ├── base.py
│   │   ├── development.py
│   │   └── production.py
│   ├── urls.py
│   └── wsgi.py
├── apps/
│   ├── users/
│   │   ├── models.py
│   │   ├── views.py
│   │   ├── serializers.py
│   │   ├── urls.py
│   │   └── tests.py
│   └── core/
│       └── models.py        # Base models
├── tests/
├── manage.py
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

### Express (Node.js)

```
project_name/
├── src/
│   ├── index.ts             # Entry point
│   ├── app.ts               # Express app
│   ├── config/
│   │   └── index.ts         # Configuration
│   ├── routes/
│   │   ├── index.ts
│   │   └── users.ts
│   ├── controllers/
│   │   └── userController.ts
│   ├── services/
│   │   └── userService.ts
│   ├── models/
│   │   └── user.ts
│   ├── middleware/
│   │   ├── auth.ts
│   │   └── errorHandler.ts
│   └── utils/
│       └── logger.ts
├── tests/
│   └── users.test.ts
├── package.json
├── tsconfig.json
├── Dockerfile
└── docker-compose.yml
```

## Essential Files to Generate

### 1. Health Check Endpoint

```python
# FastAPI
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.VERSION}
```

### 2. Error Handling

```python
# FastAPI
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )
```

### 3. Configuration Management

```python
# FastAPI - config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    DEBUG: bool = False

    class Config:
        env_file = ".env"
```

### 4. Docker Setup

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
RUN pip install .

COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5. Environment Template

```bash
# .env.example
DATABASE_URL=postgresql://user:pass@localhost/dbname
SECRET_KEY=your-secret-key-here
DEBUG=true
```

## Output Format

```markdown
## API Project Scaffold: [Project Name]

### Configuration

| Setting | Value |
|---------|-------|
| Framework | [FastAPI/Django/Express] |
| API Style | [REST/GraphQL] |
| Database | [PostgreSQL/MongoDB/None] |
| Auth | [JWT/OAuth2/API Keys/None] |

### Generated Structure

```
[Directory tree]
```

### Key Files Created

1. **Entry point**: [path] - Main application
2. **Configuration**: [path] - Environment settings
3. **Routes**: [path] - API endpoints
4. **Health check**: [endpoint] - Status endpoint

### Next Steps

1. [ ] Copy `.env.example` to `.env` and configure
2. [ ] Install dependencies: `[command]`
3. [ ] Run migrations: `[command]`
4. [ ] Start server: `[command]`
5. [ ] Test health endpoint: `curl localhost:8000/health`

### Recommended Additions

Based on your requirements:
- [ ] Add authentication ([skill reference])
- [ ] Set up database ([skill reference])
- [ ] Configure CI/CD ([skill reference])
```

## Cross-Pack Discovery

```python
import glob

# For Python patterns
python_pack = glob.glob("plugins/axiom-python-engineering/plugin.json")
if python_pack:
    print("Available: axiom-python-engineering for Python best practices")

# For security
security_pack = glob.glob("plugins/ordis-security-architect/plugin.json")
if security_pack:
    print("Available: ordis-security-architect for API security patterns")

# For testing
quality_pack = glob.glob("plugins/ordis-quality-engineering/plugin.json")
if quality_pack:
    print("Available: ordis-quality-engineering for API testing strategies")
```

## Load Detailed Guidance

For framework-specific patterns:
```
Load skill: axiom-web-backend:using-web-backend
Then read: fastapi-development.md, django-development.md, or express-development.md
```

For API design:
```
Then read: rest-api-design.md or graphql-api-design.md
```
