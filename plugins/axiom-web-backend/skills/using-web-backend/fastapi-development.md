
# FastAPI Development

## Overview

**FastAPI specialist skill providing production-ready patterns, anti-patterns to avoid, and testing strategies.**

**Core principle**: FastAPI's type hints, dependency injection, and async-first design enable fast, maintainable APIs - but require understanding async/sync boundaries, proper dependency management, and production hardening patterns.

## When to Use This Skill

Use when encountering:

- **Dependency injection**: Database connections, auth, shared resources, testing overrides
- **Async/sync boundaries**: Mixing blocking I/O with async endpoints, performance issues
- **Background tasks**: Choosing between BackgroundTasks, Celery, or other task queues
- **File uploads**: Streaming large files, memory management
- **Testing**: Dependency overrides, async test clients, fixture patterns
- **Production deployment**: ASGI servers, lifespan management, connection pooling
- **Security**: SQL injection, CORS, authentication patterns
- **Performance**: Connection pooling, query optimization, caching

## Quick Reference - Common Patterns

| Pattern | Use Case | Code Snippet |
|---------|----------|--------------|
| **DB dependency with pooling** | Per-request database access | `def get_db(): db = SessionLocal(); try: yield db; finally: db.close()` |
| **Dependency override for testing** | Test with mock/test DB | `app.dependency_overrides[get_db] = override_get_db` |
| **Lifespan events** | Startup/shutdown resources | `@asynccontextmanager async def lifespan(app): ... yield ...` |
| **Streaming file upload** | Large files without memory issues | `async with aiofiles.open(...) as f: while chunk := await file.read(CHUNK_SIZE): await f.write(chunk)` |
| **Background tasks (short)** | < 30 sec tasks | `background_tasks.add_task(func, args)` |
| **Task queue (long)** | > 1 min tasks, retries needed | Use Celery/Arq with Redis |
| **Parameterized queries** | Prevent SQL injection | `cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))` |

## Core Patterns

### 1. Dependency Injection Architecture

**Pattern: Connection pooling with yield dependencies**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi import Depends, FastAPI

# One-time pool creation at module level
engine = create_engine(
    "postgresql://user:pass@localhost/db",
    pool_size=20,          # Max connections
    max_overflow=0,        # No overflow beyond pool_size
    pool_pre_ping=True,    # Verify connection health before use
    pool_recycle=3600      # Recycle connections every hour
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

# Dependency pattern with automatic cleanup
def get_db() -> Session:
    """
    Yields database session from pool.
    Ensures cleanup even if endpoint raises exception.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Usage in endpoints
@app.get("/items/{item_id}")
def get_item(item_id: int, db: Session = Depends(get_db)):
    return db.query(Item).filter(Item.id == item_id).first()
```

**Why this pattern**:
- Pool created once (expensive operation)
- Per-request connections from pool (cheap)
- `yield` ensures cleanup on success AND exceptions
- `pool_pre_ping` prevents stale connection errors
- `pool_recycle` prevents long-lived connection issues

**Testing pattern**:

```python
# conftest.py
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def test_db():
    """Test database fixture"""
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.rollback()
        db.close()

@pytest.fixture
def client(test_db):
    """Test client with overridden dependencies"""
    def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

# test_items.py
def test_get_item(client, test_db):
    # Setup test data
    test_db.add(Item(id=1, name="Test"))
    test_db.commit()

    # Test endpoint
    response = client.get("/items/1")
    assert response.status_code == 200
```

### 2. Async/Sync Boundary Management

**❌ Anti-pattern: Blocking calls in async endpoints**

```python
# BAD - Blocks event loop
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    conn = psycopg2.connect(...)  # Blocking!
    cursor = conn.cursor()
    cursor.execute(...)           # Blocking!
    return cursor.fetchone()
```

**✅ Pattern: Use async libraries or run_in_threadpool**

```python
# GOOD Option 1: Async database library
from databases import Database

database = Database("postgresql://...")

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    query = "SELECT * FROM users WHERE id = :user_id"
    return await database.fetch_one(query=query, values={"user_id": user_id})

# GOOD Option 2: Run blocking code in thread pool
from fastapi.concurrency import run_in_threadpool

def blocking_db_call(user_id: int):
    conn = psycopg2.connect(...)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    return cursor.fetchone()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return await run_in_threadpool(blocking_db_call, user_id)
```

**Decision table**:

| Scenario | Use |
|----------|-----|
| PostgreSQL with async needed | `asyncpg` or `databases` library |
| PostgreSQL, sync is fine | `psycopg2` with `def` (not `async def`) endpoints |
| MySQL with async | `aiomysql` |
| SQLite | `aiosqlite` (async) or sync with `def` endpoints |
| External API calls | `httpx.AsyncClient` |
| CPU-intensive work | `run_in_threadpool` or Celery |

### 3. Lifespan Management (Modern Pattern)

**✅ Use lifespan context manager** (replaces deprecated `@app.on_event`)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

# Global resources
resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    resources["db_pool"] = await create_async_pool(
        "postgresql://...",
        min_size=10,
        max_size=20
    )
    resources["redis"] = await aioredis.create_redis_pool("redis://...")
    resources["ml_model"] = load_ml_model()  # Can be sync or async

    yield  # Application runs

    # Shutdown
    await resources["db_pool"].close()
    resources["redis"].close()
    await resources["redis"].wait_closed()
    resources.clear()

app = FastAPI(lifespan=lifespan)

# Access resources in endpoints
@app.get("/predict")
async def predict(data: dict):
    model = resources["ml_model"]
    return {"prediction": model.predict(data)}
```

### 4. File Upload Patterns

**For 100MB+ files: Stream to disk, never load into memory**

```python
from fastapi import UploadFile, File, HTTPException
import aiofiles
import os

UPLOAD_DIR = "/var/uploads"
CHUNK_SIZE = 1024 * 1024  # 1MB chunks
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

@app.post("/upload")
async def upload_large_file(file: UploadFile = File(...)):
    # Validate content type
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "Only video files accepted")

    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    size = 0

    try:
        async with aiofiles.open(filepath, 'wb') as f:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(413, "File too large")
                await f.write(chunk)
    except Exception as e:
        # Cleanup on failure
        if os.path.exists(filepath):
            os.remove(filepath)
        raise

    return {"filename": file.filename, "size": size}
```

**For very large files (1GB+): Direct S3 upload with presigned URLs**

```python
import boto3

@app.post("/upload/presigned-url")
async def get_presigned_upload_url(filename: str):
    s3_client = boto3.client('s3')
    presigned_post = s3_client.generate_presigned_post(
        Bucket='my-bucket',
        Key=f'uploads/{uuid.uuid4()}_{filename}',
        ExpiresIn=3600
    )
    return presigned_post  # Client uploads directly to S3
```

### 5. Background Task Decision Matrix

| Task Duration | Needs Retries? | Needs Monitoring? | Solution |
|---------------|----------------|-------------------|----------|
| < 30 seconds | No | No | `BackgroundTasks` |
| < 30 seconds | Yes | Maybe | Celery/Arq |
| > 1 minute | Don't care | Don't care | Celery/Arq |
| Any | Yes | Yes | Celery/Arq with monitoring |

**BackgroundTasks pattern** (simple, in-process):

```python
from fastapi import BackgroundTasks

async def send_email(email: str):
    await asyncio.sleep(2)  # Async work
    print(f"Email sent to {email}")

@app.post("/register")
async def register(email: str, background_tasks: BackgroundTasks):
    # ... save user ...
    background_tasks.add_task(send_email, email)
    return {"status": "registered"}  # Returns immediately
```

**Celery pattern** (distributed, persistent):

```python
# celery_app.py
from celery import Celery

celery_app = Celery('tasks', broker='redis://localhost:6379/0')

@celery_app.task(bind=True, max_retries=3)
def process_video(self, filepath: str):
    try:
        # Long-running work
        extract_frames(filepath)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)

# main.py
from celery_app import process_video

@app.post("/upload")
async def upload(file: UploadFile):
    filepath = await save_file(file)
    task = process_video.delay(filepath)
    return {"task_id": task.id}

@app.get("/status/{task_id}")
async def get_status(task_id: str):
    from celery_app import celery_app
    result = celery_app.AsyncResult(task_id)
    return {"status": result.state, "result": result.result}
```

## Security Patterns

### SQL Injection Prevention

**❌ NEVER use f-strings or string concatenation**

```python
# DANGEROUS
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
cursor.execute("SELECT * FROM users WHERE email = '" + email + "'")
```

**✅ ALWAYS use parameterized queries**

```python
# SQLAlchemy ORM (safe)
db.query(User).filter(User.id == user_id).first()

# Raw SQL (safe with parameters)
cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
cursor.execute("SELECT * FROM users WHERE email = :email", {"email": email})
```

### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific origins, not "*" in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
```

### Authentication Pattern

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
        return await get_user_by_id(user_id)
    except jwt.JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")

@app.get("/protected")
async def protected_route(current_user = Depends(get_current_user)):
    return {"user": current_user}
```

## Middleware Ordering

**Critical: Middleware wraps in order added, executes in reverse for responses**

```python
# Correct order:
app.add_middleware(CORSMiddleware, ...)       # 1. FIRST - handles preflight
app.add_middleware(RequestLoggingMiddleware)  # 2. Logs entire request
app.add_middleware(ErrorHandlingMiddleware)   # 3. Catches errors from auth/routes
app.add_middleware(AuthenticationMiddleware)  # 4. LAST - closest to routes
```

## Common Anti-Patterns

| Anti-Pattern | Why Bad | Fix |
|--------------|---------|-----|
| Global database connection | Not thread-safe, connection leaks | Use connection pool with dependency injection |
| `async def` with blocking I/O | Blocks event loop, kills performance | Use async libraries or `run_in_threadpool` |
| `time.sleep()` in async code | Blocks entire event loop | Use `asyncio.sleep()` |
| Loading large files into memory | Memory exhaustion, OOM crashes | Stream with `aiofiles` and chunks |
| BackgroundTasks for long work | Lost on restart, no retries | Use Celery/Arq |
| String formatting in SQL | SQL injection vulnerability | Parameterized queries only |
| `allow_origins=["*"]` with credentials | Security vulnerability | Specify exact origins |
| Not closing database connections | Connection pool exhaustion | Use `yield` in dependencies |

## Testing Best Practices

```python
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Sync tests (simpler, faster for most cases)
def test_read_item(client):
    response = client.get("/items/1")
    assert response.status_code == 200

# Async tests (needed for testing async endpoints with real async operations)
@pytest.mark.asyncio
async def test_async_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/items/1")
    assert response.status_code == 200

# Dependency override pattern
def test_with_mock_db(client):
    def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    response = client.get("/items/1")
    app.dependency_overrides.clear()
    assert response.status_code == 200
```

## Production Deployment

**ASGI server configuration** (Uvicorn + Gunicorn):

```bash
# gunicorn with uvicorn workers (production)
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --graceful-timeout 30 \
  --keep-alive 5
```

**Environment-based configuration**:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    redis_url: str
    secret_key: str
    debug: bool = False

    class Config:
        env_file = ".env"

settings = Settings()

# Use in app
engine = create_engine(settings.database_url)
```

## Cross-References

**Related skills**:
- **Security** → `ordis-security-architect` (threat modeling, OWASP top 10)
- **Python patterns** → `axiom-python-engineering` (async patterns, type hints)
- **API testing** → `api-testing` (contract testing, integration tests)
- **API documentation** → `api-documentation` or `muna-technical-writer`
- **Database optimization** → `database-integration` (query optimization, migrations)
- **Authentication deep dive** → `api-authentication` (OAuth2, JWT patterns)
- **GraphQL alternative** → `graphql-api-design`

## Performance Tips

1. **Use connection pooling** - Create pool once, not per-request
2. **Enable response caching** - Use `fastapi-cache2` for expensive queries
3. **Limit response size** - Paginate large result sets
4. **Use async for I/O** - Database, HTTP calls, file operations
5. **Profile slow endpoints** - Use `starlette-prometheus` for monitoring
6. **Enable gzip compression** - `GZipMiddleware` for large JSON responses

## When NOT to Use FastAPI

- **Simple CRUD with admin panel** → Django (has built-in admin)
- **Heavy template rendering** → Django or Flask
- **Mature ecosystem needed** → Django (more third-party packages)
- **Team unfamiliar with async** → Flask or Django (simpler mental model)

FastAPI excels at: Modern APIs, microservices, ML model serving, real-time features, high performance requirements.
