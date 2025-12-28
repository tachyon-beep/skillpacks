
# Language/Framework Patterns

## Purpose

Technology-specific patterns to check during codebase analysis. Supplements generic architecture analysis with language and framework-aware concerns, entry points, and common patterns.

## When to Use

- After identifying technology stack in discovery phase
- When analyzing subsystems built with known frameworks
- To catch framework-specific anti-patterns during cataloging
- As reference during systematic subsystem analysis
- Technology stack identified: Python, JavaScript/TypeScript, or Rust

## Core Principle: Technology Informs Analysis

**Generic analysis misses technology-specific concerns. Framework awareness catches more.**

A Django project has specific places to check (settings.py, middleware, admin). A React app has different patterns (state management, route guards). Use this reference to know WHERE to look based on the technology.

## How to Use This Reference

1. **Identify technology stack** during discovery phase
2. **Load relevant section** for the detected stack
3. **Check each pattern location** during subsystem analysis
4. **Document findings** in Patterns Observed and Concerns sections of catalog

---

# Python Patterns

## Django

### Configuration & Security

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Secret key** | `settings.py` | `SECRET_KEY` source - env var or hardcoded? |
| **Debug mode** | `settings.py` | `DEBUG` should be False in production config |
| **Allowed hosts** | `settings.py` | `ALLOWED_HOSTS` not empty or wildcard |
| **Database config** | `settings.py` | Credentials from env vars, not hardcoded |
| **CSRF protection** | `settings.py` | `CSRF_COOKIE_SECURE`, `CSRF_TRUSTED_ORIGINS` |
| **Session security** | `settings.py` | `SESSION_COOKIE_SECURE`, `SESSION_COOKIE_HTTPONLY` |

### Authentication

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Custom user model** | `settings.py`, `models.py` | `AUTH_USER_MODEL` defined? Custom or default? |
| **Auth backends** | `settings.py` | `AUTHENTICATION_BACKENDS` - what methods supported? |
| **Password validators** | `settings.py` | `AUTH_PASSWORD_VALIDATORS` configured? |
| **Login/logout views** | `urls.py`, `views.py` | `django.contrib.auth.views` or custom? |
| **Permission classes** | `views.py` | `permission_classes`, `@permission_required` |

### Middleware

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Middleware order** | `settings.py` | `MIDDLEWARE` - order matters (security first) |
| **Custom middleware** | `middleware/` | What each middleware does |
| **Auth middleware** | `settings.py` | `AuthenticationMiddleware` present? |
| **CORS middleware** | `settings.py` | `corsheaders.middleware` if API serves other origins |

### ORM Patterns

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **N+1 queries** | `views.py`, `serializers.py` | `select_related`, `prefetch_related` usage |
| **Raw SQL** | `*.py` | `raw()`, `extra()` - injection risks |
| **Transactions** | `views.py` | `@transaction.atomic` for multi-model operations |
| **Migrations** | `migrations/` | Migration strategy, data migrations |

### Admin Exposure

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Admin URL** | `urls.py` | Is `/admin/` at default path or obscured? |
| **Admin customization** | `admin.py` | `ModelAdmin` classes, what's exposed |
| **Admin permissions** | `admin.py` | Who can access what models |

### Django Concerns Checklist

```markdown
**Django Patterns Observed:**
- [ ] SECRET_KEY from environment (settings.py)
- [ ] DEBUG=False in production config
- [ ] Custom user model: [Yes/No] - [location]
- [ ] Auth backend: [Default/Custom] - [type]
- [ ] Middleware order: [Correct/Incorrect] - [issues]
- [ ] N+1 prevention: [select_related observed/not observed]
- [ ] Admin exposure: [Default path/Custom path/Disabled]

**Django Concerns:**
- [List any issues found]
```

---

## FastAPI

### Dependency Injection

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Security deps** | `dependencies/`, `Depends()` | Auth/permission dependencies |
| **Database session** | `Depends()` usage | Session lifecycle, cleanup |
| **Config injection** | `Depends()` | Settings loaded how? |
| **Request validation** | Pydantic models | Input validation automatic |

### Middleware & CORS

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **CORS config** | `main.py`, `app.add_middleware()` | Origins, methods, headers allowed |
| **Auth middleware** | Custom middleware | JWT/session validation |
| **Request logging** | Middleware | What's logged, PII concerns |

### Exception Handling

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Global handlers** | `@app.exception_handler` | Custom error responses |
| **HTTP exceptions** | `HTTPException` usage | Consistent error format |
| **Validation errors** | Pydantic handling | Client-friendly messages |

### Response Models

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Response types** | `response_model=` | Output validation |
| **Exclude sensitive** | `response_model_exclude` | PII not leaked |
| **Status codes** | `status_code=` | Correct HTTP semantics |

### FastAPI Concerns Checklist

```markdown
**FastAPI Patterns Observed:**
- [ ] Auth dependency: [location, type]
- [ ] CORS configuration: [permissive/restricted]
- [ ] Database session injection: [pattern]
- [ ] Exception handlers: [global/per-route/none]
- [ ] Response models: [validated/unvalidated]

**FastAPI Concerns:**
- [List any issues found]
```

---

## Flask

### Application Factory

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Factory pattern** | `__init__.py`, `create_app()` | App creation pattern |
| **Config loading** | `config.py`, `from_object()` | Environment-based config |
| **Extension init** | `create_app()` | `init_app()` pattern for extensions |

### Blueprints

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Blueprint structure** | `Blueprint()` definitions | Route organization |
| **URL prefixes** | `register_blueprint()` | Consistent URL hierarchy |
| **Blueprint-local handlers** | Blueprint error handlers | Scoped error handling |

### Extensions

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Flask-Login** | `login_manager` setup | Session management |
| **Flask-SQLAlchemy** | `db` initialization | Database handling |
| **Flask-Migrate** | Migration setup | Schema management |
| **Flask-CORS** | CORS configuration | Cross-origin policy |

### Request Context

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Current user** | `current_user` usage | Auth state access |
| **Request data** | `request.form`, `request.json` | Input handling |
| **Session usage** | `session` object | Session data storage |

### Flask Concerns Checklist

```markdown
**Flask Patterns Observed:**
- [ ] App factory pattern: [Yes/No]
- [ ] Config management: [env-based/file-based/hardcoded]
- [ ] Blueprint organization: [by feature/by layer/monolithic]
- [ ] Auth extension: [Flask-Login/Custom/None]
- [ ] Database: [SQLAlchemy/raw/other]

**Flask Concerns:**
- [List any issues found]
```

---

# JavaScript/TypeScript Patterns

## Express.js

### Middleware Chain

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Middleware order** | `app.use()` calls | Order matters (body parser before routes) |
| **Error middleware** | 4-arg function | `(err, req, res, next)` signature |
| **Auth middleware** | Custom middleware | JWT/session validation |
| **CORS** | `cors()` middleware | Origin restrictions |

### Route Organization

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Router modules** | `Router()` usage | Resource-based separation |
| **Route mounting** | `app.use('/path', router)` | URL structure |
| **Controller pattern** | Handlers | Business logic separation |

### Error Handling

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Async errors** | `express-async-errors` or try/catch | Unhandled rejection handling |
| **Global error handler** | Final middleware | Consistent error responses |
| **404 handling** | Catch-all route | Not found responses |

### Express Concerns Checklist

```markdown
**Express Patterns Observed:**
- [ ] Middleware order: [correct/issues noted]
- [ ] Error handling: [global handler/per-route/none]
- [ ] CORS: [configured/wide open/none]
- [ ] Route organization: [resource-based/feature-based/flat]
- [ ] Auth middleware: [JWT/session/none]

**Express Concerns:**
- [List any issues found]
```

---

## React

### State Management

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Global state** | Redux/Zustand/Context | State management approach |
| **State location** | Store files | What's in global vs local state |
| **Async state** | React Query/SWR/Redux thunks | Data fetching pattern |

### Component Patterns

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Component types** | `components/` | Functional vs class |
| **Hooks usage** | `use*` functions | Custom hooks, dependency arrays |
| **Container/Presenter** | File structure | Separation of concerns |

### Route Protection

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Private routes** | Route config | Auth guards |
| **Role-based access** | Route components | Permission checks |
| **Redirect logic** | Auth components | Login redirect flow |

### Security Considerations

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **XSS prevention** | JSX usage | `dangerouslySetInnerHTML` usage |
| **Token storage** | Auth code | localStorage vs httpOnly cookies |
| **API calls** | API layer | Auth header injection |

### React Concerns Checklist

```markdown
**React Patterns Observed:**
- [ ] State management: [Redux/Zustand/Context/local only]
- [ ] Data fetching: [React Query/SWR/useEffect/other]
- [ ] Route protection: [implemented/partial/none]
- [ ] Component organization: [feature-based/type-based/flat]
- [ ] Token storage: [httpOnly cookie/localStorage/sessionStorage]

**React Concerns:**
- [List any issues found]
```

---

## Node.js (Runtime Patterns)

### Event Loop

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Blocking operations** | Sync fs, crypto | `*Sync` function usage |
| **CPU-intensive** | Computation code | Worker threads for heavy compute |
| **Async patterns** | Async/await usage | Proper error handling |

### Process Management

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Clustering** | `cluster` module, PM2 | Multi-core utilization |
| **Graceful shutdown** | SIGTERM handling | Connection draining |
| **Health checks** | Health endpoint | Liveness/readiness |

### Environment Config

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **dotenv usage** | `require('dotenv')` | Environment loading |
| **Config validation** | Config files | Required vars checked |
| **Secrets handling** | Environment access | Not logged, not committed |

### Node.js Concerns Checklist

```markdown
**Node.js Patterns Observed:**
- [ ] Blocking operations: [none found/found at locations]
- [ ] Process management: [PM2/cluster/single process]
- [ ] Graceful shutdown: [implemented/not implemented]
- [ ] Config approach: [dotenv/config module/hardcoded]

**Node.js Concerns:**
- [List any issues found]
```

---

# Rust Patterns

## Ownership & Borrowing

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Clone usage** | `.clone()` calls | Excessive cloning (performance) |
| **Lifetime annotations** | Function signatures | Complex lifetimes (maintainability) |
| **Arc/Rc usage** | Shared ownership | Reference counting patterns |
| **Mutex/RwLock** | Concurrent access | Lock contention potential |

## Error Handling

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Result types** | Function returns | Proper error propagation |
| **Error types** | `thiserror`, custom | Error type organization |
| **Unwrap usage** | `.unwrap()`, `.expect()` | Panic risks in production code |
| **? operator** | Error propagation | Consistent error handling |

### Error Handling Checklist

| Pattern | Good | Concerning |
|---------|------|------------|
| `.unwrap()` in lib code | Rare, justified | Frequent, unjustified |
| Custom error types | `thiserror` derive | String errors everywhere |
| Error context | `anyhow::Context` | Raw errors, no context |

## Unsafe Blocks

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **unsafe usage** | `unsafe {}` blocks | Justification, minimization |
| **FFI boundaries** | `extern` functions | Safety at boundaries |
| **Raw pointers** | `*const`, `*mut` | Necessity and safety |

### Unsafe Audit Checklist

```markdown
**Unsafe Usage:**
- [ ] Count of unsafe blocks: [N]
- [ ] Locations: [files:lines]
- [ ] Justification documented: [Yes/No]
- [ ] Minimized scope: [Yes/No]
- [ ] Safety invariants documented: [Yes/No]
```

## Async Patterns

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Runtime** | `tokio`, `async-std` | Which async runtime |
| **Spawn patterns** | `tokio::spawn` | Task lifecycle management |
| **Channel usage** | `mpsc`, `broadcast` | Communication patterns |
| **Cancellation** | `CancellationToken` | Graceful shutdown |

## Dependency Patterns

| Pattern | Location | What to Check |
|---------|----------|---------------|
| **Cargo.toml** | Dependencies section | Version constraints |
| **Feature flags** | `[features]` | Optional functionality |
| **Workspace** | `[workspace]` | Multi-crate organization |

## Rust Concerns Checklist

```markdown
**Rust Patterns Observed:**
- [ ] Error handling: [Result-based/panic-heavy]
- [ ] Unwrap usage: [minimal/concerning count]
- [ ] Unsafe blocks: [count, justified/unjustified]
- [ ] Async runtime: [tokio/async-std/none]
- [ ] Clone patterns: [minimal/excessive]

**Rust Concerns:**
- [List any issues found]
```

---

## Unknown or Custom Frameworks

### When Standard Patterns Don't Apply

If the codebase uses frameworks not covered in this reference:

**Step 1: Detect and Document**
```markdown
**Framework Detection:**
- Language: [Python/JavaScript/Rust/Other]
- Framework: [Name] - NOT in standard pattern reference
- Version: [if determinable]
- Type: [Web framework/ORM/Auth library/etc.]
```

**Step 2: Apply Language-Level Patterns Only**

When framework is unknown, fall back to language patterns:

| Language | Applicable Patterns |
|----------|-------------------|
| Python | Import structure, module organization, error handling, type hints |
| JavaScript | Module system (ESM/CJS), async patterns, error handling |
| Rust | Ownership, error handling, unsafe usage, async runtime |

**Step 3: Document Limitation**
```markdown
**Framework Analysis Limitation:**
- Framework [X] not in standard pattern reference
- Applied: General [language] patterns only
- NOT checked: Framework-specific security locations, configuration patterns
- Recommendation: Create custom checklist for [framework] if recurring
```

### Common Custom/Unknown Scenarios

| Scenario | Approach |
|----------|----------|
| **In-house framework** | Document architecture, apply language patterns, note custom conventions |
| **Obscure open-source** | Check for documentation, apply language patterns, lower confidence |
| **Framework fork** | Check if parent framework patterns apply, note deviations |
| **No framework (vanilla)** | Apply language patterns only, document as "framework-less" |

### Creating Ad-Hoc Checklists

For unknown frameworks, build minimal checklist from:

1. **Entry points** - Where do requests enter? (routes, handlers, main)
2. **Configuration** - Where are settings? (config files, env vars)
3. **Authentication** - How is auth handled? (middleware, decorators, guards)
4. **Data access** - How is data fetched? (ORM, raw queries, API calls)
5. **Error handling** - How are errors managed? (try/catch, Result types, middleware)

```markdown
**Ad-Hoc Checklist for [Framework Name]:**

| Category | Location | Observed Pattern |
|----------|----------|------------------|
| Entry points | [path] | [description] |
| Configuration | [path] | [description] |
| Authentication | [path] | [description] |
| Data access | [path] | [description] |
| Error handling | [path] | [description] |

**Confidence:** Medium - Custom checklist, not validated against framework best practices
```

### Confidence Adjustment for Unknown Frameworks

| Situation | Confidence Cap | Reason |
|-----------|---------------|--------|
| Unknown framework, good docs | Medium | Can't verify against known patterns |
| Unknown framework, no docs | Low | Limited verification possible |
| Custom in-house framework | Medium | Internal knowledge may exist |
| Vanilla (no framework) | High | Language patterns fully apply |

---

## Using Patterns in Analysis

### Integration with Catalog Entry

When analyzing a subsystem:

1. **Identify primary language/framework** from files
2. **Load relevant pattern section** from this reference
3. **Check each location** listed in pattern tables
4. **Document in catalog entry:**

```markdown
**Patterns Observed:**
- Django auth: Custom AUTH_USER_MODEL using AbstractUser (models/user.py:15)
- Middleware: 5 custom middlewares (settings.py:45-52)
- N+1 prevention: select_related used in queryset (views/api.py:89)

**Concerns:**
- Missing: CSRF protection disabled for API views (settings.py:78)
- Warning: DEBUG=True in settings.py (not env-based)
```

### Multi-Technology Projects

For projects using multiple technologies:

1. **Identify technology per subsystem** during discovery
2. **Apply relevant patterns** when analyzing each subsystem
3. **Note technology boundaries** in trust boundary analysis
4. **Document integration patterns** (e.g., Django backend + React frontend)

```markdown
**Technology Stack:**
- Backend: Django 4.2 (Python 3.11)
- Frontend: React 18 (TypeScript)
- API: Django REST Framework
- Database: PostgreSQL

**Integration Pattern:**
- React calls Django REST API via fetch
- JWT tokens for auth (djangorestframework-simplejwt)
- CORS configured for frontend origin
```

## Success Criteria

**You succeeded when:**
- Technology stack identified accurately
- Relevant pattern sections applied during analysis
- Framework-specific locations checked
- Findings documented with file:line evidence
- Technology-specific concerns captured

**You failed when:**
- Used generic analysis for framework project
- Missed framework-specific security locations
- Didn't check standard configuration files
- Documented patterns without evidence
- Ignored technology-specific anti-patterns
