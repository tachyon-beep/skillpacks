---
name: database-integration
description: Use when working with SQLAlchemy, database connection pooling, N+1 queries, migrations, transactions, or ORM vs raw SQL decisions - covers production patterns for connection management, query optimization, zero-downtime migrations, and testing strategies
---

# Database Integration

## Overview

**Database integration specialist covering SQLAlchemy, connection pooling, query optimization, migrations, transactions, and production patterns.**

**Core principle**: Databases are stateful, high-latency external systems requiring careful connection management, query optimization, and migration strategies to maintain performance and reliability at scale.

## When to Use This Skill

Use when encountering:

- **Connection pooling**: Pool exhaustion, "too many connections" errors, pool configuration
- **Query optimization**: N+1 queries, slow endpoints, eager loading strategies
- **Migrations**: Schema changes, zero-downtime deployments, data backfills
- **Transactions**: Multi-step operations, rollback strategies, isolation levels
- **ORM vs Raw SQL**: Complex queries, performance optimization, query readability
- **Testing**: Database test strategies, fixtures, test isolation
- **Monitoring**: Query performance tracking, connection pool health

**Do NOT use for**:
- Database selection (PostgreSQL vs MySQL vs MongoDB)
- Database administration (backup, replication, sharding)
- Schema design principles (see general architecture resources)

## Connection Pool Configuration

### Pool Sizing Formula

**Calculate pool size based on deployment architecture**:

```python
# Formula: pool_size × num_workers ≤ (postgres_max_connections - reserved)
# Example: 10 workers × 5 connections = 50 total ≤ (100 - 10) reserved

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

DATABASE_URL = "postgresql://user:pass@host/db"

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,              # Connections per worker
    max_overflow=10,          # Additional connections during spikes
    pool_pre_ping=True,       # CRITICAL: Verify connection before use
    pool_recycle=3600,        # Recycle after 1 hour (prevent stale connections)
    pool_timeout=30,          # Wait max 30s for connection from pool
    echo_pool=False,          # Enable for debugging pool issues
    connect_args={
        "connect_timeout": 10,
        "options": "-c statement_timeout=30000"  # 30s query timeout
    }
)
```

**Environment-based configuration**:

```python
import os
from pydantic import BaseSettings

class DatabaseSettings(BaseSettings):
    database_url: str
    pool_size: int = 5
    max_overflow: int = 10
    pool_pre_ping: bool = True
    pool_recycle: int = 3600

    class Config:
        env_file = ".env"

settings = DatabaseSettings()

engine = create_engine(
    settings.database_url,
    pool_size=settings.pool_size,
    max_overflow=settings.max_overflow,
    pool_pre_ping=settings.pool_pre_ping,
    pool_recycle=settings.pool_recycle
)
```

**Async configuration** (asyncpg):

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@host/db",
    pool_size=20,           # Async handles more concurrent connections
    max_overflow=0,         # No overflow - fail fast
    pool_pre_ping=False,    # asyncpg handles internally
    pool_recycle=3600
)

async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)
```

### Pool Health Monitoring

**Health check endpoint**:

```python
from fastapi import FastAPI, HTTPException
from sqlalchemy import text

app = FastAPI()

@app.get("/health/database")
async def database_health(db: Session = Depends(get_db)):
    """Check database connectivity and pool status"""
    try:
        # Simple query to verify connection
        result = db.execute(text("SELECT 1"))

        # Check pool statistics
        pool = db.get_bind().pool
        pool_status = {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total_connections": pool.size() + pool.overflow()
        }

        return {
            "status": "healthy",
            "pool": pool_status
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unhealthy: {e}")
```

**Pool exhaustion debugging**:

```python
import logging

logger = logging.getLogger(__name__)

# Enable pool event logging
from sqlalchemy import event

@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    logger.info(f"New connection created: {id(dbapi_conn)}")

@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    logger.debug(f"Connection checked out: {id(dbapi_conn)}")

    pool = connection_proxy._pool
    logger.debug(
        f"Pool status - size: {pool.size()}, "
        f"checked_out: {pool.checkedout()}, "
        f"overflow: {pool.overflow()}"
    )

@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    logger.debug(f"Connection checked in: {id(dbapi_conn)}")
```

### Testing with NullPool

**Disable pooling in tests**:

```python
from sqlalchemy.pool import NullPool

# Test configuration - no connection pooling
test_engine = create_engine(
    "postgresql://user:pass@localhost/test_db",
    poolclass=NullPool,  # No pooling - each query gets new connection
    echo=True            # Log all SQL queries
)
```

## Query Optimization

### N+1 Query Detection

**Automatic detection in tests**:

```python
from sqlalchemy import event
from sqlalchemy.engine import Engine
import pytest

class QueryCounter:
    """Count queries executed during test"""
    def __init__(self):
        self.queries = []

    def __enter__(self):
        event.listen(Engine, "before_cursor_execute", self._before_cursor_execute)
        return self

    def __exit__(self, *args):
        event.remove(Engine, "before_cursor_execute", self._before_cursor_execute)

    def _before_cursor_execute(self, conn, cursor, statement, *args):
        self.queries.append(statement)

    @property
    def count(self):
        return len(self.queries)

# Test usage
def test_no_n_plus_1():
    with QueryCounter() as counter:
        users = get_users_with_posts()  # Should use eager loading

        # Access posts (should not trigger additional queries)
        for user in users:
            _ = [post.title for post in user.posts]

        # Should be 1-2 queries, not 101
        assert counter.count <= 2, f"N+1 detected: {counter.count} queries"
```

### Eager Loading Strategies

**Decision matrix**:

| Pattern | Queries | Use When | Example |
|---------|---------|----------|---------|
| `joinedload()` | 1 (JOIN) | One-to-one, small one-to-many | User → Profile |
| `selectinload()` | 2 (IN clause) | One-to-many with many rows | User → Posts |
| `subqueryload()` | 2 (subquery) | Legacy alternative | Use selectinload instead |
| `raiseload()` | 0 (raises error) | Prevent lazy loading | Production safety |

**joinedload() - Single query with JOIN**:

```python
from sqlalchemy.orm import joinedload

# Single query: SELECT * FROM users LEFT OUTER JOIN posts ON ...
users = db.query(User).options(
    joinedload(User.posts)
).all()

# Best for one-to-one or small one-to-many
user = db.query(User).options(
    joinedload(User.profile)  # One-to-one
).filter(User.id == user_id).first()
```

**selectinload() - Two queries (more efficient for many rows)**:

```python
from sqlalchemy.orm import selectinload

# Query 1: SELECT * FROM users
# Query 2: SELECT * FROM posts WHERE user_id IN (1, 2, 3, ...)
users = db.query(User).options(
    selectinload(User.posts)
).all()

# Best for one-to-many with many related rows
```

**Nested eager loading**:

```python
# Load users → posts → comments (3 queries total)
users = db.query(User).options(
    selectinload(User.posts).selectinload(Post.comments)
).all()
```

**Conditional eager loading**:

```python
from sqlalchemy.orm import selectinload, Load

# Only load published posts
users = db.query(User).options(
    selectinload(User.posts).options(
        Load(Post).filter(Post.published == True)
    )
).all()
```

**Prevent lazy loading in production** (raiseload):

```python
from sqlalchemy.orm import raiseload

# Raise error if any relationship accessed without eager loading
users = db.query(User).options(
    raiseload('*')  # Disable all lazy loading
).all()

# This will raise an error:
# user.posts  # InvalidRequestError: 'User.posts' is not available due to lazy='raise'
```

### Query Performance Measurement

**Log slow queries**:

```python
from sqlalchemy import event
from sqlalchemy.engine import Engine
import time
import logging

logger = logging.getLogger(__name__)

SLOW_QUERY_THRESHOLD = 1.0  # seconds

@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.time())

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total_time = time.time() - conn.info['query_start_time'].pop()

    if total_time > SLOW_QUERY_THRESHOLD:
        logger.warning(
            f"Slow query ({total_time:.2f}s): {statement[:200]}",
            extra={
                "duration": total_time,
                "statement": statement,
                "parameters": parameters
            }
        )
```

**EXPLAIN ANALYZE for query optimization**:

```python
from sqlalchemy import text

def explain_query(db: Session, query):
    """Get query execution plan"""
    compiled = query.statement.compile(
        compile_kwargs={"literal_binds": True}
    )

    explain_result = db.execute(
        text(f"EXPLAIN ANALYZE {compiled}")
    ).fetchall()

    return "\n".join([row[0] for row in explain_result])

# Usage
query = db.query(User).join(Post).filter(Post.published == True)
plan = explain_query(db, query)
print(plan)
```

### Deferred Column Loading

**Exclude large columns from initial query**:

```python
from sqlalchemy.orm import defer, undefer

# Don't load large 'bio' column initially
users = db.query(User).options(
    defer(User.bio),  # Skip this column
    defer(User.profile_image)  # Skip binary data
).all()

# Load specific user's bio when needed
user = db.query(User).options(
    undefer(User.bio)  # Load this column
).filter(User.id == user_id).first()
```

**Load only specific columns**:

```python
from sqlalchemy.orm import load_only

# Only load id and name (ignore all other columns)
users = db.query(User).options(
    load_only(User.id, User.name)
).all()
```

## Zero-Downtime Migrations

### Migration Decision Matrix

| Operation | Locking | Approach | Downtime |
|-----------|---------|----------|----------|
| Add nullable column | None | Single migration | No |
| Add NOT NULL column | Table lock | Multi-phase (nullable → backfill → NOT NULL) | No |
| Add index | Share lock | `CREATE INDEX CONCURRENTLY` | No |
| Add foreign key | Share lock | `NOT VALID` → `VALIDATE` | No |
| Drop column | None | Multi-phase (stop using → drop) | No |
| Rename column | None | Multi-phase (add new → dual write → drop old) | No |
| Alter column type | Table lock | Multi-phase or rebuild table | Maybe |

### Multi-Phase NOT NULL Migration

**Phase 1: Add nullable column**:

```python
# migrations/versions/001_add_email_verified.py
def upgrade():
    # Fast: no table rewrite
    op.add_column('users', sa.Column('email_verified', sa.Boolean(), nullable=True))

    # Set default for new rows
    op.execute("ALTER TABLE users ALTER COLUMN email_verified SET DEFAULT false")

def downgrade():
    op.drop_column('users', 'email_verified')
```

**Phase 2: Backfill in batches**:

```python
# migrations/versions/002_backfill_email_verified.py
from alembic import op
import sqlalchemy as sa

def upgrade():
    """Backfill existing rows in batches"""
    connection = op.get_bind()

    # Process in batches to avoid long transactions
    batch_size = 10000
    total_updated = 0

    while True:
        result = connection.execute(sa.text("""
            UPDATE users
            SET email_verified = false
            WHERE email_verified IS NULL
            AND id IN (
                SELECT id FROM users
                WHERE email_verified IS NULL
                ORDER BY id
                LIMIT :batch_size
            )
        """), {"batch_size": batch_size})

        rows_updated = result.rowcount
        total_updated += rows_updated

        if rows_updated == 0:
            break

        print(f"Backfilled {total_updated} rows")

def downgrade():
    pass  # No rollback needed
```

**Phase 3: Add NOT NULL constraint**:

```python
# migrations/versions/003_make_email_verified_not_null.py
def upgrade():
    # Verify no NULLs remain
    connection = op.get_bind()
    result = connection.execute(sa.text(
        "SELECT COUNT(*) FROM users WHERE email_verified IS NULL"
    ))
    null_count = result.scalar()

    if null_count > 0:
        raise Exception(f"Cannot add NOT NULL: {null_count} NULL values remain")

    # Add NOT NULL constraint (fast since all values are set)
    op.alter_column('users', 'email_verified', nullable=False)

def downgrade():
    op.alter_column('users', 'email_verified', nullable=True)
```

### Concurrent Index Creation

**Without CONCURRENTLY (blocks writes)**:

```python
# BAD: Locks table during index creation
def upgrade():
    op.create_index('idx_users_email', 'users', ['email'])
```

**With CONCURRENTLY (no locks)**:

```python
# GOOD: No blocking, safe for production
def upgrade():
    # Requires raw SQL for CONCURRENTLY
    op.execute("""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email
        ON users (email)
    """)

def downgrade():
    op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_users_email")
```

**Partial index for efficiency**:

```python
def upgrade():
    op.execute("""
        CREATE INDEX CONCURRENTLY idx_users_active_email
        ON users (email)
        WHERE deleted_at IS NULL
    """)
```

### Adding Foreign Keys Without Blocking

**Using NOT VALID constraint**:

```python
# migrations/versions/004_add_foreign_key.py
def upgrade():
    # Phase 1: Add constraint without validating existing rows (fast)
    op.execute("""
        ALTER TABLE posts
        ADD CONSTRAINT fk_posts_user_id
        FOREIGN KEY (user_id)
        REFERENCES users (id)
        NOT VALID
    """)

    # Phase 2: Validate constraint in background (can be canceled/restarted)
    op.execute("""
        ALTER TABLE posts
        VALIDATE CONSTRAINT fk_posts_user_id
    """)

def downgrade():
    op.drop_constraint('fk_posts_user_id', 'posts', type_='foreignkey')
```

### Migration Monitoring

**Track migration progress**:

```sql
-- Check backfill progress
SELECT
    COUNT(*) FILTER (WHERE email_verified IS NULL) as null_count,
    COUNT(*) as total_count,
    ROUND(100.0 * COUNT(*) FILTER (WHERE email_verified IS NOT NULL) / COUNT(*), 2) as pct_complete
FROM users;

-- Check index creation progress (PostgreSQL 12+)
SELECT
    phase,
    ROUND(100.0 * blocks_done / NULLIF(blocks_total, 0), 2) as pct_complete
FROM pg_stat_progress_create_index
WHERE relid = 'users'::regclass;
```

## Transaction Management

### Basic Transaction Pattern

**Context manager with automatic rollback**:

```python
from contextlib import contextmanager
from sqlalchemy.orm import Session

@contextmanager
def transactional_session(db: Session):
    """Context manager for automatic rollback on error"""
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()

# Usage
with transactional_session(db) as session:
    user = User(name="Alice")
    session.add(user)
    # Automatic commit on success, rollback on exception
```

### Savepoints for Partial Rollback

**Nested transactions with savepoints**:

```python
def create_order_with_retry(db: Session, order_data: dict):
    """Use savepoints to retry failed steps without losing entire transaction"""
    # Start main transaction
    order = Order(**order_data)
    db.add(order)
    db.flush()  # Get order.id

    # Try payment with savepoint
    sp = db.begin_nested()  # Create savepoint
    try:
        payment = process_payment(order.total)
        order.payment_id = payment.id
    except PaymentError as e:
        sp.rollback()  # Rollback to savepoint (keep order)

        # Try alternative payment method
        sp = db.begin_nested()
        try:
            payment = process_backup_payment(order.total)
            order.payment_id = payment.id
        except PaymentError:
            sp.rollback()
            raise HTTPException(status_code=402, detail="All payment methods failed")

    db.commit()  # Commit entire transaction
    return order
```

### Locking Strategies

**Optimistic locking with version column**:

```python
from sqlalchemy import Column, Integer, String

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    inventory = Column(Integer)
    version = Column(Integer, nullable=False, default=1)  # Version column

# Usage
def decrement_inventory(db: Session, product_id: int, quantity: int):
    product = db.query(Product).filter(Product.id == product_id).first()

    if product.inventory < quantity:
        raise ValueError("Insufficient inventory")

    # Update with version check
    rows_updated = db.execute(
        sa.update(Product)
        .where(Product.id == product_id)
        .where(Product.version == product.version)  # Check version hasn't changed
        .values(
            inventory=Product.inventory - quantity,
            version=Product.version + 1
        )
    ).rowcount

    if rows_updated == 0:
        # Version mismatch - another transaction modified this row
        raise HTTPException(status_code=409, detail="Product was modified by another transaction")

    db.commit()
```

**Pessimistic locking with SELECT FOR UPDATE**:

```python
def decrement_inventory_with_lock(db: Session, product_id: int, quantity: int):
    """Acquire row lock to prevent concurrent modifications"""
    # Lock the row (blocks other transactions)
    product = db.query(Product).filter(
        Product.id == product_id
    ).with_for_update().first()  # SELECT ... FOR UPDATE

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    if product.inventory < quantity:
        raise HTTPException(status_code=400, detail="Insufficient inventory")

    product.inventory -= quantity
    db.commit()
    # Lock released after commit
```

**Lock timeout to prevent deadlocks**:

```python
from sqlalchemy import text

def with_lock_timeout(db: Session, timeout_ms: int = 5000):
    """Set lock timeout for this transaction"""
    db.execute(text(f"SET LOCAL lock_timeout = '{timeout_ms}ms'"))

# Usage
try:
    with_lock_timeout(db, 3000)  # 3 second timeout
    product = db.query(Product).with_for_update().filter(...).first()
except Exception as e:
    if "lock timeout" in str(e).lower():
        raise HTTPException(status_code=409, detail="Resource locked by another transaction")
    raise
```

### Isolation Levels

**Configure isolation level**:

```python
from sqlalchemy import create_engine

# Default: READ COMMITTED
engine = create_engine(
    DATABASE_URL,
    isolation_level="REPEATABLE READ"  # Options: READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE
)

# Per-transaction isolation
from sqlalchemy.orm import Session

with Session(engine) as session:
    session.connection(execution_options={"isolation_level": "SERIALIZABLE"})
    # ... transaction logic ...
```

## Raw SQL vs ORM

### Decision Matrix

| Use ORM When | Use Raw SQL When |
|--------------|------------------|
| CRUD operations | Complex CTEs (Common Table Expressions) |
| Simple joins (<3 tables) | Window functions with PARTITION BY |
| Type safety critical | Performance-critical queries |
| Database portability needed | Database-specific optimizations (PostgreSQL arrays, JSONB) |
| Code readability with ORM is good | ORM query becomes unreadable (>10 lines) |

### Raw SQL with Type Safety

**Parameterized queries with Pydantic results**:

```python
from sqlalchemy import text
from pydantic import BaseModel
from typing import List

class CustomerReport(BaseModel):
    id: int
    name: str
    region: str
    total_spent: float
    order_count: int
    rank_in_region: int

@app.get("/reports/top-customers")
def get_top_customers(
    db: Session = Depends(get_db),
    region: str = None,
    limit: int = 100
) -> List[CustomerReport]:
    """Complex report with CTEs and window functions"""
    query = text("""
        WITH customer_totals AS (
            SELECT
                u.id,
                u.name,
                u.region,
                COUNT(o.id) as order_count,
                COALESCE(SUM(o.total), 0) as total_spent
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.deleted_at IS NULL
                AND (:region IS NULL OR u.region = :region)
            GROUP BY u.id, u.name, u.region
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY region
                    ORDER BY total_spent DESC
                ) as rank_in_region
            FROM customer_totals
        )
        SELECT * FROM ranked
        WHERE total_spent > 0
        ORDER BY total_spent DESC
        LIMIT :limit
    """)

    result = db.execute(query, {"region": region, "limit": limit})

    # Type-safe results with Pydantic
    return [CustomerReport(**dict(row._mapping)) for row in result]
```

### Hybrid Approach

**Combine ORM and raw SQL**:

```python
def get_user_analytics(db: Session, user_id: int):
    """Use raw SQL for complex aggregation, ORM for simple queries"""

    # Complex aggregation in raw SQL
    analytics_query = text("""
        SELECT
            COUNT(*) as total_orders,
            SUM(total) as lifetime_value,
            AVG(total) as avg_order_value,
            MAX(created_at) as last_order_date,
            MIN(created_at) as first_order_date
        FROM orders
        WHERE user_id = :user_id
    """)

    analytics = db.execute(analytics_query, {"user_id": user_id}).first()

    # Simple ORM query for user details
    user = db.query(User).filter(User.id == user_id).first()

    return {
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email
        },
        "analytics": {
            "total_orders": analytics.total_orders,
            "lifetime_value": float(analytics.lifetime_value or 0),
            "avg_order_value": float(analytics.avg_order_value or 0),
            "first_order": analytics.first_order_date,
            "last_order": analytics.last_order_date
        }
    }
```

### Query Optimization Checklist

**Before optimizing**:

1. **Measure with EXPLAIN ANALYZE**:
   ```sql
   EXPLAIN ANALYZE
   SELECT * FROM users JOIN orders ON users.id = orders.user_id;
   ```

2. **Look for**:
   - Sequential scans on large tables → Add index
   - High loop counts → N+1 query problem
   - Hash joins on small tables → Consider nested loop
   - Sort operations → Consider index on ORDER BY columns

3. **Optimize**:
   - Add indexes on foreign keys, WHERE clauses, ORDER BY columns
   - Use LIMIT for pagination
   - Use EXISTS instead of IN for large subqueries
   - Denormalize for read-heavy workloads

**Index usage verification**:

```sql
-- Check if index is being used
EXPLAIN SELECT * FROM users WHERE email = 'test@example.com';
-- Look for "Index Scan using idx_users_email"

-- Check index statistics
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
WHERE tablename = 'users';
```

## Testing Strategies

### Test Database Setup

**Separate test database with fixtures**:

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine"""
    engine = create_engine(
        "postgresql://user:pass@localhost/test_db",
        poolclass=NullPool,  # No pooling in tests
        echo=True            # Log all queries
    )

    # Create all tables
    Base.metadata.create_all(engine)

    yield engine

    # Drop all tables after tests
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="function")
def db_session(test_engine):
    """Create fresh database session for each test"""
    connection = test_engine.connect()
    transaction = connection.begin()

    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    # Rollback transaction (undo all changes)
    session.close()
    transaction.rollback()
    connection.close()
```

### Factory Pattern for Test Data

**Use factories for consistent test data**:

```python
from factory import Factory, Faker, SubFactory
from factory.alchemy import SQLAlchemyModelFactory

class UserFactory(SQLAlchemyModelFactory):
    class Meta:
        model = User
        sqlalchemy_session = db_session

    name = Faker('name')
    email = Faker('email')
    created_at = Faker('date_time')

class PostFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Post
        sqlalchemy_session = db_session

    title = Faker('sentence')
    content = Faker('text')
    user = SubFactory(UserFactory)  # Auto-create related user

# Test usage
def test_get_user_posts(db_session):
    user = UserFactory.create()
    PostFactory.create_batch(5, user=user)  # Create 5 posts for user

    posts = db_session.query(Post).filter(Post.user_id == user.id).all()
    assert len(posts) == 5
```

### Testing Transactions

**Test rollback behavior**:

```python
def test_transaction_rollback(db_session):
    """Verify rollback on error"""
    user = User(name="Alice", email="alice@example.com")
    db_session.add(user)

    with pytest.raises(IntegrityError):
        # This should fail (duplicate email)
        user2 = User(name="Bob", email="alice@example.com")
        db_session.add(user2)
        db_session.commit()

    # Verify rollback occurred
    db_session.rollback()
    assert db_session.query(User).count() == 0
```

### Testing Migrations

**Test migration up and down**:

```python
from alembic import command
from alembic.config import Config

def test_migration_upgrade_downgrade():
    """Test migration can be applied and reversed"""
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)

    # Apply migration
    command.upgrade(alembic_cfg, "head")

    # Verify schema changes
    # ... assertions ...

    # Rollback migration
    command.downgrade(alembic_cfg, "-1")

    # Verify rollback
    # ... assertions ...
```

## Monitoring and Observability

### Query Performance Tracking

**Track slow queries with middleware**:

```python
from fastapi import Request
import time
import logging

logger = logging.getLogger(__name__)

@app.middleware("http")
async def track_db_queries(request: Request, call_next):
    """Track database query performance per request"""
    query_count = 0
    total_query_time = 0.0

    def track_query(conn, cursor, statement, parameters, context, executemany):
        nonlocal query_count, total_query_time
        start = time.time()

        # Execute query
        cursor.execute(statement, parameters)

        duration = time.time() - start
        query_count += 1
        total_query_time += duration

        if duration > 1.0:  # Log slow queries
            logger.warning(
                f"Slow query ({duration:.2f}s): {statement[:200]}",
                extra={
                    "duration": duration,
                    "path": request.url.path
                }
            )

    # Attach listener
    event.listen(Engine, "before_cursor_execute", track_query)

    response = await call_next(request)

    # Remove listener
    event.remove(Engine, "before_cursor_execute", track_query)

    # Add headers
    response.headers["X-DB-Query-Count"] = str(query_count)
    response.headers["X-DB-Query-Time"] = f"{total_query_time:.3f}s"

    return response
```

### Connection Pool Metrics

**Expose pool metrics for monitoring**:

```python
from prometheus_client import Gauge

pool_size_gauge = Gauge('db_pool_size', 'Number of connections in pool')
pool_checked_out_gauge = Gauge('db_pool_checked_out', 'Connections currently checked out')
pool_overflow_gauge = Gauge('db_pool_overflow', 'Overflow connections')

@app.on_event("startup")
async def start_pool_metrics():
    """Collect pool metrics periodically"""
    import asyncio

    async def collect_metrics():
        while True:
            pool = engine.pool
            pool_size_gauge.set(pool.size())
            pool_checked_out_gauge.set(pool.checkedout())
            pool_overflow_gauge.set(pool.overflow())

            await asyncio.sleep(10)  # Every 10 seconds

    asyncio.create_task(collect_metrics())
```

## Anti-Patterns

| Anti-Pattern | Why Bad | Fix |
|--------------|---------|-----|
| **No connection pooling** | Creates new connection per request (slow) | Use `create_engine()` with pool |
| **pool_pre_ping=False** | Fails on stale connections | Always `pool_pre_ping=True` in production |
| **Lazy loading in loops** | N+1 query problem | Use `joinedload()` or `selectinload()` |
| **No query timeout** | Slow queries block workers | Set `statement_timeout` in connect_args |
| **Large transactions** | Locks held too long, blocking | Break into smaller transactions |
| **No migration rollback** | Can't undo bad migrations | Always test downgrade path |
| **String interpolation in SQL** | SQL injection vulnerability | Use parameterized queries with `text()` |
| **No index on foreign keys** | Slow joins | Add index on all foreign key columns |
| **Blocking migrations** | Downtime during deployment | Use `CONCURRENTLY`, `NOT VALID` patterns |

## Cross-References

**Related skills**:
- **FastAPI dependency injection** → `fastapi-development` (database dependencies)
- **API testing** → `api-testing` (testing database code)
- **Microservices** → `microservices-architecture` (per-service databases)
- **Security** → `ordis-security-architect` (SQL injection, connection security)

## Further Reading

- **SQLAlchemy docs**: https://docs.sqlalchemy.org/
- **Alembic migrations**: https://alembic.sqlalchemy.org/
- **PostgreSQL performance**: https://www.postgresql.org/docs/current/performance-tips.html
- **Database Reliability Engineering** by Laine Campbell
