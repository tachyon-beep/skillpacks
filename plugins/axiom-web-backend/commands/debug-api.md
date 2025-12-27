---
description: Debug API issues - performance problems, errors, connection issues, and unexpected behavior
allowed-tools: ["Read", "Bash", "Glob", "Grep", "Skill"]
argument-hint: "[symptom_or_endpoint]"
---

# Debug API Command

Diagnose and fix API issues including performance problems, errors, and unexpected behavior.

## Core Principle

**Reproduce first, then isolate. Most API bugs are at boundaries: input validation, database queries, or external calls.**

## Common Symptoms and Causes

| Symptom | Likely Causes | First Check |
|---------|---------------|-------------|
| **500 errors** | Unhandled exception, DB error | Application logs |
| **Slow responses** | N+1 queries, missing indexes | Database query time |
| **Timeouts** | External service, deadlock | Connection pool status |
| **Intermittent failures** | Race condition, connection exhaustion | Concurrent request count |
| **Wrong data** | Caching, stale reads | Cache invalidation |
| **Auth failures** | Token expiry, clock skew | Token timestamps |

## Debugging Protocol

### Step 1: Reproduce the Issue

```bash
# Capture failing request
curl -v -X POST "http://localhost:8000/api/endpoint" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"key": "value"}' \
  2>&1 | tee debug_output.txt

# Check response code and timing
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/api/endpoint"
```

### Step 2: Check Logs

```bash
# Application logs (last 100 lines around error)
tail -100 /var/log/app/error.log | grep -A5 -B5 "ERROR\|Exception"

# Request logs
grep "endpoint_name" /var/log/app/access.log | tail -20

# Database logs (slow queries)
grep "duration:" /var/log/postgresql/postgresql.log | awk '$NF > 100'
```

### Step 3: Profile the Request

```python
# FastAPI - Add timing middleware
import time
from fastapi import Request

@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    response.headers["X-Process-Time"] = str(duration)
    return response
```

### Step 4: Isolate the Component

```
Request → [Validation] → [Business Logic] → [Database] → [Response]
                ↓               ↓                ↓
           Input error?    Logic bug?      Query slow?
```

Test each component independently:
1. **Validation**: Try with minimal valid input
2. **Business logic**: Call service directly in tests
3. **Database**: Run query directly in DB client
4. **External services**: Mock and test

## Issue-Specific Debugging

### Slow API Responses

```bash
# Check database query time
EXPLAIN ANALYZE SELECT * FROM users WHERE ...;

# Check connection pool
SELECT count(*) FROM pg_stat_activity WHERE datname = 'mydb';

# Check N+1 queries (look for repeated queries)
grep "SELECT" /var/log/app/sql.log | sort | uniq -c | sort -rn | head
```

**Common fixes:**
- Add database indexes
- Use eager loading (joinedload, select_related)
- Add response caching
- Optimize query structure

### 500 Internal Server Errors

```bash
# Find the exception
grep -A 20 "500\|Internal Server Error" /var/log/app/error.log

# Check for unhandled exceptions
grep "Traceback\|Exception" /var/log/app/error.log | tail -50
```

**Common fixes:**
- Add proper exception handling
- Validate input before processing
- Check for null/None values
- Handle database connection errors

### Connection Timeouts

```bash
# Check connection pool exhaustion
netstat -an | grep :5432 | grep ESTABLISHED | wc -l

# Check for long-running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE state = 'active' AND now() - pg_stat_activity.query_start > interval '5 seconds';
```

**Common fixes:**
- Increase connection pool size
- Add connection timeouts
- Close connections properly
- Add circuit breakers for external services

### Authentication Failures

```bash
# Check token validity
echo $TOKEN | cut -d'.' -f2 | base64 -d 2>/dev/null | jq '.exp'

# Compare with current time
date +%s

# Check clock skew between servers
ntpq -p
```

**Common fixes:**
- Extend token expiry
- Sync server clocks (NTP)
- Add clock skew tolerance
- Check token refresh logic

### Intermittent Failures

```bash
# Run load test to reproduce
ab -n 1000 -c 100 http://localhost:8000/api/endpoint

# Check for race conditions
grep -E "deadlock|lock wait" /var/log/app/error.log
```

**Common fixes:**
- Add proper locking
- Use database transactions
- Implement idempotency
- Add retry logic with backoff

## Output Format

```markdown
## API Debug Report: [Endpoint/Issue]

### Issue Summary

**Symptom**: [What's happening]
**Severity**: [Critical/High/Medium/Low]
**Frequency**: [Always/Intermittent/Under load]

### Root Cause

**Component**: [Validation/Logic/Database/External]
**Cause**: [Specific issue identified]
**Evidence**: [Logs, metrics, reproduction steps]

### Investigation Steps

1. [What was checked]
   - Finding: [Result]
2. [Next check]
   - Finding: [Result]

### Solution

**Immediate fix**:
```[language]
[Code fix]
```

**Why this works**: [Explanation]

### Prevention

- [ ] Add test case for this scenario
- [ ] Add monitoring/alerting
- [ ] Update documentation

### Related Issues

- [Similar issues to watch for]
```

## Quick Diagnostic Commands

```bash
# API health check
curl -s http://localhost:8000/health | jq

# Response time baseline
for i in {1..10}; do
  curl -s -o /dev/null -w "%{time_total}\n" http://localhost:8000/api/endpoint
done | awk '{sum+=$1} END {print "Avg:", sum/NR, "s"}'

# Concurrent request test
seq 1 100 | xargs -P 10 -I {} curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/api/endpoint | sort | uniq -c

# Memory usage during request
ps aux | grep uvicorn | awk '{print $4}'
```

## Cross-Pack Discovery

```python
import glob

# For database debugging
python_pack = glob.glob("plugins/axiom-python-engineering/plugin.json")
if python_pack:
    print("For Python debugging patterns: use axiom-python-engineering")

# For performance testing
quality_pack = glob.glob("plugins/ordis-quality-engineering/plugin.json")
if quality_pack:
    print("For load testing: use ordis-quality-engineering")
```

## Load Detailed Guidance

For database issues:
```
Load skill: axiom-web-backend:using-web-backend
Then read: database-integration.md
```

For authentication issues:
```
Then read: api-authentication.md
```
