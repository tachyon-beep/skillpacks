---
name: load-testing-patterns
description: Use when designing load tests, choosing tools (k6, JMeter, Gatling), calculating concurrent users from DAU, interpreting latency degradation, identifying bottlenecks, or running spike/soak/stress tests - provides test patterns, anti-patterns, and load calculation frameworks
---

# Load Testing Patterns

## Overview

**Core principle:** Test realistic load patterns, not constant artificial load. Find limits before users do.

**Rule:** Load testing reveals system behavior under stress. Without it, production is your load test.

## Tool Selection Decision Tree

| Your Need | Protocol | Team Skills | Use | Why |
|-----------|----------|-------------|-----|-----|
| Modern API testing | HTTP/REST/GraphQL | JavaScript | **k6** | Best dev experience, CI/CD friendly |
| Enterprise/complex protocols | HTTP/SOAP/JMS/JDBC | Java/GUI comfort | **JMeter** | Mature, comprehensive protocols |
| Python team | HTTP/WebSocket | Python | **Locust** | Pythonic, easy scripting |
| High performance/complex scenarios | HTTP/gRPC | Scala/Java | **Gatling** | Best reports, high throughput |
| Cloud-native at scale | HTTP/WebSocket | Any (SaaS) | **Artillery, Flood.io** | Managed, distributed |

**First choice:** k6 (modern, scriptable, excellent CI/CD integration)

**Why not ApacheBench/wrk:** Too simple for realistic scenarios, no complex user flows

## Test Pattern Library

| Pattern | Purpose | Duration | When to Use |
|---------|---------|----------|-------------|
| **Smoke Test** | Verify test works | 1-2 min | Before every test run |
| **Load Test** | Normal/peak capacity | 10-30 min | Regular capacity validation |
| **Stress Test** | Find breaking point | 20-60 min | Understand limits |
| **Spike Test** | Sudden traffic surge | 5-15 min | Black Friday, launch events |
| **Soak Test** | Memory leaks, stability | 1-8 hours | Pre-release validation |
| **Capacity Test** | Max sustainable load | Variable | Capacity planning |

### Smoke Test

**Goal:** Verify test script works with minimal load

```javascript
// k6 smoke test
export let options = {
  vus: 1,
  duration: '1m',
  thresholds: {
    http_req_duration: ['p(95)<500'],  // 95% < 500ms
    http_req_failed: ['rate<0.01'],     // <1% errors
  }
}
```

**Purpose:** Catch test script bugs before running expensive full tests

### Load Test (Ramp-Up Pattern)

**Goal:** Test normal and peak expected load

```javascript
// k6 load test with ramp-up
export let options = {
  stages: [
    { duration: '5m', target: 100 },   // Ramp to normal load
    { duration: '10m', target: 100 },  // Hold at normal
    { duration: '5m', target: 200 },   // Ramp to peak
    { duration: '10m', target: 200 },  // Hold at peak
    { duration: '5m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.05'],
  }
}
```

**Pattern:** Gradual ramp-up → sustain → ramp down. Never start at peak.

### Stress Test (Breaking Point)

**Goal:** Find system limits

```javascript
// k6 stress test
export let options = {
  stages: [
    { duration: '5m', target: 100 },   // Normal
    { duration: '5m', target: 300 },   // Above peak
    { duration: '5m', target: 600 },   // 2x peak
    { duration: '5m', target: 900 },   // 3x peak (expect failure)
    { duration: '10m', target: 0 },    // Recovery
  ]
}
```

**Success:** Identify at what load system degrades (not necessarily breaking completely)

### Spike Test (Sudden Surge)

**Goal:** Test sudden traffic bursts (viral post, email campaign)

```javascript
// k6 spike test
export let options = {
  stages: [
    { duration: '1m', target: 100 },   // Normal
    { duration: '30s', target: 1000 }, // SPIKE to 10x
    { duration: '5m', target: 1000 },  // Hold spike
    { duration: '2m', target: 100 },   // Back to normal
    { duration: '5m', target: 100 },   // Recovery check
  ]
}
```

**Tests:** Auto-scaling, circuit breakers, rate limiting

### Soak Test (Endurance)

**Goal:** Find memory leaks, resource exhaustion over time

```javascript
// k6 soak test
export let options = {
  stages: [
    { duration: '5m', target: 100 },   // Ramp
    { duration: '4h', target: 100 },   // Soak (sustained load)
    { duration: '5m', target: 0 },     // Ramp down
  ]
}
```

**Monitor:** Memory growth, connection leaks, disk space, file descriptors

**Duration:** Minimum 1 hour, ideally 4-8 hours

## Load Calculation Framework

**Problem:** Convert "10,000 daily active users" to concurrent load

### Step 1: DAU to Concurrent Users

```
Concurrent Users = DAU × Concurrency Ratio × Peak Multiplier

Concurrency Ratios by App Type:
- Web apps: 5-10%
- Social media: 10-20%
- Business apps: 20-30% (work hours)
- Gaming: 15-25%

Peak Multiplier: 1.5-2x for safety margin
```

**Example:**
```
DAU = 10,000
Concurrency = 10% (web app)
Peak Multiplier = 1.5

Concurrent Users = 10,000 × 0.10 × 1.5 = 1,500 concurrent users
```

### Step 2: Concurrent Users to Requests/Second

```
RPS = (Concurrent Users × Requests per Session) / (Session Duration × Think Time Ratio)

Think Time Ratio:
- Active browsing: 0.3-0.5 (30-50% time clicking/typing)
- Reading-heavy: 0.1-0.2 (10-20% active)
- API clients: 0.8-1.0 (80-100% active)
```

**Example:**
```
Concurrent Users = 1,500
Requests per Session = 20
Session Duration = 10 minutes = 600 seconds
Think Time Ratio = 0.3 (web browsing)

RPS = (1,500 × 20) / (600 × 0.3) = 30,000 / 180 = 167 RPS
```

### Step 3: Model Realistic Patterns

Don't use constant load. Use realistic traffic patterns:

```javascript
// Realistic daily pattern
export let options = {
  stages: [
    // Morning ramp
    { duration: '2h', target: 500 },    // 08:00-10:00
    { duration: '2h', target: 1000 },   // 10:00-12:00 (peak)
    // Lunch dip
    { duration: '1h', target: 600 },    // 12:00-13:00
    // Afternoon peak
    { duration: '2h', target: 1200 },   // 13:00-15:00 (peak)
    { duration: '2h', target: 800 },    // 15:00-17:00
    // Evening drop
    { duration: '2h', target: 300 },    // 17:00-19:00
  ]
}
```

## Anti-Patterns Catalog

### ❌ Coordinated Omission
**Symptom:** Fixed rate load generation ignores slow responses, underestimating latency

**Why bad:** Hides real latency impact when system slows down

**Fix:** Use arrival rate (requests/sec) not iteration rate

```javascript
// ❌ Bad - coordinated omission
export default function() {
  http.get('https://api.example.com')
  sleep(1)  // Wait 1s between requests
}

// ✅ Good - arrival rate pacing
export let options = {
  scenarios: {
    constant_arrival_rate: {
      executor: 'constant-arrival-rate',
      rate: 100,  // 100 RPS regardless of response time
      timeUnit: '1s',
      duration: '10m',
      preAllocatedVUs: 50,
      maxVUs: 200,
    }
  }
}
```

---

### ❌ Cold Start Testing
**Symptom:** Running load test immediately after deployment without warm-up

**Why bad:** JIT compilation, cache warming, connection pooling haven't stabilized

**Fix:** Warm-up phase before measurement

```javascript
// ✅ Good - warm-up phase
export let options = {
  stages: [
    { duration: '2m', target: 50 },    // Warm-up (not measured)
    { duration: '10m', target: 100 },  // Actual test
  ]
}
```

---

### ❌ Unrealistic Test Data
**Symptom:** Using same user ID, same query parameters for all virtual users

**Why bad:** Caches give unrealistic performance, doesn't test real database load

**Fix:** Parameterized, realistic data

```javascript
// ❌ Bad - same data
http.get('https://api.example.com/users/123')

// ✅ Good - parameterized data
import { SharedArray } from 'k6/data'
import papaparse from 'https://jslib.k6.io/papaparse/5.1.1/index.js'

const csvData = new SharedArray('users', function () {
  return papaparse.parse(open('./users.csv'), { header: true }).data
})

export default function() {
  const user = csvData[__VU % csvData.length]
  http.get(`https://api.example.com/users/${user.id}`)
}
```

---

### ❌ Constant Load Pattern
**Symptom:** Running with constant VUs instead of realistic traffic pattern

**Why bad:** Real traffic has peaks, valleys, not flat line

**Fix:** Use realistic daily/hourly patterns

---

### ❌ Ignoring Think Time
**Symptom:** No delays between requests, hammering API as fast as possible

**Why bad:** Unrealistic user behavior, overestimates load

**Fix:** Add realistic think time based on user behavior

```javascript
// ✅ Good - realistic think time
import { sleep } from 'k6'

export default function() {
  http.get('https://api.example.com/products')
  sleep(Math.random() * 3 + 2)  // 2-5 seconds browsing

  http.post('https://api.example.com/cart', {...})
  sleep(Math.random() * 5 + 5)  // 5-10 seconds deciding

  http.post('https://api.example.com/checkout', {...})
}
```

## Result Interpretation Guide

### Latency Degradation Patterns

| Pattern | Cause | What to Check |
|---------|-------|---------------|
| **Linear growth** (2x users → 2x latency) | CPU-bound | Thread pool, CPU usage |
| **Exponential growth** (2x users → 10x latency) | Resource saturation | Connection pools, locks, queues |
| **Sudden cliff** (works until X, then fails) | Hard limit hit | Max connections, memory, file descriptors |
| **Gradual degradation** (slow increase over time) | Memory leak, cache pollution | Memory trends, GC activity |

### Bottleneck Classification

**Symptom: p95 latency 10x at 2x load**
→ **Resource saturation** (database connection pool, thread pool, queue)

**Symptom: Errors increase with load**
→ **Hard limit** (connection limit, rate limiting, timeout)

**Symptom: Latency grows over time at constant load**
→ **Memory leak** or **cache pollution**

**Symptom: High variance (p50 good, p99 terrible)**
→ **GC pauses**, **lock contention**, or **slow queries**

### What to Monitor

| Layer | Metrics to Track |
|-------|------------------|
| **Application** | Request rate, error rate, p50/p95/p99 latency, active requests |
| **Runtime** | GC pauses (JVM, .NET), thread pool usage, heap/memory |
| **Database** | Connection pool usage, query latency, lock waits, slow queries |
| **Infrastructure** | CPU %, memory %, disk I/O, network throughput |
| **External** | Third-party API latency, rate limit hits |

### Capacity Planning Formula

```
Safe Capacity = (Breaking Point × Degradation Factor) × Safety Margin

Breaking Point = VUs where p95 latency > threshold
Degradation Factor = 0.7 (start degradation before break)
Safety Margin = 0.5-0.7 (handle traffic spikes)

Example:
- System breaks at 1000 VUs (p95 > 1s)
- Start seeing degradation at 700 VUs (70%)
- Safe capacity: 700 × 0.7 = 490 VUs
```

## Authentication and Session Management

**Problem:** Real APIs require authentication. Can't use same token for all virtual users.

### Token Strategy Decision Framework

| Scenario | Strategy | Why |
|----------|----------|-----|
| **Short test (<10 min)** | Pre-generate tokens | Fast, simple, no login load |
| **Long test (soak)** | Login during test + refresh | Realistic, tests auth system |
| **Testing auth system** | Simulate login flow | Auth is part of load |
| **Read-only testing** | Shared token (single user) | Simplest, adequate for API-only tests |

**Default:** Pre-generate tokens for load tests, simulate login for auth system tests

### Pre-Generated Tokens Pattern

**Best for:** API testing where auth system isn't being tested

```javascript
// k6 with pre-generated JWT tokens
import http from 'k6/http'
import { SharedArray } from 'k6/data'

// Load tokens from file (generated externally)
const tokens = new SharedArray('auth tokens', function () {
  return JSON.parse(open('./tokens.json'))
})

export default function() {
  const token = tokens[__VU % tokens.length]

  const headers = {
    'Authorization': `Bearer ${token}`
  }

  http.get('https://api.example.com/protected', { headers })
}
```

**Generate tokens externally:**

```bash
# Script to generate 1000 tokens
for i in {1..1000}; do
  curl -X POST https://api.example.com/login \
    -d "username=loadtest_user_$i&password=test" \
    | jq -r '.token'
done > tokens.json
```

**Pros:** No login load, fast test setup
**Cons:** Tokens may expire during long tests, not testing auth flow

---

### Login Flow Simulation Pattern

**Best for:** Testing auth system, soak tests where tokens expire

```javascript
// k6 with login simulation
import http from 'k6/http'
import { SharedArray } from 'k6/data'

const users = new SharedArray('users', function () {
  return JSON.parse(open('./users.json'))  // [{username, password}, ...]
})

export default function() {
  const user = users[__VU % users.length]

  // Login to get token
  const loginRes = http.post('https://api.example.com/login', {
    username: user.username,
    password: user.password
  })

  const token = loginRes.json('token')

  // Use token for subsequent requests
  const headers = { 'Authorization': `Bearer ${token}` }

  http.get('https://api.example.com/protected', { headers })
  http.post('https://api.example.com/data', {}, { headers })
}
```

**Token refresh for long tests:**

```javascript
// k6 with token refresh
import { sleep } from 'k6'

let token = null
let tokenExpiry = 0

export default function() {
  const now = Date.now() / 1000

  // Refresh token if expired or about to expire
  if (!token || now > tokenExpiry - 300) {  // Refresh 5 min before expiry
    const loginRes = http.post('https://api.example.com/login', {...})
    token = loginRes.json('token')
    tokenExpiry = loginRes.json('expires_at')
  }

  http.get('https://api.example.com/protected', {
    headers: { 'Authorization': `Bearer ${token}` }
  })

  sleep(1)
}
```

---

### Session Cookie Management

**For cookie-based auth:**

```javascript
// k6 with session cookies
import http from 'k6/http'

export default function() {
  // k6 automatically handles cookies with jar
  const jar = http.cookieJar()

  // Login (sets session cookie)
  http.post('https://example.com/login', {
    username: 'user',
    password: 'pass'
  })

  // Subsequent requests use session cookie automatically
  http.get('https://example.com/dashboard')
  http.get('https://example.com/profile')
}
```

---

### Rate Limiting Detection

**Pattern:** Detect when hitting rate limits during load test

```javascript
// k6 rate limit detection
import { check } from 'k6'

export default function() {
  const res = http.get('https://api.example.com/data')

  check(res, {
    'not rate limited': (r) => r.status !== 429
  })

  if (res.status === 429) {
    console.warn(`Rate limited at VU ${__VU}, iteration ${__ITER}`)
    const retryAfter = res.headers['Retry-After']
    console.warn(`Retry-After: ${retryAfter} seconds`)
  }
}
```

**Thresholds for rate limiting:**

```javascript
export let options = {
  thresholds: {
    'http_req_failed{status:429}': ['rate<0.01']  // <1% rate limited
  }
}
```

## Third-Party Dependency Handling

**Problem:** APIs call external services (payment, email, third-party APIs). Should you mock them?

### Mock vs Real Decision Framework

| External Service | Mock or Real? | Why |
|------------------|---------------|-----|
| **Payment gateway** | Real (sandbox) | Need to test integration, has sandbox mode |
| **Email provider** | Mock | Cost ($0.001/email × 1000 VUs = expensive), no value testing |
| **Third-party API (has staging)** | Real (staging) | Test integration, realistic latency |
| **Third-party API (no staging)** | Mock | Can't load test production, rate limits |
| **Internal microservices** | Real | Testing real integration points |
| **Analytics/tracking** | Mock | High volume, no functional impact |

**Rule:** Use real services if they have sandbox/staging. Mock if expensive, rate-limited, or no test environment.

---

### Service Virtualization with WireMock

**Best for:** Mocking HTTP APIs with realistic responses

```javascript
// k6 test pointing to WireMock
export default function() {
  // WireMock running on localhost:8080 mocks external API
  const res = http.get('http://localhost:8080/api/payment/process')

  check(res, {
    'payment mock responds': (r) => r.status === 200
  })
}
```

**WireMock stub setup:**

```json
{
  "request": {
    "method": "POST",
    "url": "/api/payment/process"
  },
  "response": {
    "status": 200,
    "jsonBody": {
      "transaction_id": "{{randomValue type='UUID'}}",
      "status": "approved"
    },
    "headers": {
      "Content-Type": "application/json"
    },
    "fixedDelayMilliseconds": 200
  }
}
```

**Why WireMock:** Realistic latency simulation, dynamic responses, stateful mocking

---

### Partial Mocking Pattern

**Pattern:** Mock some services, use real for others

```javascript
// k6 with partial mocking
import http from 'k6/http'

export default function() {
  // Real API (points to staging)
  const productRes = http.get('https://staging-api.example.com/products')

  // Mock email service (points to WireMock)
  http.post('http://localhost:8080/mock/email/send', {
    to: 'user@example.com',
    subject: 'Order confirmation'
  })

  // Real payment sandbox
  http.post('https://sandbox-payment.stripe.com/charge', {
    amount: 1000,
    currency: 'usd',
    source: 'tok_visa'
  })
}
```

**Decision criteria:**
- Real: Services with sandbox, need integration validation, low cost
- Mock: No sandbox, expensive, rate-limited, testing failure scenarios

---

### Testing External Service Failures

**Use mocks to simulate failures:**

```javascript
// WireMock stub for failure scenarios
{
  "request": {
    "method": "POST",
    "url": "/api/payment/process"
  },
  "response": {
    "status": 503,
    "jsonBody": {
      "error": "Service temporarily unavailable"
    },
    "fixedDelayMilliseconds": 5000  // Slow failure
  }
}
```

**k6 test for resilience:**

```javascript
export default function() {
  const res = http.post('http://localhost:8080/api/payment/process', {})

  // Verify app handles payment failures gracefully
  check(res, {
    'handles payment failure': (r) => r.status === 503,
    'returns within timeout': (r) => r.timings.duration < 6000
  })
}
```

---

### Cost and Compliance Guardrails

**Before testing with real external services:**

| Check | Why |
|-------|-----|
| **Sandbox mode exists?** | Avoid production costs/rate limits |
| **Cost per request?** | 1000 VUs × 10 req/s × 600s = 6M requests |
| **Rate limits?** | Will you hit external service limits? |
| **Terms of service?** | Does load testing violate TOS? |
| **Data privacy?** | Using real user emails/PII? |

**Example cost calculation:**

```
Email service: $0.001/email
Load test: 100 VUs × 5 emails/session × 600s = 300,000 emails
Cost: 300,000 × $0.001 = $300

Decision: Mock email service, use real payment sandbox (free)
```

**Compliance:**
- Don't use real user data in load tests (GDPR, privacy)
- Check third-party TOS (some prohibit load testing)
- Use synthetic test data only

## Your First Load Test

**Goal:** Basic load test in one day

**Hour 1-2: Install tool and write smoke test**

```bash
# Install k6
brew install k6  # macOS
# or snap install k6  # Linux

# Create test.js
cat > test.js <<'EOF'
import http from 'k6/http'
import { check, sleep } from 'k6'

export let options = {
  vus: 1,
  duration: '30s'
}

export default function() {
  let res = http.get('https://your-api.com/health')
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response < 500ms': (r) => r.timings.duration < 500
  })
  sleep(1)
}
EOF

# Run smoke test
k6 run test.js
```

**Hour 3-4: Calculate target load**

```
Your DAU: 10,000
Concurrency: 10%
Peak multiplier: 1.5
Target: 10,000 × 0.10 × 1.5 = 1,500 VUs
```

**Hour 5-6: Write load test with ramp-up**

```javascript
export let options = {
  stages: [
    { duration: '5m', target: 750 },   // Ramp to normal (50%)
    { duration: '10m', target: 750 },  // Hold normal
    { duration: '5m', target: 1500 },  // Ramp to peak
    { duration: '10m', target: 1500 }, // Hold peak
    { duration: '5m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.05']  // < 5% errors
  }
}
```

**Hour 7-8: Run test and analyze**

```bash
# Run load test
k6 run --out json=results.json test.js

# Check summary output for:
# - p95/p99 latency trends
# - Error rates
# - When degradation started
```

**If test fails:** Check thresholds, adjust targets, investigate bottlenecks

## Common Mistakes

### ❌ Testing Production Without Safeguards
**Fix:** Use feature flags, test environment, or controlled percentage

---

### ❌ No Baseline Performance Metrics
**Fix:** Run smoke test first to establish baseline before load testing

---

### ❌ Using Iteration Duration Instead of Arrival Rate
**Fix:** Use `constant-arrival-rate` executor in k6

---

### ❌ Not Warming Up Caches/JIT
**Fix:** 2-5 minute warm-up phase before measurement

## Quick Reference

**Tool Selection:**
- Modern API: k6
- Enterprise: JMeter
- Python team: Locust

**Test Patterns:**
- Smoke: 1 VU, 1 min
- Load: Ramp-up → peak → ramp-down
- Stress: Increase until break
- Spike: Sudden 10x surge
- Soak: 4-8 hours constant

**Load Calculation:**
```
Concurrent = DAU × 0.10 × 1.5
RPS = (Concurrent × Requests/Session) / (Duration × Think Time)
```

**Anti-Patterns:**
- Coordinated omission (use arrival rate)
- Cold start (warm-up first)
- Unrealistic data (parameterize)
- Constant load (use realistic patterns)

**Result Interpretation:**
- Linear growth → CPU-bound
- Exponential growth → Resource saturation
- Sudden cliff → Hard limit
- Gradual degradation → Memory leak

**Authentication:**
- Short tests: Pre-generate tokens
- Long tests: Login + refresh
- Testing auth: Simulate login flow

**Third-Party Dependencies:**
- Has sandbox: Use real (staging/sandbox)
- Expensive/rate-limited: Mock (WireMock)
- No sandbox: Mock

## Bottom Line

**Start with smoke test (1 VU). Calculate realistic load from DAU. Use ramp-up pattern (never start at peak). Monitor p95/p99 latency. Find breaking point before users do.**

Test realistic scenarios with think time, not hammer tests.
