---
name: reliability-engineering
description: Use when establishing system reliability through SLOs, error budgets, graceful degradation, circuit breakers, retries with backoff, bulkhead isolation, and SRE practices for production resilience
---

# Reliability Engineering

## Overview

Reliability engineering (SRE - Site Reliability Engineering) applies software engineering principles to operations. Define reliability targets (SLOs), measure error budgets, implement resilience patterns (circuit breakers, retries, graceful degradation), and balance feature velocity with stability.

**Core Principle**: Define SLOs first, calculate error budget, use budget to make deployment decisions. Implement circuit breakers to prevent cascading failures. Retry with exponential backoff. Design for graceful degradation.

**Ordis Identity**: Reliability is the foundation - structural integrity that supports all functionality, systematic defenses against cascading failures, resilience through engineering discipline.

## When to Use

**Use this skill when**:
- Defining reliability targets (SLOs, SLAs)
- System has cascading failures
- Need graceful degradation during outages
- Implementing circuit breakers, retries, timeouts
- Balancing feature development vs stability
- Calculating error budgets
- Improving system uptime

**Don't use for**:
- Initial prototyping (premature optimization)
- Systems without monitoring (implement observability first)

## SLI/SLO/Error Budget

### SLI (Service Level Indicator)

**What**: Quantitative measure of service behavior.

**Examples**:
- Request success rate
- p95/p99 latency
- System availability (uptime %)
- Throughput (requests/second)

```
SLI: Percentage of requests returning 2xx or 3xx status
SLI: p95 latency of API requests
SLI: Uptime percentage (ratio of successful health checks)
```

### SLO (Service Level Objective)

**What**: Target value for SLI. Internal reliability goal.

**Format**: "X% of [SLI] over [time window]"

**Examples**:
```
SLO: 99.9% of requests succeed (error rate < 0.1%)
SLO: 99% of requests have p95 latency < 200ms
SLO: 99.95% uptime (4.38 hours downtime/year allowed)
```

### Error Budget

**What**: Allowed failure rate = (100% - SLO)

```
SLO: 99.9% availability
Error budget: 0.1% = 43.8 minutes downtime per month

Month with 30 days = 43,200 minutes
Error budget = 43,200 × 0.001 = 43.2 minutes
```

**Using error budget**:
```
If error budget remaining:
  → Deploy new features
  → Take risks
  → Move fast

If error budget exhausted:
  → Stop feature development
  → Focus on reliability
  → Fix issues, add monitoring
  → Improve tests
```

**Example decision**:
```
Week 1: Deploy 5 features, 10 min downtime (budget: 43 min)
Week 2: Deploy 3 features, 15 min downtime (budget: 18 min remaining)
Week 3: Deploy 1 feature, causes 20 min downtime (budget exhausted!)

Action: FREEZE deployments, focus on stability
  - Fix issues from Week 3 deployment
  - Add more tests
  - Improve monitoring
  - Resume deployments next month
```

## Circuit Breaker Pattern

**Problem**: Cascading failures

```
Service A calls Service B
Service B is down (100% errors)
Service A keeps calling (wasting resources)
Service A gets overwhelmed (cascading failure)
```

**Solution**: Circuit breaker stops calls to failing service.

### States

```
┌─────────┐ Success rate OK ┌────────┐
│         │◄─────────────────┤        │
│ Closed  │                  │  Open  │
│ (Normal)│                  │(Failing)
│         │──────────────────►│        │
└─────────┘  Failures exceed └────────┘
             threshold            │
                 ▲                │
                 │                │ After timeout,
                 │                │ try again
                 │            ┌───▼────┐
                 │            │        │
                 └────────────┤Half-   │
                  Success     │Open    │
                              │(Testing)
                              └────────┘
```

**Closed**: Normal operation, requests pass through
**Open**: Too many failures, requests fail fast (don't call service)
**Half-Open**: After timeout, try a few requests to test if service recovered

### Implementation

```javascript
class CircuitBreaker {
  constructor({ failureThreshold = 5, timeout = 60000 }) {
    this.failureThreshold = failureThreshold;
    this.timeout = timeout;
    this.state = 'CLOSED';
    this.failureCount = 0;
    this.nextAttempt = Date.now();
  }

  async call(fn) {
    if (this.state === 'OPEN') {
      if (Date.now() < this.nextAttempt) {
        throw new Error('Circuit breaker is OPEN');
      }
      this.state = 'HALF_OPEN';
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  onSuccess() {
    this.failureCount = 0;
    this.state = 'CLOSED';
  }

  onFailure() {
    this.failureCount++;
    if (this.failureCount >= this.failureThreshold) {
      this.state = 'OPEN';
      this.nextAttempt = Date.now() + this.timeout;
    }
  }
}

// Usage
const paymentCircuit = new CircuitBreaker({
  failureThreshold: 5,  // Open after 5 failures
  timeout: 60000        // Try again after 60 seconds
});

async function processPayment(orderId) {
  try {
    return await paymentCircuit.call(() => paymentService.charge(orderId));
  } catch (error) {
    if (error.message === 'Circuit breaker is OPEN') {
      // Fallback: Queue payment for later
      await paymentQueue.enqueue(orderId);
      return { status: 'queued' };
    }
    throw error;
  }
}
```

**Libraries**: Opossum (Node.js), Resilience4j (Java), Polly (.NET)

## Retry with Exponential Backoff

**Problem**: Temporary failures (network glitch, rate limit)

**Wrong approach**: Immediate retry

```javascript
// ❌ BAD: Immediate retry
async function fetchData() {
  for (let i = 0; i < 3; i++) {
    try {
      return await api.get('/data');
    } catch (error) {
      // Retry immediately (hammers failing service)
    }
  }
  throw new Error('Failed after 3 retries');
}
```

**Right approach**: Exponential backoff with jitter

```javascript
// ✅ GOOD: Exponential backoff with jitter
async function fetchDataWithRetry(maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await api.get('/data');
    } catch (error) {
      if (attempt === maxRetries - 1) {
        throw error; // Final attempt failed
      }

      // Exponential backoff: 1s, 2s, 4s
      const baseDelay = 1000 * Math.pow(2, attempt);

      // Jitter: randomize delay ±25%
      const jitter = baseDelay * 0.5 * Math.random();
      const delay = baseDelay + jitter;

      console.log(`Retry ${attempt + 1} after ${delay}ms`);
      await sleep(delay);
    }
  }
}
```

**Why jitter?** Prevents thundering herd (all clients retrying at same time).

```
Without jitter:
Client 1 retries at: 1s, 2s, 4s
Client 2 retries at: 1s, 2s, 4s
Client 3 retries at: 1s, 2s, 4s
→ All hit server simultaneously (overwhelming spike)

With jitter:
Client 1 retries at: 0.8s, 1.9s, 4.2s
Client 2 retries at: 1.1s, 2.3s, 3.7s
Client 3 retries at: 0.9s, 2.1s, 4.5s
→ Retries spread out (smooth load)
```

### Idempotency for Safe Retries

**Idempotent**: Same request can be repeated safely without side effects.

**Idempotent operations** (safe to retry):
- GET, HEAD, OPTIONS (read-only)
- PUT, DELETE (same result if repeated)

**Non-idempotent** (unsafe to retry):
- POST (creates new resource each time)

**Solution**: Idempotency keys

```javascript
// Client sends idempotency key
await api.post('/payments', {
  amount: 9900,
  userId: 'user_123'
}, {
  headers: {
    'Idempotency-Key': 'pay_abc123xyz'  // Unique per payment
  }
});

// Server stores idempotency key
app.post('/payments', async (req, res) => {
  const idempotencyKey = req.headers['idempotency-key'];

  // Check if already processed
  const existing = await db.payments.findByIdempotencyKey(idempotencyKey);
  if (existing) {
    return res.json(existing);  // Return cached result
  }

  // Process payment
  const payment = await processPayment(req.body);

  // Store with idempotency key
  await db.payments.create({ ...payment, idempotencyKey });

  res.json(payment);
});
```

## Timeouts

**Problem**: Hanging requests consume resources.

**Solution**: Set aggressive timeouts.

```javascript
// ✅ Set timeout for all external calls
const response = await fetch('https://api.example.com/data', {
  signal: AbortSignal.timeout(5000)  // 5 second timeout
});

// Or with axios
const response = await axios.get('https://api.example.com/data', {
  timeout: 5000
});
```

**Timeout guidelines**:
- **User-facing APIs**: 1-5 seconds
- **Background jobs**: 30-60 seconds
- **Batch processing**: Minutes (with progress tracking)

**Cascading timeouts**: Set shorter timeouts for downstream calls

```
User request timeout: 10s
  └─> Service A timeout: 8s
       └─> Service B timeout: 6s
            └─> Database timeout: 4s
```

## Graceful Degradation

**Principle**: Partial functionality > complete failure.

### Strategies

**1. Fallback to cache**:
```javascript
async function getProduct(id) {
  try {
    return await database.products.findById(id);
  } catch (error) {
    // Fallback to cache
    const cached = await cache.get(`product:${id}`);
    if (cached) {
      return cached;
    }
    throw error;  // No cache, propagate error
  }
}
```

**2. Return stale data**:
```javascript
async function getRecommendations(userId) {
  try {
    return await mlService.getRecommendations(userId);
  } catch (error) {
    // Return stale recommendations
    return await cache.get(`recommendations:${userId}`);
  }
}
```

**3. Feature degradation**:
```javascript
async function checkout(cart) {
  const order = await createOrder(cart);

  try {
    // Try to send confirmation email
    await emailService.send(order.userEmail, order);
  } catch (error) {
    // Email failed, but order succeeded
    // Queue email for later
    await emailQueue.enqueue(order);
    console.warn('Email queued for later delivery');
  }

  return order;  // Return success even if email failed
}
```

**4. Disable non-critical features**:
```javascript
// Feature flags for graceful degradation
if (featureFlag.isEnabled('recommendations') && mlService.isHealthy()) {
  recommendations = await mlService.getRecommendations(userId);
} else {
  recommendations = [];  // Empty array if service down
}
```

## Bulkhead Pattern

**Principle**: Isolate resources to prevent cascading failures.

**Analogy**: Ship bulkheads prevent one compartment flood from sinking entire ship.

### Thread Pool Isolation

```javascript
// Separate thread pools for different dependencies
const paymentPool = new ThreadPool({ size: 10 });
const emailPool = new ThreadPool({ size: 5 });
const dbPool = new ThreadPool({ size: 20 });

// If email service hangs, only email pool exhausted
// Payment and DB operations still work
```

### Connection Pool Sizing

```
Database connection pool: 20
Payment API pool: 10
Email service pool: 5

Total connections: 35

If email service fails and all 5 connections hang:
- Payment still has 10 connections available
- Database still has 20 connections available
- Only email features degraded
```

## Capacity Planning

**Formula**: Capacity = (desired throughput) × (safety margin)

**Example**:
```
Peak traffic: 1000 requests/second
Safety margin: 2x
Required capacity: 2000 requests/second

Current capacity: 1500 requests/second
→ Need to scale up
```

**Load testing**: Validate capacity before traffic increases.

## Toil Reduction

**Toil**: Manual, repetitive operational work.

**Examples**:
- Manually restarting services
- Manually running database migrations
- Manually investigating same alert repeatedly

**Solution**: Automate toil

```
Manual toil: Restart API service when memory exceeds 90%
  ↓
Automated: Configure auto-restart on OOM errors
  ↓
Better: Fix memory leak (eliminate root cause)
```

**50% rule**: SREs should spend ≤50% time on toil, ≥50% on engineering projects.

## Quick Reference

| Pattern | Use When | Prevents |
|---------|----------|----------|
| **Circuit breaker** | Calling failing service | Cascading failures |
| **Retry with backoff** | Temporary failures | Overwhelming service |
| **Timeout** | Calls might hang | Resource exhaustion |
| **Graceful degradation** | Dependency unavailable | Complete service failure |
| **Bulkhead** | Isolating failures | Cross-service contamination |

| SLO | Error Budget | Downtime/Month |
|-----|--------------|----------------|
| **99%** | 1% | 7.2 hours |
| **99.9%** ("three nines") | 0.1% | 43.2 minutes |
| **99.95%** | 0.05% | 21.6 minutes |
| **99.99%** ("four nines") | 0.01% | 4.32 minutes |

## Common Mistakes

### ❌ No SLOs

**Wrong**: "We want 100% uptime"
**Right**: "99.9% availability SLO (43 min downtime/month)"

**Why**: 100% impossible, SLOs define realistic targets.

### ❌ Unlimited Retries

**Wrong**: Retry forever until success
**Right**: Retry 3 times with exponential backoff, then fail

**Why**: Unlimited retries overwhelm failing service.

### ❌ No Circuit Breaker

**Wrong**: Keep calling failing service
**Right**: Circuit breaker stops calls after failures

**Why**: Prevents cascading failures.

### ❌ Synchronous Everything

**Wrong**: Wait for email to send before returning success
**Right**: Queue email, return success immediately

**Why**: Non-critical operations shouldn't block critical path.

### ❌ Ignoring Error Budget

**Wrong**: Deploy features even with error budget exhausted
**Right**: Freeze deploys when budget exhausted, focus on reliability

**Why**: Error budget is policy for balancing velocity and stability.

## Real-World Impact

**Before Reliability Engineering**:
- No SLOs defined
- Cascading failures (payment down → entire site down)
- No retries or circuit breakers
- 95% uptime (36 hours downtime/month)

**After Reliability Engineering**:
- SLO: 99.9% availability
- Error budget: 43 min/month
- Circuit breakers for all external dependencies
- Retry with exponential backoff
- Graceful degradation (payment down → queue for later)
- 99.95% uptime achieved (21 min downtime/month)

## Summary

**Reliability engineering builds resilient systems through SRE principles:**

1. **Define SLOs** (reliability targets) and error budgets
2. **Circuit breakers** (stop calling failing services)
3. **Retry with exponential backoff** (handle temporary failures)
4. **Timeouts** (prevent resource exhaustion)
5. **Graceful degradation** (partial functionality > complete failure)
6. **Bulkhead isolation** (prevent cascading failures)
7. **Use error budget** (balance feature velocity with stability)

**Ordis Principle**: Reliability is structural integrity - systematic defenses against cascading failures, resilience through engineering discipline.
