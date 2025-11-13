# Git History Analysis - Architecture Decision Timeline

**Source:** Git log analysis + code archaeology
**Codebase:** /legacy-ecommerce-platform
**Analysis Period:** 2018-2024

---

## 2018: Monolith Architecture

**Commit:** `a3f7c21` (2018-03-15)
**Author:** Sarah Chen <sarah@company.com>
**Message:** "Initial commit - Rails monolith for MVP"

**Code Structure:**
```
app/
  controllers/
  models/
  views/
config/
db/
```

**Team Size:** 2 developers (Sarah Chen, Mike Rodriguez)
**Rationale (from Slack archives):**
> Sarah: "Starting with Rails monolith - we need to ship MVP fast"
> Mike: "Agreed, microservices would be overkill for 2 devs"

**Assessment:** ✅ **Appropriate decision** - Monolith is correct choice for 2-person MVP team

---

## 2019: Split to Microservices

**Commit:** `b8e4d32` (2019-08-22)
**Author:** Sarah Chen <sarah@company.com>
**Message:** "Refactor: Split into microservices architecture"

**Code Structure After:**
```
services/
  user-service/
  product-service/
  order-service/
  payment-service/
```

**Team Size:** Still 2 developers
**Rationale (from design doc):**
> "As we scale, microservices will allow us to:
> - Deploy services independently
> - Scale components based on load
> - Use different tech stacks per service
> - Enable parallel development"

**Reality Check:**
- No evidence of scaling problems (100 users, <10 req/sec)
- Team still 2 developers (cannot work in parallel)
- All services deployed together (monorepo, single deploy script)
- All services using same tech stack (Ruby)

**Assessment:** ⚠️ **Premature optimization** - Split to microservices before team/scale justified it

**Code Evidence:**
```ruby
# user-service/app/controllers/users_controller.rb (2019-08-22)
class UsersController < ApplicationController
  def index
    # Direct database query - same as monolith
    @users = User.all
  end
end
```

Nothing changed except directory structure.

---

## 2020: Shared Database Added

**Commit:** `c9a2f43` (2020-04-10)
**Author:** Mike Rodriguez <mike@company.com>
**Message:** "Fix: Share database across services for data consistency"

**Context (from code review):**
> Mike: "Services keep getting out of sync. Order service creates order,
> but product service doesn't see inventory update. Users seeing stale data."
>
> Sarah: "Yeah, the event bus is too complex for us to maintain. Let's just
> share the database for now."

**Change:**
```yaml
# config/database.yml (all services)
production:
  database: ecommerce_shared
  host: db.internal.company.com
```

**Assessment:** ⚠️ **Defeated microservices purpose** - Shared database creates tight coupling, eliminates deployment independence

**Evidence of Harm:**
- 2020-06-15: Database migration broke all 4 services simultaneously
- 2020-09-22: Order service schema change required changes in 3 other services
- 2021-01-30: Cannot deploy services independently due to schema dependencies

---

## 2021: REST Calls Between Services

**Commit:** `d4b8e51` (2021-02-18)
**Author:** Jake Williams <jake@company.com> (new hire)
**Message:** "Add inter-service communication via REST"

**Team Size:** 4 developers (Sarah, Mike, Jake, + 1 contractor)

**Context:**
Jake joined from previous company with microservices experience.

**Code Example:**
```ruby
# order-service/app/services/order_creator.rb
def create_order(user_id, product_id)
  # Call user-service via REST
  user = HTTP.get("http://user-service/api/users/#{user_id}").parse

  # Call product-service via REST
  product = HTTP.get("http://product-service/api/products/#{product_id}").parse

  # But also direct DB access for inventory check
  inventory = DB[:inventory].where(product_id: product_id).first

  # Create order in shared database
  Order.create(user_id: user_id, product_id: product_id)
end
```

**Assessment:** ⚠️ **Created distributed monolith** - Mixed REST calls + shared database = worst of both worlds

**Problems Introduced:**
- Network latency (3 HTTP calls per order)
- Failure cascade (if user-service down, orders fail)
- Still coupled via database
- Circular dependencies (order → product → inventory → order)

**Evidence from Incidents:**
- 2021-04-12: User-service timeout caused 3-hour order service outage
- 2021-07-22: Circular dependency deadlock, required service restarts
- 2021-11-05: Order creation 10x slower than monolith (network overhead)

---

## 2022: Message Queue Added (But REST Kept)

**Commit:** `e7c3d62` (2022-05-30)
**Author:** Sarah Chen <sarah@company.com>
**Message:** "Add RabbitMQ for async event processing"

**Context (from planning meeting notes):**
> Sarah: "We need async processing for email notifications. Adding RabbitMQ."
> Jake: "Should we migrate all inter-service calls to events?"
> Sarah: "No time, let's just add it for emails. REST calls work fine."

**Result:**
- **Pattern 1:** Email service uses message queue (RabbitMQ)
- **Pattern 2:** All other services still use REST calls
- **Pattern 3:** All services still share database

**Code Example:**
```ruby
# order-service/app/services/order_creator.rb
def create_order(user_id, product_id)
  # Still using REST for user/product (unchanged from 2021)
  user = HTTP.get("http://user-service/api/users/#{user_id}").parse
  product = HTTP.get("http://product-service/api/products/#{product_id}").parse

  order = Order.create(user_id: user_id, product_id: product_id)

  # NEW: Message queue for email (async)
  MessageBus.publish('order.created', order.to_json)
end
```

**Assessment:** ⚠️ **Inconsistent patterns** - Now have 3 communication patterns for same logical operations

**Team Confusion:**
- 2022-08-10: New developer asks "When should I use REST vs message queue?"
- 2022-10-15: Documentation says "use events" but most code uses REST
- 2023-01-20: Failed feature (inventory updates) - developer used wrong pattern

---

## 2023: Caching Layer for Performance

**Commit:** `f9d4e73` (2023-03-15)
**Author:** Mike Rodriguez <mike@company.com>
**Message:** "Add Redis caching to improve response times"

**Context (from incident report):**
> Users complaining about slow product catalog (5-10 second load times).
> Added Redis cache in front of product-service.

**Implementation:**
```ruby
# product-service/app/controllers/products_controller.rb
def index
  cached = Redis.get('products:all')
  return JSON.parse(cached) if cached

  # Still makes REST call to user-service to check permissions
  user = HTTP.get("http://user-service/api/users/#{current_user_id}").parse

  products = Product.all  # Direct DB query
  Redis.set('products:all', products.to_json, ex: 300)
  products
end
```

**Assessment:** ⚠️ **Treating symptom, not cause** - Performance problems caused by distributed architecture + N+1 queries

**Root Causes NOT Addressed:**
- REST call overhead (product → user for every request)
- N+1 database queries (each product loads categories separately)
- Inefficient query patterns (no indexes on foreign keys)

**Cache Invalidation Problems:**
- 2023-05-22: Users seeing stale product data (cache not invalidated on updates)
- 2023-08-10: Cache stampede during sale (all caches expired simultaneously)
- 2023-11-15: Memory issues (Redis hitting RAM limit, evicting hot data)

---

## 2024: Current State

**Team Size:** 8 developers
**Microservices Count:** 14 services
**Communication Patterns:**
- Shared database (all services)
- REST calls (12 service pairs)
- Message queue (2 service pairs)
- Direct DB access (as fallback)
- Caching (4 services)

**Architecture Assessment:**
- ❌ Not truly microservices (shared database, tight coupling)
- ❌ More complex than monolith (distributed system overhead)
- ❌ Slower than monolith was (network + cache overhead)
- ❌ Harder to maintain (inconsistent patterns)
- ❌ No benefits realized (cannot deploy independently, cannot scale independently)

**Costs of Current Architecture vs. 2018 Monolith:**
| Metric | 2018 Monolith | 2024 "Microservices" | Change |
|--------|--------------|---------------------|--------|
| Response time (p95) | 150ms | 850ms | **5.7x slower** |
| Deployment time | 5 min | 45 min | **9x slower** |
| Bug resolution time | 2 hours | 8 hours | **4x slower** |
| Onboarding time | 2 weeks | 12 weeks | **6x slower** |
| Infrastructure cost | $500/mo | $3,200/mo | **6.4x more expensive** |

---

## Decision Pattern Summary

| Year | Decision | Justification Claimed | Reality | Outcome |
|------|----------|---------------------|---------|---------|
| 2018 | Monolith | Fast MVP delivery | True | ✅ Success |
| 2019 | Microservices | Future scalability | Premature | ⚠️ Unnecessary complexity |
| 2020 | Shared database | Data consistency | Defeated microservices | ❌ Tight coupling |
| 2021 | REST calls | Service communication | Created distributed monolith | ❌ Performance issues |
| 2022 | Message queue | Async processing | Inconsistent patterns | ⚠️ Developer confusion |
| 2023 | Caching | Performance improvement | Symptom treatment | ⚠️ More complexity |

**Pattern:** Each decision made sense in isolation, but compounded into architectural mess. Classic case of **incremental rational decisions → irrational outcome**.

---

## Key Insight

**The decisions weren't irrational when made.** They were locally optimal but globally harmful.

**Evidence:**
- 2019: "We might need to scale" → Split to microservices (reasonable forecast)
- 2020: "Services getting out of sync" → Share database (solves immediate problem)
- 2021: "Need service communication" → Add REST (standard pattern)
- 2022: "Need async" → Add queue (standard solution)
- 2023: "Slow" → Add cache (standard optimization)

**But:** No architectural governance. No consolidation. No "stop and redesign" moment.

**Result:** Distributed monolith with 6.4x higher costs and 5.7x worse performance than original.
