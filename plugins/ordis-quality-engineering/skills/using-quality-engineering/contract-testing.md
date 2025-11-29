---
name: contract-testing
description: Use when implementing Pact contracts, choosing consumer-driven vs provider-driven approaches, handling breaking API changes, setting up contract brokers, or preventing service integration issues - provides tool selection, anti-patterns, and workflow patterns
---

# Contract Testing

## Overview

**Core principle:** Test the contract, not the implementation. Verify integration points independently.

**Rule:** Contract tests catch breaking changes before deployment, not in production.

## Tool Selection Decision Tree

| Your Stack | Team Structure | Use | Why |
|-----------|----------------|-----|-----|
| Polyglot microservices | Multiple teams | **Pact** | Language-agnostic, mature broker |
| Java Spring ecosystem | Coordinated teams | **Spring Cloud Contract** | Spring integration, code-first |
| GraphQL APIs | Known consumers | **Pact + GraphQL** | Query validation |
| OpenAPI/REST | Public/many consumers | **OpenAPI Spec Testing** | Schema-first, documentation |

**First choice:** Pact (most mature ecosystem, widest language support)

**Why contract testing:** Catches API breaking changes in CI, not production. Teams test independently without running dependencies.

## Contract Type Decision Framework

| Scenario | Approach | Tools |
|----------|----------|-------|
| **Internal microservices, known consumers** | Consumer-Driven (CDC) | Pact, Spring Cloud Contract |
| **Public API, many unknown consumers** | Provider-Driven (Schema-First) | OpenAPI validation, Spectral |
| **Both internal and external consumers** | Bi-Directional | Pact + OpenAPI |
| **Event-driven/async messaging** | Message Pact | Pact (message provider/consumer) |

**Default:** Consumer-driven for internal services, schema-first for public APIs

## Anti-Patterns Catalog

### ❌ Over-Specification
**Symptom:** Contract tests verify exact response format, including fields consumer doesn't use

**Why bad:** Brittle tests, provider can't evolve API, false positives

**Fix:** Only specify what consumer actually uses

```javascript
// ❌ Bad - over-specified
.willRespondWith({
  status: 200,
  body: {
    id: 123,
    name: 'John',
    email: 'john@example.com',
    created_at: '2023-01-01',
    updated_at: '2023-01-02',
    phone: '555-1234',
    address: {...}  // Consumer doesn't use these
  }
})

// ✅ Good - specify only what's used
.willRespondWith({
  status: 200,
  body: {
    id: Matchers.integer(123),
    name: Matchers.string('John')
  }
})
```

---

### ❌ Testing Implementation Details
**Symptom:** Contract tests verify database queries, internal logic, or response timing

**Why bad:** Couples tests to implementation, not contract

**Fix:** Test only request/response contract, not how provider implements it

```javascript
// ❌ Bad - testing implementation
expect(provider.database.queryCalled).toBe(true)

// ✅ Good - testing contract only
expect(response.status).toBe(200)
expect(response.body.name).toBe('John')
```

---

### ❌ Brittle Provider States
**Symptom:** Provider states hardcode IDs, dates, or specific data that changes

**Why bad:** Tests fail randomly, high maintenance

**Fix:** Use matchers, generate data in state setup

```javascript
// ❌ Bad - hardcoded state
.given('user 123 exists')
.uponReceiving('request for user 123')
.withRequest({ path: '/users/123' })

// ✅ Good - flexible state
.given('a user exists')
.uponReceiving('request for user')
.withRequest({ path: Matchers.regex('/users/\\d+', '/users/123') })
.willRespondWith({
  body: {
    id: Matchers.integer(123),
    name: Matchers.string('John')
  }
})
```

---

### ❌ No Contract Versioning
**Symptom:** Breaking changes deployed without consumer coordination

**Why bad:** Runtime failures, production incidents

**Fix:** Use can-i-deploy, tag contracts by environment

```bash
# ✅ Good - check before deploying
pact-broker can-i-deploy \
  --pacticipant UserService \
  --version 2.0.0 \
  --to production
```

---

### ❌ Missing Can-I-Deploy
**Symptom:** Deploying without checking if all consumers compatible

**Why bad:** Deploy provider changes that break consumers

**Fix:** Run can-i-deploy in CI before deployment

## Pact Broker Workflow

**Core workflow:**

1. **Consumer:** Write contract test → Generate pact file
2. **Consumer CI:** Publish pact to broker with version tag
3. **Provider CI:** Fetch contracts → Verify → Publish results
4. **Provider CD:** Run can-i-deploy → Deploy if compatible

### Publishing Contracts

```bash
# Consumer publishes pact with version and branch
pact-broker publish pacts/ \
  --consumer-app-version ${GIT_SHA} \
  --branch ${GIT_BRANCH} \
  --tag ${ENV}
```

### Verifying Contracts

```javascript
// Provider verifies against broker
const { Verifier } = require('@pact-foundation/pact')

new Verifier({
  providerBaseUrl: 'http://localhost:8080',
  pactBrokerUrl: process.env.PACT_BROKER_URL,
  provider: 'UserService',
  publishVerificationResult: true,
  providerVersion: process.env.GIT_SHA,
  consumerVersionSelectors: [
    { mainBranch: true },  // Latest from main
    { deployed: 'production' },  // Currently in production
    { deployed: 'staging' }  // Currently in staging
  ]
}).verifyProvider()
```

### Can-I-Deploy Check

```yaml
# CI/CD pipeline (GitHub Actions example)
- name: Check if can deploy
  run: |
    pact-broker can-i-deploy \
      --pacticipant UserService \
      --version ${{ github.sha }} \
      --to-environment production
```

**Rule:** Never deploy without can-i-deploy passing

## Breaking Change Taxonomy

| Change Type | Breaking? | Migration Strategy |
|-------------|-----------|-------------------|
| Add optional field | No | Deploy provider first |
| Add required field | Yes | Use expand/contract pattern |
| Remove field | Yes | Deprecate → verify no consumers use → remove |
| Change field type | Yes | Add new field → migrate consumers → remove old |
| Rename field | Yes | Add new → deprecate old → remove old |
| Change status code | Yes | Version API or expand responses |

### Expand/Contract Pattern

**For adding required field:**

**Expand (Week 1-2):**
```javascript
// Provider adds NEW field (optional), keeps OLD field
{
  user_name: "John",  // Old field (deprecated)
  name: "John"        // New field
}
```

**Migrate (Week 3-4):**
- Consumers update to use new field
- Update contracts
- Verify all consumers migrated

**Contract (Week 5):**
```javascript
// Provider removes old field
{
  name: "John"  // Only new field remains
}
```

## Provider State Patterns

**Purpose:** Set up test data before verification

**Pattern:** Use state handlers to create/clean up data

```javascript
// Provider state setup
const { Verifier } = require('@pact-foundation/pact')

new Verifier({
  stateHandlers: {
    'a user exists': async () => {
      // Setup: Create test user
      await db.users.create({
        id: 123,
        name: 'John Doe'
      })
    },
    'no users exist': async () => {
      // Setup: Clear users
      await db.users.deleteAll()
    }
  },
  afterEach: async () => {
    // Cleanup after each verification
    await db.users.deleteAll()
  }
}).verifyProvider()
```

**Best practices:**
- States should be independent
- Clean up after each verification
- Use transactions for database tests
- Don't hardcode IDs (use matchers)

## Async/Event-Driven Messaging Contracts

**For Kafka, RabbitMQ, SNS/SQS:** Use Message Pact (different API than HTTP Pact)

### Consumer Message Contract

```javascript
const { MessageConsumerPact, MatchersV3 } = require('@pact-foundation/pact')

describe('User Event Consumer', () => {
  const messagePact = new MessageConsumerPact({
    consumer: 'NotificationService',
    provider: 'UserService'
  })

  it('processes user created events', () => {
    return messagePact
      .expectsToReceive('user created event')
      .withContent({
        userId: MatchersV3.integer(123),
        email: MatchersV3.string('user@example.com'),
        eventType: 'USER_CREATED'
      })
      .withMetadata({
        'content-type': 'application/json'
      })
      .verify((message) => {
        processUserCreatedEvent(message.contents)
      })
  })
})
```

### Provider Message Verification

```javascript
// Provider verifies it can produce matching messages
const { MessageProviderPact } = require('@pact-foundation/pact')

describe('User Event Producer', () => {
  it('publishes user created events matching contracts', () => {
    return new MessageProviderPact({
      messageProviders: {
        'user created event': () => ({
          contents: {
            userId: 123,
            email: 'test@example.com',
            eventType: 'USER_CREATED'
          },
          metadata: {
            'content-type': 'application/json'
          }
        })
      }
    }).verify()
  })
})
```

### Key Differences from HTTP Contracts

- **No request/response:** Only message payload
- **Metadata:** Headers, content-type, message keys
- **Ordering:** Don't test message ordering in contracts (infrastructure concern)
- **Delivery:** Don't test delivery guarantees (wrong layer)

**Workflow:** Same as HTTP (publish pact → verify → can-i-deploy)

## CI/CD Integration Quick Reference

### GitHub Actions

```yaml
# Consumer publishes contracts
- name: Run Pact tests
  run: npm test

- name: Publish pacts
  run: |
    npm run pact:publish
  env:
    PACT_BROKER_URL: ${{ secrets.PACT_BROKER_URL }}
    PACT_BROKER_TOKEN: ${{ secrets.PACT_BROKER_TOKEN }}

# Provider verifies and checks deployment
- name: Verify contracts
  run: npm run pact:verify

- name: Can I deploy?
  run: |
    pact-broker can-i-deploy \
      --pacticipant UserService \
      --version ${{ github.sha }} \
      --to-environment production
```

### GitLab CI

```yaml
pact_test:
  script:
    - npm test
    - npm run pact:publish

pact_verify:
  script:
    - npm run pact:verify
    - pact-broker can-i-deploy --pacticipant UserService --version $CI_COMMIT_SHA --to-environment production
```

## Your First Contract Test

**Goal:** Prevent breaking changes between two services in one week

**Day 1-2: Consumer Side**

```javascript
// Install Pact
npm install --save-dev @pact-foundation/pact

// Consumer contract test (order-service)
const { PactV3, MatchersV3 } = require('@pact-foundation/pact')
const { getUserById } = require('./userClient')

describe('User API', () => {
  const provider = new PactV3({
    consumer: 'OrderService',
    provider: 'UserService'
  })

  it('gets user by id', () => {
    provider
      .given('a user exists')
      .uponReceiving('a request for user')
      .withRequest({
        method: 'GET',
        path: '/users/123'
      })
      .willRespondWith({
        status: 200,
        body: {
          id: MatchersV3.integer(123),
          name: MatchersV3.string('John')
        }
      })

    return provider.executeTest(async (mockServer) => {
      const user = await getUserById(mockServer.url, 123)
      expect(user.name).toBe('John')
    })
  })
})
```

**Day 3-4: Set Up Pact Broker**

```bash
# Docker Compose
docker-compose up -d

# Or use hosted Pactflow (SaaS)
# https://pactflow.io
```

**Day 5-6: Provider Side**

```javascript
// Provider verification (user-service)
const { Verifier } = require('@pact-foundation/pact')
const app = require('./app')

describe('Pact Verification', () => {
  it('validates contracts from broker', () => {
    return new Verifier({
      provider: 'UserService',
      providerBaseUrl: 'http://localhost:8080',
      pactBrokerUrl: process.env.PACT_BROKER_URL,
      publishVerificationResult: true,
      providerVersion: '1.0.0',

      stateHandlers: {
        'a user exists': async () => {
          await db.users.create({ id: 123, name: 'John' })
        }
      }
    }).verifyProvider()
  })
})
```

**Day 7: Add to CI**

```yaml
# Add can-i-deploy before deployment
- pact-broker can-i-deploy --pacticipant UserService --version $VERSION --to production
```

## Common Mistakes

### ❌ Testing Business Logic in Contracts
**Fix:** Contract tests verify integration only. Test business logic separately.

---

### ❌ Not Using Matchers
**Fix:** Use `Matchers.string()`, `Matchers.integer()` for flexible matching

---

### ❌ Skipping Can-I-Deploy
**Fix:** Always run can-i-deploy before deployment. Automate in CI.

---

### ❌ Hardcoding Test Data
**Fix:** Generate data in provider states, use matchers in contracts

## Quick Reference

**Tool Selection:**
- Polyglot/multiple teams: Pact
- Java Spring only: Spring Cloud Contract
- Public API: OpenAPI validation

**Contract Type:**
- Internal services: Consumer-driven (Pact)
- Public API: Provider-driven (OpenAPI)
- Both: Bi-directional

**Pact Broker Workflow:**
1. Consumer publishes pact
2. Provider verifies
3. Can-i-deploy checks compatibility
4. Deploy if compatible

**Breaking Changes:**
- Add optional field: Safe
- Add required field: Expand/contract pattern
- Remove/rename field: Deprecate → migrate → remove

**Provider States:**
- Set up test data
- Clean up after each test
- Use transactions for DB
- Don't hardcode IDs

**CI/CD:**
- Consumer: Test → publish pacts
- Provider: Verify → can-i-deploy → deploy

## Bottom Line

**Contract testing prevents API breaking changes by testing integration points independently. Use Pact for internal microservices, publish contracts to broker, run can-i-deploy before deployment.**

Test the contract (request/response), not the implementation. Use consumer-driven contracts for known consumers, schema-first for public APIs.
