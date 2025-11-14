---
name: contract-testing
description: Use when testing distributed system boundaries without E2E tests - provides consumer-driven contracts, schema validation, backward compatibility verification, API versioning, and reducing E2E test burden
---

# Contract Testing

## Overview

Contract testing validates that services communicate correctly without requiring E2E tests. Consumers define expectations (contracts), providers verify they meet them. This catches breaking changes before deployment while avoiding the complexity and flakiness of E2E tests.

**Core Principle**: Consumer defines contract (what it expects from provider), provider verifies contract (that it meets expectations). Contracts are verified on both sides independently. Faster and more reliable than E2E tests.

**Ordis Identity**: Contract testing is defensive verification at system boundaries - formal agreements between services that prevent integration failures without complex E2E orchestration.

## When to Use

**Use this skill when**:
- Testing microservice integration without E2E tests
- API changes might break downstream consumers
- Need to verify backward compatibility
- Reducing E2E test burden for distributed systems
- Schema validation between services
- API versioning and deprecation

**Don't use for**:
- Monolithic applications (use integration tests)
- Testing business logic (use unit tests)
- UI testing (use E2E tests for critical paths)

## Consumer-Driven Contracts

### The Problem

**Traditional approach**: E2E tests

```
Test: Create order end-to-end
  ├─ Start all services (Order, Payment, Inventory, Shipping)
  ├─ Make HTTP request to Order Service
  ├─ Order Service calls Payment Service
  ├─ Payment Service calls Bank API
  ├─ Inventory Service checks stock
  ├─ Shipping Service calculates shipping
  └─ Verify order created

Problems:
- Slow (start all services)
- Flaky (any service fails → test fails)
- Complex (maintain test infrastructure)
- Expensive (test all combinations)
```

**Contract testing approach**:

```
Consumer (Order Service) defines contract:
  "When I POST /payments, I expect 200 with { id, status }"

Provider (Payment Service) verifies contract:
  "I will return 200 with { id, status } for POST /payments"

Benefits:
- Fast (no service orchestration)
- Reliable (isolated verification)
- Simple (test one contract at a time)
- Scales (each service tested independently)
```

### Consumer Test (Pact Example)

**Consumer**: Order Service expects Payment Service to process payments.

```javascript
// order-service/pacts/payment.spec.js
const { Pact } = require('@pact-foundation/pact');
const { createOrder } = require('../order-service');

describe('Payment Service Contract', () => {
  const provider = new Pact({
    consumer: 'OrderService',
    provider: 'PaymentService',
  });

  beforeAll(() => provider.setup());
  afterAll(() => provider.finalize());

  test('processes payment', async () => {
    // Define expected interaction
    await provider.addInteraction({
      state: 'user has valid payment method',
      uponReceiving: 'a payment request',
      withRequest: {
        method: 'POST',
        path: '/payments',
        headers: { 'Content-Type': 'application/json' },
        body: {
          amount: 9900,
          currency: 'USD',
          userId: 'user_123'
        }
      },
      willRespondWith: {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: {
          id: Matchers.uuid(),
          status: 'succeeded',
          amount: 9900
        }
      }
    });

    // Execute consumer code against mock provider
    const result = await createOrder({
      userId: 'user_123',
      amount: 9900
    });

    expect(result.payment.status).toBe('succeeded');
  });

  // Contract published to Pact Broker
  afterAll(() => {
    return provider.verify().then(() => {
      return provider.publish({ consumerVersion: '1.0.0' });
    });
  });
});
```

**Result**: Contract file generated:

```json
{
  "consumer": { "name": "OrderService" },
  "provider": { "name": "PaymentService" },
  "interactions": [
    {
      "description": "a payment request",
      "request": {
        "method": "POST",
        "path": "/payments",
        "body": { "amount": 9900, "currency": "USD", "userId": "user_123" }
      },
      "response": {
        "status": 200,
        "body": { "id": "...", "status": "succeeded", "amount": 9900 }
      }
    }
  ]
}
```

### Provider Verification

**Provider**: Payment Service verifies it meets all consumer contracts.

```javascript
// payment-service/pacts/verify.spec.js
const { Verifier } = require('@pact-foundation/pact');
const { server } = require('../payment-service');

describe('Payment Service Provider Verification', () => {
  let serverInstance;

  beforeAll(() => {
    serverInstance = server.listen(8080);
  });

  afterAll(() => {
    serverInstance.close();
  });

  test('validates all consumer contracts', () => {
    return new Verifier({
      provider: 'PaymentService',
      providerBaseUrl: 'http://localhost:8080',

      // Fetch contracts from Pact Broker
      pactBrokerUrl: 'https://pact-broker.example.com',

      // Provider states (test data setup)
      stateHandlers: {
        'user has valid payment method': async () => {
          await database.users.create({
            id: 'user_123',
            paymentMethod: 'card_valid'
          });
        }
      },

      // Publish verification results
      publishVerificationResult: true,
      providerVersion: '2.5.0'
    }).verifyProvider();
  });
});
```

**Verification process**:
1. Fetch all contracts where provider = "PaymentService"
2. For each contract:
   - Set up provider state (e.g., "user has valid payment method")
   - Replay consumer request against real provider
   - Verify provider response matches contract
3. Publish verification results to Pact Broker

## Contract Testing Flow

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Consumer writes contract test                             │
│    - Defines expected interactions                           │
│    - Runs against mock provider                              │
│    - Publishes contract to Pact Broker                       │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Provider verifies contracts                                │
│    - Fetches contracts from Pact Broker                      │
│    - Runs real provider against contract requests            │
│    - Publishes verification results                          │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. Deployment gated on verification                           │
│    - Provider can deploy if all contracts verified           │
│    - Consumer can deploy if provider has verified contract   │
└──────────────────────────────────────────────────────────────┘
```

## Detecting Breaking Changes

**Scenario**: Provider changes response format

```javascript
// Provider v1: Returns { id, status, amount }
{
  "id": "pay_123",
  "status": "succeeded",
  "amount": 9900
}

// Provider v2: Adds field, changes status values
{
  "id": "pay_123",
  "status": "complete",  // ❌ BREAKING: consumer expects "succeeded"
  "amount": 9900,
  "currency": "USD"  // ✓ Non-breaking: new field
}
```

**Contract verification fails**:
```
❌ Contract verification failed

Expected: { "status": "succeeded" }
Actual:   { "status": "complete" }

Provider change breaks consumer contract.
Deploy blocked until contract updated.
```

## Schema Validation

**Alternative to contract testing**: Validate against OpenAPI/GraphQL schemas.

### OpenAPI Schema Validation

```javascript
// Validate provider matches OpenAPI spec
const Validator = require('express-validator-swagger');

test('Payment endpoint matches OpenAPI schema', async () => {
  const validator = new Validator('openapi.yaml');

  const response = await request(app)
    .post('/payments')
    .send({ amount: 9900, currency: 'USD', userId: 'user_123' });

  // Validate response against schema
  const result = validator.validate(response, '/payments', 'post');
  expect(result.errors).toHaveLength(0);
});
```

**OpenAPI schema**:
```yaml
paths:
  /payments:
    post:
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required: [amount, currency, userId]
              properties:
                amount: { type: integer }
                currency: { type: string }
                userId: { type: string }
      responses:
        '200':
          content:
            application/json:
              schema:
                type: object
                required: [id, status, amount]
                properties:
                  id: { type: string }
                  status: { type: string, enum: [succeeded, failed] }
                  amount: { type: integer }
```

### GraphQL Schema Validation

```javascript
// GraphQL schema is contract
const { buildSchema } = require('graphql');

const schema = buildSchema(`
  type Payment {
    id: ID!
    status: PaymentStatus!
    amount: Int!
  }

  enum PaymentStatus {
    SUCCEEDED
    FAILED
    PENDING
  }

  type Mutation {
    createPayment(amount: Int!, userId: ID!): Payment!
  }
`);

// GraphQL automatically validates requests/responses against schema
// Type errors caught at runtime or with tools like GraphQL Code Generator
```

## Backward Compatibility

### Versioning Strategies

**1. URL versioning**:
```
GET /v1/payments/123  ← Old consumers
GET /v2/payments/123  ← New consumers
```

**2. Header versioning**:
```
GET /payments/123
Accept: application/vnd.api+json; version=1
```

**3. Non-breaking changes only**:
```
✅ Add new fields (ignored by old consumers)
✅ Add new endpoints
✅ Make required fields optional
✅ Expand enum values (if consumers don't validate)

❌ Remove fields
❌ Rename fields
❌ Change field types
❌ Add required fields
```

### Deprecation Strategy

```
1. Announce deprecation
   - Update docs
   - Notify consumers
   - Set deprecation date (e.g., 90 days)

2. Add deprecation warnings
   - Return warning header: "X-API-Deprecated: true"
   - Log which consumers still use deprecated endpoint

3. Monitor usage
   - Dashboard showing deprecated endpoint usage by consumer
   - Alert when usage drops to zero

4. Remove after grace period
   - Verify no consumers remain
   - Delete deprecated code
```

## CI/CD Integration

```yaml
# Consumer CI pipeline
name: Consumer Tests
on: [pull_request]

jobs:
  contract-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run contract tests
        run: npm run test:contract

      - name: Publish contracts to Pact Broker
        run: |
          npm run pact:publish -- \
            --consumer-app-version=${{ github.sha }} \
            --branch=${{ github.ref }}

# Provider CI pipeline
name: Provider Tests
on: [pull_request]

jobs:
  verify-contracts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Verify consumer contracts
        run: npm run pact:verify

      - name: Publish verification results
        run: |
          npm run pact:publish-verification -- \
            --provider-app-version=${{ github.sha }}

      - name: Can I deploy?
        run: npx pact-broker can-i-deploy \
          --pacticipant PaymentService \
          --version ${{ github.sha }} \
          --to production
```

## Contract Testing vs E2E Testing

| Aspect | Contract Testing | E2E Testing |
|--------|-----------------|-------------|
| **Speed** | Fast (isolated services) | Slow (start all services) |
| **Reliability** | Reliable (no flakiness) | Flaky (any service fails) |
| **Scope** | Service boundaries | Complete workflows |
| **Maintenance** | Low (one contract per interaction) | High (complex test infrastructure) |
| **Coverage** | API contracts | Business logic + integration |
| **When to use** | Service integration | Critical user journeys |

**Recommendation**: Use contract testing for service boundaries, minimal E2E for critical paths only.

## Quick Reference

| Tool | Language | Use Case |
|------|----------|----------|
| **Pact** | Multi-language | Consumer-driven contracts |
| **Spring Cloud Contract** | Java/Spring | JVM microservices |
| **OpenAPI/Swagger** | Any | Schema validation (provider-driven) |
| **GraphQL** | Any | GraphQL schema enforcement |

| Change Type | Breaking? | Safe? |
|-------------|-----------|-------|
| **Add field** | No | ✅ Safe |
| **Remove field** | Yes | ❌ Breaking |
| **Rename field** | Yes | ❌ Breaking |
| **Change type** | Yes | ❌ Breaking |
| **Add endpoint** | No | ✅ Safe |
| **Remove endpoint** | Yes | ❌ Breaking (deprecate first) |

## Common Mistakes

### ❌ Testing Implementation, Not Contract

**Wrong**: Contract tests provider's internal logic
**Right**: Contract tests provider's interface only

**Why**: Contracts should verify API surface, not implementation.

### ❌ Too Many E2E Tests

**Wrong**: Test all service integration with E2E tests
**Right**: Use contract tests for integration, E2E for critical paths only

**Why**: Contract tests are faster, more reliable.

### ❌ Provider-Driven Contracts

**Wrong**: Provider defines contract, consumers must adapt
**Right**: Consumers define what they need, provider verifies

**Why**: Consumer-driven ensures provider meets actual needs.

### ❌ Not Versioning APIs

**Wrong**: Make breaking changes, assume consumers will update
**Right**: Version APIs, maintain backward compatibility

**Why**: Breaking changes without versioning break production.

## Real-World Impact

**Before Contract Testing**:
- 50 E2E tests for microservice integration
- 45-minute test suite
- 15% flake rate
- Breaking changes discovered in production

**After Contract Testing**:
- 5 E2E tests (critical paths)
- 120 contract tests
- 8-minute test suite
- <1% flake rate
- Breaking changes caught before deployment

## Summary

**Contract testing verifies service boundaries without E2E complexity:**

1. **Consumer-driven contracts** (consumer defines needs, provider verifies)
2. **Independent verification** (each service tests separately)
3. **Fast and reliable** (no service orchestration, no flakiness)
4. **Breaking change detection** (contracts fail if API changes break consumers)
5. **CI/CD integration** (block deployment if contracts fail)
6. **Replace E2E where possible** (use contracts for integration, E2E for critical paths only)

**Ordis Principle**: Contract testing is defensive verification at boundaries - formal agreements that prevent integration failures without complex orchestration.
