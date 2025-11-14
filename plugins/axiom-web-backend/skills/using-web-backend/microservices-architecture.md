
# Microservices Architecture

## Overview

**Microservices architecture specialist covering service boundaries, communication patterns, data consistency, and operational concerns.**

**Core principle**: Microservices decompose applications into independently deployable services organized around business capabilities - enabling team autonomy and technology diversity at the cost of operational complexity and distributed system challenges.

## When to Use This Skill

Use when encountering:

- **Service boundaries**: Defining service scope, applying domain-driven design
- **Monolith decomposition**: Strategies for splitting existing systems
- **Data consistency**: Sagas, event sourcing, eventual consistency patterns
- **Communication**: Sync (REST/gRPC) vs async (events/messages)
- **API gateways**: Routing, authentication, rate limiting
- **Service discovery**: Registry patterns, DNS, configuration
- **Resilience**: Circuit breakers, retries, timeouts, bulkheads
- **Observability**: Distributed tracing, logging aggregation, metrics
- **Deployment**: Containers, orchestration, blue-green deployments

**Do NOT use for**:
- Monolithic architectures (microservices aren't always better)
- Single-team projects < 5 services (overhead exceeds benefits)
- Simple CRUD applications (microservices add unnecessary complexity)

## When NOT to Use Microservices

**Stay monolithic if**:
- Team < 10 engineers
- Domain is not well understood yet
- Strong consistency required everywhere
- Network latency is critical
- You can't invest in observability/DevOps infrastructure

**Microservices require**: Mature DevOps, monitoring, distributed systems expertise, organizational support.

## Service Boundary Patterns (Domain-Driven Design)

### 1. Bounded Contexts

**Pattern: One microservice = One bounded context**

```
❌ Too fine-grained (anemic services):
- UserService (just CRUD)
- OrderService (just CRUD)
- PaymentService (just CRUD)

✅ Business capability alignment:
- CustomerManagementService (user profiles, preferences, history)
- OrderFulfillmentService (order lifecycle, inventory, shipping)
- PaymentProcessingService (payment, billing, invoicing, refunds)
```

**Identifying boundaries**:
1. **Ubiquitous language** - Different terms for same concept = different contexts
2. **Change patterns** - Services that change together should stay together
3. **Team ownership** - One team should own one service
4. **Data autonomy** - Each service owns its data, no shared databases

### 2. Strategic DDD Patterns

| Pattern | Use When | Example |
|---------|----------|---------|
| **Separate Ways** | Contexts are independent | Analytics service, main app service |
| **Partnership** | Teams must collaborate closely | Order + Inventory services |
| **Customer-Supplier** | Upstream/downstream relationship | Payment gateway (upstream) → Order service |
| **Conformist** | Accept upstream model as-is | Third-party API integration |
| **Anti-Corruption Layer** | Isolate from legacy/external systems | ACL between new microservices and legacy monolith |

### 3. Service Sizing Guidelines

**Too small (Nanoservices)**:
- Excessive network calls
- Distributed monolith
- Coordination overhead exceeds benefits

**Too large (Minimonoliths)**:
- Multiple teams modifying same service
- Mixed deployment frequencies
- Tight coupling re-emerges

**Right size indicators**:
- Single team can own it
- Deployable independently
- Changes don't ripple to other services
- Clear business capability
- 100-10,000 LOC (highly variable)

## Communication Patterns

### Synchronous Communication

**REST APIs**:

```python
# Order service calling Payment service
async def create_order(order: Order):
    # Synchronous REST call
    payment = await payment_service.charge(
        amount=order.total,
        customer_id=order.customer_id
    )

    if payment.status == "success":
        order.status = "confirmed"
        await db.save(order)
        return order
    else:
        raise PaymentFailedException()
```

**Pros**: Simple, request-response, easy to debug
**Cons**: Tight coupling, availability dependency, latency cascades

**gRPC**:

```python
# Proto definition
service OrderService {
    rpc CreateOrder (OrderRequest) returns (OrderResponse);
}

# Implementation
class OrderServicer(order_pb2_grpc.OrderServiceServicer):
    async def CreateOrder(self, request, context):
        # Type-safe, efficient binary protocol
        payment = await payment_stub.Charge(
            PaymentRequest(amount=request.total)
        )
        return OrderResponse(order_id=order.id)
```

**Pros**: Type-safe, efficient, streaming support
**Cons**: HTTP/2 required, less human-readable, proto dependencies

### Asynchronous Communication

**Event-Driven (Pub/Sub)**:

```python
# Order service publishes event
await event_bus.publish("order.created", {
    "order_id": order.id,
    "customer_id": customer.id,
    "total": order.total
})

# Inventory service subscribes
@event_bus.subscribe("order.created")
async def reserve_inventory(event):
    await inventory.reserve(event["order_id"])
    await event_bus.publish("inventory.reserved", {...})

# Notification service subscribes
@event_bus.subscribe("order.created")
async def send_confirmation(event):
    await email.send_order_confirmation(event)
```

**Pros**: Loose coupling, services independent, scalable
**Cons**: Eventual consistency, harder to trace, ordering challenges

**Message Queues (Point-to-Point)**:

```python
# Producer
await queue.send("payment-processing", {
    "order_id": order.id,
    "amount": order.total
})

# Consumer
@queue.consumer("payment-processing")
async def process_payment(message):
    result = await payment_gateway.charge(message["amount"])
    if result.success:
        await message.ack()
    else:
        await message.nack(requeue=True)
```

**Pros**: Guaranteed delivery, work distribution, retry handling
**Cons**: Queue becomes bottleneck, requires message broker

### Communication Pattern Decision Matrix

| Scenario | Pattern | Why |
|----------|---------|-----|
| User-facing request/response | Sync (REST/gRPC) | Low latency, immediate feedback |
| Background processing | Async (queue) | Don't block user, retry support |
| Cross-service notifications | Async (pub/sub) | Loose coupling, multiple consumers |
| Real-time updates | WebSocket/SSE | Bidirectional, streaming |
| Data replication | Event sourcing | Audit trail, rebuild state |
| High throughput | Async (messaging) | Buffer spikes, backpressure |

## Data Consistency Patterns

### 1. Saga Pattern (Distributed Transactions)

**Choreography (Event-Driven)**:

```python
# Order Service
async def create_order(order):
    order.status = "pending"
    await db.save(order)
    await events.publish("order.created", order)

# Payment Service
@events.subscribe("order.created")
async def handle_order(event):
    try:
        await charge_customer(event["total"])
        await events.publish("payment.completed", event)
    except PaymentError:
        await events.publish("payment.failed", event)

# Inventory Service
@events.subscribe("payment.completed")
async def reserve_items(event):
    try:
        await reserve(event["items"])
        await events.publish("inventory.reserved", event)
    except InventoryError:
        await events.publish("inventory.failed", event)

# Order Service (Compensation)
@events.subscribe("payment.failed")
async def cancel_order(event):
    order = await db.get(event["order_id"])
    order.status = "cancelled"
    await db.save(order)

@events.subscribe("inventory.failed")
async def refund_payment(event):
    await payment.refund(event["order_id"])
    await cancel_order(event)
```

**Orchestration (Coordinator)**:

```python
class OrderSaga:
    def __init__(self, order):
        self.order = order
        self.completed_steps = []

    async def execute(self):
        try:
            # Step 1: Reserve inventory
            await self.reserve_inventory()
            self.completed_steps.append("inventory")

            # Step 2: Process payment
            await self.process_payment()
            self.completed_steps.append("payment")

            # Step 3: Confirm order
            await self.confirm_order()

        except Exception as e:
            # Compensate in reverse order
            await self.compensate()
            raise

    async def compensate(self):
        for step in reversed(self.completed_steps):
            if step == "inventory":
                await inventory_service.release(self.order.id)
            elif step == "payment":
                await payment_service.refund(self.order.id)
```

**Choreography vs Orchestration**:

| Aspect | Choreography | Orchestration |
|--------|--------------|---------------|
| Coordination | Decentralized (events) | Centralized (orchestrator) |
| Coupling | Loose | Tight to orchestrator |
| Complexity | Distributed across services | Concentrated in orchestrator |
| Tracing | Harder (follow events) | Easier (single coordinator) |
| Failure handling | Implicit (event handlers) | Explicit (orchestrator logic) |
| Best for | Simple workflows | Complex workflows |

### 2. Event Sourcing

**Pattern: Store events, not state**

```python
# Traditional approach (storing state)
class Order:
    id: int
    status: str  # "pending" → "confirmed" → "shipped"
    total: float

# Event sourcing (storing events)
class OrderCreated(Event):
    order_id: int
    total: float

class OrderConfirmed(Event):
    order_id: int

class OrderShipped(Event):
    order_id: int

# Rebuild state from events
def rebuild_order(order_id):
    events = event_store.get_events(order_id)
    order = Order()
    for event in events:
        order.apply(event)  # Apply each event to rebuild state
    return order
```

**Pros**: Complete audit trail, time travel, event replay
**Cons**: Complexity, eventual consistency, schema evolution challenges

### 3. CQRS (Command Query Responsibility Segregation)

**Separate read and write models**:

```python
# Write model (commands)
class CreateOrder:
    def execute(self, data):
        order = Order(**data)
        await db.save(order)
        await event_bus.publish("order.created", order)

# Read model (projections)
class OrderReadModel:
    # Denormalized for fast reads
    def __init__(self):
        self.cache = {}

    @event_bus.subscribe("order.created")
    async def on_order_created(self, event):
        self.cache[event["order_id"]] = {
            "id": event["order_id"],
            "customer_name": await get_customer_name(event["customer_id"]),
            "status": "pending",
            "total": event["total"]
        }

    def get_order(self, order_id):
        return self.cache.get(order_id)  # Fast read, no joins
```

**Use when**: Read/write patterns differ significantly (e.g., analytics dashboards)

## Resilience Patterns

### 1. Circuit Breaker

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
async def call_payment_service(amount):
    response = await http.post("http://payment-service/charge", json={"amount": amount})
    if response.status >= 500:
        raise PaymentServiceError()
    return response.json()

# Circuit states:
# CLOSED → normal operation
# OPEN → fails fast after threshold
# HALF_OPEN → test if service recovered
```

### 2. Retry with Exponential Backoff

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def call_with_retry(url):
    return await http.get(url)

# Retries: 2s → 4s → 8s
```

### 3. Timeout

```python
import asyncio

async def call_with_timeout(url):
    try:
        return await asyncio.wait_for(
            http.get(url),
            timeout=5.0  # 5 second timeout
        )
    except asyncio.TimeoutError:
        return {"error": "Service timeout"}
```

### 4. Bulkhead

**Isolate resources to prevent cascade failures**:

```python
# Separate thread pools for different services
payment_pool = ThreadPoolExecutor(max_workers=10)
inventory_pool = ThreadPoolExecutor(max_workers=5)

async def call_payment():
    return await asyncio.get_event_loop().run_in_executor(
        payment_pool,
        payment_service.call
    )

# If payment service is slow, it only exhausts payment_pool,
# inventory calls still work
```

## API Gateway Pattern

**Centralized entry point for client requests**:

```
Client → API Gateway → [Order, Payment, Inventory services]
```

**Responsibilities**:
- Routing requests to services
- Authentication/authorization
- Rate limiting
- Request/response transformation
- Caching
- Logging/monitoring

**Example (Kong, AWS API Gateway, Nginx)**:

```yaml
# API Gateway config
routes:
  - path: /orders
    service: order-service
    auth: jwt
    ratelimit: 100/minute

  - path: /payments
    service: payment-service
    auth: oauth2
    ratelimit: 50/minute
```

**Backend for Frontend (BFF) Pattern**:

```
Web Client → Web BFF → Services
Mobile App → Mobile BFF → Services
```

Each client type has optimized gateway.

## Service Discovery

### 1. Client-Side Discovery

```python
# Service registry (Consul, Eureka)
registry = ServiceRegistry("http://consul:8500")

# Client looks up service
instances = registry.get_instances("payment-service")
instance = load_balancer.choose(instances)
response = await http.get(f"http://{instance.host}:{instance.port}/charge")
```

### 2. Server-Side Discovery (Load Balancer)

```
Client → Load Balancer → [Service Instance 1, Instance 2, Instance 3]
```

**DNS-based**: Kubernetes services, AWS ELB

## Observability

### Distributed Tracing

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

async def create_order(order):
    with tracer.start_as_current_span("create-order") as span:
        span.set_attribute("order.id", order.id)
        span.set_attribute("order.total", order.total)

        # Trace propagates to payment service
        payment = await payment_service.charge(
            amount=order.total,
            trace_context=span.context
        )

        span.add_event("payment-completed")
        return order
```

**Tools**: Jaeger, Zipkin, AWS X-Ray, Datadog APM

### Log Aggregation

**Structured logging with correlation IDs**:

```python
import logging
import uuid

logger = logging.getLogger(__name__)

async def handle_request(request):
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())

    logger.info("Processing request", extra={
        "correlation_id": correlation_id,
        "service": "order-service",
        "user_id": request.user_id
    })
```

**Tools**: ELK stack (Elasticsearch, Logstash, Kibana), Splunk, Datadog

## Monolith Decomposition Strategies

### 1. Strangler Fig Pattern

**Gradually replace monolith with microservices**:

```
Phase 1: Monolith handles everything
Phase 2: Extract service, proxy some requests to it
Phase 3: More services extracted, proxy more requests
Phase 4: Monolith retired
```

### 2. Branch by Abstraction

1. Create abstraction layer in monolith
2. Implement new service
3. Gradually migrate code behind abstraction
4. Remove old implementation
5. Extract as microservice

### 3. Extract by Bounded Context

Priority order:
1. Services with clear boundaries (authentication, payments)
2. Services changing frequently
3. Services with different scaling needs
4. Services with technology mismatches (e.g., Java monolith, Python ML service)

## Anti-Patterns

| Anti-Pattern | Why Bad | Fix |
|--------------|---------|-----|
| **Distributed Monolith** | Services share database, deploy together | One DB per service, independent deployment |
| **Nanoservices** | Too fine-grained, excessive network calls | Merge related services, follow DDD |
| **Shared Database** | Tight coupling, schema changes break multiple services | Database per service |
| **Synchronous Chains** | A→B→C→D, latency adds up, cascading failures | Async events, parallelize where possible |
| **Chatty Services** | N+1 calls, excessive network overhead | Batch APIs, caching, coarser boundaries |
| **No Circuit Breakers** | Cascading failures bring down system | Circuit breakers + timeouts + retries |
| **No Distributed Tracing** | Impossible to debug cross-service issues | OpenTelemetry, correlation IDs |

## Cross-References

**Related skills**:
- **Message queues** → `message-queues` (RabbitMQ, Kafka patterns)
- **REST APIs** → `rest-api-design` (service interface design)
- **gRPC** → Check if gRPC skill exists
- **Security** → `ordis-security-architect` (service-to-service auth, zero trust)
- **Database** → `database-integration` (per-service databases, migrations)
- **Deployment** → `backend-deployment` (Docker, Kubernetes, CI/CD)
- **Testing** → `api-testing` (contract testing, integration testing)

## Further Reading

- **Building Microservices** by Sam Newman
- **Domain-Driven Design** by Eric Evans
- **Release It!** by Michael Nygard (resilience patterns)
- **Microservices Patterns** by Chris Richardson
