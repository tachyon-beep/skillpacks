---
name: diagram-conventions
description: Choose diagram types (sequence, component, data flow, state) with semantic labeling standards
---

# Diagram Conventions

## Overview

Choose the right diagram type for what you're documenting. Core principle: **Diagram type matches what you're showing** (time-based interactions → sequence, static structure → component, etc.).

**Key insight**: Wrong diagram type obscures meaning. Right diagram type makes it obvious.

## When to Use

Load this skill when:
- Creating diagrams for documentation
- Choosing between diagram types
- Labeling components and relationships
- Reviewing diagrams for clarity

**Symptoms you need this**:
- "Should I use a flowchart or sequence diagram?"
- Creating architecture documentation
- Documenting API flows, system interactions
- Explaining complex decision logic

**Don't use for**:
- Writing code (not documentation)
- Non-technical diagrams (org charts, process flows)

## Decision Tree: Choosing Diagram Type

```
What are you documenting?
│
├─ Interactions between systems over time?
│  (API calls, message exchanges, request-response flows)
│  └─→ Use SEQUENCE DIAGRAM
│
├─ System components and their relationships?
│  (Services, databases, queues, static architecture)
│  └─→ Use COMPONENT/ARCHITECTURE DIAGRAM
│
├─ Data movement through transformations?
│  (ETL pipelines, data processing, input→output)
│  └─→ Use DATA FLOW DIAGRAM
│
├─ State changes over lifecycle?
│  (Order states: pending→paid→shipped, connection states)
│  └─→ Use STATE DIAGRAM
│
└─ Simple decision logic with branches?
   (2-4 conditions, clear branching)
   ├─ Small (≤3 conditions) → FLOWCHART acceptable
   └─ Large (>3 conditions) → Use DECISION TABLE or PSEUDO-CODE instead
```

---

## Diagram Type 1: Sequence Diagram

**Use for**: Interactions between systems/actors over time.

**When to use**:
- API request-response flows
- Authentication sequences
- Message exchanges between services
- Anything with temporal ordering (this happens THEN that happens)

### Structure

```
Actor/System 1   Actor/System 2   Actor/System 3
     |                |                |
     |-- message 1 -->|                |
     |                |-- message 2 -->|
     |                |<-- response ---|
     |<-- response ---|                |
     |                |                |
```

**Time flows downward**. Each arrow = message/call with label showing WHAT is sent.

### Example: OAuth Authentication

```
User          Frontend        Google Auth      Backend
  |               |               |               |
  |-- Click Login →              |               |
  |               |-- Redirect -->|               |
  |<------------- Redirect to Google Auth --------|
  |-- Enter credentials -------->|               |
  |<-- Auth code ----------------|               |
  |               |<-- Redirect with code --------|
  |               |-- POST /auth/callback ------->|
  |               |               |<-- Exchange code for token
  |               |               |--- Access token
  |               |<-- Session token -------------|
  |<-- Redirect to dashboard ----|               |
```

### Labeling Rules

**Arrow labels** = What is sent/requested:
- ✅ "POST /users with user_data"
- ✅ "Return 200 OK with user_id"
- ✅ "Publish OrderCreated event"
- ❌ "Request" (too vague)
- ❌ "Step 3" (not semantic)

**Actor/System names** = Specific entities:
- ✅ "API Gateway", "Auth Service", "Users Database"
- ❌ "Service1", "Database" (too generic)

---

## Diagram Type 2: Component/Architecture Diagram

**Use for**: Static system structure and relationships.

**When to use**:
- Microservices architecture
- System components and dependencies
- Database relationships
- Infrastructure layout
- No temporal aspect (not "then what happens", just "what connects to what")

### Structure

```
┌─────────────┐
│ Component A │
└──────┬──────┘
       │ relationship_label
       ↓
┌─────────────┐
│ Component B │
└─────────────┘
```

**Components** = boxes with names. **Relationships** = arrows with meaningful labels.

### Example: Microservices Architecture

```
                    ┌──────────────────┐
                    │   API Gateway    │
                    │ (Routes requests)│
                    └────┬────────┬────┘
                         │        │
           authenticates │        │ queries orders
                         │        │
                    ┌────▼───┐  ┌─▼──────────────┐
                    │ Auth   │  │ Order Service  │
                    │Service │  └────┬───────────┘
                    └────┬───┘       │
                         │           │ publishes OrderCreated
                         │ queries   │
                         ↓           ↓
                   ┌──────────┐  ┌────────────┐
                   │ Users DB │  │   Queue    │
                   └──────────┘  └─────┬──────┘
                                       │ consumes
                                       ↓
                             ┌──────────────────┐
                             │ Notification     │
                             │ Service          │
                             └──────────────────┘
```

### Labeling Rules

**Component names** = What they are + brief function:
- ✅ "Auth Service (validates tokens)"
- ✅ "Users Database (PostgreSQL)"
- ✅ "Message Queue (RabbitMQ)"
- ❌ "Service", "DB", "Queue" (too generic)

**Relationship labels** = Specific action:
- ✅ "authenticates user", "queries orders", "publishes OrderCreated"
- ✅ "reads from", "writes to", "subscribes to"
- ❌ "uses", "talks to", "connects" (too vague)

**Consistency**: Use same terminology as code/documentation.

---

## Diagram Type 3: Data Flow Diagram

**Use for**: Data movement and transformations.

**When to use**:
- ETL pipelines
- Data processing workflows
- Input → transformation → output flows

### Structure

```
[Input Source] → [Transform] → [Transform] → [Output Destination]
```

### Example: Data Pipeline

```
CSV Files       Parse CSV      Validate      Enrich with      Write to
(S3 Bucket) →  (extract) →    (check) →    Metadata   →    Database
                  │              │             │              (Postgres)
                  ↓              ↓             ↓
              JSON objects  Valid records  Records +
                                           timestamps
```

### Labeling Rules

**Transformation steps** = What happens to data:
- ✅ "Parse CSV to JSON"
- ✅ "Validate schema"
- ✅ "Enrich with timestamps"
- ❌ "Process", "Handle" (not specific)

**Data labels** = What format/content:
- ✅ "CSV records", "JSON objects", "Valid records"
- ✅ Show intermediate formats if they change

---

## Diagram Type 4: State Diagram

**Use for**: State changes over entity lifecycle.

**When to use**:
- Order states (pending → paid → shipped)
- Connection states (disconnected → connecting → connected)
- Workflow states (draft → review → approved)

### Structure

```
[State 1] --event/condition--> [State 2] --event/condition--> [State 3]
```

### Example: Order Lifecycle

```
┌─────────┐   payment     ┌──────────┐   fulfill    ┌──────────┐
│ Pending │  received     │   Paid   │   order      │ Shipped  │
└─────────┘ ───────────→  └──────────┘ ──────────→  └──────────┘
     │                          │                         │
     │ cancel                   │ refund                  │ deliver
     ↓                          ↓                         ↓
┌──────────┐             ┌──────────┐             ┌──────────┐
│Cancelled │             │Refunded  │             │Delivered │
└──────────┘             └──────────┘             └──────────┘
```

### Labeling Rules

**States** = Noun describing entity status:
- ✅ "Pending", "Paid", "Shipped"
- ✅ "Connected", "Disconnected"
- ❌ "Processing" (too vague - processing what?)

**Transitions** = Event or condition causing change:
- ✅ "payment received", "cancel order", "fulfill order"
- ✅ "timeout expires", "user clicks submit"
- ❌ "go to next state" (not semantic)

---

## When Flowcharts Become Anti-Patterns

Flowcharts are overused. Use alternatives for:

### Anti-Pattern 1: Complex Business Logic

❌ **Wrong**: Flowchart with 15+ decision diamonds

✅ **Right**: Decision table or pseudo-code

**Example**: Authorization logic (authenticated? admin? owns resource?)

**Better as decision table**:
| Authenticated | Admin | Owns Resource | Result |
|---|---|---|---|
| No | - | - | 401 |
| Yes | Yes | - | Allow |
| Yes | No | Yes | Allow |
| Yes | No | No | Deny |

**Why**: Flowchart with 4+ conditions becomes spaghetti. Table is scannable.

---

### Anti-Pattern 2: Long Procedures

❌ **Wrong**: Flowchart showing deployment steps (20 boxes)

✅ **Right**: Numbered list

**Example**:
```markdown
## Deployment Steps

1. Build Docker image: `docker build -t app:v1.0 .`
2. Push to registry: `docker push registry/app:v1.0`
3. Update Kubernetes: `kubectl set image deployment/app app=registry/app:v1.0`
4. Verify pods running: `kubectl get pods -l app=app`
5. Check logs: `kubectl logs -f deployment/app`
```

**Why**: Sequential steps don't need visual diagram. Numbered list is clearer.

---

### Anti-Pattern 3: Duplicating Code

❌ **Wrong**: Flowchart replicating function logic that exists in code

✅ **Right**: Link to code, don't duplicate

**Example**:
```markdown
## Token Validation

See `validate_token()` in `auth/token_validator.py:45-78`

High-level: Checks signature, expiration, scopes.
```

**Why**: Flowchart duplicates code. Gets out of sync when code changes.

---

## Flowchart Usage Checklist

**Use flowchart ONLY if all these are true:**
- [ ] Fewer than 4 decision points
- [ ] Not duplicating existing code
- [ ] Branching logic is core to understanding (not just procedural steps)
- [ ] No simpler alternative (decision table, list, pseudo-code)

**If any are false**: Use alternative format.

---

## Semantic Labeling Standards

### Rule 1: No Generic Names

❌ **Wrong**:
- "Service1", "Service2"
- "Step1", "Step2"
- "Database", "Queue"
- "Process", "Handle"

✅ **Right**:
- "Auth Service", "Order Service"
- "Parse CSV", "Validate Schema"
- "Users Database (PostgreSQL)", "Message Queue (RabbitMQ)"
- "Authenticate user", "Publish OrderCreated event"

**Principle**: Names should have semantic meaning. If you removed the diagram and only saw labels, you'd understand what they do.

---

### Rule 2: Consistent Terminology

**Use same terms as code/documentation.**

If code has `AuthenticationService`, diagram should say "Authentication Service", not "Login Handler".

If code publishes `OrderCreatedEvent`, diagram should say "publishes OrderCreated", not "sends message".

**Why**: Readers switching between diagram and code should see same concepts.

---

### Rule 3: Meaningful Relationships

❌ **Wrong**:
- Arrow with no label
- "connects to", "uses"
- "talks to", "calls"

✅ **Right**:
- "authenticates user with JWT"
- "queries orders by user_id"
- "publishes OrderCreated event to queue"
- "consumes from notifications topic"

**Pattern**: `[Verb] [Object] [with/via/using] [Details]`

**Examples**:
- "queries users with SQL SELECT"
- "publishes to orders_topic via Kafka"
- "validates signature using RSA public key"

---

## Quick Reference: Diagram Selection

| What You're Documenting | Use This Diagram | Key Feature |
|---|---|---|
| **API calls between services** | Sequence | Shows temporal order (time flows down) |
| **Microservices architecture** | Component | Shows static structure (boxes and relationships) |
| **ETL pipeline** | Data Flow | Shows transformations (input → process → output) |
| **Order/connection states** | State | Shows lifecycle (state → event → state) |
| **Simple decision (≤3 conditions)** | Flowchart | Shows branching logic |
| **Complex decision (>3 conditions)** | Decision Table | Scannable conditions and outcomes |
| **Sequential steps (deployment)** | Numbered List | No visual needed for linear steps |

---

## Common Mistakes

### ❌ Wrong Diagram Type for Content

**Wrong**: Sequence diagram for system architecture (no temporal aspect)

**Right**: Component diagram (static structure)

**Why**: Sequence diagrams imply ordering over time. Architecture is static.

---

### ❌ Generic Labels

**Wrong**:
```
Service1 → Service2 → Database
```

**Right**:
```
Auth Service (validates JWT)
  → User Service (queries user data)
  → Users Database (PostgreSQL)
```

**Why**: Generic labels force reader to guess. Semantic labels explain.

---

### ❌ Flowchart for Complex Logic

**Wrong**: Flowchart with 10+ decision diamonds (authentication logic)

**Right**: Decision table showing all auth outcomes

**Why**: Large flowcharts are spaghetti. Tables are scannable.

---

### ❌ Missing Relationship Labels

**Wrong**:
```
Auth Service → Database
(arrow with no label)
```

**Right**:
```
Auth Service → Database
   "queries users by email"
```

**Why**: Unlabeled arrows are ambiguous. Does it read? Write? Both?

---

### ❌ Inconsistent Terminology

**Wrong**: Code calls it `OrderService`, diagram says "Purchase Handler"

**Right**: Code and diagram both say "Order Service"

**Why**: Different terms confuse readers switching between diagram and code.

---

## Cross-References

**Use WITH this skill**:
- `muna/technical-writer/documentation-structure` - Diagrams go in specific sections (architecture docs, API flows)
- `muna/technical-writer/clarity-and-style` - Diagrams should be scannable, well-labeled

**Use AFTER this skill**:
- `muna/technical-writer/documentation-testing` - Verify diagrams are understandable

## Real-World Impact

**Well-chosen diagrams using these conventions:**
- **Sequence diagram for OAuth flow**: Onboarding developers understood flow in 5 minutes (vs 30 minutes reading prose)
- **Decision table replacing 12-branch flowchart**: Authorization logic bugs reduced from 8 to 0 (scannable table caught missed cases)
- **Component diagram with semantic labels**: New engineers could navigate codebase without asking "what is Service2?" (eliminated 15+ Slack questions per week)

**Key lesson**: **Right diagram type + semantic labels = immediate understanding. Wrong type or generic labels = confusion.**
