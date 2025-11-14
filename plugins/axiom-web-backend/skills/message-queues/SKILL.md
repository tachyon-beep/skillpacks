---
name: message-queues
description: Use when implementing message queues, choosing between RabbitMQ/Kafka/SQS, ensuring reliable delivery, handling ordering/DLQ/scaling, or building event-driven architectures - covers reliability patterns, schema evolution, monitoring, and production best practices
---

# Message Queues

## Overview

**Message queue specialist covering technology selection, reliability patterns, ordering guarantees, schema evolution, and production operations.**

**Core principle**: Message queues decouple producers from consumers, enabling async processing, load leveling, and resilience - but require careful design for reliability, ordering, monitoring, and operational excellence.

## When to Use This Skill

Use when encountering:

- **Technology selection**: RabbitMQ vs Kafka vs SQS vs SNS
- **Reliability**: Guaranteed delivery, acknowledgments, retries, DLQ
- **Ordering**: Partition keys, FIFO queues, ordered processing
- **Scaling**: Consumer groups, parallelism, backpressure
- **Schema evolution**: Message versioning, Avro, Protobuf
- **Monitoring**: Lag tracking, alerting, distributed tracing
- **Advanced patterns**: Outbox, saga, CQRS, event sourcing
- **Security**: Encryption, IAM, Kafka authentication
- **Testing**: Local testing, chaos engineering, load testing

**Do NOT use for**:
- Request/response APIs → Use REST or GraphQL instead
- Strong consistency required → Use database transactions
- Real-time streaming analytics → See if streaming-specific skill exists

## Technology Selection Matrix

| Factor | RabbitMQ | Apache Kafka | AWS SQS | AWS SNS |
|--------|----------|--------------|---------|---------|
| **Use Case** | Task queues, routing | Event streaming, logs | Simple queues | Pub/sub fanout |
| **Throughput** | 10k-50k msg/s | 100k+ msg/s | 3k msg/s (std), 300 msg/s (FIFO) | 100k+ msg/s |
| **Ordering** | Queue-level | Partition-level (strong) | FIFO queues only | None |
| **Persistence** | Durable queues | Log-based (default) | Managed | Ephemeral (SNS → SQS for durability) |
| **Retention** | Until consumed | Days to weeks | 4 days (std), 14 days max | None (delivery only) |
| **Routing** | Exchanges (topic, fanout, headers) | Topics only | None | Topic-based filtering |
| **Message size** | Up to 128 MB | Up to 1 MB (configurable) | 256 KB | 256 KB |
| **Ops complexity** | Medium (clustering) | High (partitions, replication) | Low (managed) | Low (managed) |
| **Cost** | EC2 self-hosted | Self-hosted or MSK | Pay-per-request | Pay-per-request |

### Decision Tree

```
Are you on AWS and need simple async processing?
  → Yes → **AWS SQS** (start simple)
  → No → Continue...

Do you need event replay or stream processing?
  → Yes → **Kafka** (log-based, replayable)
  → No → Continue...

Do you need complex routing (topic exchange, headers)?
  → Yes → **RabbitMQ** (rich exchange types)
  → No → Continue...

Do you need pub/sub fanout to multiple subscribers?
  → Yes → **SNS** (or Kafka topics with multiple consumer groups)
  → No → **SQS** or **RabbitMQ** for task queues
```

### Migration Path

| Current State | Next Step | Why |
|---------------|-----------|-----|
| No queue | Start with SQS (if AWS) or RabbitMQ | Lowest operational complexity |
| SQS → 1k+ msg/s | Consider Kafka or sharded SQS | SQS throttles at 3k msg/s |
| RabbitMQ → Event sourcing needed | Migrate to Kafka | Kafka's log retention enables replay |
| Kafka → Simple task queue | Consider RabbitMQ or SQS | Kafka is overkill for simple queues |

## Reliability Patterns

### Acknowledgment Modes

| Mode | When Ack Sent | Reliability | Performance | Use Case |
|------|---------------|-------------|-------------|----------|
| **Auto-ack** | On receive | Low (lost on crash) | High | Logs, analytics, best-effort |
| **Manual ack (after processing)** | After success | High (at-least-once) | Medium | Standard production pattern |
| **Transactional** | In transaction | Highest (exactly-once) | Low | Financial, critical data |

### At-Least-Once Delivery Pattern

**SQS**:

```python
# WRONG: Delete before processing
message = sqs.receive_message(QueueUrl=queue_url)['Messages'][0]
sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message['ReceiptHandle'])
process(message['Body'])  # ❌ If this fails, message is lost

# CORRECT: Process, then delete
message = sqs.receive_message(
    QueueUrl=queue_url,
    VisibilityTimeout=300  # 5 minutes to process
)['Messages'][0]

try:
    process(json.loads(message['Body']))
    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message['ReceiptHandle'])
except Exception as e:
    # Message becomes visible again after timeout
    logger.error(f"Processing failed, will retry: {e}")
```

**Kafka**:

```python
# WRONG: Auto-commit before processing
consumer = KafkaConsumer(
    'orders',
    enable_auto_commit=True,  # ❌ Commits offset before processing
    auto_commit_interval_ms=5000
)

for msg in consumer:
    process(msg.value)  # Crash here = message lost

# CORRECT: Manual commit after processing
consumer = KafkaConsumer(
    'orders',
    enable_auto_commit=False
)

for msg in consumer:
    try:
        process(msg.value)
        consumer.commit()  # ✓ Commit only after success
    except Exception as e:
        logger.error(f"Processing failed, will retry: {e}")
        # Don't commit - message will be reprocessed
```

**RabbitMQ**:

```python
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

def callback(ch, method, properties, body):
    try:
        process(json.loads(body))
        ch.basic_ack(delivery_tag=method.delivery_tag)  # ✓ Ack after success
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)  # Requeue

channel.basic_consume(
    queue='orders',
    on_message_callback=callback,
    auto_ack=False  # ✓ Manual acknowledgment
)

channel.start_consuming()
```

### Idempotency (Critical for At-Least-Once)

Since at-least-once delivery guarantees duplicates, **all processing must be idempotent**:

```python
# Pattern 1: Database unique constraint
def process_order(order_id, data):
    db.execute(
        "INSERT INTO orders (id, user_id, amount, created_at) "
        "VALUES (%s, %s, %s, NOW()) "
        "ON CONFLICT (id) DO NOTHING",  # Idempotent
        (order_id, data['user_id'], data['amount'])
    )

# Pattern 2: Distributed lock (Redis)
def process_order_with_lock(order_id, data):
    lock_key = f"lock:order:{order_id}"

    # Try to acquire lock (60s TTL)
    if not redis.set(lock_key, "1", nx=True, ex=60):
        logger.info(f"Order {order_id} already being processed")
        return  # Duplicate, skip

    try:
        # Process order
        create_order(data)
        charge_payment(data['amount'])
    finally:
        redis.delete(lock_key)

# Pattern 3: Idempotency key table
def process_with_idempotency_key(message_id, data):
    with db.transaction():
        # Check if already processed
        result = db.execute(
            "SELECT 1 FROM processed_messages WHERE message_id = %s FOR UPDATE",
            (message_id,)
        )

        if result:
            return  # Already processed

        # Process + record atomically
        process_order(data)
        db.execute(
            "INSERT INTO processed_messages (message_id, processed_at) VALUES (%s, NOW())",
            (message_id,)
        )
```

## Ordering Guarantees

### Kafka: Partition-Level Ordering

**Kafka guarantees ordering within a partition**, not across partitions.

```python
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=['kafka:9092'],
    key_serializer=str.encode,
    value_serializer=lambda v: json.dumps(v).encode()
)

# ✓ Partition key ensures ordering
def publish_order_event(user_id, event_type, data):
    producer.send(
        'orders',
        key=str(user_id),  # All user_id events go to same partition
        value={
            'event_type': event_type,
            'user_id': user_id,
            'data': data,
            'timestamp': time.time()
        }
    )

# User 123's events all go to partition 2 → strict ordering
publish_order_event(123, 'order_placed', {...})
publish_order_event(123, 'payment_processed', {...})
publish_order_event(123, 'shipped', {...})
```

**Partition count determines max parallelism**:

```
Topic: orders (4 partitions)
Consumer group: order-processors

2 consumers → Each processes 2 partitions
4 consumers → Each processes 1 partition (max parallelism)
5 consumers → 1 consumer idle (wasted)

Rule: partition_count >= max_consumers_needed
```

### SQS FIFO: MessageGroupId Ordering

```python
import boto3

sqs = boto3.client('sqs')

# FIFO queue guarantees ordering per MessageGroupId
sqs.send_message(
    QueueUrl='orders.fifo',
    MessageBody=json.dumps(event),
    MessageGroupId=f"user-{user_id}",  # Like Kafka partition key
    MessageDeduplicationId=f"{event_id}-{timestamp}"  # Prevent duplicates
)

# Throughput limit: 300 msg/s per MessageGroupId
# Workaround: Use multiple MessageGroupIds if possible
```

### RabbitMQ: Single Consumer Ordering

```python
# RabbitMQ guarantees ordering if single consumer
channel.basic_qos(prefetch_count=1)  # Process one at a time

channel.basic_consume(
    queue='orders',
    on_message_callback=callback,
    auto_ack=False
)

# Multiple consumers break ordering unless using consistent hashing
```

## Dead Letter Queues (DLQ)

### Retry Strategy with Exponential Backoff

**SQS with DLQ**:

```python
# Infrastructure setup
main_queue = sqs.create_queue(
    QueueName='orders',
    Attributes={
        'RedrivePolicy': json.dumps({
            'deadLetterTargetArn': dlq_arn,
            'maxReceiveCount': '3'  # After 3 failures → DLQ
        }),
        'VisibilityTimeout': '300'
    }
)

# Consumer with retry logic
def process_with_retry(message):
    attempt = int(message.attributes.get('ApproximateReceiveCount', 0))

    try:
        process_order(json.loads(message.body))
        message.delete()

    except RetriableError as e:
        # Exponential backoff: 10s, 20s, 40s, 80s, ...
        backoff = min(300, 2 ** attempt * 10)
        message.change_visibility(VisibilityTimeout=backoff)
        logger.warning(f"Retriable error (attempt {attempt}), retry in {backoff}s")

    except PermanentError as e:
        # Send to DLQ immediately
        logger.error(f"Permanent error: {e}")
        send_to_dlq(message, error=str(e))
        message.delete()

# Error classification
class RetriableError(Exception):
    """Network timeout, rate limit, DB unavailable"""
    pass

class PermanentError(Exception):
    """Invalid data, missing field, business rule violation"""
    pass
```

**Kafka DLQ Pattern**:

```python
from kafka import KafkaConsumer, KafkaProducer

consumer = KafkaConsumer('orders', group_id='processor')
dlq_producer = KafkaProducer(bootstrap_servers=['kafka:9092'])

def process_with_dlq(message):
    retry_count = message.headers.get('retry_count', 0)

    try:
        process_order(message.value)
        consumer.commit()

    except RetriableError as e:
        if retry_count < 3:
            # Send to retry topic with delay
            delay_minutes = 2 ** retry_count  # 1min, 2min, 4min
            retry_producer.send(
                f'orders-retry-{delay_minutes}min',
                value=message.value,
                headers={'retry_count': retry_count + 1}
            )
        else:
            # Max retries → DLQ
            dlq_producer.send(
                'orders-dlq',
                value=message.value,
                headers={'error': str(e), 'retry_count': retry_count}
            )
        consumer.commit()  # Don't reprocess from main topic

    except PermanentError as e:
        # Immediate DLQ
        dlq_producer.send('orders-dlq', value=message.value, headers={'error': str(e)})
        consumer.commit()
```

### DLQ Monitoring & Recovery

```python
# Alert on DLQ depth
def check_dlq_depth():
    attrs = sqs.get_queue_attributes(
        QueueUrl=dlq_url,
        AttributeNames=['ApproximateNumberOfMessages']
    )
    depth = int(attrs['Attributes']['ApproximateNumberOfMessages'])

    if depth > 10:
        alert(f"DLQ has {depth} messages - investigate!")

# Manual recovery
def replay_from_dlq():
    """Fix root cause, then replay"""
    messages = dlq.receive_messages(MaxNumberOfMessages=10)

    for msg in messages:
        data = json.loads(msg.body)

        # Fix data issue
        if 'customer_email' not in data:
            data['customer_email'] = lookup_email(data['user_id'])

        # Replay to main queue
        main_queue.send_message(MessageBody=json.dumps(data))
        msg.delete()
```

## Message Schema Evolution

### Versioning Strategies

**Pattern 1: Version field in message**:

```python
# v1 message
{
  "version": "1.0",
  "order_id": "123",
  "amount": 99.99
}

# v2 message (added currency)
{
  "version": "2.0",
  "order_id": "123",
  "amount": 99.99,
  "currency": "USD"
}

# Consumer handles both versions
def process_order(message):
    if message['version'] == "1.0":
        amount = message['amount']
        currency = "USD"  # Default for v1
    elif message['version'] == "2.0":
        amount = message['amount']
        currency = message['currency']
    else:
        raise ValueError(f"Unsupported version: {message['version']}")
```

**Pattern 2: Apache Avro (Kafka best practice)**:

```python
from confluent_kafka import avro
from confluent_kafka.avro import AvroProducer, AvroConsumer

# Define schema
value_schema = avro.loads('''
{
  "type": "record",
  "name": "Order",
  "fields": [
    {"name": "order_id", "type": "string"},
    {"name": "amount", "type": "double"},
    {"name": "currency", "type": "string", "default": "USD"}  # Backward compatible
  ]
}
''')

# Producer
producer = AvroProducer({
    'bootstrap.servers': 'kafka:9092',
    'schema.registry.url': 'http://schema-registry:8081'
}, default_value_schema=value_schema)

producer.produce(topic='orders', value={
    'order_id': '123',
    'amount': 99.99,
    'currency': 'USD'
})

# Consumer automatically validates schema
consumer = AvroConsumer({
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'processor',
    'schema.registry.url': 'http://schema-registry:8081'
})
```

**Avro Schema Evolution Rules**:

| Change | Compatible? | Notes |
|--------|-------------|-------|
| Add field with default | ✓ Backward compatible | Old consumers ignore new field |
| Remove field | ✓ Forward compatible | New consumers must handle missing field |
| Rename field | ❌ Breaking | Requires migration |
| Change field type | ❌ Breaking | Requires new topic or migration |

**Pattern 3: Protobuf (alternative to Avro)**:

```protobuf
syntax = "proto3";

message Order {
  string order_id = 1;
  double amount = 2;
  string currency = 3;  // New field, backward compatible
}
```

### Schema Registry (Kafka)

```
Producer → Schema Registry (validate) → Kafka
Consumer → Kafka → Schema Registry (deserialize)

Benefits:
- Centralized schema management
- Automatic validation
- Schema evolution enforcement
- Type safety
```

## Monitoring & Observability

### Key Metrics

| Metric | Alert Threshold | Why It Matters |
|--------|----------------|----------------|
| **Queue depth** | > 1000 (or 5min processing time) | Consumers can't keep up |
| **Consumer lag** (Kafka) | > 100k messages or > 5 min | Consumers falling behind |
| **DLQ depth** | > 10 | Messages failing repeatedly |
| **Processing time p99** | > 5 seconds | Slow processing blocks queue |
| **Error rate** | > 5% | Widespread failures |
| **Redelivery rate** | > 10% | Idempotency issues or transient errors |

### Consumer Lag Monitoring (Kafka)

```python
from kafka import KafkaAdminClient, TopicPartition

admin = KafkaAdminClient(bootstrap_servers=['kafka:9092'])

def check_consumer_lag(group_id, topic):
    # Get committed offsets
    committed = admin.list_consumer_group_offsets(group_id)

    # Get latest offsets (highwater mark)
    consumer = KafkaConsumer(bootstrap_servers=['kafka:9092'])
    partitions = [TopicPartition(topic, p) for p in range(partition_count)]
    latest = consumer.end_offsets(partitions)

    # Calculate lag
    total_lag = 0
    for partition in partitions:
        committed_offset = committed[partition].offset
        latest_offset = latest[partition]
        lag = latest_offset - committed_offset
        total_lag += lag

        if lag > 10000:
            alert(f"Partition {partition.partition} lag: {lag}")

    return total_lag

# Alert if total lag > 100k
if check_consumer_lag('order-processor', 'orders') > 100000:
    alert("Consumer lag critical!")
```

### Distributed Tracing Across Queues

```python
from opentelemetry import trace
from opentelemetry.propagate import inject, extract

tracer = trace.get_tracer(__name__)

# Producer: Inject trace context
def publish_with_trace(topic, message):
    with tracer.start_as_current_span("publish-order") as span:
        headers = {}
        inject(headers)  # Inject trace context into headers

        producer.send(
            topic,
            value=message,
            headers=list(headers.items())
        )

# Consumer: Extract trace context
def consume_with_trace(message):
    context = extract(dict(message.headers))

    with tracer.start_as_current_span("process-order", context=context) as span:
        process_order(message.value)
        span.set_attribute("order.id", message.value['order_id'])

# Trace spans: API → Producer → Queue → Consumer → DB
# Shows end-to-end latency including queue wait time
```

## Backpressure & Circuit Breakers

### Rate Limiting Consumers

```python
import time
from collections import deque

class RateLimitedConsumer:
    def __init__(self, max_per_second=100):
        self.max_per_second = max_per_second
        self.requests = deque()

    def consume(self, message):
        now = time.time()

        # Remove requests older than 1 second
        while self.requests and self.requests[0] < now - 1:
            self.requests.popleft()

        # Check rate limit
        if len(self.requests) >= self.max_per_second:
            sleep_time = 1 - (now - self.requests[0])
            time.sleep(sleep_time)

        self.requests.append(time.time())
        process(message)
```

### Circuit Breaker for Downstream Dependencies

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def call_payment_service(order_id, amount):
    response = requests.post(
        'https://payment-service/charge',
        json={'order_id': order_id, 'amount': amount},
        timeout=5
    )

    if response.status_code >= 500:
        raise ServiceUnavailableError()

    return response.json()

def process_order(message):
    try:
        result = call_payment_service(message['order_id'], message['amount'])
        # ... continue processing
    except CircuitBreakerError:
        # Circuit open - don't overwhelm failing service
        logger.warning("Payment service circuit open, requeueing message")
        raise RetriableError("Circuit breaker open")
```

## Advanced Patterns

### Outbox Pattern (Reliable Publishing)

**Problem**: How to atomically update database AND publish message?

```python
# ❌ WRONG: Dual write (can fail between DB and queue)
def create_order(data):
    db.execute("INSERT INTO orders (...) VALUES (...)")
    producer.send('orders', data)  # ❌ If this fails, DB updated but no event

# ✓ CORRECT: Outbox pattern
def create_order_with_outbox(data):
    with db.transaction():
        # 1. Insert order
        db.execute("INSERT INTO orders (id, user_id, amount) VALUES (%s, %s, %s)",
                   (data['id'], data['user_id'], data['amount']))

        # 2. Insert into outbox (same transaction)
        db.execute("INSERT INTO outbox (event_type, payload) VALUES (%s, %s)",
                   ('order.created', json.dumps(data)))

    # Separate process reads outbox and publishes

# Outbox processor (separate worker)
def process_outbox():
    while True:
        events = db.execute("SELECT * FROM outbox WHERE published_at IS NULL LIMIT 10")

        for event in events:
            try:
                producer.send(event['event_type'], json.loads(event['payload']))
                db.execute("UPDATE outbox SET published_at = NOW() WHERE id = %s", (event['id'],))
            except Exception as e:
                logger.error(f"Failed to publish event {event['id']}: {e}")
                # Will retry on next iteration

        time.sleep(1)
```

### Saga Pattern (Distributed Transactions)

See `microservices-architecture` skill for full saga patterns (choreography vs orchestration).

**Quick reference for message-based saga**:

```python
# Order saga coordinator publishes commands
def create_order_saga(order_data):
    saga_id = str(uuid.uuid4())

    # Step 1: Reserve inventory
    producer.send('inventory-commands', {
        'command': 'reserve',
        'saga_id': saga_id,
        'order_id': order_data['order_id'],
        'items': order_data['items']
    })

    # Inventory service responds on 'inventory-events'
    # If success → proceed to step 2
    # If failure → compensate (cancel order)
```

## Security

### Message Encryption

**SQS**: Server-side encryption (SSE) with KMS

```python
sqs.create_queue(
    QueueName='orders-encrypted',
    Attributes={
        'KmsMasterKeyId': 'alias/my-key',  # AWS KMS
        'KmsDataKeyReusePeriodSeconds': '300'
    }
)
```

**Kafka**: Encryption in transit + at rest

```python
# SSL/TLS for in-transit encryption
producer = KafkaProducer(
    bootstrap_servers=['kafka:9093'],
    security_protocol='SSL',
    ssl_cafile='/path/to/ca-cert',
    ssl_certfile='/path/to/client-cert',
    ssl_keyfile='/path/to/client-key'
)

# Encryption at rest (Kafka broker config)
# log.dirs=/encrypted-volume  # Use encrypted EBS volumes
```

### Authentication & Authorization

**SQS**: IAM policies

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"AWS": "arn:aws:iam::123456789012:role/OrderService"},
    "Action": ["sqs:SendMessage"],
    "Resource": "arn:aws:sqs:us-east-1:123456789012:orders"
  }]
}
```

**Kafka**: SASL/SCRAM authentication

```python
producer = KafkaProducer(
    bootstrap_servers=['kafka:9093'],
    security_protocol='SASL_SSL',
    sasl_mechanism='SCRAM-SHA-512',
    sasl_plain_username='order-service',
    sasl_plain_password='secret'
)
```

**Kafka ACLs** (authorization):

```bash
# Grant order-service permission to write to orders topic
kafka-acls --add \
  --allow-principal User:order-service \
  --operation Write \
  --topic orders
```

## Testing Strategies

### Local Testing

**LocalStack for SQS/SNS**:

```python
# docker-compose.yml
services:
  localstack:
    image: localstack/localstack
    environment:
      - SERVICES=sqs,sns

# Test code
import boto3

sqs = boto3.client(
    'sqs',
    endpoint_url='http://localhost:4566',  # LocalStack
    region_name='us-east-1'
)

queue_url = sqs.create_queue(QueueName='test-orders')['QueueUrl']
sqs.send_message(QueueUrl=queue_url, MessageBody='test')
```

**Kafka in Docker**:

```yaml
# docker-compose.yml
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:latest
    ports:
      - "9092:9092"
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
```

### Integration Testing

```python
import pytest
from testcontainers.kafka import KafkaContainer

@pytest.fixture
def kafka():
    with KafkaContainer() as kafka:
        yield kafka.get_bootstrap_server()

def test_order_processing(kafka):
    producer = KafkaProducer(bootstrap_servers=kafka)
    consumer = KafkaConsumer('orders', bootstrap_servers=kafka, auto_offset_reset='earliest')

    # Publish message
    producer.send('orders', value=b'{"order_id": "123"}')
    producer.flush()

    # Consume and verify
    message = next(consumer)
    assert json.loads(message.value)['order_id'] == '123'
```

### Chaos Engineering

```python
# Test consumer failure recovery
def test_consumer_crash_recovery():
    # Start consumer
    consumer_process = subprocess.Popen(['python', 'consumer.py'])
    time.sleep(2)

    # Publish message
    producer.send('orders', value=test_order)
    producer.flush()

    # Kill consumer mid-processing
    consumer_process.kill()

    # Restart consumer
    consumer_process = subprocess.Popen(['python', 'consumer.py'])
    time.sleep(5)

    # Verify message was reprocessed (idempotency!)
    assert db.execute("SELECT COUNT(*) FROM orders WHERE id = %s", (test_order['id'],))[0] == 1
```

## Anti-Patterns

| Anti-Pattern | Why Bad | Fix |
|--------------|---------|-----|
| **Auto-ack before processing** | Messages lost on crash | Manual ack after processing |
| **No idempotency** | Duplicates cause data corruption | Unique constraints, locks, or idempotency keys |
| **No DLQ** | Poison messages block queue | Configure DLQ with maxReceiveCount |
| **No monitoring** | Can't detect consumer lag or failures | Monitor lag, depth, error rate |
| **Synchronous message processing** | Low throughput | Batch processing, parallel consumers |
| **Large messages** | Exceeds queue limits, slow transfer | Store in S3, send reference in message |
| **No schema versioning** | Breaking changes break consumers | Use Avro/Protobuf with schema registry |
| **Shared consumer instances** | Race conditions, duplicate processing | Use consumer groups (Kafka) or visibility timeout (SQS) |

## Technology-Specific Patterns

### RabbitMQ Exchanges

```python
# Topic exchange for routing
channel.exchange_declare(exchange='orders', exchange_type='topic')

# Bind queues with patterns
channel.queue_bind(exchange='orders', queue='us-orders', routing_key='order.us.*')
channel.queue_bind(exchange='orders', queue='eu-orders', routing_key='order.eu.*')

# Publish with routing key
channel.basic_publish(
    exchange='orders',
    routing_key='order.us.california',  # Goes to us-orders queue
    body=json.dumps(order)
)

# Fanout exchange for pub/sub
channel.exchange_declare(exchange='analytics', exchange_type='fanout')
# All bound queues receive every message
```

### Kafka Connect (Data Integration)

```json
{
  "name": "mysql-source",
  "config": {
    "connector.class": "io.confluent.connect.jdbc.JdbcSourceConnector",
    "connection.url": "jdbc:mysql://localhost:3306/mydb",
    "table.whitelist": "orders",
    "mode": "incrementing",
    "incrementing.column.name": "id",
    "topic.prefix": "mysql-"
  }
}
```

**Use cases**:
- Stream DB changes to Kafka (CDC)
- Sink Kafka to Elasticsearch, S3, databases
- No custom code needed for common integrations

## Batching Optimizations

### Batch Size Tuning

```python
# SQS batch receiving (up to 10 messages)
messages = sqs.receive_messages(
    QueueUrl=queue_url,
    MaxNumberOfMessages=10,  # Fetch 10 at once
    WaitTimeSeconds=20  # Long polling (reduces empty receives)
)

# Process in parallel
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(process, msg) for msg in messages]
    for future in futures:
        future.result()

# Kafka batch consuming
consumer = KafkaConsumer(
    'orders',
    max_poll_records=500,  # Fetch 500 messages per poll
    fetch_min_bytes=1024  # Wait for at least 1KB before returning
)

for messages in consumer:
    batch_process(messages)  # Process 500 at once
```

**Batch size tradeoffs**:

| Batch Size | Throughput | Latency | Memory |
|------------|------------|---------|--------|
| 1 | Low | Low | Low |
| 10-100 | Medium | Medium | Medium |
| 500+ | High | High | High |

**Recommendation**: Start with 10-100, increase for higher throughput if latency allows.

## Cross-References

**Related skills**:
- **Microservices communication** → `microservices-architecture` (saga, event-driven)
- **FastAPI async** → `fastapi-development` (consuming queues in FastAPI)
- **REST vs async** → `rest-api-design` (when to use queues vs HTTP)
- **Security** → `ordis-security-architect` (encryption, IAM, compliance)
- **Testing** → `api-testing` (integration testing strategies)

## Further Reading

- **Enterprise Integration Patterns** by Gregor Hohpe (message patterns)
- **Designing Data-Intensive Applications** by Martin Kleppmann (Kafka internals)
- **RabbitMQ in Action** by Alvaro Videla
- **Kafka: The Definitive Guide** by Neha Narkhede
- **AWS SQS Best Practices**: https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-best-practices.html
