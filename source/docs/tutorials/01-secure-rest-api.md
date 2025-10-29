# Tutorial 1: Securing a REST API (End-to-End)

**Estimated Time:** 45-60 minutes

**Difficulty:** Intermediate

---

## Introduction

This tutorial walks you through securing a customer data platform REST API from initial threat modeling through to final architecture review. You'll learn to apply systematic security analysis using Claude Code skills for security architecture.

### Scenario

You're building a REST API for a customer data platform that handles personally identifiable information (PII). The API must be secure, compliant with data protection regulations, and production-ready.

**API Details:**
- **Endpoints:**
  - `GET /customers` - List all customers (paginated)
  - `POST /customers` - Create new customer
  - `GET /customers/{id}` - Retrieve customer details
  - `PUT /customers/{id}` - Update customer information
  - `DELETE /customers/{id}` - Remove customer record

- **Data Handled:**
  - Customer names (first, last)
  - Email addresses
  - Phone numbers
  - Physical addresses
  - Account creation dates
  - Customer preferences

- **Technology Stack:**
  - **API Gateway:** AWS API Gateway
  - **Compute:** AWS Lambda (Node.js 20.x)
  - **Database:** AWS RDS PostgreSQL 15
  - **Authentication:** OAuth 2.0 with JWT tokens
  - **Hosting:** AWS (us-east-1)

- **Compliance Requirements:**
  - GDPR (European customers)
  - CCPA (California customers)
  - General data protection best practices

### What You'll Accomplish

By the end of this tutorial, you will have:

1. Performed comprehensive threat modeling using STRIDE methodology
2. Designed security controls mapped to identified threats
3. Created complete security documentation with threat-control traceability
4. Reviewed the overall security architecture for gaps and improvements

---

## Prerequisites

Before starting this tutorial, ensure you have:

- **Claude Code installed and configured**
- **Access to the security-architect skillpack** (ordis/security-architect)
- **Basic understanding of REST APIs** (HTTP methods, status codes, headers)
- **Familiarity with OAuth 2.0 and JWT** (high-level concept)
- **Understanding of cloud deployment** (AWS basics helpful but not required)
- **Text editor** for documenting outputs (VS Code, Sublime, etc.)

**Optional:**
- API design document or OpenAPI specification
- Architecture diagram (can be created during tutorial)
- Existing security policies or compliance requirements

---

## Step 1: Initial Threat Model

**Estimated Time:** 15 minutes

### Loading the Skill

In Claude Code, use the threat modeling skill:

```
I'm using ordis/security-architect/threat-modeling to analyze my customer data REST API.
```

### Input Prompt

Provide Claude with detailed context about your system:

```
I'm building a REST API for a customer data platform with the following details:

**System Architecture:**
- API Gateway: AWS API Gateway (public endpoint)
- Backend: AWS Lambda functions (Node.js 20.x)
- Database: AWS RDS PostgreSQL 15 (private subnet)
- Authentication: OAuth 2.0 with JWT tokens
- Region: AWS us-east-1

**API Endpoints:**
1. GET /customers - List customers (paginated, max 100 per page)
2. POST /customers - Create customer record
3. GET /customers/{id} - Get customer by ID
4. PUT /customers/{id} - Update customer
5. DELETE /customers/{id} - Delete customer

**Data Handled:**
- Customer PII: names, emails, phone numbers, physical addresses
- Account metadata: creation dates, last updated timestamps
- Customer preferences: marketing opt-in/out

**Current Security Controls:**
- OAuth 2.0 authentication via Auth0
- JWT tokens (1-hour expiration)
- HTTPS/TLS 1.3 for all API traffic
- VPC with private subnets for database

**Compliance Requirements:**
- GDPR compliance (European customers)
- CCPA compliance (California customers)

Please perform a comprehensive STRIDE threat analysis of this API, focusing on:
1. Authentication and authorization threats
2. Data exposure risks
3. API endpoint-specific vulnerabilities
4. Infrastructure security concerns
```

### Expected Output

The threat modeling skill will generate structured STRIDE analysis. Here's what you should expect:

#### STRIDE Analysis Summary

**Spoofing Threats:**

| Threat ID | Threat | Component | Risk Level |
|-----------|--------|-----------|------------|
| S-01 | JWT token theft via XSS in client application | Authentication | High |
| S-02 | Token replay attack using stolen valid JWT | API Gateway | High |
| S-03 | Impersonation via compromised OAuth credentials | Auth0 Integration | Critical |
| S-04 | API key leakage in client-side code | Client Applications | Medium |

**Tampering Threats:**

| Threat ID | Threat | Component | Risk Level |
|-----------|--------|-----------|------------|
| T-01 | Unauthorized modification of customer data via PUT endpoint | Lambda Functions | High |
| T-02 | SQL injection in customer lookup queries | Database Layer | Critical |
| T-03 | Request parameter manipulation to access other customers | API Endpoints | High |
| T-04 | JWT claims manipulation (e.g., role elevation) | Token Validation | Critical |

**Repudiation Threats:**

| Threat ID | Threat | Component | Risk Level |
|-----------|--------|-----------|------------|
| R-01 | No audit trail for customer data modifications | Application Logic | High |
| R-02 | Insufficient logging of authentication failures | API Gateway | Medium |
| R-03 | Missing correlation IDs for request tracing | Logging System | Medium |
| R-04 | No retention of deleted customer records for compliance | Database | High |

**Information Disclosure Threats:**

| Threat ID | Threat | Component | Risk Level |
|-----------|--------|-----------|------------|
| I-01 | PII exposure in API error messages | Error Handling | High |
| I-02 | Customer data leakage via overly broad GET responses | API Endpoints | Medium |
| I-03 | Database credentials in Lambda environment variables | Infrastructure | Critical |
| I-04 | CloudWatch logs containing PII | Logging | High |
| I-05 | Enumeration of customer IDs via sequential access | API Design | Medium |

**Denial of Service Threats:**

| Threat ID | Threat | Component | Risk Level |
|-----------|--------|-----------|------------|
| D-01 | API rate limiting bypass | API Gateway | Medium |
| D-02 | Large payload attacks on POST/PUT endpoints | Input Validation | Medium |
| D-03 | Database connection pool exhaustion | Lambda/RDS | High |
| D-04 | Expensive queries via uncontrolled pagination | GET /customers | Medium |

**Elevation of Privilege Threats:**

| Threat ID | Threat | Component | Risk Level |
|-----------|--------|-----------|------------|
| E-01 | Horizontal privilege escalation (user A accessing user B's data) | Authorization | Critical |
| E-02 | Vertical privilege escalation via JWT manipulation | Role-Based Access Control | Critical |
| E-03 | Lambda execution role over-permissions | IAM Policies | High |
| E-04 | Admin endpoint exposure without proper authorization | API Endpoints | High |

#### Attack Tree: Customer Data Breach

```
Goal: Exfiltrate Customer PII Database
├── OR: Exploit API Vulnerabilities
│   ├── AND: Bypass Authentication
│   │   ├── Steal Valid JWT Token (S-01)
│   │   ├── Replay Token Before Expiration (S-02)
│   │   └── Use Token to Access API
│   ├── AND: Exploit Authorization Flaws
│   │   ├── Obtain Low-Privilege Account
│   │   ├── Manipulate Customer ID Parameter (E-01)
│   │   └── Enumerate All Customer Records (I-05)
│   └── AND: SQL Injection Attack
│       ├── Craft Malicious Input (T-02)
│       ├── Bypass Input Validation
│       └── Extract Database Contents
├── OR: Compromise Infrastructure
│   ├── AND: Access Lambda Execution Environment
│   │   ├── Exploit Lambda Function Vulnerability
│   │   ├── Extract Database Credentials (I-03)
│   │   └── Direct Database Access
│   └── AND: Exploit Logging System
│       ├── Gain Access to CloudWatch Logs (I-04)
│       ├── Extract PII from Log Entries
│       └── Aggregate Customer Data
└── OR: Social Engineering
    ├── Phish OAuth Credentials (S-03)
    ├── Impersonate Administrator
    └── Bulk Export Customer Data
```

#### Data Flow Diagram with Threats

```
[Client App] --HTTPS/JWT--> [API Gateway] --Invoke--> [Lambda] --SQL--> [RDS PostgreSQL]
     |                            |                      |                   |
   S-01: XSS                   S-02: Replay          T-02: SQLi          I-03: Creds
   S-04: Key leak              D-01: Rate limit      E-01: Authz
                               I-01: Error msgs       I-04: Logs
                               E-04: Admin access     E-03: IAM perms
```

### Key Observations

As you review the threat model output, look for:

1. **Critical Threats:** Threats marked as "Critical" risk level (S-03, T-02, T-04, I-03, E-01, E-02) require immediate attention.

2. **Common Attack Patterns:**
   - Authentication bypass (S-01, S-02, S-03)
   - Authorization failures (E-01, E-02, E-04)
   - Injection attacks (T-02)
   - Information leakage (I-01, I-03, I-04)

3. **Infrastructure vs. Application:** Note which threats are infrastructure-level (I-03, E-03) versus application-level (T-02, E-01).

4. **Compliance Impact:** Some threats directly impact GDPR/CCPA compliance (R-01, R-04, I-04).

5. **Attack Chains:** The attack tree shows how threats combine - a single vulnerability might not be critical, but chained attacks are dangerous.

### What to Document

Save the following from this step:

- Complete STRIDE threat table (all 6 categories)
- Attack tree diagram
- Data flow diagram with threat annotations
- List of critical threats (for prioritization)

---

## Step 2: Design Security Controls

**Estimated Time:** 15 minutes

### Loading the Skill

Use the security controls design skill:

```
I'm using ordis/security-architect/security-controls-design to create controls for the threats identified in my customer data API threat model.
```

### Input Prompt

Provide the critical threats from Step 1 and ask for mapped controls:

```
Based on my threat model of a customer data REST API, I need to design security controls for these critical threats:

**Critical Threats Requiring Controls:**

1. **S-03:** Impersonation via compromised OAuth credentials (Critical)
2. **T-02:** SQL injection in customer lookup queries (Critical)
3. **T-04:** JWT claims manipulation for privilege escalation (Critical)
4. **I-03:** Database credentials exposed in Lambda environment variables (Critical)
5. **E-01:** Horizontal privilege escalation (user accessing another user's data) (Critical)
6. **E-02:** Vertical privilege escalation via JWT manipulation (Critical)

**High Priority Threats:**

7. **S-01:** JWT token theft via XSS (High)
8. **T-01:** Unauthorized customer data modification (High)
9. **R-01:** No audit trail for data changes (High)
10. **I-01:** PII in error messages (High)
11. **I-04:** PII in CloudWatch logs (High)
12. **D-03:** Database connection pool exhaustion (High)

**Technology Stack:**
- AWS API Gateway, Lambda (Node.js), RDS PostgreSQL
- OAuth 2.0/JWT authentication via Auth0
- VPC with private subnets

Please design specific, implementable security controls for each threat, including:
- Control type (preventive, detective, corrective)
- Implementation details
- Validation/testing approach
- Compliance mapping (GDPR/CCPA where applicable)
```

### Expected Output

The security controls design skill will provide detailed controls mapped to each threat:

#### Control Catalog

**Authentication & Authorization Controls**

**Control AC-01: Multi-Factor Authentication (MFA)**
- **Threat Addressed:** S-03 (Compromised OAuth credentials)
- **Control Type:** Preventive
- **Implementation:**
  - Enable Auth0 MFA for all user accounts
  - Require MFA for administrative accounts (mandatory)
  - Support TOTP (Time-based One-Time Password) and push notifications
  - Enforce MFA re-authentication every 12 hours
- **Configuration:**
  ```javascript
  // Auth0 Rule: Require MFA
  function requireMFA(user, context, callback) {
    if (!context.authentication.methods.find(m => m.name === 'mfa')) {
      return callback(new UnauthorizedError('MFA required'));
    }
    callback(null, user, context);
  }
  ```
- **Testing:** Attempt login without MFA - should be rejected
- **Compliance:** GDPR Article 32 (Security of Processing)

**Control AC-02: JWT Token Security Hardening**
- **Threat Addressed:** S-01 (XSS token theft), S-02 (Token replay), T-04 (Claims manipulation)
- **Control Type:** Preventive
- **Implementation:**
  - Store tokens in httpOnly, secure, SameSite cookies (prevent XSS)
  - Implement token binding via confirmation claims
  - Use short-lived access tokens (15 minutes) with refresh tokens
  - Validate JWT signature using RS256 (asymmetric)
  - Include jti (JWT ID) claim for revocation tracking
- **Configuration:**
  ```javascript
  // Lambda: JWT Validation
  const jwt = require('jsonwebtoken');
  const jwksClient = require('jwks-rsa');

  const client = jwksClient({
    jwksUri: 'https://your-domain.auth0.com/.well-known/jwks.json'
  });

  function getKey(header, callback) {
    client.getSigningKey(header.kid, (err, key) => {
      const signingKey = key.publicKey || key.rsaPublicKey;
      callback(null, signingKey);
    });
  }

  function validateToken(token) {
    return new Promise((resolve, reject) => {
      jwt.verify(token, getKey, {
        algorithms: ['RS256'],
        issuer: 'https://your-domain.auth0.com/',
        audience: 'customer-api'
      }, (err, decoded) => {
        if (err) return reject(err);
        resolve(decoded);
      });
    });
  }
  ```
- **Testing:** Attempt to modify JWT claims - signature verification should fail
- **Compliance:** GDPR Article 32 (Security of Processing)

**Control AC-03: Resource-Level Authorization**
- **Threat Addressed:** E-01 (Horizontal privilege escalation), E-02 (Vertical privilege escalation)
- **Control Type:** Preventive
- **Implementation:**
  - Implement attribute-based access control (ABAC)
  - Validate user ownership of resources before all operations
  - Check JWT sub claim matches customer_id for GET/PUT/DELETE
  - Separate admin and user roles with distinct permissions
  - Deny by default (whitelist approach)
- **Configuration:**
  ```javascript
  // Lambda: Authorization Middleware
  async function authorizeCustomerAccess(userId, customerId, operation) {
    // Verify user owns the customer record
    const customer = await db.query(
      'SELECT user_id FROM customers WHERE id = $1',
      [customerId]
    );

    if (!customer.rows.length) {
      throw new Error('Customer not found');
    }

    if (customer.rows[0].user_id !== userId) {
      throw new Error('Unauthorized: User does not own this customer record');
    }

    // Check operation-specific permissions
    const userRole = await getUserRole(userId);
    if (operation === 'DELETE' && userRole !== 'admin') {
      throw new Error('Unauthorized: Delete requires admin role');
    }

    return true;
  }

  // Example usage in Lambda handler
  exports.handler = async (event) => {
    const token = await validateToken(event.headers.Authorization);
    const userId = token.sub;
    const customerId = event.pathParameters.id;

    await authorizeCustomerAccess(userId, customerId, 'READ');

    // Proceed with operation...
  };
  ```
- **Testing:** User A attempts to access User B's customer record - should return 403 Forbidden
- **Compliance:** GDPR Article 32 (Security of Processing)

**Data Protection Controls**

**Control DP-01: SQL Injection Prevention**
- **Threat Addressed:** T-02 (SQL injection attacks)
- **Control Type:** Preventive
- **Implementation:**
  - Use parameterized queries exclusively (never string concatenation)
  - Implement ORM layer (e.g., Sequelize, TypeORM)
  - Validate and sanitize all user inputs
  - Apply least privilege to database user accounts
  - Enable database query logging for anomaly detection
- **Configuration:**
  ```javascript
  // Lambda: Safe Database Queries
  const { Pool } = require('pg');
  const pool = new Pool({
    host: process.env.DB_HOST,
    database: process.env.DB_NAME,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD,
    ssl: { rejectUnauthorized: true }
  });

  // SECURE: Parameterized query
  async function getCustomer(customerId) {
    const result = await pool.query(
      'SELECT id, first_name, last_name, email FROM customers WHERE id = $1',
      [customerId]
    );
    return result.rows[0];
  }

  // INSECURE: Never do this
  // const query = `SELECT * FROM customers WHERE id = '${customerId}'`;
  ```
- **Testing:** Inject malicious SQL in customer_id parameter - should be treated as literal string
- **Compliance:** GDPR Article 32 (Security of Processing)

**Control DP-02: Secrets Management**
- **Threat Addressed:** I-03 (Database credentials in environment variables)
- **Control Type:** Preventive
- **Implementation:**
  - Store credentials in AWS Secrets Manager
  - Rotate credentials automatically (every 90 days)
  - Use IAM roles for Lambda-to-RDS authentication (preferred)
  - Encrypt environment variables at rest with KMS
  - Never log or expose secrets in error messages
- **Configuration:**
  ```javascript
  // Lambda: Retrieve Secrets from AWS Secrets Manager
  const AWS = require('aws-sdk');
  const secretsManager = new AWS.SecretsManager();

  async function getDatabaseCredentials() {
    const secretName = 'customer-api/database';
    const secret = await secretsManager.getSecretValue({
      SecretId: secretName
    }).promise();

    return JSON.parse(secret.SecretString);
  }

  // Or better: Use IAM Database Authentication
  const signer = new AWS.RDS.Signer({
    region: 'us-east-1',
    hostname: process.env.DB_HOST,
    port: 5432,
    username: 'iam_db_user'
  });

  const token = signer.getAuthToken({
    username: 'iam_db_user'
  });

  // Use token as password for PostgreSQL connection
  ```
- **Testing:** Inspect Lambda environment variables - should not contain plaintext credentials
- **Compliance:** GDPR Article 32 (Security of Processing)

**Control DP-03: Data Minimization in API Responses**
- **Threat Addressed:** I-02 (Overly broad GET responses)
- **Control Type:** Preventive
- **Implementation:**
  - Return only requested fields (support field filtering)
  - Exclude sensitive fields by default (e.g., internal IDs, metadata)
  - Implement response pagination (max 100 records per page)
  - Use DTOs (Data Transfer Objects) to control response shape
- **Configuration:**
  ```javascript
  // Lambda: Response Filtering
  function filterCustomerResponse(customer, requestedFields) {
    const allowedFields = ['id', 'first_name', 'last_name', 'email', 'phone'];
    const fieldsToReturn = requestedFields
      ? requestedFields.filter(f => allowedFields.includes(f))
      : ['id', 'first_name', 'last_name', 'email']; // Default fields

    const filtered = {};
    fieldsToReturn.forEach(field => {
      if (customer[field] !== undefined) {
        filtered[field] = customer[field];
      }
    });

    return filtered;
  }

  // Usage: GET /customers?fields=id,first_name,email
  ```
- **Testing:** Request customer without field filter - verify sensitive fields excluded
- **Compliance:** GDPR Article 5 (Data Minimization)

**Control DP-04: Secure Error Handling**
- **Threat Addressed:** I-01 (PII in error messages)
- **Control Type:** Preventive
- **Implementation:**
  - Return generic error messages to clients
  - Log detailed errors internally (without PII)
  - Implement error codes for client-side handling
  - Sanitize all error responses before returning
- **Configuration:**
  ```javascript
  // Lambda: Secure Error Handler
  function handleError(error, event) {
    // Log detailed error internally (CloudWatch)
    console.error('Error processing request:', {
      requestId: event.requestContext.requestId,
      errorType: error.constructor.name,
      errorMessage: error.message,
      stackTrace: error.stack,
      // Do NOT log PII
    });

    // Return generic error to client
    const statusCode = error.statusCode || 500;
    const errorCode = error.code || 'INTERNAL_ERROR';

    return {
      statusCode,
      body: JSON.stringify({
        error: {
          code: errorCode,
          message: getPublicErrorMessage(errorCode)
        }
      })
    };
  }

  function getPublicErrorMessage(code) {
    const messages = {
      'UNAUTHORIZED': 'Authentication required',
      'FORBIDDEN': 'Access denied',
      'NOT_FOUND': 'Resource not found',
      'VALIDATION_ERROR': 'Invalid request parameters',
      'INTERNAL_ERROR': 'An error occurred processing your request'
    };
    return messages[code] || messages['INTERNAL_ERROR'];
  }
  ```
- **Testing:** Trigger database error - verify response doesn't include SQL details or PII
- **Compliance:** GDPR Article 32 (Security of Processing)

**Logging & Monitoring Controls**

**Control LM-01: Audit Logging**
- **Threat Addressed:** R-01 (No audit trail), R-03 (Missing correlation IDs)
- **Control Type:** Detective
- **Implementation:**
  - Log all customer data access and modifications
  - Include: timestamp, user ID, action, resource ID, result
  - Use structured logging (JSON format)
  - Generate correlation IDs for request tracing
  - Retain audit logs for 7 years (compliance requirement)
- **Configuration:**
  ```javascript
  // Lambda: Audit Logging
  const { v4: uuidv4 } = require('uuid');

  function logAuditEvent(event, userId, action, resourceId, result) {
    const auditLog = {
      timestamp: new Date().toISOString(),
      correlationId: event.requestContext.requestId,
      userId: userId,
      action: action, // CREATE, READ, UPDATE, DELETE
      resourceType: 'customer',
      resourceId: resourceId,
      result: result, // SUCCESS, FAILURE
      ipAddress: event.requestContext.identity.sourceIp,
      userAgent: event.headers['User-Agent'],
      statusCode: result.statusCode
    };

    // Send to CloudWatch Logs or dedicated audit log stream
    console.log('AUDIT:', JSON.stringify(auditLog));

    // Also store in database for long-term retention
    await db.query(
      'INSERT INTO audit_logs (correlation_id, user_id, action, resource_type, resource_id, result, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7)',
      [auditLog.correlationId, auditLog.userId, auditLog.action, auditLog.resourceType, auditLog.resourceId, auditLog.result, auditLog.timestamp]
    );
  }
  ```
- **Testing:** Perform customer update - verify audit log entry created with all required fields
- **Compliance:** GDPR Article 30 (Records of Processing Activities)

**Control LM-02: PII Redaction in Logs**
- **Threat Addressed:** I-04 (PII in CloudWatch logs)
- **Control Type:** Preventive
- **Implementation:**
  - Redact PII before logging (emails, phone numbers, addresses)
  - Use tokenization for identifiers in logs
  - Configure CloudWatch log retention (30 days for application logs)
  - Encrypt logs at rest with KMS
  - Implement log access controls (least privilege)
- **Configuration:**
  ```javascript
  // Lambda: PII Redaction Utility
  function redactPII(data) {
    const redacted = { ...data };

    // Redact email
    if (redacted.email) {
      redacted.email = redacted.email.replace(/(.{2}).*(@.*)/, '$1***$2');
    }

    // Redact phone
    if (redacted.phone) {
      redacted.phone = redacted.phone.replace(/(\d{3}).*(\d{4})/, '$1***$2');
    }

    // Redact address
    if (redacted.address) {
      redacted.address = '[REDACTED]';
    }

    // Keep customer ID for correlation
    return redacted;
  }

  // Usage in logging
  console.log('Customer retrieved:', redactPII(customer));
  // Output: { id: '123', email: 'jo***@example.com', phone: '555***7890' }
  ```
- **Testing:** Trigger error with customer data - verify CloudWatch logs contain redacted PII
- **Compliance:** GDPR Article 32 (Security of Processing)

**Resilience Controls**

**Control RS-01: API Rate Limiting**
- **Threat Addressed:** D-01 (Rate limiting bypass), D-04 (Expensive queries)
- **Control Type:** Preventive
- **Implementation:**
  - Configure AWS API Gateway usage plans
  - Rate limits: 100 requests/second per API key
  - Burst limit: 200 requests
  - Implement per-user rate limiting (10 requests/second)
  - Return 429 Too Many Requests with Retry-After header
- **Configuration:**
  ```yaml
  # API Gateway Usage Plan
  UsagePlan:
    Type: AWS::ApiGateway::UsagePlan
    Properties:
      UsagePlanName: customer-api-standard
      Throttle:
        RateLimit: 100
        BurstLimit: 200
      Quota:
        Limit: 100000
        Period: DAY

  # Lambda: Additional per-user rate limiting
  const rateLimit = require('lambda-rate-limiter')({
    interval: 1000, // 1 second
    uniqueTokenPerInterval: 500
  });

  async function checkRateLimit(userId) {
    try {
      await rateLimit.check(10, userId); // 10 requests per interval
    } catch (error) {
      throw new Error('Rate limit exceeded');
    }
  }
  ```
- **Testing:** Send >100 requests/second - verify 429 responses
- **Compliance:** N/A (Availability control)

**Control RS-02: Input Validation**
- **Threat Addressed:** T-01 (Unauthorized modifications), D-02 (Large payloads)
- **Control Type:** Preventive
- **Implementation:**
  - Validate all input against defined schemas (JSON Schema)
  - Limit request body size (100KB max)
  - Sanitize string inputs (remove special characters)
  - Validate data types, formats, and ranges
  - Reject requests with invalid input (400 Bad Request)
- **Configuration:**
  ```javascript
  // Lambda: Input Validation with JSON Schema
  const Ajv = require('ajv');
  const ajv = new Ajv();

  const customerSchema = {
    type: 'object',
    properties: {
      first_name: { type: 'string', minLength: 1, maxLength: 100 },
      last_name: { type: 'string', minLength: 1, maxLength: 100 },
      email: { type: 'string', format: 'email', maxLength: 255 },
      phone: { type: 'string', pattern: '^\\+?[1-9]\\d{1,14}$' },
      address: {
        type: 'object',
        properties: {
          street: { type: 'string', maxLength: 255 },
          city: { type: 'string', maxLength: 100 },
          state: { type: 'string', maxLength: 50 },
          zip: { type: 'string', pattern: '^\\d{5}(-\\d{4})?$' }
        }
      }
    },
    required: ['first_name', 'last_name', 'email'],
    additionalProperties: false
  };

  const validate = ajv.compile(customerSchema);

  function validateCustomerInput(data) {
    const valid = validate(data);
    if (!valid) {
      throw new ValidationError(validate.errors);
    }
    return true;
  }
  ```
- **Testing:** Send invalid email format - verify 400 Bad Request response
- **Compliance:** GDPR Article 32 (Security of Processing)

**Control RS-03: Database Connection Pooling**
- **Threat Addressed:** D-03 (Connection pool exhaustion)
- **Control Type:** Preventive
- **Implementation:**
  - Configure connection pool with max connections (10 per Lambda)
  - Implement connection timeout (5 seconds)
  - Reuse connections across Lambda invocations
  - Monitor active connections via CloudWatch
  - Implement circuit breaker pattern for database failures
- **Configuration:**
  ```javascript
  // Lambda: Connection Pool Configuration
  const { Pool } = require('pg');

  // Initialize pool outside handler for reuse
  const pool = new Pool({
    host: process.env.DB_HOST,
    database: process.env.DB_NAME,
    max: 10, // Max connections per Lambda instance
    idleTimeoutMillis: 30000,
    connectionTimeoutMillis: 5000,
    maxUses: 7500 // Rotate connections
  });

  pool.on('error', (err) => {
    console.error('Unexpected database pool error', err);
  });

  // Lambda handler
  exports.handler = async (event) => {
    let client;
    try {
      client = await pool.connect();
      const result = await client.query('SELECT NOW()');
      return result.rows[0];
    } finally {
      if (client) client.release();
    }
  };
  ```
- **Testing:** Simulate high concurrent Lambda invocations - monitor connection pool metrics
- **Compliance:** N/A (Availability control)

**Infrastructure Controls**

**Control IN-01: Least Privilege IAM Policies**
- **Threat Addressed:** E-03 (Lambda over-permissions)
- **Control Type:** Preventive
- **Implementation:**
  - Grant minimum permissions required for Lambda execution
  - Separate IAM roles per Lambda function
  - Use resource-level permissions (specific RDS instance)
  - Enable IAM Access Analyzer to detect over-permissions
  - Review and audit IAM policies quarterly
- **Configuration:**
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "rds-db:connect"
        ],
        "Resource": "arn:aws:rds-db:us-east-1:123456789012:dbuser:db-ABCDEFGHIJKLMNOPQRST/iam_db_user"
      },
      {
        "Effect": "Allow",
        "Action": [
          "secretsmanager:GetSecretValue"
        ],
        "Resource": "arn:aws:secretsmanager:us-east-1:123456789012:secret:customer-api/database-AbCdEf"
      },
      {
        "Effect": "Allow",
        "Action": [
          "kms:Decrypt"
        ],
        "Resource": "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012",
        "Condition": {
          "StringEquals": {
            "kms:ViaService": "secretsmanager.us-east-1.amazonaws.com"
          }
        }
      },
      {
        "Effect": "Allow",
        "Action": [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        "Resource": "arn:aws:logs:us-east-1:123456789012:log-group:/aws/lambda/customer-api-*"
      }
    ]
  }
  ```
- **Testing:** Attempt to invoke Lambda with overly broad permissions - should fail AWS security scans
- **Compliance:** GDPR Article 32 (Security of Processing)

#### Control Summary Table

| Control ID | Control Name | Threat(s) Addressed | Type | Priority |
|------------|--------------|---------------------|------|----------|
| AC-01 | Multi-Factor Authentication | S-03 | Preventive | Critical |
| AC-02 | JWT Token Security Hardening | S-01, S-02, T-04 | Preventive | Critical |
| AC-03 | Resource-Level Authorization | E-01, E-02 | Preventive | Critical |
| DP-01 | SQL Injection Prevention | T-02 | Preventive | Critical |
| DP-02 | Secrets Management | I-03 | Preventive | Critical |
| DP-03 | Data Minimization in API Responses | I-02 | Preventive | High |
| DP-04 | Secure Error Handling | I-01 | Preventive | High |
| LM-01 | Audit Logging | R-01, R-03 | Detective | High |
| LM-02 | PII Redaction in Logs | I-04 | Preventive | High |
| RS-01 | API Rate Limiting | D-01, D-04 | Preventive | Medium |
| RS-02 | Input Validation | T-01, D-02 | Preventive | High |
| RS-03 | Database Connection Pooling | D-03 | Preventive | Medium |
| IN-01 | Least Privilege IAM Policies | E-03 | Preventive | High |

### Key Observations

As you review the security controls output, note:

1. **Defense in Depth:** Multiple controls address overlapping threats (e.g., AC-02 and AC-03 both prevent privilege escalation).

2. **Preventive vs. Detective:** Most controls are preventive (stop attacks before they happen), with LM-01 being detective (detect attacks after they occur).

3. **Implementation Specificity:** Controls include actual code samples, not just high-level descriptions.

4. **Compliance Mapping:** Each control is mapped to specific GDPR articles for traceability.

5. **Priority Alignment:** Critical controls address critical threats from the threat model.

6. **Technology-Specific:** Controls are tailored to AWS Lambda/API Gateway/RDS stack.

### What to Document

Save the following from this step:

- Complete control catalog (all 13 controls)
- Control-to-threat mapping table
- Implementation code snippets for each control
- Testing procedures for each control

---

## Step 3: Document Threats and Controls

**Estimated Time:** 10 minutes

### Loading the Skill

Use the documenting threats and controls skill:

```
I'm using ordis/security-architect/documenting-threats-and-controls to create comprehensive security documentation for my customer data API.
```

### Input Prompt

Request documentation that traces threats to controls:

```
I need to create complete security documentation for my customer data REST API that provides full traceability between threats, controls, and compliance requirements.

**Documentation Requirements:**

1. **Threat Register:** Complete catalog of all threats from STRIDE analysis
2. **Control Register:** Detailed description of all implemented controls
3. **Traceability Matrix:** Map threats → controls → compliance requirements
4. **Risk Assessment:** Residual risk after control implementation
5. **Compliance Evidence:** Mapping to GDPR/CCPA articles

**Threats Summary:**
- 24 total threats identified (6 Critical, 6 High, 12 Medium)
- Categories: Spoofing (4), Tampering (4), Repudiation (4), Information Disclosure (5), Denial of Service (4), Elevation of Privilege (4)

**Controls Summary:**
- 13 security controls designed (5 Critical, 6 High, 2 Medium priority)
- Types: 11 Preventive, 1 Detective, 0 Corrective, 1 Compensating

**Compliance Requirements:**
- GDPR: Articles 5, 30, 32
- CCPA: Data protection and breach notification

Please create documentation that will support:
- Security audits and assessments
- Compliance reviews (GDPR/CCPA)
- Incident response planning
- Developer onboarding
```

### Expected Output

The documenting skill will generate comprehensive security documentation:

#### Complete Threat Register

**Threat Register: Customer Data REST API**

| ID | Threat | Category | Component | Likelihood | Impact | Risk Score | Status |
|----|--------|----------|-----------|------------|--------|------------|--------|
| S-01 | JWT token theft via XSS in client application | Spoofing | Authentication | Medium | High | 8/10 | Mitigated |
| S-02 | Token replay attack using stolen valid JWT | Spoofing | API Gateway | Medium | High | 8/10 | Mitigated |
| S-03 | Impersonation via compromised OAuth credentials | Spoofing | Auth0 | Low | Critical | 9/10 | Mitigated |
| S-04 | API key leakage in client-side code | Spoofing | Client Apps | Medium | Medium | 6/10 | Accepted |
| T-01 | Unauthorized modification of customer data via PUT | Tampering | Lambda | Medium | High | 8/10 | Mitigated |
| T-02 | SQL injection in customer lookup queries | Tampering | Database | Low | Critical | 9/10 | Mitigated |
| T-03 | Request parameter manipulation to access other customers | Tampering | API Endpoints | Medium | High | 8/10 | Mitigated |
| T-04 | JWT claims manipulation for privilege escalation | Tampering | Token Validation | Low | Critical | 9/10 | Mitigated |
| R-01 | No audit trail for customer data modifications | Repudiation | Application | High | High | 9/10 | Mitigated |
| R-02 | Insufficient logging of authentication failures | Repudiation | API Gateway | Medium | Medium | 6/10 | Accepted |
| R-03 | Missing correlation IDs for request tracing | Repudiation | Logging | Medium | Medium | 6/10 | Mitigated |
| R-04 | No retention of deleted customer records | Repudiation | Database | Medium | High | 8/10 | Transferred |
| I-01 | PII exposure in API error messages | Information Disclosure | Error Handling | Medium | High | 8/10 | Mitigated |
| I-02 | Customer data leakage via overly broad GET responses | Information Disclosure | API Endpoints | High | Medium | 7/10 | Mitigated |
| I-03 | Database credentials in Lambda environment variables | Information Disclosure | Infrastructure | Low | Critical | 9/10 | Mitigated |
| I-04 | CloudWatch logs containing PII | Information Disclosure | Logging | High | High | 9/10 | Mitigated |
| I-05 | Enumeration of customer IDs via sequential access | Information Disclosure | API Design | High | Medium | 7/10 | Accepted |
| D-01 | API rate limiting bypass | Denial of Service | API Gateway | Medium | Medium | 6/10 | Mitigated |
| D-02 | Large payload attacks on POST/PUT endpoints | Denial of Service | Input Validation | Low | Medium | 5/10 | Mitigated |
| D-03 | Database connection pool exhaustion | Denial of Service | Lambda/RDS | Medium | High | 8/10 | Mitigated |
| D-04 | Expensive queries via uncontrolled pagination | Denial of Service | GET /customers | Medium | Medium | 6/10 | Mitigated |
| E-01 | Horizontal privilege escalation | Elevation of Privilege | Authorization | Medium | Critical | 9/10 | Mitigated |
| E-02 | Vertical privilege escalation via JWT manipulation | Elevation of Privilege | RBAC | Low | Critical | 9/10 | Mitigated |
| E-03 | Lambda execution role over-permissions | Elevation of Privilege | IAM | Medium | High | 8/10 | Mitigated |
| E-04 | Admin endpoint exposure without authorization | Elevation of Privilege | API Endpoints | Low | High | 7/10 | Accepted |

**Risk Scoring:**
- **Likelihood:** Low (1-3), Medium (4-6), High (7-9)
- **Impact:** Low (1-3), Medium (4-6), High (7-8), Critical (9-10)
- **Risk Score:** Likelihood × Impact / 10 (rounded)

**Risk Status:**
- **Mitigated:** Control implemented to reduce risk to acceptable level
- **Accepted:** Risk accepted by business (documented in risk acceptance register)
- **Transferred:** Risk transferred to third party (e.g., backup provider for R-04)

#### Complete Control Register

**Control Register: Customer Data REST API**

**Control AC-01: Multi-Factor Authentication (MFA)**

- **Control ID:** AC-01
- **Control Family:** Access Control
- **Control Type:** Preventive
- **Implementation Status:** Implemented
- **Owner:** Security Team
- **Last Review:** 2025-10-29

**Description:**
Enforce multi-factor authentication (MFA) for all user accounts accessing the customer data API. MFA requires users to provide two or more verification factors to gain access, significantly reducing the risk of account compromise.

**Implementation Details:**
- MFA provider: Auth0 with TOTP and push notification support
- Enforcement: Mandatory for all administrative accounts, optional for standard users
- Re-authentication interval: Every 12 hours
- Supported factors: SMS, authenticator app (Google Authenticator, Authy), push notifications

**Threats Mitigated:**
- S-03: Impersonation via compromised OAuth credentials (Critical)

**Testing/Validation:**
1. Attempt login with valid credentials but without MFA - expect rejection
2. Attempt login with valid credentials and MFA - expect success
3. Attempt login with expired MFA token - expect rejection
4. Verify MFA re-authentication after 12 hours

**Compliance Mapping:**
- GDPR Article 32(1)(b): Ability to ensure ongoing confidentiality of processing systems
- NIST SP 800-53: IA-2 (Identification and Authentication)

**Monitoring:**
- CloudWatch metric: MFA enrollment rate (target: >95%)
- CloudWatch alarm: Spike in MFA failures (potential attack)

**Residual Risk:**
- MFA bypass via SIM swap attack (Low likelihood)
- Social engineering of MFA codes (Low likelihood)

---

**Control AC-02: JWT Token Security Hardening**

- **Control ID:** AC-02
- **Control Family:** Access Control
- **Control Type:** Preventive
- **Implementation Status:** Implemented
- **Owner:** Development Team
- **Last Review:** 2025-10-29

**Description:**
Implement comprehensive JWT token security measures to prevent token theft, replay, and manipulation attacks. This includes secure storage, short token lifetimes, cryptographic validation, and token binding.

**Implementation Details:**
- Token storage: httpOnly, secure, SameSite cookies (prevent XSS extraction)
- Access token lifetime: 15 minutes (short-lived)
- Refresh token lifetime: 7 days with rotation on use
- Signature algorithm: RS256 (asymmetric, prevents tampering)
- Token binding: Confirmation claims tied to client
- Revocation: JWT ID (jti) claim tracked in Redis for revocation

**Threats Mitigated:**
- S-01: JWT token theft via XSS (High)
- S-02: Token replay attack (High)
- T-04: JWT claims manipulation (Critical)

**Testing/Validation:**
1. Verify tokens stored in httpOnly cookies (not accessible via JavaScript)
2. Attempt to modify JWT claims - signature verification should fail
3. Attempt to use token after 15 minutes - should be expired
4. Verify token revocation works (jti blacklist check)

**Compliance Mapping:**
- GDPR Article 32(1)(b): Ability to ensure ongoing confidentiality
- OWASP ASVS 2.0: Section 3.5 (Token-Based Session Management)

**Monitoring:**
- CloudWatch metric: Token validation failures (detect manipulation attempts)
- CloudWatch alarm: High rate of expired token usage (potential replay attack)

**Residual Risk:**
- Token theft via compromised HTTPS/TLS (Low likelihood with TLS 1.3)
- Token leakage via compromised browser (Low likelihood, mitigated by short lifetime)

---

**Control AC-03: Resource-Level Authorization**

- **Control ID:** AC-03
- **Control Family:** Access Control
- **Control Type:** Preventive
- **Implementation Status:** Implemented
- **Owner:** Development Team
- **Last Review:** 2025-10-29

**Description:**
Implement attribute-based access control (ABAC) to ensure users can only access customer records they own. This prevents horizontal privilege escalation (user A accessing user B's data) and vertical privilege escalation (user gaining admin privileges).

**Implementation Details:**
- Authorization model: Attribute-Based Access Control (ABAC)
- User-resource ownership validation before all GET/PUT/DELETE operations
- JWT sub claim matched against customer.user_id in database
- Role-based permissions for administrative operations (DELETE requires admin role)
- Default deny policy (whitelist approach)

**Threats Mitigated:**
- E-01: Horizontal privilege escalation (Critical)
- E-02: Vertical privilege escalation (Critical)
- T-03: Request parameter manipulation (High)

**Testing/Validation:**
1. User A attempts GET /customers/{B-id} - expect 403 Forbidden
2. User A attempts PUT /customers/{B-id} - expect 403 Forbidden
3. Non-admin user attempts DELETE /customers/{own-id} - expect 403 Forbidden
4. Admin user attempts DELETE /customers/{any-id} - expect 200 OK

**Compliance Mapping:**
- GDPR Article 32(1)(b): Ability to ensure ongoing confidentiality
- NIST SP 800-53: AC-3 (Access Enforcement)

**Monitoring:**
- CloudWatch metric: Authorization failures (403 responses)
- CloudWatch alarm: Spike in authorization failures from single user (potential attack)

**Residual Risk:**
- Insider threat (authorized user abusing legitimate access) - mitigated by audit logging

---

*(Continues for all 13 controls...)*

#### Threat-Control Traceability Matrix

**Traceability Matrix: Threats → Controls → Compliance**

| Threat ID | Threat | Risk | Control(s) | Control Type | GDPR Article | CCPA Section |
|-----------|--------|------|------------|--------------|--------------|--------------|
| S-01 | JWT token theft via XSS | High | AC-02 | Preventive | 32(1)(b) | 1798.150(a)(1) |
| S-02 | Token replay attack | High | AC-02 | Preventive | 32(1)(b) | 1798.150(a)(1) |
| S-03 | OAuth credential compromise | Critical | AC-01 | Preventive | 32(1)(b) | 1798.150(a)(1) |
| S-04 | API key leakage | Medium | (Accepted) | N/A | N/A | N/A |
| T-01 | Unauthorized data modification | High | AC-03, RS-02 | Preventive | 32(1)(b) | 1798.150(a)(1) |
| T-02 | SQL injection | Critical | DP-01 | Preventive | 32(1)(b) | 1798.150(a)(1) |
| T-03 | Parameter manipulation | High | AC-03 | Preventive | 32(1)(b) | 1798.150(a)(1) |
| T-04 | JWT claims manipulation | Critical | AC-02 | Preventive | 32(1)(b) | 1798.150(a)(1) |
| R-01 | No audit trail | High | LM-01 | Detective | 30(1), 32(1)(d) | 1798.150(a)(1) |
| R-02 | Insufficient auth logging | Medium | (Accepted) | N/A | N/A | N/A |
| R-03 | Missing correlation IDs | Medium | LM-01 | Detective | 30(1) | N/A |
| R-04 | No retention of deleted records | High | (Transferred) | N/A | 17 (Right to erasure) | 1798.105 |
| I-01 | PII in error messages | High | DP-04 | Preventive | 32(1)(b) | 1798.150(a)(1) |
| I-02 | Overly broad GET responses | Medium | DP-03 | Preventive | 5(1)(c) | 1798.100(a)(4) |
| I-03 | Database credentials exposed | Critical | DP-02 | Preventive | 32(1)(b) | 1798.150(a)(1) |
| I-04 | PII in CloudWatch logs | High | LM-02 | Preventive | 32(1)(b) | 1798.150(a)(1) |
| I-05 | Customer ID enumeration | Medium | (Accepted) | N/A | N/A | N/A |
| D-01 | Rate limiting bypass | Medium | RS-01 | Preventive | 32(1)(b) | N/A |
| D-02 | Large payload attacks | Low | RS-02 | Preventive | 32(1)(b) | N/A |
| D-03 | Connection pool exhaustion | High | RS-03 | Preventive | 32(1)(b) | N/A |
| D-04 | Expensive queries | Medium | RS-01 | Preventive | 32(1)(b) | N/A |
| E-01 | Horizontal privilege escalation | Critical | AC-03 | Preventive | 32(1)(b) | 1798.150(a)(1) |
| E-02 | Vertical privilege escalation | Critical | AC-03 | Preventive | 32(1)(b) | 1798.150(a)(1) |
| E-03 | Lambda over-permissions | High | IN-01 | Preventive | 32(1)(b) | 1798.150(a)(1) |
| E-04 | Admin endpoint exposure | High | (Accepted) | N/A | N/A | N/A |

**Coverage Analysis:**
- **Total Threats:** 24
- **Threats Mitigated:** 19 (79%)
- **Threats Accepted:** 4 (17%)
- **Threats Transferred:** 1 (4%)
- **Critical Threats Mitigated:** 6/6 (100%)
- **High Threats Mitigated:** 5/6 (83%)

**Compliance Coverage:**
- **GDPR Article 5 (Data Minimization):** 1 control (DP-03)
- **GDPR Article 17 (Right to Erasure):** Risk transferred (R-04)
- **GDPR Article 30 (Records of Processing):** 1 control (LM-01)
- **GDPR Article 32 (Security of Processing):** 11 controls
- **CCPA Section 1798.100 (Consumer Rights):** 1 control (DP-03)
- **CCPA Section 1798.105 (Deletion Rights):** Risk transferred (R-04)
- **CCPA Section 1798.150 (Data Security):** 11 controls

#### Risk Assessment: Before and After Controls

**Risk Heat Map: Before Controls**

| | **Low Impact** | **Medium Impact** | **High Impact** | **Critical Impact** |
|---------------|----------------|-------------------|-----------------|---------------------|
| **High Likelihood** | | I-02, I-05 | I-04, R-01 | |
| **Medium Likelihood** | | S-04, R-02, R-03, D-01, D-04 | S-01, S-02, T-01, T-03, I-01, D-03, E-03 | E-01 |
| **Low Likelihood** | D-02 | | E-04 | S-03, T-02, T-04, I-03, E-02 |

**Total Initial Risk Score:** 172/240 (High)

**Risk Heat Map: After Controls**

| | **Low Impact** | **Medium Impact** | **High Impact** | **Critical Impact** |
|---------------|----------------|-------------------|-----------------|---------------------|
| **High Likelihood** | | I-05 (Accepted) | | |
| **Medium Likelihood** | S-04 (Accepted), R-02 (Accepted) | | | |
| **Low Likelihood** | All mitigated threats | E-04 (Accepted) | | R-04 (Transferred) |

**Total Residual Risk Score:** 28/240 (Low)

**Risk Reduction:** 84% reduction in overall risk

**Accepted Risks (Documented):**
- **S-04:** API key leakage in client-side code - Mitigated by key rotation and monitoring
- **R-02:** Insufficient auth failure logging - API Gateway provides basic logging, full logging is expensive
- **I-05:** Customer ID enumeration - Using UUIDs would break existing clients, monitoring for abuse instead
- **E-04:** Admin endpoint exposure - Admin endpoints require high-privilege JWT, risk acceptable

**Transferred Risks:**
- **R-04:** Deleted customer record retention - Using third-party backup service (AWS Backup) with 7-year retention

#### Compliance Evidence Package

**GDPR Compliance Evidence**

**Article 5(1)(c) - Data Minimization**

*Requirement:* Personal data shall be adequate, relevant and limited to what is necessary.

*Evidence:*
- Control DP-03 (Data Minimization in API Responses) implemented
- API returns only requested fields via field filtering
- Default response excludes sensitive metadata
- Pagination limits response size (max 100 records)

**Article 30(1) - Records of Processing Activities**

*Requirement:* Each controller shall maintain a record of processing activities.

*Evidence:*
- Control LM-01 (Audit Logging) implemented
- All customer data access/modification logged with: timestamp, user ID, action, resource ID, result
- Audit logs retained for 7 years
- Correlation IDs enable request tracing

**Article 32(1)(b) - Security of Processing**

*Requirement:* Ability to ensure the ongoing confidentiality, integrity, availability and resilience of processing systems.

*Evidence:*
- 11 controls implemented to ensure confidentiality:
  - AC-01: Multi-Factor Authentication
  - AC-02: JWT Token Security
  - AC-03: Resource-Level Authorization
  - DP-01: SQL Injection Prevention
  - DP-02: Secrets Management
  - DP-04: Secure Error Handling
  - LM-02: PII Redaction in Logs
  - RS-01: API Rate Limiting
  - RS-02: Input Validation
  - RS-03: Database Connection Pooling
  - IN-01: Least Privilege IAM

**CCPA Compliance Evidence**

**Section 1798.100(a)(4) - Right to Know**

*Requirement:* Consumers have the right to request specific pieces of personal information collected.

*Evidence:*
- Control DP-03 enables consumers to request specific data fields
- GET /customers/{id}?fields=email,phone returns only requested PII
- API supports data portability in JSON format

**Section 1798.150(a)(1) - Data Security**

*Requirement:* Implement reasonable security procedures to protect personal information.

*Evidence:*
- 11 preventive security controls implemented (see GDPR Article 32 evidence)
- Regular security audits conducted
- Incident response plan documented
- Security controls tested and validated

### Key Observations

As you review the documentation output, note:

1. **Complete Traceability:** Every threat is mapped to specific controls and compliance requirements.

2. **Risk Quantification:** Before/after risk assessment shows measurable security improvement (84% reduction).

3. **Compliance Ready:** Documentation directly supports GDPR and CCPA compliance audits.

4. **Accepted Risks Documented:** Risks not mitigated are documented with justification.

5. **Evidence-Based:** Compliance evidence cites specific controls with implementation details.

6. **Audit Trail:** Documentation suitable for security audits, penetration test scoping, and compliance reviews.

### What to Document

Save the following from this step:

- Complete threat register (24 threats)
- Complete control register (13 controls)
- Threat-control-compliance traceability matrix
- Risk assessment (before/after heat maps)
- Compliance evidence package (GDPR/CCPA)

---

## Step 4: Review Architecture

**Estimated Time:** 10-15 minutes

### Loading the Skill

Use the security architecture review skill:

```
I'm using ordis/security-architect/security-architecture-review to perform a final assessment of my customer data API security posture.
```

### Input Prompt

Request a comprehensive architecture review:

```
I've completed threat modeling, control design, and documentation for my customer data REST API. I need a final security architecture review to identify any gaps, weaknesses, or areas for improvement.

**System Summary:**
- REST API for customer PII (names, emails, phone numbers, addresses)
- AWS API Gateway + Lambda (Node.js) + RDS PostgreSQL
- OAuth 2.0/JWT authentication via Auth0
- 24 threats identified, 19 mitigated, 4 accepted, 1 transferred
- 13 security controls implemented (5 Critical, 6 High, 2 Medium)
- GDPR and CCPA compliance required

**Key Security Controls Implemented:**
1. Multi-Factor Authentication (Auth0)
2. JWT token security (httpOnly cookies, 15-min expiration, RS256)
3. Resource-level authorization (ABAC)
4. SQL injection prevention (parameterized queries)
5. Secrets management (AWS Secrets Manager)
6. Audit logging (7-year retention)
7. PII redaction in logs
8. API rate limiting (100 req/sec)
9. Input validation (JSON Schema)
10. Database connection pooling
11. Least privilege IAM policies

**Review Focus Areas:**
1. **Architecture Patterns:** Are security patterns (defense in depth, least privilege, fail secure) properly applied?
2. **Coverage Gaps:** Any threats or attack vectors not addressed?
3. **Implementation Risks:** Potential issues in control implementation?
4. **Compliance Alignment:** Does architecture meet GDPR/CCPA requirements?
5. **Resilience:** Can the system withstand attacks and recover?
6. **Operational Security:** Are monitoring, incident response, and maintenance considered?

Please provide:
- Overall security posture assessment (score/rating)
- Identified gaps or weaknesses
- Recommendations for improvement (prioritized)
- Compliance validation
```

### Expected Output

The architecture review skill will provide a comprehensive assessment:

#### Security Posture Assessment

**Overall Security Rating: 8.5/10 (Strong)**

**Assessment Summary:**

The customer data REST API demonstrates a mature security posture with comprehensive threat modeling, well-designed controls, and strong compliance alignment. The architecture implements defense-in-depth principles across multiple layers (authentication, authorization, data protection, infrastructure) and addresses all critical threats.

**Strengths:**
- Comprehensive STRIDE threat analysis with 24 identified threats
- All critical threats (6) mitigated with appropriate controls
- Strong authentication (MFA + JWT hardening)
- Robust authorization (resource-level ABAC)
- Data protection controls (encryption, redaction, minimization)
- Compliance-ready documentation (GDPR/CCPA traceability)
- Infrastructure security (secrets management, least privilege IAM)

**Areas for Improvement:**
- Monitoring and alerting could be enhanced
- Incident response procedures not fully documented
- Some accepted risks may warrant reconsideration
- Disaster recovery and business continuity planning needed
- Third-party dependency security assessment required

---

#### Architecture Pattern Analysis

**Defense in Depth: ✓ Strong**

Multiple layers of security controls protect customer data:

```
Layer 1: Network Security
├── HTTPS/TLS 1.3 (encryption in transit)
├── AWS VPC with private subnets (network isolation)
└── API Gateway WAF rules (traffic filtering)

Layer 2: Authentication & Authorization
├── OAuth 2.0 with MFA (strong authentication)
├── JWT with RS256 signature (tamper-proof tokens)
├── Resource-level authorization (ABAC)
└── Short-lived tokens (15-min expiration)

Layer 3: Application Security
├── Input validation (JSON Schema)
├── SQL injection prevention (parameterized queries)
├── Secure error handling (no PII leakage)
└── Rate limiting (DoS protection)

Layer 4: Data Security
├── Secrets management (AWS Secrets Manager)
├── PII redaction in logs
├── Data minimization (field filtering)
└── Encryption at rest (RDS encryption)

Layer 5: Monitoring & Response
├── Audit logging (all data access)
├── CloudWatch monitoring (anomaly detection)
└── Correlation IDs (request tracing)
```

**Verdict:** Strong defense-in-depth implementation. If one layer fails (e.g., JWT stolen), other layers (MFA, authorization) provide protection.

**Recommendation:** Add API Gateway WAF with OWASP Top 10 rule set for additional Layer 1 protection.

---

**Least Privilege: ✓ Good**

Principle applied across multiple dimensions:

- **IAM Policies:** Lambda execution role has minimal permissions (IN-01)
- **Database Access:** Lambda uses connection pooling with limited queries
- **API Authorization:** Users can only access their own customer records (AC-03)
- **Token Permissions:** JWT scopes limit API operations

**Gap Identified:** Database user account permissions not documented.

**Recommendation:**
- Create separate database users for read-only vs. read-write operations
- Lambda functions performing GET operations should use read-only database user
- Document database user permissions in control register

---

**Fail Secure: ✓ Good**

Error conditions default to secure state:

- **Authorization Failures:** Deny by default (whitelist approach in AC-03)
- **Token Validation Errors:** Reject request if JWT validation fails
- **Database Errors:** Return generic error, don't expose details (DP-04)
- **Rate Limiting:** Block requests when limit exceeded

**Gap Identified:** Lambda function error handling not fully specified.

**Recommendation:**
- Document Lambda error handling strategy (timeout, out-of-memory, uncaught exceptions)
- Ensure errors don't bypass authorization checks
- Test failure scenarios (database unreachable, Auth0 down)

---

**Zero Trust: ⚠ Partial**

Zero trust principles partially implemented:

- **✓ Verify Explicitly:** All requests authenticated (OAuth 2.0/JWT)
- **✓ Least Privilege Access:** Resource-level authorization (AC-03)
- **⚠ Assume Breach:** Limited containment measures

**Gap Identified:**
- No network segmentation beyond VPC (Lambda can access all RDS instances in VPC)
- No egress filtering (Lambda can make outbound requests to any endpoint)
- Limited anomaly detection (no baseline behavior modeling)

**Recommendation:**
- Implement Lambda-level network controls (VPC endpoint policies)
- Use AWS Security Groups to restrict RDS access to specific Lambda functions
- Add anomaly detection for unusual access patterns (e.g., user accessing 1000s of customer records)

---

#### Identified Gaps and Weaknesses

**Gap 1: Monitoring and Alerting (Medium Priority)**

**Issue:**
While audit logging (LM-01) is implemented, there is no comprehensive monitoring and alerting strategy for security events.

**Specific Concerns:**
- No real-time alerts for suspicious activity (e.g., repeated authorization failures)
- No dashboards for security metrics (failed logins, authorization failures, rate limit hits)
- No integration with SIEM (Security Information and Event Management) system
- Limited anomaly detection (e.g., user accessing unusual number of records)

**Recommendation:**
- Create CloudWatch Alarms for security events:
  - Alert on >10 authorization failures from same user in 5 minutes
  - Alert on >50 JWT validation failures per minute
  - Alert on >1000 requests from single IP in 1 minute
- Build CloudWatch Dashboard for security metrics
- Consider integration with AWS GuardDuty for threat detection
- Implement automated response (e.g., temporary account lock after failures)

**Implementation Estimate:** 4-8 hours

---

**Gap 2: Incident Response Procedures (High Priority)**

**Issue:**
Security controls are preventive and detective, but no corrective controls or incident response procedures are documented.

**Specific Concerns:**
- No incident response plan for security events (e.g., compromised JWT, SQL injection attempt)
- No playbooks for common scenarios (account compromise, data breach, DDoS)
- No defined roles and responsibilities during incidents
- No communication plan for breach notification (GDPR/CCPA requirement)

**Recommendation:**
- Create incident response plan with:
  - Incident classification (P0-P4 severity levels)
  - Response procedures for common scenarios
  - Contact information for incident response team
  - Escalation paths and decision trees
- Develop runbooks for:
  - Compromised user account (revoke tokens, reset credentials)
  - Suspected data breach (isolate systems, preserve evidence, notify stakeholders)
  - API availability issues (check Lambda, RDS, API Gateway health)
- Document GDPR breach notification process (72-hour timeline)
- Conduct tabletop exercises to test procedures

**Implementation Estimate:** 8-16 hours

---

**Gap 3: Accepted Risk Re-evaluation (Low Priority)**

**Issue:**
Four threats were accepted without mitigation controls. Some may warrant reconsideration.

**Specific Concerns:**

**S-04 (API key leakage in client-side code):**
- **Current Status:** Accepted
- **Rationale:** Mitigated by key rotation and monitoring
- **Concern:** If API keys are exposed in mobile apps or JavaScript, rotation is insufficient (keys already public)
- **Recommendation:** Consider removing API keys entirely and using OAuth 2.0 PKCE flow for public clients

**I-05 (Customer ID enumeration via sequential access):**
- **Current Status:** Accepted
- **Rationale:** UUIDs would break existing clients, monitoring for abuse
- **Concern:** Enumeration enables attackers to discover all customer IDs, facilitating other attacks
- **Recommendation:** Migrate to UUIDs for new customer records, monitor enumeration attempts, implement rate limiting per customer ID

**E-04 (Admin endpoint exposure without authorization):**
- **Current Status:** Accepted
- **Rationale:** Admin endpoints require high-privilege JWT
- **Concern:** If JWT is compromised (e.g., S-01, S-02), admin endpoints are exposed
- **Recommendation:** Add IP whitelisting for admin endpoints, require re-authentication for sensitive operations

---

**Gap 4: Third-Party Dependency Security (Medium Priority)**

**Issue:**
The architecture relies on several third-party services (Auth0, AWS services), but their security is not assessed.

**Specific Concerns:**
- Auth0 security configuration not documented (password policies, session management, etc.)
- AWS service configurations not hardened (API Gateway WAF not mentioned, RDS backup retention unknown)
- Node.js dependencies in Lambda functions not scanned for vulnerabilities
- No supply chain security controls (e.g., dependency pinning, vulnerability scanning)

**Recommendation:**
- Document Auth0 security configuration:
  - Password policy (min length, complexity, breach detection)
  - Session timeout and re-authentication requirements
  - Account lockout policies
- Harden AWS services:
  - Enable API Gateway WAF with OWASP Top 10 rules
  - Configure RDS automated backups (7-day retention minimum)
  - Enable RDS encryption at rest (if not already enabled)
  - Enable AWS CloudTrail for audit logging of infrastructure changes
- Implement dependency scanning:
  - Use npm audit or Snyk to scan Node.js dependencies
  - Automate scanning in CI/CD pipeline
  - Pin dependency versions in package-lock.json

**Implementation Estimate:** 8-12 hours

---

**Gap 5: Disaster Recovery and Business Continuity (Medium Priority)**

**Issue:**
No disaster recovery (DR) or business continuity (BC) planning is documented.

**Specific Concerns:**
- No RTO (Recovery Time Objective) or RPO (Recovery Point Objective) defined
- No backup and restore procedures documented
- No multi-region failover strategy
- No testing of recovery procedures

**Recommendation:**
- Define RTO/RPO targets (e.g., RTO: 4 hours, RPO: 1 hour)
- Document backup strategy:
  - RDS automated backups (point-in-time recovery)
  - Database snapshots (weekly)
  - Audit log backups (S3 with versioning)
- Create disaster recovery runbook:
  - Restore RDS from snapshot
  - Redeploy Lambda functions in alternate region
  - Update API Gateway endpoint
- Test DR procedures quarterly

**Implementation Estimate:** 12-16 hours

---

#### Compliance Validation

**GDPR Compliance: ✓ Strong**

**Article 5 (Data Processing Principles):**
- **✓ Lawfulness, Fairness, Transparency:** OAuth consent flow ensures transparency
- **✓ Purpose Limitation:** Customer data used only for specified purposes
- **✓ Data Minimization:** Control DP-03 implements field filtering
- **✓ Accuracy:** PUT endpoint allows customers to update information
- **✓ Storage Limitation:** (Not addressed - retention policy needed)
- **✓ Integrity and Confidentiality:** 11 security controls ensure data protection

**Gap:** Data retention policy not defined.

**Recommendation:** Define and implement data retention policy (e.g., customer data deleted 90 days after account closure).

**Article 30 (Records of Processing Activities):**
- **✓ Implemented:** Control LM-01 provides comprehensive audit trail
- **✓ Retention:** 7-year retention meets regulatory requirements

**Article 32 (Security of Processing):**
- **✓ Strong implementation:** 11 controls address confidentiality, integrity, availability
- **✓ Ongoing monitoring:** CloudWatch logging enables security monitoring
- **✓ Regular testing:** Validation procedures documented for each control

**Article 33 (Breach Notification):**
- **⚠ Partially addressed:** Audit logging enables breach detection
- **Gap:** 72-hour notification process not documented

**Recommendation:** Document breach notification procedure (see Gap 2).

**Overall GDPR Compliance: 90% (Strong)**

---

**CCPA Compliance: ✓ Strong**

**Section 1798.100 (Consumer Rights - Right to Know):**
- **✓ Implemented:** GET /customers/{id}?fields=... supports right to know
- **✓ Data portability:** JSON format enables data export

**Section 1798.105 (Consumer Rights - Right to Delete):**
- **✓ Implemented:** DELETE /customers/{id} supports deletion
- **⚠ Gap:** Backup retention (R-04 transferred risk) may conflict with deletion requirement

**Recommendation:** Document backup retention policy and procedure for deleting customer data from backups upon request.

**Section 1798.150 (Data Security Requirements):**
- **✓ Strong implementation:** 11 security controls meet "reasonable security" standard
- **✓ Breach notification:** (Gap identified above - see GDPR Article 33)

**Overall CCPA Compliance: 85% (Strong)**

---

#### Prioritized Recommendations

**Priority 1: Critical (Implement Within 1 Month)**

1. **Document Incident Response Procedures (Gap 2)**
   - **Effort:** 8-16 hours
   - **Impact:** High (GDPR/CCPA breach notification requirement)
   - **Action:** Create incident response plan, runbooks, and breach notification process

2. **Implement Database User Least Privilege**
   - **Effort:** 2-4 hours
   - **Impact:** Medium (reduces impact of SQL injection)
   - **Action:** Create read-only and read-write database users, assign to Lambda functions

---

**Priority 2: High (Implement Within 3 Months)**

3. **Enhance Monitoring and Alerting (Gap 1)**
   - **Effort:** 4-8 hours
   - **Impact:** High (early detection of attacks)
   - **Action:** Create CloudWatch alarms for security events, build security dashboard

4. **Harden Third-Party Dependencies (Gap 4)**
   - **Effort:** 8-12 hours
   - **Impact:** Medium (prevents supply chain attacks)
   - **Action:** Document Auth0 config, enable API Gateway WAF, implement dependency scanning

5. **Define Data Retention Policy**
   - **Effort:** 4-8 hours
   - **Impact:** Medium (GDPR Article 5 compliance)
   - **Action:** Document retention policy, implement automated deletion

---

**Priority 3: Medium (Implement Within 6 Months)**

6. **Implement Disaster Recovery Procedures (Gap 5)**
   - **Effort:** 12-16 hours
   - **Impact:** Medium (business continuity)
   - **Action:** Define RTO/RPO, document DR runbook, test procedures quarterly

7. **Re-evaluate Accepted Risks (Gap 3)**
   - **Effort:** 8-12 hours
   - **Impact:** Low-Medium (additional risk reduction)
   - **Action:** Migrate to UUIDs (I-05), implement IP whitelisting for admin endpoints (E-04)

8. **Implement Zero Trust Network Controls**
   - **Effort:** 8-12 hours
   - **Impact:** Medium (containment of breaches)
   - **Action:** Configure VPC Security Groups, implement egress filtering, add anomaly detection

---

**Priority 4: Low (Nice to Have)**

9. **Add API Gateway WAF**
   - **Effort:** 2-4 hours
   - **Impact:** Low (additional Layer 1 defense)
   - **Action:** Enable WAF with OWASP Top 10 rules

10. **Implement Automated Security Testing**
    - **Effort:** 12-16 hours
    - **Impact:** Low (continuous validation)
    - **Action:** Add automated security tests to CI/CD pipeline (DAST, SAST)

---

#### Final Verdict

**Security Posture: Strong (8.5/10)**

The customer data REST API demonstrates a mature security architecture with comprehensive threat analysis, well-designed controls, and strong compliance alignment. The architecture successfully addresses all critical threats and implements defense-in-depth principles.

**Key Achievements:**
- 100% of critical threats mitigated
- 84% reduction in overall risk score
- Strong GDPR (90%) and CCPA (85%) compliance
- Comprehensive documentation with full traceability

**Remaining Work:**
- 10 prioritized recommendations (focus on Priority 1-2)
- Estimated effort: 40-60 hours over 3 months
- Primary focus: Incident response, monitoring, third-party hardening

**Recommendation:** The API is production-ready from a security perspective, with Priority 1 recommendations implemented before launch. Priority 2-3 recommendations should be implemented within 3-6 months post-launch.

### Key Observations

As you review the architecture assessment, note:

1. **Quantified Security:** Clear security rating (8.5/10) with measurable improvement areas.

2. **Actionable Recommendations:** Each gap includes specific implementation steps and effort estimates.

3. **Prioritization:** Recommendations prioritized by risk impact and compliance requirements.

4. **Compliance Validation:** Explicit validation against GDPR and CCPA with percentage scores.

5. **Production Readiness:** Clear verdict on whether the system is ready for production.

6. **Pattern Analysis:** Evaluation of security patterns (defense-in-depth, least privilege, fail secure, zero trust).

### What to Document

Save the following from this step:

- Overall security rating and assessment summary
- Identified gaps with recommendations (all 10)
- Prioritized implementation roadmap
- Compliance validation results (GDPR/CCPA scores)
- Production readiness verdict

---

## Conclusion

Congratulations! You've completed an end-to-end security analysis of a customer data REST API using Claude Code security architecture skills.

### What You Accomplished

Over the past 45-60 minutes, you:

1. **Performed Threat Modeling (Step 1)**
   - Identified 24 threats using STRIDE methodology
   - Created attack trees showing threat chains
   - Classified threats by risk level (6 Critical, 6 High, 12 Medium)
   - Documented data flow with threat annotations

2. **Designed Security Controls (Step 2)**
   - Created 13 security controls mapped to threats
   - Provided implementation code for each control
   - Defined testing and validation procedures
   - Mapped controls to GDPR/CCPA requirements

3. **Created Security Documentation (Step 3)**
   - Built complete threat and control registers
   - Developed threat-control-compliance traceability matrix
   - Performed before/after risk assessment (84% risk reduction)
   - Generated compliance evidence packages

4. **Reviewed Architecture (Step 4)**
   - Assessed overall security posture (8.5/10)
   - Identified 10 gaps with prioritized recommendations
   - Validated GDPR (90%) and CCPA (85%) compliance
   - Determined production readiness

### Key Takeaways

**Security Architecture Process:**
- Systematic threat analysis before control design prevents gaps
- Defense-in-depth provides resilience against control failures
- Documentation enables compliance, audits, and knowledge transfer
- Architecture reviews validate coverage and identify improvements

**Skill Workflow:**
- `threat-modeling` → identifies what to protect against
- `security-controls-design` → designs how to protect
- `documenting-threats-and-controls` → creates compliance evidence
- `security-architecture-review` → validates completeness

**Real-World Application:**
- Adapt the API scenario to your specific system (web app, microservice, mobile backend)
- Use the same 4-step process for any security architecture work
- Leverage the documentation templates for audits and compliance reviews
- Iterate: revisit threat model as system evolves

### Next Steps

**Immediate Actions:**
1. **Implement Priority 1 Recommendations**
   - Document incident response procedures (8-16 hours)
   - Configure database user least privilege (2-4 hours)
   - **Timeline:** Within 1 month before production launch

2. **Set Up Monitoring**
   - Create CloudWatch alarms for security events (4-8 hours)
   - Build security metrics dashboard
   - **Timeline:** Within 1 week of production launch

3. **Schedule Architecture Review**
   - Quarterly review of threat model (new threats?)
   - Semi-annual review of controls (still effective?)
   - Annual compliance audit (GDPR/CCPA)

**Advanced Learning:**
- Apply these skills to more complex scenarios (microservices, multi-tenant SaaS)
- Explore additional security architecture skills:
  - `zero-trust-architecture` for network security
  - `defense-in-depth-design` for layered controls
  - `classified-systems-security` for high-security environments
- Integrate security into CI/CD pipeline (automated threat modeling)

**Share Your Work:**
- Use the documentation as a template for other projects
- Present the threat model to stakeholders for risk acceptance
- Submit compliance evidence to auditors for GDPR/CCPA reviews

### Documentation Package

You now have a complete security architecture package:

```
Customer Data API Security Documentation/
├── 01-threat-model.md
│   ├── STRIDE analysis (24 threats)
│   ├── Attack trees
│   └── Data flow diagrams
├── 02-security-controls.md
│   ├── Control catalog (13 controls)
│   ├── Implementation code
│   └── Testing procedures
├── 03-traceability-matrix.md
│   ├── Threat-control mapping
│   ├── Compliance mapping
│   └── Risk assessment
└── 04-architecture-review.md
    ├── Security posture assessment (8.5/10)
    ├── Gap analysis (10 recommendations)
    └── Production readiness verdict
```

This documentation supports:
- **Security Audits:** Complete threat and control catalog
- **Compliance Reviews:** GDPR/CCPA evidence with traceability
- **Penetration Testing:** Threat model guides test scope
- **Developer Onboarding:** Controls document secure coding practices
- **Incident Response:** Threat analysis informs detection and response

### Estimated Time Breakdown

- **Step 1 (Threat Modeling):** 15 minutes
- **Step 2 (Control Design):** 15 minutes
- **Step 3 (Documentation):** 10 minutes
- **Step 4 (Architecture Review):** 10-15 minutes
- **Total:** 45-60 minutes

### Tutorial Complete

You've successfully secured a REST API from threat identification through production readiness validation. These skills and processes apply to any system requiring security architecture analysis.

**Questions or Issues?**
- Review skill documentation: `ordis/security-architect/[skill-name]`
- Consult STRIDE methodology references
- Refer to GDPR/CCPA compliance frameworks

**Ready for more?** Try Tutorial 2: Achieving HIPAA Compliance for a Healthcare API (coming soon).

---

**Tutorial Version:** 1.0.0
**Last Updated:** 2025-10-29
**Skills Used:** threat-modeling, security-controls-design, documenting-threats-and-controls, security-architecture-review
**Compliance Frameworks:** GDPR, CCPA
**Technology Stack:** AWS API Gateway, Lambda, RDS PostgreSQL, OAuth 2.0/JWT
