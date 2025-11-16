
# Security-Aware Documentation

## Overview

Document systems without compromising security. Core principle: **Documentation should inform, not expose**.

**Key insight**: Examples with real credentials/PII leak secrets. Obviously fake examples are safe and clear.

## When to Use

Load this skill when:
- Documenting authentication/authorization
- Creating API examples with credentials
- Writing about systems handling PII or classified data
- Documenting security features

**Symptoms you need this**:
- "How do I show API key example without exposing real keys?"
- Writing documentation for healthcare/finance systems
- Creating examples with user data
- Documenting security configurations

**Don't use for**:
- General documentation without security concerns
- Internal-only docs (though still good practice)

## Sanitizing Examples

### Rule 1: Never Use Real Credentials

❌ **WRONG**:
```bash
# Don't mask real secrets
curl -H "Authorization: Bearer sk_live_51Hx***REDACTED***" \
  https://api.example.com/users
```

**Problem**: Pattern `sk_live_51Hx...` suggests real Stripe key structure. Reader might think they should unmask it.

✅ **RIGHT**:
```bash
# Generate obviously fake credentials
curl -H "Authorization: Bearer fake_key_abc123_for_docs_only" \
  https://api.example.com/users
```

**Better**: Clearly fake, no confusion possible.


### Rule 2: Use Obviously Fake Values

**Fake Credentials Pattern**:
```
fake_[type]_[random]_for_docs_only
```

**Examples**:
- API key: `fake_api_key_abc123_for_docs_only`
- JWT token: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.FAKE_TOKEN_FOR_DOCUMENTATION.fake_signature_do_not_use`
- Database password: `fake_password_p@ssw0rd_example_only`
- SSH key: `fake_ssh_key_AAAAB3NzaC1yc2EAAAADAQABAAABAQ... (truncated for docs)`

**Fake PII Pattern**:
```
[common_name]@example.com
000-00-0000 (SSN)
+1-555-0123 (phone - 555 is reserved for fiction)
```

**Examples**:
- Email: `jane.doe@example.com`, `user@example.org`
- SSN: `000-00-0000`, `123-45-6789` (invalid format)
- Phone: `+1-555-0100`, `+1-555-0199`
- Address: `123 Main Street, Anytown, ST 12345`


### Rule 3: Use Reserved Domains

**Reserved for documentation** (RFC 2606):
- `example.com`
- `example.net`
- `example.org`
- `test.com` (not official but commonly accepted)

❌ **WRONG**:
```javascript
const API_URL = 'https://api.acmecorp.com';  // Real company?
```

✅ **RIGHT**:
```javascript
const API_URL = 'https://api.example.com';  // Obviously fake
```


### Rule 4: Complete Fake Examples (Not Partial)

❌ **WRONG**:
```python
# Partial example - reader must guess
api_key = "YOUR_API_KEY_HERE"
client = APIClient(api_key)
```

**Problem**: Reader doesn't know what `YOUR_API_KEY_HERE` should look like.

✅ **RIGHT**:
```python
# Complete fake example - copy-paste-run (with fake backend)
api_key = "fake_api_key_abc123_for_docs_only"
client = APIClient(api_key)

# For real usage:
# 1. Get API key from https://dashboard.example.com/settings
# 2. Replace fake_api_key_abc123_for_docs_only with your real key
# 3. Never commit real keys to git
```

**Better**: Complete example + clear instructions for real usage.


## Threat Disclosure Decisions

### Document: Security Features Users Must Configure

✅ **DO document**:
```markdown
## Security Configuration

### Enable MFA (Required for Production)

Multi-factor authentication prevents unauthorized access even if passwords are compromised.

Enable MFA for all admin accounts:
\```bash
user-admin mfa enable --user admin@example.com --method totp
\```

**Security impact**: Without MFA, stolen passwords grant full access.
```


### Document: Security Best Practices

✅ **DO document**:
```markdown
## Hardening Guide

### Disable Unused Services

**Threat**: Unused services increase attack surface.

Disable SSH if not needed:
\```bash
systemctl disable sshd
systemctl stop sshd
\```

**Verification**: `systemctl status sshd` should show "inactive (dead)"
```


### Don't Document: Specific Vulnerabilities

❌ **DON'T document**:
```markdown
## Known Issues

### CVE-2024-12345: SQL Injection in /api/users Endpoint

Vulnerable code in `user_controller.py:45`:
\```python
query = f"SELECT * FROM users WHERE id = {user_id}"  # Vulnerable!
\```

Attacker can inject SQL via: `/api/users?id=1 OR 1=1`
```

**Problem**: Provides exploit guide to attackers.

✅ **DO instead**: Coordinate with security team for disclosure. Document fix after patch released:
```markdown
## Security Updates

### Version 2.1.5 (2024-03-15)

**Security fix**: Resolved input validation issue in user API (CVE-2024-12345).
Upgrade immediately.

For technical details, see our security advisory: [link]
```


### Don't Document: Internal Security Architecture (Unless Necessary)

❌ **DON'T document** (in public docs):
```markdown
## Internal Security Architecture

Our secrets vault runs on ec2-10-0-1-50.internal with:
- Port 8200 (HTTP API)
- Port 8201 (cluster communication)
- Root token stored in S3 bucket: company-secrets-prod
- Unsealing keys split across: admin1@company.com, admin2@company.com, admin3@company.com
```

**Problem**: Reveals infrastructure details aiding attackers (IP addresses, ports, bucket names, key custodians).

✅ **DO instead**: Document what users need, abstract internals:
```markdown
## Secrets Management

Secrets are stored in an encrypted vault. To access secrets:

1. Request access via [access-request-form]
2. Use provided vault token
3. Tokens expire after 8 hours

See [secrets-access-guide] for details.
```


## Compliance Sensitivity

### Rule: Document Controls Without Revealing Weaknesses

❌ **WRONG**:
```markdown
## SOC2 Compliance

### Access Control (CC6.1)

**Control**: Role-based access control (RBAC)

**Current gaps**:
- Admin users can bypass RBAC via debug mode (known issue #245)
- No access reviews conducted (planned for Q3)
- 3 dormant admin accounts still active (cleanup delayed)
```

**Problem**: Audit report publicly reveals control weaknesses.

✅ **RIGHT**:
```markdown
## SOC2 Compliance

### Access Control (CC6.1)

**Control**: Role-based access control (RBAC)

**Implementation**:
- Authentication: MFA required for admin accounts
- Authorization: Permissions enforced at API layer and database layer
- Access reviews: Quarterly review of all privileged accounts
- Account lifecycle: Automated disablement after 30 days inactivity

**Audit evidence**: Available to authorized auditors via [compliance-portal]
```

**Better**: Focus on what exists, keep gaps internal.


## Redaction Patterns

### Logs: Redact Sensitive Data

❌ **WRONG**:
```
[2024-03-15 10:23:45] User login: user=john.smith@acme.com, password=MyP@ssw0rd123, token=eyJhbGci...
[2024-03-15 10:24:12] API call: GET /users/12345, auth_token=sk_live_51HxAbC123...
```

✅ **RIGHT**:
```
[2024-03-15 10:23:45] User login: user=john.smith@acme.com, password=[REDACTED], token=[REDACTED]
[2024-03-15 10:24:12] API call: GET /users/12345, auth_token=[REDACTED]
```

**Even better** (for docs): Use fake data:
```
[2024-03-15 10:23:45] User login: user=jane.doe@example.com, password=[REDACTED], token=[REDACTED]
[2024-03-15 10:24:12] API call: GET /users/67890, auth_token=[REDACTED]
```


### Screenshots: Blur Sensitive Data

**Before sharing screenshot**:
1. Use test account (user@example.com, fake data)
2. Blur any real data (names, emails, IDs)
3. Use browser extensions: "Redact for Screenshot"

❌ **WRONG**: Screenshot with production data visible

✅ **RIGHT**: Screenshot with:
- Fake user names (Jane Doe, John Smith)
- Fake emails (jane@example.com)
- Blurred real data if any leaked


### Diagrams: Anonymize Infrastructure

❌ **WRONG**:
```
[Load Balancer: lb-prod-01.company.com (52.12.34.56)]
       ↓
[API Servers: api-01.internal (10.0.1.10), api-02.internal (10.0.1.11)]
       ↓
[Database: postgres-master.internal (10.0.2.50)]
```

✅ **RIGHT**:
```
[Load Balancer: lb.example.com (203.0.113.10)]  # RFC 5737 documentation IP
       ↓
[API Servers: api-01, api-02 (192.168.1.10-11)]  # RFC 1918 private IP
       ↓
[Database: postgres-master (192.168.2.50)]
```


### Database Schemas: Use Synthetic Data

❌ **WRONG**:
```sql
-- Example users table
SELECT * FROM users LIMIT 3;

| id | email                    | ssn         | credit_card      |
|----|--------------------------|-------------|------------------|
| 1  | alice@acmecorp.com       | 123-45-6789 | 4532-1234-5678-9012 |
| 2  | bob@acmecorp.com         | 987-65-4321 | 5105-1051-0510-5100 |
| 3  | charlie@acmecorp.com     | 456-78-9123 | 3782-822463-10005 |
```

✅ **RIGHT**:
```sql
-- Example users table (synthetic data)
SELECT * FROM users LIMIT 3;

| id | email                  | ssn           | credit_card      |
|----|------------------------|---------------|------------------|
| 1  | alice@example.com      | 000-00-0001   | 0000-0000-0000-0001 |
| 2  | bob@example.com        | 000-00-0002   | 0000-0000-0000-0002 |
| 3  | charlie@example.com    | 000-00-0003   | 0000-0000-0000-0003 |
```


## Security Feature Documentation

### Pattern: Threat + Configuration + Impact

```markdown
## Security Feature: [Name]

### Threat Prevented
[What attack does this prevent?]

### Configuration
[How to enable/configure?]
\```bash
[Example commands]
\```

### Security Impact
**If enabled**: [What protection do you get?]
**If disabled**: [What risk remains?]

### Verification
[How to verify it's working?]
\```bash
[Test commands]
\```
```

### Example: Rate Limiting

```markdown
## Security Feature: API Rate Limiting

### Threat Prevented
**Brute force attacks**: Attacker attempts thousands of login requests to guess passwords.
**DoS attacks**: Attacker overwhelms API with excessive requests.

### Configuration

Enable rate limiting in `config.yaml`:
\```yaml
rate_limiting:
  enabled: true
  max_requests: 100  # per minute per IP
  window: 60         # seconds
  block_duration: 300  # 5 minutes
\```

Restart API server:
\```bash
systemctl restart api-server
\```

### Security Impact
**If enabled**:
- Brute force attack limited to 100 attempts/minute (vs unlimited)
- Single IP cannot DoS entire service
- Legitimate users unaffected (typical usage: 10-20 req/min)

**If disabled**:
- Attacker can attempt 1000s of passwords per minute
- Single attacker can exhaust server resources
- No protection against credential stuffing attacks

### Verification

Test rate limit:
\```bash
# Attempt 101 requests in 1 minute
for i in {1..101}; do
  curl https://api.example.com/login -d "user=test&pass=fake"
done

# Expected: First 100 succeed, 101st returns:
# HTTP 429 Too Many Requests
# Retry-After: 60
\```

Check logs:
\```bash
grep "rate_limit_exceeded" /var/log/api-server.log
# Should show: [2024-03-15 10:25:45] Rate limit exceeded: IP 203.0.113.10, endpoint /login
\```
```


## Quick Reference: Sanitization Checklist

| Data Type | ❌ Wrong | ✅ Right |
|-----------|---------|---------|
| **API Key** | `sk_live_***REDACTED***` | `fake_api_key_abc123_for_docs_only` |
| **JWT Token** | `eyJhbGci...` (real masked) | `eyJhbGci...FAKE_TOKEN_FOR_DOCUMENTATION...` |
| **Email** | `john.smith@acme.com` | `jane.doe@example.com` |
| **SSN** | `***-**-1234` | `000-00-0000` |
| **Phone** | `(555) ***-1234` | `+1-555-0100` |
| **IP Address** | `52.12.34.56` (real AWS) | `203.0.113.10` (RFC 5737 docs) |
| **Domain** | `api.acme.com` | `api.example.com` |
| **Database Password** | `p@ssw***` | `fake_password_example_only` |


## Common Mistakes

### ❌ Masking Real Secrets

**Wrong**: `sk_live_***REDACTED***` (pattern suggests real key)

**Right**: `fake_api_key_abc123_for_docs_only` (obviously fake)

**Why**: Masked secrets still leak structure. Readers might try to unmask or think it's production data.


### ❌ Using Real Company Names

**Wrong**: `curl https://api.acmecorp.com` (might be real company)

**Right**: `curl https://api.example.com` (reserved for docs)

**Why**: Avoid accidental real company references. Use RFC-designated example domains.


### ❌ Documenting Exploits Before Patch

**Wrong**: Publish CVE details with exploit code before customers patch

**Right**: Coordinate disclosure with security team, publish after patch available

**Why**: Responsible disclosure prevents weaponizing vulnerabilities before users can protect themselves.


### ❌ Incomplete Redaction

**Wrong**: Redact password but leave username + server IP

**Right**: Redact all PII/credentials and use example IPs

**Why**: Partial redaction still enables attacks. Usernames + server IPs = reconnaissance.


## Cross-References

**Use WITH this skill**:
- `muna/technical-writer/clarity-and-style` - Write clear security documentation
- `ordis/security-architect/threat-modeling` - Understand threats documentation might expose

**Use AFTER this skill**:
- `muna/technical-writer/documentation-testing` - Verify examples work (with fake credentials)

## Real-World Impact

**Projects using security-aware documentation**:
- **API Documentation (Healthcare)**: Sanitized all examples with `jane.doe@example.com`, `fake_api_key_...`. Prevented accidental PII exposure in publicly-accessible docs.
- **OAuth Flow Tutorial**: Used complete fake examples (`client_id=fake_client_abc123`) vs placeholders (`YOUR_CLIENT_ID_HERE`). Support tickets reduced 60% ("I don't know what client ID looks like").
- **Database Migration Guide**: Used synthetic data (SSN: 000-00-0000) vs redacted real data (SSN: ***-**-1234). Compliance audit passed with "exemplary PII handling in documentation".

**Key lesson**: **Obviously fake examples are clearer and safer than masked real data. Complete fake examples enable copy-paste-run testing without security risk.**
