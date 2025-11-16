
# API Authentication

## Overview

**API authentication specialist covering token patterns, OAuth2 flows, security hardening, compliance, monitoring, and production operations.**

**Core principle**: Authentication proves identity; authorization controls access - implement defense-in-depth with short-lived tokens, secure storage, rotation, monitoring, and assume breach to minimize blast radius.

## When to Use This Skill

Use when encountering:

- **Authentication strategy**: JWT vs sessions vs OAuth2 vs API keys
- **OAuth2 flows**: Authorization Code, PKCE, Client Credentials, token exchange
- **Token security**: Storage, rotation, revocation, theft detection
- **Service-to-service**: mTLS, service mesh, zero-trust
- **Mobile auth**: Secure storage, biometrics, certificate pinning
- **Security hardening**: Rate limiting, abuse prevention, anomaly detection
- **Monitoring**: Auth metrics, distributed tracing, audit logs
- **Compliance**: GDPR, PCI-DSS, SOC 2, audit trails
- **Multi-tenancy**: Tenant isolation, per-tenant policies
- **Testing**: Mock auth, development workflows

**Do NOT use for**:
- Application-specific business logic → Use domain skills
- Infrastructure security (firewalls, IDS) → `ordis-security-architect`
- Frontend auth UI → `lyra-ux-designer`

## Quick Reference - Authentication Patterns

| Pattern | Use Case | Security | Complexity | Revocation |
|---------|----------|----------|------------|------------|
| **JWT** | Mobile apps, APIs | Medium | Low | Hard (requires blacklist) |
| **Sessions** | Web apps, admin panels | High | Medium | Easy (delete session) |
| **OAuth2** | Third-party access, SSO | High | High | Medium (refresh rotation) |
| **API Keys** | Service-to-service, webhooks | Medium | Low | Easy (rotate keys) |
| **mTLS** | Service mesh, zero-trust | Very High | High | Medium (cert revocation) |

## JWT vs Sessions Decision Matrix

| Factor | JWT | Server-Side Sessions | Winner |
|--------|-----|---------------------|--------|
| **Mobile apps** | Excellent (stateless) | Poor (sticky sessions needed) | JWT |
| **Horizontal scaling** | Excellent (no shared state) | Requires sticky sessions or Redis | JWT |
| **Revocation** | Poor (need blacklist or short TTL) | Excellent (delete session) | Sessions |
| **Payload size** | Large (sent every request) | Small (session ID only) | Sessions |
| **Server memory** | None (stateless) | High (session store) | JWT |
| **XSS vulnerability** | High (if stored in localStorage) | Low (httpOnly cookies) | Sessions |
| **CSRF vulnerability** | None (bearer token) | High (requires CSRF tokens) | JWT |

**Production Recommendation**: **Hybrid Approach**

```
Architecture:
- Short-lived JWTs (15 min) for API access
- Long-lived refresh tokens stored server-side (session-like)
- Refresh endpoint returns new JWT + rotates refresh token

Benefits:
- Stateless API access (JWT)
- Secure revocation (server-side refresh tokens)
- Mobile-friendly (no cookies required)
- Horizontal scaling (minimal session state)
```

## OAuth2 Grant Types

### Grant Type Selection Matrix

| Client Type | Grant Type | Security | Use Case |
|-------------|-----------|----------|----------|
| **Web app (server-side)** | Authorization Code + PKCE | High | User login with backend |
| **SPA** | Authorization Code + PKCE | Medium-High | React/Vue/Angular apps |
| **Mobile app** | Authorization Code + PKCE | High | iOS/Android apps |
| **Service-to-service** | Client Credentials | High | Background jobs, APIs |
| **Device** | Device Authorization Grant | Medium | Smart TV, IoT devices |
| **Legacy** | ~~Password Grant~~ | DEPRECATED | Don't use |

### Authorization Code + PKCE (RFC 7636)

**Why PKCE?** Prevents authorization code interception attacks

```javascript
// Step 1: Generate PKCE challenge
const codeVerifier = crypto.randomBytes(32).toString('base64url');
const codeChallenge = crypto
  .createHash('sha256')
  .update(codeVerifier)
  .digest('base64url');

// Step 2: Redirect to authorization endpoint
const authUrl = new URL('https://auth.example.com/authorize');
authUrl.searchParams.set('response_type', 'code');
authUrl.searchParams.set('client_id', 'your_client_id');
authUrl.searchParams.set('redirect_uri', 'https://yourapp.com/callback');
authUrl.searchParams.set('scope', 'read write offline_access');
authUrl.searchParams.set('code_challenge', codeChallenge);
authUrl.searchParams.set('code_challenge_method', 'S256');
authUrl.searchParams.set('state', generateStateToken());  // CSRF protection

// Step 3: Exchange code for token
const tokenResponse = await fetch('https://auth.example.com/token', {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: new URLSearchParams({
    grant_type: 'authorization_code',
    code: receivedCode,
    redirect_uri: 'https://yourapp.com/callback',
    client_id: 'your_client_id',
    code_verifier: codeVerifier  // Proves you initiated the flow
  })
});

// Response
{
  "access_token": "eyJhbGc...",
  "token_type": "Bearer",
  "expires_in": 900,
  "refresh_token": "zxcvbnm...",
  "scope": "read write offline_access"
}
```

### Client Credentials (Service-to-Service)

```javascript
const tokenResponse = await fetch('https://auth.example.com/token', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Authorization': `Basic ${base64(client_id + ':' + client_secret)}`
  },
  body: new URLSearchParams({
    grant_type: 'client_credentials',
    scope: 'api.read api.write',
    audience: 'https://api.example.com'
  })
});
```

## Token Storage Security

### Storage Security Matrix

| Storage Location | XSS Risk | CSRF Risk | Accessible to JS | Production Use |
|------------------|----------|-----------|------------------|----------------|
| **localStorage** | ❌ HIGH | ✅ None | Yes | NEVER for tokens |
| **sessionStorage** | ❌ HIGH | ✅ None | Yes | NEVER for tokens |
| **Memory only** | ✅ None | ✅ None | Yes (in-app) | ✅ Access tokens (SPA) |
| **httpOnly cookie** | ✅ None | ❌ HIGH | No | ✅ Refresh tokens (+SameSite) |
| **Secure + httpOnly + SameSite=Strict** | ✅ None | ✅ Low | No | ✅ BEST for web |
| **iOS Keychain** | ✅ None | ✅ N/A | No (secure enclave) | ✅ Mobile apps |
| **Android Keystore** | ✅ None | ✅ N/A | No (hardware-backed) | ✅ Mobile apps |

### Web App Pattern (BFF - Backend For Frontend)

```javascript
// Frontend - access token in memory only
class AuthService {
  #accessToken = null;  // Private field, lost on refresh

  async callAPI(endpoint) {
    if (!this.#accessToken || this.isExpired(this.#accessToken)) {
      this.#accessToken = await this.refreshAccessToken();
    }

    return fetch(endpoint, {
      headers: { 'Authorization': `Bearer ${this.#accessToken}` }
    });
  }

  async refreshAccessToken() {
    // Calls BFF, which reads httpOnly cookie
    const response = await fetch('/api/auth/refresh', {
      method: 'POST',
      credentials: 'include'  // Send httpOnly cookie
    });

    const { access_token } = await response.json();
    return access_token;
  }
}

// Backend (BFF) - refresh endpoint
app.post('/api/auth/refresh', async (req, res) => {
  const refreshToken = req.cookies.refresh_token;  // httpOnly cookie

  // Validate and rotate refresh token
  const newTokens = await rotateRefreshToken(refreshToken);

  // Set new httpOnly cookie
  res.cookie('refresh_token', newTokens.refresh_token, {
    httpOnly: true,
    secure: true,
    sameSite: 'strict',
    maxAge: 7 * 24 * 60 * 60 * 1000  // 7 days
  });

  res.json({ access_token: newTokens.access_token, expires_in: 900 });
});
```

### Mobile App Pattern

```swift
// iOS - Keychain storage
import Security

class TokenStorage {
    func saveToken(_ token: String, forKey key: String) {
        let data = token.data(using: .utf8)!

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]

        SecItemDelete(query as CFDictionary)  // Delete old
        SecItemAdd(query as CFDictionary, nil)  // Add new
    }

    func getToken(forKey key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true
        ]

        var result: AnyObject?
        SecItemCopyMatching(query as CFDictionary, &result)

        guard let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }
}
```

```kotlin
// Android - EncryptedSharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

class TokenStorage(context: Context) {
    private val masterKey = MasterKey.Builder(context)
        .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
        .build()

    private val prefs = EncryptedSharedPreferences.create(
        context,
        "secure_prefs",
        masterKey,
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM
    )

    fun saveToken(key: String, token: String) {
        prefs.edit().putString(key, token).apply()
    }

    fun getToken(key: String): String? {
        return prefs.getString(key, null)
    }
}
```

## Refresh Token Rotation

### Pattern: Token Families with Replay Detection

```javascript
// Database schema
CREATE TABLE refresh_tokens (
  token_hash VARCHAR(64) PRIMARY KEY,
  user_id UUID NOT NULL,
  family_id UUID NOT NULL,
  parent_token_hash VARCHAR(64),
  device_id VARCHAR(255),
  ip_address INET,
  user_agent TEXT,
  created_at TIMESTAMP NOT NULL,
  expires_at TIMESTAMP NOT NULL,
  revoked BOOLEAN DEFAULT false,
  revoked_at TIMESTAMP,
  revoked_reason TEXT,
  INDEX idx_family (family_id),
  INDEX idx_user (user_id),
  INDEX idx_expires (expires_at)
);

// Refresh endpoint with rotation
async function refreshTokens(refreshToken, clientInfo) {
  const tokenHash = sha256(refreshToken);
  const dbToken = await db.query(
    'SELECT * FROM refresh_tokens WHERE token_hash = $1',
    [tokenHash]
  );

  // Case 1: Token not found or already revoked
  if (!dbToken || dbToken.revoked) {
    // Check if this token existed in history
    const historical = await db.query(
      'SELECT family_id FROM refresh_tokens WHERE token_hash = $1',
      [tokenHash]
    );

    if (historical.length > 0) {
      // REPLAY ATTACK DETECTED!
      // Revoke entire token family
      await db.query(
        'UPDATE refresh_tokens SET revoked = true, revoked_at = NOW(), ' +
        'revoked_reason = $1 WHERE family_id = $2',
        ['Replay attack detected', historical[0].family_id]
      );

      await auditLog.critical({
        event: 'token_replay_attack',
        user_id: historical[0].user_id,
        family_id: historical[0].family_id,
        ip: clientInfo.ip
      });

      throw new SecurityError('Token reuse detected - all sessions revoked');
    }

    throw new AuthError('Invalid refresh token');
  }

  // Case 2: Token expired
  if (dbToken.expires_at < new Date()) {
    throw new AuthError('Refresh token expired');
  }

  // Case 3: Valid token - rotate it
  const newRefreshToken = crypto.randomBytes(32).toString('base64url');
  const newAccessToken = generateJWT({
    sub: dbToken.user_id,
    scopes: ['read', 'write'],
    exp: Math.floor(Date.now() / 1000) + 900  // 15 min
  });

  // Revoke current token
  await db.query(
    'UPDATE refresh_tokens SET revoked = true WHERE token_hash = $1',
    [tokenHash]
  );

  // Create new token in same family
  await db.query(
    'INSERT INTO refresh_tokens ' +
    '(token_hash, user_id, family_id, parent_token_hash, device_id, ' +
    'ip_address, user_agent, created_at, expires_at) ' +
    'VALUES ($1, $2, $3, $4, $5, $6, $7, NOW(), NOW() + INTERVAL \'7 days\')',
    [
      sha256(newRefreshToken),
      dbToken.user_id,
      dbToken.family_id,      // Same family
      tokenHash,               // Track lineage
      clientInfo.device_id,
      clientInfo.ip,
      clientInfo.user_agent
    ]
  );

  return {
    access_token: newAccessToken,
    refresh_token: newRefreshToken,
    expires_in: 900,
    token_type: 'Bearer'
  };
}
```

### Advanced Refresh Patterns

**Absolute expiry** (max lifetime regardless of rotation):

```javascript
// Add max_family_age to family tracking
CREATE TABLE token_families (
  family_id UUID PRIMARY KEY,
  user_id UUID NOT NULL,
  created_at TIMESTAMP NOT NULL,
  max_lifetime_hours INT DEFAULT 720,  // 30 days max
  INDEX idx_user (user_id)
);

// Check absolute expiry
const familyAge = Date.now() - family.created_at;
const maxAge = family.max_lifetime_hours * 60 * 60 * 1000;

if (familyAge > maxAge) {
  throw new AuthError('Session expired - please re-authenticate');
}
```

**Grace period for concurrent requests**:

```javascript
// Allow small window for race conditions
const ROTATION_GRACE_PERIOD_MS = 5000;  // 5 seconds

if (dbToken.revoked && dbToken.revoked_at) {
  const timeSinceRevocation = Date.now() - dbToken.revoked_at;

  if (timeSinceRevocation < ROTATION_GRACE_PERIOD_MS) {
    // Within grace period - might be concurrent refresh
    // Return cached new tokens instead of replay alert
    const newTokens = await getChildToken(tokenHash);
    if (newTokens) return newTokens;
  }

  // Outside grace period - likely replay attack
  await revokeTokenFamily(dbToken.family_id);
}
```

## Rate Limiting & Abuse Prevention

### Authentication Endpoint Rate Limits

```javascript
const rateLimit = require('express-rate-limit');
const RedisStore = require('rate-limit-redis');

// Login endpoint - strict limits
const loginLimiter = rateLimit({
  store: new RedisStore({ client: redisClient }),
  windowMs: 15 * 60 * 1000,  // 15 minutes
  max: 5,  // Max 5 attempts
  message: 'Too many login attempts, please try again later',
  keyGenerator: (req) => {
    // Rate limit by IP + username combination
    return `login:${req.ip}:${req.body.username}`;
  },
  handler: (req, res) => {
    auditLog.warning({
      event: 'rate_limit_exceeded',
      endpoint: '/auth/login',
      ip: req.ip,
      username: req.body.username
    });

    res.status(429).json({
      error: 'rate_limit_exceeded',
      retry_after: res.getHeader('Retry-After')
    });
  }
});

app.post('/auth/login', loginLimiter, async (req, res) => {
  // Login logic
});

// Refresh endpoint - moderate limits
const refreshLimiter = rateLimit({
  store: new RedisStore({ client: redisClient }),
  windowMs: 60 * 1000,  // 1 minute
  max: 10,  // 10 refreshes per minute
  keyGenerator: (req) => `refresh:${req.ip}`
});

app.post('/auth/refresh', refreshLimiter, async (req, res) => {
  // Refresh logic
});
```

### Account Lockout After Failed Attempts

```javascript
async function attemptLogin(username, password, clientInfo) {
  const lockoutKey = `lockout:${username}`;
  const attemptsKey = `attempts:${username}`;

  // Check if account is locked
  const lockedUntil = await redis.get(lockoutKey);
  if (lockedUntil && Date.now() < parseInt(lockedUntil)) {
    throw new AuthError('Account temporarily locked due to failed login attempts');
  }

  // Verify credentials
  const user = await db.findUser(username);
  const valid = await bcrypt.compare(password, user.password_hash);

  if (!valid) {
    // Increment failed attempts
    const attempts = await redis.incr(attemptsKey);
    await redis.expire(attemptsKey, 15 * 60);  // 15 min window

    if (attempts >= 5) {
      // Lock account for 30 minutes
      const lockUntil = Date.now() + 30 * 60 * 1000;
      await redis.set(lockoutKey, lockUntil.toString(), 'EX', 30 * 60);

      await auditLog.warning({
        event: 'account_locked',
        user_id: user.id,
        attempts,
        ip: clientInfo.ip
      });

      throw new AuthError('Account locked due to too many failed attempts');
    }

    throw new AuthError('Invalid credentials');
  }

  // Success - clear attempts
  await redis.del(attemptsKey);

  // Check for anomalies
  await detectAnomalies(user.id, clientInfo);

  return generateTokens(user);
}
```

### Anomaly Detection

```javascript
async function detectAnomalies(userId, clientInfo) {
  // Get user's login history
  const recentLogins = await db.query(
    'SELECT ip_address, country, city FROM login_history ' +
    'WHERE user_id = $1 AND created_at > NOW() - INTERVAL \'30 days\' ' +
    'ORDER BY created_at DESC LIMIT 100',
    [userId]
  );

  // Check for new location
  const knownLocations = new Set(recentLogins.map(l => `${l.country}:${l.city}`));
  const currentLocation = `${clientInfo.country}:${clientInfo.city}`;

  if (!knownLocations.has(currentLocation)) {
    // New location - require additional verification
    await sendSecurityAlert(userId, {
      type: 'new_location',
      location: currentLocation,
      ip: clientInfo.ip
    });

    // Could require:
    // - Email verification
    // - 2FA challenge
    // - Security question
    // - Temporary session with limited access
  }

  // Check for impossible travel
  if (recentLogins.length > 0) {
    const lastLogin = recentLogins[0];
    const timeDiff = Date.now() - lastLogin.created_at;
    const distance = calculateDistance(
      lastLogin.country,
      clientInfo.country
    );

    // If 500+ km traveled in < 1 hour, flag as suspicious
    if (distance > 500 && timeDiff < 60 * 60 * 1000) {
      await auditLog.warning({
        event: 'impossible_travel',
        user_id: userId,
        from: lastLogin.country,
        to: clientInfo.country,
        time_diff_minutes: timeDiff / 60000
      });

      // Require step-up authentication
      return { require_2fa: true };
    }
  }
}
```

## Monitoring & Observability

### Key Metrics to Track

| Metric | Alert Threshold | Why It Matters |
|--------|----------------|----------------|
| **Login success rate** | < 80% | Credentials issues, attacks |
| **Token refresh failures** | > 5% | Rotation bugs, clock skew |
| **Rate limit hits** | > 100/hour | Brute force attempts |
| **Account lockouts** | > 10/hour | Credential stuffing attack |
| **Token replay attempts** | > 0 | Security breach |
| **Failed 2FA attempts** | > 3/user/day | Account compromise |
| **New device logins** | Monitor trends | Unusual activity |
| **p99 auth latency** | > 500ms | Performance degradation |

### Distributed Tracing for Auth Flows

```javascript
const { trace, context } = require('@opentelemetry/api');

const tracer = trace.getTracer('auth-service');

async function handleLogin(req, res) {
  return tracer.startActiveSpan('auth.login', async (span) => {
    span.setAttribute('user.username', req.body.username);
    span.setAttribute('client.ip', req.ip);
    span.setAttribute('client.user_agent', req.headers['user-agent']);

    try {
      // Nested span for credential validation
      const user = await tracer.startActiveSpan('auth.validate_credentials', async (validateSpan) => {
        const result = await validateCredentials(req.body.username, req.body.password);
        validateSpan.setAttribute('validation.success', !!result);
        validateSpan.end();
        return result;
      });

      if (!user) {
        span.setAttribute('auth.result', 'invalid_credentials');
        throw new AuthError('Invalid credentials');
      }

      // Nested span for token generation
      const tokens = await tracer.startActiveSpan('auth.generate_tokens', async (tokenSpan) => {
        const result = await generateTokens(user);
        tokenSpan.setAttribute('tokens.access_expiry', result.expires_in);
        tokenSpan.end();
        return result;
      });

      span.setAttribute('auth.result', 'success');
      span.setAttribute('user.id', user.id);

      res.json(tokens);
    } catch (error) {
      span.recordException(error);
      span.setAttribute('auth.result', 'error');
      throw error;
    } finally {
      span.end();
    }
  });
}

// Trace shows:
// auth.login (500ms)
//   ├── auth.validate_credentials (300ms)  // DB query
//   ├── auth.generate_tokens (50ms)        // JWT signing
//   └── auth.audit_log (150ms)             // Logging

// Can identify bottlenecks:
// - Slow password hashing (increase bcrypt rounds?)
// - Slow DB queries (add indexes?)
// - Network latency to Redis
```

### Audit Logging

```javascript
class AuditLogger {
  async log(event) {
    const entry = {
      timestamp: new Date().toISOString(),
      event_type: event.type,
      user_id: event.user_id,
      ip_address: event.ip,
      user_agent: event.user_agent,
      resource: event.resource,
      action: event.action,
      result: event.result,
      metadata: event.metadata,
      trace_id: context.active().getValue('trace_id')
    };

    // Write to multiple destinations
    await Promise.all([
      // 1. Append-only audit table (compliance)
      db.query('INSERT INTO audit_log (...) VALUES (...)', entry),

      // 2. Time-series database (analytics)
      influxdb.write('auth_events', entry),

      // 3. SIEM (security monitoring)
      siem.send(entry),

      // 4. Compliance log (immutable, encrypted)
      complianceLog.append(encrypt(entry))
    ]);
  }

  async critical(event) {
    await this.log({ ...event, severity: 'critical' });

    // Alert on critical events
    await alerting.send({
      title: `Critical Auth Event: ${event.event_type}`,
      details: event,
      severity: 'critical'
    });
  }
}

// Usage
await auditLog.log({
  type: 'login_success',
  user_id: user.id,
  ip: req.ip,
  user_agent: req.headers['user-agent'],
  result: 'success'
});

await auditLog.critical({
  type: 'token_replay_attack',
  user_id: user.id,
  family_id: token.family_id,
  ip: req.ip
});
```

## Multi-Tenancy Patterns

### Tenant Isolation in Tokens

```javascript
// JWT with tenant claim
const accessToken = jwt.sign({
  sub: user.id,
  tenant_id: user.tenant_id,        // Tenant isolation
  tenant_tier: tenant.tier,         // For rate limiting
  roles: user.roles,                // ['admin', 'user']
  scopes: ['read:orders', 'write:orders'],
  iss: 'https://auth.example.com',
  aud: 'https://api.example.com',
  exp: Math.floor(Date.now() / 1000) + 900
}, privateKey, { algorithm: 'RS256' });

// Middleware to enforce tenant isolation
function tenantIsolation(req, res, next) {
  const token = verifyJWT(req.headers.authorization);

  // Extract tenant from token
  req.tenant_id = token.tenant_id;

  // Add tenant filter to all DB queries
  req.dbFilter = { tenant_id: req.tenant_id };

  next();
}

// All queries automatically filtered
app.get('/orders', tenantIsolation, async (req, res) => {
  // Automatically filtered by tenant
  const orders = await db.query(
    'SELECT * FROM orders WHERE tenant_id = $1',
    [req.tenant_id]
  );
  res.json(orders);
});
```

### Per-Tenant Rate Limits

```javascript
const getTenantRateLimit = (tier) => {
  const limits = {
    free: { windowMs: 60000, max: 100 },      // 100/min
    pro: { windowMs: 60000, max: 1000 },      // 1000/min
    enterprise: { windowMs: 60000, max: 10000 } // 10k/min
  };
  return limits[tier] || limits.free;
};

app.use(async (req, res, next) => {
  const token = verifyJWT(req.headers.authorization);
  const tenant = await getTenant(token.tenant_id);

  const limit = getTenantRateLimit(tenant.tier);

  // Apply tenant-specific rate limit
  const limiter = rateLimit({
    ...limit,
    keyGenerator: () => `api:${tenant.id}`
  });

  limiter(req, res, next);
});
```

## Service-to-Service Authentication

### Zero-Trust Architecture

```
Principles:
1. Never trust, always verify
2. Assume breach
3. Verify explicitly (identity + device + location)
4. Least privilege access
5. Micro-segmentation
```

### Mutual TLS (mTLS) Pattern

```yaml
# Kubernetes with cert-manager
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: service-a-cert
spec:
  secretName: service-a-tls
  issuerRef:
    name: internal-ca
    kind: ClusterIssuer
  dnsNames:
  - service-a.default.svc.cluster.local
  usages:
  - digital signature
  - key encipherment
  - client auth  # Client authentication
  - server auth  # Server authentication

# Service configuration
apiVersion: v1
kind: Service
metadata:
  name: service-b
  annotations:
    service.alpha.kubernetes.io/app-protocols: '{"https":"HTTPS"}'
spec:
  ports:
  - port: 443
    protocol: TCP
    targetPort: 8443

# Pod configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: service-a
spec:
  template:
    spec:
      containers:
      - name: app
        volumeMounts:
        - name: tls
          mountPath: /etc/tls
          readOnly: true
      volumes:
      - name: tls
        secret:
          secretName: service-a-tls
```

```javascript
// Node.js client with mTLS
const https = require('https');
const fs = require('fs');

const options = {
  hostname: 'service-b.default.svc.cluster.local',
  port: 443,
  path: '/api/orders',
  method: 'GET',

  // Client certificate
  cert: fs.readFileSync('/etc/tls/tls.crt'),
  key: fs.readFileSync('/etc/tls/tls.key'),

  // CA certificate to verify server
  ca: fs.readFileSync('/etc/tls/ca.crt'),

  // Verify server identity
  checkServerIdentity: (hostname, cert) => {
    // Custom verification logic
    if (cert.subject.CN !== 'service-b.default.svc.cluster.local') {
      throw new Error('Server identity mismatch');
    }
  }
};

https.get(options, (res) => {
  // Handle response
});
```

### Service Mesh (Istio) Pattern

```yaml
# Automatic mTLS for all services
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: default
spec:
  mtls:
    mode: STRICT  # Require mTLS

# Authorization policy
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: service-b-policy
spec:
  selector:
    matchLabels:
      app: service-b
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/default/sa/service-a"]
    to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/api/orders/*"]
  - from:
    - source:
        principals: ["cluster.local/ns/default/sa/service-c"]
    to:
    - operation:
        methods: ["GET"]
        paths: ["/api/orders/*/status"]

# Request authentication (JWT validation)
apiVersion: security.istio.io/v1beta1
kind: RequestAuthentication
metadata:
  name: jwt-auth
spec:
  selector:
    matchLabels:
      app: service-b
  jwtRules:
  - issuer: "https://auth.example.com"
    jwksUri: "https://auth.example.com/.well-known/jwks.json"
    audiences:
    - "service-b"
```

## Mobile-Specific Patterns

### Certificate Pinning

```swift
// iOS - Certificate pinning with URLSession
class CertificatePinner: NSObject, URLSessionDelegate {
    let pinnedCertificates: [SecCertificate]

    init(pinnedCertificates: [SecCertificate]) {
        self.pinnedCertificates = pinnedCertificates
    }

    func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
              let serverTrust = challenge.protectionSpace.serverTrust else {
            completionHandler(.cancelAuthenticationChallenge, nil)
            return
        }

        // Get server certificate
        guard let serverCertificate = SecTrustGetCertificateAtIndex(serverTrust, 0) else {
            completionHandler(.cancelAuthenticationChallenge, nil)
            return
        }

        // Check if server cert matches any pinned cert
        let serverCertData = SecCertificateCopyData(serverCertificate) as Data

        for pinnedCert in pinnedCertificates {
            let pinnedCertData = SecCertificateCopyData(pinnedCert) as Data

            if serverCertData == pinnedCertData {
                // Certificate matches - allow connection
                let credential = URLCredential(trust: serverTrust)
                completionHandler(.useCredential, credential)
                return
            }
        }

        // Certificate not pinned - reject connection
        completionHandler(.cancelAuthenticationChallenge, nil)
    }
}
```

### Biometric Authentication

```swift
// iOS - Biometric auth (Face ID / Touch ID)
import LocalAuthentication

class BiometricAuth {
    func authenticate(reason: String, completion: @escaping (Bool, Error?) -> Void) {
        let context = LAContext()
        var error: NSError?

        // Check if biometric auth is available
        guard context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) else {
            completion(false, error)
            return
        }

        // Attempt biometric authentication
        context.evaluatePolicy(
            .deviceOwnerAuthenticationWithBiometrics,
            localizedReason: reason
        ) { success, error in
            DispatchQueue.main.async {
                if success {
                    // Biometric auth successful - retrieve token from Keychain
                    let token = TokenStorage().getToken(forKey: "refresh_token")
                    completion(true, nil)
                } else {
                    completion(false, error)
                }
            }
        }
    }
}
```

## Compliance & Regulations

### GDPR Considerations

```javascript
// Right to be forgotten - token revocation
async function deleteUserData(userId) {
  await db.transaction(async (tx) => {
    // 1. Revoke all active tokens
    await tx.query(
      'UPDATE refresh_tokens SET revoked = true, ' +
      'revoked_reason = $1 WHERE user_id = $2',
      ['GDPR deletion request', userId]
    );

    // 2. Anonymize audit logs (keep for compliance)
    await tx.query(
      'UPDATE audit_log SET user_id = NULL, ' +
      'ip_address = NULL, user_agent = NULL WHERE user_id = $1',
      [userId]
    );

    // 3. Delete user data
    await tx.query('DELETE FROM users WHERE id = $1', [userId]);
  });
}

// Data portability - export auth history
async function exportAuthData(userId) {
  const data = {
    login_history: await db.query(
      'SELECT created_at, ip_address, user_agent, result ' +
      'FROM login_history WHERE user_id = $1',
      [userId]
    ),
    active_sessions: await db.query(
      'SELECT created_at, device_id, ip_address, expires_at ' +
      'FROM refresh_tokens WHERE user_id = $1 AND revoked = false',
      [userId]
    )
  };

  return JSON.stringify(data, null, 2);
}
```

### PCI-DSS for Payment Systems

```javascript
// Requirements for authentication in payment systems

// 1. Strong access control (8.2)
const PASSWORD_REQUIREMENTS = {
  minLength: 12,
  requireUppercase: true,
  requireLowercase: true,
  requireNumbers: true,
  requireSpecialChars: true,
  preventReuse: 4,  // Can't reuse last 4 passwords
  maxAge: 90 * 24 * 60 * 60 * 1000  // 90 days
};

// 2. Multi-factor authentication (8.3)
async function loginWithMFA(username, password, mfaCode) {
  const user = await validateCredentials(username, password);
  if (!user) throw new AuthError('Invalid credentials');

  // Require MFA for all administrative access
  if (user.roles.includes('admin')) {
    const validMFA = await validateTOTP(user.id, mfaCode);
    if (!validMFA) throw new AuthError('Invalid MFA code');
  }

  return generateTokens(user);
}

// 3. Session timeout (8.1.8)
const SESSION_TIMEOUT = 15 * 60 * 1000;  // 15 minutes idle

// 4. Audit logging (10.2)
await auditLog.log({
  type: 'cardholder_data_access',
  user_id: user.id,
  resource: 'payment_methods',
  action: 'read',
  result: 'success',
  timestamp: new Date().toISOString()
});
```

## Testing Strategies

### Mock Auth for Development

```javascript
// Development-only bypass (NEVER in production)
if (process.env.NODE_ENV === 'development') {
  app.use('/dev-auth/login-as/:userId', async (req, res) => {
    if (process.env.ENABLE_DEV_AUTH !== 'true') {
      return res.status(403).json({ error: 'Dev auth not enabled' });
    }

    const user = await db.findUser(req.params.userId);
    const tokens = await generateTokens(user);

    res.json(tokens);
  });
}

// Environment check middleware
app.use((req, res, next) => {
  if (req.path.startsWith('/dev-auth') && process.env.NODE_ENV !== 'development') {
    return res.status(404).json({ error: 'Not found' });
  }
  next();
});
```

### Integration Testing

```javascript
const request = require('supertest');
const app = require('./app');

describe('OAuth2 Authorization Code Flow', () => {
  let authCode, codeVerifier;

  it('should initiate authorization', async () => {
    codeVerifier = generatePKCEVerifier();
    const codeChallenge = generatePKCEChallenge(codeVerifier);

    const res = await request(app)
      .get('/oauth/authorize')
      .query({
        response_type: 'code',
        client_id: 'test_client',
        redirect_uri: 'http://localhost:3000/callback',
        scope: 'read write',
        code_challenge: codeChallenge,
        code_challenge_method: 'S256',
        state: 'random_state'
      });

    expect(res.status).toBe(302);
    expect(res.headers.location).toContain('code=');

    // Extract code from redirect
    const url = new URL(res.headers.location);
    authCode = url.searchParams.get('code');
  });

  it('should exchange code for tokens', async () => {
    const res = await request(app)
      .post('/oauth/token')
      .send({
        grant_type: 'authorization_code',
        code: authCode,
        redirect_uri: 'http://localhost:3000/callback',
        client_id: 'test_client',
        code_verifier: codeVerifier
      });

    expect(res.status).toBe(200);
    expect(res.body).toHaveProperty('access_token');
    expect(res.body).toHaveProperty('refresh_token');
    expect(res.body.token_type).toBe('Bearer');
    expect(res.body.expires_in).toBe(900);
  });

  it('should detect PKCE verification failure', async () => {
    const res = await request(app)
      .post('/oauth/token')
      .send({
        grant_type: 'authorization_code',
        code: authCode,
        redirect_uri: 'http://localhost:3000/callback',
        client_id: 'test_client',
        code_verifier: 'wrong_verifier'  // Wrong verifier
      });

    expect(res.status).toBe(400);
    expect(res.body.error).toBe('invalid_grant');
  });
});

describe('Refresh Token Rotation', () => {
  let refreshToken1, refreshToken2;

  it('should rotate refresh token on use', async () => {
    // First refresh
    const res1 = await request(app)
      .post('/auth/refresh')
      .send({ refresh_token: originalRefreshToken });

    expect(res1.status).toBe(200);
    refreshToken1 = res1.body.refresh_token;

    // Second refresh with new token
    const res2 = await request(app)
      .post('/auth/refresh')
      .send({ refresh_token: refreshToken1 });

    expect(res2.status).toBe(200);
    refreshToken2 = res2.body.refresh_token;

    expect(refreshToken1).not.toBe(refreshToken2);
  });

  it('should detect refresh token replay', async () => {
    // Try to reuse first refresh token (already rotated)
    const res = await request(app)
      .post('/auth/refresh')
      .send({ refresh_token: refreshToken1 });

    expect(res.status).toBe(401);
    expect(res.body.error).toContain('replay');

    // Entire family should be revoked
    const familyCheck = await request(app)
      .post('/auth/refresh')
      .send({ refresh_token: refreshToken2 });

    expect(familyCheck.status).toBe(401);  // Also revoked
  });
});
```

## Token Validation Patterns

### JWT Validation with Caching

```javascript
const jwt = require('jsonwebtoken');
const { NodeCache } = require('node-cache');

const publicKeyCache = new NodeCache({ stdTTL: 3600 });  // 1 hour

async function validateJWT(token) {
  // Decode without verification to get header
  const decoded = jwt.decode(token, { complete: true });
  if (!decoded) throw new AuthError('Invalid token format');

  const keyId = decoded.header.kid;

  // Try cache first
  let publicKey = publicKeyCache.get(keyId);

  if (!publicKey) {
    // Fetch from JWKS endpoint
    const jwks = await fetch('https://auth.example.com/.well-known/jwks.json');
    const keys = await jwks.json();

    const key = keys.keys.find(k => k.kid === keyId);
    if (!key) throw new AuthError('Public key not found');

    publicKey = jwkToPem(key);
    publicKeyCache.set(keyId, publicKey);
  }

  // Verify signature and claims
  try {
    const payload = jwt.verify(token, publicKey, {
      algorithms: ['RS256'],
      issuer: 'https://auth.example.com',
      audience: 'https://api.example.com'
    });

    // Additional validation
    if (!payload.sub) throw new AuthError('Missing subject claim');
    if (!payload.scopes || !Array.isArray(payload.scopes)) {
      throw new AuthError('Missing or invalid scopes');
    }

    return payload;
  } catch (error) {
    if (error.name === 'TokenExpiredError') {
      throw new AuthError('Token expired');
    }
    throw new AuthError('Token validation failed');
  }
}
```

### Key Rotation Without Downtime

```javascript
// Support multiple signing keys simultaneously
const CURRENT_KEY_ID = 'key-2024-11';
const PREVIOUS_KEY_ID = 'key-2024-10';

const signingKeys = new Map([
  [CURRENT_KEY_ID, fs.readFileSync('/keys/current-private.pem')],
  [PREVIOUS_KEY_ID, fs.readFileSync('/keys/previous-private.pem')]
]);

// Sign with current key
function generateJWT(payload) {
  return jwt.sign(payload, signingKeys.get(CURRENT_KEY_ID), {
    algorithm: 'RS256',
    keyid: CURRENT_KEY_ID,
    expiresIn: '15m'
  });
}

// Validate with either key (grace period)
function validateJWT(token) {
  const decoded = jwt.decode(token, { complete: true });
  const keyId = decoded.header.kid;

  if (!signingKeys.has(keyId)) {
    throw new AuthError('Unknown signing key');
  }

  return jwt.verify(token, signingKeys.get(keyId), {
    algorithms: ['RS256']
  });
}

// Key rotation process:
// 1. Generate new key pair → key-2024-12
// 2. Add to signingKeys map (validation now accepts 3 keys)
// 3. Update CURRENT_KEY_ID to key-2024-12 (new tokens use new key)
// 4. Wait for old tokens to expire (15 min)
// 5. Remove key-2024-10 from signingKeys map
```

## Anti-Patterns

| Anti-Pattern | Why Bad | Fix |
|--------------|---------|-----|
| **Long-lived JWTs** | Can't revoke, security risk | Max 15-60 min, use refresh tokens |
| **Tokens in localStorage** | XSS vulnerability | httpOnly cookies or memory-only |
| **No refresh rotation** | Stolen token = permanent access | Rotate on every use, detect replay |
| **Password Grant** | App handles credentials, no MFA | Authorization Code + PKCE |
| **Shared secrets across services** | One breach = all compromised | Per-service secrets, rotate regularly |
| **No rate limiting** | Brute force attacks | Rate limit login, refresh, sensitive endpoints |
| **Ignoring anomalies** | Account takeover undetected | Monitor location, device, behavior |
| **No audit logging** | Can't investigate breaches | Log all auth events, immutable storage |
| **Weak password requirements** | Easy to crack | 12+ chars, complexity, no common passwords |
| **No MFA for admins** | Privileged account compromise | Require MFA for elevated access |

## Cross-References

**Related skills**:
- **Security architecture** → `ordis-security-architect` (threat modeling, defense-in-depth)
- **FastAPI implementation** → `fastapi-development` (FastAPI auth middleware)
- **REST API design** → `rest-api-design` (Bearer tokens, auth headers)
- **GraphQL auth** → `graphql-api-design` (context-based auth, directives)
- **Microservices** → `microservices-architecture` (service mesh, mTLS)

## Further Reading

- **OAuth 2.1**: Latest OAuth spec (consolidates best practices)
- **RFC 7636**: PKCE specification
- **RFC 8693**: Token exchange for delegation
- **OWASP Auth Cheat Sheet**: https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html
- **JWT Best Practices**: https://datatracker.ietf.org/doc/html/rfc8725
- **Zero Trust Architecture**: NIST SP 800-207
