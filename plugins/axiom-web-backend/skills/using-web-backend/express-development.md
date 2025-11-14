
# Express Development

## Overview

**Express.js development specialist covering middleware organization, error handling, validation, database integration, testing, and production deployment.**

**Core principle**: Express's minimalist philosophy requires disciplined patterns - without structure, Express apps become tangled middleware chains with inconsistent error handling and poor testability.

## When to Use This Skill

Use when encountering:

- **Middleware organization**: Ordering, async error handling, custom middleware
- **Error handling**: Centralized handlers, custom error classes, async/await errors
- **Request validation**: Zod, express-validator, type-safe validation
- **Database patterns**: Connection pooling, transactions, graceful shutdown
- **Testing**: Supertest, mocking, middleware isolation
- **Production deployment**: PM2, clustering, Docker, environment management
- **Performance**: Compression, caching, clustering
- **Security**: Helmet, rate limiting, CORS, input sanitization

**DO NOT use for**:
- General TypeScript patterns (use `axiom-python-engineering` equivalents)
- API design principles (use `rest-api-design`)
- Database-agnostic patterns (use `database-integration`)

## Middleware Organization

### Correct Middleware Order

**Order matters** - middleware executes top to bottom:

```typescript
import express from 'express';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';

const app = express();

// 1. Security (FIRST - before any parsing)
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
    },
  },
}));

// 2. CORS (before routes)
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(','),
  credentials: true,
  maxAge: 86400, // 24 hours
}));

// 3. Parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// 4. Compression
app.use(compression());

// 5. Logging
app.use(morgan('combined', { stream: logger.stream }));

// 6. Authentication (before routes that need it)
app.use('/api', authenticationMiddleware);

// 7. Routes
app.use('/api/users', userRoutes);
app.use('/api/posts', postRoutes);

// 8. 404 handler (AFTER all routes)
app.use((req, res) => {
  res.status(404).json({
    status: 'error',
    message: 'Route not found',
    path: req.path,
  });
});

// 9. Error handler (LAST)
app.use(errorHandler);
```

### Async Error Wrapper

**Problem**: Express doesn't catch async errors automatically

```typescript
// src/middleware/asyncHandler.ts
import { Request, Response, NextFunction } from 'express';

export const asyncHandler = <T>(
  fn: (req: Request, res: Response, next: NextFunction) => Promise<T>
) => {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};

// Usage
router.get('/:id', asyncHandler(async (req, res) => {
  const user = await userService.findById(req.params.id);
  if (!user) throw new NotFoundError('User not found');
  res.json(user);
}));
```

**Alternative**: Use express-async-errors (automatic)

```typescript
// At top of app.ts (BEFORE routes)
import 'express-async-errors';

// Now all async route handlers auto-catch errors
router.get('/:id', async (req, res) => {
  const user = await userService.findById(req.params.id);
  res.json(user);
}); // Errors automatically forwarded to error handler
```

## Error Handling

### Custom Error Classes

```typescript
// src/errors/AppError.ts
export class AppError extends Error {
  constructor(
    public readonly message: string,
    public readonly statusCode: number,
    public readonly isOperational: boolean = true
  ) {
    super(message);
    Error.captureStackTrace(this, this.constructor);
  }
}

// src/errors/HttpErrors.ts
export class BadRequestError extends AppError {
  constructor(message: string) {
    super(message, 400);
  }
}

export class UnauthorizedError extends AppError {
  constructor(message = 'Unauthorized') {
    super(message, 401);
  }
}

export class ForbiddenError extends AppError {
  constructor(message = 'Forbidden') {
    super(message, 403);
  }
}

export class NotFoundError extends AppError {
  constructor(message: string) {
    super(message, 404);
  }
}

export class ConflictError extends AppError {
  constructor(message: string) {
    super(message, 409);
  }
}

export class TooManyRequestsError extends AppError {
  constructor(message = 'Too many requests', public retryAfter?: number) {
    super(message, 429);
  }
}
```

### Centralized Error Handler

```typescript
// src/middleware/errorHandler.ts
import { Request, Response, NextFunction } from 'express';
import { AppError } from '../errors/AppError';
import { logger } from '../config/logger';

export const errorHandler = (
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction
) => {
  // Log error with context
  logger.error('Error occurred', {
    error: {
      message: err.message,
      stack: err.stack,
      name: err.name,
    },
    request: {
      method: req.method,
      url: req.url,
      ip: req.ip,
      userAgent: req.get('user-agent'),
    },
  });

  // Operational errors (expected)
  if (err instanceof AppError && err.isOperational) {
    const response: any = {
      status: 'error',
      message: err.message,
    };

    // Add retry-after for rate limiting
    if (err instanceof TooManyRequestsError && err.retryAfter) {
      res.setHeader('Retry-After', err.retryAfter);
      response.retryAfter = err.retryAfter;
    }

    return res.status(err.statusCode).json(response);
  }

  // Validation errors (Zod, express-validator)
  if (err.name === 'ZodError') {
    return res.status(400).json({
      status: 'error',
      message: 'Validation failed',
      errors: (err as any).errors,
    });
  }

  // Database constraint violations
  if ((err as any).code === '23505') { // PostgreSQL unique violation
    return res.status(409).json({
      status: 'error',
      message: 'Resource already exists',
    });
  }

  if ((err as any).code === '23503') { // Foreign key violation
    return res.status(400).json({
      status: 'error',
      message: 'Invalid reference',
    });
  }

  // Unexpected errors (don't leak details in production)
  res.status(500).json({
    status: 'error',
    message: process.env.NODE_ENV === 'production'
      ? 'Internal server error'
      : err.message,
    ...(process.env.NODE_ENV !== 'production' && { stack: err.stack }),
  });
};
```

### Global Error Handlers

```typescript
// src/server.ts
process.on('unhandledRejection', (reason: Error) => {
  logger.error('Unhandled Rejection', { reason });
  // Graceful shutdown
  server.close(() => process.exit(1));
});

process.on('uncaughtException', (error: Error) => {
  logger.error('Uncaught Exception', { error });
  process.exit(1);
});
```

## Request Validation

### Zod Integration (Type-Safe)

```typescript
// src/schemas/userSchema.ts
import { z } from 'zod';

export const createUserSchema = z.object({
  body: z.object({
    email: z.string().email('Invalid email'),
    password: z.string()
      .min(8, 'Password must be at least 8 characters')
      .regex(/[A-Z]/, 'Password must contain uppercase')
      .regex(/[0-9]/, 'Password must contain number'),
    name: z.string().min(2).max(100),
    age: z.number().int().positive().max(150).optional(),
  }),
});

export const getUserSchema = z.object({
  params: z.object({
    id: z.string().regex(/^\d+$/, 'ID must be numeric'),
  }),
});

export const getUsersSchema = z.object({
  query: z.object({
    page: z.string().regex(/^\d+$/).transform(Number).default('1'),
    limit: z.string().regex(/^\d+$/).transform(Number).default('10'),
    search: z.string().optional(),
    sortBy: z.enum(['name', 'created_at', 'updated_at']).optional(),
    order: z.enum(['asc', 'desc']).optional(),
  }),
});

// Type inference
export type CreateUserInput = z.infer<typeof createUserSchema>['body'];
export type GetUserParams = z.infer<typeof getUserSchema>['params'];
export type GetUsersQuery = z.infer<typeof getUsersSchema>['query'];
```

**Validation middleware**:

```typescript
// src/middleware/validate.ts
import { Request, Response, NextFunction } from 'express';
import { AnyZodObject, ZodError } from 'zod';

export const validate = (schema: AnyZodObject) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    try {
      const validated = await schema.parseAsync({
        body: req.body,
        query: req.query,
        params: req.params,
      });

      // Replace with validated data (transforms applied)
      req.body = validated.body || req.body;
      req.query = validated.query || req.query;
      req.params = validated.params || req.params;

      next();
    } catch (error) {
      if (error instanceof ZodError) {
        return res.status(400).json({
          status: 'error',
          message: 'Validation failed',
          errors: error.errors.map(err => ({
            field: err.path.join('.'),
            message: err.message,
            code: err.code,
          })),
        });
      }
      next(error);
    }
  };
};
```

**Usage in routes**:

```typescript
import { Router } from 'express';
import { validate } from '../middleware/validate';
import * as schemas from '../schemas/userSchema';

const router = Router();

router.post('/', validate(schemas.createUserSchema), async (req, res) => {
  // req.body is now typed as CreateUserInput
  const user = await userService.create(req.body);
  res.status(201).json(user);
});

router.get('/:id', validate(schemas.getUserSchema), async (req, res) => {
  // req.params.id is validated
  const user = await userService.findById(req.params.id);
  if (!user) throw new NotFoundError('User not found');
  res.json(user);
});
```

## Database Connection Pooling

### PostgreSQL with pg

```typescript
// src/config/database.ts
import { Pool, PoolConfig } from 'pg';
import { logger } from './logger';

const config: PoolConfig = {
  host: process.env.DB_HOST || 'localhost',
  port: Number(process.env.DB_PORT) || 5432,
  database: process.env.DB_NAME,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  max: Number(process.env.DB_POOL_MAX) || 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
  statement_timeout: 30000, // 30s query timeout
};

export const pool = new Pool(config);

// Event handlers
pool.on('connect', (client) => {
  logger.debug('Database client connected');
});

pool.on('acquire', (client) => {
  logger.debug('Client acquired from pool');
});

pool.on('error', (err, client) => {
  logger.error('Unexpected pool error', { error: err });
  process.exit(-1);
});

// Health check
export const testConnection = async () => {
  try {
    const client = await pool.connect();
    const result = await client.query('SELECT NOW()');
    client.release();
    logger.info('Database connection successful', {
      serverTime: result.rows[0].now,
    });
  } catch (err) {
    logger.error('Database connection failed', { error: err });
    throw err;
  }
};

// Graceful shutdown
export const closePool = async () => {
  logger.info('Closing database pool');
  await pool.end();
  logger.info('Database pool closed');
};
```

### Transaction Helper

```typescript
// src/utils/transaction.ts
import { Pool, PoolClient } from 'pg';

export async function withTransaction<T>(
  pool: Pool,
  callback: (client: PoolClient) => Promise<T>
): Promise<T> {
  const client = await pool.connect();

  try {
    await client.query('BEGIN');
    const result = await callback(client);
    await client.query('COMMIT');
    return result;
  } catch (error) {
    await client.query('ROLLBACK');
    throw error;
  } finally {
    client.release();
  }
}

// Usage
import { pool } from '../config/database';

async function createUserWithProfile(userData, profileData) {
  return withTransaction(pool, async (client) => {
    const userResult = await client.query(
      'INSERT INTO users (email, name) VALUES ($1, $2) RETURNING id',
      [userData.email, userData.name]
    );
    const userId = userResult.rows[0].id;

    await client.query(
      'INSERT INTO profiles (user_id, bio) VALUES ($1, $2)',
      [userId, profileData.bio]
    );

    return userId;
  });
}
```

## Testing

### Integration Tests with Supertest

```typescript
// tests/integration/userRoutes.test.ts
import request from 'supertest';
import app from '../../src/app';
import { pool } from '../../src/config/database';

describe('User Routes', () => {
  beforeAll(async () => {
    await pool.query('CREATE TABLE IF NOT EXISTS users (...)');
  });

  afterEach(async () => {
    await pool.query('TRUNCATE TABLE users CASCADE');
  });

  afterAll(async () => {
    await pool.end();
  });

  describe('POST /api/users', () => {
    it('should create user with valid data', async () => {
      const response = await request(app)
        .post('/api/users')
        .send({
          email: 'test@example.com',
          name: 'Test User',
          password: 'Password123',
        })
        .expect(201);

      expect(response.body).toHaveProperty('id');
      expect(response.body.email).toBe('test@example.com');
      expect(response.body).not.toHaveProperty('password');
    });

    it('should return 400 for invalid email', async () => {
      const response = await request(app)
        .post('/api/users')
        .send({
          email: 'invalid',
          name: 'Test',
          password: 'Password123',
        })
        .expect(400);

      expect(response.body.status).toBe('error');
      expect(response.body.errors).toContainEqual(
        expect.objectContaining({
          field: 'body.email',
          message: expect.stringContaining('email'),
        })
      );
    });
  });

  describe('GET /api/users/:id', () => {
    it('should return user by ID', async () => {
      const createRes = await request(app)
        .post('/api/users')
        .send({
          email: 'test@example.com',
          name: 'Test User',
          password: 'Password123',
        });

      const response = await request(app)
        .get(`/api/users/${createRes.body.id}`)
        .expect(200);

      expect(response.body.id).toBe(createRes.body.id);
    });

    it('should return 404 for non-existent user', async () => {
      await request(app)
        .get('/api/users/99999')
        .expect(404);
    });
  });
});
```

### Unit Tests with Mocks

```typescript
// tests/unit/userService.test.ts
import { userService } from '../../src/services/userService';
import { pool } from '../../src/config/database';

jest.mock('../../src/config/database');

const mockPool = pool as jest.Mocked<typeof pool>;

describe('UserService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('findById', () => {
    it('should return user when found', async () => {
      mockPool.query.mockResolvedValue({
        rows: [{ id: 1, email: 'test@example.com', name: 'Test' }],
        command: 'SELECT',
        rowCount: 1,
        oid: 0,
        fields: [],
      });

      const result = await userService.findById('1');

      expect(result).toEqual(
        expect.objectContaining({ id: 1, email: 'test@example.com' })
      );
    });

    it('should return null when not found', async () => {
      mockPool.query.mockResolvedValue({
        rows: [],
        command: 'SELECT',
        rowCount: 0,
        oid: 0,
        fields: [],
      });

      const result = await userService.findById('999');
      expect(result).toBeNull();
    });
  });
});
```

## Production Deployment

### PM2 Configuration

```javascript
// ecosystem.config.js
module.exports = {
  apps: [{
    name: 'api',
    script: './dist/server.js',
    instances: 'max', // Use all CPU cores
    exec_mode: 'cluster',
    env: {
      NODE_ENV: 'production',
      PORT: 3000,
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    max_memory_restart: '500M',
    wait_ready: true,
    listen_timeout: 10000,
    kill_timeout: 5000,
  }],
};
```

**Graceful shutdown with PM2**:

```typescript
// src/server.ts
const server = app.listen(PORT, () => {
  logger.info(`Server started on port ${PORT}`);

  // Signal PM2 ready
  if (process.send) {
    process.send('ready');
  }
});

// Graceful shutdown
process.on('SIGINT', async () => {
  logger.info('SIGINT received, closing server');

  server.close(async () => {
    await closePool();
    logger.info('Server closed');
    process.exit(0);
  });

  // Force shutdown after 10s
  setTimeout(() => {
    logger.error('Forcing shutdown');
    process.exit(1);
  }, 10000);
});
```

### Dockerfile

```dockerfile
# Multi-stage build
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci

# Copy source
COPY src ./src

# Build TypeScript
RUN npm run build

# Production image
FROM node:18-alpine

WORKDIR /app

# Install production dependencies only
COPY package*.json ./
RUN npm ci --omit=dev && npm cache clean --force

# Copy built files
COPY --from=builder /app/dist ./dist

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

USER nodejs

EXPOSE 3000

CMD ["node", "dist/server.js"]
```

### Health Check Endpoint

```typescript
// src/routes/healthRoutes.ts
import { Router } from 'express';
import { pool } from '../config/database';

const router = Router();

router.get('/health', async (req, res) => {
  const health = {
    uptime: process.uptime(),
    message: 'OK',
    timestamp: Date.now(),
  };

  try {
    await pool.query('SELECT 1');
    health.database = 'connected';
  } catch (error) {
    health.database = 'disconnected';
    return res.status(503).json(health);
  }

  res.json(health);
});

router.get('/health/ready', async (req, res) => {
  // Readiness check
  try {
    await pool.query('SELECT 1');
    res.status(200).json({ status: 'ready' });
  } catch (error) {
    res.status(503).json({ status: 'not ready' });
  }
});

router.get('/health/live', (req, res) => {
  // Liveness check (simpler)
  res.status(200).json({ status: 'alive' });
});

export default router;
```

## Performance Optimization

### Response Caching

```typescript
import Redis from 'ioredis';

const redis = new Redis({
  host: process.env.REDIS_HOST,
  port: Number(process.env.REDIS_PORT),
});

export const cacheMiddleware = (duration: number) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    if (req.method !== 'GET') return next();

    const key = `cache:${req.originalUrl}`;

    try {
      const cached = await redis.get(key);
      if (cached) {
        return res.json(JSON.parse(cached));
      }

      // Capture response
      const originalJson = res.json.bind(res);
      res.json = (body: any) => {
        redis.setex(key, duration, JSON.stringify(body));
        return originalJson(body);
      };

      next();
    } catch (error) {
      next();
    }
  };
};

// Usage
router.get('/users', cacheMiddleware(300), async (req, res) => {
  const users = await userService.findAll();
  res.json(users);
});
```

## Security

### Rate Limiting

```typescript
import rateLimit from 'express-rate-limit';
import RedisStore from 'rate-limit-redis';
import Redis from 'ioredis';

const redis = new Redis();

export const apiLimiter = rateLimit({
  store: new RedisStore({ client: redis }),
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // 100 requests per window
  message: 'Too many requests, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});

export const authLimiter = rateLimit({
  store: new RedisStore({ client: redis }),
  windowMs: 15 * 60 * 1000,
  max: 5, // 5 attempts
  skipSuccessfulRequests: true,
});

// Usage
app.use('/api/', apiLimiter);
app.use('/api/auth/login', authLimiter);
```

## Anti-Patterns

| Anti-Pattern | Why Bad | Fix |
|--------------|---------|-----|
| **No async error handling** | Crashes server | Use asyncHandler or express-async-errors |
| **Inconsistent error responses** | Poor DX | Centralized error handler |
| **New DB connection per request** | Exhausts connections | Use connection pool |
| **No graceful shutdown** | Data loss, broken requests | Handle SIGTERM/SIGINT |
| **Logging to console in production** | Lost logs, no structure | Use Winston/Pino with transports |
| **No request validation** | Security vulnerabilities | Zod/express-validator |
| **Synchronous operations in routes** | Blocks event loop | Use async/await |
| **No health checks** | Can't monitor service | /health endpoints |

## Cross-References

**Related skills**:
- **Database patterns** → `database-integration` (pooling, transactions)
- **API testing** → `api-testing` (supertest patterns)
- **REST design** → `rest-api-design` (endpoint patterns)
- **Authentication** → `api-authentication` (JWT, sessions)

## Further Reading

- **Express docs**: https://expressjs.com/
- **Express.js Best Practices**: https://expressjs.com/en/advanced/best-practice-performance.html
- **Node.js Production Best Practices**: https://github.com/goldbergyoni/nodebestpractices
