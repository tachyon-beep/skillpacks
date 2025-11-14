---
name: graphql-api-design
description: Use when designing GraphQL schemas, solving N+1 query problems, implementing pagination/subscriptions, or choosing between REST and GraphQL - covers DataLoader, schema patterns, federation, performance optimization, and security
---

# GraphQL API Design

## Overview

**GraphQL API specialist covering schema design, query optimization, real-time subscriptions, federation, and production patterns.**

**Core principle**: GraphQL enables clients to request exactly the data they need in a single query - but requires careful schema design, batching strategies, and security measures to prevent performance and security issues.

## When to Use This Skill

Use when encountering:

- **N+1 query problems**: Too many database queries for nested resolvers
- **Schema design**: Types, interfaces, unions, input types, directives
- **Pagination**: Connections, cursors, offset patterns
- **Performance**: Query complexity, caching, batching, persisted queries
- **Real-time**: Subscriptions, WebSocket patterns, live queries
- **Federation**: Splitting schema across multiple services
- **Security**: Query depth limiting, cost analysis, allowlisting
- **Testing**: Schema validation, resolver testing, integration tests
- **Migrations**: Schema evolution, deprecation, versioning

**Do NOT use for**:
- REST API design → `rest-api-design`
- Framework-specific implementation → `fastapi-development`, `express-development`
- Microservices architecture → `microservices-architecture` (use with Federation)

## GraphQL vs REST Decision Matrix

| Factor | Choose GraphQL | Choose REST |
|--------|----------------|-------------|
| **Client needs** | Mobile apps, varying data needs | Uniform data requirements |
| **Over/under-fetching** | Problem | Not a problem |
| **Real-time features** | Subscriptions built-in | Need SSE/WebSockets separately |
| **Schema-first** | Strong typing required | Flexible, schema optional |
| **Caching** | Complex (field-level) | Simple (HTTP caching) |
| **File uploads** | Non-standard (multipart) | Native (multipart/form-data) |
| **Team expertise** | GraphQL experience | REST experience |
| **API consumers** | Known clients | Public/third-party |
| **Rate limiting** | Complex (field-level) | Simple (endpoint-level) |

**Hybrid approach**: GraphQL for internal/mobile, REST for public APIs

## Quick Reference - Core Patterns

| Pattern | Use Case | Key Concept |
|---------|----------|-------------|
| **DataLoader** | N+1 queries | Batch and cache within request |
| **Connection** | Pagination | Cursor-based with edges/nodes |
| **Union** | Heterogeneous results | Search, activity feeds |
| **Interface** | Shared fields | Polymorphic types with guarantees |
| **Directive** | Field behavior | @auth, @deprecated, custom logic |
| **Input types** | Mutations | Type-safe input validation |
| **Federation** | Microservices | Distributed schema composition |
| **Subscription** | Real-time | WebSocket-based live updates |

## N+1 Query Optimization

### The Problem

```javascript
// Schema
type Post {
  id: ID!
  title: String!
  author: User!  // Requires fetching user
}

type Query {
  posts: [Post!]!
}

// Naive resolver (N+1 problem)
const resolvers = {
  Query: {
    posts: () => db.posts.findAll()  // 1 query
  },
  Post: {
    author: (post) => db.users.findOne(post.authorId)  // N queries!
  }
};

// Result: 100 posts = 101 database queries
```

### DataLoader Solution

```javascript
const DataLoader = require('dataloader');

// Batch loading function
const batchUsers = async (userIds) => {
  const users = await db.users.findMany({
    where: { id: { in: userIds } }
  });

  // CRITICAL: Return in same order as requested IDs
  const userMap = new Map(users.map(u => [u.id, u]));
  return userIds.map(id => userMap.get(id) || null);
};

// Create loader per-request (avoid stale cache)
const createLoaders = () => ({
  user: new DataLoader(batchUsers),
  post: new DataLoader(batchPosts),
  // ... other loaders
});

// Add to context
const server = new ApolloServer({
  typeDefs,
  resolvers,
  context: () => ({
    loaders: createLoaders(),
    db,
    user: getCurrentUser()
  })
});

// Use in resolver
const resolvers = {
  Post: {
    author: (post, args, { loaders }) => {
      return loaders.user.load(post.authorId);  // Batched!
    }
  }
};
```

**Result**: 100 posts = 2 queries (1 for posts, 1 batched for unique authors)

### Advanced DataLoader Patterns

**Composite Keys**:

```javascript
// For multi-field lookups
const batchUsersByEmail = async (keys) => {
  // keys = [{domain: 'example.com', email: 'user@example.com'}, ...]
  const users = await db.users.findMany({
    where: {
      OR: keys.map(k => ({ email: k.email, domain: k.domain }))
    }
  });

  const userMap = new Map(
    users.map(u => [`${u.domain}:${u.email}`, u])
  );

  return keys.map(k => userMap.get(`${k.domain}:${k.email}`));
};

const userByEmailLoader = new DataLoader(batchUsersByEmail, {
  cacheKeyFn: (key) => `${key.domain}:${key.email}`
});
```

**Priming Cache**:

```javascript
// After fetching posts, prime user loader
const posts = await db.posts.findAll();
posts.forEach(post => {
  if (post.authorData) {
    loaders.user.prime(post.authorId, post.authorData);
  }
});
return posts;
```

**Error Handling in Batch**:

```javascript
const batchUsers = async (userIds) => {
  const users = await db.users.findMany({
    where: { id: { in: userIds } }
  });

  const userMap = new Map(users.map(u => [u.id, u]));

  return userIds.map(id => {
    const user = userMap.get(id);
    if (!user) {
      return new Error(`User ${id} not found`);  // Per-item error
    }
    return user;
  });
};
```

## Schema Design Patterns

### Interface vs Union

**Interface** (shared fields enforced):

```graphql
interface Node {
  id: ID!
}

interface Timestamped {
  createdAt: DateTime!
  updatedAt: DateTime!
}

type User implements Node & Timestamped {
  id: ID!
  createdAt: DateTime!
  updatedAt: DateTime!
  email: String!
  name: String!
}

type Post implements Node & Timestamped {
  id: ID!
  createdAt: DateTime!
  updatedAt: DateTime!
  title: String!
  content: String!
}

type Query {
  node(id: ID!): Node  # Can return any Node implementer
  nodes(ids: [ID!]!): [Node!]!
}
```

**Query**:
```graphql
{
  node(id: "user_123") {
    id
    ... on User {
      email
      name
    }
    ... on Post {
      title
    }
  }
}
```

**Union** (no shared fields required):

```graphql
union SearchResult = User | Post | Comment

type Query {
  search(query: String!): [SearchResult!]!
}
```

**When to use each**:

| Use Case | Pattern | Why |
|----------|---------|-----|
| Global ID lookup | Interface (Node) | Guarantees `id` field |
| Polymorphic lists with shared fields | Interface | Can query shared fields without fragments |
| Heterogeneous results | Union | No shared field requirements |
| Activity feeds | Union | Different event types |
| Search results | Union | Mixed content types |

### Input Types and Validation

```graphql
input CreatePostInput {
  title: String!
  content: String!
  tags: [String!]
  publishedAt: DateTime
}

input UpdatePostInput {
  title: String
  content: String
  tags: [String!]
}

type Mutation {
  createPost(input: CreatePostInput!): Post!
  updatePost(id: ID!, input: UpdatePostInput!): Post!
}
```

**Benefits**:
- Reusable across multiple mutations
- Clear separation of create vs update requirements
- Type-safe in generated code
- Can add descriptions per field

### Custom Directives

```graphql
directive @auth(requires: Role = USER) on FIELD_DEFINITION
directive @rateLimit(limit: Int!, window: Int!) on FIELD_DEFINITION
directive @deprecated(reason: String) on FIELD_DEFINITION | ENUM_VALUE

enum Role {
  USER
  ADMIN
  SUPER_ADMIN
}

type Query {
  publicData: String
  userData: User @auth(requires: USER)
  adminData: String @auth(requires: ADMIN)
  expensiveQuery: Result @rateLimit(limit: 10, window: 60)
}

type User {
  id: ID!
  email: String! @auth(requires: USER)  # Only authenticated users
  internalId: String @deprecated(reason: "Use `id` instead")
}
```

## Pagination Patterns

### Relay Connection Specification

**Standard connection pattern**:

```graphql
type PostConnection {
  edges: [PostEdge!]!
  pageInfo: PageInfo!
  totalCount: Int  # Optional
}

type PostEdge {
  node: Post!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

type Query {
  posts(
    first: Int
    after: String
    last: Int
    before: String
  ): PostConnection!
}
```

**Implementation**:

```javascript
const resolvers = {
  Query: {
    posts: async (parent, { first, after, last, before }) => {
      const limit = first || last || 10;
      const cursor = after || before;

      // Decode cursor
      const offset = cursor ? decodeCursor(cursor) : 0;

      // Fetch one extra to determine hasNextPage
      const posts = await db.posts.findMany({
        skip: offset,
        take: limit + 1,
        orderBy: { createdAt: 'desc' }
      });

      const hasNextPage = posts.length > limit;
      const edges = posts.slice(0, limit).map((post, index) => ({
        node: post,
        cursor: encodeCursor(offset + index)
      }));

      return {
        edges,
        pageInfo: {
          hasNextPage,
          hasPreviousPage: offset > 0,
          startCursor: edges[0]?.cursor,
          endCursor: edges[edges.length - 1]?.cursor
        }
      };
    }
  }
};

// Opaque cursor encoding
const encodeCursor = (offset) =>
  Buffer.from(`arrayconnection:${offset}`).toString('base64');
const decodeCursor = (cursor) =>
  parseInt(Buffer.from(cursor, 'base64').toString().split(':')[1]);
```

**Alternative: Offset pagination** (simpler but less robust):

```graphql
type PostPage {
  items: [Post!]!
  total: Int!
  page: Int!
  pageSize: Int!
}

type Query {
  posts(page: Int = 1, pageSize: Int = 20): PostPage!
}
```

## Performance Optimization

### Query Complexity Analysis

**Prevent expensive queries**:

```javascript
const depthLimit = require('graphql-depth-limit');
const { createComplexityLimitRule } = require('graphql-validation-complexity');

const server = new ApolloServer({
  typeDefs,
  resolvers,
  validationRules: [
    depthLimit(10),  // Max 10 levels deep
    createComplexityLimitRule(1000, {
      scalarCost: 1,
      objectCost: 2,
      listFactor: 10
    })
  ]
});
```

**Custom complexity**:

```graphql
type Query {
  posts(first: Int!): [Post!]! @cost(complexity: 10, multipliers: ["first"])
  expensiveAnalytics: AnalyticsReport! @cost(complexity: 1000)
}
```

### Automatic Persisted Queries (APQ)

**Client sends hash instead of full query**:

```javascript
// Client
const query = gql`
  query GetUser($id: ID!) {
    user(id: $id) { name email }
  }
`;

const queryHash = sha256(query);

// First request: Send hash only
fetch('/graphql', {
  body: JSON.stringify({
    extensions: {
      persistedQuery: {
        version: 1,
        sha256Hash: queryHash
      }
    },
    variables: { id: '123' }
  })
});

// If server doesn't have it (PersistedQueryNotFound)
// Second request: Send full query + hash
fetch('/graphql', {
  body: JSON.stringify({
    query,
    extensions: {
      persistedQuery: {
        version: 1,
        sha256Hash: queryHash
      }
    },
    variables: { id: '123' }
  })
});

// Future requests: Just send hash
```

**Benefits**:
- Reduced bandwidth (hash << full query)
- CDN caching of GET requests
- Query allowlisting (if configured)

### Field-Level Caching

```javascript
const resolvers = {
  Query: {
    user: async (parent, { id }, { cache }) => {
      const cacheKey = `user:${id}`;
      const cached = await cache.get(cacheKey);
      if (cached) return JSON.parse(cached);

      const user = await db.users.findOne(id);
      await cache.set(cacheKey, JSON.stringify(user), { ttl: 300 });
      return user;
    }
  }
};
```

## Subscriptions (Real-Time)

### Basic Subscription

```graphql
type Subscription {
  postAdded: Post!
  commentAdded(postId: ID!): Comment!
}

type Mutation {
  createPost(input: CreatePostInput!): Post!
}
```

**Implementation (Apollo Server)**:

```javascript
const { PubSub } = require('graphql-subscriptions');
const pubsub = new PubSub();

const resolvers = {
  Mutation: {
    createPost: async (parent, { input }) => {
      const post = await db.posts.create(input);
      pubsub.publish('POST_ADDED', { postAdded: post });
      return post;
    }
  },
  Subscription: {
    postAdded: {
      subscribe: () => pubsub.asyncIterator(['POST_ADDED'])
    },
    commentAdded: {
      subscribe: (parent, { postId }) =>
        pubsub.asyncIterator([`COMMENT_ADDED_${postId}`])
    }
  }
};

// Client
subscription {
  postAdded {
    id
    title
    author { name }
  }
}
```

### Scaling Subscriptions

**Problem**: In-memory PubSub doesn't work across servers

**Solution**: Redis PubSub

```javascript
const { RedisPubSub } = require('graphql-redis-subscriptions');
const Redis = require('ioredis');

const pubsub = new RedisPubSub({
  publisher: new Redis(),
  subscriber: new Redis()
});

// Now works across multiple server instances
```

### Subscription Authorization

```javascript
const resolvers = {
  Subscription: {
    secretDataUpdated: {
      subscribe: withFilter(
        () => pubsub.asyncIterator(['SECRET_DATA']),
        (payload, variables, context) => {
          // Only admin users can subscribe
          return context.user?.role === 'ADMIN';
        }
      )
    }
  }
};
```

## Federation (Distributed Schema)

**Split schema across multiple services**:

### User Service

```graphql
# user-service schema
type User @key(fields: "id") {
  id: ID!
  email: String!
  name: String!
}

type Query {
  user(id: ID!): User
}
```

### Post Service

```graphql
# post-service schema
extend type User @key(fields: "id") {
  id: ID! @external
  posts: [Post!]!
}

type Post {
  id: ID!
  title: String!
  content: String!
  authorId: ID!
  author: User!
}
```

### Gateway

Composes schemas and routes requests:

```javascript
const { ApolloGateway } = require('@apollo/gateway');

const gateway = new ApolloGateway({
  serviceList: [
    { name: 'users', url: 'http://user-service:4001/graphql' },
    { name: 'posts', url: 'http://post-service:4002/graphql' }
  ]
});

const server = new ApolloServer({
  gateway,
  subscriptions: false  // Not yet supported in federation
});
```

**Reference Resolver** (fetch extended fields):

```javascript
// post-service resolvers
const resolvers = {
  User: {
    __resolveReference: async (user) => {
      // Receive { __typename: 'User', id: '123' }
      // Don't need to fetch user, just return it for field resolution
      return user;
    },
    posts: async (user) => {
      return db.posts.findMany({ where: { authorId: user.id } });
    }
  }
};
```

## Security Patterns

### Query Depth Limiting

```javascript
const depthLimit = require('graphql-depth-limit');

const server = new ApolloServer({
  validationRules: [depthLimit(7)]  // Max 7 levels deep
});

// Prevents: user { posts { author { posts { author { ... } } } }
```

### Query Allowlisting (Production)

```javascript
const allowedQueries = new Map([
  ['GetUser', 'query GetUser($id: ID!) { user(id: $id) { name } }'],
  ['ListPosts', 'query ListPosts { posts { title } }']
]);

const server = new ApolloServer({
  validationRules: [
    (context) => ({
      Document(node) {
        const queryName = node.definitions[0]?.name?.value;
        if (!allowedQueries.has(queryName)) {
          context.reportError(
            new GraphQLError('Query not allowed')
          );
        }
      }
    })
  ]
});
```

### Rate Limiting (Field-Level)

```javascript
const { shield, rule, and } = require('graphql-shield');

const isRateLimited = rule({ cache: 'contextual' })(
  async (parent, args, ctx, info) => {
    const key = `rate:${ctx.user.id}:${info.fieldName}`;
    const count = await redis.incr(key);
    if (count === 1) {
      await redis.expire(key, 60);  // 1 minute window
    }
    return count <= 10;  // 10 requests per minute
  }
);

const permissions = shield({
  Query: {
    expensiveQuery: isRateLimited
  }
});
```

## Schema Evolution

### Deprecation

```graphql
type User {
  id: ID!
  username: String @deprecated(reason: "Use `name` instead")
  name: String!
}
```

**Tooling shows warnings to clients**

### Breaking Changes (Avoid)

❌ **Breaking**:
- Removing fields
- Changing field types
- Making nullable → non-nullable
- Removing enum values
- Changing arguments

✅ **Non-breaking**:
- Adding fields
- Adding types
- Deprecating fields
- Making non-nullable → nullable
- Adding arguments with defaults

### Versioning Strategy

**Don't version schema** - evolve incrementally:

1. Add new field
2. Deprecate old field
3. Monitor usage
4. Remove old field in next major version (if removing)

## Testing Strategies

### Schema Validation

```javascript
const { buildSchema, validateSchema } = require('graphql');

test('schema is valid', () => {
  const schema = buildSchema(typeDefs);
  const errors = validateSchema(schema);
  expect(errors).toHaveLength(0);
});
```

### Resolver Testing

```javascript
const resolvers = require('./resolvers');

test('user resolver fetches user', async () => {
  const mockDb = {
    users: { findOne: jest.fn().mockResolvedValue({ id: '1', name: 'Alice' }) }
  };

  const result = await resolvers.Query.user(
    null,
    { id: '1' },
    { db: mockDb, loaders: { user: mockDataLoader() } }
  );

  expect(result).toEqual({ id: '1', name: 'Alice' });
  expect(mockDb.users.findOne).toHaveBeenCalledWith('1');
});
```

### Integration Testing

```javascript
const { ApolloServer } = require('apollo-server');
const { createTestClient } = require('apollo-server-testing');

const server = new ApolloServer({ typeDefs, resolvers });
const { query } = createTestClient(server);

test('GetUser query', async () => {
  const GET_USER = gql`
    query GetUser($id: ID!) {
      user(id: $id) {
        name
        email
      }
    }
  `;

  const res = await query({ query: GET_USER, variables: { id: '1' } });

  expect(res.errors).toBeUndefined();
  expect(res.data.user).toMatchObject({
    name: 'Alice',
    email: 'alice@example.com'
  });
});
```

## Anti-Patterns

| Anti-Pattern | Why Bad | Fix |
|--------------|---------|-----|
| **No DataLoader** | N+1 queries kill performance | Use DataLoader for all entity fetching |
| **Offset pagination** | Breaks with real-time data | Use cursor-based connections |
| **No query complexity** | DoS via deeply nested queries | Set depth/complexity limits |
| **Shared DataLoader instances** | Stale cache across requests | Create new loaders per request |
| **No error masking** | Leaks internal errors to clients | Mask in production, log internally |
| **mutations returning Boolean** | Can't extend response | Return object type |
| **Nullable IDs** | IDs should never be null | Use `ID!` not `ID` |
| **Over-fetching in resolvers** | Selecting * wastes bandwidth | Select only requested fields |

## Common Mistakes

### 1. DataLoader Return Order

```javascript
// ❌ WRONG - Returns in database order
const batchUsers = async (ids) => {
  return await db.users.findMany({ where: { id: { in: ids } } });
};

// ✅ CORRECT - Returns in requested order
const batchUsers = async (ids) => {
  const users = await db.users.findMany({ where: { id: { in: ids } } });
  const userMap = new Map(users.map(u => [u.id, u]));
  return ids.map(id => userMap.get(id));
};
```

### 2. Mutations Returning Primitives

```graphql
# ❌ BAD - Can't extend
type Mutation {
  deletePost(id: ID!): Boolean!
}

# ✅ GOOD - Extensible
type DeletePostPayload {
  success: Boolean!
  deletedPostId: ID
  message: String
}

type Mutation {
  deletePost(id: ID!): DeletePostPayload!
}
```

### 3. No Context in Subscriptions

```javascript
// ❌ Missing auth context
const server = new ApolloServer({
  subscriptions: {
    onConnect: () => {
      return {};  // No user context!
    }
  }
});

// ✅ Include auth
const server = new ApolloServer({
  subscriptions: {
    onConnect: (connectionParams) => {
      const token = connectionParams.authToken;
      const user = verifyToken(token);
      return { user };
    }
  }
});
```

## Tooling Ecosystem

**Schema Management**:
- **Apollo Studio**: Schema registry, operation tracking, metrics
- **GraphQL Inspector**: Schema diffing, breaking change detection
- **Graphql-eslint**: Linting for schema and queries

**Code Generation**:
- **GraphQL Code Generator**: TypeScript types from schema
- **Apollo Codegen**: Client types for queries

**Development**:
- **GraphiQL**: In-browser IDE
- **Apollo Sandbox**: Modern GraphQL explorer
- **Altair**: Desktop GraphQL client

**Testing**:
- **EasyGraphQL Test**: Schema mocking
- **GraphQL Tools**: Schema stitching, mocking

## Cross-References

**Related skills**:
- **REST comparison** → `rest-api-design` (when to use each)
- **FastAPI implementation** → `fastapi-development` (Strawberry, Graphene)
- **Express implementation** → `express-development` (Apollo Server, GraphQL Yoga)
- **Microservices** → `microservices-architecture` (use with Federation)
- **Security** → `ordis-security-architect` (OWASP API Security)
- **Testing** → `api-testing` (integration testing strategies)
- **Authentication** → `api-authentication` (JWT, OAuth2 with GraphQL)

## Further Reading

- **GraphQL Spec**: https://spec.graphql.org/
- **Apollo Docs**: Federation, caching, tooling
- **Relay Spec**: Connection specification
- **DataLoader GitHub**: facebook/dataloader
- **Production Ready GraphQL**: Book by Marc-André Giroux
