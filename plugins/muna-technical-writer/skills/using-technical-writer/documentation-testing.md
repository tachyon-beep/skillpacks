
# Documentation Testing

## Overview

Test documentation like you test code. Core principle: **If you haven't tried it, it's broken**.

**Key insight**: Untested documentation always has issues. Copy-paste-run test finds them before users do.

## When to Use

Load this skill when:
- Finalizing documentation before release
- Reviewing documentation quality
- Creating documentation quality gates
- After major doc updates

**Symptoms you need this**:
- "Is this documentation good enough to ship?"
- Preparing installation guides, quick starts, tutorials
- Documentation quality review
- Pre-release documentation checklist

**Don't use for**:
- Writing documentation (use `muna/technical-writer/clarity-and-style`)
- Structuring documentation (use `muna/technical-writer/documentation-structure`)

## Five Testing Dimensions

Test documentation across 5 dimensions:

### 1. Completeness Testing
**Question**: Can reader accomplish the task with ONLY this documentation?

### 2. Accuracy Testing
**Question**: Do all examples, commands, and instructions actually work?

### 3. Findability Testing
**Question**: Can users find this documentation when they need it?

### 4. Example Verification
**Question**: Can you copy-paste every example and have it run without modification?

### 5. Walkthrough Testing
**Question**: Can a new user follow this successfully on a clean system?


## Dimension 1: Completeness Testing

**Goal**: Verify reader can complete task without external resources.

### Checklist

- [ ] **All prerequisites listed** - OS, language versions, required accounts, tools
- [ ] **All configuration options documented** - Every setting, not just defaults
- [ ] **Error cases covered** - What can go wrong and how to fix
- [ ] **Troubleshooting section** - Common issues with solutions
- [ ] **Success criteria** - "How do I know it worked?"
- [ ] **Next steps** - What to do after completing this doc

### Example: Installation Guide

❌ **Incomplete**:
```markdown
## Installation

Run:
\`\`\`bash
npm install our-app
\`\`\`

You're done!
```

**Missing**: Prerequisites, error handling, verification

✅ **Complete**:
```markdown
## Installation

### Prerequisites
- Node.js 18+ (`node --version` to check)
- npm 9+ (`npm --version` to check)
- Active internet connection

### Install

\`\`\`bash
npm install our-app
\`\`\`

### Verify Installation

Check installed version:
\`\`\`bash
npx our-app --version
# Expected output: our-app v2.1.0
\`\`\`

### Troubleshooting

**Error: "EACCES: permission denied"**
- Solution: Run with sudo: `sudo npm install -g our-app`

**Error: "Unsupported engine"**
- Solution: Upgrade Node.js to 18+

### Next Steps
- [Quick Start Guide](./quick-start.md)
- [Configuration Reference](./config.md)
```

### Testing Method

**The "Clean Slate Test"**:
1. Can someone with ZERO context complete this?
2. Read doc, follow instructions
3. Note every moment you had to Google or guess


## Dimension 2: Accuracy Testing

**Goal**: Verify all examples, commands, and instructions work.

### Checklist

- [ ] **Code examples run** - Copy-paste into environment, executes without errors
- [ ] **Commands correct** - No typos, correct options, work on stated OS
- [ ] **Version numbers current** - Not referencing outdated versions
- [ ] **Screenshots up-to-date** - Match current UI
- [ ] **Links work** - No 404s, links go to correct pages
- [ ] **Output matches examples** - Documented output = actual output

### Example: API Documentation

❌ **Inaccurate**:
```markdown
Make a request:
\`\`\`bash
curl https://api.example.com/users
\`\`\`

Returns user list.
```

**Issues**: No authentication, vague output

✅ **Accurate**:
```markdown
Make a request:
\`\`\`bash
curl -H "Authorization: Bearer YOUR_API_KEY" \\
     https://api.example.com/v1/users
\`\`\`

Response:
\`\`\`json
{
  "users": [
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"}
  ],
  "total": 2
}
\`\`\`

Status: 200 OK
```

### Testing Method

**Copy-Paste-Run Test**:
1. Copy EVERY code example
2. Paste into clean environment
3. Run without modifications
4. Verify output matches documentation
5. If ANY example fails, documentation is inaccurate


## Dimension 3: Findability Testing

**Goal**: Verify users can find documentation when needed.

### Checklist

- [ ] **Keywords present** - Terms users would search for
- [ ] **Linked from related pages** - Cross-references in both directions
- [ ] **In navigation/TOC** - Appears in sidebar, index, sitemap
- [ ] **Search engine optimized** - Title, headers, meta description
- [ ] **Clear title** - Describes content accurately

### Example: Deployment Guide

❌ **Not Findable**:
```markdown
# Guide

Deploy the app...
```

**Issues**: Generic title, no keywords

✅ **Findable**:
```markdown
# Deploying to AWS ECS with Docker (Production)

**Keywords**: AWS, ECS, Fargate, Docker, deployment, production, continuous deployment, CI/CD

Deploy our application to AWS Elastic Container Service (ECS) using Docker containers...

**Related**:
- [Docker Configuration](./docker.md)
- [CI/CD Pipeline](./cicd.md)
- [Environment Variables](./env-vars.md)
```

### Testing Method

**Search Simulation**:
1. What would user search for? ("deploy to AWS", "ECS deployment", "Docker production")
2. Search your docs with those terms
3. Does this page appear in top 3 results?


## Dimension 4: Example Verification

**Goal**: All examples work without modification.

### Checklist

- [ ] **Examples are complete** - Include all necessary imports, setup
- [ ] **No placeholders without explanation** - If using `YOUR_API_KEY`, explain how to get it
- [ ] **Environment specified** - Language version, OS, dependencies
- [ ] **Async/await correct** - Don't forget await on promises
- [ ] **Error handling shown** - Not just happy path

### Example: Code Example

❌ **Unverified**:
```javascript
const users = client.get('/users');
console.log(users);
```

**Issues**: Missing await, won't work

✅ **Verified**:
```javascript
// Prerequisites: npm install api-client
// Environment: Node.js 18+

import APIClient from 'api-client';

// Get API key from https://dashboard.example.com/settings
const client = new APIClient(process.env.API_KEY);

async function getUsers() {
  try {
    const users = await client.get('/users');
    console.log('Users:', users);
    // Output: Users: [{id: 1, name: 'Alice'}, {id: 2, name: 'Bob'}]
  } catch (error) {
    console.error('Failed to fetch users:', error.message);
  }
}

getUsers();
```

### Testing Method

**Literal Copy-Paste Test**:
1. Copy example
2. Create new file
3. Paste (no modifications)
4. Run
5. Does it work? If not, example is broken.


## Dimension 5: Walkthrough Testing

**Goal**: New user can follow successfully on clean system.

### Checklist

- [ ] **Test on clean system** - Fresh VM/container, not your dev machine
- [ ] **Follow every step literally** - No shortcuts, no cached knowledge
- [ ] **Note confusion points** - Every time you have to guess or Google
- [ ] **Verify timing claims** - "5 minute setup" actually takes 5 minutes?
- [ ] **Test with beginner** - Colleague unfamiliar with project

### Example: Quick Start

❌ **Untested**:
```markdown
# Quick Start (5 minutes)

1. Install the CLI
2. Configure your credentials
3. Deploy your first app

Done!
```

**Issues**: Vague steps, unverified timing

✅ **Walkthrough-Tested**:
```markdown
# Quick Start (15 minutes)

## Prerequisites (2 min)
- [ ] Ubuntu 22.04 or macOS 12+
- [ ] 2GB free disk space
- [ ] Internet connection

## Step 1: Install CLI (5 min)

\`\`\`bash
curl -L https://releases.example.com/cli.sh | bash
\`\`\`

Verify:
\`\`\`bash
our-cli --version
# Should output: our-cli v2.1.0
\`\`\`

**Troubleshooting**: If command not found, close and reopen terminal.

## Step 2: Configure Credentials (3 min)

Get API key: https://dashboard.example.com/settings

\`\`\`bash
our-cli auth login YOUR_API_KEY_HERE
\`\`\`

Success message: "✓ Authenticated as user@example.com"

## Step 3: Deploy First App (5 min)

\`\`\`bash
mkdir my-app && cd my-app
our-cli init
our-cli deploy
\`\`\`

**Success criteria**: URL shown: "✓ Deployed to https://my-app-abc123.example.com"

Visit URL in browser - you should see "Hello World"

## Next Steps
- [Add custom domain](./custom-domains.md)
- [Configure environment variables](./env-vars.md)
```

### Testing Method

**New User Walkthrough**:
1. Spin up clean VM/container
2. Follow guide step-by-step as written
3. Don't use any cached knowledge
4. Note EVERY point of confusion
5. Time how long it actually takes

**Or**: Give to colleague who's never used the product. Watch them follow it. Note every question they ask.


## Testing Workflow

Use this workflow before releasing documentation:

### Phase 1: Quick Checks (10 min)

- [ ] Read through once - obvious errors?
- [ ] Check all links - do they work?
- [ ] Scan for placeholders - any unexplained `YOUR_X_HERE`?
- [ ] Verify versions - are version numbers current?

### Phase 2: Example Verification (30 min)

- [ ] Copy EVERY code example
- [ ] Paste into clean environment
- [ ] Run without modifications
- [ ] Verify output matches docs

### Phase 3: Completeness Check (15 min)

- [ ] Prerequisites listed?
- [ ] Error cases covered?
- [ ] Troubleshooting section?
- [ ] Success criteria ("how do I know it worked")?

### Phase 4: Walkthrough Test (60 min)

- [ ] Fresh VM/container
- [ ] Follow as new user
- [ ] Note confusion points
- [ ] Verify timing claims

### Phase 5: Findability Check (10 min)

- [ ] Search docs with user keywords
- [ ] Check cross-references
- [ ] Verify in navigation/TOC

**Total time**: ~2 hours for thorough documentation testing


## Common Issues Found By Testing

### Issue: Async/Await Missing

**Example**:
```javascript
const data = api.get('/endpoint'); // Missing await
console.log(data); // Prints Promise object, not data
```

**Found by**: Copy-paste-run test


### Issue: Prerequisites Not Listed

**Example**:
```markdown
Run: `docker-compose up`
```

**Missing**: Docker installed, docker-compose.yml file exists

**Found by**: Clean system walkthrough


### Issue: Environment Variables Not Explained

**Example**:
```javascript
const key = process.env.API_KEY; // How do I set this?
```

**Found by**: New user walkthrough (where do I get API key?)


### Issue: Timing Claims Unverified

**Example**: "Setup in 5 minutes" actually takes 20 minutes (npm install, account creation, key generation)

**Found by**: Walkthrough testing with timer


### Issue: Success Criteria Missing

**Example**:
```markdown
Deploy your app:
\`\`\`bash
deploy.sh
\`\`\`
```

**Missing**: How do I know it worked? What URL? What should I see?

**Found by**: Completeness testing


## Documentation Testing Report Template

```markdown
# Documentation Testing Report

**Document**: [Quick Start Guide / API Reference / etc.]
**Tester**: [Name]
**Date**: [Date]
**Environment**: [Clean Ubuntu 22.04 VM / macOS 13 / etc.]

## Test Results

### Completeness ✅ / ❌
- [ ] Prerequisites listed
- [ ] Error cases covered
- [ ] Troubleshooting included
- [ ] Success criteria present

**Issues Found**: [List any gaps]

### Accuracy ✅ / ❌
- [ ] All code examples run
- [ ] All commands correct
- [ ] Links work
- [ ] Output matches docs

**Issues Found**: [List inaccuracies]

### Findability ✅ / ❌
- [ ] Keywords present
- [ ] Linked from related pages
- [ ] In navigation

**Issues Found**: [List findability gaps]

### Examples ✅ / ❌
**Copy-Paste-Run Results**:
- Example 1: ✅ Works / ❌ Failed - [error]
- Example 2: ✅ Works / ❌ Failed - [error]

### Walkthrough ✅ / ❌
**Confusion Points**: [List every point where you had to guess or Google]
**Actual Time**: [X minutes] vs Claimed: [Y minutes]
**Success**: ✅ Completed task / ❌ Got stuck at step X

## Recommendations
1. [Fix async/await in example 2]
2. [Add prerequisites section]
3. [Update timing claim from 5 to 15 minutes]

## Overall: Ready for Release ✅ / Needs Work ❌
```


## Quick Reference: Testing Checklist

| Dimension | Key Question | Quick Test |
|-----------|--------------|------------|
| **Completeness** | Can task be done with ONLY this doc? | List everything needed - is it in the doc? |
| **Accuracy** | Do examples run? | Copy-paste every example, run it |
| **Findability** | Can users find this? | Search with user keywords - does it appear? |
| **Examples** | Copy-paste-run works? | Literally copy-paste, no modifications, run |
| **Walkthrough** | Does it work for new user? | Fresh VM, follow as beginner, time it |


## Cross-References

**Use BEFORE this skill**:
- `muna/technical-writer/clarity-and-style` - Write clear docs
- `muna/technical-writer/documentation-structure` - Structure docs properly

**Use AFTER this skill**:
- Fix issues found, then re-test

## Real-World Impact

**Documentation tested with this framework:**
- **Quick start guide testing caught async/await bug** in 3/5 examples (would have broken for every user)
- **Walkthrough testing revealed "5 minute setup" actually took 22 minutes** (including account creation and key generation not mentioned in docs)
- **Copy-paste-run test found missing `import` statement** that prevented example from running (developer's IDE auto-imported it)
- **Clean system test revealed missing prerequisite** (Docker Compose not documented) that blocked 40% of users

**Key lesson**: **Untested documentation always has issues. 2 hours of testing prevents weeks of user confusion and support tickets.**
