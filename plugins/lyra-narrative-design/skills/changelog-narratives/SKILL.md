---
name: changelog-narratives
description: Use when maintaining CHANGELOG.md or version history - transforms technical change lists into narrative progression that helps developers understand evolution, breaking changes, and upgrade paths
---

# Changelog Narratives

## Overview

**Changelogs tell the story of your project's evolution.** They're not just commit logs—they're a narrative of how your software grew, what problems you solved, and where it's heading.

**Core principle:** Developers read changelogs to understand "what changed and how does it affect me?" Answer that question clearly, and you build trust.

**Key insight:** Good changelogs reduce support burden (fewer "how do I upgrade?" issues), increase adoption (confident upgrades), and document decision-making for future maintainers.

## When to Use

**Use this skill when:**
- Creating or maintaining CHANGELOG.md
- Preparing version releases
- Documenting breaking changes
- Explaining deprecations
- Writing upgrade guides
- Managing semantic versioning communication

**Symptoms you need this:**
- "Users fear upgrading" (breaking changes unclear)
- "Support tickets spike after releases"
- "CHANGELOG is auto-generated commit dump"
- "No one reads release history"
- "Deprecations surprise users"

**Don't use when:**
- Writing release notes for non-developers (use `release-notes-that-resonate`)
- Internal changelogs that never leave team (lower stakes, different audience)
- Documentation changes that belong in docs, not changelog

## Changelog Philosophy

### What Belongs in a Changelog

**Include:**
- ✅ **Added** - New features
- ✅ **Changed** - Changes to existing functionality
- ✅ **Deprecated** - Soon-to-be-removed features
- ✅ **Removed** - Removed features
- ✅ **Fixed** - Bug fixes
- ✅ **Security** - Security patches
- ✅ **Breaking** - Breaking changes (critical!)

**Exclude:**
- ❌ Internal refactoring (unless it affects users)
- ❌ Dependency updates (unless it fixes user-facing issue)
- ❌ Documentation fixes (minor typos)
- ❌ Development tooling changes

**Test:** "Does this affect how users interact with my software?" → Include. Otherwise, skip.

---

## Standard Format: Keep a Changelog

**Follow [keepachangelog.com](https://keepachangelog.com) structure:**

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Feature X for use case Y

### Changed
- Behavior Z now defaults to W for better performance

## [2.0.0] - 2025-01-15

### Breaking
- Removed deprecated `oldMethod()` - use `newMethod()` instead
- Changed `config.timeout` from seconds to milliseconds

### Added
- Real-time subscriptions via WebSocket
- New query builder API

### Fixed
- Cache invalidation race condition (#234)
- Memory leak in connection pooling (#256)

## [1.5.0] - 2024-12-01
...
```

**Why this format works:**
- **Scannable**: Categories make it easy to find relevant changes
- **Chronological**: Latest changes first
- **Semantic versioning aligned**: Version numbers match change severity
- **Predictable**: Users know what to expect

---

## Writing Effective Changelog Entries

### Pattern: Impact + Context + Action

```markdown
❌ BAD (commit message dump):
- Updated authentication

❌ BAD (vague):
- Improved performance

✅ GOOD (impact + context + action):
- **Performance**: Reduced API response time by 60% by implementing
  request-level caching. Upgrade: no code changes required.

✅ GOOD (breaking change with migration):
- **Breaking**: `authenticate(token)` now returns `Promise<User>` instead
  of `User` to support async validation. Migration: add `await` before calls.
  See [migration guide](link).
```

**Formula:**
1. **Category tag** (Added/Changed/Fixed/Breaking)
2. **Impact** (what users experience)
3. **Context** (why this changed)
4. **Action** (what users need to do, if anything)

---

### Handling Breaking Changes

**Breaking changes are the most critical changelog entries.** Handle with care:

```markdown
## [2.0.0] - 2025-01-15

### Breaking Changes

#### Removed `cache.clear()` in favor of `cache.invalidate()`

**Why**: `clear()` was ambiguous—did it clear local cache or distributed
cache? This caused production incidents.

**Migration**:
```typescript
// Before (v1.x)
cache.clear();

// After (v2.x)
cache.invalidate();  // Clears local cache
cache.invalidateAll();  // Clears distributed cache
```

**Affected users**: Anyone calling `cache.clear()`. Compiler will show
error at upgrade.

**Timeline**: `clear()` was deprecated in v1.8.0 (June 2024), removed in v2.0.0.

---

#### Changed `timeout` config from seconds to milliseconds

**Why**: Milliseconds give finer control and align with industry standards
(Node.js setTimeout, etc.).

**Migration**:
```typescript
// Before (v1.x)
{ timeout: 5 }  // 5 seconds

// After (v2.x)
{ timeout: 5000 }  // 5000 milliseconds = 5 seconds
```

**Affected users**: Anyone setting custom `timeout` values.

**Detection**: v2.0.0 logs warnings if timeout values look suspiciously
small (< 100), suggesting you may need to update.
```

**Breaking change checklist:**
- [ ] Explain WHY breaking change was necessary
- [ ] Provide before/after code examples
- [ ] Link to migration guide (if complex)
- [ ] Specify who's affected
- [ ] Mention deprecation timeline (if applicable)
- [ ] Describe detection/error messages users will see

---

### Writing Deprecation Warnings

**Deprecations are "polite breaking changes"—give users time to migrate.**

```markdown
## [1.8.0] - 2024-06-01

### Deprecated

#### `cache.clear()` deprecated in favor of `cache.invalidate()`

**Reason**: Ambiguous behavior led to production incidents. New methods
make local vs distributed cache operations explicit.

**Timeline**:
- v1.8.0 (now): `clear()` works but logs deprecation warning
- v1.9.0 (Aug 2024): Warning becomes more prominent
- v2.0.0 (Dec 2024): `clear()` removed entirely

**Migration**:
```typescript
// Replace this:
cache.clear();

// With one of these:
cache.invalidate();     // Local cache only
cache.invalidateAll();  // Distributed cache
```

**Tracking**: Run `npm run check-deprecations` to find all uses in your code.
```

**Deprecation best practices:**
- ✅ Give 6-12 months notice for major features
- ✅ Provide automated detection (linter rules, runtime warnings)
- ✅ Show migration path clearly
- ✅ Explain the "why" (builds trust)
- ❌ Deprecate and remove in same version
- ❌ Deprecate without replacement

---

## Narrative Techniques for Changelogs

### Technique 1: Group Related Changes

**Instead of flat list:**
```markdown
❌ HARD TO SCAN:
### Changed
- Updated WebSocket library
- Modified connection timeout handling
- Changed reconnection logic
- Adjusted ping/pong intervals
```

**Use thematic grouping:**
```markdown
✅ EASY TO SCAN:
### Changed

**Real-time connection reliability improvements:**
- Reduced WebSocket reconnection time from 5s to 1s
- Added exponential backoff for connection retries
- Improved timeout handling for flaky networks
- Upgraded WebSocket library to v4 for better mobile support

Impact: Fewer disconnections on mobile and unstable networks.
```

---

### Technique 2: Quantify Impact

**Add numbers wherever possible:**

```markdown
❌ VAGUE:
- Improved performance

✅ SPECIFIC:
- **Performance**: Reduced cache lookup time from 12ms to 0.8ms (15x faster)
  by implementing hash-based indexing. Affects all cache reads.

❌ VAGUE:
- Fixed memory leak

✅ SPECIFIC:
- **Fixed**: Memory leak causing 200MB/hour growth in long-running workers.
  Affected users running background jobs > 6 hours. (#234)
```

**Quantifiable metrics:**
- Performance: "40% faster", "reduced from X to Y"
- Scale: "supports 10K concurrent connections (up from 2K)"
- Size: "bundle size reduced 30%"
- Reliability: "99.9% uptime (vs 94% in v1)"

---

### Technique 3: Link Context

**Tie changes to issues, PRs, and docs:**

```markdown
### Fixed
- Cache invalidation race condition causing stale reads (#234)
- Memory leak in connection pooling (#256, thanks @contributor)
- Type errors with TypeScript 5.0 (#271)

See [v2.0 migration guide](link) for upgrade instructions.
```

**Benefits:**
- Issue numbers let users verify the fix
- Contributor credits build community
- Links reduce "how do I upgrade?" questions

---

### Technique 4: Tell the Version Story

**Each version should have a theme/narrative:**

```markdown
## [2.0.0] - 2025-01-15 - "Real-Time Ready"

This release focuses on real-time capabilities and developer experience.

**Headline features:**
- Real-time subscriptions replace polling (see `Added` section)
- 60% faster cache reads (see `Performance` section)
- Simplified API for common patterns (see `Changed` section)

**Breaking changes:** We removed deprecated v1.0 APIs and changed some
defaults for better production behavior. Budget 1-2 hours for migration.
See [v2 migration guide](link).

### Added
...
```

**Version narratives provide:**
- **Context**: Why this version exists
- **Expectations**: How much effort to upgrade
- **Priorities**: What team focused on

---

## Changelog Structure Patterns

### Pattern 1: Standard OSS Project

```markdown
# Changelog

## [Unreleased]
### Added
- WIP features not yet released

## [2.1.0] - 2025-02-01
### Added
### Changed
### Fixed

## [2.0.0] - 2025-01-15
### Breaking
### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security
```

**Use when:** Most open source projects, libraries, frameworks

---

### Pattern 2: API-Focused Changelog

```markdown
# API Changelog

## v2.1 (2025-02-01)

### New Endpoints
- `POST /webhooks` - Register webhook subscriptions
- `GET /analytics/summary` - Usage analytics

### Modified Endpoints
- `POST /auth/token` - Now accepts `refresh_token` parameter

### Deprecated Endpoints
- `GET /users/:id/legacy` - Use `/users/:id` instead (removal: v3.0)

### Breaking Changes
None

### Bug Fixes
- `GET /orders` pagination now respects `limit` parameter (#122)
```

**Use when:** REST APIs, GraphQL APIs, external-facing interfaces

---

### Pattern 3: Database Migration Changelog

```markdown
# Database Changelog

## Migration 2025-01-15-001 (v2.0.0)

### Schema Changes
- Added `users.email_verified_at` column (nullable timestamp)
- Dropped `sessions.legacy_data` column (unused since v1.5)
- Created index on `orders.created_at` (performance)

### Data Migrations
- Backfilled `email_verified_at` for existing users
- Migrated legacy session data to new format

### Rollback Instructions
```sql
-- To rollback:
ALTER TABLE users DROP COLUMN email_verified_at;
```

**Affected environments**: Production, staging
**Estimated downtime**: ~30 seconds for index creation
```

**Use when:** Database schemas, infrastructure changes

---

## Automation & Tooling

### Auto-Generated vs Hand-Written

**Auto-generated changelogs (from commits):**

✅ **Pros:**
- Zero manual effort
- Never forget to update
- Links to commits/PRs automatically

❌ **Cons:**
- Commit messages ≠ user-facing changes
- Includes internal changes users don't care about
- No narrative structure
- No impact explanation

**Recommendation:** Auto-generate as draft, then edit manually

---

### Tooling

**Changelog generators:**
- **conventional-changelog**: Auto-generate from conventional commits
- **release-please**: Google's release automation
- **semantic-release**: Automated versioning + changelog
- **Keep a Changelog**: Manual template

**Workflow:**

```bash
# 1. Generate draft from commits
npx conventional-changelog -p angular -i CHANGELOG.md -s

# 2. Edit manually
# - Add impact statements
# - Group related changes
# - Remove internal changes
# - Clarify breaking changes

# 3. Commit
git add CHANGELOG.md
git commit -m "docs: update changelog for v2.0.0"
```

---

## Common Mistakes & Fixes

| Mistake | Impact | Fix |
|---------|--------|-----|
| **Commit log dump** | Unreadable, includes internal changes | Curate for user-facing changes |
| **No breaking changes section** | Users surprised by breaks | Always separate `### Breaking` |
| **Vague descriptions** | "Fixed bug", "Updated feature" | Specify what changed and impact |
| **No dates** | Can't tell if old or new | Always include `[version] - YYYY-MM-DD` |
| **Not following semver** | Unpredictable upgrades | Breaking = major, features = minor, fixes = patch |
| **No migration guides** | Users can't upgrade | Link to migration docs for breaking changes |
| **Unreleased section stale** | Looks abandoned | Keep `[Unreleased]` up-to-date or remove it |
| **All changes equal weight** | Can't scan for critical changes | Put breaking changes first, group related changes |

---

## Changelog Checklist

Before releasing:

**Structure:**
- [ ] Follows Keep a Changelog format
- [ ] Latest version at top (reverse chronological)
- [ ] Version number + date for each release
- [ ] Categories: Added, Changed, Deprecated, Removed, Fixed, Security, Breaking
- [ ] `[Unreleased]` section present (if using)

**Content:**
- [ ] Breaking changes clearly marked and explained
- [ ] Migration paths provided for breaking changes
- [ ] Impact stated for major changes ("40% faster", "supports 10K users")
- [ ] Issue/PR numbers linked
- [ ] Contributors credited (if open source)
- [ ] User-facing changes only (no internal refactoring unless it affects users)

**Clarity:**
- [ ] Each entry answers "how does this affect me?"
- [ ] Before/after code examples for breaking changes
- [ ] Grouped related changes thematically
- [ ] No commit message jargon ("refactored widget factory")
- [ ] Links to migration guides for complex changes

**Communication:**
- [ ] Deprecation warnings include timeline and migration path
- [ ] Security fixes noted (without exposing vulnerabilities)
- [ ] Version story/theme summarized (for major releases)

---

## Advanced: Changelogs as Product Marketing

**Changelogs aren't just for developers—they're product history.**

### Use in Product Marketing

```markdown
## [2.0.0] - 2025-01-15 - "Real-Time Ready"

After 18 months of development and feedback from 500+ production users,
v2.0 brings the most requested feature: real-time subscriptions.

**The headline:** What took 200 lines of polling code now takes 3 lines
of subscription code. Your users get live updates, you get simpler code.

**By the numbers:**
- 60% faster cache reads
- 95% reduction in polling-related code
- Zero-config real-time subscriptions
- Used in production by companies serving 50M+ users

[Read the launch post](link) | [Watch the demo](link)
```

**Marketing changelog benefits:**
- Announcements reference changelog for technical details
- Changelog becomes SEO content ("FastCache 2.0 release notes")
- Tells story of product evolution for case studies

---

## Changelogs for Different Audiences

### Developer Changelog (technical)

```markdown
### Breaking
- `authenticate()` now returns `Promise<User>` (was `User`)
- Requires Node.js 18+ (dropped support for 16)
```

### User Changelog (non-technical)

```markdown
### What's New
- **Real-time updates**: See changes instantly without refreshing
- **Faster dashboard**: Loads 60% faster
```

**Strategy:** Maintain both if you have distinct developer vs end-user audiences.

---

## The Lyra Approach

**"Changelogs are version stories, not version lists."**

Balance:
- **Technical accuracy** (precise breaking change descriptions)
- **Narrative flow** (group related changes, provide context)
- **User empathy** (explain impact, not just changes)
- **Migration support** (make upgrades confident, not scary)

Great changelogs turn "scary upgrade" into "exciting new capabilities."

Your users will upgrade more confidently when they understand what changed and why.
