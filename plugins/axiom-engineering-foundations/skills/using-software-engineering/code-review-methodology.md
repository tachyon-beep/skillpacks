# Code Review Methodology

Systematic code review process. Find real issues, give actionable feedback.

## Core Principle

**Code review catches what automation misses.** Linters catch syntax, tests catch regressions, but humans catch design problems, maintainability issues, and "this will bite us later" code. Focus your attention where it uniquely matters.

## When to Use This

- Reviewing someone else's PR
- Self-reviewing before requesting review
- Receiving and responding to feedback
- Teaching through code review
- Large PR feels overwhelming

**Don't use for**: Debugging (use [complex-debugging.md](complex-debugging.md)), refactoring decisions (use [systematic-refactoring.md](systematic-refactoring.md)).

---

## The Review Process

```
┌──────────────────┐
│ 1. CONTEXT       │ ← Understand what and why
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 2. HIGH-LEVEL    │ ← Architecture, approach
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 3. CORRECTNESS   │ ← Does it work? Edge cases?
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 4. QUALITY       │ ← Maintainability, clarity
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 5. FEEDBACK      │ ← Clear, actionable comments
└──────────────────┘
```

**Order matters.** Don't bikeshed naming (step 4) if the approach is wrong (step 2).

---

## Phase 1: Context

**Understand before judging.**

### Read First

1. **PR title and description** - What is this trying to do?
2. **Linked issue/ticket** - What's the requirement?
3. **Conversation history** - Any context from prior discussion?

### Questions to Answer

| Question | Why It Matters |
|----------|----------------|
| What problem does this solve? | Can't review solution without understanding problem |
| Why this approach? | Alternatives may have been considered |
| What's the scope? | Avoid asking for out-of-scope changes |
| Who's affected? | Other teams, systems, users? |

### Claude-Specific

You can quickly scan the entire codebase for context:
- Related code that interacts with changed files
- Similar patterns elsewhere
- Tests that exercise this code

```bash
# Find callers of changed code
grep -r "changed_function" --include="*.py"

# Find similar patterns
grep -r "similar_pattern" --include="*.py"
```

---

## Phase 2: High-Level Review

**Architecture and approach before details.**

### Questions to Ask

| Question | Red Flag If... |
|----------|----------------|
| Does the approach make sense? | Overcomplicated for the problem |
| Is this the right place for this code? | Wrong module/layer |
| Does it fit existing patterns? | Invents new pattern unnecessarily |
| Will it scale? | O(n²) when data grows |
| Is it testable? | Hard to unit test |
| Does it introduce coupling? | New dependencies between unrelated modules |

### Architecture Issues

**Stop if you find architecture problems.** Don't review details of code that needs fundamental redesign.

```markdown
## Architecture Concern

This adds direct database access from the API controller. Our pattern is:
Controller → Service → Repository → Database

Suggested approach: Create a UserService method instead.

Happy to review details once we align on approach.
```

### When to Request Approach Discussion

- Fundamental design choice you disagree with
- Multiple valid approaches, unclear which was considered
- Change affects system architecture

---

## Phase 3: Correctness

**Does it actually work?**

### What to Check

| Category | Questions |
|----------|-----------|
| **Logic** | Does the algorithm do what it claims? |
| **Edge cases** | Empty input? Null? Max values? Concurrent access? |
| **Error handling** | What if X fails? Is the error message helpful? |
| **State** | Are assumptions about state valid? Race conditions? |
| **Data** | Correct types? Encoding? Timezone handling? |
| **Security** | Input validation? SQL injection? XSS? Auth checks? |

### Test Coverage

```markdown
I don't see tests for:
- Empty input case (line 42 would crash)
- Concurrent access scenario
- Error case when API returns 500

Could you add tests covering these?
```

### Security Checklist

| Risk | Check For |
|------|-----------|
| **Injection** | User input in queries/commands/templates |
| **Auth** | All endpoints check permissions |
| **Data exposure** | Sensitive data in logs, responses, errors |
| **CSRF/XSS** | Token validation, output encoding |
| **Secrets** | Hardcoded credentials, keys in code |

---

## Phase 4: Quality

**Maintainability and clarity.**

### What to Check

| Category | Questions |
|----------|-----------|
| **Readability** | Can I understand this in 6 months? |
| **Naming** | Do names describe purpose? |
| **Complexity** | Is this simpler than it needs to be? |
| **Duplication** | Repeated code that should be extracted? |
| **Comments** | Explains WHY, not WHAT? |
| **Documentation** | API changes documented? |

### Readability Heuristics

- **Function length** - Can you see entire function without scrolling?
- **Nesting depth** - More than 3 levels of nesting?
- **Parameter count** - More than 3-4 parameters?
- **Single responsibility** - Does function do one thing?

### When to Comment on Style

**Only if**:
- Not caught by automated linting
- Significantly impacts readability
- Pattern should be established for project

**Skip if**:
- Personal preference ("I would do it differently")
- Minor and doesn't affect understanding
- Automated tools should catch it

---

## Phase 5: Feedback

**Clear, actionable, kind.**

### Comment Structure

```markdown
## [Category] Brief Issue

[Context: Why this matters]

[Current code or problem]

[Suggested fix or question]

[Optional: Example of fix]
```

### Example Comments

**Good: Actionable with Context**
```markdown
## Bug: Race condition

If two requests hit this endpoint simultaneously, both could
read `count=5`, increment, and write `count=6` instead of `count=7`.

Consider using a database transaction or atomic increment:

```python
# Option 1: Atomic
Counter.objects.filter(id=id).update(count=F('count') + 1)

# Option 2: Transaction with SELECT FOR UPDATE
```
```

**Bad: Vague**
```markdown
This could have issues.
```

**Bad: Style preference without substance**
```markdown
I would write this differently.
```

### Comment Categories

Use consistent prefixes:

| Prefix | Meaning | Blocking? |
|--------|---------|-----------|
| **Bug:** | Incorrect behavior | Yes |
| **Security:** | Security vulnerability | Yes |
| **Question:** | Need clarification | Maybe |
| **Suggestion:** | Improvement idea | No |
| **Nit:** | Minor style/preference | No |
| **Praise:** | Something done well | No |

### When NOT to Comment

- Caught by linter/CI (they'll learn from automation)
- Personal preference with no clear benefit
- Out of scope for this PR
- Already at diminishing returns (stop at ~10 comments)

---

## Large PR Strategy

**Don't boil the ocean.**

### For Overwhelming PRs

1. **Skim all files first** - Get mental map
2. **Identify critical paths** - What's the core change?
3. **Review critical paths deeply** - Correctness, security
4. **Scan remaining files** - High-level only
5. **Time-box** - 1 hour max per review session

### Request Split if Needed

```markdown
This PR has 3 distinct changes:
1. Database migration
2. API endpoint changes
3. Frontend updates

Could we split into separate PRs? It would make review safer
and allow independent rollback if issues arise.
```

---

## Receiving Feedback

**Feedback is a gift, even when it stings.**

### How to Respond

| Feedback Type | Response |
|---------------|----------|
| **Valid bug** | "Good catch, fixed!" |
| **Good suggestion** | "Makes sense, updated" |
| **Disagree but minor** | Just do it (not worth debate) |
| **Disagree fundamentally** | Explain reasoning, seek alignment |
| **Unclear** | Ask for clarification |

### Don't Do This

- Get defensive ("It works though")
- Dismiss without consideration ("That's not a problem")
- Take it personally (it's about code, not you)
- Argue style preferences (agree or defer to team norms)

### When to Push Back

Push back professionally when:
- Requested change contradicts requirements
- Suggestion would introduce bugs
- Scope creep beyond PR purpose
- Fundamental disagreement on approach (escalate to tech lead)

```markdown
I considered that approach, but went with this because:
1. [Reason 1]
2. [Reason 2]

Happy to discuss further if you think I'm missing something.
```

---

## Review Anti-Patterns

| Anti-Pattern | Problem | Instead |
|--------------|---------|---------|
| **Rubber stamping** | "LGTM" without reading | Actually review or don't approve |
| **Gatekeeping** | Blocking for preferences | Focus on correctness and clarity |
| **Bikeshedding** | Debating trivial choices | Save energy for important issues |
| **Drive-by criticism** | Comment without offering solution | Suggest alternatives |
| **Scope creep** | Requesting unrelated changes | Open separate issue |
| **Perfectionism** | Blocking for minor issues | Use "nit:" and approve |
| **Delayed review** | PR sits for days | Review within 24 hours or hand off |

---

## Integration with Other Skills

| Skill | Relationship |
|-------|--------------|
| [codebase-confidence-building.md](codebase-confidence-building.md) | Build context for unfamiliar areas |
| [complex-debugging.md](complex-debugging.md) | If review reveals bugs |
| [systematic-refactoring.md](systematic-refactoring.md) | When suggesting refactoring |
| [technical-debt-triage.md](technical-debt-triage.md) | Prioritize "should fix" comments |

---

## Quick Reference

### Review Checklist

- [ ] **Context**: Understand what and why
- [ ] **Architecture**: Does approach make sense?
- [ ] **Correctness**: Logic, edge cases, error handling
- [ ] **Security**: Input validation, auth, data exposure
- [ ] **Tests**: Adequate coverage of changes
- [ ] **Quality**: Readable, maintainable, documented
- [ ] **Feedback**: Clear, actionable, categorized

### Comment Template

```markdown
## [Category]: [Brief Issue]

[Why this matters]

[Specific problem]

[Suggested fix or question]
```

### Review Priority

1. **Security vulnerabilities** - Block and fix immediately
2. **Bugs** - Block until resolved
3. **Design/architecture** - Discuss before details
4. **Test coverage** - Request additions
5. **Code quality** - Suggest improvements
6. **Style nits** - Comment or skip
