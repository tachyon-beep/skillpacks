# Reference Sheet: Peer Reviews

## Purpose & Context

Provides frameworks for effective code reviews beyond checklists - addresses social dynamics, psychological safety, and reviewer accountability.

**When to apply**: Reviews ineffective (rubber stamps), unclear what reviewers should focus on, giving critical feedback difficult

**Prerequisites**: Basic code review process exists (PRs, review tool)

---

## CMMI Maturity Scaling

### Level 2: Managed
- Peer review recommended (not enforced)
- Informal feedback acceptable
- Single reviewer sufficient

### Level 3: Defined
- **2+ reviewers required** (platform-enforced)
- Review checklist used
- Substantive feedback required (not just "LGTM")
- Review metrics tracked (finding rate 20-40% target)

### Level 4: Quantitatively Managed
- Review effectiveness in statistical control
- Defect prediction models based on review findings
- Review time optimized via historical data

---

## The Social Dynamics Problem

**Core Issue**: Reviews fail due to culture, not lack of checklist

### Why Reviewers Give Rubber Stamps

1. **Social Pressure**: Don't want to block teammates or seem difficult
2. **Conflict Aversion**: Giving critical feedback is uncomfortable
3. **Time Pressure**: "Thorough review takes too long"
4. **Lack of Accountability**: No consequences for missing bugs
5. **Unclear Responsibility**: "Not my job to find bugs, that's QA's job"

**Result**: "LGTM" approvals in <5 minutes, defects escape to production

---

## Reviewer Accountability Framework

### What Reviewers Are Responsible For

**Must check** (reviewer accountable if missed):
- **Correctness**: Logic bugs, edge cases, null checks
- **Tests**: New code has tests, tests actually verify behavior
- **Security**: No SQL injection, XSS, hardcoded secrets
- **Performance**: No obvious N+1 queries, unnecessary loops

**Should check** (best effort):
- **Design**: Appropriate abstractions, clear interfaces
- **Readability**: Clear naming, reasonable function length
- **Maintainability**: No code duplication, good comments

**Not responsible for** (author's job):
- **Style**: Automated linters handle this
- **Requirements**: Product owner validates feature correctness
- **Complete feature testing**: QA does comprehensive testing

**Accountability metric**: If bug found in production that reviewer should have caught → retrospective required

---

## Review Checklist (Level 3)

### Functionality
- [ ] Code does what PR description claims
- [ ] Edge cases handled (null, empty, boundary values, errors)
- [ ] No obvious logic bugs
- [ ] Errors propagated or handled appropriately

### Tests
- [ ] New code has tests (unit + integration as needed)
- [ ] Tests actually test the code (not just pass trivially)
- [ ] Edge cases covered in tests
- [ ] Existing tests still pass

### Design
- [ ] Follows existing patterns in codebase
- [ ] No code duplication (DRY)
- [ ] Appropriate abstraction (not over/under-engineered)
- [ ] Clear module boundaries

### Security
- [ ] No hardcoded secrets (API keys, passwords)
- [ ] Input validation present where needed
- [ ] No SQL injection, XSS, or other OWASP Top 10 vulnerabilities

### Performance
- [ ] No obvious performance issues (N+1 queries, unnecessary loops)
- [ ] Database queries efficient
- [ ] Caching used appropriately if needed

### Readability
- [ ] Clear, descriptive naming
- [ ] Functions reasonably sized (<50 lines)
- [ ] Comments explain WHY where needed (not WHAT)
- [ ] No commented-out code

---

## Giving Critical Feedback: The Psychological Safety Playbook

### Distinguish MUST-FIX from SUGGESTIONS

```
# ❌ BAD (ambiguous severity)
"This could be better"

# ✅ GOOD (clear severity)
MUST-FIX: This will cause a null pointer exception if user.email is None.

SUGGESTION: Consider extracting this 80-line function into smaller functions
for readability. The current approach works fine though.
```

### Be Specific and Actionable

```
# ❌ BAD (vague)
"The logic here seems off"

# ✅ GOOD (specific)
"Lines 45-50: This assumes input is always >0, but the function signature
doesn't require that. Add validation or update the docstring to document
the precondition."
```

### Provide Rationale

```
# ❌ BAD (dictatorial)
"Use a try-finally block here"

# ✅ GOOD (explains why)
"Use a try-finally block here to ensure the file handle is closed even if
an exception occurs. Current code will leak file handles on error."
```

### Frame as Collaboration

```
# ❌ BAD (adversarial)
"You got this wrong"

# ✅ GOOD (collaborative)
"I might be misunderstanding the requirements - could you explain the
intended behavior when user_id is None? I'm concerned line 67 will crash."
```

### Approve When Appropriate

**Don't let perfect be enemy of good**:
- MUST-FIX issues → Request changes
- Only SUGGESTIONS → Approve with comments

**Trust the author**: If you've pointed out issues and author addressed them, approve. Don't require perfection.

---

## Review Taxonomy: Different Depths for Different Changes

### Critical Change (45-60 minutes)

**What**: Security patches, payment logic, auth changes, data migrations

**Review depth**:
- Pull code, run locally, test manually
- Review every line
- Check tests comprehensively
- Validate security implications
- 2+ reviewers required

### Feature (20-30 minutes)

**What**: New user-facing functionality

**Review depth**:
- Review main logic and tests
- Check for obvious bugs
- Validate design approach
- Spot-check edge cases
- 1-2 reviewers

### Refactor (10-20 minutes)

**What**: Code restructuring, no behavior change

**Review depth**:
- Verify behavior unchanged (tests pass)
- Check readability improved
- Ensure no new bugs introduced
- 1 reviewer sufficient

### Hotfix (5-10 minutes + retrospective)

**What**: Emergency production fix

**Review depth**:
- Verify fix addresses immediate issue
- Check for obvious side effects
- Approve quickly to unblock
- **Retrospective review within 48h**: Proper solution, tests, RCA

**Classification Requirements**:
- Hotfix classification requires verifiable production outage ticket
- **Abuse detection**: >20% of changes classified as "hotfix" = escalate to manager
- Review taxonomy guidelines must be approved by team lead
- False hotfix classification = process violation

### Trivial (2-5 minutes)

**What**: Documentation, comments, formatting

**Review depth**:
- Quick scan for typos or errors
- Approve rapidly
- 1 reviewer

---

## Review Metrics (Level 3+)

### Finding Rate

**Formula**: Bugs found in review / Total bugs found (review + QA + production)

**Target**: 20-40%

**Interpretation**:
- <20% → Reviews too shallow or reviewers undertrained
- 20-40% → Healthy (catching meaningful issues)
- >60% → Might be over-reviewing (diminishing returns)

**Measurement**: Track bugs by discovery phase (review, QA, production)

### Review Turnaround Time

**Formula**: Time from PR creation to approval

**Target**: <4 hours median for non-critical changes

**Why it matters**: Long delays block progress, reduce iteration speed

**Intervention**: If >24h median, investigate (too busy? unclear responsibilities?)

### Review Depth (Time Spent)

**Proxy**: Minutes spent per 100 lines changed

**Target**: ~2-5 minutes per 100 LOC (varies by change type)

**Caveat**: Don't optimize for speed at expense of thoroughness

---

## Common Anti-Patterns

| Anti-Pattern | Symptoms | Why It Fails | Better Approach |
|--------------|----------|--------------|-----------------|
| **Rubber Stamp** | "LGTM" in <5 min, no comments, bugs escape | Social pressure > quality | Reviewer accountability, metrics (20-40% finding rate) |
| **Bikeshedding** | Hour-long debates about variable names | Waste time on trivial issues | Automate style with linters, focus on logic |
| **Review Backlog** | PRs sit for days without review | Blocks progress, code goes stale | Review SLA (<4h), rotate review duty |
| **Perfection Paralysis** | Request 10 rounds of changes for minor issues | Demotivates authors, slows delivery | MUST-FIX vs SUGGESTIONS, trust authors |
| **No Accountability** | Reviewers miss bugs, no retrospective | Reviews become checkboxes | Track finding rate, retrospective if production bug was reviewable |

---

## Real-World Example: Rubber Stamps → Effective Reviews

**Context**:
- Team of 10, reviews <5 minutes each
- 80% of production bugs were visible in code review
- Finding rate: 10% (should be 20-40%)
- Team culture: "Don't block teammates"

**Actions**:
1. **Week 1**: Established reviewer accountability ("If you approve it, you own bugs you could have caught")
2. **Week 2**: Introduced review checklist, trained team on how to give feedback
3. **Week 3**: Started tracking finding rate metric, made it visible
4. **Week 4**: Retrospective on production bugs - which reviews missed them?

**Results after 2 months**:
- Finding rate: 10% → 35% (catching bugs pre-production)
- Production bugs: 40% reduction
- Review time: 5 min → 15 min average (appropriate depth)
- Culture shift: "Blocker = helper" (catching bugs early saves time)

**Key learning**: Problem was cultural (social pressure), not technical (checklist). Accountability + metrics changed behavior.

---

**Last Updated**: 2026-01-24
**Review Schedule**: Quarterly or when production bugs spike
