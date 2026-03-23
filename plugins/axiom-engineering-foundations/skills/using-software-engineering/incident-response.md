# Incident Response

Production fire methodology. Contain damage, restore service, learn from it.

## Core Principle

**Restore service first, investigate later.** When production is down, users are suffering NOW. Your job is to stop the bleeding, not to understand why you're bleeding. Root cause analysis comes after the fire is out.

## When to Use This

- Production is down or degraded
- Users are actively affected
- Alerts are firing
- "Everything is broken"
- Need to coordinate incident response

**Don't use for**: Normal debugging (use [complex-debugging.md](complex-debugging.md)), planned maintenance, performance optimization.

---

## Incident Response Process

```
┌─────────────────┐
│ 1. ASSESS       │ ← What's the blast radius?
└────────┬────────┘
         ↓
┌─────────────────┐
│ 2. CONTAIN      │ ← Stop the bleeding
└────────┬────────┘
         ↓
┌─────────────────┐
│ 3. MITIGATE     │ ← Restore service (even partially)
└────────┬────────┘
         ↓
┌─────────────────┐
│ 4. DIAGNOSE     │ ← Now find root cause
└────────┬────────┘
         ↓
┌─────────────────┐
│ 5. FIX          │ ← Permanent resolution
└────────┬────────┘
         ↓
┌─────────────────┐
│ 6. POSTMORTEM   │ ← Learn and prevent
└─────────────────┘
```

**Golden rule**: Phases 1-3 are about SPEED. Phases 4-6 are about THOROUGHNESS.

---

## Phase 1: Assess

**30 seconds to understand scope.**

### Triage Questions

| Question | Why It Matters |
|----------|----------------|
| What's broken? | Scope of the problem |
| Who's affected? | User impact, business priority |
| When did it start? | What changed recently? |
| What changed? | Deploy? Config? Traffic spike? |
| How bad? | 10% of users or 100%? |

### Severity Classification

| Severity | Impact | Response |
|----------|--------|----------|
| **SEV-1** | Complete outage, all users | All hands on deck |
| **SEV-2** | Major feature broken, many users | Immediate response |
| **SEV-3** | Minor feature broken, some users | Business hours |
| **SEV-4** | Cosmetic, edge case | Ticket for later |

### Ask the User Immediately

If you're helping with an incident, get this information NOW:
- "What symptoms are users seeing?"
- "When did this start?"
- "What deployed or changed recently?"
- "What does your monitoring show?"
- "What have you already tried?"

---

## Phase 2: Contain

**Stop it from getting worse.**

### Containment Actions

| Situation | Containment |
|-----------|-------------|
| Bad deploy | Rollback to previous version |
| Feature bug | Disable feature flag |
| Traffic spike | Enable rate limiting |
| Data corruption | Halt writes, preserve state |
| Security breach | Revoke access, isolate system |
| Cascading failure | Shed load, circuit break |

### Rollback Decision

```
Recent deploy + New symptoms = ROLLBACK FIRST, ASK QUESTIONS LATER
```

```bash
# If you have deploy history
kubectl rollout undo deployment/app

# Or revert to known-good version
git revert HEAD
deploy_to_prod
```

**Don't wait for root cause to rollback.** You can always re-deploy after investigation.

### Feature Flags

If you have feature flags, disable suspicious features:

```bash
# Disable feature
feature_flag disable new-checkout-flow

# Monitor if symptoms resolve
```

---

## Phase 3: Mitigate

**Restore service, even partially.**

### Mitigation Priority

1. **Full restore** - Rollback works, back to normal
2. **Partial restore** - Core features work, edge cases broken
3. **Degraded mode** - Limited functionality, clear messaging
4. **Maintenance mode** - System down, but communicating status

### Communication

Tell users what's happening:

```markdown
## Status: Investigating

We're aware of issues with [feature]. Our team is investigating.

Current status: [Degraded/Down]
Workaround: [If any]
Next update: [Time]
```

### Temporary Fixes

Temporary fixes are OKAY during incidents:

```python
# INCIDENT FIX: Bypass broken cache (REMOVE AFTER INCIDENT)
# Ticket: INC-1234
def get_user(id):
    # return cache.get(f"user:{id}") or db.get_user(id)
    return db.get_user(id)  # Bypass cache temporarily
```

**Document temporary fixes** - They become permanent if forgotten.

---

## Phase 4: Diagnose

**NOW find root cause.**

### After Service Is Restored

Use [complex-debugging.md](complex-debugging.md) methodology:
1. Reproduce (if possible in non-prod)
2. Isolate (what component failed?)
3. Hypothesize (what caused it?)
4. Verify (prove the hypothesis)

### Incident-Specific Investigation

| Data Source | What to Look For |
|-------------|------------------|
| **Logs** | Errors, stack traces, timing |
| **Metrics** | When did graphs change? |
| **Deploys** | What shipped recently? |
| **Config changes** | Any manual changes? |
| **External dependencies** | Third-party status pages |
| **Traffic patterns** | Unusual load? DDoS? |

### Timeline Reconstruction

Build a timeline of events:

```markdown
14:30 - Deploy v2.3.4 completed
14:32 - Error rate increased 5% → 20%
14:35 - First user complaint
14:38 - Alert fired
14:42 - Incident declared, rollback initiated
14:45 - Rollback complete, errors dropping
14:50 - Error rate back to baseline
```

---

## Phase 5: Fix

**Permanent resolution.**

### Fix Requirements

- [ ] Root cause identified and documented
- [ ] Fix addresses root cause (not symptom)
- [ ] Fix has tests
- [ ] Fix reviewed
- [ ] Rollback plan if fix fails

### Testing Before Deploy

```bash
# Test fix in staging
run_regression_tests

# Verify fix addresses specific issue
run_incident_reproduction_test

# Deploy with monitoring
deploy_to_prod --monitor
```

### Remove Temporary Fixes

After permanent fix is verified:
1. Remove temporary mitigations
2. Verify system works without them
3. Update documentation

---

## Phase 6: Postmortem

**Learn so it doesn't happen again.**

### Postmortem Template

```markdown
# Incident Postmortem: [Title]

**Date**: YYYY-MM-DD
**Duration**: X hours Y minutes
**Severity**: SEV-N
**Author**: [Name]

## Summary
One-paragraph summary of what happened.

## Impact
- Users affected: [Number/percentage]
- Revenue impact: [If applicable]
- Duration of impact: [Time]

## Timeline
- HH:MM - Event
- HH:MM - Event
...

## Root Cause
Technical explanation of why this happened.

## Contributing Factors
- Factor 1
- Factor 2

## Resolution
What fixed it (temporary and permanent).

## What Went Well
- Item 1
- Item 2

## What Went Poorly
- Item 1
- Item 2

## Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| Item 1 | Name  | Date     | Open   |

## Lessons Learned
Key takeaways for the team.
```

### Blameless Culture

**Focus on systems, not people.**

❌ "John deployed buggy code"
✅ "Code review didn't catch the bug; deploy pipeline lacked tests"

❌ "Sarah didn't check before deploying"
✅ "Pre-deploy checklist wasn't enforced automatically"

### Action Items

Good action items:
- Specific (not "improve monitoring")
- Actionable (not "be more careful")
- Owned (someone responsible)
- Time-bound (due date)

```markdown
## Action Items
| Action | Owner | Due | Status |
|--------|-------|-----|--------|
| Add integration test for checkout flow | Alex | 3/15 | Open |
| Add alert for error rate > 5% | Sam | 3/10 | Open |
| Document rollback procedure | Jordan | 3/12 | Open |
```

---

## Incident Communication

### Internal Updates

Keep stakeholders informed:

```markdown
## Incident Update #3 - 15:30

**Status**: Mitigating
**Impact**: 30% of checkout attempts failing

**What we know**:
- Payment API returning timeouts
- Started after 14:30 deploy

**What we're doing**:
- Rolling back to v2.3.3
- ETA for rollback: 15:45

**Next update**: 16:00 or when status changes
```

### External Communication

For user-facing incidents:

```markdown
## [Service] - Degraded Performance

We're experiencing issues with [feature]. Some users may see
[symptom]. Our team is actively working on a resolution.

**Workaround**: [If any]
**Status**: Investigating
**Last updated**: HH:MM timezone

Updates will be posted as available.
```

---

## On-Call Handoff

If you need to hand off an incident:

```markdown
## Incident Handoff

**Incident**: [Brief description]
**Status**: [Contain/Mitigate/Diagnose]
**Current severity**: SEV-N

**Context**:
- What happened
- What we've tried
- What we learned

**Current hypothesis**: [If any]

**Next steps**:
1. [Action 1]
2. [Action 2]

**Contacts**:
- Original reporter: [Name]
- Subject matter expert: [Name]
```

---

## Red Flags During Incidents

| Thought | Reality | Action |
|---------|---------|--------|
| "Let me understand the code first" | Users are suffering NOW | Rollback first |
| "This fix might work" | Untested fix might make it worse | Rollback to known good |
| "It's just a small issue" | Small issues cascade | Take it seriously |
| "We need the permanent fix" | Temporary mitigation is fine | Restore service first |
| "I can debug this quickly" | Debugging takes time | Mitigate, then debug |
| "One more minute..." | Scope creep delays recovery | Time-box, escalate |

---

## Related Skills

### During Incident

| Need | Skill | When |
|------|-------|------|
| Find root cause | [complex-debugging.md](complex-debugging.md) | After service is restored |
| Understand unfamiliar code | [codebase-confidence-building.md](codebase-confidence-building.md) | Incident in code you don't know |

### Setting Up for Future Incidents

| Need | Skill | When |
|------|-------|------|
| Observability | `ordis-quality-engineering:observability-and-monitoring` | Improve detection/diagnosis |
| Chaos testing | `ordis-quality-engineering:chaos-engineering-principles` | Find weaknesses before incidents |
| Load testing | `ordis-quality-engineering:load-testing-patterns` | Validate capacity |

### After Incident

| Need | Skill | When |
|------|-------|------|
| Document decisions | `muna-technical-writer:create-adr` | Capture architectural choices made during incident |
| Systemic issues | `yzmir-systems-thinking:systems-archetypes-reference` | Incident reveals systemic pattern |

---

## Quick Reference

### Incident Checklist

- [ ] **Assess**: What's broken? Who's affected? How bad?
- [ ] **Contain**: Stop it from getting worse (rollback, disable)
- [ ] **Mitigate**: Restore service (even partially)
- [ ] **Communicate**: Keep stakeholders informed
- [ ] **Diagnose**: Find root cause (after service is up)
- [ ] **Fix**: Permanent resolution with tests
- [ ] **Postmortem**: Document and create action items

### Severity Quick Guide

| Sev | Impact | Response Time | Who |
|-----|--------|---------------|-----|
| 1 | All users, core function | Immediate | Everyone |
| 2 | Many users, important feature | < 1 hour | On-call + backup |
| 3 | Some users, minor feature | < 4 hours | On-call |
| 4 | Edge case, cosmetic | Next business day | Ticket |

### First 5 Minutes

1. What's broken?
2. What changed recently?
3. Can we rollback?
4. Who needs to know?
5. What's our mitigation?
