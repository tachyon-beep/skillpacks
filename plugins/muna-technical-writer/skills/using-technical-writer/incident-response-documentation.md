
# Incident Response Documentation

## Overview

Create actionable runbooks for high-stress incidents. Core principle: **Clear, numbered steps with decision trees - no ambiguity under pressure**.

**Key insight**: Incidents are chaotic. Good runbooks provide structure when thinking is hardest.

## When to Use

Load this skill when:
- Creating security incident runbooks
- Documenting operational emergency procedures
- Writing post-incident reports
- Establishing escalation paths

**Symptoms you need this**:
- "What do we do if [security incident/outage]?"
- Creating runbooks for PII exposure, DDoS, database failure
- Documenting on-call procedures
- Writing post-mortems

**Don't use for**:
- General documentation (use `muna/technical-writer/documentation-structure`)
- Non-urgent procedures

## Response Template: 5 Phases

### Phase 1: Detection

**What to document**:
```markdown
## Detection

### Symptoms
- [What you see: alerts, error messages, user reports]
- [Monitoring dashboard links]

### Severity Classification
**P1 (Critical)**: [Description, e.g., PII exposure, complete outage]
**P2 (High)**: [Description, e.g., degraded performance, partial outage]
**P3 (Medium)**: [Description, e.g., minor issue affecting few users]
**P4 (Low)**: [Description, e.g., cosmetic issue, no user impact]

### Initial Triage
1. Check monitoring dashboard: [link]
2. Run diagnostic query:
   \```bash
   [Query to check system health]
   \```
3. Expected output: [What healthy looks like]
4. If [symptom X], proceed to Containment
```

### Phase 2: Containment

**What to document**:
```markdown
## Containment

### Goal
Stop the bleeding. Prevent further damage.

###Critical Actions (Do First)
1. **[Action 1]**: [Command/procedure]
   - Why: [Rationale]
   - Success criteria: [How to verify]

2. **[Action 2]**: [Command/procedure]
   - Why: [Rationale]
   - Success criteria: [How to verify]

### Communication Holds
❌ **DO NOT**: [Actions that tip off attacker or cause panic]
- Don't post to public Slack before containment
- Don't email affected users before scope known
- Don't restart services that destroy forensic evidence

✅ **DO**: [Immediate notifications]
- Alert security team via [pager]
- Notify incident commander via [phone]
```

### Phase 3: Investigation

**What to document**:
```markdown
## Investigation

### Log Collection
Collect logs for forensic analysis:
\```bash
# Application logs (last 24 hours)
aws logs filter-log-events --log-group-name /aws/app \
  --start-time $(date -d '24 hours ago' +%s)000 \
  --output json > incident-logs.json

# Access logs
aws s3 cp s3://logs/access-logs/ ./access-logs/ --recursive
\```

### Forensic Procedures
1. **Preserve evidence**: Take snapshots before changes
   \```bash
   aws ec2 create-snapshot --volume-id vol-abc123
   \```

2. **Timeline reconstruction**: When did compromise occur?
   - Check authentication logs for unauthorized access
   - Review deployment history for recent changes
   - Identify first appearance of anomaly

3. **Impact assessment**:
   - How many users/records affected?
   - What data was accessed/modified?
   - Did attacker establish persistence?

### Investigation Checklist
- [ ] Logs collected and preserved
- [ ] Timeline reconstructed (first compromise to detection)
- [ ] Scope determined (affected users, data, systems)
- [ ] Attack vector identified (how did they get in?)
- [ ] Persistence mechanisms found (backdoors, cron jobs, etc.)
```

### Phase 4: Recovery

**What to document**:
```markdown
## Recovery

### Restoration Procedure
1. **Remove attacker access**:
   \```bash
   # Rotate all credentials
   aws secretsmanager rotate-secret --secret-id prod/db/password

   # Revoke suspicious sessions
   redis-cli KEYS "session:suspicious_*" | xargs redis-cli DEL
   \```

2. **Patch vulnerability**:
   - Deploy fix: [git commit hash, deployment command]
   - Verify patch: [test procedure]

3. **Restore service**:
   - Bring systems back online
   - Verify functionality
   - Monitor for recurrence

### Verification Steps
- [ ] Vulnerability patched and verified
- [ ] All malicious access removed
- [ ] Service restored to normal operation
- [ ] Monitoring shows no anomalies for [duration]

### Monitoring for Recurrence
\```bash
# Watch for suspicious activity
watch -n 60 'aws logs tail /aws/app --follow --filter-pattern "[attack-pattern]"'
\```

Continue monitoring for 24-48 hours post-recovery.
```

### Phase 5: Lessons Learned

**What to document**:
```markdown
## Lessons Learned (Post-Incident Report)

### Timeline
| Time | Event | Actor |
|------|-------|-------|
| 10:23 | First compromise detected in logs | Attacker |
| 10:45 | Alert triggered, pager sent | Monitoring |
| 10:50 | On-call engineer acknowledged | John Doe |
| 11:05 | Containment actions completed | John Doe |
| 11:30 | Investigation confirmed SQL injection | Jane Smith |
| 12:15 | Patch deployed to production | DevOps |
| 12:30 | Service fully restored | Team |

### Root Cause
[Single-sentence root cause: "SQL injection in /api/users endpoint due to unparameterized query"]

### Impact
- **Users affected**: 1,247 users
- **Data exposed**: Email addresses and usernames (no passwords or payment data)
- **Downtime**: 2 hours (10:45-12:30)
- **Revenue impact**: ~$5,000 (estimated)

### What Went Well
- ✅ Alert fired within 20 minutes of compromise
- ✅ On-call responded in 5 minutes
- ✅ Containment completed within 20 minutes
- ✅ Clear runbook followed, no confusion

### What Could Improve
- ❌ Initial alert lacked severity context (delayed triage)
- ❌ Log retention only 7 days (lost pre-compromise forensics)
- ❌ No automated rollback procedure (manual steps delayed recovery)

### Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| Implement parameterized queries in all endpoints | Dev Team | 2024-04-01 | ✅ Done |
| Extend log retention to 90 days | Platform | 2024-04-15 | 🔄 In Progress |
| Add severity to alert messages | SRE | 2024-04-05 | ✅ Done |
| Create automated rollback procedure | DevOps | 2024-05-01 | 📋 Planned |

### Prevention
- Implement SAST scan in CI/CD to catch SQL injection
- Quarterly penetration testing
- Code review checklist updated with parameterized query requirement
```


## Escalation Paths

### Severity-Based Escalation

```markdown
## Escalation Matrix

### P1 (Critical): Immediate Escalation
**Definition**: Complete outage, security breach, data loss
**Response Time**: 15 minutes
**Escalation Path**:
1. On-call engineer (immediately via pager)
2. If no response in 5 min → Escalate to backup on-call
3. If no resolution in 15 min → Page incident commander
4. If ongoing after 30 min → Notify VP Engineering

**External Notifications**:
- Customer communication: Status page update within 30 min
- Regulatory notification (if PII breach): Within 72 hours
- Media inquiry response: Refer to PR team

**Contacts**:
- On-call: [pagerduty-link]
- Backup: Jane Doe (+1-555-0100)
- Incident Commander: John Smith (+1-555-0200)
- VP Engineering: Alice Johnson (+1-555-0300)
- PR Team: pr@example.com

### P2 (High): Escalate if Not Resolved
**Definition**: Degraded performance, partial outage, security concern
**Response Time**: 1 hour
**Escalation Path**:
1. On-call engineer (page)
2. If no resolution in 2 hours → Page incident commander
3. If ongoing after 4 hours → Notify VP Engineering

### P3 (Medium): Standard Response
**Definition**: Minor issue, workaround available
**Response Time**: 4 hours
**Escalation Path**:
1. Create ticket in incident system
2. On-call reviews and assigns
3. If no progress in 8 hours → Escalate to team lead

### P4 (Low): Track for Later
**Definition**: Cosmetic issue, no user impact
**Response Time**: Next business day
**Escalation Path**:
1. Create ticket
2. Addressed in next sprint planning
```


## Time-Critical Clarity Patterns

### Rule 1: Numbered Steps (Not Paragraphs)

❌ **WRONG**:
```
To respond to a database outage, you should first check if the primary is down by running a connection test. If the primary is unresponsive, you might want to consider promoting the replica to primary, but first make sure replication is up-to-date by checking the lag metrics. After promoting, update the connection string in the application configuration and restart the application servers.
```

**Problem**: Paragraph format, vague ("you might want to"), no clear sequence.

✅ **RIGHT**:
```
## Database Outage Response

1. **Check primary status**:
   \```bash
   pg_isready -h primary.db.internal -p 5432
   \```
   - If returns "accepting connections" → Primary is healthy, check replica
   - If returns "no response" → Primary is down, proceed to step 2

2. **Verify replica health**:
   \```bash
   psql -h replica.db.internal -c "SELECT pg_last_wal_replay_lsn();"
   \```
   - If replication lag < 1MB → Safe to promote
   - If lag > 1MB → Wait or accept data loss, document decision

3. **Promote replica to primary**:
   \```bash
   ssh replica.db.internal
   sudo pg_ctl promote -D /var/lib/postgresql/data
   \```
   - Success message: "server promoting"
   - Verify: `psql -c "SELECT pg_is_in_recovery();"` returns `f` (false = primary)

4. **Update application config**:
   \```bash
   kubectl set env deployment/app DATABASE_URL=postgresql://replica.db.internal:5432/app
   kubectl rollout restart deployment/app
   \```
   - Wait for rollout: `kubectl rollout status deployment/app`
   - Success: "successfully rolled out"

5. **Verify application health**:
   - Check monitoring dashboard: [link]
   - Test query: `curl https://api.example.com/health`
   - Expected: `{"status": "healthy", "database": "connected"}`
```

**Better**: Numbered steps, specific commands, success criteria, decision points.


### Rule 2: Decision Trees for Triage

❌ **WRONG**:
```
If you see high latency, check the database. If the database is slow, check for lock contention. If there's no lock contention, check for slow queries. Also check if the cache is working.
```

**Problem**: Unstructured, reader doesn't know priority.

✅ **RIGHT**:
```
## High Latency Triage

**Symptom**: API response time > 2 seconds

### Decision Tree

1. **Check cache hit rate**:
   \```bash
   redis-cli INFO stats | grep keyspace_hits
   \```
   - If hit rate < 80% → Cache miss issue, see [Cache Troubleshooting](#cache)
   - If hit rate ≥ 80% → Proceed to step 2

2. **Check database connection pool**:
   \```bash
   psql -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"
   \```
   - If active connections > 90 → Connection pool exhausted, see [Pool Tuning](#pool)
   - If active connections ≤ 90 → Proceed to step 3

3. **Check for slow queries**:
   \```bash
   psql -c "SELECT query, state, query_start FROM pg_stat_activity WHERE state != 'idle' AND (now() - query_start) > interval '5 seconds';"
   \```
   - If slow queries found → See [Query Optimization](#queries)
   - If no slow queries → Proceed to step 4

4. **Check for lock contention**:
   \```bash
   psql -c "SELECT * FROM pg_locks WHERE NOT granted;"
   \```
   - If locks found → See [Lock Resolution](#locks)
   - If no locks → Escalate to database team
```

**Better**: Clear decision tree, priority order, specific thresholds.


## Cross-References

**Use WITH this skill**:
- `ordis/security-architect/security-controls-design` - Understand control failure scenarios
- `muna/technical-writer/clarity-and-style` - Write clear steps under stress

**Use AFTER this skill**:
- `muna/technical-writer/documentation-testing` - Test runbooks with tabletop exercises

## Real-World Impact

**Runbooks using this framework**:
- **PII Exposure Response**: Detection→Containment→Investigation→Recovery structure enabled 45-minute response (vs 2-hour average without runbook). Forensic evidence preserved.
- **Database Outage**: Decision tree (primary down → check replica lag → promote if <1MB lag) reduced promotion decision from 15 minutes (ad-hoc discussion) to 2 minutes (follow runbook).
- **DDoS Response**: Numbered steps with success criteria enabled junior engineer to respond effectively during P1 incident without senior engineer (first time).

**Key lesson**: **Time-critical clarity (numbered steps, decision trees, success criteria) enables effective response under stress. Paragraphs fail during incidents.**
