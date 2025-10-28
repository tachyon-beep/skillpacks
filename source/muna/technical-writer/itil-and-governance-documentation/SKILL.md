---
name: itil-and-governance-documentation
description: Use when working in formal IT service management environments - covers change requests (RFC), service documentation, configuration management (CMDB), release docs, operational handover, business continuity (DR/RTO/RPO), capacity planning, and problem management
---

# ITIL and Governance Documentation

## Overview

Document systems for formal IT service management environments. Core principle: **Governance documentation enables controlled change and accountability**.

**Key insight**: Enterprise operations require structured documentation for change management, service delivery, and business continuity.

## When to Use

Load this skill when:
- Working in ITIL/ITSM environments
- Preparing change requests (RFC)
- Documenting enterprise services
- Creating disaster recovery plans

**Symptoms you need this**:
- "How do I document a production change?"
- Creating service catalog entries
- Writing DR/BCP documentation
- Formal change advisory board (CAB) processes

**Don't use for**:
- Informal/startup environments (too heavyweight)
- Quick fixes without governance

## Change Request (RFC) Documentation

### Pattern: Request for Change

```markdown
# RFC-1234: Database Schema Migration

## Change Type
**NORMAL** (Standard | Normal | Emergency)

## Summary
Migrate user authentication schema from legacy format to OAuth2-compatible format. Adds `oauth_provider` and `oauth_token` columns to `users` table.

## Business Justification
- Enable SSO integration with enterprise identity providers
- Customer requirement for Fortune 500 clients
- Revenue impact: Unblocks $500k in contracts

## Impact Analysis

### Affected Services
- User Authentication Service (primary impact)
- API Gateway (configuration change required)
- Mobile App (token format change)

### Affected Users/Teams
- All active users (15,000 users) - No visible impact
- Engineering team - Deployment coordination required
- Customer support - FAQs updated

### Dependencies
- Requires database maintenance window
- OAuth provider configuration must be complete
- Mobile app v2.5+ must be deployed first (rollout complete as of 2024-03-10)

### Risk Assessment
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Migration fails mid-process | Low | High | Transaction-based migration, automatic rollback |
| Users locked out post-migration | Medium | High | Keep legacy auth fallback for 30 days |
| Performance degradation | Low | Medium | Load testing completed, indexes added |

**Overall Risk**: MEDIUM

## Implementation Plan

### Pre-Implementation (1 day before)
1. **2024-03-19 10:00**: Send user notification ("scheduled maintenance 2024-03-20 02:00-04:00 UTC, no action required")
2. **2024-03-19 14:00**: Final QA testing in staging environment
3. **2024-03-19 16:00**: Freeze code deployments (change freeze begins)

### Implementation Window (2024-03-20 02:00-04:00 UTC)
**Duration**: 2 hours
**Teams required**: 2x SRE, 1x Database Admin, 1x Backend Engineer

1. **02:00**: Enable maintenance mode
   \```bash
   kubectl scale deployment/api-gateway --replicas=0
   \```
   - Success: No active API requests

2. **02:05**: Backup database
   \```bash
   pg_dump production_db > backup-pre-migration-20240320.sql
   aws s3 cp backup-pre-migration-20240320.sql s3://backups/
   \```
   - Success: Backup file uploaded (verify size ~5GB)

3. **02:15**: Run migration
   \```bash
   psql production_db < migration-20240320-oauth-schema.sql
   \```
   - Success: "ALTER TABLE" messages, no errors
   - Rollback if: Any ERROR message → Restore from backup

4. **02:45**: Deploy new API version
   \```bash
   kubectl set image deployment/api-gateway api=registry/api:v2.6.0
   kubectl scale deployment/api-gateway --replicas=6
   kubectl rollout status deployment/api-gateway
   \```
   - Success: "successfully rolled out"

5. **03:00**: Smoke tests
   \```bash
   # Test legacy auth (backward compatibility)
   curl -X POST https://api.example.com/auth/legacy -d '{"user":"test@example.com","pass":"fake_password"}'
   # Expected: 200 OK, token returned

   # Test OAuth auth (new feature)
   curl -X POST https://api.example.com/auth/oauth -d '{"provider":"google","code":"fake_code"}'
   # Expected: 200 OK, token returned
   \```

6. **03:15**: Monitor for 15 minutes
   - Check error rate dashboard: [link]
   - Check authentication success rate: [link]
   - Expected: Error rate <0.1%, auth success >99%

7. **03:30**: Disable maintenance mode
   \```bash
   # Restore normal operation
   \```

8. **03:45**: Post-deployment verification
   - Verify mobile app can authenticate
   - Verify web app can authenticate
   - Check customer support queue (any authentication issues?)

### Post-Implementation
1. **2024-03-20 08:00**: Team standup - review deployment, any issues?
2. **2024-03-20 09:00**: Send "maintenance complete" notification
3. **2024-03-21**: Monitor for 24 hours, watch for late-breaking issues
4. **2024-03-27**: Remove legacy auth fallback (7 days post-migration)

## Rollback Plan

### Trigger Conditions
Rollback if ANY of these occur:
- Migration script errors
- Auth success rate drops below 95%
- Critical bugs reported by >5 users
- Performance degradation >20%

### Rollback Procedure (30 minutes)
1. **Enable maintenance mode** (same as step 1 above)

2. **Restore database from backup**:
   \```bash
   aws s3 cp s3://backups/backup-pre-migration-20240320.sql ./
   psql production_db < backup-pre-migration-20240320.sql
   \```
   - Duration: ~15 minutes for 5GB database

3. **Revert to previous API version**:
   \```bash
   kubectl set image deployment/api-gateway api=registry/api:v2.5.9
   kubectl rollout restart deployment/api-gateway
   \```

4. **Verify rollback success**: Run smoke tests (same as implementation step 5)

5. **Disable maintenance mode**

6. **Post-rollback**: Investigate failure, update RFC with findings, reschedule

## Testing Plan

### Pre-Production Testing (Completed)
- [x] Unit tests pass (2024-03-15)
- [x] Integration tests pass (2024-03-16)
- [x] Load testing: 10,000 concurrent users, <200ms latency (2024-03-17)
- [x] Staging deployment: Migration successful (2024-03-18)

### Production Verification
- Smoke tests (included in implementation plan step 5)
- 24-hour monitoring period
- User acceptance: No critical issues reported within 7 days

## Approval Chain

| Role | Name | Approval Date | Status |
|------|------|---------------|--------|
| Backend Engineer | John Doe | 2024-03-14 | ✅ Approved |
| Database Admin | Jane Smith | 2024-03-14 | ✅ Approved |
| SRE Lead | Bob Johnson | 2024-03-15 | ✅ Approved |
| Product Manager | Alice Williams | 2024-03-15 | ✅ Approved |
| Change Advisory Board (CAB) | CAB Chair | 2024-03-18 | ✅ Approved |

**Change Authorized by**: CAB Chair (Alice Johnson) on 2024-03-18

## Communication Plan

| Audience | Message | Channel | Timing |
|----------|---------|---------|--------|
| All users | "Scheduled maintenance 2024-03-20 02:00-04:00 UTC. No action required." | Email, status page | 1 day before |
| Engineering team | Implementation details, on-call requirements | Slack #engineering | 1 day before |
| Customer support | FAQ updates, escalation path | Support portal | 1 day before |
| Executive team | Change summary, business impact | Email | 1 day before |
| All users | "Maintenance complete. New SSO features available." | Email, status page | After completion |
```

---

## Service Documentation

### Service Catalog Entry

```markdown
# Service: User Authentication API

## Service Overview
**Service ID**: SVC-0042
**Service Name**: User Authentication API
**Description**: Provides authentication and authorization for all company applications. Supports password-based login, OAuth SSO, and API token management.

## Service Owner
**Team**: Platform Security Team
**Primary Contact**: security-team@example.com
**Escalation**: VP Engineering (vp-eng@example.com)

## Service Details

### Purpose
Enable secure user authentication across all company products.

### Scope
**Included**:
- User login (password, OAuth)
- Session management
- API token generation/validation
- MFA enforcement

**Not Included**:
- User registration (owned by User Management Service)
- Password reset (owned by User Management Service)

### Users
- **Primary Users**: All application users (15,000 active users)
- **Internal Users**: API Gateway, Mobile App, Web App

## Service Level Agreement (SLA)

### Availability
**Target**: 99.9% uptime
**Measurement**: Monthly uptime excluding planned maintenance
**Planned Maintenance**: Max 4 hours/month, communicated 7 days in advance

### Performance
**Target**: 95th percentile latency <100ms
**Measurement**: API response time from request to auth token response

### Support Hours
**P1 (Critical)**: 24/7/365, 15-minute response time
**P2 (High)**: Business hours (9am-5pm PT), 2-hour response time
**P3/P4**: Business hours, next business day response

## Operational Level Agreements (OLA)

### Database Team OLA
**Commitment**: Database backup every 6 hours, 30-day retention
**Response Time**: 1-hour response for database issues affecting auth service

### Network Team OLA
**Commitment**: 99.95% network availability for auth service subnet
**Response Time**: 30-minute response for network issues

## Monitoring and Alerts

**Dashboard**: https://grafana.example.com/d/auth-service
**Key Metrics**:
- Uptime: Target 99.9%
- Latency (p95): Target <100ms
- Error rate: Target <0.1%
- Auth success rate: Target >99%

**Alerts**:
- P1: Service down (5-minute window), latency >500ms, error rate >1%
- P2: Elevated errors (>0.5%), auth success rate <95%

## Dependencies
- **Database**: PostgreSQL (SVC-0010)
- **Cache**: Redis (SVC-0015)
- **Secrets Management**: Vault (SVC-0020)

## Disaster Recovery
**RTO**: 1 hour (service restored within 1 hour of total failure)
**RPO**: 15 minutes (max 15 minutes of data loss acceptable)
**DR Site**: us-west-2 (failover from us-east-1)
**Testing**: Quarterly DR drills
```

---

## Business Continuity Documentation

### Disaster Recovery Plan

```markdown
# Disaster Recovery Plan: User Authentication Service

## RTO and RPO

**RTO (Recovery Time Objective)**: 1 hour
- Time to restore service after total regional failure

**RPO (Recovery Point Objective)**: 15 minutes
- Maximum acceptable data loss (last database backup)

## Disaster Scenarios

### Scenario 1: Complete Regional Outage (AWS us-east-1)
**Trigger**: All us-east-1 availability zones unavailable
**Impact**: PRIMARY - Service completely unavailable
**Recovery**: Failover to us-west-2 (DR region)

### Scenario 2: Database Corruption
**Trigger**: Database integrity check fails, data corruption detected
**Impact**: CRITICAL - Service degraded or unavailable
**Recovery**: Restore from most recent backup (15-minute data loss)

### Scenario 3: Critical Security Incident
**Trigger**: Confirmed breach, attacker has system access
**Impact**: CRITICAL - Service must be taken offline immediately
**Recovery**: Forensic analysis, patch vulnerabilities, restore from known-good state

## Failover Procedure (Scenario 1: Regional Outage)

### Prerequisites
- DR region (us-west-2) database in standby replication mode
- DR region compute resources pre-provisioned (scaled to zero, can scale up instantly)

### Failover Steps (45 minutes)

1. **Declare disaster** (5 minutes)
   - Incident commander confirms regional outage
   - Notify stakeholders: "Initiating failover to DR region"

2. **Promote DR database to primary** (10 minutes)
   \```bash
   # SSH to DR database server
   ssh db-dr.us-west-2.internal

   # Promote standby to primary
   sudo -u postgres pg_ctl promote -D /var/lib/postgresql/data

   # Verify promotion
   psql -c "SELECT pg_is_in_recovery();"  # Should return 'f' (not in recovery = primary)
   \```

3. **Scale up DR compute** (10 minutes)
   \```bash
   # Scale API servers from 0 to 6 replicas
   kubectl --context us-west-2 scale deployment/auth-api --replicas=6

   # Wait for readiness
   kubectl --context us-west-2 rollout status deployment/auth-api
   \```

4. **Update DNS to DR region** (10 minutes)
   \```bash
   # Update Route53 to point to DR load balancer
   aws route53 change-resource-record-sets --hosted-zone-id Z123 --change-batch file://dns-failover.json

   # Verify DNS propagation (may take 5-10 minutes for full propagation)
   dig auth-api.example.com
   \```

5. **Verify service health** (5 minutes)
   - Smoke test: Authenticate test user
   - Check monitoring dashboard: Error rate, latency
   - Success criteria: Error rate <1%, latency <200ms

6. **Communicate completion** (5 minutes)
   - Status page: "Service restored, operating from backup region"
   - Engineering team: "Failover complete, monitor for issues"
   - Customer support: "Service operational, escalate any issues to incident channel"

### Failback Procedure (After Primary Region Restored)

**Wait period**: 24 hours of stable DR operation before failback

1. Restore primary region database from DR backup
2. Establish replication DR → Primary (reverse direction)
3. During maintenance window: Failover back to primary (same procedure as above)

## Testing Schedule

**DR Drills**: Quarterly (January, April, July, October)
**Procedure**: Simulate regional failure, execute failover, measure RTO/RPO
**Success Criteria**: Complete failover within 1-hour RTO, data loss <15 minutes
```

---

## Cross-References

**Use WITH this skill**:
- `muna/technical-writer/documentation-structure` - RFC and service docs follow structured patterns
- `muna/technical-writer/incident-response-documentation` - DR procedures are incident responses

## Real-World Impact

**Governance documentation using these patterns**:
- **Database Migration RFC**: Change Advisory Board approved in first review (vs 3-review avg) due to comprehensive impact analysis, rollback plan, and testing documentation.
- **Service Catalog**: SLA documentation enabled proactive capacity planning. When auth service approached 99.9% SLA threshold, automatic scaling prevented SLA breach.
- **DR Plan**: Quarterly DR drill revealed 2-hour RTO vs documented 1-hour. Plan updated with pre-provisioned DR resources, actual RTO now 45 minutes.

**Key lesson**: **Structured governance documentation (RFC, SLA, DR plans) enables controlled change, accountability, and measurable service delivery.**
