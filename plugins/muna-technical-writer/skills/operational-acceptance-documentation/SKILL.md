---
name: operational-acceptance-documentation
description: Security authorization (SSP/SAR/ATO), operational readiness, go-live approval
---

# Operational Acceptance Documentation

## Overview

Prepare complete acceptance packages for production deployment. Core principle: **Document readiness, risks, and acceptance criteria for informed go-live decisions**.

**Key insight**: Acceptance documentation enables stakeholders to make informed risk decisions about production deployment.

## When to Use

Load this skill when:
- Preparing systems for production launch
- Seeking executive go-live approval
- Completing operational handover
- Government/defense system authorization

**Symptoms you need this**:
- "How do I get approval to launch?"
- Preparing production readiness checklist
- Creating go-live approval package
- Operational handover to support team

**Don't use for**:
- Development/staging deployments
- Internal-only tools (unless high-risk)

## Production Readiness Checklist

### Infrastructure Readiness

```markdown
## Infrastructure Readiness

### Compute Resources
- [ ] Production servers provisioned (6x API servers, 2x database servers)
- [ ] Auto-scaling configured (scale 2-20 instances based on CPU >70%)
- [ ] Load balancer configured with health checks
- [ ] SSL/TLS certificates installed and valid (expires 2025-12-01)

### Storage
- [ ] Database provisioned (PostgreSQL 14, 500GB storage)
- [ ] Database backups configured (automated hourly backups, 30-day retention)
- [ ] Backup restoration tested (RTO: 1 hour, RPO: 1 hour)

### Network
- [ ] VPC configured with public/private subnets
- [ ] Firewall rules implemented (allow HTTPS 443, deny all other inbound)
- [ ] DNS configured (api.example.com → load balancer)

### Monitoring and Logging
- [ ] Application metrics instrumented (Prometheus)
- [ ] Logs centralized (CloudWatch Logs, 90-day retention)
- [ ] Dashboards created ([Grafana dashboard link])
- [ ] Alerts configured (error rate, latency, uptime)

### Security
- [ ] Secrets stored in secrets manager (not environment variables)
- [ ] TLS 1.3 enforced
- [ ] Authentication implemented (MFA for admins)
- [ ] Security scan completed (no HIGH/CRITICAL findings)

**Infrastructure Status**: ✅ READY (all criteria met)
```

---

## Operational Readiness Checklist

```markdown
## Operational Readiness

### Monitoring Coverage
- [ ] **Availability**: Uptime monitoring ([UptimeRobot link])
- [ ] **Performance**: Latency tracking (p50, p95, p99)
- [ ] **Errors**: Error rate monitoring (<0.1% threshold)
- [ ] **Business metrics**: User signups, API calls, revenue

**Success criteria**: All critical metrics have dashboards + alerts

### Alerting Configuration
- [ ] **P1 alerts** (PagerDuty): Service down, error rate >1%, security incident
- [ ] **P2 alerts** (Slack #ops): Elevated errors >0.5%, latency >500ms
- [ ] **P3 alerts** (Email): Performance degradation, capacity warnings

**Success criteria**: Alerts tested and verified to fire correctly

### Backup and Recovery
- [ ] **Backup procedure**: Automated hourly PostgreSQL dumps to S3
- [ ] **Backup testing**: Restored from backup on 2024-03-10 (successful)
- [ ] **Recovery time**: 1-hour RTO verified
- [ ] **Recovery point**: 1-hour RPO (acceptable data loss)

**Success criteria**: Restore from backup completes within RTO

### Runbooks and Documentation
- [ ] **Incident response runbooks**: Database outage, API errors, security incidents
- [ ] **Operational procedures**: Deployment, rollback, scaling
- [ ] **Architecture documentation**: System diagram, data flows, integrations
- [ ] **API documentation**: Endpoint reference, authentication guide

**Success criteria**: On-call engineer can respond to P1 incident using runbooks alone

**Operational Status**: ✅ READY (all criteria met)
```

---

## Test and Evaluation Documentation

### Test Summary Report

```markdown
# Test Summary Report: Customer Portal Launch

## Test Objectives
1. Verify functional requirements (user registration, login, profile management)
2. Validate performance requirements (p95 latency <200ms, support 1000 concurrent users)
3. Confirm security requirements (authentication, authorization, data encryption)

## Test Methodology

### Functional Testing
- **Unit tests**: 487 tests, 100% pass rate
- **Integration tests**: 156 tests, 100% pass rate
- **End-to-end tests**: 45 scenarios, 44 passed, 1 defect (LOW severity, workaround available)

### Performance Testing
- **Load test**: 1000 concurrent users, 10,000 requests/min
  - p50 latency: 45ms ✅
  - p95 latency: 180ms ✅ (target: <200ms)
  - p99 latency: 350ms ⚠️ (target: <500ms)
  - Error rate: 0.02% ✅ (target: <0.1%)

### Security Testing
- **Vulnerability scan**: Nessus scan completed 2024-03-15
  - Critical: 0 ✅
  - High: 0 ✅
  - Medium: 3 (remediated)
  - Low: 8 (accepted risk)
- **Penetration test**: External pentest completed 2024-03-18
  - HIGH findings: 1 (SQL injection, fixed on 2024-03-19)
  - MEDIUM findings: 2 (remediated)

## Defect Summary

| Defect ID | Severity | Description | Status | Disposition |
|-----------|----------|-------------|--------|-------------|
| DEF-001 | LOW | Profile image upload fails for files >10MB | Open | Workaround: Resize before upload (documented) |
| DEF-002 | MEDIUM | Password reset email delayed 5-10 minutes | Fixed | Fixed on 2024-03-20 |
| DEF-003 | HIGH | SQL injection in /api/users | Fixed | Fixed on 2024-03-19, re-tested |

## Test Completion Criteria

- [ ] ✅ All HIGH/CRITICAL defects fixed
- [ ] ✅ All MEDIUM defects fixed or have workarounds
- [ ] ✅ LOW defects documented (1 open, workaround available)
- [ ] ✅ Performance requirements met
- [ ] ✅ Security requirements met (no HIGH/CRITICAL findings)

**Test Status**: ✅ PASSED (all criteria met, 1 LOW defect acceptable)
```

---

## Go-Live Approval Package

### Executive Summary

```markdown
# Go-Live Approval Request: Customer Portal

## System Overview
**System Name**: Customer Portal
**Purpose**: Enable customers to self-serve account management, reducing support tickets by 40%
**Business Value**: $2M annual revenue enabler (enterprise customers require self-service portal)
**Target Launch**: 2024-04-01

## Readiness Status

### Infrastructure: ✅ READY
- All production servers provisioned and tested
- Auto-scaling configured
- Backups automated and tested (1-hour RTO/RPO)

### Operations: ✅ READY
- Monitoring and alerting configured
- Runbooks complete
- On-call rotation staffed (3 SREs, 2 backend engineers)

### Testing: ✅ PASSED
- Functional tests: 100% pass (1 LOW defect with workaround)
- Performance tests: p95 latency 180ms (target: <200ms)
- Security tests: 0 HIGH/CRITICAL findings

### Security Authorization: ✅ AUTHORIZED
- ATO granted on 2024-03-25 (valid for 3 years)
- POA&M with 2 LOW-risk items (tracked, non-blocking)

## Residual Risks

### Risk 1: Performance Degradation Above 1000 Users (MEDIUM)
**Description**: Load testing validated 1000 concurrent users. Performance above 1000 users unknown.
**Mitigation**:
- Auto-scaling configured to add capacity at 70% CPU
- Gradual rollout plan (100 users week 1, 500 week 2, all users week 4)
- Performance monitoring with alerts at 800ms latency threshold
**Accepted by**: CTO on 2024-03-28

### Risk 2: Profile Image Upload Limitation (LOW)
**Description**: Images >10MB fail to upload (DEF-001)
**Mitigation**:
- Workaround documented in user help center
- Fix planned for v1.1 release (2024-05-01)
**Accepted by**: Product Manager on 2024-03-28

## Launch Criteria

### Success Metrics
**Immediate (Week 1)**:
- Uptime: >99% (target: 99.9%)
- Error rate: <0.5% (target: <0.1%)
- p95 latency: <300ms (target: <200ms)

**Medium-term (Month 1)**:
- User adoption: 30% of customers use portal
- Support ticket reduction: 20% decrease

### Abort Criteria
**Immediate rollback if**:
- Uptime drops below 95% for >1 hour
- Error rate exceeds 5%
- Data breach or security incident
- Critical functionality broken for >50% of users

### Monitoring Plan
- **Real-time**: Grafana dashboard monitored by on-call
- **Daily**: Morning standup reviews previous 24 hours
- **Weekly**: Executive summary report (metrics vs targets)

## Rollback Plan

**Trigger**: Any abort criterion met

**Rollback Procedure** (30 minutes):
1. Enable maintenance page
2. Scale production deployment to 0 replicas
3. Restore database from pre-launch backup (if data changes occurred)
4. Re-enable previous customer support workflow
5. Communicate to customers via email

**Testing**: Rollback procedure tested in staging on 2024-03-27 (successful, 25-minute duration)

## Recommendation

**Status**: ✅ APPROVED FOR LAUNCH

All readiness criteria met. Residual risks identified and accepted by stakeholders. Launch criteria defined with clear success metrics and abort criteria. Rollback plan tested and ready.

**Requested Approval**: Executive Go-Live Approval

**Approvals Required**:
- [ ] VP Engineering (technical readiness)
- [ ] CTO (security and risk acceptance)
- [ ] VP Product (business value and user experience)
- [ ] CFO (budget and revenue impact)
```

---

## Operational Handover Checklist

```markdown
# Operational Handover: Customer Portal

## Knowledge Transfer

### Documentation Delivered
- [ ] ✅ Architecture documentation (`/docs/architecture.md`)
- [ ] ✅ API reference (`/docs/api-reference.md`)
- [ ] ✅ Runbooks (`/runbooks/` - 12 runbooks)
- [ ] ✅ Deployment procedures (`/docs/deployment.md`)
- [ ] ✅ Troubleshooting guide (`/docs/troubleshooting.md`)

### Training Completed
- [ ] ✅ On-call training (2024-03-20): 3 SREs, 2 backend engineers
- [ ] ✅ Runbook walkthrough (2024-03-22): All on-call staff
- [ ] ✅ Incident response drill (2024-03-25): Simulated database outage, responded successfully

### Handoff Meeting
- **Date**: 2024-03-28
- **Attendees**: Development team (6), Operations team (5), Product (2)
- **Agenda**:
  1. System overview and architecture
  2. Common issues and troubleshooting
  3. Escalation paths and contact information
  4. Q&A session
- **Outcome**: ✅ Operations team confident in supporting system

## Support Model

### On-Call Rotation
**Primary On-Call**: Rotating weekly schedule (3 SREs)
**Backup On-Call**: Backend engineer (2-person rotation)

**Schedule**: https://pagerduty.example.com/schedules/customer-portal

### Escalation Paths

**P1 (Critical)**:
1. Primary on-call (page immediately)
2. If no response in 5 min → Backup on-call
3. If no resolution in 30 min → Incident commander
4. If ongoing after 1 hour → VP Engineering

**P2 (High)**:
1. Primary on-call (page)
2. If no response in 15 min → Backup on-call
3. If no resolution in 4 hours → Team lead

**Contacts**:
- Primary on-call: [PagerDuty link]
- Incident commander: John Doe (+1-555-0100)
- Team lead: Jane Smith (+1-555-0200)
- VP Engineering: Bob Johnson (+1-555-0300)

### SLA Commitments
**Uptime**: 99.9% (measured monthly)
**Performance**: p95 latency <200ms
**Support Response**:
- P1: 15-minute response, 4-hour resolution target
- P2: 2-hour response, 1-day resolution target
- P3: Next business day response

## Acceptance Criteria Met

- [ ] ✅ All documentation delivered and reviewed
- [ ] ✅ Operations team trained
- [ ] ✅ Incident response drill successful
- [ ] ✅ On-call rotation staffed
- [ ] ✅ Escalation paths defined
- [ ] ✅ SLA commitments documented

**Handover Status**: ✅ COMPLETE

**Signed Off**:
- Development Team Lead: John Doe (2024-03-28)
- Operations Team Lead: Jane Smith (2024-03-28)
```

---

## Cross-References

**Use WITH this skill**:
- `ordis/security-architect/security-authorization-and-accreditation` - For government/defense ATO requirements
- `muna/technical-writer/itil-and-governance-documentation` - For RFC and service documentation

## Real-World Impact

**Systems using operational acceptance documentation**:
- **Customer Portal Launch**: Go-live approval package enabled same-day executive approval (vs 1-week review cycle). Clear risk acceptance + rollback plan gave confidence to approve.
- **Government System**: Complete readiness checklist (infrastructure, operations, testing, security authorization) passed IRAP assessment on first attempt. Assessor: "Most comprehensive readiness documentation in 5 years".
- **Operational Handover**: Training + runbooks + incident drill enabled junior SRE to respond to P1 database outage successfully within 45 minutes (first week post-handover).

**Key lesson**: **Comprehensive acceptance documentation (readiness, risks, criteria, handover) enables informed go-live decisions and smooth operational transitions.**
