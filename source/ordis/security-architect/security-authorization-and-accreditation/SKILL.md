---
name: security-authorization-and-accreditation
description: Use when preparing systems for production authorization in government/defense environments - covers ATO (Authority to Operate), AIS (Authorization to Interconnect), T&E (Test & Evaluation), SSP, SAR, POA&M, and continuous monitoring processes
---

# Security Authorization and Accreditation

## Overview

Navigate government/defense security authorization processes. Core principle: **Authorization is risk acceptance by an official with authority**.

**Key insight**: ATO is not a checklist - it's formal risk acceptance documentation enabling informed decision-making by leadership.

## When to Use

Load this skill when:
- Deploying systems for government/defense
- Preparing for ATO (Authority to Operate)
- Connecting to classified networks
- Formal security testing and evaluation

**Symptoms you need this**:
- "How do I get ATO for production?"
- Government/defense contracts requiring authorization
- "What is an SSP/SAR/POA&M?"
- Preparing for security assessment

**Don't use for**:
- Commercial compliance (use `ordis/security-architect/compliance-awareness-and-mapping`)
- General security (use `ordis/security-architect/security-controls-design`)

## The Authorization Process

### Core Concept: Risk Management Framework (RMF)

**RMF** (NIST SP 800-37) has 7 steps:

```
1. PREPARE → 2. CATEGORIZE → 3. SELECT → 4. IMPLEMENT → 5. ASSESS → 6. AUTHORIZE → 7. MONITOR
     ↓              ↓             ↓           ↓             ↓            ↓             ↓
  Planning    Impact Level   Controls    Build System    Test      Get ATO    Ongoing Ops
```

### Step 1: PREPARE (Pre-Authorization)

**Activities**:
- Define authorization boundary (what's in scope?)
- Identify Authorizing Official (AO) - person with authority to accept risk
- Assemble authorization team (ISSM, ISSO, assessors)
- Review organizational risk tolerance

**Deliverable**: Authorization strategy and team assigned

---

### Step 2: CATEGORIZE (Impact Analysis)

**Activities**:
- Security categorization using FIPS 199
- Determine impact level: Low, Moderate, or High
- Based on Confidentiality, Integrity, Availability (CIA)

**Impact Levels**:
```
Low: Limited adverse effect
Moderate: Serious adverse effect
High: Severe or catastrophic adverse effect
```

**Example**:
```
System: Healthcare Records Database
Confidentiality: HIGH (patient privacy breach = severe impact)
Integrity: HIGH (incorrect medical records = life-threatening)
Availability: MODERATE (temporary outage = serious but not life-threatening)

Overall System Impact: HIGH (highest of C/I/A)
```

**Deliverable**: Security categorization document

---

### Step 3: SELECT (Control Selection)

**Activities**:
- Select control baseline (NIST SP 800-53)
- Low baseline → 125 controls
- Moderate baseline → 325 controls
- High baseline → 421 controls
- Tailor controls (add/remove based on organizational needs)

**Control Families**:
- AC (Access Control)
- AT (Awareness and Training)
- AU (Audit and Accountability)
- CA (Assessment, Authorization, and Monitoring)
- CM (Configuration Management)
- CP (Contingency Planning)
- IA (Identification and Authentication)
- IR (Incident Response)
- MA (Maintenance)
- MP (Media Protection)
- PE (Physical and Environmental Protection)
- PL (Planning)
- PS (Personnel Security)
- RA (Risk Assessment)
- SA (System and Services Acquisition)
- SC (System and Communications Protection)
- SI (System and Information Integrity)

**Deliverable**: Control baseline with tailoring decisions

---

### Step 4: IMPLEMENT (Build System with Controls)

**Activities**:
- Implement selected security controls
- Document implementation in SSP (System Security Plan)
- Common control inheritance (use organization-wide controls)

**Example Control Implementation**:
```
Control: AC-2 (Account Management)
Implementation:
- All accounts created via ServiceNow ticket
- Manager approval required
- Accounts disabled after 30 days inactivity
- Monthly access reviews by data owners
- Evidence: ServiceNow workflow, access review reports
```

**Deliverable**: Implemented system with documented controls (SSP)

---

### Step 5: ASSESS (Security Assessment)

**Activities**:
- Independent assessment by certified assessor
- Test controls (interviews, configuration review, penetration testing)
- Document findings in SAR (Security Assessment Report)
- Classify findings by severity (Critical/High/Medium/Low)

**Assessment Methods**:
- **Examine**: Review documentation, configurations, logs
- **Interview**: Question staff about procedures
- **Test**: Execute tests (penetration test, scan, functional test)

**Deliverable**: SAR (Security Assessment Report) with findings

---

### Step 6: AUTHORIZE (Risk Acceptance)

**Activities**:
- Remediate or accept findings
- Create POA&M (Plan of Action & Milestones) for residual risks
- Authorizing Official reviews SSP + SAR + POA&M
- AO makes risk acceptance decision
- Issue ATO (Authority to Operate) or denial

**ATO Types**:
- **Full ATO**: System fully compliant, authorized for 3 years
- **Interim ATO (IATO)**: Temporary authorization (6-12 months) with conditions
- **Denial**: System does not meet minimum security requirements

**Deliverable**: ATO Letter from Authorizing Official

---

### Step 7: MONITOR (Continuous Monitoring)

**Activities**:
- Ongoing assessment of controls
- Change impact analysis (new vulnerabilities, configuration changes)
- Update POA&M as risks remediated
- Annual re-assessment or earlier if major changes
- Trigger re-authorization when needed

**Deliverable**: Continuous monitoring reports, updated POA&M

---

## Key Documents

### 1. SSP (System Security Plan)

**Purpose**: Comprehensive description of the system and its security controls.

**Contents**:
```markdown
# System Security Plan (SSP)

## 1. System Identification
- System name, acronym, unique identifier
- System owner, ISSO, ISSM contact info
- Authorization boundary (what's included/excluded)

## 2. System Categorization
- FIPS 199 categorization (Low/Moderate/High)
- CIA impact levels with justification
- Overall system impact level

## 3. System Description
- System purpose and functionality
- Users and use cases
- Data types processed (PII, classified, etc.)
- System architecture diagram
- Network diagram with trust boundaries
- Technology stack (OS, database, languages)

## 4. Authorization Boundary
- Components within boundary (servers, databases, networks)
- External connections (APIs, data feeds)
- Interconnection agreements for external systems

## 5. Security Control Implementation
For EACH control in baseline:

### AC-2: Account Management
**Control Requirement**: Organization manages information system accounts
**Implementation**:
- ServiceNow ticketing system for account requests
- Manager approval required via workflow
- Automated 30-day inactivity disablement
- Monthly access reviews by data owners
**Responsible Role**: System Administrator
**Assessment Procedure**:
- Interview: Ask sysadmins about account creation process
- Examine: Review ServiceNow tickets, approval records
- Test: Attempt to create account without approval (should fail)
**Evidence Location**:
- ServiceNow workflow documentation: /docs/account-mgmt.pdf
- Sample access review: /evidence/access-review-2025-01.xlsx

(Repeat for all ~125-421 controls depending on baseline)

## 6. Related Documents
- Contingency Plan (CP)
- Incident Response Plan (IRP)
- Configuration Management Plan (CMP)
- Continuous Monitoring Plan

## 7. Approval Signatures
- System Owner: [Signature] [Date]
- ISSO: [Signature] [Date]
- ISSM: [Signature] [Date]
```

**Typical Length**: 200-500 pages for HIGH system

---

### 2. SAR (Security Assessment Report)

**Purpose**: Independent assessment results documenting control effectiveness.

**Contents**:
```markdown
# Security Assessment Report (SAR)

## Executive Summary
- System assessed: [Name]
- Assessment dates: [Start] to [End]
- Assessor: [Name, Credentials]
- Overall assessment: [Pass with findings / Conditional / Fail]
- Critical findings: [Count]
- High findings: [Count]
- Recommendations: [Summary]

## Assessment Scope
- Controls assessed: [List control families or specific controls]
- Assessment methods: Examine, Interview, Test
- Limitations: [Any scope exclusions or constraints]

## Assessment Results by Control

### AC-2: Account Management
**Assessment Procedures**:
1. Interview: System administrators (2025-03-15)
2. Examine: ServiceNow workflow, access review reports
3. Test: Attempted unauthorized account creation

**Finding AC-2-001: MEDIUM**
**Status**: OPEN
**Description**: Access reviews conducted monthly but no evidence of remediation for identified excessive privileges. 3 accounts with admin access had not logged in for 6 months but remain enabled.
**Risk**: Dormant privileged accounts increase attack surface.
**Recommendation**: Implement automated disablement of inactive admin accounts within 30 days. Conduct immediate review of all privileged accounts.
**AO Response**: Accepted. Remediation planned in POA&M item #5.

**Overall Control Result**: PARTIALLY SATISFIED (findings require remediation)

(Repeat for all assessed controls)

## Findings Summary

| Finding ID | Control | Severity | Status | Remediation Due |
|------------|---------|----------|--------|-----------------|
| AC-2-001 | AC-2 | Medium | Open | 2025-06-01 |
| IA-5-001 | IA-5 | High | Open | 2025-04-15 |
| SC-7-001 | SC-7 | Low | Open | 2025-08-01 |

## Assessor Recommendation
Recommend INTERIM ATO for 6 months conditional on remediation of HIGH findings within 30 days. Re-assessment required before full ATO.

**Assessor Signature**: [Name] [Date]
```

**Typical Length**: 100-300 pages

---

### 3. POA&M (Plan of Action & Milestones)

**Purpose**: Track remediation of security weaknesses and residual risks.

**Contents**:
```markdown
# Plan of Action & Milestones (POA&M)

## POA&M Item #1
**Finding ID**: IA-5-001 (from SAR)
**Control**: IA-5 (Authenticator Management)
**Weakness**: Password policy allows 8-character passwords; NIST recommends minimum 12 characters.
**Risk Level**: HIGH
**Risk Description**: Weak passwords vulnerable to brute-force attacks. Estimated 15% of user accounts have 8-character passwords.
**Milestones**:
- [ ] Milestone 1: Update password policy to require 12 characters (2025-04-01) - System Admin
- [ ] Milestone 2: Force password reset for all 8-char passwords (2025-04-15) - System Admin
- [ ] Milestone 3: Verify 100% compliance via audit query (2025-04-20) - ISSO
- [ ] Milestone 4: Provide evidence to assessor (2025-04-25) - ISSO
**Resources Required**: 40 hours engineering time, communication campaign for users
**Status**: IN PROGRESS (Milestone 1 complete, Milestone 2 in progress)
**Scheduled Completion**: 2025-04-25
**Actual Completion**: [TBD]

## POA&M Item #2
**Finding ID**: AC-2-001 (from SAR)
**Control**: AC-2 (Account Management)
**Weakness**: Dormant privileged accounts not automatically disabled.
**Risk Level**: MEDIUM
**Milestones**:
- [ ] Implement automated script to disable admin accounts after 30 days inactivity (2025-05-01)
- [ ] Immediate review and disable of current dormant admin accounts (2025-04-10)
- [ ] Monthly verification report to ISSO (ongoing)
**Scheduled Completion**: 2025-05-01

## POA&M Item #3 (Risk Acceptance)
**Finding ID**: SC-8-001
**Control**: SC-8 (Transmission Confidentiality)
**Weakness**: Legacy internal API uses HTTP (not HTTPS) for non-sensitive configuration data.
**Risk Level**: LOW
**Recommendation**: Migrate to HTTPS
**AO Decision**: RISK ACCEPTED
**Justification**:
- API only accessible on isolated management network (not exposed to internet)
- Data transmitted is non-sensitive system configuration (no PII, credentials, or classified data)
- Migration to HTTPS requires vendor upgrade (cost: $50k, timeline: 12 months)
- Risk mitigated by network segmentation
**Acceptance Date**: 2025-03-20
**Acceptance Authority**: Authorizing Official [Name]
**Re-evaluation Date**: 2026-03-20 (annual review)

---

**POA&M Metrics**:
- Total items: 15
- Critical: 0
- High: 1 (in progress)
- Medium: 8 (6 in progress, 2 accepted risk)
- Low: 6 (4 in progress, 2 accepted risk)
```

**Updates**: Monthly or when milestone completed

---

### 4. ATO Letter (Authority to Operate)

**Purpose**: Formal authorization from Authorizing Official.

**Contents**:
```markdown
MEMORANDUM FOR: System Owner, [Name]

SUBJECT: Authority to Operate (ATO) for [System Name]

1. AUTHORIZATION DECISION

After careful review of the System Security Plan (SSP), Security Assessment Report (SAR), and Plan of Action & Milestones (POA&M), I hereby grant an INTERIM AUTHORITY TO OPERATE for [System Name] for a period of SIX (6) MONTHS, effective [Start Date] to [End Date].

2. AUTHORIZATION BOUNDARY

This authorization covers the system as described in SSP version 2.1, including:
- Application servers (5x EC2 instances)
- PostgreSQL database cluster (2 nodes)
- AWS resources within VPC vpc-abc123
- Users: 500 internal staff with SECRET clearance

3. CONDITIONS OF AUTHORIZATION

This interim ATO is granted subject to the following conditions:

a) HIGH-severity finding IA-5-001 (password policy weakness) must be remediated within 30 days (by 2025-04-25). Failure to remediate will result in suspension of ATO.

b) All MEDIUM-severity findings must be remediated or risk-accepted within 90 days (by 2025-07-01).

c) Monthly POA&M status reports submitted to ISSM.

d) No major changes to system without prior authorization (change impact analysis required).

e) Full security re-assessment required at end of 6-month period for consideration of full 3-year ATO.

4. RESIDUAL RISKS ACCEPTED

I accept the following residual risks as documented in POA&M:
- SC-8-001 (LOW): HTTP on internal API (mitigated by network segmentation)

5. CONTINUOUS MONITORING

System owner must maintain continuous monitoring program including:
- Weekly vulnerability scans
- Monthly access reviews
- Quarterly control spot-checks
- Annual contingency plan testing

6. AUTHORIZATION TERMINATION

This authorization may be suspended or revoked if:
- Conditions of authorization not met
- New vulnerabilities with HIGH or CRITICAL severity discovered
- Significant security incident occurs
- Major system changes without authorization

Authorizing Official: [Signature]
[Name], [Title]
[Date]
```

---

## Authorization Types

### ATO (Authority to Operate)

**Full ATO**:
- Duration: 3 years (typically)
- Conditions: All HIGH/CRITICAL findings remediated or risk-accepted
- Re-authorization: Every 3 years or upon major change

**Interim ATO (IATO)**:
- Duration: 6-12 months
- Conditions: HIGH findings have remediation plan, progress tracked
- Purpose: Allow operation while remediating non-critical issues
- Requires full assessment at expiration

**Denial**:
- System does not meet minimum security requirements
- CRITICAL/HIGH findings with unacceptable risk
- Must remediate before re-submission

---

### AIS (Authorization to Interconnect)

**Purpose**: Authorization to connect two systems across trust boundaries.

**When needed**:
- Connecting to external networks
- Connecting systems at different classification levels
- Sharing data between organizations

**Requirements**:
- Both systems have current ATO
- Interconnection Security Agreement (ISA)
- Boundary protection documented
- Data flow diagrams
- Security controls at boundary (firewall, data diode, etc.)

**ISA Contents**:
```markdown
# Interconnection Security Agreement (ISA)

## Systems
- System A: [Name], ATO valid until [Date]
- System B: [Name], ATO valid until [Date]

## Data Flows
- Direction: System A → System B
- Data type: Transaction records (OFFICIAL classification)
- Frequency: Real-time API calls
- Volume: 10,000 records/day

## Boundary Protection
- Firewall: Palo Alto PA-5000 at boundary
- Allowed traffic: HTTPS port 443 only, source IP whitelisted
- Data validation: System B validates all incoming data
- Encryption: TLS 1.3 with mutual authentication

## Security Controls
- AC-4: Information flow enforcement (firewall rules)
- SC-7: Boundary protection (dedicated firewall)
- SC-8: Transmission confidentiality (TLS 1.3)

## Roles and Responsibilities
- System A ISSO: Monitor outbound connections, alert on anomalies
- System B ISSO: Validate incoming data, monitor for security events
- Network team: Maintain firewall, apply security patches

## Incident Response
- Security incident on either system triggers review of interconnection
- Contact: [System A ISSO], [System B ISSO]

## Review and Re-authorization
- Annual review of ISA
- Re-authorization required if either system ATO expires or major changes

**Approvals**:
- System A AO: [Signature] [Date]
- System B AO: [Signature] [Date]
```

---

### T&E (Test & Evaluation)

**Purpose**: Independent security testing before authorization.

**Testing Types**:

#### 1. Vulnerability Assessment
```bash
# Authenticated vulnerability scan
nessus scan --authenticated --target 10.0.1.0/24 \
  --policy "Government High Baseline" \
  --output vuln-report.pdf

# Results:
# Critical: 0
# High: 2 (outdated SSL certificates, missing patches)
# Medium: 15
# Low: 43
```

#### 2. Penetration Testing
```markdown
# Penetration Test Report

## Scope
- External penetration test (internet-facing)
- Internal penetration test (insider threat simulation)
- Duration: 2 weeks

## Rules of Engagement
- No DoS attacks
- No social engineering of executives
- Data exfiltration limited to test accounts

## Findings
### FINDING PT-001: HIGH
**Vulnerability**: SQL injection in /api/users endpoint
**Exploit**: Extracted 10 test user records via injection
**Impact**: Attacker could exfiltrate entire user database
**Recommendation**: Implement parameterized queries, input validation
**Remediation**: Developer committed fix (commit abc123), deployed 2025-03-20
**Verification**: Re-tested 2025-03-22, vulnerability no longer exploitable

## Summary
- High findings: 1 (remediated)
- Medium findings: 3 (2 remediated, 1 in POA&M)
- Low findings: 8 (accepted risk)
```

#### 3. Functional Security Testing
```markdown
# Functional Security Test: Access Control

## Test Case TC-AC-001: Unauthorized Access Attempt
**Objective**: Verify user cannot access resources without authorization
**Procedure**:
1. Login as user@example.com (no admin privileges)
2. Attempt to access /admin/users endpoint
3. Expected result: 403 Forbidden
**Actual Result**: 403 Forbidden ✅
**Status**: PASS

## Test Case TC-AC-002: Privilege Escalation
**Objective**: Verify user cannot elevate own privileges
**Procedure**:
1. Login as user@example.com
2. Attempt to modify own role via API: PUT /api/users/me {"role": "admin"}
3. Expected result: 403 Forbidden
**Actual Result**: 200 OK, role changed to admin ❌
**Status**: FAIL - CRITICAL
**Remediation Required**: Implement server-side role validation, users cannot modify own roles
```

**T&E Report**: Submitted to assessor as input to SAR.

---

## Continuous Monitoring

**Purpose**: Ongoing assurance that controls remain effective.

### Monitoring Activities

#### 1. Automated Scanning
```bash
# Weekly vulnerability scan
cron: 0 2 * * 1 /opt/nessus/scan.sh

# Monthly configuration compliance check
cron: 0 3 1 * * /opt/scap/compliance-check.sh
```

#### 2. Access Reviews
```markdown
# Monthly Access Review

## Review Date: 2025-04-01
## Reviewer: Data Owner (Jane Smith)

| User | Role | Last Login | Action |
|------|------|------------|--------|
| john.doe@example.com | Admin | 2025-03-28 | ✅ Retain |
| alice.smith@example.com | User | 2025-03-25 | ✅ Retain |
| bob.jones@example.com | Admin | 2024-12-15 | ❌ Remove (dormant 4 months) |

**Actions Taken**:
- Disabled bob.jones@example.com on 2025-04-02
- Notified ISSO of access review completion
```

#### 3. Change Impact Analysis
```markdown
# Change Impact Analysis: Database Upgrade

## Change Description
Upgrade PostgreSQL from 13.2 to 13.10 (security patches)

## Security Impact Assessment
- Controls affected: SC-28 (Protection of information at rest - encryption still enabled)
- New vulnerabilities: CVE-2024-XXXX patched in 13.10
- Configuration changes: None (encryption settings preserved)

## Authorization Impact
- Change Type: Minor (security patch)
- ATO Impact: None (no re-authorization required per continuous monitoring plan)
- ISSO Notification: Required (notified 2025-03-15)

## Testing
- Tested in dev environment: 2025-03-10 (PASS)
- Contingency plan: Snapshot before upgrade, 4-hour rollback window

## Approval
- ISSO: Approved (2025-03-16)
- System Owner: Approved (2025-03-17)
- Deployed: 2025-03-20
```

#### 4. Trigger Events for Re-Authorization

**Major change** (requires re-authorization before implementation):
- New system interconnection
- Change in classification level
- Major architectural change (e.g., move to cloud)
- Change in data types processed (e.g., add PII)

**Minor change** (notify ISSO, may not require re-authorization):
- Security patches
- Configuration hardening
- Adding users

---

## Quick Reference: Authorization Timeline

```
Month 1-2: PREPARE + CATEGORIZE + SELECT
  - Define boundary, assemble team
  - Determine impact level (Low/Moderate/High)
  - Select control baseline (125-421 controls)

Month 3-6: IMPLEMENT
  - Build system with security controls
  - Write SSP (200-500 pages)
  - Prepare evidence

Month 7-8: ASSESS
  - Independent security assessment
  - Penetration testing
  - SAR with findings

Month 9: AUTHORIZE
  - Remediate HIGH/CRITICAL findings
  - Create POA&M for residual risks
  - AO reviews package
  - ATO issued (or IATO or Denial)

Ongoing: MONITOR
  - Weekly scans
  - Monthly access reviews
  - Quarterly spot-checks
  - Annual re-assessment or every 3 years
```

**Typical timeline**: 9-12 months for HIGH system, 6-9 months for MODERATE, 3-6 months for LOW.

---

## Common Mistakes

### ❌ Treating ATO as Checklist

**Wrong**: "We have all controls implemented, give us ATO"

**Right**: ATO is risk acceptance decision. Document residual risks, let AO decide.

**Why**: AO accepts risk on behalf of organization. Your job is to inform that decision, not make it.

---

### ❌ Starting SSP After Implementation

**Wrong**: Build system, then write SSP to document what exists

**Right**: Write SSP during design/implementation, update as you build

**Why**: SSP informs design decisions. Writing after implementation often reveals missing controls.

---

### ❌ Ignoring Continuous Monitoring

**Wrong**: Get ATO, assume you're done for 3 years

**Right**: Continuous monitoring is mandatory. Weekly scans, monthly reviews, change impact analysis.

**Why**: ATO can be revoked if controls degrade. Continuous monitoring proves ongoing compliance.

---

### ❌ No Risk Acceptance for Residual Risks

**Wrong**: Hide findings or claim "we'll fix later"

**Right**: Document in POA&M, explicit risk acceptance by AO for LOW/MEDIUM items

**Why**: Transparency builds trust. AO needs full picture to make informed decision.

---

### ❌ Vague Control Implementation Descriptions

**Wrong**: "AC-2: We manage accounts properly"

**Right**: "AC-2: ServiceNow ticket → manager approval → automated creation → 30-day inactivity disablement → monthly review. Evidence: /docs/account-mgmt.pdf"

**Why**: Assessor cannot verify vague descriptions. Specific implementation enables assessment.

---

## Cross-References

**Use WITH this skill**:
- `ordis/security-architect/security-controls-design` - Implement controls for SSP
- `ordis/security-architect/threat-modeling` - Inform security categorization
- `muna/technical-writer/operational-acceptance-documentation` - Write SSP/SAR/POA&M

**Use AFTER this skill**:
- `ordis/security-architect/classified-systems-security` - If system handles classified data

## Real-World Impact

**Systems using formal authorization processes**:
- **DoD Cloud Migration**: RMF process revealed 47 missing controls before migration. Remediated during IMPLEMENT phase (vs discovering at ASSESS). Achieved ATO in 9 months vs industry avg 14 months.
- **Healthcare System (Moderate)**: POA&M transparency with 12 accepted LOW risks enabled IATO while remediating MEDIUM findings. System operational 6 months earlier than "wait for perfect" approach.
- **Continuous Monitoring**: Weekly scans detected CVE-2024-XXXX within 48 hours of disclosure. Change impact analysis approved emergency patch in 24 hours (vs 3-week change control for non-monitored systems).

**Key lesson**: **Formal authorization process enables informed risk decisions by leadership. SSP+SAR+POA&M transparency beats "security by checklist" or "security by obscurity".**
