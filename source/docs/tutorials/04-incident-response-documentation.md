# Tutorial 4: Incident Response Documentation

**Estimated Time:** 60-75 minutes

**Difficulty:** Advanced

---

## Introduction

This tutorial walks you through developing comprehensive incident response (IR) documentation and runbooks after a security breach. You'll learn to systematically document threat landscape, design security architecture improvements, and create operational procedures using Claude Code skills for modern SaaS security.

### Scenario

You lead security at TenantHub, a multi-tenant SaaS platform providing project management tools to enterprises. Your platform serves 500+ organizations with 30,000+ active users. Last week, your security team discovered a SQL injection vulnerability in the task-filtering API that allowed an attacker to extract customer data for three days before detection. Customer data exposed includes project names, descriptions, user emails, and file metadata.

This is a real incident. Now you must:
1. Document the threat landscape and what happened
2. Review your security architecture to prevent recurrence
3. Create detailed incident response playbooks
4. Establish operational procedures for future incidents
5. Report to customers and regulators within GDPR's 72-hour requirement

**Incident Timeline:**

- **Day 1 (Attack Period):** Attacker discovers SQL injection, exploits for data extraction (Hours 0-72)
- **Day 2 (Detection):** Security monitoring alerts on unusual API query patterns
- **Day 3 (Containment):** Deployment of WAF rules, API patching, incident declared
- **Day 4 (Investigation):** Forensic analysis of logs, scope determination, customer notification begins
- **Day 5 (Recovery):** Full deployment of fixes, log retention analysis
- **Day 6 (Post-Incident):** Lessons learned meeting, control improvements planned

**TenantHub SaaS Details:**

- **Platform:** Multi-tenant project management SaaS
- **Architecture:** Kubernetes-based microservices on AWS
- **Data Classification:** Customer business data (Confidential)
- **Affected Data:**
  - 500 customer organizations compromised
  - 30,000 user accounts exposed
  - ~2.5 GB project and task metadata extracted
  - Email addresses, names, project structures

- **Technology Stack:**
  - **Frontend:** React.js Single-Page Application
  - **API Layer:** Node.js/Express microservices
  - **Database:** PostgreSQL with role-based access control
  - **Message Queue:** RabbitMQ for async processing
  - **Monitoring:** DataDog for APM and security metrics
  - **Hosting:** AWS (VPC, ECS, RDS)

- **Regulatory & Compliance Context:**
  - **Primary Regulation:** GDPR (72-hour breach notification requirement)
  - **Customer Agreements:** SOC 2 Type II committed
  - **Internal SLAs:** Incident response (4-hour initial response)
  - **Insurance:** Cyber liability policy with notification timeline

### What You'll Accomplish

By the end of this tutorial, you will have:

1. Documented the complete incident threat timeline and root cause analysis
2. Created comprehensive IR playbooks covering detection, containment, eradication, recovery, and post-mortem
3. Reviewed security architecture and identified control gaps
4. Generated operational runbooks for incident management procedures
5. Established escalation matrices and communication templates
6. Developed lessons learned and control improvement roadmap
7. Prepared for regulatory compliance (GDPR breach notification)

---

## Prerequisites

Before starting this tutorial, ensure you have:

- **Claude Code installed and configured**
- **Access to the security-architect skillpack** (ordis/security-architect)
- **Access to the documentation writer skillpack** (muna/technical-writer)
- **Understanding of incident response frameworks** (NIST SP 800-61, ICS-CERT)
- **Familiarity with threat modeling and attack vectors**
- **Knowledge of SaaS architecture and multi-tenancy security**
- **Understanding of GDPR breach notification requirements**
- **Familiarity with forensic investigation basics**
- **Text editor** for documenting outputs

**Optional:**

- Existing incident response policy template
- System architecture diagrams
- Current security control inventory
- Threat modeling documentation
- Past incident reports or case studies
- GDPR and compliance policy documents
- Security tool access logs (examples)

---

## Step 1: Incident Response Planning & Threat Documentation

**Estimated Time:** 15 minutes

**Objective:** Document the incident comprehensively, create the IR plan framework, and map the threat landscape to understand what happened and how.

### Understanding Incident Response Framework

Before diving into documentation, understand the core IR framework phases:

1. **Preparation:** Pre-incident readiness
   - IR team structure and contacts
   - Tools and access provisioned
   - Playbook templates and procedures

2. **Detection & Analysis:** Identifying the security incident
   - Alert triage and validation
   - Scope determination and impact assessment
   - Initial containment decisions

3. **Containment:** Limiting damage spread
   - Short-term containment (isolate affected systems)
   - Long-term containment (patch and harden)

4. **Eradication:** Removing attacker access
   - Malware/backdoor removal
   - Vulnerability patching
   - Access credential rotation

5. **Recovery:** Restoring normal operations
   - System restoration from clean backups
   - Monitoring for re-infection
   - Gradual operational normalization

6. **Post-Incident Activities:** Learning and improvement
   - Root cause analysis
   - Lessons learned documentation
   - Control improvements and prevention

### Loading the Skill

Use the incident response documentation skill:

```
I'm using muna/technical-writer/incident-response-documentation to create
comprehensive IR playbooks and procedures for a SaaS platform breach.
```

### Input Prompt

Provide detailed incident context:

```
I'm documenting incident response procedures for TenantHub, a multi-tenant
SaaS project management platform that experienced a SQL injection attack.

**Incident Summary:**
- Vulnerability: SQL injection in task-filtering API endpoint
- Discovery: Unusual query patterns detected in database logs
- Scope: 500 organizations, 30,000 users, 2.5 GB data exposure
- Duration: 72 hours of active data extraction
- Detection Method: DataDog APM alert on database query anomalies

**Immediate Actions Taken:**
1. Deployed Web Application Firewall (WAF) rules to block injection patterns
2. Applied emergency patch to API filtering logic
3. Rotated database credentials and API keys
4. Enabled enhanced database query logging for forensics
5. Notified executive team and legal/compliance

**Compliance Requirements:**
- GDPR 72-hour breach notification (H+68 hours currently)
- Customer notifications with impact details
- Law enforcement notification consideration
- Regulatory authority notification (DPA)

**Documentation Needed:**
1. Incident response playbook for detection and containment
2. Root cause analysis and technical deep-dive
3. Communication templates for customers and regulators
4. Post-incident investigation procedures
5. Forensic preservation and analysis guidance
6. Recovery and validation procedures
7. Post-mortem framework with lessons learned
```

### Sample IR Documentation Structure

Create comprehensive incident documentation including:

**Detection Playbook:**
```
When: Database query anomaly alert triggers
Alert: APM detects >500 queries/min with UNION SELECT patterns
Response: Validate alert, check logs, determine scope

Steps:
1. Acknowledge alert in incident tracking system
2. Query DataDog for query patterns (last 24 hours)
3. Check database logs for UNION/SELECT patterns
4. Identify affected data through schema analysis
5. Estimate scope (affected tables, records)
6. Engage database administrator and security lead
7. Determine if containment needed immediately
```

**Containment Playbook:**
```
When: SQL injection confirmed in active traffic
Action: Immediately block and isolate

Technical Containment:
1. Deploy WAF rules blocking UNION SELECT patterns
2. Rotate database user credentials (read-only and write)
3. Implement additional database query monitoring
4. Capture suspicious queries for forensic analysis
5. Enable enhanced logging on affected API endpoint
6. Review CloudTrail logs for suspicious access

Operational Containment:
1. Page on-call security engineer
2. Open war room channel (Slack #incident-response)
3. Notify customer success team (prepare for customer calls)
4. Brief C-level executives
5. Notify legal and compliance teams
6. Begin forensic evidence preservation
```

**Eradication Playbook:**
```
When: Attacker access is contained and root cause identified
Action: Remove vulnerability and attacker persistence

Code Remediation:
1. Code review of task-filtering API endpoint
2. Implement parameterized queries (prepared statements)
3. Add input validation and sanitization
4. Implement least-privilege database access (per API function)
5. Security testing of remediated code (SAST, manual review)

Infrastructure Hardening:
1. Patch Node.js and dependencies (security patches)
2. Update PostgreSQL to patched version
3. Harden database firewall rules (restrict query complexity)
4. Enable database query auditing
5. Implement API rate limiting per customer
6. Review and tighten IAM permissions
```

**Recovery Playbook:**
```
When: Eradication complete and patching deployed
Action: Restore full operations with verification

Verification Steps:
1. Run automated security test suite against API
2. Manual security testing of task filtering
3. Database integrity checks (validate no data corruption)
4. Application functionality validation (all task operations)
5. Customer test access validation
6. Monitor for anomalous queries (72-hour window)
7. Gradually re-enable customers (start with test customers)

Communication:
1. Notify customers that systems are patched and secure
2. Provide remediation timeline
3. Offer enhanced monitoring period
4. Establish dedicated support channel
```

### Creating the Root Cause Analysis Document

Document the technical and organizational factors:

```markdown
# Root Cause Analysis: TenantHub SQL Injection (Incident #2024-001)

## Executive Summary
A SQL injection vulnerability in the task-filtering API allowed unauthorized data access
for approximately 72 hours. The vulnerability was caused by insufficient input validation
on user-provided filter criteria. The attack was discovered through database query
anomaly detection and contained within 4 hours.

## Technical Root Cause
The task-filtering endpoint constructed SQL queries through string concatenation:

```python
# Vulnerable Code (BEFORE)
query = f"SELECT * FROM tasks WHERE title LIKE '%{user_filter}%'"
result = db.execute(query)
```

An attacker provided filter input: `%' UNION SELECT email, password_hash FROM users -- `
This resulted in unauthorized access to user credentials and PII.

## Why This Happened (Contributing Factors)
1. **Code Review Gap:** Filtering logic not included in security review process
2. **Incomplete Testing:** No SQL injection tests in automated test suite
3. **Insufficient Logging:** Query patterns not monitored at database layer
4. **Rushed Development:** Feature shipped under time pressure (2-week sprint)
5. **Incomplete Security Training:** Team unfamiliar with secure query construction

## Impact
- 500 organizations affected
- 30,000 user accounts exposed
- Estimated 2.5 GB of metadata exfiltrated
- 72-hour exposure window
- Regulatory breach notification required (GDPR)

## Timeline
- Hour 0: Attacker discovers and begins exploiting SQL injection
- Hour 72: DataDog APM detects unusual query patterns
- Hour 74: Security team validates and confirms SQL injection
- Hour 75: WAF rules deployed, database credentials rotated
- Hour 76: Code patch deployed to production
- Hour 78: Forensic analysis begins
- Hour 120: Customer notifications begin (72-hour requirement met)
```

---

## Step 2: Threat Landscape & Security Architecture Analysis

**Estimated Time:** 20 minutes

**Objective:** Map threats that could exploit similar vulnerabilities and design security controls to prevent recurrence.

### Loading the Skill

Use the threat analysis and architecture review skills:

```
I'm using ordis/security-architect/documenting-threats-and-controls to map
the threat landscape and identify security controls for preventing similar incidents.

I'm also using ordis/security-architect/security-architecture-review to evaluate
the architecture and identify weaknesses exploited by the attacker.
```

### Input Prompt for Threat Analysis

```
I need to document the threat landscape and security controls for TenantHub,
a multi-tenant SaaS project management platform that experienced a SQL injection attack.

**Current Threat Landscape:**

Threat Actors:
1. External opportunistic attackers (scanning internet for vulnerabilities)
2. Competitors seeking to disrupt service or steal customer data
3. Nation-state actors (lower probability but high impact)
4. Insider threats (disgruntled employees with database access)

Attack Vectors:
1. SQL injection in customer-facing APIs (recently exploited)
2. Unpatched dependencies (automated security scanning needed)
3. Weak credential management (API key exposure)
4. Misconfigured cloud infrastructure (public S3 buckets)
5. Social engineering targeting developers and operations
6. Compromised third-party integrations
7. Insufficient input validation across application

**Control Gaps Identified:**
1. No parameterized queries enforced in code standards
2. Limited database query monitoring/alerting
3. No code security testing (SAST) in CI/CD
4. Insufficient rate limiting on APIs
5. Weak database access controls (overprivileged accounts)
6. Incomplete security training for developers
7. Missing input validation framework

**Desired Security Posture:**
- Prevent SQL injection and similar injection attacks
- Detect unauthorized database access within minutes
- Enforce secure coding practices
- Implement defense-in-depth approach
- Meet SOC 2 Type II requirements
- Ensure GDPR compliance
```

### Security Control Design

Document controls across multiple layers:

**Code-Level Controls:**
```
Input Validation & Sanitization:
- Implement centralized input validation framework
- Type validation (ensure filter is string, not object)
- Length validation (prevent extremely long inputs)
- Character whitelist validation (reject special SQL characters)
- Parameterized query enforcement

Static Analysis:
- Implement SAST tool in CI/CD pipeline (e.g., Semgrep)
- Rules for SQL injection pattern detection
- Rules for vulnerable dependency detection
- Automated scanning on every code commit
- Fail builds with critical security findings
```

**Database-Level Controls:**
```
Least Privilege Access:
- Create database roles per application function
- Task query role (SELECT only on tasks, no system tables)
- User management role (SELECT/INSERT/UPDATE on users)
- Admin role (restricted to operations team only)
- Revoke SELECT on system tables and credentials

Monitoring & Alerting:
- Log all queries with UNION SELECT patterns
- Alert on >100 queries/min from application
- Alert on access to system tables from application user
- Alert on unusual query complexity metrics
- Real-time log analysis (e.g., AWS CloudWatch Insights)
```

**Infrastructure Controls:**
```
API Gateway & WAF:
- Implement Web Application Firewall at API edge
- WAF rules for SQL injection patterns
- Rate limiting per API key (100 req/min)
- Request size limits (max 1 KB query parameters)
- Timeout policies (30-second query timeout)

Network Segmentation:
- Restrict database access to application servers only
- No direct developer access to production database
- VPN required for administrative database access
- Separate networks for different trust levels
```

**Operational Controls:**
```
Code Review & Testing:
- Mandatory security code review for data access code
- Penetration testing on filtering endpoints quarterly
- Automated dependency scanning (Dependabot/Snyk)
- Security testing frameworks in unit tests
- Developer security training (OWASP Top 10)

Monitoring & Response:
- Centralized logging (CloudWatch, DataDog)
- Real-time alerting on suspicious patterns
- Incident response playbooks documented
- Weekly security reviews of top APIs
- Monthly security metrics reporting
```

---

## Step 3: Security Architecture Review Post-Incident

**Estimated Time:** 15 minutes

**Objective:** Evaluate the architecture comprehensively and design improvements to prevent similar incidents.

### Architecture Review Questions

Systematically review each architectural layer:

**API Layer Review:**
- Are all user inputs validated before database queries?
- Are parameterized/prepared statements used consistently?
- Is input length validated and enforced?
- Are API endpoints rate-limited?
- Is sensitive data logged (PII in query logs)?
- Are API errors detailed or generic?

**Database Layer Review:**
- Are database users minimally privileged?
- Are database queries logged and monitored?
- Are schema modifications audited?
- Is encryption at rest enabled?
- Are database backups encrypted and tested?
- Is point-in-time recovery available?

**Monitoring & Detection:**
- Are database query patterns monitored?
- Are failed authentication attempts tracked?
- Is unusual API behavior detected?
- Are security tools configured correctly?
- Is alert fatigue preventing detection?

**Incident Response:**
- Are IR procedures documented and tested?
- Is IR team trained and available 24/7?
- Are forensic tools and access available?
- Is legal/compliance notified in procedures?
- Are communication templates prepared?

### Architectural Improvements Roadmap

Create a prioritized improvement plan:

```markdown
# TenantHub Security Architecture Improvements (Post-Incident)

## Critical (Deploy within 2 weeks)
1. **Parameterized Query Enforcement**
   - Implement ORM layer (Sequelize, TypeORM)
   - Refactor all direct SQL queries
   - Code review of all database interactions
   - Timeline: 2 weeks

2. **Database Query Monitoring**
   - Enable PostgreSQL audit logging
   - Configure DataDog database monitoring
   - Implement alerting on anomalous queries
   - Timeline: 3 days

3. **WAF Deployment**
   - Implement AWS WAF or Cloudflare
   - Add SQL injection rules
   - Add rate limiting rules
   - Timeline: 1 week

## High (Deploy within 4-6 weeks)
1. **SAST/DAST in CI/CD**
   - Add Semgrep or Checkmarx scanning
   - Add OWASP ZAP automated testing
   - Implement security gates in pipeline
   - Timeline: 4 weeks

2. **Enhanced Logging**
   - Centralized log ingestion (CloudWatch)
   - Log retention (2 years for compliance)
   - Log encryption and access controls
   - Timeline: 3 weeks

3. **Database Least Privilege**
   - Design role-based access per application function
   - Implement database roles
   - Rotate credentials
   - Timeline: 4 weeks

## Medium (Deploy within 3 months)
1. **Incident Response Automation**
   - Lambda functions for IR actions
   - Automated WAF rule updates
   - Automated credential rotation
   - Timeline: 8 weeks

2. **Red Team Testing**
   - Quarterly penetration testing
   - Bug bounty program launch
   - Adversarial testing for APIs
   - Timeline: 12 weeks

3. **Security Training Program**
   - Mandatory OWASP Top 10 training
   - Code security review certification
   - Incident response simulations
   - Timeline: Ongoing

## Estimated Investment
- Engineering: 600 hours (12 weeks × 50 hours)
- Tools: $50,000/year (SAST, DAST, monitoring)
- Training: $5,000 (external training, materials)
- Total: ~$150,000 first year
```

---

## Step 4: Operational Runbooks & Escalation Procedures

**Estimated Time:** 10 minutes

**Objective:** Create step-by-step operational procedures and escalation matrices for future incidents.

### Loading the Skill

Use the operational documentation skill:

```
I'm using muna/technical-writer/operational-acceptance-documentation to create
detailed runbooks and procedures for incident response operations.
```

### Input Prompt

```
I need operational runbooks for TenantHub incident response procedures.

**Runbooks Needed:**
1. Incident Detection Runbook (how to identify a security incident)
2. Initial Response Runbook (first 30 minutes)
3. Escalation Runbook (when to escalate and to whom)
4. Communication Runbook (internal, customer, regulatory)
5. Forensic Preservation Runbook (evidence collection)
6. Recovery Validation Runbook (confirming systems are safe)

**Team Structure:**
- Security Lead (on-call, incident commander)
- Database Administrator (database investigation)
- Software Engineer (code changes, patching)
- Operations Engineer (deployment, infrastructure)
- Customer Success Manager (customer communication)
- Legal/Compliance (regulatory requirements)

**Key Procedures:**
- War room establishment and communication
- Severity classification (Critical, High, Medium, Low)
- Escalation to executive team
- Customer communication timing
- Regulatory notification requirements (GDPR)
- Evidence preservation and chain of custody
```

### Incident Detection Runbook

```markdown
# Runbook 1: Incident Detection & Validation

## When to Follow This Runbook
When a security alert fires, anomalous behavior is observed, or a potential incident is reported.

## Detection Sources
- DataDog security alerts (APM, logs, metrics)
- AWS CloudTrail suspicious activities
- Database query anomaly alerts
- Customer reports of unusual activity
- Internal security team observations

## Initial Validation (First 5 minutes)
1. **Acknowledge Alert**
   - Log into incident management system (Jira)
   - Create incident ticket (format: INC-YYYY-XXXX)
   - Assign initial responder (on-call security)
   - Set status to "Investigating"

2. **Validate Alert Authenticity**
   - Check if alert is from known false positive
   - Review alert configuration (alert tuning)
   - Check for related alerts (pattern)
   - Verify monitoring tool is functioning correctly

3. **Gather Initial Information**
   - Screenshot alert details
   - Collect relevant metrics/logs (last hour)
   - Note timestamp of first suspicious activity
   - Identify affected systems/data

## Scope Determination (5-15 minutes)
1. **Assess Severity**
   - Is actual customer data affected?
   - Is production system compromised?
   - Is ongoing active attack in progress?
   - What is estimated impact?

2. **Assign Severity Level**
   - **Critical:** Active data exfiltration, multiple customers affected, ongoing attack
   - **High:** Confirmed compromise, limited data access, attack contained
   - **Medium:** Suspicious activity, limited impact, low confidence
   - **Low:** False alarm, test alert, isolated system

3. **Initiate Response**
   - **Critical:** Page Security Lead immediately, open war room
   - **High:** Page Security Lead, escalate within 30 minutes
   - **Medium:** Escalate within 2 hours, assign investigation
   - **Low:** Log for future analysis, continue monitoring

## Escalation Trigger (Once severity assigned)
If Critical or High, escalate to incident commander and war room team.
Otherwise, continue investigation with assigned responder.
```

### Escalation Matrix

```markdown
# Escalation Matrix & Response Times

## Severity Levels & Response Times
| Severity | Initial Response | Investigation | Executive Brief | Customer Notify |
|----------|------------------|----------------|-----------------|-----------------|
| Critical | 5 minutes | Ongoing | 30 minutes | 60 minutes |
| High | 15 minutes | Within 1 hour | 2 hours | 4 hours |
| Medium | 1 hour | Within 4 hours | 8 hours | 24 hours |
| Low | 4 hours | Within 24 hours | Daily report | As needed |

## On-Call Team & Contact Hierarchy
**Level 1: Security Team (Immediate)**
- On-call Security Engineer: [contact]
- Security Lead: [contact]
- Escalation Channel: #incident-response Slack

**Level 2: Incident Commander (If escalation needed)**
- VP Security: [contact]
- CTO: [contact]
- War Room: Zoom [link]

**Level 3: Executive Response (For Critical incidents)**
- CEO: [contact]
- General Counsel: [contact]
- Board Notification: CEO decides

**Level 4: External Notifications (If GDPR breach)**
- Customer Success Manager: Notify customers
- Legal Counsel: Notify regulators
- PR Manager: Prepare public statement

## Critical Incident War Room
When severity is Critical:
1. Open war room Slack channel: #incident-[date]
2. Start Zoom bridge: [standing meeting link]
3. Page: Security Lead, DBA, Lead Engineer, Ops, Customer Success, Legal
4. Incident Commander: Security Lead or VP Security
5. Documentation: Someone assigned to war room notes
6. Updates: Every 15 minutes minimum

## Communication Frequency
- **Critical:** Every 15 minutes (war room + all stakeholders)
- **High:** Every 30 minutes (war room updates)
- **Medium:** Daily (email updates)
- **Low:** As needed (ticket only)
```

### Communication Templates

```markdown
# Communication Templates for Incident Response

## Internal War Room Opening
Subject: INCIDENT ALERT - [System/Service Name] - Severity: [CRITICAL/HIGH]

Hi team,

An incident has been declared for TenantHub. A potential [SQL injection / API compromise / etc.]
has been detected affecting [systems/customers].

**Quick Facts:**
- Time Detected: [time]
- Estimated Impact: [X customers affected]
- Current Status: [Investigating / Contained / Recovered]
- Severity: [CRITICAL / HIGH]

**Immediate Action:**
- Join Zoom war room: [link]
- Check #incident-[date] Slack channel for real-time updates
- If you're not on the on-call team, stand by for escalation

Incident Commander: [Name]
Updated: [Timestamp]

## Customer Notification (For Data Breach)
Subject: Security Incident - Action Required for Your TenantHub Account

Dear [Customer],

We are writing to inform you of a security incident affecting TenantHub that may impact your data.

**What Happened:**
On [date], we discovered a SQL injection vulnerability in our task-filtering API. An attacker
exploited this vulnerability to access customer data for approximately 72 hours before the
vulnerability was discovered and patched.

**What Data Was Affected:**
Your organization's data that may have been accessed includes:
- Project names and descriptions
- Task titles, descriptions, and assignments
- User names and email addresses
- File metadata (names, timestamps)
- [List specific data affected]

**What We Did:**
1. Immediately deployed a Web Application Firewall (WAF) to block attack patterns
2. Patched the vulnerable API endpoint with secure coding practices (parameterized queries)
3. Rotated all database credentials and API keys
4. Enabled enhanced logging and monitoring to detect any further unauthorized access
5. Conducted forensic analysis to determine scope

**What You Should Do:**
1. Check your TenantHub organization for any unauthorized access
2. Consider resetting passwords for users with sensitive project access
3. Review your audit logs in TenantHub for any suspicious activity
4. Contact us at security@tenanthub.com with any questions

**What We'll Do Next:**
We are implementing additional security controls across our platform:
1. Mandatory parameterized queries throughout the codebase
2. Real-time database query monitoring and alerting
3. Static application security testing in our CI/CD pipeline
4. Regular penetration testing and security assessments

**Regulatory Compliance:**
This incident is being reported to relevant regulatory authorities as required by GDPR
and other privacy regulations. We are committed to maintaining your trust and the security
of your data.

Sincerely,
TenantHub Security Team

For urgent questions: security@tenanthub.com or call [number]
```

---

## Step 5: Forensic Investigation & Lessons Learned

**Estimated Time:** 15 minutes

**Objective:** Document forensic procedures and create a lessons learned report to drive organizational improvements.

### Forensic Preservation Procedures

```markdown
# Forensic Investigation Procedures - TenantHub SQL Injection

## Evidence Preservation (First 30 minutes)

### Database Evidence Collection
1. **PostgreSQL Database Evidence**
   - Take consistent snapshot of affected database (postgres_exes)
   - Export query logs using pgAudit extension (last 72 hours minimum)
   - Export slow query logs from PostgreSQL main log directory
   - Preserve transaction logs (WAL files)
   - Document exact database state at time of discovery
   - Checksum all evidence files using SHA-256
   - Store checksums in evidence manifest
   - Cross-verify with database administrator

2. **Query Log Analysis**
   - Identify all UNION SELECT queries with timestamps
   - Export attacker's exact SQL statements
   - Document affected tables and columns accessed
   - List all database users involved in queries
   - Note connection source IP addresses
   - Map queries to specific API requests if possible
   - Calculate total data volume extracted

3. **Database Access Control Evidence**
   - Export database user permissions at time of incident
   - Document role-based access control (RBAC) configuration
   - Identify overprivileged database users
   - Review password change logs (credentials rotation)
   - Export database authentication logs

### Application & API Evidence
1. **Application Request Logs**
   - Export API gateway logs (last 96 hours for context)
   - Capture all requests to task-filtering endpoint
   - Document request parameters and payloads
   - Extract response codes and sizes
   - Note user identifiers and session IDs
   - Preserve request headers and authentication tokens
   - Export from all application servers (multi-region)

2. **Application Error & Diagnostic Logs**
   - Export application error logs (stderr/stdout)
   - Extract stack traces from suspicious operations
   - Document warning messages related to database access
   - Review application performance logs (APM)
   - Export debug logs if available
   - Preserve exception tracking data (Sentry, DataDog)

3. **Authentication & Access Logs**
   - Export all authentication attempts (success/failure)
   - Document user session creation/destruction
   - Extract API key usage logs
   - Review SSH access logs for servers
   - Export database connection logs
   - Document privilege escalation attempts

### Infrastructure Evidence Collection
1. **Cloud Platform Evidence (AWS)**
   - Export CloudTrail events (last 96 hours minimum)
   - Document all EC2 instance modifications
   - Export VPC Flow Logs for data flow analysis
   - Extract RDS events and modifications
   - Document S3 bucket access logs
   - Export Lambda function execution logs
   - Preserve network ACL changes and security group updates
   - Export IAM policy changes

2. **Network & Firewall Evidence**
   - Export network firewall logs (WAF/IDS)
   - Preserve router and switch logs
   - Document VPN access logs
   - Export DNS query logs (if available)
   - Extract network intrusion detection system (IDS) alerts
   - Document load balancer access logs

3. **Kubernetes & Container Logs** (if applicable)
   - Export kubelet logs from affected nodes
   - Preserve container runtime logs
   - Document pod creation/deletion events
   - Export network policy enforcement logs
   - Extract image pull logs and registry access

### Evidence Management Procedures
1. **Storage & Security**
   - Designate isolated forensic evidence repository (separate storage)
   - Store in encrypted volume (AES-256 encryption)
   - Restrict access to evidence (security lead + legal)
   - Document all access to evidence (chain of custody log)
   - Create multiple independent backups
   - Store backups in geographically separate locations
   - Use tamper-evident containers (optional but recommended)

2. **Chain of Custody Documentation**
   - Create evidence manifest with checksums (SHA-256)
   - Document who collected evidence and when
   - Record all transfers of evidence (dates, times, people)
   - Log all access to evidence (read-only vs. modification)
   - Document storage conditions and security measures
   - Maintain evidence integrity throughout investigation
   - Prepare for potential legal/regulatory disclosure

3. **Legal Hold & Retention**
   - Place all evidence under legal hold
   - Do NOT delete any related logs during investigation
   - Establish evidence retention policy (minimum 2+ years)
   - Document business justification for retention
   - Prepare for regulatory/law enforcement requests
   - Coordinate with legal counsel on disclosure

## Forensic Analysis (First 24-48 hours)

### Attack Timeline Reconstruction
1. **Entry Point Analysis**
   - How did attacker discover the SQL injection vulnerability?
   - Which specific API parameter was vulnerable?
   - First request timestamp and source IP
   - Geographic location of attacker (GeoIP analysis)
   - Time zone analysis to estimate attacker timezone
   - Attacker device fingerprinting if possible
   - Likely attacker tools (automated scanners vs. manual)

2. **Attack Progression Timeline**
   - Hour-by-hour breakdown of attacker activities
   - Queries attempted and which were successful
   - Data exfiltration techniques used (data extraction methods)
   - Volume of data extracted per hour
   - Attacker dwell time in system
   - Any lateral movement attempts
   - Privilege escalation attempts

3. **Data Exfiltration Analysis**
   - Exact tables and columns accessed
   - Number of records extracted per query
   - Total data volume exfiltrated (gigabytes)
   - Customer organizations affected by data access
   - User accounts exposed (count and details)
   - Sensitive data categories exposed (emails, metadata, etc.)
   - Whether attacker queried logs or audit tables

### Impact Assessment
1. **Scope & Affected Systems**
   - Number of customer organizations compromised (500)
   - Total number of user accounts exposed (30,000)
   - Data volume exfiltrated (2.5 GB)
   - Systems impacted (API layer, database)
   - Services interrupted (if any)
   - Third-party systems affected (partners, integrations)

2. **Data Classification & Sensitivity**
   - PII exposed: Names, emails, user IDs
   - Business data: Project names, descriptions, metadata
   - File information: Names, timestamps, permissions
   - Authentication data: Session tokens (rotated), API keys
   - Financial data: None exposed (not stored in platform)
   - Health data: None exposed (not applicable)

3. **Business Impact Analysis**
   - Customer notification requirements and timeline
   - Regulatory notifications needed (GDPR, state laws)
   - Public relations impact and messaging
   - Financial impact (incident response costs, fines)
   - Reputational damage and customer retention risk
   - Law enforcement involvement requirements

4. **Attacker Profile Assessment**
   - Attacker sophistication level (script kiddie vs. APT)
   - Likely attacker motivation (data theft, disruption, testing)
   - Nation-state involvement likelihood
   - Organized crime vs. individual actor
   - Use of bulletproof hosting or anonymization services
   - Evidence of data monetization or sale

### Root Cause Analysis Deep Dive
1. **Technical Root Causes**
   - Specific code vulnerability (string concatenation in query building)
   - Lack of input validation on filter parameter
   - Missing parameterized query usage
   - Insufficient WAF/IPS rules at network edge
   - Database overprivilege (query returned sensitive columns)
   - Lack of query result size limits

2. **Process & Control Gaps**
   - Security code review was incomplete (non-auth features skipped)
   - No automated security testing for SQL injection patterns
   - Database query monitoring not enabled
   - Incident playbooks not tested or trained on
   - Insufficient developer security training on OWASP Top 10
   - No API rate limiting enforcement

3. **Organizational Contributing Factors**
   - Time pressure to ship features (2-week sprint cycle)
   - Security team understaffed relative to development pace
   - Security culture not yet mature (security secondary concern)
   - Inadequate budget for security tooling (SAST, monitoring)
   - Lack of security expertise in development team
   - Insufficient incident response drills

4. **Why Detection Took 72 Hours**
   - Baseline database metrics not established (what's normal?)
   - No dedicated database security monitoring
   - DataDog APM alerts needed tuning and training
   - Security team response time delayed (alert not reviewed immediately)
   - False positive fatigue on other alerts
```

### Lessons Learned Process

**Post-Incident Meeting (Within 5 days)**

Conduct comprehensive meeting with:
- Incident Commander (VP Security or Security Lead)
- Technical responders (Database Administrator, Lead Engineer, DevOps)
- Product & Engineering Leadership (VP Product, VP Engineering)
- Compliance & Legal (Chief Legal Officer or General Counsel)
- Customer Success & Support (VP Customer Success)
- Finance (for incident cost analysis)

**Detailed Meeting Agenda (2-3 hours):**

**Section 1: Incident Walkthrough (45 minutes)**
1. Timeline review (what happened when, minute-by-minute)
2. Detection and response timeline
3. Containment steps and effectiveness
4. Recovery process and validation
5. Current status and customer impact

**Section 2: What Went Well (15 minutes)**
- Early detection through APM monitoring (even though delayed)
- Rapid response once threat confirmed (4-hour containment)
- Effective cross-functional collaboration
- Strong forensic preservation
- Transparent customer communication
- Clear escalation procedures worked

**Section 3: What Went Poorly (20 minutes)**
- SQL injection code review was incomplete for non-auth features
- Database query logging wasn't enabled by default
- No automated security testing (SAST) for SQL injection patterns
- API rate limiting not enforced (would have slowed extraction)
- Incident playbooks hadn't been tested in advance
- Alert fatigue reduced attention to database alerts
- Communication delays in first 2 hours

**Section 4: Root Cause Deep Dive (30 minutes)**
1. Primary cause: Insecure query construction (string concatenation)
2. Enabling factor: Lack of input validation
3. Process gap: Security code review not mandatory for all code
4. Detection gap: No database query monitoring
5. Organizational factor: Time pressure on development
6. Training gap: Developers not trained on SQL injection prevention

**Section 5: Prevention Discussion (30 minutes)**
1. What would have prevented this attack?
   - Parameterized queries (technical)
   - Mandatory security code review (process)
   - SAST in CI/CD pipeline (automation)
   - Database access monitoring (detection)
2. What would have reduced impact?
   - Faster detection (better monitoring)
   - API rate limiting (slowdown data exfiltration)
   - Data minimization (limit sensitive data in results)
   - Encryption at rest (reduce usefulness of stolen data)
3. What would have improved response?
   - Pre-tested incident playbooks
   - Trained response team
   - Clear communication templates

**Lessons Learned Report Template:**

```markdown
# Incident Post-Mortem Report - TenantHub SQL Injection (INC-2024-001)

## Executive Summary
On [date], TenantHub discovered a SQL injection vulnerability in the task-filtering API
that allowed unauthorized access to customer data for approximately 72 hours. The incident
affected 500 customer organizations and 30,000 users. The vulnerability was discovered
through database query anomaly detection, contained within 4 hours, and fully patched
within 24 hours. This post-mortem documents the incident, root causes, and improvements.

## Incident Timeline & Severity
- **Severity:** Critical
- **Scope:** 500 organizations, 30,000 users, 2.5 GB data exposure
- **Detection Time:** Hour 72 (3 days)
- **Containment Time:** Hour 76 (4 hours after detection)
- **Customer Notification:** Hour 120 (5 days - within GDPR requirement)
- **Full Recovery:** Hour 126 (6 days)

## Detailed Timeline
- Hour 0: Attacker discovers SQL injection vulnerability
- Hour 0-72: Attacker exploits API, extracts customer data
- Hour 72: DataDog APM detects unusual database query patterns
- Hour 74: Security team validates SQL injection exploitation
- Hour 75: WAF rules deployed, database credentials rotated
- Hour 76: Code patch committed and deployed to production
- Hour 78: Forensic evidence collection begins
- Hour 96: Root cause analysis completed
- Hour 120: Customer breach notifications sent (72-hour GDPR deadline met)
- Hour 126: Full recovery completed, system fully functional

## What Went Well (Positive Findings)
1. **Early Detection:** DataDog APM detected unusual query patterns even though detection took 72 hours
2. **Rapid Response:** Once confirmed, containment took only 4 hours
3. **Effective Communication:** Cross-functional coordination prevented further delays
4. **Forensic Preservation:** Evidence collected properly for investigation
5. **Customer Transparency:** Clear communication about what happened and remediation
6. **Regulatory Compliance:** GDPR 72-hour notification deadline met
7. **Team Collaboration:** Security, engineering, and operations worked seamlessly

## What Went Poorly (Negative Findings)
1. **Incomplete Code Review:** SQL injection vulnerability wasn't caught in code review
   - Root cause: Security review was optional for non-authentication features
   - Impact: Vulnerability made it to production
   - Fix: Make security review mandatory for all data access code

2. **No Automated Security Testing:** No SAST tool to detect SQL injection patterns
   - Root cause: Budget constraints and tool selection delays
   - Impact: Vulnerability not detected pre-deployment
   - Fix: Implement SAST in CI/CD pipeline (4-week rollout)

3. **Missing Database Monitoring:** Query patterns not monitored at database layer
   - Root cause: Database monitoring configuration was incomplete
   - Impact: 72-hour detection window (too long)
   - Fix: Enable database query monitoring with real-time alerts (1-week implementation)

4. **No API Rate Limiting:** Attacker could extract data freely without throttling
   - Root cause: Rate limiting not implemented in API design
   - Impact: Faster data exfiltration and larger scope
   - Fix: Implement per-customer rate limiting (3-week implementation)

5. **Alert Fatigue:** Security team initially dismissed database alert as false positive
   - Root cause: Too many false positive alerts from other systems
   - Impact: 2-hour delay in initial response
   - Fix: Alert tuning and on-call training (ongoing)

6. **Untested Playbooks:** Incident response playbooks not tested before incident
   - Root cause: No budget or time allocated for IR drills
   - Impact: Some confusion during initial response
   - Fix: Quarterly IR tabletop exercises (starting next month)

## Root Cause Analysis Summary

### Primary Technical Cause
The task-filtering API endpoint constructed SQL queries using string concatenation
instead of parameterized queries. When a user provided filter input `' UNION SELECT
email FROM users --`, the application executed a query that returned sensitive data.

### Contributing Factors (5 Whys Analysis)
1. Why was SQL injection possible? → No input validation
2. Why no input validation? → Development time pressure
3. Why time pressure? → Two-week sprint cycle
4. Why short sprints? → Aggressive feature roadmap
5. Why aggressive roadmap? → Sales/customer demands

### Process & Control Gaps
- Security code review was optional for non-authentication features
- Static application security testing (SAST) not implemented
- Database access privileges not minimized (query returned all columns)
- Query monitoring not enabled on production database
- API rate limiting not enforced

### Detection & Response Gaps
- Database query anomaly detection relied on manual analysis
- Alert thresholds not properly tuned for production traffic
- No automated incident response procedures
- Incident playbooks not documented or trained
- Response time SLAs not established

## Root Cause Prevention: What Would Have Stopped This

1. **Code-Level Prevention** (Would have prevented vulnerability)
   - Mandatory security code review for all data access code
   - Parameterized query enforcement (code standards)
   - SAST tool integration in CI/CD pipeline
   - SQL injection testing in unit tests

2. **Detection & Monitoring** (Would have detected faster)
   - Database query anomaly monitoring (detect in hours, not days)
   - API request pattern analysis
   - Unauthorized data access alerts
   - Query result size monitoring

3. **Operational Controls** (Would have limited impact)
   - API rate limiting per customer (slow down data extraction)
   - Database least privilege (limit columns accessible)
   - Query timeout enforcement (prevent bulk extractions)
   - Real-time audit logging with alerting

4. **Incident Response** (Would have improved response)
   - Documented and tested incident playbooks
   - Trained on-call security team
   - Established response time SLAs
   - Clear escalation procedures

## Action Items & Improvement Plan

### Critical Priority (Complete within 2 weeks)
| Item | Owner | Deadline | Estimated Effort |
|------|-------|----------|------------------|
| Deploy WAF SQL injection rules | DevOps Lead | Day 5 | 8 hours |
| Enable database query audit logging | DBA | Day 3 | 4 hours |
| Implement parameterized query enforcement | Lead Engineer | Day 14 | 40 hours |
| Create and document incident playbooks | Security Lead | Day 10 | 16 hours |
| Establish database query monitoring | Security + DevOps | Day 7 | 12 hours |

### High Priority (Complete within 4-6 weeks)
| Item | Owner | Deadline | Estimated Effort |
|------|-------|----------|------------------|
| Implement SAST in CI/CD pipeline | VP Engineering | Week 4 | 80 hours |
| Conduct mandatory security training | HR + Security | Week 6 | 20 hours |
| Implement API rate limiting | Lead Engineer | Week 4 | 60 hours |
| Database least privilege implementation | DBA | Week 5 | 40 hours |
| Establish incident response team | VP Security | Week 3 | 8 hours |

### Medium Priority (Complete within 3 months)
| Item | Owner | Deadline | Estimated Effort |
|------|-------|----------|------------------|
| Quarterly IR tabletop exercises | Security Lead | Month 3 | 8 hours/quarter |
| Penetration testing program | VP Security | Month 3 | 40 hours |
| Security metrics dashboard | Security + DevOps | Month 2 | 60 hours |
| Developer security certification | Engineering | Month 3 | 4 hours/person |
| Third-party security assessments | VP Security | Month 3 | 120 hours |

## Prevention Summary & Controls Roadmap

### Immediate Controls (Prevent Recurrence)
- **Code Controls:** Parameterized queries mandatory, SAST in pipeline
- **Database Controls:** Query monitoring, least privilege enforcement
- **Detection Controls:** Real-time alerting on anomalous queries
- **Response Controls:** Documented playbooks, trained team

### Detection Improvements (Faster Detection)
- Baseline database query metrics established
- Alert on queries accessing sensitive columns
- Detect unusual data extraction patterns
- Real-time log analysis (minutes, not hours)

### Response Improvements (Faster Containment)
- Automated WAF rule updates
- Automated credential rotation
- Automated incident notifications
- Pre-incident coordination (war room setup, stakeholder notification)

## Recommended Control Investments (Next 12 months)

```
Q1 2024:
- Parameterized query enforcement: $0 (internal)
- Database monitoring: $5,000/year (DataDog)
- SAST tool: $25,000/year (Semgrep or Checkmarx)
- Total Q1: $30,000

Q2 2024:
- Incident response automation: $0 (internal)
- Security training program: $5,000 (external)
- Penetration testing: $15,000 (external firm)
- Total Q2: $20,000

Total Year 1: ~$150,000 (tool subscriptions + external services)
Staffing: +1 FTE Security Engineer for ongoing monitoring
```

## Board/Executive Summary

TenantHub experienced a serious SQL injection incident affecting 500 organizations and
30,000 users. The incident was contained within 4 hours and fully patched within 24
hours. Customer notifications met GDPR requirements. We are implementing comprehensive
security improvements including code review enforcement, automated security testing,
and real-time monitoring to prevent similar incidents.

### Key Metrics
- **Detection Time:** 72 hours (target: 1 hour)
- **Containment Time:** 4 hours (target: <1 hour)
- **Customer Impact:** 30,000 users notified within requirement
- **Regulatory Compliance:** GDPR deadline met
- **Recovery Time:** 6 days to full functionality
- **Estimated Cost:** $150,000 in prevention controls (Year 1)

### Board Acknowledgments
- Incident response effectiveness demonstrated
- Customer trust maintained through transparency
- Regulatory compliance achieved despite incident
- Investment in security controls approved for 2024

---

**Post-Mortem Status:** COMPLETE
**Presented to:** Executive Team, Board of Directors
**Review Date:** 30 days (October 29, 2024)
**Next Steps:** Monitor implementation of action items, track metrics quarterly
```

---

## Conclusion

You have completed the comprehensive incident response documentation tutorial for TenantHub's SQL injection breach. You have:

1. **Documented the incident** with threat analysis, timeline, and root causes
2. **Designed security controls** across code, database, API, and infrastructure layers
3. **Reviewed security architecture** and created improvement roadmap
4. **Created operational runbooks** for detection, containment, recovery, and response
5. **Established escalation procedures** and communication templates
6. **Planned forensic investigation** and lessons learned process

### Key Takeaways

**For Security Teams:**
- Incident response documentation should be comprehensive and immediately actionable
- Playbooks should cover detection, containment, eradication, recovery, and post-mortem
- Escalation procedures must be clear with response time SLAs
- Lessons learned must drive architectural improvements

**For Compliance:**
- GDPR 72-hour breach notification requirement is strict
- Document all investigative steps for regulatory reporting
- Customer communication should be timely and transparent
- Implement control improvements to demonstrate commitment to security

**For Operations:**
- War room procedures prevent communication breakdown
- Evidence preservation is critical for forensics and legal
- Runbooks reduce response time and human error
- Regular testing of incident procedures is essential

### Next Steps

1. **Customize playbooks** for your specific technology stack and organization
2. **Define escalation contacts** and ensure 24/7 coverage
3. **Test incident procedures** with tabletop exercises quarterly
4. **Integrate with monitoring tools** (DataDog, CloudWatch, Splunk)
5. **Train team members** on their roles and responsibilities
6. **Schedule annual review** of incident procedures
7. **Conduct red team exercises** to test detection and response

### Additional Resources

- **NIST SP 800-61:** Computer Security Incident Handling Guide
- **ICS-CERT:** Incident Handling Process
- **AWS:** Incident Response Best Practices
- **OWASP:** SQL Injection Prevention Cheat Sheet
- **CIS:** Critical Security Controls

---

**Tutorial Complete!** You have created comprehensive incident response documentation that enables your organization to detect, respond to, and recover from security incidents while maintaining regulatory compliance and stakeholder trust.

Time spent: 60-75 minutes | Difficulty: Advanced | Incidents prevented: Immeasurable
