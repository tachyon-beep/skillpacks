# Tutorial 2: Healthcare Compliance Documentation (HIPAA-Focused)

**Estimated Time:** 60-75 minutes

**Difficulty:** Advanced

---

## Introduction

This tutorial walks you through creating comprehensive compliance documentation for a healthcare system subject to HIPAA and SOC 2 Type II requirements. You'll learn to systematically map regulatory requirements to technical controls, document security safeguards, establish operational procedures, and define governance frameworks using Claude Code skills.

### Scenario

You're building an Electronic Health Records (EHR) system for a healthcare startup serving multiple clinics. The platform must protect sensitive patient information (PHI) while meeting strict regulatory compliance standards.

**EHR System Details:**

- **Protected Health Information (PHI) Handled:**
  - Patient names, dates of birth, social security numbers
  - Medical record numbers (MRNs)
  - Clinical diagnoses and treatment plans
  - Prescription information
  - Laboratory test results and imaging reports
  - Insurance information
  - Emergency contact details

- **Compliance Requirements:**
  - HIPAA Security Rule (Administrative, Physical, Technical safeguards)
  - HIPAA Privacy Rule (PHI access controls)
  - HIPAA Breach Notification Rule
  - SOC 2 Type II (Security, Availability, Processing Integrity, Confidentiality)
  - State privacy regulations (varies by jurisdiction)

- **Technology Stack:**
  - **Frontend:** React 18.x (TypeScript)
  - **Backend:** Python Django 4.x with DRF
  - **Database:** PostgreSQL 15 (encrypted at rest)
  - **Cache:** Redis (for session management)
  - **Authentication:** OAuth 2.0 + MFA (TOTP)
  - **Hosting:** AWS with HIPAA Business Associate Agreement (BAA)
  - **Regions:** Multi-region deployment (us-east-1, us-west-2)

- **Users:**
  - Clinic administrators (10-20 per clinic)
  - Healthcare providers (physicians, nurses)
  - Medical staff (lab technicians, imaging specialists)
  - Patients (reading own records)
  - Auditors (compliance and security reviews)

### What You'll Accomplish

By the end of this tutorial, you will have:

1. Created comprehensive HIPAA-to-technical controls mapping
2. Documented all three types of HIPAA safeguards (Administrative, Physical, Technical)
3. Generated SOC 2 Trust Service Criteria alignment documentation
4. Developed operational acceptance documentation and runbooks
5. Established IT governance policies and procedures
6. Created evidence collection protocols for compliance audits

---

## Prerequisites

Before starting this tutorial, ensure you have:

- **Claude Code installed and configured**
- **Access to the security-architect skillpack** (ordis/security-architect)
- **Understanding of HIPAA regulations** (high-level overview of Security Rule)
- **Familiarity with healthcare IT architecture**
- **Knowledge of compliance frameworks** (HIPAA, SOC 2 basics)
- **Text editor** for documenting outputs

**Optional:**
- Healthcare compliance policies (if existing)
- Current risk assessment documentation
- System architecture diagrams
- Data flow diagrams showing PHI handling

---

## Step 1: Map Compliance Requirements to Technical Controls

**Estimated Time:** 20 minutes

**Objective:** Create detailed mapping of HIPAA Security Rule requirements to your EHR system's technical controls.

### Loading the Skill

Use the compliance mapping skill:

```
I'm using ordis/security-architect/compliance-awareness-and-mapping to map HIPAA requirements to our EHR system.
```

### Input Prompt

Provide comprehensive context about your compliance scope:

```
I'm documenting compliance requirements for a healthcare EHR system. Please help me create detailed mappings of HIPAA Security Rule requirements to technical controls.

**System Scope:**
- Electronic Health Records (EHR) platform
- Multi-clinic deployment (5+ clinics initially, scaling to 50+)
- Cloud-hosted on AWS with HIPAA BAA
- Processes patient PHI: names, SSNs, MRNs, diagnoses, prescriptions, test results

**HIPAA Security Rule Areas to Map:**

1. Administrative Safeguards
   - Security management process and risk assessment
   - Assigned security responsibility
   - Workforce security (access controls)
   - Information access management
   - Security awareness and training
   - Security incident procedures
   - Contingency planning

2. Physical Safeguards
   - Facility access and controls
   - Workstation use and security
   - Workstation device and media controls

3. Technical Safeguards
   - Access controls (user ID, emergency access, encryption, key management)
   - Audit controls and integrity controls
   - Transmission security

**Current Technical Implementation:**
- React frontend with TypeScript
- Python Django backend with role-based access control
- PostgreSQL database with AES-256 encryption at rest
- Redis for session management
- AWS IAM for infrastructure access
- VPC isolation and security groups
- SSL/TLS for all data in transit
- AWS CloudTrail for audit logging
- CloudWatch for monitoring
- Automated daily backups to S3

**Additional Requirements:**
- SOC 2 Type II compliance (Security, Availability, Processing Integrity, Confidentiality)
- Must support multi-tenancy (data isolation between clinics)
- Disaster recovery with <4 hour RTO, <1 hour RPO
- Encryption key rotation (quarterly)
- Annual compliance audits

Please create:
1. HIPAA Security Rule to technical control mapping matrix
2. SOC 2 Trust Service Criteria crosswalk
3. Control effectiveness assessment
4. Gap analysis for any unmet requirements
```

### Expected Output

The skill will generate:

**HIPAA Control Mapping Matrix Example:**

| HIPAA Requirement | Technical Control | Implementation | Evidence |
|------------------|------------------|----------------|----------|
| **Security Management Process (164.308(a)(1))** | Risk assessment program | Annual risk assessments documented | Risk assessment reports (2023, 2024) |
| | Risk management planning | Documented remediation for identified risks | Risk register with remediation tracking |
| | Sanctions policy | Non-compliance escalation procedures | Policy GOV-2024-001, training records |
| | Information system monitoring | CloudWatch + CloudTrail logging | Log retention in S3 (7 years) |
| **Assigned Security Responsibility (164.308(a)(2))** | HIPAA compliance officer | Designated CISO with written authorization | Org chart, job description |
| | Compliance committee | Monthly security review meetings | Meeting minutes, attendees, action items |
| **Workforce Security (164.308(a)(3))** | Authorization procedures | Role-based access control matrix | Access approval workflow (ServiceNow) |
| | Supervision & oversight | Quarterly access reviews by managers | Completed access review certifications |
| | Termination procedures | Access removal checklist within 24 hours | Termination tickets with verification |
| **Information Access Management (164.308(a)(4))** | Access determination | Role definitions (8 roles) with permissions | RACI matrix, access control list |
| | Access review | Quarterly access reviews for all users | Manager certifications (100% coverage) |
| **Security Awareness & Training (164.308(a)(5))** | Security training | Annual HIPAA training (mandatory) | Completion tracking in LMS |
| | Password management training | MFA setup training for new users | Training checklist per employee |
| | Log-in monitoring training | Alerting on unusual access patterns | Training module + test |
| **Security Incident Procedures (164.308(a)(6))** | Incident response plan | Documented IR procedures with SLAs | Policy GOV-2024-003, drill results |
| | Incident reporting | Breach notification process (<60 days) | Breach log, HHS notifications |
| **Contingency Planning (164.308(a)(7))** | Data backup & recovery | Daily incremental, weekly full backups | Backup logs, restore test results |
| | Disaster recovery plan | RTO <4 hours, RPO <1 hour | DR plan, failover procedures, tests |
| **Business Associate Management (164.308(b))** | BA agreements | HIPAA BAA with AWS, vendors | Signed BAAs on file (AWS, Okta, etc.) |
| | BA compliance monitoring | Annual BA compliance assessments | Assessment reports, audit results |
| **Physical Access Controls (164.310(a))** | Facility access | Data center with biometric access | AWS compliance certificate |
| | Workstation placement | Servers in secure, locked cabinets | Data center photos, access logs |
| **Workstation Use Security (164.310(b))** | Workstation use policy | Screen lock after 15 min inactivity | Endpoint security policy |
| | Secure destruction | Disk wiping before device reuse | Data disposal procedures |
| **Access Controls (164.312(a)(2))** | Unique user ID | Each user has unique identifier | Active Directory user list |
| | Emergency access | Break-glass access with audit trail | Emergency access procedures |
| | Encryption & key management | AES-256 at rest, TLS 1.3 in transit | Key rotation log (quarterly) |
| **Audit Controls (164.312(b))** | Audit logging | All PHI access logged with user/timestamp | CloudTrail, RDS audit logs |
| | Log review | Daily automated log review with alerts | CloudWatch rules, SNS notifications |
| **Integrity Controls (164.312(c)(2))** | Checksums | MD5 checksums on backups | Backup verification logs |
| | Data integrity monitoring | Database constraints + application validation | Schema documentation |
| **Transmission Security (164.312(e))** | TLS encryption | TLS 1.3 for all data in transit | SSL certificate audit |
| | VPN for remote access | VPN required for remote clinician access | VPN logs, device management |

**SOC 2 Trust Service Criteria Alignment:**

**CC (Common Criteria):**
- CC1: Organization obtains or generates, uses, and communicates relevant, quality information
  - HIPAA Control: Audit logging and monitoring
  - Implementation: CloudTrail for all API calls
  - Evidence: Log retention policy (7 years)

- CC6: The organization obtains or generates, uses, and communicates relevant, quality information
  - HIPAA Control: Unique user identification
  - Implementation: RBAC with 8 roles per user
  - Evidence: Access control list

**C (Confidentiality):**
- C1: The system is protected against unauthorized internal and external access
  - HIPAA Control: Encryption at rest and in transit
  - Implementation: AES-256 + TLS 1.3
  - Evidence: Key management procedures

**A (Availability):**
- A1: The system is protected against disruption of operations
  - HIPAA Control: Disaster recovery planning
  - Implementation: RTO <4 hours, RPO <1 hour
  - Evidence: DR plan + monthly test results

**Gap Analysis Example:**

| Control Area | Status | Gap | Priority | Remediation |
|-------------|--------|-----|----------|-------------|
| Encryption at rest | Implemented | Uses AES-256, no gaps | N/A | Continue current practice |
| Encryption in transit | Implemented | TLS 1.3 on APIs, but legacy SFTP unencrypted | High | Implement SFTP over TLS for vendor data |
| Key rotation | Implemented | Quarterly rotation documented | Low | Automate key rotation (currently manual) |
| Audit logging | Implemented | Logs retained 7 years, meets HIPAA | N/A | Review annually |
| Access reviews | Partial | Quarterly for employees, missing contractors | Medium | Expand access review scope to all contractors |
| Incident response | Implemented | Plan exists, but no recent drill | Medium | Schedule quarterly IR drills |
| Business associate agreements | Partial | AWS, Okta signed; missing small vendors | High | Execute BAAs with all vendors accessing PHI |
| Workstation security | Gap | No mandatory endpoint encryption | High | Deploy endpoint encryption to all workstations |
| Multi-factor authentication | Implemented | TOTP enforced for all users, no gaps | N/A | Continue current practice |

### Observations

**Key Points:**
- HIPAA requires documentation of all security controls, not just technical ones
- Administrative safeguards (policies, training, incident response) are equally important
- SOC 2 audit validates these controls through independent assessment
- Control effectiveness depends on consistent implementation and enforcement
- Evidence trails are critical for audit compliance

### Time Estimate

15-20 minutes to generate mappings and review for completeness.

---

## Step 2: Document Security Controls and Threats

**Estimated Time:** 20 minutes

**Objective:** Create comprehensive documentation of security threats, mitigating controls, and effectiveness.

### Loading the Skill

Use the threat and control documentation skill:

```
I'm using ordis/security-architect/documenting-threats-and-controls for threat-control relationships.
```

### Input Prompt

Document specific threats and how controls mitigate them:

```
I need comprehensive threat-control documentation for our HIPAA EHR system. Please document threats, mitigating controls, and effectiveness.

**Key Threats to Address:**

1. Unauthorized Access to PHI
   - Threat: Disgruntled staff accessing patient records
   - Threat: External attackers compromising credentials
   - Threat: Insider threats accessing more data than role permits

2. Data Breach/Exposure
   - Threat: Database breach exposing encrypted PHI
   - Threat: Unencrypted data in transit
   - Threat: Backup data exposure
   - Threat: Log files containing sensitive information

3. System Availability/Downtime
   - Threat: DDoS attacks affecting patient access
   - Threat: Infrastructure failure causing service outage
   - Threat: Ransomware attack on database

4. Audit Trail Manipulation
   - Threat: Tampering with access logs
   - Threat: Incomplete audit logging
   - Threat: Log deletion or rotation without retention

5. Credential Compromise
   - Threat: Password reuse across systems
   - Threat: Phishing attacks targeting staff
   - Threat: Compromised API keys or secrets

**Implemented Controls:**

Administrative:
- Role-based access control (RBAC) with 8 privilege levels
- Monthly access reviews and recertification
- Quarterly security awareness training (mandatory)
- Incident response plan with <1 hour escalation
- Data steward program with clinical oversight

Physical:
- Data center with ISO 27001 certification
- Video surveillance in server rooms
- Biometric access to secure areas
- Environmental controls (fire suppression, temperature)

Technical:
- AES-256 encryption at rest (database, backups, archives)
- TLS 1.3 for all data in transit
- HashiCorp Vault for secret management
- AWS WAF for DDoS protection
- CloudTrail and VPC Flow Logs enabled
- Multi-factor authentication (TOTP) for all staff
- Database activity monitoring with real-time alerts
- Automated backup verification and restoration testing

Please create:
1. Threat register with severity ratings
2. Control-to-threat mapping matrix
3. Control effectiveness assessment (strong/moderate/weak)
4. Residual risk analysis
5. Control testing recommendations
```

### Expected Output

**Threat Register:**
- Comprehensive list of identified threats
- Severity ratings (Critical/High/Medium/Low)
- Likelihood and impact assessment
- Owner and remediation timeline

**Control Effectiveness Matrix:**
- Each threat mapped to 1+ mitigating controls
- Effectiveness rating (Strong/Moderate/Weak)
- Evidence of control operation
- Testing frequency and results

**Residual Risk Assessment:**
- Remaining risk after all controls applied
- Risk acceptance decisions
- Compensating controls where needed
- Monitoring and alerting configuration

### Observations

**Key Insights:**
- Multiple controls layered for defense-in-depth
- Administrative controls (policies, training) support technical controls
- Audit trails provide evidence for compliance investigations
- Regular control testing validates effectiveness
- Residual risk must be documented and accepted by management

### Time Estimate

15-20 minutes to document threats, controls, and effectiveness.

---

## Step 3: Create Operational Acceptance Documentation

**Estimated Time:** 20 minutes

**Objective:** Develop operational runbooks and acceptance criteria for PHI handling procedures.

### Loading the Skill

Use the operational acceptance documentation skill:

```
I'm using muna/technical-writer/operational-acceptance-documentation for operational procedures.
```

### Runbook Template: User Access Provisioning

Create detailed runbooks following this structure:

**RUNBOOK: New User Access Provisioning**

**Purpose:** Establish secure access for new clinical and administrative staff to the EHR system

**Scope:** Applies to all new hires, contractors, and third-party users

**Prerequisites:**
- Manager approval documented
- Role and access levels pre-approved
- Identity verification completed
- HIPAA training certification completed

**Procedure Steps:**

1. **Access Request Submission**
   - Manager submits access request via ServiceNow
   - Include: User name, role, clinic location, start date, managed by manager name
   - Compliance officer reviews within 24 hours
   - Approval triggers provisioning automation

2. **Identity and Access Control Setup**
   - Create Active Directory account following naming convention: firstname.lastname@company.ehr
   - Assign to appropriate security groups based on role:
     - Clinical staff: [clinic-name]-clinicians
     - Administrative: [clinic-name]-admin
     - Physicians: [clinic-name]-physicians
     - Patient access: patient-portal-users
   - Generate temporary password using HashiCorp Vault
   - Create MFA secret (TOTP) and backup codes
   - Log all actions in CloudTrail (automated)

3. **EHR System Provisioning**
   - Provision Django user account with assigned role
   - Apply row-level security rules for clinic data isolation
   - Grant database schema access per role requirements
   - Configure audit logging to capture all user actions
   - Enable session monitoring and real-time alerting

4. **First-Time Onboarding**
   - User receives credential delivery package (encrypted)
   - Forced password change on first login
   - Required acceptance of HIPAA Business Associate Agreement
   - MFA setup verification
   - System orientation training (30 minutes)
   - Test access to assigned clinics and data

5. **Audit Trail Documentation**
   - Record: creation timestamp, creator, approver, access level
   - Generate access certificate for audit file
   - Store credentials in encrypted credential vault
   - Enable real-time audit logging in CloudWatch

**Success Criteria:**
- User successfully logs in with MFA
- Access to assigned clinic data only
- No access to other clinics' PHI
- Audit logs show all provisioning steps
- User completes MFA setup and training
- Completion time: <2 business hours

**Exception Handling:**
- Emergency access: Requires VP approval + 24-hour audit review
- Vendor access: Separate process with temporary credentials (90-day max)
- After-hours access: Logs require next-day review by security officer

---

### Runbook Template: Backup and Disaster Recovery

**RUNBOOK: Daily Backup Verification and Testing**

**Purpose:** Ensure backup integrity and disaster recovery capability

**Frequency:** Daily for incremental, weekly for full backups, monthly for restore testing

**Critical Metrics:**
- RTO (Recovery Time Objective): <4 hours for full system recovery
- RPO (Recovery Point Objective): <1 hour of data loss
- Backup success rate: 99.9%
- Restore verification: 100% of backups tested monthly

**Backup Schedule:**
- Full backup: Sundays 2:00-4:00 AM EST (off-peak)
- Incremental backups: Daily 3:00 AM EST
- Backup retention: 30 days local, 90 days in S3 Glacier
- All backups encrypted with AES-256 (managed by AWS KMS)

**Daily Backup Verification Process:**

1. **Automated Backup Completion Check**
   - CloudWatch monitors backup completion status
   - Alert if backup takes >120 minutes (threshold)
   - Alert if backup fails or is missing
   - Verify backup size within expected range (±10%)

2. **Backup Integrity Validation**
   - Calculate MD5 checksum of backup file
   - Compare against expected hash (stored in DynamoDB)
   - Flag any discrepancies for investigation
   - Document validation in backup log

3. **Encryption Verification**
   - Confirm all backups encrypted with correct KMS key
   - Verify key access logs (CloudTrail)
   - Ensure key rotation schedule maintained
   - Alert on any unauthorized key access attempts

4. **Backup Restore Testing (Monthly)**
   - Restore from previous month's full backup to isolated environment
   - Verify data completeness (row count, checksum)
   - Test database functionality:
     - Patient record retrieval
     - Report generation
     - User authentication
     - Audit log access
   - Document restore time (should be <2 hours for full database)
   - Clean up test environment

5. **Documentation and Reporting**
   - Log backup details: size, duration, checksum, completion time
   - Generate monthly backup report for compliance
   - Include: success rate, failures, restore test results
   - Attach to audit file for annual review

**Success Criteria:**
- All backups complete within SLA window
- Backup integrity validation passes 100%
- Monthly restore test successful
- Recovery time within RTO
- No data loss during restore (RPO met)

**Failure Procedures:**
- Backup failure: Alert security team immediately, investigate root cause, manual backup if needed
- Restore failure: Escalate to DBA, analyze logs, restore from previous backup
- Encryption failure: Check KMS key status, verify IAM permissions, attempt re-encryption

---

### Healthcare-Specific Acceptance Criteria

**Electronic Protected Health Information (ePHI) Handling Verification:**

1. **Access Logging Requirements**
   - Every read/write of patient ePHI logged with:
     - User ID and timestamp
     - Patient MRN accessed
     - Data elements accessed (diagnosis, prescription, lab)
     - Purpose of access
     - System action (view, download, print, export)
   - Logs retained for 6 years (HIPAA requirement)
   - Real-time alerting for suspicious access patterns

2. **HIPAA Breach Notification Rule Compliance**
   - Breach definition: Unauthorized access/disclosure of unsecured ePHI
   - Detection: System alert within 1 hour of breach discovery
   - Investigation: Root cause analysis within 24 hours
   - Notification timeline:
     - Affected individuals: within 60 days
     - Media (if 500+ affected): within 60 days
     - HHS Secretary: within 60 days
   - Documentation required:
     - Dates of breach and discovery
     - Affected individuals and data elements
     - Remedial actions taken
     - Investigation findings

3. **Data Minimization and Audit Trail**
   - Only necessary ePHI displayed in UI (no SSN in logs)
   - Bulk exports require multi-factor approval
   - Export audit trail includes recipient and use case
   - Business associate data sharing tracked separately
   - Patient can request access log audit

**Acceptance Testing Checklist:**
- [ ] All new user access verified in CloudWatch logs
- [ ] Privilege escalation blocked and alerted
- [ ] MFA required for all users (100% enforcement)
- [ ] Backup restoration successful in <4 hours
- [ ] No ePHI found in application logs or error messages
- [ ] Patient access audit trail generated successfully
- [ ] Encrypt-in-transit validation (TLS 1.3)
- [ ] Encrypt-at-rest validation (AES-256)
- [ ] Segregation of duties enforced (no single admin access)
- [ ] Audit logs immutable and tamper-evident

### Input Prompt

Document standard operating procedures for compliance operations:

```
I need operational acceptance documentation for our EHR system. Please help create runbooks and operational acceptance criteria.

**Key Operational Processes to Document:**

1. User Access Provisioning
   - Access request and approval workflow
   - Credential generation and delivery
   - MFA setup and backup codes
   - First-time login procedures
   - Access audit trail

2. Privileged Access Management (PAM)
   - Database admin access procedures
   - Emergency access procedures
   - Session recording and audit
   - Privilege escalation controls
   - Just-in-time (JIT) access approval

3. Incident Response
   - Breach detection and validation
   - Evidence preservation procedures
   - Notification workflow (HHS, patients, media)
   - Forensic investigation procedures
   - System recovery and rebuild

4. Data Access and Export
   - Patient record access logging
   - Bulk data export controls
   - Business associate data sharing
   - Secondary use approvals
   - Data retention and deletion

5. Backup and Disaster Recovery
   - Backup schedule and retention
   - Backup encryption verification
   - Restore testing procedures
   - RTO/RPO validation
   - Disaster recovery drill procedures

6. Security Update Management
   - Patch assessment and scheduling
   - Testing in non-production
   - Production deployment procedures
   - Rollback procedures
   - Audit trail of all changes

**Operational Teams:**
- Clinical Operations (10 staff)
- IT Operations (5 staff)
- Security Operations (2-3 staff)
- Database Administration (2 staff)
- System Engineering (3 staff)

**Compliance Requirements:**
- All procedures documented and approved
- Staff trained on procedures (annually)
- Procedure compliance monitored
- Non-compliance triggers remediation
- Audit evidence collected per procedure

Please create:
1. Operational runbooks for each process
2. Acceptance criteria and success measures
3. Training requirements and certifications
4. Audit evidence collection procedures
5. Procedure version control and change log
```

### Expected Output

**Operational Runbooks:**
- Step-by-step procedures with screenshots/examples
- Decision trees for exception handling
- Contact information for escalation
- Success criteria for each step
- Estimated execution time

**Acceptance Criteria:**
- Measurable indicators of successful operation
- Logging and audit trail requirements
- Performance baselines (speed, accuracy)
- Quality assurance checkpoints
- Compliance verification methods

**Training and Certification:**
- Role-specific training modules
- Competency assessments
- Certification renewal schedule
- Documentation of training completion

### Observations

**Implementation Notes:**
- Operational procedures are often the weakest link in compliance
- Staff training and certification are critical
- Procedure compliance must be audited regularly
- Exception handling must be documented and approved
- Incident response procedures require regular drills

### Time Estimate

12-15 minutes to develop comprehensive runbooks and acceptance criteria.

---

## Step 4: Establish Governance Policies and ITIL Framework

**Estimated Time:** 25 minutes

**Objective:** Create IT governance policies and establish ITIL-based service management framework.

### Loading the Skill

Use the governance and ITIL documentation skill:

```
I'm using muna/technical-writer/itil-and-governance-documentation for governance policies.
```

### IT Governance Policy Templates

#### Policy Template: Change Management

**POLICY: Change Management for Healthcare EHR Systems**

**Policy Number:** GOV-2024-001-CM
**Effective Date:** [Date]
**Last Reviewed:** [Date]
**Next Review:** [Date + 1 year]
**Owner:** Chief Information Officer

**1. Purpose and Scope**

This policy establishes mandatory change management procedures for all changes to the healthcare EHR system to ensure:
- Service availability and reliability (99.95% target)
- Regulatory compliance (HIPAA, SOC 2, state laws)
- Data integrity and security
- Audit trail for all changes
- Zero unplanned outages

Applies to:
- Production database schema changes
- Application code deployments
- Infrastructure changes (servers, networks)
- Security configuration modifications
- Third-party integration updates

Does NOT apply to:
- Configuration within application UI
- Non-production environments
- Emergency patches (managed separately)

**2. Change Classification and Approval Authority**

**Standard Changes (Low Risk):**
- Configuration updates within existing parameters
- Documentation updates
- Approved vendor patches
- Approval: Change Manager
- Testing: Non-production only
- Deployment window: Anytime (non-production)

**Normal Changes (Medium Risk):**
- Application feature releases
- Infrastructure scaling
- Moderate database modifications
- Approval: Change Advisory Board (CAB)
- Testing: Staging environment (production-equivalent)
- Deployment window: 8 AM - 2 PM EST, non-weekend
- Rollback plan: Required

**Emergency Changes (High Risk):**
- Security patches for vulnerabilities
- Emergency infrastructure changes
- Incident remediation
- Approval: VP Operations + CISO
- Testing: Expedited (minimum 1 hour)
- Deployment window: Immediate
- Rollback plan: Tested before deployment

**3. Change Request Process (Normal Changes)**

**Step 1: Initiation**
- Requestor completes change request form in ServiceNow:
  - Change title and description
  - Business justification and risk assessment
  - Affected systems and dependencies
  - Estimated duration and testing time
  - Rollback procedure
  - Estimated downtime (if any)

**Step 2: Assessment**
- Change Manager reviews for completeness within 24 hours
- Technical lead performs impact analysis:
  - System dependencies map
  - Database schema impact
  - Performance implications
  - Security implications
- Estimated risk level assigned (Low/Medium/High)

**Step 3: CAB Approval**
- CAB meets weekly (Mondays, 2 PM EST)
- Members: CISO, VP Operations, Engineering Lead, Compliance Officer
- Voting: All present members must approve
- Approval documented in ServiceNow
- Requestor notified within 2 hours of meeting

**Step 4: Implementation Planning**
- Change owner develops detailed implementation plan:
  - Step-by-step procedures
  - Parallel testing procedures
  - Success criteria and validation
  - Rollback triggers and procedures
  - Communication plan for affected users
- Test environment setup (mirror of production)
- Data backup before change (automated)

**Step 5: Testing and Validation**
- Minimum 1-week testing window for Normal changes
- Functional testing: All affected features verified
- Security testing: No new vulnerabilities introduced
- Performance testing: Baseline metrics met
- Accessibility testing: Compliance verified
- Document test results in change log

**Step 6: Deployment and Monitoring**
- Deployment window approved by CAB
- Deployment team follows procedures exactly
- Real-time monitoring during deployment:
  - Application performance metrics
  - Error rate and exceptions
  - User access and authentication
  - Database performance
- Canary deployment (10% of servers first) for large changes
- Full rollback capability until change is declared "stable"

**Step 7: Post-Implementation Review**
- Change verified successful within 24 hours
- Actual vs. estimated duration recorded
- Any issues or workarounds documented
- User feedback collected and documented
- Change marked "Completed" or "Closed"
- Lessons learned documented for future changes

**4. Deployment Windows and Schedules**

**Standard Deployment Windows:**
- **Production:** 8 AM - 2 PM EST, Monday-Friday
- **Estimated downtime:** Maximum 15 minutes (communicated 48 hours prior)
- **Patient notification:** Automated email if downtime expected
- **Clinician notification:** In-app banner 48 hours before

**Emergency Deployment:**
- Available 24/7 for critical security patches
- VP Operations approval required
- Incident ticket created and referenced
- Downtime minimized with load balancing

**Blackout Periods (No Changes Allowed):**
- December 23 - January 2 (holiday)
- Week before and after major compliance audits
- During disaster recovery drills

**5. Audit Trail and Compliance**

All changes logged with:
- Change request ID and approver
- Deployment timestamp and duration
- Changed components and before/after states
- CloudTrail entries for AWS changes
- Git commit log for code changes
- Database schema version records
- Rollback history (if applicable)

Retention: 7 years (per HIPAA requirements)

Access control: Only authorized personnel can view change details

---

#### Policy Template: Access and Authorization

**POLICY: Access and Authorization Governance**

**Policy Number:** GOV-2024-002-ACCESS
**Owner:** Chief Information Security Officer

**1. Access Principle: Least Privilege**

Every user granted only minimum access necessary for their role:
- Role-based access control (RBAC) with 8 predefined roles
- No permanent super-admin access
- Periodic access reviews (quarterly minimum)
- Just-in-time (JIT) privileged access (temporary, time-limited)

**2. Access Roles and Permissions**

| Role | Data Access | Write Access | Admin Functions |
|------|-----------|-------------|-----------------|
| Patient | Own records only | Update own profile | None |
| Clinician | Assigned clinic records | Enter clinical notes | None |
| Physician | Full clinic records | Clinical decisions | Referrals |
| Clinic Admin | Clinic billing/admin | User management | Clinic reporting |
| System Admin | All data (audited) | Emergency only | System configuration |
| Auditor | All records (read-only) | None | Audit logging |
| DBA | Database schema | Migration scripts | Database tuning |
| Security Officer | Security logs | Alert review | Security reporting |

**3. Access Request and Approval Workflow**

- Employee's manager submits request (with business justification)
- Role owner (department head) approves appropriateness
- Compliance officer reviews for segregation of duties violations
- All approvals documented in access request ticket
- System provisioning within 24 hours of approval
- User notified with access details and responsibilities
- Training required before access granted

**4. Periodic Access Reviews**

**Quarterly Access Review:**
- For all users (100% coverage)
- Manager certifies: "I have reviewed access for [user] and confirm it is appropriate for their current role"
- Compliance officer flags any exceptions
- Exceptions require VP sign-off
- Results documented in audit file

**Annual Comprehensive Review:**
- Access rights recertified from scratch (zero-based)
- Reports generated: users by role, permissions by data type
- Segregation of duties violations identified
- Unnecessary access removed

**Triggered Reviews:**
- Within 24 hours of:
  - Role change
  - Department transfer
  - Disciplinary action
  - Suspicious access detected

**5. Segregation of Duties Enforcement**

These combinations of access are prohibited (mutually exclusive):
- Approver + Implementer: Can't approve own change requests
- Preparer + Approver: Can't approve own financial transactions
- Access Approver + Access Recipient: Compliance officer can't request own access
- Audit access + System admin: Can't audit own administrative actions

Violations automatically prevented by role configuration.

**6. Emergency Access Procedures**

**Authorization:**
- VP Operations approves emergency access
- Reason documented: security incident, production outage, etc.
- Time limit: 4 hours maximum (renewable)

**Execution:**
- Temporary account created with elevated privileges
- All actions logged to tamper-evident audit trail
- Real-time alerts if access used unusually
- Screenshots/recordings of all system changes

**Post-Event:**
- Access revoked immediately when not needed
- Audit logs reviewed by CISO within 24 hours
- Any system changes independently verified
- Documentation added to incident file

**7. Credential Management**

- **Passwords:** Minimum 12 characters, complexity required
- **Password expiration:** 90 days (enforced by Active Directory)
- **Password reuse:** Last 10 passwords blocked
- **Shared accounts:** Prohibited (except break-glass emergency)
- **API keys:** Rotated every 30 days
- **Secrets storage:** HashiCorp Vault (encrypted)
- **Temporary credentials:** 4-hour maximum validity
- **MFA:** Mandatory for all users (TOTP or hardware key)

---

### ITIL Service Management Process Flowchart

**Incident Management Workflow:**

```
User reports issue (phone/email/portal)
                 |
                 v
Create Incident ticket (ServiceNow)
Assign severity (Critical/High/Medium/Low)
                 |
    +--------+---+---+--------+
    |        |       |        |
(Critical)(High)(Medium)(Low)
    |        |       |        |
    v        v       v        v
Response SLA: 1h   4h    8h   24h
Resolution SLA: 4h  8h   24h  72h
                 |
    +-------+----+----+
    |       |    |    |
Incident assigned to support teams
                 |
                 v
Diagnosis: Root cause identified
                 |
    +----+---+---+
    |    |   |
Can we fix it? (Yes/No/Escalate)
    |    |   |
    v    v   v
   Fix   Workaround  Escalate to
                     Engineering
    |    |   |
    +----+---+
         |
         v
    Apply solution
         |
         v
    User verifies fix
         |
    Yes-+--No
    |       |
    v       v
Incident  Reopen &
Closed    Reassign
    |       |
    +---+---+
        |
        v
    Schedule post-incident review
    (if sev 1-2)
```

**Change Management Workflow:**

```
Change request submitted → CAB review → Approved?
                                         Yes → Implementation planning
                                         No  → Return to requestor

Implementation planning → Testing in staging
                              ↓
                         All tests pass?
                         Yes → Schedule deployment
                         No  → Fix issues, retest

Deploy to production → Monitor for 4 hours
                        ↓
                     Issues detected?
                     Yes → Rollback & investigate
                     No  → Mark complete
```

---

### Governance Dashboard and Metrics

Track compliance with governance framework:

**Key Performance Indicators (KPIs):**
- **Change Success Rate:** % of changes deployed without rollback (Target: 99%)
- **Change Cycle Time:** Days from request to deployment (Target: <7 days for normal changes)
- **Incident MTTR:** Mean time to resolution by severity
- **Access Review Compliance:** % of users with current access certification (Target: 100%)
- **Policy Compliance:** % of staff trained on policies (Target: 100% annually)
- **Audit Findings:** Critical/high findings from annual compliance audit
- **Security Incidents:** Number and severity of security incidents
- **Unplanned Outages:** Incidents causing service unavailability

**Monthly Governance Report:**
- Summary of approved/deployed changes
- Incidents by severity and resolution time
- Access review status and exceptions
- Security findings and remediation status
- Policy compliance metrics
- Risk register updates
- Trends and recommendations

### Input Prompt

Establish comprehensive IT governance framework:

```
I need to establish IT governance and service management for our healthcare EHR system. Please help create governance policies and ITIL framework.

**Governance Areas to Address:**

1. Change Management
   - Change request process and approval
   - Risk assessment for changes
   - Testing and validation requirements
   - Rollback procedures and triggers
   - Change calendar and scheduling
   - Change audit trail and evidence

2. Incident Management
   - Incident classification and severity levels
   - Incident workflow and SLAs
   - Root cause analysis (RCA) procedures
   - Incident communication protocols
   - Post-incident review requirements

3. Problem Management
   - Known error database
   - Problem trend analysis
   - Permanent fix implementation
   - Workaround documentation
   - Problem lifecycle tracking

4. Service Level Management
   - Service availability targets (99.95% uptime)
   - Response time SLAs
   - Resolution time objectives
   - Performance baselines
   - Escalation procedures

5. Capacity and Performance Management
   - Capacity planning and forecasting
   - Performance monitoring baselines
   - Resource utilization thresholds
   - Scaling procedures
   - Performance optimization initiatives

6. Configuration Management
   - Configuration item (CI) database
   - Asset tracking and inventory
   - CI relationships and dependencies
   - Change tracking per CI
   - Audit trail for all changes

7. Access and Authorization Governance
   - Access governance policy
   - Segregation of duties (SOD) controls
   - Periodic access review (quarterly)
   - Access removal procedures
   - Emergency access logging

8. Compliance and Audit Governance
   - Compliance calendar and schedule
   - Audit coordination
   - Evidence collection procedures
   - Remediation tracking
   - Annual compliance certification

**Current Organization:**
- Chief Information Security Officer (CISO)
- VP of Operations
- Operations Managers (3)
- Service Delivery Leads (5)
- Change Advisory Board (CAB): CISO, VP Ops, Engineering Lead, Compliance Officer

**Regulatory Oversight:**
- HIPAA compliance officer (required)
- Security officer (internal)
- Business associate audit (annual)
- Internal audit (annual)
- External compliance assessments (as needed)

Please create:
1. Comprehensive IT governance policy document
2. ITIL process definitions and workflows
3. Governance roles and responsibilities matrix
4. Decision-making authority matrix
5. Compliance monitoring and reporting procedures
```

### Expected Output

**Governance Policies:**
- Change Management Policy (with approval authorities)
- Incident Management Policy (with SLAs)
- Problem Management Policy (with RCA procedures)
- Access Management Policy (with segregation of duties)
- Service Level Policy (with targets and escalations)

**ITIL Service Management Framework:**
- Service Catalog definition
- Change Advisory Board charter
- Service Level Agreements (SLAs)
- Incident Response procedures
- Problem management workflows
- Configuration Management Database (CMDB) structure

**Roles and Responsibilities:**
- RACI matrix for key processes
- Decision authority for approvals
- Escalation paths
- Training requirements
- Performance metrics and KPIs

**Compliance Monitoring:**
- Dashboard and reporting
- Metrics collection procedures
- Audit evidence gathering
- Non-compliance escalation
- Continuous improvement process

### Observations

**Governance Framework Benefits:**
- Provides structure and accountability
- Enables consistent, repeatable processes
- Supports compliance audits with evidence
- Facilitates continuous improvement
- Builds institutional knowledge

**Critical Success Factors:**
- Executive sponsorship and commitment
- Clear roles and responsibilities
- Regular training and competency assessment
- Metrics and performance monitoring
- Continuous process improvement

### Time Estimate

12-15 minutes to establish governance framework and policies.

---

## Conclusion

You have successfully created comprehensive compliance documentation for a HIPAA-regulated healthcare system. This documentation provides:

**Compliance Foundation:**
- HIPAA Security Rule controls mapped to technical implementations
- SOC 2 Type II Trust Service Criteria alignment
- Documented threats and mitigating controls
- Residual risk assessment and acceptance

**Operational Excellence:**
- Detailed runbooks for key operational processes
- Acceptance criteria and success measures
- Training and certification programs
- Audit evidence collection procedures

**Governance Framework:**
- IT governance policies and procedures
- ITIL-based service management
- Clear roles, responsibilities, and decision authority
- Compliance monitoring and reporting

**Audit Readiness:**
- Comprehensive documentation for auditors
- Evidence trails for all controls
- Documented compliance procedures
- Regular testing and validation results

### Next Steps

1. **Training and Implementation:**
   - Conduct staff training on documented procedures
   - Certify compliance with operational requirements
   - Monitor procedure compliance through audits

2. **Audit Preparation:**
   - Organize documentation for auditors
   - Prepare evidence collection procedures
   - Schedule internal audit reviews

3. **Continuous Improvement:**
   - Review and update policies annually
   - Incorporate audit findings
   - Improve processes based on metrics
   - Stay current with regulatory changes

4. **System Evolution:**
   - Plan for regulatory updates
   - Monitor healthcare IT trends
   - Enhance security controls as needed
   - Scale governance for growth

### Key Takeaways

- **Documentation is the foundation** of healthcare compliance; controls must be documented and evidenced
- **Layered controls** (administrative, physical, technical) are required by HIPAA
- **Operational procedures** are critical; policies alone don't ensure compliance
- **Governance framework** provides accountability and consistency
- **Regular auditing and testing** validates control effectiveness
- **Training and awareness** are essential for compliance culture

---

**Total Estimated Time:** 60-75 minutes

**Difficulty:** Advanced (requires healthcare IT and compliance knowledge)

**Skillpack Required:** ordis/security-architect

**Skills Used:**
1. compliance-awareness-and-mapping
2. documenting-threats-and-controls
3. operational-acceptance-documentation
4. itil-and-governance-documentation
