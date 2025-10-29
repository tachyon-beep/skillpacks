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

**HIPAA Control Mapping Matrix:**
- Administrative Safeguards → Access Control, IAM, Audit Logging, Training Programs
- Physical Safeguards → Data Center Controls, Workstation Security, Device Management
- Technical Safeguards → Encryption, Access Controls, Audit Controls, Key Management

**SOC 2 Alignment:**
- Security: Unauthorized access prevention measures
- Availability: Disaster recovery and business continuity
- Processing Integrity: Data accuracy and completeness
- Confidentiality: PHI encryption and access controls

**Gap Analysis:**
- Identify any HIPAA requirements not currently addressed
- Prioritize gaps by risk level and implementation effort

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

**Estimated Time:** 15 minutes

**Objective:** Develop operational runbooks and acceptance criteria for PHI handling procedures.

### Loading the Skill

Use the operational acceptance documentation skill:

```
I'm using ordis/security-architect/operational-acceptance-documentation for operational procedures.
```

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

**Estimated Time:** 15 minutes

**Objective:** Create IT governance policies and establish ITIL-based service management framework.

### Loading the Skill

Use the governance and ITIL documentation skill:

```
I'm using ordis/security-architect/itil-and-governance-documentation for governance policies.
```

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
