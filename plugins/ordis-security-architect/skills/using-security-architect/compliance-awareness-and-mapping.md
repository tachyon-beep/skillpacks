
# Compliance Awareness and Mapping

## Overview

Navigate diverse compliance frameworks systematically. Core principle: **ALWAYS ask which frameworks apply - don't assume**.

**Key insight**: Frameworks vary by jurisdiction and industry. Discovery process matters more than memorizing frameworks.

## When to Use

Load this skill when:
- Starting projects in regulated environments
- Preparing for compliance audits
- Mapping technical controls to requirements
- Working with healthcare, finance, government data

**Symptoms you need this**:
- "What compliance frameworks do we need?"
- Preparing for SOC2, HIPAA, PCI-DSS, IRAP audits
- Government/defense contract compliance
- "How do I map our controls to [framework]?"

**Don't use for**:
- Implementation of specific controls (use `ordis/security-architect/security-controls-design`)
- Security architecture without compliance requirements

## The Discovery Process

### Step 1: Ask Three Questions

**Before** identifying frameworks, ask:

1. **"What jurisdiction?"**
   - Australia → ISM (current edition), IRAP, Privacy Act 1988 (as amended 2024), PSPF
   - United Kingdom → Cyber Essentials / Cyber Essentials Plus, NCSC guidance, UK GDPR, DPA 2018
   - United States → NIST CSF 2.0 (Feb 2024), FedRAMP (Rev 5 baselines), FISMA, EO 14028
   - European Union → NIS2 (Dir. 2022/2555, transposition deadline Oct 2024), GDPR, ISO/IEC 27001:2022, EU AI Act (Reg. 2024/1689), CRA (Reg. 2024/2847)

2. **"What industry?"**
   - Healthcare → HIPAA (US), GDPR (EU), Australian Privacy Principles
   - Finance → PCI-DSS (payments), SOX (US), Basel III
   - Government/Defense → Jurisdiction-specific (ISM, FedRAMP, etc.)
   - General SaaS → SOC2, ISO 27001

3. **"What data types?"**
   - Personal data → Privacy laws (GDPR, Privacy Act)
   - Payment card data → PCI-DSS
   - Health records → Healthcare-specific (HIPAA, etc.)
   - Classified data → Government frameworks (ISM, FedRAMP)

**Never assume.** Same project can have multiple frameworks (e.g., Australian hospital SaaS = Privacy Act + Healthcare-specific + possibly SOC2 if B2B).


### Step 2: Framework Stacking

**Multiple frameworks often apply simultaneously.**

#### Example: Australian Healthcare SaaS

```
Data type: Patient health records
Jurisdiction: Australia
Industry: Healthcare
Business model: SaaS (B2B to hospitals)

Applicable frameworks:
1. Privacy Act 1988 (Australian Privacy Principles) - MANDATORY
2. My Health Records Act 2012 - if using My Health Records system
3. State-specific health regulations (e.g., NSW Health Privacy Manual)
4. SOC2 (if hospitals require it for vendor assurance)
5. ISO 27001 (if targeting enterprise healthcare market)

Priority:
- MANDATORY: Privacy Act (legal requirement)
- HIGHLY RECOMMENDED: SOC2 (market expectation for B2B SaaS)
- OPTIONAL: ISO 27001 (competitive advantage)
```

**Key insight**: Don't just pick one framework. Identify ALL that apply, then prioritize by legal vs market requirements.


### Step 3: Understand Framework Structure

Each framework has structure - learn it before mapping controls.

#### Common Framework Components

**1. Control Categories** (groups of related controls):
- Access Control
- Encryption
- Audit Logging
- Incident Response
- Vulnerability Management
- Configuration Management
- Personnel Security
- Physical Security

**2. Control Requirements** (specific technical/operational requirements):
- "System must enforce least privilege access"
- "All sensitive data encrypted at rest using AES-256"
- "Security events logged and retained for 90 days"

**3. Evidence Requirements** (what auditors need):
- Configuration screenshots
- Log samples
- Policy documents
- Test results
- Interview records

**4. Assessment Procedures** (how controls are tested):
- Configuration review
- Log analysis
- Penetration testing
- Interviews with staff


### Step 4: Control Inventory

Before mapping to framework, inventory what you have.

#### Technical Controls Inventory Template

```markdown
## Access Control

| Control | Description | Implementation | Evidence Location |
|---------|-------------|----------------|-------------------|
| AC-1 | User authentication | AWS IAM with MFA | aws-iam-config.json |
| AC-2 | Role-based access | PostgreSQL roles | postgres-roles.sql |
| AC-3 | Session timeout | 30-minute idle timeout | app-config.yaml |

## Encryption

| Control | Description | Implementation | Evidence Location |
|---------|-------------|----------------|-------------------|
| ENC-1 | Data at rest | AES-256, AWS KMS | kms-config.json |
| ENC-2 | Data in transit | TLS 1.3 | nginx-ssl-config |

## Audit Logging

| Control | Description | Implementation | Evidence Location |
|---------|-------------|----------------|-------------------|
| LOG-1 | Authentication events | CloudWatch Logs | cloudwatch-config |
| LOG-2 | Data access logs | PostgreSQL query logs | pg-audit-config |
| LOG-3 | Admin actions | Audit trail table | audit-schema.sql |
```

**Why inventory first**: Easier to map existing controls to requirements than build from scratch.


### Step 5: Create Traceability Matrix

Map your controls to framework requirements.

#### SOC2 Traceability Matrix Example

```markdown
| SOC2 Criterion | Control Category | Our Control | Implementation | Evidence | Status |
|----------------|------------------|-------------|----------------|----------|--------|
| CC6.1 (Logical access) | Access Control | AC-1: MFA | AWS IAM | aws-iam-config.json | ✅ Complete |
| CC6.1 | Access Control | AC-2: RBAC | PostgreSQL | postgres-roles.sql | ✅ Complete |
| CC6.6 (Encryption) | Encryption | ENC-1: At rest | AWS KMS | kms-config.json | ✅ Complete |
| CC6.6 | Encryption | ENC-2: In transit | TLS 1.3 | nginx-ssl-config | ✅ Complete |
| CC7.2 (Monitoring) | Audit Logging | LOG-1: Auth events | CloudWatch | cloudwatch-config | ✅ Complete |
| CC7.2 | Audit Logging | LOG-2: Data access | PostgreSQL | pg-audit-config | ⚠️ Partial (retention = 30 days, need 90) |
| CC8.1 (Change mgmt) | Config Mgmt | CM-1: Version control | GitHub | github-repos | ❌ Missing approval workflow |
```

**Gap identification**: ⚠️ Partial and ❌ Missing items become your remediation backlog.


### Step 6: Gap Analysis and Remediation

Identify missing/insufficient controls, prioritize by risk.

#### Gap Analysis Template

```markdown
# Compliance Gap Analysis: SOC2

## Critical Gaps (Block Audit)

### GAP-1: Missing Change Management Approval Workflow
- **Requirement**: CC8.1 - Changes must have approval before production
- **Current state**: Git commits directly to main without approval
- **Impact**: HIGH - Cannot pass SOC2 without this
- **Remediation**:
  - Implement GitHub branch protection (require PR approval)
  - Create approval policy (2 reviewers for production changes)
  - Document change management policy
- **Timeline**: 2 weeks
- **Owner**: DevOps Lead
- **Cost**: $0 (GitHub feature)

## High-Priority Gaps (Remediate Before Audit)

### GAP-2: Insufficient Log Retention
- **Requirement**: CC7.2 - Logs retained for 90 days minimum
- **Current state**: PostgreSQL logs retained 30 days
- **Impact**: MEDIUM - Auditor will note as deficiency
- **Remediation**:
  - Extend PostgreSQL log retention to 90 days
  - Archive to S3 for long-term storage
  - Update retention policy document
- **Timeline**: 1 week
- **Owner**: Platform Engineer
- **Cost**: ~$50/month (S3 storage)

## Low-Priority (Post-Audit)

### GAP-3: No Formal Incident Response Tabletop Exercises
- **Requirement**: CC7.5 - Test incident response procedures
- **Current state**: IR runbooks exist but not tested
- **Impact**: LOW - Can be remediated post-audit
- **Remediation**: Schedule quarterly IR tabletop exercises
- **Timeline**: 3 months
- **Owner**: Security Team
```

**Prioritization**:
1. **Critical**: Must fix before audit (blocks compliance)
2. **High**: Should fix before audit (reduces risk of findings)
3. **Low**: Can defer to post-audit (continuous improvement)


## Universal Control Categories

**Frameworks differ in details but share core categories.**

### 1. Access Control
- Authentication (passwords, MFA, SSO)
- Authorization (RBAC, ABAC, least privilege)
- Session management (timeouts, revocation)

### 2. Encryption
- Data at rest (AES-256, authenticated key management; envelope encryption with HSM/KMS where required)
- Data in transit (TLS 1.3 default; TLS 1.2 acceptable for legacy interop only — TLS 1.0/1.1 deprecated by RFC 8996 and most frameworks)
- Key rotation, separation of duties, and access control on key material
- **Crypto-agility** — design for algorithm migration; track post-quantum (FIPS 203/204/205) as a multi-year roadmap concern

### 3. Audit Logging
- Authentication events (login, logout, failures)
- Data access (who accessed what, when)
- Admin actions (configuration changes, user management)
- Log retention (30-90 days typical, varies by framework)

### 4. Incident Response
- Detection (monitoring, alerting)
- Containment (isolation procedures)
- Recovery (restoration procedures)
- Lessons learned (post-incident reviews)

### 5. Vulnerability Management
- Patch management (timely updates)
- Vulnerability scanning (regular cadence)
- Penetration testing (annual or on major changes)

### 6. Configuration Management
- Secure baselines (hardening guides)
- Change control (approval processes)
- Configuration monitoring (detect drift)

### 7. Personnel Security
- Background checks (role-appropriate)
- Security training (annual, role-specific)
- Offboarding procedures (revoke access)

### 8. Physical Security
- Facility access controls
- Environmental controls (fire, flood)
- Equipment disposal (data sanitization)

**Use this as checklist**: Most frameworks require these categories. Implement once, map to multiple frameworks.


## Framework-Specific Nuances

### Standards Versions Cheat-Sheet

Always cite the version when mapping controls. Frameworks change
materially between editions.

| Framework | Current edition | Key change to know |
|-----------|-----------------|--------------------|
| **NIST CSF** | **2.0** (Feb 2024) | Added the **Govern** function (sixth alongside Identify/Protect/Detect/Respond/Recover); broadened scope beyond critical infrastructure |
| **NIST SP 800-53** | **Rev 5** (Sep 2020, with updates) | Privacy controls integrated; supply-chain (SR) family added |
| **NIST SP 800-37 (RMF)** | **Rev 2** (Dec 2018) | Prepare step added |
| **NIST SP 800-218 (SSDF)** | **v1.1** (Feb 2022) | Secure development lifecycle; cited by EO 14028 |
| **NIST AI RMF** | **1.0** (Jan 2023) + **GenAI Profile** (July 2024) | AI risk management |
| **ISO/IEC 27001** | **2022** | Annex A controls reduced from 114 to 93, reorganized into 4 themes; the 2013 edition retired Oct 2025 |
| **ISO/IEC 27002** | **2022** | Companion to 27001:2022 |
| **ISO/IEC 42001** | **2023** | AI management system |
| **PCI-DSS** | **v4.0.1** (June 2024) | Several future-dated requirements became mandatory 31 March 2025 (auth, scripts, log review, etc.) |
| **SOC 2 (TSC)** | **2017 TSC, points of focus revised 2022** | Continuous monitoring expectations strengthened |
| **HIPAA Security Rule** | 2013 (HITECH); **2024 NPRM** for modernization in flight | Watch for finalization; encryption likely to move from "addressable" to required |
| **GDPR** | Reg. (EU) 2016/679 (in force May 2018) | UK GDPR is the post-Brexit equivalent |
| **NIS2** | Dir. (EU) 2022/2555 | Transposition deadline Oct 2024; broadens scope vs original NIS |
| **EU AI Act** | Reg. (EU) 2024/1689 | In force Aug 2024; obligations phase in 2025–2027 |
| **EU CRA** | Reg. (EU) 2024/2847 | In force Dec 2024; main obligations apply Dec 2027 |
| **OWASP Top 10 (web)** | **2021** | Watch for next edition |
| **OWASP API Security Top 10** | **2023** | Successor to 2019 list |
| **OWASP Top 10 for LLM Apps** | **2025** | See `llm-and-ai-security.md` |
| **OWASP ASVS** | **5.0** (2024) | Verification standard |
| **CIS Controls** | **v8.1** (2024) | Implementation Groups (IG1/2/3) |

When you cite a framework in a control mapping, **include the version**.
"NIST CSF" without a version is ambiguous after Feb 2024.

---

### SOC2 (Service Organization Control 2)

**Purpose**: Trust assurance for service providers (SaaS, cloud)

**Trust Service Criteria** (2017 TSC, points of focus revised 2022):
- Security (always required)
- Availability (optional)
- Processing Integrity (optional)
- Confidentiality (optional)
- Privacy (optional)

**Key Requirements**:
- Annual audit (Type I = point-in-time, Type II = over period, typically 6-12 months)
- Control documentation (policies, procedures)
- Evidence of operation (logs, reports, test results)
- Continuous monitoring

**Common Gap**: SOC2 Type II requires evidence of controls operating over time (not just implemented). Need historical logs, incident reports, change records.


### PCI-DSS (Payment Card Industry Data Security Standard)

**Purpose**: Protect payment card data

**Current version**: **PCI-DSS v4.0.1** (June 2024). Several future-dated
requirements added in v4.0 became **mandatory 31 March 2025** — any
control mapping done after that date must treat them as in scope, not
"future":

- Authenticated internal vulnerability scans
- Multi-factor authentication for all access into the CDE (not just remote/admin)
- Detection and response for changes to payment-page scripts
- Targeted risk analyses replacing prescriptive frequencies for several requirements
- Phishing-resistant MFA expectations strengthened

**12 Requirements** (grouped into 6 control objectives):
1. Build and maintain secure network
2. Protect cardholder data
3. Maintain vulnerability management
4. Implement strong access control
5. Regularly monitor and test networks
6. Maintain information security policy

**Key Requirements**:
- Internal and external vulnerability scans on defined cadences (quarterly external by ASV; internal authenticated)
- Annual penetration testing (segmentation testing if segmentation is claimed)
- Cardholder data encryption (PAN never stored plainly)
- Strict access control (need-to-know basis)

**Common Gap**: Many developers store PAN (Primary Account Number) in logs or databases. PCI-DSS forbids this - use tokenization instead.


### HIPAA (Health Insurance Portability and Accountability Act) - US Healthcare

**Purpose**: Protect patient health information (PHI)

**Key Rules**:
- **Privacy Rule**: Patient rights to access/control their PHI
- **Security Rule**: Technical safeguards for electronic PHI (ePHI)
- **Breach Notification Rule**: Report breaches affecting 500+ individuals

**Key Requirements**:
- Encryption (not strictly required but effectively mandatory via "addressable" safeguards)
- Access controls (role-based, minimum necessary)
- Audit trails (track all ePHI access)
- Business Associate Agreements (BAAs with vendors)

**Common Gap**: Developers forget BAAs are required for ANY vendor processing ePHI (cloud providers, analytics tools, etc.).


### ISM (Information Security Manual) + IRAP - Australia Government

**Purpose**: Protect government information systems

**IRAP**: Infosec Registered Assessors Program (authorized assessors)

**Key Requirements**:
- ISM compliance (Essential Eight at minimum)
  1. Application control (whitelisting)
  2. Patch applications
  3. Configure Microsoft Office macros
  4. User application hardening
  5. Restrict administrative privileges
  6. Patch operating systems
  7. Multi-factor authentication
  8. Daily backups
- Classification handling (UNOFFICIAL, OFFICIAL, SECRET, TOP SECRET)
- IRAP assessment (required for government contracts)

**Common Gap**: Essential Eight is minimum, but full ISM compliance has hundreds of controls. Scope carefully based on classification level.


### GDPR (General Data Protection Regulation) - EU

**Purpose**: Protect personal data of EU residents

**Key Principles**:
- Lawful basis for processing (consent, contract, legal obligation, etc.)
- Data minimization (collect only necessary data)
- Right to access, rectify, erase (data subject rights)
- Data breach notification (72 hours to supervisory authority)

**Key Requirements**:
- Privacy by design and default
- Data Protection Impact Assessment (DPIA) for high-risk processing
- Data Processing Agreements (DPAs with processors)
- EU representative (if outside EU but processing EU data)

**Common Gap**: GDPR applies to ANY company processing EU resident data, regardless of company location. Many US companies underestimate scope.


### NIS2 Directive — EU Cybersecurity (Dir. 2022/2555)

**Purpose**: Replace and broaden NIS1; raise the cybersecurity baseline
across "essential" and "important" entities in the EU.

**Status**: In force; transposition deadline was 17 October 2024.
National laws in member states are now the operative compliance text.

**Scope**: Significantly broader than NIS1 — energy, transport, banking,
health, drinking water, digital infrastructure, ICT service management,
public administration, space, postal services, waste management,
chemicals, food, certain manufacturing, digital providers, research.

**Key Requirements**:
- Risk-management measures (incident handling, business continuity, supply-chain security, vulnerability handling)
- Incident reporting (early warning within 24h; full notification within 72h; final report within one month)
- Management-body accountability (named officers; potential personal liability)
- Supply-chain due diligence

**Common Gap**: NIS2 holds the **management body** personally
accountable. Boards and execs need named cybersecurity training and
documented oversight, not just delegation to a CISO.


### EU AI Act (Reg. (EU) 2024/1689)

**Purpose**: Risk-tiered regulation of AI systems sold or used in the EU.

**Status**: In force August 2024. Obligations phase in:
- Prohibitions and AI literacy: ~Feb 2025
- General-purpose AI (GPAI) model obligations: ~Aug 2025
- Most high-risk system obligations: ~Aug 2026
- Embedded high-risk (Annex II) systems: ~Aug 2027

**Risk Tiers**:
- **Prohibited**: Social scoring by public authorities, certain biometric categorization, etc.
- **High-risk** (Annex III): Critical infrastructure, education, employment, essential services, law enforcement, migration, justice. Require conformity assessment, risk management, data governance, transparency, human oversight, accuracy/robustness/cybersecurity, post-market monitoring, registration in EU database.
- **GPAI** (General-purpose AI models): Documentation, copyright, summaries of training data; additional obligations for "systemic risk" models.
- **Limited risk**: Transparency obligations (disclose AI interaction).
- **Minimal risk**: Voluntary codes of conduct.

**Common Gap**: Treating AI Act as "EU only" — like GDPR, it applies
extraterritorially when output is used in the EU.

**Cross-link**: `llm-and-ai-security.md` covers the security-architect
contribution to high-risk-system obligations (cybersecurity controls,
robustness, post-market monitoring).


### EU CRA — Cyber Resilience Act (Reg. (EU) 2024/2847)

**Purpose**: Harmonize cybersecurity requirements for "products with
digital elements" (PDEs) sold in the EU. Mandates secure-by-design,
vulnerability handling, and SBOM-equivalent inventories.

**Status**: In force December 2024; main obligations apply
~December 2027. Earlier obligations on importers/manufacturers begin
sooner; consult current European Commission guidance.

**Scope**: Most software and connected hardware sold in the EU
(open-source maintained by stewards has tailored treatment).

**Cross-link**: `supply-chain-security.md` for SBOM, vulnerability
handling, and signing controls aligned with CRA expectations.


### Post-Quantum Cryptography Note

NIST has finalized the first three post-quantum standards (Aug 2024):

- **FIPS 203 (ML-KEM)** — module-lattice key encapsulation (CRYSTALS-Kyber)
- **FIPS 204 (ML-DSA)** — module-lattice digital signature (CRYSTALS-Dilithium)
- **FIPS 205 (SLH-DSA)** — stateless hash-based signature (SPHINCS+)

**For 2026-era architecture reviews**: at minimum, flag PQ readiness as
a forward-looking concern. **Crypto-agility** — the ability to swap
algorithms without re-architecting — matters more than picking the
"right" PQ algorithm today. Identify long-lived secrets (data with
multi-decade confidentiality requirements) and treat "harvest now,
decrypt later" as a real threat for those.

US national-security systems are subject to **CNSA 2.0** (NSA, Sep 2022,
updated 2024) which sets a phased migration timeline to PQ algorithms
for classified-adjacent systems.


## Evidence Collection

**Auditors need evidence that controls are operating, not just documented.**

### Evidence Types

#### 1. Configuration Evidence
```bash
# Example: TLS configuration
openssl s_client -connect api.example.com:443 -tls1_3
# Save output showing TLS 1.3 enabled

# Example: IAM MFA enforcement
aws iam get-account-password-policy
# Save JSON showing MFA required
```

#### 2. Log Evidence
```bash
# Example: Authentication logs (last 7 days)
aws logs filter-log-events \
  --log-group-name /aws/lambda/auth \
  --start-time $(date -d '7 days ago' +%s)000 \
  --filter-pattern 'authentication'
# Save sample showing successful/failed logins logged
```

#### 3. Policy Documentation
```markdown
# Example: Access Control Policy
- All users must authenticate with MFA
- Role-based access (roles defined in roles.md)
- Session timeout: 30 minutes
- Annual access review by manager
```

#### 4. Test Results
```markdown
# Penetration Test Report (Annual)
- Date: {{YYYY-MM-DD — date of test}}
- Tester: Acme Security (SOC2 requirement)
- Findings: 2 medium, 5 low
- Remediation: All medium fixed within 30 days
- Evidence: pentest-report-{{YYYY}}.pdf
```

#### 5. Interview Records
```markdown
# Auditor Interview: DevOps Lead ({{YYYY-MM-DD}})
Q: How do you handle production changes?
A: Pull request → 2 approvals → CI/CD deploy → post-deploy verification

Q: How long are logs retained?
A: 90 days in CloudWatch, then archived to S3 for 7 years

Evidence: interview-notes-devops-{{YYYY-MM-DD}}.md
```


## Control Mapping Workflow

### Workflow: Preparing for Audit

```
1. Discovery (Week 1)
   └─→ Identify applicable frameworks (jurisdiction + industry + data type)
   └─→ Understand framework structure (categories, requirements, evidence)

2. Inventory (Week 2-3)
   └─→ Document existing technical controls
   └─→ Document operational controls (policies, procedures)
   └─→ Document evidence locations

3. Mapping (Week 4)
   └─→ Create traceability matrix (controls → requirements)
   └─→ Identify gaps (missing/insufficient controls)

4. Gap Analysis (Week 5)
   └─→ Prioritize gaps (critical/high/low)
   └─→ Estimate remediation effort and cost

5. Remediation (Week 6-12)
   └─→ Fix critical gaps (blockers)
   └─→ Fix high-priority gaps (reduce risk)
   └─→ Document all changes

6. Evidence Collection (Week 13-14)
   └─→ Gather configuration evidence
   └─→ Gather log samples
   └─→ Finalize policy documents
   └─→ Conduct test activities (if needed)

7. Audit (Week 15-16)
   └─→ Provide evidence to auditor
   └─→ Answer auditor questions
   └─→ Address any findings

8. Continuous Monitoring (Ongoing)
   └─→ Maintain controls
   └─→ Collect evidence continuously
   └─→ Annual re-assessment
```


## Quick Reference: Framework Selection

| If you have... | Consider these frameworks... | Priority |
|----------------|------------------------------|----------|
| **Australian government data** | ISM (current), IRAP, PSPF | Mandatory |
| **Australian private healthcare** | Privacy Act 1988 (as amended), Healthcare-specific | Mandatory |
| **US healthcare (HIPAA data)** | HIPAA, HITECH (and watch HHS NPRM updates) | Mandatory |
| **EU resident data** | GDPR | Mandatory |
| **Payment card data** | PCI-DSS v4.0.1 | Mandatory |
| **US government contracts** | FedRAMP (Rev 5), FISMA, NIST 800-53 Rev 5, EO 14028 | Mandatory |
| **EU "essential / important" entities** | NIS2 (national transposition) | Mandatory |
| **EU-marketed connected products** | EU CRA | Mandatory (phased) |
| **AI systems sold in EU** | EU AI Act (risk tier dependent) | Mandatory |
| **B2B SaaS (any jurisdiction)** | SOC 2 | High priority |
| **Enterprise software** | ISO/IEC 27001:2022 | Medium priority |
| **AI management system** | ISO/IEC 42001:2023 | Medium priority |
| **UK government** | Cyber Essentials / Plus, NCSC, UK GDPR/DPA 2018 | Mandatory |

**Always verify**: This table is a starting point, not definitive. Consult legal/compliance experts for your specific situation.


## Common Mistakes

### ❌ Assuming Frameworks Without Asking

**Wrong**: "We need SOC2" (without checking customer requirements)

**Right**: "What do our customers/contracts require? What jurisdiction are we in?"

**Why**: You might need multiple frameworks, or different ones than assumed.


### ❌ Memorizing Framework Details

**Wrong**: Try to remember all SOC2 criteria, all PCI-DSS requirements

**Right**: Learn discovery process, reference frameworks as needed

**Why**: Frameworks update (e.g., PCI-DSS v4.0.1 effective 2024 with mandatory requirements landing March 2025; NIST CSF 2.0 added the Govern function in Feb 2024; ISO 27001:2022 superseded the 2013 edition with retirement in Oct 2025). Process is stable, details change — always cite the version.


### ❌ Mapping Without Inventory

**Wrong**: Read framework, try to build controls from scratch to match

**Right**: Inventory existing controls first, then map to framework

**Why**: Easier to map existing controls than build from requirements. Avoids duplicate implementations.


### ❌ No Gap Prioritization

**Wrong**: List all gaps, start working on first one

**Right**: Prioritize by impact (critical = blocks audit, high = findings, low = post-audit)

**Why**: Time/budget limited. Fix blockers first, optimize later.


### ❌ Treating Compliance as One-Time

**Wrong**: Pass audit, stop maintaining controls

**Right**: Continuous monitoring, annual re-assessment, maintain evidence

**Why**: Most audits are annual or more frequent. Controls must operate continuously.


## Cross-References

**Use WITH this skill**:
- `ordis/security-architect/security-controls-design` - Implement controls identified in gap analysis
- `muna/technical-writer/clarity-and-style` - Write clear policy documentation

**Use AFTER this skill**:
- `ordis/security-architect/security-authorization-and-accreditation` - If government/defense (ATO/SSP/SAR)
- `muna/technical-writer/operational-acceptance-documentation` - Document audit package

## Real-World Impact

**Projects using systematic compliance mapping**:
- **Healthcare SaaS (Australia)**: Discovered 4 applicable frameworks in discovery (not just Privacy Act). Avoided surprise compliance requirements pre-launch.
- **SOC2 Type II (US)**: Gap analysis found 12 missing controls, prioritized 3 as critical. Remediated in 6 weeks, passed audit on first attempt (vs industry avg 2-3 attempts).
- **IRAP Assessment (Australia Gov)**: Traceability matrix with 200+ ISM controls mapped to 47 technical implementations. Assessor praised "clearest control mapping in 5 years".

**Key lesson**: **Discovery process (ask jurisdiction/industry/data type) finds ALL applicable frameworks. Control mapping with traceability prevents gaps and audit failures.**
