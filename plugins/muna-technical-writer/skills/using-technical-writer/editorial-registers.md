
# Editorial Registers

## Overview

A **register** is the set of conventions — tone, authority markers, structural expectations, vocabulary, and hedging patterns — that a document follows based on its institutional context and purpose. Register is distinct from audience:

- **Audience** (who-receives): Determines *what information* to include
- **Register** (how-text-operates): Determines *how to express it*

A document has both. A technical architecture review written for a CTO uses the **technical register** with an **executive audience**. A public FAQ about a government program uses the **public-facing register** with a **general audience**.

**Language scope**: The six built-in registers are defined with English-language conventions, authority markers, and examples. Non-English documents are out of scope for the built-in registers. Users working in other languages can define custom registers using the extension point below.

## Built-In Registers

This reference defines six registers. Legal register is explicitly excluded — it requires domain-specific expertise around liability and jurisdiction that is out of scope. The custom register extension point allows users to define their own.

---

### 1. Technical

#### When to Use
Internal engineering documentation, architecture docs, API references, design documents, code comments, technical specifications. Documents where the reader has domain knowledge and needs precision.

#### Voice & Tone
- Direct and precise — no hedging, no softening
- Third person or imperative ("the service returns", "call the endpoint")
- Assumes shared vocabulary — domain terms used without definition
- Terse over verbose — say it once, clearly

#### Authority Markers
- Definitive statements: "X does Y", "X requires Y"
- RFC 2119 when specifying behavior: "MUST", "SHOULD", "MAY"
- Version-pinned references: "as of v2.3.1"
- Code as evidence: inline code, code blocks, command examples

#### Structural Expectations
- Headings map to components or concepts
- Code examples for every non-trivial claim
- Tables for configuration options, parameters, error codes
- Links to source code, API docs, related specs
- Minimal prose between structured elements

#### Audience Assumptions
- Reader has domain expertise (understands the tech stack)
- Reader wants HOW — implementation details, not business justification
- Reader will copy-paste commands and code examples
- Reader uses docs as reference, not tutorial

#### Common Mistakes
- Hedging: "This might help with performance" — state the measured result instead
- Business framing: "This creates value by..." — describe the implementation instead
- Undefined acronyms for cross-team docs — define on first use even in technical register
- Prose where a table or code block would be clearer

#### Example: Characteristic Paragraph

> The `AuthMiddleware` validates JWT tokens on every request. It extracts the `Authorization` header, verifies the RS256 signature against the public key at `/keys`, checks the `exp` claim, and compares `scope` claims against the endpoint's required permissions. Requests with invalid or expired tokens receive a 401 response. Requests with valid tokens but insufficient scopes receive 403.

#### Example: Characteristic Structure

```
## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `JWT_PUBLIC_KEY_PATH` | string | required | Path to PEM-encoded public key |
| `TOKEN_EXPIRY_SECONDS` | int | 3600 | Token lifetime in seconds |
| `CLOCK_SKEW_SECONDS` | int | 30 | Tolerance for clock drift |

## Error Codes

| Code | Meaning | Resolution |
|------|---------|------------|
| 401 | Invalid or expired token | Re-authenticate |
| 403 | Insufficient scopes | Request elevated permissions |
```

#### Calibration Examples
- **IS** Technical register: "The service processes requests asynchronously using a work-stealing thread pool with 4 workers."
- **Is NOT** Technical register (it is Public-facing): "Our service handles your requests quickly and efficiently in the background."
- **Is NOT** Technical register (it is Executive): "The new processing architecture reduces response times by 60%, improving customer satisfaction scores."

---

### 2. Policy

#### When to Use
Organizational standards, compliance frameworks, internal procedures, data governance policies, acceptable use policies, security policies. Documents that define rules and their enforcement.

#### Voice & Tone
- Normative and authoritative — this is a rule, not a suggestion
- Third person, role-based: "the data owner shall...", "authorized users may..."
- Formal but not ornate — clear obligation, not bureaucratic padding
- No hedging — obligations are stated, not suggested

#### Authority Markers
- RFC 2119 modal verbs with precise meaning:
  - "shall" / "must" — mandatory requirement
  - "should" — recommended but not mandatory
  - "may" — permitted, optional
  - "shall not" / "must not" — prohibited
- Numbered requirements for traceability (REQ-001, POL-3.2)
- Defined terms capitalized: "Authorized User", "Covered Data", "Data Owner"
- Effective dates and review cycles

#### Structural Expectations
- Numbered sections and subsections (1.1, 1.2, 2.1)
- Scope and applicability section upfront
- Definitions section for capitalized terms
- Roles and responsibilities matrix
- Compliance and enforcement section
- Version history / revision log

#### Audience Assumptions
- Reader needs to know what is required of them
- Reader may need to audit compliance against this document
- Reader expects unambiguous obligations — no room for interpretation
- Reader will reference specific numbered sections

#### Common Mistakes
- Casual language: "Try to encrypt data when possible" — use "shall encrypt" with specific standards
- Undefined terms: Using "sensitive data" without defining scope — define "Covered Data" in Definitions section
- Missing enforcement: What happens when someone violates the policy?
- Mixing recommendation with requirement: "should" used where "shall" is intended

#### Example: Characteristic Paragraph

> All Authorized Users shall complete security awareness training within 30 calendar days of their access provisioning date. The Information Security Office shall maintain training completion records for a minimum of three (3) years. Users who do not complete training within the required timeframe shall have their access privileges suspended until training is completed.

#### Example: Characteristic Structure

```
## 3. Data Classification

### 3.1 Scope
This policy applies to all data processed, stored, or transmitted
by Company systems, including data held by third-party processors.

### 3.2 Classification Levels
| Level | Definition | Examples |
|-------|-----------|----------|
| Restricted | Data whose unauthorized disclosure would cause severe harm | PII, credentials, financial records |
| Internal | Data intended for internal use only | Architecture docs, roadmaps |
| Public | Data approved for external release | Marketing materials, published APIs |

### 3.3 Requirements
- REQ-3.3.1: Data Owners shall classify all data assets within 90 days of creation.
- REQ-3.3.2: Restricted data shall be encrypted at rest and in transit.
- REQ-3.3.3: Classification labels must be applied to all documents and repositories.
```

#### Calibration Examples
- **IS** Policy register: "Data Owners shall review access permissions quarterly and revoke access for personnel who no longer require it."
- **Is NOT** Policy register (it is Technical): "Use the `revoke-access` CLI command to remove user permissions from the IAM service."
- **Is NOT** Policy register (it is Government): "The Department will ensure that all access reviews are conducted in accordance with Section 5.2 of the Federal Information Security Modernization Act."

---

### 3. Government/Regulatory

#### When to Use
Regulatory communications, agency reports, Federal Register notices, compliance guidance for regulated entities, public accountability documents, government program documentation. Documents issued by or for government institutions communicating to external stakeholders.

#### Voice & Tone
- Institutional voice — the organization speaks, not an individual
- Formal but accessible — plain language mandates apply (e.g., Plain Writing Act)
- Authoritative without being adversarial — the institution has statutory authority
- Measured and deliberate — every word carries weight

#### Authority Markers
- Office-based accountability: "the Department will...", "the Agency is responsible for..."
- Statutory references: "pursuant to 44 U.S.C. § 3551", "in accordance with Executive Order 14028"
- Regulatory language: "is required to", "will", "is responsible for" (not RFC 2119 "shall")
- Effective dates tied to regulatory calendars: "effective 90 days after publication in the Federal Register"
- Docket and citation numbers

#### Structural Expectations
- Executive summary / purpose statement upfront
- Legal authority and statutory basis
- Plain language summaries alongside regulatory text
- Public comment / feedback mechanisms
- Contact information for responsible office
- Appendices for detailed technical content

#### Audience Assumptions
- Reader may be a member of the public with no technical background
- Reader may be a regulated entity needing to understand obligations
- Reader may be an oversight body evaluating compliance
- Reader expects institutional accountability — who is responsible

#### Common Mistakes
- Jargon without plain language alternative: "Implement FISMA controls" — explain in plain language first
- Individual voice: "I recommend..." — use "The Department recommends..."
- Using RFC 2119 "shall/must" instead of regulatory "is required to/will"
- Missing statutory basis — government documents need legal grounding

#### Disambiguation: Government vs Policy

| Dimension | Policy | Government/Regulatory |
|-----------|--------|----------------------|
| **Purpose** | Define organizational rules | Communicate institutional decisions externally |
| **Authority source** | Organizational mandate | Statutory/regulatory authority |
| **Language mandate** | No specific requirement | Plain language mandates apply |
| **Audience** | Internal compliance staff | Public, regulated entities, oversight bodies |
| **Normative verbs** | "shall", "must", "may" | "is required to", "will", "is responsible for" |
| **Accountability** | Role-based ("the data owner shall...") | Office-based ("the Department will...") |

**When a document is both**: The audience and accountability structure determine the register. Internal policy for a regulator uses the policy register. External communication to regulated entities uses the government register. When genuinely ambiguous, the distinguishing question is: "Who is the primary reader — internal compliance staff or external stakeholders?"

#### Example: Characteristic Paragraph

> The Cybersecurity and Infrastructure Security Agency (CISA) is issuing this Binding Operational Directive to ensure that federal agencies take immediate action to remediate known exploited vulnerabilities. Agencies are required to review the catalog of known exploited vulnerabilities maintained at cisa.gov/known-exploited-vulnerabilities and apply available patches or mitigations within the timeframes specified. The Department of Homeland Security will provide technical assistance to agencies that identify resource constraints in meeting these requirements.

#### Example: Characteristic Structure

```
## Purpose

This directive establishes requirements for federal civilian executive
branch agencies to remediate known exploited vulnerabilities.

## Authority

This directive is issued pursuant to 44 U.S.C. § 3553(b)(2), which
authorizes the Secretary of Homeland Security to issue binding
operational directives.

## Required Actions

1. Within 14 calendar days of a vulnerability being added to the
   catalog, agencies are required to apply available patches.
2. Where patches are not available, agencies will implement published
   mitigations or discontinue use of the affected product.

## Reporting

Agencies will report compliance status to CISA using the CyberScope
reporting tool within 72 hours of the remediation deadline.

## Contact

For questions regarding this directive, contact:
Cybersecurity Division, CISA
cyberdirectives@cisa.dhs.gov
```

#### Calibration Examples
- **IS** Government register: "The Agency will publish updated guidance within 60 days of this directive's effective date, in accordance with Executive Order 14028."
- **Is NOT** Government register (it is Policy): "Authorized Users shall complete remediation of critical vulnerabilities within 14 calendar days of notification."
- **Is NOT** Government register (it is Public-facing): "We're working to make sure your data stays safe by fixing security issues as quickly as possible."

---

### 4. Public-Facing

#### When to Use
User-facing documentation, help centers, product announcements, public FAQs, onboarding guides, customer communications, open-source project docs aimed at non-technical users. Documents where the reader may have no domain expertise.

#### Voice & Tone
- Warm and approachable — conversational without being casual
- Second person: "you can...", "your account..."
- Active voice, short sentences — scannable
- Confident and reassuring — builds trust
- No assumed expertise — explain everything or link to an explanation

#### Authority Markers
- Plain language: "you need to" not "you shall" or "you must"
- Action-oriented: "To do X, follow these steps"
- Benefit-first framing: "This keeps your data safe" before "Enable 2FA"
- Visual cues: icons, callout boxes, highlighted tips
- Human tone: "If something goes wrong, we're here to help"

#### Structural Expectations
- Short paragraphs (2-3 sentences max)
- Bullet points and numbered lists for any sequence
- Callout boxes for warnings, tips, prerequisites
- Screenshots or diagrams for visual learners
- "Next steps" or "Related articles" at the end
- Search-optimized headings (what users would type)

#### Audience Assumptions
- Reader may have no technical background at all
- Reader wants to accomplish a specific task, not understand the system
- Reader will scan, not read — make key information visually prominent
- Reader will give up if the first paragraph doesn't help

#### Common Mistakes
- Technical jargon: "Configure the SMTP relay" — say "Set up email forwarding"
- Passive voice: "Your password can be reset by..." — say "To reset your password, click..."
- Wall of text — break into bullets, add headings
- Assuming prior knowledge: "As you know..." — the reader may NOT know
- Normative language: "Users shall..." — say "You'll need to..."

#### Example: Characteristic Paragraph

> You can turn on two-factor authentication to add an extra layer of security to your account. When it's enabled, you'll enter a short code from your phone each time you log in. This means that even if someone learns your password, they still can't access your account without your phone.

#### Example: Characteristic Structure

```
## How to Reset Your Password

If you've forgotten your password, you can reset it in a few steps.

### Before You Start
- You'll need access to the email address on your account
- The reset link expires after 24 hours

### Steps

1. Go to the **login page** and click "Forgot password?"
2. Enter your email address and click **Send reset link**
3. Check your email for a message from us (check spam if you
   don't see it within 5 minutes)
4. Click the link in the email
5. Choose a new password and click **Save**

> Tip: Your new password must be at least 12 characters.
> Try a phrase like "correct horse battery staple" — it's easier
> to remember than random characters.

### Still Having Trouble?

If you didn't receive the email or the link has expired,
contact our support team.
```

#### Calibration Examples
- **IS** Public-facing register: "You can connect your calendar in just a few clicks — here's how to get started."
- **Is NOT** Public-facing (it is Technical): "The CalDAV integration endpoint accepts PROPFIND requests with a Depth: 1 header to enumerate calendar resources."
- **Is NOT** Public-facing (it is Policy): "Users shall integrate their calendar applications using approved enterprise connectors only."

---

### 5. Executive/Business

#### When to Use
Executive summaries, board presentations, business cases, investment proposals, quarterly reviews, risk assessments for leadership, strategic planning documents. Documents where the reader makes decisions based on business impact, not technical detail.

#### Voice & Tone
- Decisive and outcome-focused — lead with the conclusion
- Confident without hedging — "This will reduce costs by 30%" not "This might help reduce costs"
- Numbers-first: revenue, cost, risk, timeline
- No technical detail — abstract to business impact
- Concise — every sentence earns its place

#### Authority Markers
- Quantified impact: "$2M annual savings", "30% reduction in incidents"
- Comparative framing: "before/after", "current state/future state"
- Timeline commitments: "delivered by Q3 2026"
- Risk quantification: "probability x impact" framing
- ROI calculations and payback periods

#### Structural Expectations
- Executive summary (3-5 bullet points) at the top
- Recommendation stated in the first paragraph
- Business impact before technical approach
- Tables for cost/benefit analysis
- Risk matrix (impact x likelihood)
- Clear ask: what decision is needed, by when

#### Audience Assumptions
- Reader has 5 minutes, maybe 2
- Reader cares about: revenue, cost, risk, timeline, competitive position
- Reader does NOT care about: implementation details, architecture, code
- Reader will make a decision based on this document

#### Common Mistakes
- Leading with technical details: "We use Kafka for event streaming..." — lead with business impact instead
- Hedging: "We think this could potentially improve..." — state the measured or projected result
- No clear ask: What decision do you need from the reader?
- Too long — if it's over 2 pages, it's not executive-level

#### Example: Characteristic Paragraph

> Migrating to the new authentication system eliminates our largest security liability and reduces annual infrastructure costs by $240K. The migration requires 4 engineering-weeks and carries low risk — we can run both systems in parallel during the transition. We recommend approving the migration for Q2, with full cutover by June 30.

#### Example: Characteristic Structure

```
## Recommendation

Approve the authentication migration for Q2 2026.

## Business Impact

| Metric | Current | After Migration |
|--------|---------|----------------|
| Annual infrastructure cost | $480K | $240K |
| Mean time to breach detection | 72 hours | 4 hours |
| Customer-facing auth failures | 12/month | <1/month |

## Investment Required

- Engineering: 4 weeks (2 engineers x 2 weeks)
- Cost: $40K fully loaded
- Payback period: 2 months

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Migration delays | Medium | Low | Parallel run, no hard cutover |
| Integration issues | Low | Medium | Staged rollout, automated testing |

## Timeline

- April: Development and testing
- May: Parallel run with monitoring
- June 30: Full cutover, decommission legacy system
```

#### Calibration Examples
- **IS** Executive register: "This investment pays for itself in 60 days and eliminates our top audit finding."
- **Is NOT** Executive register (it is Technical): "The service uses HMAC-SHA256 for request signing with a 15-minute nonce window."
- **Is NOT** Executive register (it is Academic): "Our analysis suggests a statistically significant correlation (p < 0.05) between authentication latency and user churn, though further longitudinal study is warranted."

---

### 6. Academic/Research

#### When to Use
Research papers, technical reports, literature reviews, experimental results, white papers, grant proposals, methodology descriptions. Documents where claims must be grounded in evidence and prior work.

#### Voice & Tone
- Precise and measured — claims are hedged appropriately
- Third person, passive often appropriate: "the results indicate...", "it was observed that..."
- Cautious with causation — "correlated with" not "caused by" unless proven
- Acknowledges limitations and alternative explanations
- Formal without being stiff — clarity still matters

#### Authority Markers
- Citations: (Author, Year) or [1] numbered references
- Hedged claims: "suggests", "indicates", "is consistent with", "appears to"
- Statistical evidence: p-values, confidence intervals, effect sizes
- Methodology disclosure: enough detail to reproduce
- Explicit scope limitations: "within the constraints of this study..."

#### Structural Expectations
- Abstract / summary
- Introduction with research question or hypothesis
- Related work / literature review
- Methodology
- Results (data first, interpretation second)
- Discussion (implications, limitations, future work)
- References / bibliography

#### Audience Assumptions
- Reader is a domain expert or informed practitioner
- Reader will evaluate the strength of your evidence
- Reader expects to see prior work acknowledged
- Reader will check your methodology before trusting your conclusions

#### Common Mistakes
- Overclaiming: "This proves that..." — say "These results suggest that..."
- Missing limitations: Every study has them — state them explicitly
- No citations: Claims without references appear ungrounded
- Decisive language where hedging is appropriate: "X causes Y" without controlled experiment
- Missing methodology: Reader can't evaluate what they can't reproduce

#### Example: Characteristic Paragraph

> Our results indicate that the proposed caching strategy reduces median query latency by 42% (95% CI: 38-46%) compared to the baseline, consistent with theoretical predictions by Chen et al. (2024). However, we note that tail latency (p99) showed no statistically significant improvement (p = 0.23), suggesting that the caching strategy primarily benefits common-path queries. Further investigation with a larger sample size and diverse workload patterns is warranted.

#### Example: Characteristic Structure

```
## Abstract

We present a caching strategy for distributed query engines that
reduces median latency by 42% on standard benchmarks. [...]

## 1. Introduction

Query latency remains a primary bottleneck in distributed analytics
systems (Garcia et al., 2023). Prior approaches have focused on...

## 2. Related Work

Chen et al. (2024) demonstrated that predictive caching can reduce
latency by up to 35% in single-node configurations. Our work extends
this approach to distributed settings. [...]

## 3. Methodology

### 3.1 Experimental Setup
We evaluated our approach on a 16-node cluster running [...]
Benchmarks were conducted using the TPC-DS suite at scale factor 100.

### 3.2 Metrics
- Primary: median query latency (ms)
- Secondary: p99 latency, cache hit rate, memory overhead

## 4. Results
[Tables, figures, statistical analysis]

## 5. Discussion
[Implications, limitations, future work]

## References
[1] Chen, L. et al. (2024). "Predictive Caching for Analytical Queries."
    Proc. VLDB, 17(4), 892-905.
```

#### Calibration Examples
- **IS** Academic register: "These findings are consistent with prior observations (Martinez, 2023), though the effect size in our sample (d = 0.4) was smaller than previously reported."
- **Is NOT** Academic register (it is Executive): "The new caching system cuts query times in half, saving $500K annually in compute costs."
- **Is NOT** Academic register (it is Technical): "Set `CACHE_TTL=300` and `CACHE_MAX_SIZE=10GB` in the config file to enable the query cache."

---

## Register-to-Register Relationships

### Affinity Pairs

These registers share conventions. Translation between them is light editing.

| Pair | Shared Conventions | Key Differences |
|------|-------------------|-----------------|
| Technical and Academic | Precision, domain vocabulary | Academic adds hedging, citations, methodology framing |
| Policy and Government | Formality, structured authority | Government adds plain language mandates, external accountability |
| Public-facing and Executive | Accessibility, no jargon | Executive adds outcome/ROI framing, assumes decision authority |

### Tension Pairs

These registers' conventions conflict. Translation between them is substantial rewriting.

| Pair | Core Tension |
|------|-------------|
| Technical and Public-facing | Domain knowledge vs accessibility — vocabulary, assumed context, and detail level all change |
| Policy and Public-facing | Normative authority vs approachable tone — "shall" becomes "you need to" |
| Academic and Executive | Hedged nuance vs decisive brevity — everything that makes academic writing careful makes executive writing weak |

### Mixed-Register Documents

Some documents legitimately use multiple registers. The agent distinguishes between:

- **Intentional mixing**: Register shifts at section boundaries, appropriate for each section's audience. Example: a government report (government register) with technical appendices (technical register) and a public-facing executive summary (public-facing register). This is valid structure.
- **Register drift**: The writer lost track of their audience — informal asides in policy text, jargon creeping into public-facing sections, hedging in executive summaries. This is a defect.

---

## Custom Register Extension

The register field schema is uniform. Users can define custom registers using this template. The agent treats custom registers identically to built-in ones — same detection heuristics, same review dimensions, same translation mechanics. Custom registers can be provided inline in the user's prompt or referenced from a file. The agent does not persist custom registers between sessions.

```markdown
### Custom Register: [Name]

#### When to Use
[Document types and contexts]

#### Voice & Tone
[Authority level, formality, hedging]

#### Authority Markers
[Key vocabulary and phrases]

#### Structural Expectations
[What readers expect to see]

#### Audience Assumptions
[What the reader knows and needs]

#### Common Mistakes
[What breaks this register]

#### Example: Characteristic Paragraph
> [Sample text]

#### Example: Characteristic Structure
[Sample headings/callouts/citations]

#### Calibration Examples
- **IS** this register: "[example sentence]"
- **Is NOT** this register: "[example sentence — it is [other register] because [reason]]"
```
