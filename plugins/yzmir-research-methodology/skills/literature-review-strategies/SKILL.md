---
name: literature-review-strategies
description: Use when starting research to survey a field, writing literature review sections, or need systematic search methodology - prevents scope creep, cherry-picking, stopping too early, and unsystematic searching using PRISMA-based methodology
---

# Literature Review Strategies

## Overview

Systematic literature review methodology prevents bias, scope creep, and incomplete coverage. **Literature reviews are hypothesis-neutral investigations** - you search to discover what the field knows, not to support a predetermined position.

**Core Principle**: Systematic methodology protects YOU from bias (scope creep, cherry-picking, satisficing), not just readers. PRISMA-based rigor applies regardless of output type - proposals deserve the same rigor as dissertations.

## When to Use

Use this skill when:
- Starting research on a new topic
- Writing literature review sections for papers/proposals
- Surveying a field to identify gaps
- Establishing context for your research
- Systematic review or meta-analysis

**Don't use for**: Finding one specific paper, quick background reading, reading a single paper deeply

---

## Critical Red Flags - STOP Before Shortcutting

### 🚨 "Good Enough for [Output Type]"

**If you're tempted to think**:
- "This is just a proposal, not a dissertation"
- "Conference paper doesn't need full systematic review"
- "Internal report can be less rigorous"
- "20-30 papers is enough for this purpose"

**STOP. Do NOT adjust methodology based on output type.**

**Why**: Systematic methodology prevents YOUR bias during search, regardless of who reads the output. A biased literature review in a proposal leads to biased research design.

**Reality**: PhD proposal = same rigor as dissertation. Your future research depends on accurate understanding of the field NOW.

---

### 🚨 Arbitrary Stopping Rules

**If you're about to**:
- Stop after finding N papers (10, 20, 50)
- Only search last N years (5 years, 10 years)
- Only read "top 10 most cited"
- "I found enough to write the section"

**STOP. Do NOT use arbitrary stopping criteria.**

**Why**: Arbitrary limits =  satisficing = missing key papers.

**Instead**: Stop when you reach **saturation** (multiple searches yield no new relevant papers).

---

### 🚨 PRISMA as "Overkill" or "Just for Journals"

**If you're thinking**:
- "PRISMA is only required for journal submission"
- "I'll use the principles but skip the documentation"
- "80% value in 5% of the time"
- "Adapt the process to my context"

**STOP. Do NOT separate PRISMA principles from PRISMA process.**

**Why**: The PROCESS (screening, documentation, flow charts) IS how you verify you followed the principles. Documentation catches when you deviate.

**Reality**: PRISMA methodology prevents cherry-picking. The flowchart isn't bureaucracy - it's tracking to ensure you didn't introduce bias.

---

### 🚨 Cherry-Picking for Hypothesis Support

**If you're searching**:
- "Papers supporting X hypothesis"
- "Evidence that Y approach works"
- "Examples of Z being successful"
- "Find papers showing..."

**STOP. Hypothesis-biased search = confirmation bias.**

**Why**: Literature reviews are hypothesis-NEUTRAL. You search to discover what's known, not prove what you believe.

**Instead**: Define a QUESTION (not hypothesis), use neutral search terms, include contradictory evidence.

---

## Systematic Literature Review Process

Follow this sequence - no skipping steps:

### Step 1: Define Specific Research Question

**Before searching anything, answer**:
- What SPECIFIC aspect am I investigating? (not "AI fairness" - what about fairness?)
- What population/problem/domain? (PICO framework)
- What types of studies are relevant? (empirical? theoretical? surveys?)

**Use PICO (for empirical) or FINER (for any research)**:

**PICO**:
- **P**opulation: Who/what is being studied?
- **I**ntervention: What is being applied/tested?
- **C**omparison: Compared to what?
- **O**utcome: What is being measured?

**FINER**:
- **F**easible: Can you actually review this scope?
- **I**nteresting: Meaningful research question?
- **N**ovel: Adds something to knowledge?
- **E**thical: No problematic implications?
- **R**elevant: Matters to your field?

**Example**:
- ❌ Too broad: "AI fairness"
- ✅ Specific: "What fairness metrics are used for evaluating hiring algorithms, and how do they compare?"

---

### Step 2: Identify Databases (Plural)

**DO NOT rely on Google Scholar alone.**

Google Scholar is useful for:
- Backward citation tracking (finding references)
- Forward citation tracking (finding citing papers)
- Quick exploratory search

**NOT sufficient for primary systematic search because**:
- Coverage gaps in specific fields
- No advanced boolean search in many cases
- Ranking algorithm may hide relevant papers

**Required**: Search multiple databases appropriate to your field:

| Field | Primary Databases |
|-------|-------------------|
| Computer Science | IEEE Xplore, ACM Digital Library, arXiv, Scopus |
| Psychology | PsycINFO, PubMed, Web of Science |
| Medicine/Health | PubMed, MEDLINE, Cochrane Library |
| Engineering | IEEE, Scopus, Web of Science, Compendex |
| Social Sciences | JSTOR, Web of Science, Sociological Abstracts |
| Interdisciplinary | Scopus, Web of Science, Google Scholar (supplementary) |

**Document**: Which databases you searched and why (coverage for your topic).

---

### Step 3: Develop Search Strings

**Create boolean search strings**:

**Components**:
- Core concept terms (AND between different concepts)
- Synonym variants (OR between synonyms)
- Exclusions if needed (NOT)

**Example** (hiring fairness):
```
("algorithmic fairness" OR "AI fairness" OR "ML bias")
AND
("hiring" OR "recruitment" OR "employment" OR "candidate selection")
AND
("metric*" OR "measure*" OR "evaluation" OR "assessment")
```

**Document for EACH database**:
- Exact search string used
- Date range (if limited - justify why)
- Any filters applied (peer-reviewed, language, etc.)
- Date you ran the search
- Number of results

**Why**: Enables replication and catches if you biased your search terms.

---

### Step 4: Screen and Select (PRISMA Methodology)

**DO NOT just read the first 20 papers.**

Follow PRISMA screening:

**4a. Initial Screening (Title/Abstract)**:
- Apply inclusion/exclusion criteria BEFORE reading
- Document criteria (what makes a paper relevant?)
- Screen ALL results from search, not subset

**Inclusion criteria example**:
- ✅ Empirical evaluation of fairness metrics
- ✅ Hiring/recruitment domain
- ✅ Published in peer-reviewed venue
- ✅ English language

**Exclusion criteria example**:
- ❌ Opinion pieces without data
- ❌ Non-hiring domains
- ❌ No fairness metric evaluation

**4b. Full-Text Screening**:
- Read full papers that passed title/abstract screening
- Apply more detailed criteria
- Track reasons for exclusion

**4c. Create PRISMA Flow Diagram**:

```
Records identified through database searching (n=450)
    ↓
Remove duplicates (n=120)
    ↓
Titles/abstracts screened (n=330)
    ↓ Excluded (n=280): [reasons]
Full-text articles assessed (n=50)
    ↓ Excluded (n=25): [reasons]
Studies included in synthesis (n=25)
```

**Why flowchart matters**: Forces you to account for EVERY paper. Catches when you skip papers that contradict your expectations.

---

### Step 5: Backward and Forward Citation Tracking (Snowballing)

After identifying core papers from database search:

**Backward tracking** (check references):
- For each included paper, scan reference list
- Identify relevant papers you missed in database search
- Apply same screening criteria
- Often finds 30-50% additional papers

**Forward tracking** (who cites these papers):
- Use Google Scholar "Cited by" feature
- Use Web of Science citation tracking
- Identify newer papers building on core work
- Apply same screening criteria

**Stop when**: Saturation reached (multiple rounds of snowballing yield no new relevant papers)

**NOT when**: You hit an arbitrary count or get tired

---

### Step 6: Synthesize by Themes, Not Papers

**DON'T**:
- Write one section per paper summarizing each
- "Paper A found X. Paper B found Y. Paper C found Z."
- Chronological paper-by-paper review

**DO**:
- Identify THEMES across papers
- Compare and contrast findings
- Highlight consensus and contradictions
- Identify gaps

**Synthesis frameworks**:

**Thematic Synthesis**:
- What are the recurring themes? (e.g., "metric tradeoffs", "context-dependency")
- Which papers address each theme?
- What's the consensus? Where do they disagree?

**Methodological Synthesis**:
- What methods do papers use? (simulation, field study, theoretical)
- How do findings differ by methodology?
- Are some methods more reliable?

**Chronological Synthesis** (rarely best):
- How has understanding evolved over time?
- Only use if temporal evolution is your research question

**Gap Identification**:
- What hasn't been studied?
- What populations/domains are missing?
- What methods haven't been tried?
- Where do contradictions need resolution?

---

## Stopping Criteria (When Is Review Complete?)

**Stop systematic search when**:
1. ✅ **Saturation**: Multiple database searches + snowballing yield no new relevant papers
2. ✅ **Coverage**: You've searched all major databases in your field
3. ✅ **Themes identified**: You can articulate the main themes and gaps

**DO NOT stop when**:
- ❌ You hit arbitrary paper count
- ❌ You have "enough to write the section"
- ❌ Deadline approaching (adjust scope if needed, don't cut corners)
- ❌ Papers start repeating findings (that's saturation SIGNAL, not stop condition - verify with broader search)

---

## Common Mistakes

| Mistake | Why It Happens | How to Prevent |
|---------|----------------|----------------|
| **Scope too broad** | Undefined research question | Use PICO/FINER before searching |
| **Only Google Scholar** | Convenience | Search field-specific databases |
| **Arbitrary stopping** (20 papers) | Satisficing, time pressure | Search until saturation |
| **Cherry-picking** | Confirmation bias | Hypothesis-neutral search terms |
| **No PRISMA flowchart** | "Not required for my venue" | PRISMA prevents YOUR bias, not just for readers |
| **Date filters** (5 years only) | Recency bias | Include foundational papers, justify any date limits |
| **Paper-by-paper review** | Easier than synthesis | Force thematic organization |
| **Stopped at database search** | Didn't know about snowballing | Always do backward/forward citation tracking |
| **Single reviewer** | No resources | At least document YOUR process for transparency |

---

## Rationalization Table - Don't Do These

| Excuse | Reality | What To Do |
|--------|---------|------------|
| "This is just a proposal, not dissertation" | Proposals need same rigor - bad lit review = bad research design | Use full systematic methodology |
| "20-30 papers is enough" | Arbitrary count = satisficing = missing key work | Search until saturation |
| "Last 5 years is sufficient" | Recency bias, misses foundational papers | Justify ANY date filter |
| "PRISMA is only for journal submissions" | PRISMA prevents YOUR bias, not just reader perception | Use PRISMA methodology always |
| "Use principles but skip documentation" | Documentation IS how you verify you followed principles | Document search strategy and screening |
| "Google Scholar is comprehensive" | Scholar has coverage gaps | Search multiple field-specific databases |
| "Include counter-evidence to strengthen argument" | Literature review is hypothesis-NEUTRAL, not argument | Include ALL relevant evidence regardless of hypothesis |
| "Paper-by-paper is clearer" | Actually harder to follow, misses synthesis | Synthesize by themes |
| "I found enough to understand the field" | You understand what you've read, not what exists | Search systematically, don't satisfice |
| "80% value in 5% of the time" | The other 20% is often what contradicts your assumptions | Do the full process |

---

## Quick Reference

### PRISMA Checklist (Minimum)

- [ ] Defined specific research question (PICO/FINER)
- [ ] Searched multiple databases (not just Google Scholar)
- [ ] Documented search strings for each database
- [ ] Applied consistent inclusion/exclusion criteria
- [ ] Screened ALL results (title/abstract, then full-text)
- [ ] Created PRISMA flow diagram tracking all papers
- [ ] Backward citation tracking (references in key papers)
- [ ] Forward citation tracking (papers citing key papers)
- [ ] Searched until saturation (not arbitrary count)
- [ ] Synthesized by themes (not paper-by-paper)
- [ ] Identified gaps in literature

### Database Quick Start

**Computer Science**: IEEE Xplore + ACM DL + arXiv
**Machine Learning**: arXiv + NeurIPS/ICML proceedings + Google Scholar (supplement)
**Interdisciplinary**: Web of Science + Scopus + field-specific database

### Synthesis Template

```markdown
## Theme 1: [Name]

**Papers addressing this**: [A, B, C, D]
**Consensus**: [What they agree on]
**Contradictions**: [Where they disagree]
**Gaps**: [What's missing]

## Theme 2: [Name]
...
```

---

## When NOT to Use Full Systematic Review

**Lighter methodology acceptable when**:
- Quick background reading (not for publication)
- Familiar field, just updating knowledge
- Reading one paper in depth

**Still document**:
- How you found papers (even if informal)
- Why you chose these papers
- What you might have missed

**Full systematic review required when**:
- Writing literature review section for publication
- PhD proposal or dissertation
- Making claims about "the state of the field"
- Research depends on complete understanding

---

## Examples

### Example 1: Scope Creep (50K Results)

**User**: "I searched 'AI fairness' and got 50,000 results. Where do I start?"

**Response**:
1. STOP. Define specific question FIRST using PICO:
   - Population: What systems? (hiring, lending, criminal justice?)
   - Intervention: What about fairness? (metrics? interventions? definitions?)
   - Comparison: Compared to what? (traditional ML, humans, other fairness approaches?)
   - Outcome: What matters? (accuracy? equity? discrimination measure?)

2. Example specific question: "What fairness metrics are used for hiring algorithms, and how do they compare in accuracy-fairness tradeoffs?"

3. THEN search with specific terms: `("fairness metric*" OR "bias metric*") AND ("hiring" OR "recruitment") AND ("tradeoff" OR "comparison")`

4. Results: Probably 200-500 papers (manageable), not 50,000

---

### Example 2: Cherry-Picking Prevention

**User**: "I think deep learning is inherently fairer. What papers support this?"

**Response**:
1. STOP. Reframe as hypothesis-neutral question: "How does fairness compare between deep learning and traditional ML approaches?"

2. Neutral search terms:
   - `("deep learning" OR "neural network") AND fairness AND ("traditional ML" OR "logistic regression" OR "SVM")`
   - Search for "fairness" not "fairness improvements"

3. Include papers showing:
   - ✅ Deep learning improves fairness
   - ✅ Deep learning worsens fairness
   - ✅ Mixed results / context-dependent

4. Synthesis addresses: When does DL help? When does it hurt? What moderates the effect?

---

### Example 3: PRISMA for "Just a Proposal"

**User**: "This is just a PhD proposal, not a journal submission. Do I need PRISMA?"

**Response**:
1. YES. Your proposal determines your research direction for 3-5 years.

2. PRISMA prevents you from:
   - Missing entire research branches
   - Designing experiments already done
   - Biasing your research based on incomplete literature understanding

3. Use PRISMA methodology (search, screen, document):
   - Prevents bias during YOUR search
   - Enables replication if you need to update the review later
   - Shows your committee you understand the field

4. You're not doing PRISMA "for them" - you're doing it to ensure your own research is well-founded.

---

## Summary

**Literature review is systematic investigation, not confirmation of beliefs.**

**Core rules**:
1. Define specific question BEFORE searching (PICO/FINER)
2. Search multiple databases (not just Google Scholar)
3. Document search strings and screening criteria (PRISMA)
4. Screen ALL results systematically (no arbitrary stopping)
5. Snowball (backward/forward citations)
6. Stop at saturation, not arbitrary count
7. Synthesize by themes, not paper-by-paper
8. Same rigor regardless of output type (proposal = dissertation)
9. Hypothesis-neutral search (include contradictory evidence)
10. PRISMA prevents YOUR bias, not just reader perception

**Meta-rule**: Systematic methodology = protection from bias. Shortcuts create blind spots.
