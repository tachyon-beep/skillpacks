
# User Research and Validation

## Overview

This skill teaches systematic user research methodology across all project phases, from initial discovery through post-launch optimization. Use when you need to understand users, test designs, or validate design decisions with evidence.

**Core Principle**: Design decisions based on user evidence are more successful than decisions based on assumptions, intuition, or stakeholder opinions alone.

## When to Use

Load this skill when:
- Starting a new project (need to understand users first)
- User asks: "How do I test this design?", "Will users understand this?"
- Validating design decisions before development
- Post-launch evaluation and optimization
- User mentions: "research", "testing", "validation", "users", "interviews", "usability"

**Don't use for**: Pure design specification (use visual/interaction/IA skills), technical implementation


## The User Understanding Model

A systematic 5-phase approach matching project lifecycle:

**Phase 1: DISCOVERY** - Understand users, context, problems
**Phase 2: GENERATIVE** - Explore possibilities, ideate solutions
**Phase 3: EVALUATIVE** - Test designs, identify usability issues
**Phase 4: VALIDATION** - Confirm design meets user needs
**Phase 5: POST-LAUNCH** - Measure impact, identify improvements

Each phase has specific methods, research questions, and deliverables.


## Phase 1: Discovery Research

### Purpose

Understand users deeply before designing solutions. Answer: Who are the users? What problems do they face? What's their current workflow?

**When to use**: Project kickoff, redesign, entering new market, foundational understanding

### Research Methods

#### Method 1: User Interviews (Contextual Inquiry)

**Setup:**
- 5-8 participants (representative sample)
- 30-60 minutes per interview
- User's environment (office, home, wherever they'd use product)
- Record with permission (audio/video)

**Script structure:**
1. Introduction (5 min) - Build rapport, explain purpose
2. Background (10 min) - Demographics, tech comfort, role
3. Current workflow (20 min) - "Walk me through how you currently..."
4. Pain points (15 min) - "What's frustrating about this?"
5. Goals and motivations (10 min) - "What would success look like?"

**Key questions:**
- "Tell me about the last time you [did task]..."
- "What's the hardest part of [workflow]?"
- "If you could wave a magic wand, what would you change?"
- "What workarounds have you developed?"

**Good practices:**
- Ask open-ended questions ("Tell me about..." not "Do you like...?")
- Follow up with "Why?" to understand motivations
- Observe actual behavior (not just self-reported)
- Stay neutral (don't lead toward your preferred solution)

**Anti-patterns:**
- Leading questions ("Don't you think X would be better?")
- Only talking to power users (miss mainstream needs)
- Interviewing in artificial settings (miss real context)
- Pitching your solution (you're here to learn, not sell)

#### Method 2: Field Observations (Ethnographic)

**Setup:**
- Observe users in natural environment
- 2-4 hours per observation
- Take notes on context, workarounds, interruptions
- Minimal intervention (be a fly on wall)

**What to observe:**
- Physical environment (lighting, noise, space constraints)
- Tools they use (software, hardware, paper notes)
- Workarounds (sticky notes, spreadsheets, manual steps)
- Interruptions (calls, colleagues, notifications)
- Emotional reactions (frustration, confusion, satisfaction)

**Documentation:**
- Photos of workspace (with permission)
- Video of workflows (if allowed)
- Detailed field notes
- Artifacts (forms they fill, outputs they create)

**Benefits:**
- See actual behavior (not self-reported)
- Discover unarticulated needs
- Understand real context (messy reality, not ideal)

#### Method 3: Diary Studies

**Setup:**
- Users log experiences over time (1-4 weeks)
- Mobile app, Google Form, or notebook
- Daily or event-triggered entries
- Minimal burden (3-5 minutes per entry)

**Entry prompts:**
- "What task did you complete?"
- "Rate difficulty (1-5)"
- "What went well? What was frustrating?"
- "Photo of situation (optional)"

**When to use:**
- Longitudinal behavior (patterns over time)
- Infrequent tasks (happens weekly, not daily)
- Context variety (different locations, times)

#### Method 4: Analytics Review (Quantitative Baseline)

**Data to gather:**
- Usage patterns (feature adoption, frequency)
- Drop-off points (where users abandon)
- Error rates (where things break)
- Time on task (efficiency baseline)
- Device/browser distribution

**Tools:**
- Google Analytics, Mixpanel (web)
- Firebase Analytics (mobile)
- Hotjar, FullStory (session recordings)
- Server logs (API usage, errors)

**Look for:**
- High-traffic flows (optimize these first)
- Abandonment funnels (usability issues)
- Feature deserts (built but not used)
- Error spikes (technical or UX problems)

### Research Questions to Answer

**User Characteristics:**
- Who are they? (demographics, role, tech comfort)
- What's their experience level? (novice → expert)
- What devices do they use? (mobile, desktop, tablet)
- What's their context? (office, home, on-the-go, distractions)

**Goals and Motivations:**
- What are they trying to accomplish?
- Why is this important to them?
- What's their success criteria?
- What would make them switch to competitor?

**Current Workflow:**
- How do they accomplish task today?
- What tools do they use?
- What steps are involved?
- How long does it take?

**Pain Points:**
- What's frustrating about current approach?
- Where do they get stuck?
- What errors do they encounter?
- What workarounds have they developed?

**Mental Models:**
- How do they think about the problem space?
- What terminology do they use?
- What categories make sense to them?
- What analogies do they use?

### Deliverables

#### User Personas

**Structure:**
- Name and photo (humanize)
- Demographics (age, role, tech comfort)
- Goals (what they want to accomplish)
- Pain points (current frustrations)
- Behaviors (how they work today)
- Quote (memorable user voice)

**Example:**

```
Sarah - The Efficiency-Focused Manager

Age: 35 | Role: Marketing Manager | Tech Comfort: Medium

Goals:
- Monitor campaign performance at a glance
- Quickly generate reports for executives
- Collaborate with team on content calendar

Pain Points:
- Data scattered across 5 different tools
- Takes 2 hours to build executive report
- Can't see real-time campaign status

Behaviors:
- Checks dashboard 3-4 times per day
- Works primarily on laptop, occasionally mobile
- Prefers visual summaries over detailed tables

Quote: "I just want to see what's working and what's not, without digging through spreadsheets."
```

**Good practices:**
- Base on real research (not assumptions)
- 3-5 personas (not 10+, too many to use)
- Focus on behaviors and goals (not just demographics)
- Include primary persona (design for this one first)

**Anti-patterns:**
- Marketing personas (not UX personas - different goals)
- Stereotypes without research basis
- Too many personas (design becomes unfocused)
- Perfect users (no pain points or constraints)

#### Journey Maps

**Structure:**
- Phases (stages of user journey)
- Actions (what user does at each stage)
- Thoughts (what they're thinking)
- Emotions (how they feel - graph line)
- Pain points (frustrations encountered)
- Opportunities (where design can help)

**Example: E-commerce Purchase Journey**

```
Phase 1: DISCOVERY
Actions: Search, browse categories, filter results
Thoughts: "Do they have what I need?"
Emotions: Curious → Frustrated (can't find it)
Pain points: Unclear categories, too many results
Opportunities: Better search, faceted filtering

Phase 2: EVALUATION
Actions: Compare products, read reviews, check specs
Thoughts: "Is this the right choice? Can I trust this?"
Emotions: Analytical → Uncertain
Pain points: Missing spec comparisons, conflicting reviews
Opportunities: Comparison tool, verified reviews

Phase 3: PURCHASE
Actions: Add to cart, enter shipping, payment
Thoughts: "Is this secure? What if I need to return it?"
Emotions: Cautious → Anxious
Pain points: Unexpected shipping costs, unclear return policy
Opportunities: Transparent pricing, clear return policy upfront
```

**Good practices:**
- Based on real user journeys (not ideal path)
- Include emotional arc (highs and lows)
- Highlight pain points (design opportunities)
- Show cross-channel journeys (mobile → desktop)

#### Research Insights Report

**Structure:**
1. Executive Summary (1 page)
2. Methodology (who, how, when)
3. Key Findings (5-10 insights with evidence)
4. User Personas (detailed profiles)
5. Journey Maps (visual workflows)
6. Opportunity Areas (prioritized design implications)
7. Appendix (raw data, interview transcripts)

**Key Findings Format:**

```
Finding 3: Users abandon checkout due to unexpected costs

Evidence:
- 67% of users mentioned shipping costs as frustration (8/12 interviews)
- Analytics show 43% cart abandonment at shipping step
- "I hate when they hide the total cost until the end" - multiple quotes

Implications:
- Show estimated total early in flow
- Be transparent about shipping costs upfront
- Consider free shipping threshold

Priority: HIGH (affects 43% of users at critical conversion point)
```


## Phase 2: Generative Research

### Purpose

Explore possibilities and generate solutions collaboratively with users. Answer: How do users think about organizing this? What solutions resonate with them?

**When to use**: After discovery, during ideation, before committing to specific design direction

### Research Methods

#### Method 1: Card Sorting

**Purpose**: Understand how users mentally organize information

**Types:**

**Open Card Sorting** (exploratory):
- Users create their own categories
- Reveals mental models
- Use for new IA, unclear structure

**Closed Card Sorting** (validation):
- Users place cards in predefined categories
- Tests proposed structure
- Use to validate IA decisions

**Setup:**
- 15-30 participants (for statistical validity)
- 40-60 cards (content items, features)
- Online tools: OptimalSort, UserZoom
- Physical cards: 3x5 index cards

**Process:**
1. Participant gets stack of cards (each = content item)
2. "Group these in a way that makes sense to you"
3. "Name each group you created"
4. (Optional) "Describe your thinking"

**Analysis:**
- Cluster analysis (which cards grouped together frequently)
- Category names (user terminology)
- Patterns across participants (consensus vs diversity)

**Deliverables:**
- Dendrogram (tree showing card relationships)
- Category labels (from user language)
- IA recommendations (structure that matches mental models)

**Good practices:**
- Use actual content (not generic labels)
- Test with representative users (not internal team)
- Mix easy and ambiguous items (find the edge cases)

**Anti-patterns:**
- Too many cards (>60 = cognitive overload)
- Cards that overlap (confuses participants)
- Only testing with 3 users (not statistically valid)

#### Method 2: Co-Design Workshops

**Purpose**: Generate solutions collaboratively with users

**Setup:**
- 6-10 participants (representative users)
- 2-3 hours
- Materials: paper, markers, sticky notes, templates
- Facilitators: 1 lead, 1 note-taker

**Activities:**

**Brainstorming:**
- "How might we [improve X]?"
- Quantity over quality (defer judgment)
- Build on others' ideas
- 10-15 minutes of rapid ideation

**Sketching:**
- Low-fidelity sketches (no art skills needed)
- 8-12 frames (crazy eights method)
- Focus on flows, not pixel perfection
- 5 minutes per sketch round

**Dot Voting:**
- Participants vote on favorite ideas
- 3 dots per person
- Can cluster votes or spread them
- Reveals group priorities

**Storyboarding:**
- Map ideal user journey
- Before/during/after scenarios
- Show context and emotions
- Identify key moments

**Benefits:**
- Users feel heard (buy-in)
- Uncover unarticulated needs
- Generate diverse ideas
- Validate assumptions early

#### Method 3: Concept Testing

**Purpose**: Test early ideas before investing in high-fidelity design

**Format:**
- Sketches, wireframes, or paper prototypes
- 5-8 participants
- 15-30 minutes per session
- Show concept, gather reactions

**Questions:**
- "What do you think this does?"
- "How would you use this?"
- "What's unclear or confusing?"
- "How does this compare to current approach?"
- "Would you use this? Why or why not?"

**Variations:**

**A/B Concept Testing:**
- Show two different approaches
- Ask which they prefer and why
- Not a vote (understand reasoning)

**Desirability Study:**
- Show concept
- "Pick 5 words that describe this" (from 60-word list)
- Reveals emotional response and brand perception

**Benefits:**
- Fail fast (before development)
- Understand reactions (not just usability)
- Compare directions (which resonates)

#### Method 4: Affinity Mapping (Synthesis)

**Purpose**: Synthesize research findings into insights

**Process:**
1. Write each observation on sticky note (from interviews, studies)
2. Put all notes on wall
3. Group similar observations together
4. Name each group (theme)
5. Identify patterns across themes

**Example themes:**
- "Users struggle with navigation" (12 notes)
- "Price transparency is critical" (8 notes)
- "Mobile experience is frustrating" (15 notes)

**Deliverables:**
- Thematic clusters (key insights)
- Priority order (by frequency or impact)
- Design implications (what to do about it)

**Good practices:**
- Involve cross-functional team (diverse perspectives)
- Use actual user quotes (not interpretations)
- Look for patterns (not one-off comments)
- Connect to business goals (not just user wants)

### Deliverables

**Information Architecture Proposals:**
- Sitemap (navigation structure)
- Category labels (user terminology)
- Justification (based on card sorting, mental models)

**Low-Fidelity Concepts:**
- Paper sketches (workflow options)
- Wireframes (structure, no visual design)
- Flow diagrams (user paths)

**Feature Prioritization:**
- MoSCoW method (Must/Should/Could/Won't)
- Impact vs effort matrix (quick wins vs long-term)
- User priority (from co-design voting)


## Phase 3: Evaluative Research

### Purpose

Test designs to identify usability issues before development. Answer: Can users complete tasks? Where do they struggle?

**When to use**: Mid-design, after wireframes/prototypes created, before high-fidelity investment

### Research Methods

#### Method 1: Usability Testing (Moderated)

**Purpose**: Watch users attempt tasks, identify friction points

**Setup:**
- 5-8 participants (Nielsen: 5 users find 85% of issues)
- 45-60 minutes per session
- Prototype (clickable wireframes or mockups)
- Think-aloud protocol (narrate thoughts)

**Test structure:**

1. **Introduction (5 min)**
   - Explain purpose: "Testing design, not you"
   - Ask to think aloud
   - Remind: no wrong answers

2. **Warm-up task (5 min)**
   - Easy task to practice thinking aloud
   - Builds comfort

3. **Core tasks (30 min)**
   - 5-7 realistic tasks
   - Specific, scenario-based
   - "You need to [goal]. Show me how you'd do that."

4. **Post-task questions (10 min)**
   - "What was confusing?"
   - "What would you change?"
   - "How does this compare to current approach?"

5. **Wrap-up (5 min)**
   - Overall impressions
   - Thank participant

**Good task examples:**
- "You need to change your shipping address. Show me how." (specific goal)
- "Find the return policy for this product." (findability)
- "Compare the features of these two plans." (information clarity)

**Bad task examples:**
- "Click the settings button" (too specific, not realistic)
- "What do you think of this page?" (not a task)
- "Do you like the blue button?" (leading, not behavioral)

**What to observe:**
- Task success (completed, partial, failed)
- Time on task (efficiency)
- Errors (wrong clicks, backtracking)
- Confusion (hesitation, pauses, "hmm...")
- Workarounds (unexpected paths to goal)
- Emotional reactions (frustration, delight)

**Note-taking:**
- Use observer template (task, time, success, notes, severity)
- Record session (with permission)
- Tag issues (navigation, labeling, feedback, visual hierarchy)

**Good practices:**
- Don't help (unless completely stuck)
- Probe for understanding ("What are you thinking?")
- Stay neutral (no leading)
- Test realistic tasks (not feature tours)

**Anti-patterns:**
- Leading users ("Try clicking the blue button")
- Defending design ("But that's obvious, isn't it?")
- Testing with 1-2 users only (not enough)
- Testing final-stage designs (too late to fix major issues)

#### Method 2: Unmoderated Remote Testing

**Purpose**: Scale testing across more users, faster

**Tools:**
- UserTesting.com
- Maze
- Lookback
- UsabilityHub

**Setup:**
- 10-20 participants (broader sample)
- 15-20 minutes per test
- Pre-recorded task instructions
- Automated metrics

**Advantages:**
- Faster turnaround (hours, not weeks)
- Larger sample size (statistical confidence)
- No scheduling logistics
- Users in natural environment

**Disadvantages:**
- Can't probe ("Why did you do that?")
- Technical issues harder to diagnose
- Less nuanced observations

**When to use:**
- Quick validation (A vs B)
- Large sample needed
- Limited budget/time
- Simple task flows

#### Method 3: Tree Testing

**Purpose**: Validate information architecture (findability)

**Setup:**
- Text-only hierarchy (no visual design)
- 15-30 participants
- 5-8 findability tasks
- Online tool: Treejack (Optimal Workshop)

**Process:**
1. Show text sitemap (categories and subcategories)
2. Task: "Where would you look to find [X]?"
3. Participant clicks through hierarchy
4. Success or failure recorded

**Metrics:**
- Task success rate (found correct location)
- Directness (straight path vs backtracking)
- Time to complete
- First click (did they start in right direction?)

**Analysis:**
- Which categories are clear vs confusing?
- Where do users go wrong? (misleading labels)
- Which content is hard to find? (IA gaps)

**Benefits:**
- Tests IA without visual design (pure structure)
- Identifies labeling issues
- Fast to run and analyze

#### Method 4: First-Click Testing

**Purpose**: Validate if users can start tasks correctly

**Setup:**
- Screenshot or prototype
- "To accomplish [task], what would you click first?"
- One click per task
- 15-30 participants

**Analysis:**
- Heat map (where users clicked)
- Success rate (clicked correct element)
- Misclicks (where they went wrong)

**Insight:**
- First click predicts task success (if first click right, 87% complete task successfully)
- Identifies unclear navigation
- Tests visual hierarchy (do important elements get clicked?)

### Research Questions to Answer

**Usability:**
- Can users complete core tasks?
- How long does it take? (efficiency)
- What errors do they make?
- Where do they get stuck?

**Findability:**
- Can users locate features/content?
- Do labels make sense?
- Is navigation intuitive?
- Do search results match expectations?

**Comprehension:**
- Do users understand what elements do?
- Is feedback clear?
- Are error messages helpful?
- Is terminology familiar?

**Satisfaction:**
- Is the experience pleasant?
- What delights users?
- What frustrates them?
- Would they recommend it?

### Deliverables

#### Usability Findings Report

**Structure:**
1. **Executive Summary**
   - Overall success rates
   - Top 5 critical issues
   - Recommendations priority

2. **Methodology**
   - Participants (who, how recruited)
   - Tasks tested
   - Test environment

3. **Findings by Task**
   - Task description
   - Success rate (5/8 completed)
   - Time on task (average)
   - Key issues observed
   - Severity rating

4. **Issue Details**
   - Issue description (what went wrong)
   - Severity (critical/high/medium/low)
   - Evidence (quotes, clips, frequency)
   - Recommendation (how to fix)
   - Affected users (5/8 participants)

5. **Prioritized Recommendations**
   - Quick wins (high impact, low effort)
   - Critical fixes (before launch)
   - Future improvements (next iteration)

**Issue Severity Scale:**
- **Critical**: Blocks task completion, affects all users
- **High**: Major friction, affects most users
- **Medium**: Slows users down, workarounds exist
- **Low**: Minor annoyance, rare occurrence

**Example Finding:**

```
Issue #3: Users can't find the "Save" button (HIGH SEVERITY)

Observed: 6/8 participants looked around screen for 10+ seconds before finding save button in bottom-right corner. Two gave up and asked for help.

Quotes:
- "Where's the save button? I keep looking for it..."
- "Oh, it's way down there. I didn't see it."

Recommendation: Move save button to top-right (expected location) OR make it a prominent floating action button. Consider adding keyboard shortcut (Cmd+S) for power users.

Priority: Fix before launch (blocks core workflow)
```

#### Task Success Rates

**Metrics table:**

| Task | Success | Partial | Failed | Avg Time | Notes |
|------|---------|---------|--------|----------|-------|
| Change shipping address | 7/8 (88%) | 1/8 | 0/8 | 45s | Easy |
| Find return policy | 4/8 (50%) | 2/8 | 2/8 | 2m 15s | Label unclear |
| Compare pricing plans | 2/8 (25%) | 3/8 | 3/8 | 3m 30s | Table confusing |

**Benchmark targets:**
- Core tasks: >80% success rate
- Secondary tasks: >60% success rate
- Average task time: <2 minutes for common tasks

**Issue Severity Ratings:**
- Critical: 5 issues (must fix)
- High: 8 issues (should fix)
- Medium: 12 issues (consider)
- Low: 6 issues (backlog)


## Phase 4: Validation Research

### Purpose

Confirm design meets user needs and follows best practices before launch. Answer: Is design ready to ship?

**When to use**: Pre-launch, final design validation, quality assurance

### Research Methods

#### Method 1: Heuristic Evaluation (Expert Review)

**Purpose**: Expert applies usability principles to find violations

**Framework: Nielsen's 10 Usability Heuristics**

1. **Visibility of system status** - Users know what's happening
2. **Match system and real world** - Familiar language and concepts
3. **User control and freedom** - Undo/redo, escape hatches
4. **Consistency and standards** - Patterns match conventions
5. **Error prevention** - Design prevents mistakes
6. **Recognition over recall** - Minimize memory load
7. **Flexibility and efficiency** - Shortcuts for power users
8. **Aesthetic and minimalist** - No unnecessary information
9. **Error recovery** - Help users fix mistakes
10. **Help and documentation** - When needed, accessible

**Process:**
1. Expert walks through interface
2. Evaluates against each heuristic
3. Documents violations (with severity)
4. Provides recommendations

**Severity scale:**
- 0 = Not a problem
- 1 = Cosmetic (fix if time)
- 2 = Minor usability issue
- 3 = Major usability issue (priority)
- 4 = Usability catastrophe (must fix)

**Benefits:**
- Fast (1-2 days)
- Cheap (expert time only, no users)
- Finds 30-40% of issues
- Good for catching obvious violations

**Limitations:**
- Expert opinions (not user behavior)
- Misses novel issues (not covered by heuristics)
- Should complement user testing (not replace)

#### Method 2: Cognitive Walkthrough

**Purpose**: Step through tasks to find learning barriers

**Process:**
1. Define user persona and goals
2. List task steps
3. For each step, ask:
   - "Will user know what to do?"
   - "Will user see how to do it?"
   - "Will user understand the feedback?"
   - "Will user know they're making progress?"

**Example walkthrough: Change password**

```
Step 1: User needs to find settings
→ Will user know settings exist? (Yes, common pattern)
→ Will user see settings menu? (Yes, in top-right nav)
→ Issue: None

Step 2: User needs to find "Change Password"
→ Will user know it's in Account section? (Maybe - test)
→ Will user see it in the list? (Yes, clear label)
→ Will user understand it's clickable? (Yes, styled as link)
→ Issue: None

Step 3: User needs to enter current password
→ Will user understand why? (Yes, security explained)
→ Will user remember current password? (Maybe - offer "Forgot?" link)
→ Issue: Add "Forgot password?" link
```

**Deliverable**: Step-by-step analysis with identified learning barriers

#### Method 3: Accessibility Audit (WCAG Compliance)

**Purpose**: Ensure design is accessible to people with disabilities

**Standards: WCAG 2.1 Level AA** (minimum legal requirement)

**Automated checks** (tools: Axe, Lighthouse, WAVE):
- Color contrast (4.5:1 for text, 3:1 for UI elements)
- Alt text on images
- Form labels
- Heading hierarchy
- Link text (not "click here")

**Manual checks:**
- Keyboard navigation (Tab, Enter, Esc, arrows)
- Focus indicators visible (2px outline)
- Skip links (bypass navigation)
- Screen reader compatibility (NVDA, JAWS, VoiceOver)
- Semantic HTML (button, nav, main, article)
- ARIA labels where needed

**Test scenarios:**
- Navigate entire flow with keyboard only (no mouse)
- Turn on screen reader, close eyes, complete task
- Zoom to 200%, verify no content loss
- Toggle high contrast mode, verify readable
- Test with colorblind simulator (Stark plugin)

**Deliverable**: Accessibility audit report
- Issues found (with WCAG reference)
- Severity (A, AA, AAA)
- Recommendations (how to fix)
- Priority (critical before launch vs future improvement)

**Integration**: Reference `accessibility-and-inclusive-design` skill for detailed evaluation framework

#### Method 4: Beta Testing

**Purpose**: Validate in real-world conditions with real users

**Setup:**
- Invite 50-200 users (representative sample)
- 1-4 weeks of use
- Production or staging environment
- Feedback channels (in-app, survey, support)

**Data to collect:**
- Usage metrics (adoption, frequency)
- Error rates (crashes, bugs)
- Support tickets (what confuses users)
- Satisfaction survey (NPS, satisfaction ratings)
- Feature requests (what's missing)

**Benefits:**
- Real workflows (not test scenarios)
- Real data (not lab environment)
- Uncovers edge cases (not anticipated)
- Validates at scale

**Risks:**
- Can't fix major issues this late (should be polished already)
- Users frustrated if too buggy (damages trust)
- Sensitive to first impressions

**When to skip:**
- High-stakes launches (financial, medical)
- Critical bug risk (test more before beta)
- No time to act on feedback (too close to launch)

### Deliverables

**Heuristic Evaluation Report:**
- Violations by heuristic (grouped)
- Severity ratings (prioritized)
- Screenshots (annotated)
- Recommendations (actionable)

**Accessibility Audit:**
- WCAG compliance status (pass/fail per criterion)
- Issues by severity (critical → low)
- Remediation steps (how to fix)
- Re-test checklist (verify fixes)

**Beta Feedback Synthesis:**
- Quantitative metrics (usage, errors, satisfaction)
- Qualitative themes (from support tickets, feedback)
- Priority issues (blockers vs nice-to-haves)
- Go/no-go recommendation (ready to launch?)


## Phase 5: Post-Launch Research

### Purpose

Measure impact and identify next improvements. Answer: Is it working? What should we optimize next?

**When to use**: After launch, ongoing optimization, annual reviews

### Research Methods

#### Method 1: Analytics Analysis (Behavioral Data)

**Purpose**: Understand actual usage patterns at scale

**Metrics to track:**

**Adoption:**
- New users per week/month
- Feature usage (% of users who tried feature)
- Repeat usage (% who return)
- Power users (top 10% usage)

**Engagement:**
- Sessions per user (frequency)
- Session duration (time spent)
- Actions per session (depth)
- Retention (day 1, day 7, day 30)

**Performance:**
- Task completion rate (% who finish flow)
- Time to complete (efficiency)
- Error rate (% who encounter errors)
- Drop-off points (where users abandon)

**Conversion:**
- Funnel conversion (% through each step)
- Drop-off analysis (where and why)
- Assisted conversions (multi-touch attribution)

**Tools:**
- Google Analytics 4 (web)
- Mixpanel, Amplitude (product analytics)
- Firebase (mobile)
- Hotjar, FullStory (session recordings)

**Analysis techniques:**
- Cohort analysis (compare user groups over time)
- Funnel analysis (conversion optimization)
- Path analysis (common user journeys)
- Segmentation (power users vs casual)

#### Method 2: Surveys (Satisfaction Data)

**Purpose**: Measure user satisfaction and sentiment

**Types:**

**NPS (Net Promoter Score):**
- "How likely are you to recommend [product] to a friend?" (0-10)
- Detractors (0-6), Passives (7-8), Promoters (9-10)
- NPS = % Promoters - % Detractors
- Follow-up: "Why did you give that score?"

**CSAT (Customer Satisfaction):**
- "How satisfied are you with [feature/experience]?" (1-5)
- Measure specific interactions (support, checkout, onboarding)
- Benchmark: >80% satisfied (4-5 rating)

**CES (Customer Effort Score):**
- "How easy was it to [complete task]?" (1-7)
- Predicts loyalty (ease → retention)
- Target: <2 average effort score

**Custom surveys:**
- Feature-specific questions
- Usability self-assessment
- Priorities for improvement
- Open-ended feedback

**Best practices:**
- Short surveys (5 questions max)
- In-context (after completing task)
- Right timing (not too frequent)
- Follow up on negative scores (reach out, fix issues)

#### Method 3: Session Recordings (Qualitative Observation)

**Purpose**: Watch real users interact with live product

**Tools:**
- Hotjar (web)
- FullStory (web)
- Lookback (mobile)
- Microsoft Clarity (free)

**What to look for:**
- Rage clicks (rapid clicking on unresponsive element)
- Confusion (hesitation, backtracking, random clicks)
- Error encounters (how users recover)
- Workarounds (creative solutions to design gaps)
- Unexpected paths (users find their own way)

**Analysis:**
- Tag sessions by behavior (errors, success, confusion)
- Watch sample of sessions per segment (new users, power users, churned)
- Identify patterns (common issues affecting multiple users)
- Prioritize fixes (high-impact, high-frequency)

**Benefits:**
- See actual behavior (not self-reported)
- Discover unknown issues (not in hypothesis)
- Empathy building (watch real struggles)

**Limitations:**
- Time-consuming (manual review)
- Privacy concerns (anonymize, get consent)
- Sampling bias (recorded sessions may not be representative)

#### Method 4: Support Ticket Analysis

**Purpose**: Find pain points users report

**Process:**
1. Export support tickets (last 1-3 months)
2. Categorize by issue type (bug, question, request, complaint)
3. Tag by product area (checkout, navigation, settings)
4. Quantify frequency (which issues are common)
5. Identify patterns (related issues clustering)

**Insights:**
- Usability gaps (repeated "how do I..." questions)
- Confusing flows (users need help to complete)
- Missing features (workaround requests)
- Error messaging (unclear error explanations)

**Integration with design:**
- High support volume = design issue (not just support issue)
- FAQ content becomes in-app help/tooltips
- Common questions reveal discoverability problems

### Deliverables

**Usage Metrics Dashboard:**
- Weekly active users (trend)
- Feature adoption (% using each feature)
- Task completion rate (success %)
- Drop-off points (where users abandon)
- Top user paths (common journeys)

**Satisfaction Scores:**
- NPS trend (month over month)
- CSAT by feature (which are well-received)
- Verbatim feedback (user quotes)
- Sentiment analysis (positive vs negative themes)

**Improvement Roadmap:**
- Priority 1 (critical issues, launch blockers)
- Priority 2 (high-impact improvements)
- Priority 3 (nice-to-haves, future enhancements)
- Justification (data supporting each priority)

**Example roadmap item:**

```
Improvement: Redesign checkout flow

Evidence:
- 38% cart abandonment at shipping step (analytics)
- "Checkout is too long" - 47 support tickets (3 months)
- NPS detractors cite checkout complexity (12/20 mentions)
- Session recordings show 2m 45s average (benchmark: 1m 30s)

Impact: HIGH
- Directly affects revenue (38% abandonment)
- Frequent user complaint (47 tickets)
- Competitive disadvantage (competitors faster)

Recommendation: Reduce checkout from 4 steps to 2, streamline form fields, add guest checkout option

Expected outcome: Reduce abandonment to <25%, increase conversion by 13%

Priority: P1 (next quarter)
```


## Integration with Design Skills

Research informs and validates design decisions:

### Discovery Research → Design Skills

**Findings feed into:**
- `information-architecture` - Mental models inform IA structure
- `visual-design-foundations` - Understand user context (lighting, device, distractions)
- `interaction-design-patterns` - Learn user workflows and expectations
- `accessibility-and-inclusive-design` - Identify user needs (disabilities, constraints)

**Example**: Discovery reveals users frequently switch between mobile and desktop → Design must prioritize responsive consistency and cross-device sync

### Generative Research → Design Skills

**Card sorting informs:**
- `information-architecture` - Navigation structure and labels

**Co-design feeds:**
- `visual-design-foundations` - User preferences for layout and hierarchy
- `interaction-design-patterns` - Expected interactions and feedback

### Evaluative Research → Design Skills

**Usability testing validates:**
- `visual-design-foundations` - Is hierarchy clear? Do users notice CTAs?
- `information-architecture` - Can users find content? Are labels clear?
- `interaction-design-patterns` - Do interactions feel responsive? Are affordances clear?
- `accessibility-and-inclusive-design` - Can users with disabilities complete tasks?

**Example**: Testing reveals users miss "Save" button → Visual design needs higher contrast or better placement

### Post-Launch Research → All Design Skills

**Analytics and feedback identify:**
- High-friction flows (redesign with all skills)
- Underused features (discoverability issue → IA)
- Frequent errors (interaction design → better feedback)
- Accessibility complaints (audit and fix)


## When to Use Which Phase

### Starting New Project
→ **Phase 1: DISCOVERY** (understand users, context, problems)
→ **Phase 2: GENERATIVE** (explore solutions, ideate)
→ Design phase (apply visual/IA/interaction skills)
→ **Phase 3: EVALUATIVE** (test designs)
→ **Phase 4: VALIDATION** (pre-launch check)
→ Launch
→ **Phase 5: POST-LAUNCH** (measure impact, optimize)

### Redesigning Existing Product
→ **Phase 1: DISCOVERY** (understand current pain points)
→ **Phase 3: EVALUATIVE** (test current design, identify issues)
→ Design improvements (apply skills)
→ **Phase 3: EVALUATIVE** (test improvements)
→ **Phase 4: VALIDATION** (pre-launch check)
→ Launch
→ **Phase 5: POST-LAUNCH** (measure improvement)

### Optimizing Live Product
→ **Phase 5: POST-LAUNCH** (identify issues from analytics/support)
→ **Phase 3: EVALUATIVE** (test hypothesized improvements)
→ Design iteration (apply skills)
→ **Phase 3: EVALUATIVE** (validate improvements)
→ Launch
→ **Phase 5: POST-LAUNCH** (measure impact)

### Adding New Feature
→ **Phase 2: GENERATIVE** (co-design, concept testing)
→ Design feature (apply skills)
→ **Phase 3: EVALUATIVE** (test with users)
→ **Phase 4: VALIDATION** (pre-launch check)
→ Launch
→ **Phase 5: POST-LAUNCH** (measure adoption)


## Research Methods Catalog

### Qualitative Methods (Understanding Why)

| Method | Phase | Purpose | Participants | Duration |
|--------|-------|---------|--------------|----------|
| User Interviews | Discovery | Understand needs, context | 5-8 | 30-60 min |
| Field Observations | Discovery | See real workflows | 3-5 | 2-4 hours |
| Diary Studies | Discovery | Longitudinal behavior | 10-15 | 1-4 weeks |
| Card Sorting | Generative | Mental models, IA | 15-30 | 20-30 min |
| Co-design Workshops | Generative | Generate solutions | 6-10 | 2-3 hours |
| Concept Testing | Generative | Test early ideas | 5-8 | 15-30 min |
| Usability Testing | Evaluative | Find usability issues | 5-8 | 45-60 min |
| Cognitive Walkthrough | Validation | Learning barriers | 2-3 experts | 2-4 hours |
| Session Recordings | Post-Launch | Real user behavior | Sample | 5-10 min/session |

### Quantitative Methods (Measuring What)

| Method | Phase | Purpose | Sample Size | Analysis |
|--------|-------|---------|-------------|----------|
| Analytics Review | Discovery | Usage patterns | All users | Descriptive |
| Tree Testing | Evaluative | IA validation | 15-30 | Success rate |
| First-Click Testing | Evaluative | Visual hierarchy | 15-30 | Heat map |
| A/B Testing | Evaluative | Compare designs | 100+ per variant | Statistical |
| Surveys (NPS, CSAT) | Post-Launch | Satisfaction | 50-200 | Descriptive |
| Funnel Analysis | Post-Launch | Conversion | All users | Drop-off % |

### Validation Methods (Quality Assurance)

| Method | Phase | Purpose | Evaluators | Time |
|--------|-------|---------|------------|------|
| Heuristic Evaluation | Validation | Usability principles | 3 experts | 1-2 days |
| Accessibility Audit | Validation | WCAG compliance | 1-2 specialists | 2-4 days |
| Beta Testing | Validation | Real-world validation | 50-200 | 1-4 weeks |


## Deliverable Templates

### User Persona Template

```
[Name] - [Role/Archetype]

Photo: [Realistic stock photo]

Demographics:
- Age: [Range]
- Role: [Job title or context]
- Tech Comfort: [Novice/Intermediate/Expert]
- Location: [Context for usage]

Goals:
1. [Primary goal]
2. [Secondary goal]
3. [Tertiary goal]

Pain Points:
1. [Current frustration #1]
2. [Current frustration #2]
3. [Current frustration #3]

Behaviors:
- [How they work today]
- [Tools they use]
- [Frequency of use]
- [Device preferences]

Quote: "[Memorable statement that captures their perspective]"

Scenario: [Day-in-the-life narrative showing context]
```

### Journey Map Template

```
Journey: [User goal - e.g., "Purchase product online"]

Phase 1: [STAGE NAME]
├─ Actions: [What user does]
├─ Thoughts: [What they're thinking]
├─ Emotions: [How they feel - rate 1-10]
├─ Pain Points: [Frustrations]
└─ Opportunities: [Design improvements]

Phase 2: [STAGE NAME]
[Repeat structure]

Emotion Graph: [Visual line showing emotional highs and lows across journey]

Key Insights:
- [Overall pattern]
- [Critical moments]
- [Design opportunities]
```

### Usability Findings Template

```
Issue #[X]: [Brief description]

Severity: [Critical / High / Medium / Low]

Frequency: [X/Y participants affected]

Description:
[What went wrong in detail]

Evidence:
- Observation: [What you saw]
- Quotes: "[User statement]"
- Metrics: [Time, clicks, errors]

Impact:
[How this affects users and business goals]

Recommendation:
[Specific, actionable fix with rationale]

Priority: [P0/P1/P2/P3]
- P0 = Blocker (must fix before launch)
- P1 = Critical (fix ASAP post-launch)
- P2 = Important (next sprint)
- P3 = Backlog (future improvement)

Related Issues: [Links to other findings]
```


## Common Patterns

### Pattern 1: Research-Driven Design Process
**Discovery** (understand) → **Generative** (ideate) → Design (create) → **Evaluative** (test) → **Validation** (quality check) → Launch → **Post-Launch** (optimize)

### Pattern 2: Lean Validation
**Evaluative** (test current state) → Design iteration → **Evaluative** (test improvement) → Launch if better

### Pattern 3: Continuous Improvement
**Post-Launch** (identify issues) → Design fixes → **Evaluative** (test fixes) → Launch → **Post-Launch** (measure impact) → Repeat

### Pattern 4: Feature Development
**Generative** (co-design) → Design feature → **Evaluative** (usability test) → **Validation** (QA) → Launch → **Post-Launch** (adoption metrics)


## Red Flags & Anti-Patterns

### Anti-Pattern 1: Skipping Discovery
**Problem**: Designing without understanding users
**Consequence**: Build wrong thing, wasted effort
**Fix**: Always start with discovery (even if brief)

### Anti-Pattern 2: Testing Too Late
**Problem**: Usability testing after development complete
**Consequence**: Expensive to fix, must ship with known issues
**Fix**: Test wireframes/prototypes before development

### Anti-Pattern 3: Testing with Wrong Users
**Problem**: Using internal team or unrepresentative users
**Consequence**: Miss real user issues, false confidence
**Fix**: Recruit actual target users (representative sample)

### Anti-Pattern 4: Ignoring Negative Feedback
**Problem**: Dismissing usability issues ("Users will learn it")
**Consequence**: Poor user experience, churn, support burden
**Fix**: Take all feedback seriously, prioritize based on severity and frequency

### Anti-Pattern 5: Over-Relying on Quantitative Data
**Problem**: Only looking at analytics, no qualitative insight
**Consequence**: Know what's broken, not why or how to fix
**Fix**: Combine quant (what) with qual (why) methods

### Anti-Pattern 6: Research Without Action
**Problem**: Conducting research but not using findings
**Consequence**: Wasted time, frustrated participants
**Fix**: Only research what you can act on, prioritize findings, implement changes

### Anti-Pattern 7: One-Time Research
**Problem**: Research once, never again
**Consequence**: Design becomes outdated as users evolve
**Fix**: Establish continuous research cadence (quarterly pulse checks)

### Anti-Pattern 8: Confusing Preference with Usability
**Problem**: "Users said they like blue, so we made it blue"
**Consequence**: Aesthetic preference doesn't equal usability
**Fix**: Test behavior (can they complete tasks?), not opinions (do they like it?)


## Related Skills

**Core UX Skills:**
- `lyra/ux-designer/information-architecture` - Research validates IA structure (card sorting, tree testing)
- `lyra/ux-designer/accessibility-and-inclusive-design` - Research with users with disabilities, accessibility audits
- `lyra/ux-designer/visual-design-foundations` - Research informs visual hierarchy, usability testing validates designs
- `lyra/ux-designer/interaction-design-patterns` - Research validates interaction clarity and feedback

**Meta Skills:**
- `lyra/ux-designer/using-ux-designer` - Routes to this skill when research/validation needed
- `lyra/ux-designer/ux-fundamentals` - Explains research terminology and principles

**Platform Extensions:**
- All platform skills benefit from research findings (validates platform-specific patterns)

**Cross-Faction:**
- `muna/technical-writer/*` - Research methods apply to documentation usability testing
