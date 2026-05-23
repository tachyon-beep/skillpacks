---
description: Critique an interface design with multi-competency assessment across visual, IA, interaction, and accessibility
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[design_file_or_screenshot]"
---

# Design Review Command

You are conducting a comprehensive UX design review. Apply multi-competency assessment covering visual design, information architecture, interaction patterns, and accessibility.

## Core Principle

**Accessibility constraints make design BETTER for everyone, not just users with disabilities.**

High contrast helps in sunlight. Large touch targets help when distracted. Clear language helps when stressed.

## Review Framework

### 1. Visual Design Assessment

**Evaluate:**

**Contrast & Color:**
- Text contrast ratio (4.5:1 AA, 7:1 AAA)
- UI component contrast (3:1 minimum)
- Color not sole indicator of meaning
- Colorblind-safe palette

**Typography:**
- Body text 16px+ mobile, 14px+ desktop
- Line height 1.5x for body
- Line length 45-75 characters
- Clear, legible fonts

**Visual Hierarchy:**
- Primary action clearly emphasized
- Logical grouping (proximity, similarity)
- Appropriate whitespace
- Consistent styling

**Layout:**
- Responsive across breakpoints
- Grid-based alignment
- Zoom to 200% without horizontal scroll

### 2. Information Architecture Assessment

**Evaluate:**

**Navigation:**
- Clear location indicators
- Logical hierarchy (3 levels max)
- Consistent navigation placement
- Breadcrumbs for deep structures

**Content Organization:**
- Chunked information (5-7 items per group)
- Clear headings and labels
- Progressive disclosure where appropriate
- Scannable content structure

**Findability:**
- Key actions discoverable
- Search functionality (if complex)
- Sensible defaults
- Clear categorization

### 3. Interaction Design Assessment

**Evaluate:**

**Feedback:**
- Button states (hover, active, disabled)
- Loading indicators
- Success/error messaging
- Progress indicators for multi-step

**Touch/Click Targets:**
- 44x44px minimum (iOS) / 48x48dp (Android)
- 8px spacing between targets
- Thumb-zone consideration (mobile)

**Forms:**
- Clear labels (not placeholder-only)
- Inline validation
- Error prevention
- Helpful error messages

**Keyboard:**
- Logical tab order
- Visible focus indicators
- All functions keyboard accessible

### 3a. AI-Surface Assessment (if applicable)

**Trigger** if the artefact under review is a chat assistant, AI copilot, agent loop, RAG UI, streaming-output panel, or any surface where model output is presented to the user.

**Action**: Additionally load `ai-experience-patterns.md` (sibling sheet in this pack) and apply the AI-UX Trust Stack:

- **Legibility** — does the user always know whether they are talking to a model, what model, what its limits are? Streaming-cursor distinct from final state?
- **Grounding** — citations linked to source spans, not free-floating; hover/click reveals provenance; "I don't know" is a first-class response
- **Steering** — user can interrupt, redirect, edit prior turn, set scope ("only use this document")
- **Refusal & Recovery** — refusals explain what is allowed instead; one-tap rephrase / report
- **Reversibility** — destructive tool actions (send, delete, charge, refund) require preview-then-confirm, with a visible undo window
- **Calibration** — confidence framing matches actual accuracy; no false certainty; uncertainty signalled when present

**WCAG 2.2 traps specific to AI surfaces**: `aria-live` token buffering during streaming, `prefers-reduced-motion` on streaming cursor, SC 3.3.8 (don't gate AI features behind cognitive-function authentication tests).

If the surface is **not** an AI artefact, skip this section.

### 4. Accessibility Assessment

**Universal Access Model (6 dimensions):**

**Visual:** Contrast, zoom, color independence
**Motor:** Touch targets, keyboard nav, no precision required
**Cognitive:** Clear language, chunked info, error recovery
**Screen Reader:** Semantic HTML, alt text, ARIA labels
**Temporal:** No timeouts, pauseable content, user-paced
**Situational:** Works in varied contexts (sunlight, one-handed)

## Review Process

### Step 1: First Impressions (2 min)
- What draws attention first?
- Is purpose immediately clear?
- Any obvious issues?

### Step 2: Competency-by-Competency (15-20 min)
- Visual design
- Information architecture
- Interaction design
- Accessibility

### Step 3: Platform-Specific (5 min)
- Mobile: Touch targets, thumb zones, gestures
- Web: Responsive, keyboard, data tables
- Desktop: Window management, shortcuts
- Game: Immersion, performance, controller

### Step 4: Synthesis (5 min)
- Prioritize issues (Critical > Major > Minor)
- Identify patterns
- Recommend actions

## Review Output Format

```markdown
# Design Review: [Design Name/Screen]

## Summary
**Overall Assessment:** [Strong/Acceptable/Needs Work]
**Critical Issues:** [Count]
**Major Issues:** [Count]

## Visual Design

### Strengths
- [Genuine strength with evidence]

### Issues
| Issue | Severity | Evidence | Recommendation |
|-------|----------|----------|----------------|
| [Description] | Critical/Major/Minor | [Specific location] | [Fix] |

## Information Architecture

### Strengths
- [Genuine strength]

### Issues
[Same table format]

## Interaction Design

### Strengths
- [Genuine strength]

### Issues
[Same table format]

## Accessibility

### WCAG 2.2 AA Compliance
- [ ] 1.4.3 Contrast (AA): [Pass/Fail]
- [ ] 2.1.1 Keyboard: [Pass/Fail]
- [ ] 2.4.7 Focus Visible: [Pass/Fail]
- [ ] 2.4.11 Focus Not Obscured (Minimum, new in 2.2): [Pass/Fail]
- [ ] 2.5.7 Dragging Movements (new in 2.2): [Pass/Fail]
- [ ] 2.5.8 Target Size Minimum 24px (new in 2.2): [Pass/Fail]
- [ ] 3.2.6 Consistent Help (new in 2.2): [Pass/Fail]
- [ ] 3.3.7 Redundant Entry (new in 2.2): [Pass/Fail]
- [ ] 3.3.8 Accessible Authentication Minimum (new in 2.2): [Pass/Fail]
- [ ] 4.1.2 Name/Role/Value: [Pass/Fail]

### AI Trust Stack (if AI surface)
- [ ] Legibility: [Pass/Fail]
- [ ] Grounding (citations linked to sources): [Pass/Fail]
- [ ] Steering (user can interrupt / scope): [Pass/Fail]
- [ ] Refusal & Recovery: [Pass/Fail]
- [ ] Reversibility (preview-then-confirm for tool actions): [Pass/Fail]
- [ ] Calibration (confidence matches accuracy): [Pass/Fail]

### Issues
[Same table format]

## Platform-Specific Notes
[If applicable]

## Priority Recommendations

### Critical (Fix Before Launch)
1. [Issue + action]

### Major (Fix Soon)
1. [Issue + action]

### Minor (Improvement Opportunities)
1. [Issue + action]

## Testing Recommendations
- [ ] Keyboard-only navigation test
- [ ] Screen reader test (NVDA/VoiceOver)
- [ ] Colorblind simulation
- [ ] Mobile device test
```

## Issue Severity Guide

**Critical:**
- Security/privacy concerns
- WCAG A violations
- Completely blocks functionality
- Data loss risk

**Major:**
- WCAG AA violations
- Significantly impairs usability
- Confusing flow/navigation
- Poor mobile experience

**Minor:**
- WCAG AAA opportunities
- Suboptimal but functional
- Inconsistent styling
- Polish improvements

## Cross-Pack Discovery

```python
import glob

# For documentation UX
writer_pack = glob.glob("plugins/muna-technical-writer/plugin.json")
if writer_pack:
    print("Available: muna-technical-writer for microcopy review")

# For security concerns
security_pack = glob.glob("plugins/ordis-security-architect/plugin.json")
if security_pack:
    print("Available: ordis-security-architect for auth flow review")
```

## Scope Boundaries

**This command covers:**
- Multi-competency design assessment
- WCAG compliance checking
- Platform-specific review
- Prioritized recommendations

**Not covered:**
- Detailed accessibility audit (use /accessibility-audit)
- User research (separate methodology)
- Implementation details
