---
description: Run comprehensive WCAG accessibility compliance audit with Universal Access Model assessment
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[design_or_url_to_audit]"
---

# Accessibility Audit Command

You are conducting a comprehensive accessibility audit using the Universal Access Model and WCAG guidelines. Your goal is to ensure the design works for everyone, regardless of ability or situation.

## Core Principle

**Accessibility constraints make design BETTER for everyone, not just users with disabilities.**

High contrast helps in sunlight. Large touch targets help when distracted. Clear language helps when stressed.

## Universal Access Model (6 Dimensions)

### Dimension 1: Visual Accessibility

**Contrast Checklist:**
- [ ] Text contrast 4.5:1 minimum (WCAG AA)
- [ ] Large text (18pt+) contrast 3:1 minimum
- [ ] UI components contrast 3:1 against background
- [ ] Links distinguishable without color alone

**Zoom/Resize Checklist:**
- [ ] Zoom to 200% without horizontal scrolling
- [ ] Text remains readable when zoomed
- [ ] Layouts reflow gracefully
- [ ] Font sizes use relative units (em, rem)

**Color Independence Checklist:**
- [ ] Information not conveyed by color alone
- [ ] Error states use icon + color + text
- [ ] Charts use patterns in addition to color
- [ ] Colorblind-safe palette verified

### Dimension 2: Motor Accessibility

**Touch Targets Checklist:**
- [ ] All interactive elements 44x44px+ (iOS) / 48x48dp+ (Android)
- [ ] 8px minimum spacing between targets
- [ ] Critical actions not at screen edges
- [ ] No precision required for interactions

**Keyboard Navigation Checklist:**
- [ ] All functionality accessible via keyboard
- [ ] Tab order is logical
- [ ] Focus indicators visible (2px outline minimum)
- [ ] No keyboard traps
- [ ] Skip links available

**Alternatives Checklist:**
- [ ] Drag-and-drop has keyboard alternative
- [ ] Hover interactions have keyboard equivalent
- [ ] Complex gestures have simple alternatives

### Dimension 3: Cognitive Accessibility

**Language Checklist:**
- [ ] 8th grade reading level or lower
- [ ] Sentences 20 words or less
- [ ] Jargon avoided or explained
- [ ] Instructions direct and actionable

**Cognitive Load Checklist:**
- [ ] Information chunked (5-7 items per group)
- [ ] Headings and lists organize content
- [ ] Progressive disclosure used
- [ ] No memory required across screens

**Error Handling Checklist:**
- [ ] Errors prevented with constraints
- [ ] Error messages clear and actionable
- [ ] Easy undo for mistakes
- [ ] Confirmations for destructive actions

### Dimension 4: Screen Reader Compatibility

**Semantic HTML Checklist:**
- [ ] Proper elements used (button, nav, main, article)
- [ ] Heading hierarchy logical (h1 → h2 → h3)
- [ ] Landmarks present (header, nav, main, footer)
- [ ] Lists used for lists (ul, ol)

**Alternative Text Checklist:**
- [ ] Images have descriptive alt text
- [ ] Decorative images marked as decorative (alt="")
- [ ] Icon buttons have text labels or aria-label
- [ ] Complex images explained

**ARIA Checklist:**
- [ ] Form inputs properly labeled
- [ ] Dynamic updates announced (aria-live)
- [ ] Expanded/collapsed states indicated
- [ ] Focus managed in modals

### Dimension 5: Temporal Accessibility

**Timeouts Checklist:**
- [ ] Timeouts avoidable or adjustable
- [ ] 5+ minute warning before session expires
- [ ] Users can extend timeouts easily
- [ ] Work saved if timeout occurs

**Content Control Checklist:**
- [ ] Users can pause/stop animations
- [ ] Auto-playing content controllable
- [ ] prefers-reduced-motion respected
- [ ] Carousels have pause controls

### Dimension 6: Situational Accessibility

**Environmental Checklist:**
- [ ] Works in bright sunlight (high contrast)
- [ ] Works in low light
- [ ] Works with one hand (mobile)
- [ ] Works in noisy/quiet environments

**Connection Checklist:**
- [ ] Works on slow connections (3G)
- [ ] Works offline (fallback provided)
- [ ] Large assets lazy-loaded
- [ ] Loading feedback provided

## Audit Process

### Step 1: Automated Scan

Run automated tools:
- Axe DevTools (catches 30-40% of issues)
- Lighthouse accessibility score
- WAVE for visual feedback

**Note automated findings but continue with manual testing.**

### Step 2: Dimension-by-Dimension Manual Testing

For each of the 6 dimensions:
1. Go through checklist items
2. Document issues found
3. Note specific locations
4. Rate severity

### Step 3: Screen Reader Testing

Test with:
- NVDA (Windows) or VoiceOver (macOS)
- Navigate entire interface
- Verify labels announced correctly
- Test form interactions

### Step 4: Keyboard-Only Testing

1. Unplug/hide mouse
2. Navigate entire interface with Tab, Shift+Tab, Enter, Esc, Arrows
3. Verify all functions accessible
4. Check focus visibility

### Step 5: Compile Report

## Audit Output Format

```markdown
# Accessibility Audit Report

**Audited:** [Design/URL]
**Date:** [Date]
**Standard:** WCAG 2.1 AA

## Executive Summary

**Overall Compliance:** [Compliant/Partially Compliant/Non-Compliant]
**Critical Issues:** [Count]
**Issues by Dimension:**
| Dimension | Critical | Major | Minor |
|-----------|----------|-------|-------|
| Visual | [#] | [#] | [#] |
| Motor | [#] | [#] | [#] |
| Cognitive | [#] | [#] | [#] |
| Screen Reader | [#] | [#] | [#] |
| Temporal | [#] | [#] | [#] |
| Situational | [#] | [#] | [#] |

## Automated Scan Results

**Axe DevTools:** [# issues found]
**Lighthouse Score:** [Score]/100

## Detailed Findings

### Dimension 1: Visual Accessibility

**Status:** [Pass/Partial/Fail]

| Issue | Severity | WCAG | Location | Fix |
|-------|----------|------|----------|-----|
| [Description] | Critical/Major/Minor | [Criterion] | [Location] | [Recommendation] |

### Dimension 2: Motor Accessibility

[Same format]

### Dimension 3: Cognitive Accessibility

[Same format]

### Dimension 4: Screen Reader Compatibility

[Same format]

### Dimension 5: Temporal Accessibility

[Same format]

### Dimension 6: Situational Accessibility

[Same format]

## WCAG Compliance Matrix

### Level A (Must Have)
| Criterion | Status | Notes |
|-----------|--------|-------|
| 1.1.1 Non-text Content | Pass/Fail | [Details] |
| 1.3.1 Info and Relationships | Pass/Fail | [Details] |
| 2.1.1 Keyboard | Pass/Fail | [Details] |
| 2.1.2 No Keyboard Trap | Pass/Fail | [Details] |
| 2.4.1 Bypass Blocks | Pass/Fail | [Details] |
| 3.3.1 Error Identification | Pass/Fail | [Details] |
| 3.3.2 Labels or Instructions | Pass/Fail | [Details] |

### Level AA (Should Have)
| Criterion | Status | Notes |
|-----------|--------|-------|
| 1.4.3 Contrast (Minimum) | Pass/Fail | [Details] |
| 1.4.4 Resize Text | Pass/Fail | [Details] |
| 1.4.11 Non-text Contrast | Pass/Fail | [Details] |
| 2.4.6 Headings and Labels | Pass/Fail | [Details] |
| 2.4.7 Focus Visible | Pass/Fail | [Details] |
| 3.2.3 Consistent Navigation | Pass/Fail | [Details] |
| 3.2.4 Consistent Identification | Pass/Fail | [Details] |
| 3.3.3 Error Suggestion | Pass/Fail | [Details] |
| 3.3.4 Error Prevention | Pass/Fail | [Details] |

## Remediation Plan

### Priority 1: Critical (Fix Immediately)
| Issue | Effort | Owner | Deadline |
|-------|--------|-------|----------|
| [Description] | S/M/L | [Team] | [Date] |

### Priority 2: Major (Fix Before Launch)
[Same format]

### Priority 3: Minor (Improvement)
[Same format]

## Testing Tools Used

- [ ] Axe DevTools
- [ ] WAVE
- [ ] Lighthouse
- [ ] NVDA/VoiceOver
- [ ] Keyboard-only navigation
- [ ] Colorblind simulation
- [ ] Zoom to 200%
- [ ] Mobile device testing
```

## Common Issues Quick Reference

| Issue | WCAG | Fix |
|-------|------|-----|
| Low contrast text | 1.4.3 | Increase to 4.5:1 |
| Missing alt text | 1.1.1 | Add descriptive alt |
| No focus indicator | 2.4.7 | Add 2px outline |
| Non-semantic markup | 1.3.1 | Use proper HTML elements |
| Tiny touch targets | 2.5.5 | Increase to 44x44px |
| Placeholder as label | 3.3.2 | Add visible label |
| Auto-playing video | 1.4.2 | Add pause controls |
| Color-only indicators | 1.4.1 | Add icon + text |

## Scope Boundaries

**This command covers:**
- Full WCAG 2.1 AA audit
- Universal Access Model assessment
- Remediation planning
- Compliance documentation

**Not covered:**
- General design review (use /design-review)
- User research
- Implementation fixes
