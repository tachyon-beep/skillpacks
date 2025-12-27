---
description: Design a new interface component with platform-aware patterns and accessibility built-in
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[feature_or_component_description]"
---

# Create Interface Command

You are designing a new interface component or feature. Apply platform-aware patterns with accessibility built-in from the start.

## Core Principle

**Accessibility constraints make design BETTER for everyone.**

Design for constraints first, then enhance. This creates interfaces that work for all users, all contexts.

## Design Process

### Step 1: Understand Context

**Ask clarifying questions:**

1. **Platform:** Web, mobile (iOS/Android), desktop, game?
2. **User:** Who are primary users? What context of use?
3. **Task:** What user goal does this support?
4. **Constraints:** Technical limitations? Brand guidelines?

### Step 2: Apply Platform Patterns

**Mobile (iOS/Android):**
- Touch targets 44x44px+ (iOS) / 48x48dp+ (Android)
- Thumb zone consideration (primary actions in lower half)
- Platform-specific gestures (swipe, pinch, long-press)
- iOS HIG / Material Design conventions

**Web Application:**
- Responsive breakpoints (mobile-first)
- Keyboard shortcuts for power users
- Data table patterns for complex data
- Multi-tab/window consideration

**Desktop Software:**
- Window management (panels, docking)
- Keyboard-first workflows
- Customization options
- Power-user features

**Game UI:**
- Visibility vs immersion balance
- Controller/gamepad navigation
- Readability during action
- Performance impact consideration

### Step 3: Apply Core Competencies

**Visual Design:**
- Establish visual hierarchy (primary, secondary, tertiary)
- Apply consistent spacing system
- Choose appropriate color palette (accessible)
- Select legible typography

**Information Architecture:**
- Organize content logically
- Limit options per view (5-7 rule)
- Apply progressive disclosure
- Ensure clear navigation

**Interaction Design:**
- Define all states (default, hover, active, disabled, error, loading)
- Provide appropriate feedback
- Design for keyboard navigation
- Create forgiving interactions

**Accessibility (Built-in):**
- Contrast ratios (4.5:1 text, 3:1 UI)
- Touch target sizes
- Semantic structure
- Color-independent indicators

### Step 4: Define Component Specifications

**For each UI element, specify:**

1. **Visual specs:**
   - Size (width, height, min/max)
   - Spacing (margin, padding)
   - Colors (fill, border, text)
   - Typography (font, size, weight)

2. **States:**
   - Default
   - Hover/focus
   - Active/pressed
   - Disabled
   - Error
   - Loading

3. **Interactions:**
   - Click/tap behavior
   - Keyboard interaction
   - Screen reader announcement
   - Animation/transition

4. **Responsiveness:**
   - Mobile behavior
   - Tablet behavior
   - Desktop behavior

## Design Output Format

```markdown
# Interface Design: [Component/Feature Name]

## Context

**Platform:** [Web/Mobile/Desktop/Game]
**User:** [Primary user description]
**Task:** [User goal supported]
**Constraints:** [Technical/brand limitations]

## Component Overview

[Brief description of what this component does and when to use it]

## Visual Specifications

### Layout
```
┌─────────────────────────────────┐
│  [ASCII diagram of layout]      │
│                                 │
└─────────────────────────────────┘
```

### Dimensions
| Element | Mobile | Tablet | Desktop |
|---------|--------|--------|---------|
| Container width | 100% | 400px | 480px |
| Button height | 48px | 44px | 40px |
| [etc.] | | | |

### Spacing
- Outer margin: 16px
- Inner padding: 12px
- Element gap: 8px

### Colors
| Element | Color | Usage |
|---------|-------|-------|
| Background | #FFFFFF | Container fill |
| Primary text | #1A1A1A | Headings, labels |
| Secondary text | #666666 | Helper text |
| Primary button | #0066CC | Main action |
| [etc.] | | |

### Typography
| Element | Font | Size | Weight | Line Height |
|---------|------|------|--------|-------------|
| Heading | System | 18px | 600 | 1.3 |
| Body | System | 16px | 400 | 1.5 |
| Button | System | 16px | 500 | 1.0 |

## States

### [Element Name]

**Default:**
- Background: #FFFFFF
- Border: 1px solid #CCCCCC
- Text: #1A1A1A

**Hover/Focus:**
- Background: #F5F5F5
- Border: 2px solid #0066CC
- Focus outline: 2px solid #0066CC, 2px offset

**Active/Pressed:**
- Background: #E0E0E0
- Border: 1px solid #999999

**Disabled:**
- Background: #F5F5F5
- Border: 1px solid #E0E0E0
- Text: #999999
- Cursor: not-allowed

**Error:**
- Border: 2px solid #CC0000
- Helper text: "Error message" in #CC0000
- Icon: Error icon left of field

**Loading:**
- Content replaced with spinner
- Interaction disabled

## Interactions

### Click/Tap
- [Element]: [Behavior description]

### Keyboard
- Tab: Move focus to next element
- Enter/Space: Activate button
- Escape: Close/cancel
- Arrow keys: Navigate options

### Screen Reader
- Button announces: "[Label], button"
- Input announces: "[Label], text field, [value]"
- Error announces: "[Field] error, [message]"

### Animation
- Hover transition: 150ms ease-out
- Focus transition: 100ms
- Loading spinner: Continuous rotation

## Responsive Behavior

### Mobile (< 768px)
- Full width container
- Stacked layout
- 48px touch targets

### Tablet (768px - 1024px)
- Centered container, max 400px
- Side-by-side where appropriate
- 44px touch targets

### Desktop (> 1024px)
- Fixed width container
- Multi-column layout options
- 40px click targets, keyboard shortcuts

## Accessibility Checklist

- [ ] Contrast ratio 4.5:1 for text
- [ ] Touch targets 44px+ minimum
- [ ] Focus indicators visible
- [ ] Labels associated with inputs
- [ ] Error messages announced
- [ ] Keyboard navigable
- [ ] Screen reader tested

## Usage Examples

### Example 1: [Use case]
[Description and mockup]

### Example 2: [Use case]
[Description and mockup]

## Anti-Patterns

**Don't:**
- [Anti-pattern 1 and why]
- [Anti-pattern 2 and why]

**Do:**
- [Correct pattern 1]
- [Correct pattern 2]
```

## Platform Quick Reference

### Mobile Touch Targets
- iOS: 44x44pt minimum
- Android: 48x48dp minimum
- Spacing: 8dp between targets

### Web Keyboard Shortcuts
- Tab/Shift+Tab: Navigate
- Enter/Space: Activate
- Escape: Close
- Arrow keys: Menu navigation

### Desktop Power Features
- Cmd/Ctrl+S: Save
- Cmd/Ctrl+Z: Undo
- /: Focus search (common web pattern)

## Scope Boundaries

**This command covers:**
- Component specification
- Visual design
- Interaction design
- Accessibility requirements
- Responsive behavior

**Not covered:**
- Implementation code
- Design review of existing work
- User research
