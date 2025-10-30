---
name: ux-fundamentals
description: Use when explaining UX concepts, teaching design principles, or providing foundational knowledge - covers core UX terminology, mental models, design thinking, and when to use each lyra skill
---

# UX Fundamentals

## Overview

This teaching skill explains core UX principles, terminology, and mental models. Use when users ask "What is X?" or need foundational understanding before applying other Lyra skills.

**Core Principle**: Understanding why design principles exist enables better application than memorizing rules.

## When to Use

Load this skill when:
- User asks "What is [UX concept]?"
- User needs explanation of design terminology
- User wants to understand UX thinking approach
- User is new to UX and needs orientation
- You need to explain rationale behind design recommendations

**Don't use for**: Specific design tasks (use specialized skills), detailed implementation guidance

---

## Core UX Principles

### 1. User-Centered Design

**Definition**: Design decisions driven by user needs, not business/technical constraints

**Key Concepts**:
- **Empathy**: Understand users' context, goals, frustrations
- **Iteration**: Design → Test → Refine → Repeat
- **Evidence-based**: Use research data, not assumptions

**In Practice**:
- Start with user research (interviews, observations)
- Test designs with real users early and often
- Prioritize user goals over feature lists

**Why it matters**: Designs that ignore users fail, regardless of technical excellence

---

### 2. Progressive Disclosure

**Definition**: Show only what users need now, reveal complexity gradually

**Key Concepts**:
- **Layers**: Basic → Intermediate → Advanced
- **Just-in-time**: Information appears when needed
- **Defaults**: Sensible defaults for common cases

**In Practice**:
- Simple interface by default, "Advanced" button for power features
- Tooltips on hover, not permanent labels everywhere
- Wizards guide through complex multi-step flows

**Why it matters**: Showing everything at once = cognitive overload

---

### 3. Affordances

**Definition**: Visual/physical properties that suggest how something is used

**Key Concepts**:
- **Perceived affordance**: Looks clickable/tappable (buttons raised, links colored)
- **Real affordance**: Actually is clickable/tappable
- **Signifiers**: Clues about functionality (icons, labels, shadows)

**In Practice**:
- Buttons look different from text (visual weight, borders)
- Links are colored and/or underlined
- Interactive elements change on hover (cursor, highlight)

**Why it matters**: Users shouldn't guess what's interactive

---

### 4. Feedback

**Definition**: System responds to user actions immediately and clearly

**Key Concepts**:
- **Immediate**: <100ms for most interactions (button press)
- **Appropriate**: Match feedback to action (subtle for minor, prominent for major)
- **Informative**: What happened? Success/failure? Next steps?

**In Practice**:
- Button depresses on click
- Loading spinner for async operations (>1s delay)
- Success messages confirm completion
- Error messages explain what went wrong + how to fix

**Why it matters**: No feedback = user repeats action = bugs/confusion

---

### 5. Consistency

**Definition**: Similar elements behave similarly throughout interface

**Key Concepts**:
- **Internal consistency**: Within your app (same patterns everywhere)
- **External consistency**: Match platform conventions (iOS vs Android)
- **Functional consistency**: Same action has same result

**In Practice**:
- All primary buttons same color/style
- Navigation pattern consistent across screens
- Keyboard shortcuts work the same everywhere

**Why it matters**: Inconsistency forces users to relearn patterns

---

### 6. Visibility of System Status

**Definition**: Users always know what's happening

**Key Concepts**:
- **Current state**: Where am I? (active nav item highlighted)
- **Process**: What's happening? (loading spinner, progress bar)
- **Outcome**: What happened? (success/error message)

**In Practice**:
- Active tab/page highlighted
- Progress indicators for multi-step flows (step 2 of 5)
- Breadcrumbs show navigation path
- Form validation shows which fields have errors

**Why it matters**: Users lost in interface = frustration

---

### 7. Error Prevention & Recovery

**Definition**: Design to prevent errors, make recovery easy when they happen

**Key Concepts**:
- **Prevention**: Constraints, validation, confirmations
- **Clear errors**: What went wrong? How to fix?
- **Undo**: Easy to reverse mistakes

**In Practice**:
- Disable "Submit" until form valid
- Confirm before destructive actions ("Delete all?")
- Undo/redo for content changes
- Auto-save to prevent lost work

**Why it matters**: Errors are frustrating, especially if unrecoverable

---

### 8. Recognition Over Recall

**Definition**: Minimize memory load by making information visible

**Key Concepts**:
- **Show, don't hide**: Visible options > memorized commands
- **Contextual help**: Information available when needed
- **Autocomplete**: Suggest options rather than require exact input

**In Practice**:
- Menus show available actions (vs. command-line memorization)
- Recently used items listed (vs. remembering file names)
- Autocomplete in search (vs. typing exact query)
- Date pickers (vs. typing date format)

**Why it matters**: Memory is fragile, recognition is easier

---

### 9. Flexibility & Efficiency

**Definition**: Support both novice and expert users

**Key Concepts**:
- **Novice path**: Simple, guided, obvious
- **Expert path**: Shortcuts, customization, speed
- **Progressive disclosure**: Novices aren't overwhelmed, experts aren't constrained

**In Practice**:
- Keyboard shortcuts for power users (Cmd+S)
- Command palettes (fuzzy search all actions)
- Customizable toolbars/workspaces
- Both clicking and keyboard navigation work

**Why it matters**: Users grow from novice → expert over time

---

### 10. Aesthetic & Minimalist Design

**Definition**: Remove unnecessary elements, focus on essentials

**Key Concepts**:
- **Signal vs. noise**: Every element competes for attention
- **White space**: Breathing room improves readability
- **Hierarchy**: Guide attention to important elements

**In Practice**:
- Remove decorative elements that don't serve purpose
- Use white space to group related items
- Single clear primary action per screen
- Reduce visual clutter (colors, borders, shadows)

**Why it matters**: Too much information = user misses important parts

---

## UX Terminology

### Mental Models

**Definition**: User's internal understanding of how system works

**Why it matters**: Design must match user's mental model, not system architecture

**Example**:
- Good: File structure matches user's mental organization (Projects → Project Name → Files)
- Bad: File structure matches database schema (table_id → foreign_key → blob_id)

---

### Information Architecture (IA)

**Definition**: Organizing and labeling content for findability

**Related skill**: `lyra/ux-designer/information-architecture`

**Key aspects**: Navigation structure, labeling, categorization, search

---

### Interaction Design

**Definition**: Defining how users interact with system (clicks, gestures, feedback)

**Related skill**: `lyra/ux-designer/interaction-design-patterns`

**Key aspects**: Touch targets, micro-interactions, state changes, animations

---

### Visual Hierarchy

**Definition**: Arranging elements to guide attention in order of importance

**Related skill**: `lyra/ux-designer/visual-design-foundations`

**Key aspects**: Size, color, contrast, spacing, typography

---

### Accessibility

**Definition**: Designing for people with disabilities (visual, motor, cognitive)

**Related skill**: `lyra/ux-designer/accessibility-and-inclusive-design`

**Key aspects**: WCAG compliance, screen readers, keyboard navigation, color contrast

---

### Usability

**Definition**: How easy and efficient it is to use a system

**Measured by**: Task success rate, time on task, error rate, satisfaction

**Related skill**: `lyra/ux-designer/user-research-and-validation` (testing)

---

### User Journey / User Flow

**Definition**: Path user takes to accomplish a goal

**Includes**: Entry point → Steps → Decision points → Outcome

**Related skill**: `lyra/ux-designer/user-research-and-validation` (journey mapping)

---

### Personas

**Definition**: Fictional characters representing user segments

**Includes**: Demographics, goals, behaviors, pain points, context

**Related skill**: `lyra/ux-designer/user-research-and-validation` (research methods)

---

### Wireframe

**Definition**: Low-fidelity layout sketch (structure, not visual design)

**Purpose**: Test IA and layout before investing in visual design

**Related skills**: `information-architecture`, `visual-design-foundations`

---

### Prototype

**Definition**: Interactive simulation of design (clickable, navigable)

**Purpose**: Test flows and interactions before development

**Types**: Low-fidelity (Balsamiq), high-fidelity (Figma with interactions)

---

### Heuristic Evaluation

**Definition**: Expert review against usability principles (heuristics)

**Common framework**: Nielsen's 10 Usability Heuristics

**Related skill**: `lyra/ux-designer/user-research-and-validation` (validation methods)

---

## Design Thinking Process

### 1. Empathize (Understand Users)

**Goal**: Deep understanding of user needs, context, pain points

**Methods**: Interviews, observations, diary studies

**Related skill**: `lyra/ux-designer/user-research-and-validation`

**Output**: User insights, pain points, opportunity areas

---

### 2. Define (Frame the Problem)

**Goal**: Synthesize research into clear problem statement

**Methods**: Affinity mapping, personas, journey maps

**Output**: Problem statement, design principles, success criteria

---

### 3. Ideate (Generate Solutions)

**Goal**: Explore many possible solutions without judgment

**Methods**: Brainstorming, sketching, co-design workshops

**Output**: Diverse design concepts

---

### 4. Prototype (Build to Test)

**Goal**: Make ideas tangible for testing

**Methods**: Wireframes, mockups, clickable prototypes

**Related skills**: All design skills (visual, IA, interaction)

**Output**: Testable prototypes (lo-fi or hi-fi)

---

### 5. Test (Validate with Users)

**Goal**: Learn what works, what doesn't, iterate

**Methods**: Usability testing, A/B testing, analytics

**Related skill**: `lyra/ux-designer/user-research-and-validation`

**Output**: Insights for iteration, validated design decisions

---

### 6. Iterate (Refine & Improve)

**Goal**: Continuously improve based on feedback

**Process**: Implement → Measure → Learn → Refine

**Output**: Improved design, new insights

---

## When to Use Each Lyra Skill

### lyra/ux-designer/using-ux-designer (Meta)

**When**: Starting any UX task, unsure which skill to load

**Purpose**: Routes to appropriate skills based on context

---

### lyra/ux-designer/ux-fundamentals (This Skill)

**When**: "What is...?", "Explain...", learning UX concepts

**Purpose**: Teaching and foundational knowledge

---

### lyra/ux-designer/visual-design-foundations

**When**: Color, typography, hierarchy, spacing, contrast issues

**Purpose**: Systematic framework for visual design evaluation

**Framework**: 6-dimension Visual Hierarchy Analysis (Contrast, Scale, Spacing, Color, Typography, Layout Flow)

---

### lyra/ux-designer/information-architecture

**When**: Navigation confusing, content organization, findability issues

**Purpose**: Structure content for discoverability

**Framework**: 4-layer Navigation & Discoverability Model (Mental Models, Navigation Systems, Information Scent, Discoverability)

---

### lyra/ux-designer/interaction-design-patterns

**When**: Touch targets, feedback, micro-interactions, button states

**Purpose**: Design clear, responsive interactions

**Framework**: 5-dimension Interaction Clarity Framework (Affordances, Feedback, Micro-interactions, State Changes, Touch Targets)

---

### lyra/ux-designer/accessibility-and-inclusive-design

**When**: WCAG compliance, colorblind-safe, keyboard nav, screen readers

**Purpose**: Ensure design works for everyone

**Framework**: 6-dimension Universal Access Model (Visual, Motor, Cognitive, Screen Reader, Temporal, Situational)

---

### lyra/ux-designer/user-research-and-validation

**When**: Need to understand users, test designs, validate decisions

**Purpose**: Research methods and usability testing

**Framework**: 5-phase User Understanding Model (Discovery, Generative, Evaluative, Validation, Post-Launch)

---

### lyra/ux-designer/mobile-design-patterns

**When**: iOS/Android app design, touch interactions

**Purpose**: Mobile-specific patterns and constraints

**Framework**: Mobile Interaction Evaluation Model (Reachability, Gesture Conventions, Platform Consistency, Performance Perception)

---

### lyra/ux-designer/web-application-design

**When**: Web app, dashboard, SaaS, data visualization

**Purpose**: Web-specific patterns (responsive, complex data)

**Framework**: Web Application Usability Framework (Data Clarity, Workflow Efficiency, Responsive Adaptation, Progressive Enhancement)

---

### lyra/ux-designer/desktop-software-design

**When**: Desktop app, Electron, multi-window, keyboard-first

**Purpose**: Desktop-specific patterns (windows, shortcuts)

**Framework**: Desktop Application Workflow Model (Window Organization, Keyboard Efficiency, Workspace Customization, Expert Paths)

---

### lyra/ux-designer/game-ui-design

**When**: Game, HUD, menu system, in-game interface

**Purpose**: Game-specific patterns (immersion, performance)

**Framework**: Game UI Integration Framework (Visibility vs Immersion, Input Method Optimization, Aesthetic Coherence, Performance Impact)

---

## Common UX Questions Answered

### "How do I know if my design is good?"

**Answer**: Test with users. Good design = users accomplish goals efficiently and satisfactorily.

**Objective measures**: Task success rate, time on task, error rate
**Subjective measures**: User satisfaction, perceived ease of use

**Related skill**: `user-research-and-validation`

---

### "Should I follow platform conventions (iOS HIG, Material Design)?"

**Answer**: Yes, with rare exceptions.

**Why follow conventions**:
- Users already know the patterns (less learning curve)
- Accessibility features work as expected
- Platform integration (gestures, navigation)

**When to deviate**:
- Strong brand identity requires custom patterns
- Unique use case not covered by conventions (games, creative tools)
- Cross-platform consistency is more important

**Related skill**: Platform extensions (mobile, web, desktop, game)

---

### "Design for mobile-first or desktop-first?"

**Answer**: Depends on your users' primary context.

**Mobile-first**: When most users on mobile, forces prioritization
**Desktop-first**: When desktop is primary, mobile is secondary

**Best practice**: Design for primary platform, adapt to others

**Related skills**: `mobile-design-patterns`, `web-application-design`

---

### "How do I balance aesthetics and usability?"

**Answer**: They're not in conflict. Good aesthetics enhance usability.

**Aesthetic benefits**:
- Visual hierarchy guides attention (usability)
- Consistent visual system reduces cognitive load
- Pleasant aesthetics increase trust and engagement

**Warning signs of conflict**:
- Low contrast text (aesthetic > accessibility = BAD)
- Hiding essential functions for "clean" look = BAD
- Decorative elements obscuring content = BAD

**Related skills**: `visual-design-foundations`, `accessibility-and-inclusive-design`

---

### "Should I use modals/popups?"

**Answer**: Sparingly. Modals interrupt users.

**When appropriate**:
- Requires user decision before proceeding (confirm destructive action)
- Focus needed (entering password, selecting from list)
- Contextual detail (lightbox for image, detail view)

**Alternatives**:
- Inline expansion (accordion, show/hide)
- Separate page (if lots of content)
- Toast notification (non-blocking feedback)

**Related skills**: `interaction-design-patterns`, platform extensions

---

### "How many clicks/taps is too many?"

**Answer**: 3-click rule is guideline, not law. Clarity > click count.

**What matters**:
- **Information scent**: Each step has clear clues to next
- **No dead ends**: Users can always proceed or go back
- **Progress visible**: Users know where they are in flow

**10 clear clicks > 2 confusing clicks**

**Related skill**: `information-architecture`

---

### "Should I use animation?"

**Answer**: Yes, purposefully. Animation guides attention and shows relationships.

**Good uses**:
- State transitions (show what changed)
- Drawing attention (new message, error)
- Showing relationships (element moves from A to B)

**Bad uses**:
- Gratuitous decoration (slows users down)
- Too slow (>500ms feels sluggish)
- Motion sickness triggers (parallax, constant movement)

**Related skill**: `interaction-design-patterns`

---

### "How do I design for accessibility without compromising design?"

**Answer**: Accessibility constraints make design BETTER, not worse.

**Examples**:
- High contrast = more readable for everyone
- Large touch targets = easier for everyone
- Clear labels = less confusion for everyone
- Keyboard navigation = faster for power users

**Accessibility IS good design.**

**Related skill**: `accessibility-and-inclusive-design`

---

## Related Skills

**All Lyra UX Designer skills**: This teaching skill references concepts explained in detail by specialized skills

**Cross-faction**:
- `muna/technical-writer/clarity-and-style` - Writing clear UI copy
- `ordis/security-architect/threat-modeling` - Security implications of UX decisions

---

## Further Learning

**Books**:
- "Don't Make Me Think" (Steve Krug) - Web usability fundamentals
- "The Design of Everyday Things" (Don Norman) - Affordances, mental models
- "About Face" (Alan Cooper) - Interaction design principles

**Frameworks**:
- Nielsen's 10 Usability Heuristics
- WCAG 2.1 Guidelines (accessibility)
- iOS Human Interface Guidelines
- Material Design Guidelines

**Practice**:
- Critique existing interfaces (what works? what doesn't?)
- Apply frameworks from other Lyra skills
- Test designs with real users (even 5 users find major issues)
