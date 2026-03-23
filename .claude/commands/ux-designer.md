
# Using UX Designer

## Overview

This meta-skill routes you to the right UX design skills based on your situation. Load this skill when you need UX expertise but aren't sure which specific skill to use.

**Core Principle**: Different UX tasks require different skills. Match your situation to the appropriate skill, load only what you need.

## When to Use

Load this skill when:
- Starting any UX/UI design task
- User mentions: "design", "UX", "UI", "interface", "user experience", "layout", "navigation"
- You need to critique or review a design
- You need to create a new interface or feature
- User asks about UX principles or concepts

**Don't use for**: Backend logic, database design, pure technical implementation without UX implications


## Routing by Situation

### Learning & Explanation

**Symptoms**: "What is...", "Explain...", "Teach me about...", "How does X work in UX?"

**Route to**: `lyra/ux-designer/ux-fundamentals`

**Examples**:
- "What is information architecture?" → ux-fundamentals
- "Explain visual hierarchy" → ux-fundamentals
- "How do I think about accessibility?" → ux-fundamentals


### Design Critique & Review

**Symptoms**: "Review this design", "Critique this interface", "Is this usable?", "Does this follow best practices?"

**Route to**: Relevant competency skills based on critique focus

**General Review** (no specific focus):
- `lyra/ux-designer/visual-design-foundations` (visual hierarchy, color, typography)
- `lyra/ux-designer/information-architecture` (content organization, navigation)
- `lyra/ux-designer/accessibility-and-inclusive-design` (WCAG, inclusive design)

**Specific Focus**:
- Visual issues (color, contrast, hierarchy) → `visual-design-foundations`
- Navigation/findability issues → `information-architecture`
- Interaction feedback, touch targets → `interaction-design-patterns`
- Accessibility concerns → `accessibility-and-inclusive-design`

**Add platform extension** if design is platform-specific:
- Mobile app → Add `mobile-design-patterns`
- Web dashboard → Add `web-application-design`
- Desktop software → Add `desktop-software-design`
- Game interface → Add `game-ui-design`


### New Interface Design

**Symptoms**: "Design a...", "Create interface for...", "Build a [feature] screen"

**Route to**: Competency skills + platform extension

**Standard Web/Mobile Feature**:
1. `lyra/ux-designer/visual-design-foundations` (layout, hierarchy, color)
2. `lyra/ux-designer/interaction-design-patterns` (buttons, feedback, states)
3. Platform-specific:
   - Mobile → `mobile-design-patterns`
   - Web app → `web-application-design`

**Complex Navigation/IA**:
1. `lyra/ux-designer/information-architecture` (content structure, nav systems)
2. `lyra/ux-designer/visual-design-foundations` (visual hierarchy)
3. Platform extension as needed

**Research Phase** (early discovery):
1. `lyra/ux-designer/user-research-and-validation` (understand users first)
2. Then return to design skills once research complete


### Specific UX Domains

#### Visual Design Issues

**Symptoms**: "Colors don't work", "Typography feels off", "Hierarchy unclear", "Layout cramped"

**Route to**: `lyra/ux-designer/visual-design-foundations`

**Add**: `accessibility-and-inclusive-design` if contrast/readability concerns


#### Navigation & Findability

**Symptoms**: "Users can't find features", "Navigation confusing", "Menu structure", "Content organization"

**Route to**: `lyra/ux-designer/information-architecture`

**Add**: Platform extension for platform-specific nav patterns


#### Interaction & Feedback

**Symptoms**: "Button states unclear", "No loading feedback", "Micro-interactions", "Touch targets too small"

**Route to**: `lyra/ux-designer/interaction-design-patterns`

**Add**: Platform extension for platform-specific interaction conventions


#### Accessibility & Inclusion

**Symptoms**: "WCAG compliance", "Accessibility audit", "Colorblind-safe", "Keyboard navigation", "Screen reader"

**Route to**: `lyra/ux-designer/accessibility-and-inclusive-design`

**Note**: This skill should be referenced by all other design decisions (accessibility is universal)


#### User Research & Validation

**Symptoms**: "Understand users", "User interviews", "Usability testing", "Mental models", "Journey mapping"

**Route to**: `lyra/ux-designer/user-research-and-validation`

**Add**: Other skills once research informs design direction


## Platform-Specific Routing

### Mobile (iOS/Android)

**Symptoms**: "Mobile app", "iOS", "Android", "Touch interface", "Phone", "Tablet"

**Route to**:
- Core competency skills (visual, IA, interaction) as needed
- **Always add**: `lyra/ux-designer/mobile-design-patterns`

**Mobile-Specific Concerns**:
- Touch targets (44x44pt iOS, 48x48dp Android)
- Gestures (swipe, pinch, long-press)
- Platform conventions (iOS HIG vs Material Design)
- One-handed use, thumb zones


### Web Applications

**Symptoms**: "Web app", "Dashboard", "SaaS", "Data visualization", "Admin panel", "Responsive design"

**Route to**:
- Core competency skills as needed
- **Always add**: `lyra/ux-designer/web-application-design`

**Web-Specific Concerns**:
- Responsive breakpoints
- Complex data display (tables, charts)
- Keyboard shortcuts, power-user workflows
- Multi-tasking (tabs, split views)


### Desktop Software

**Symptoms**: "Desktop app", "Electron", "Native application", "Multi-window", "Keyboard shortcuts"

**Route to**:
- Core competency skills as needed
- **Always add**: `lyra/ux-designer/desktop-software-design`

**Desktop-Specific Concerns**:
- Window management (multi-window, panels)
- Keyboard-first workflows
- Workspace customization
- Power-user features (preferences, scripting)


### Game UI

**Symptoms**: "Game", "HUD", "Menu system", "Game interface", "In-game UI", "Player experience"

**Route to**:
- Core competency skills as needed
- **Always add**: `lyra/ux-designer/game-ui-design`

**Game-Specific Concerns**:
- Visibility vs immersion (diegetic UI)
- Controller/gamepad navigation
- Readability during action
- Performance impact (frame rate)


## Multi-Skill Scenarios

### Complete Feature Design (Mobile Login)

**Load in order**:
1. `visual-design-foundations` (layout, button hierarchy)
2. `interaction-design-patterns` (form feedback, button states)
3. `accessibility-and-inclusive-design` (form labels, contrast)
4. `mobile-design-patterns` (touch targets, platform conventions)


### Dashboard Redesign (Web)

**Load in order**:
1. `information-architecture` (organize data, navigation)
2. `visual-design-foundations` (hierarchy, chart design)
3. `web-application-design` (responsive, data display patterns)
4. `accessibility-and-inclusive-design` (data table accessibility)


### Game HUD Evaluation

**Load in order**:
1. `visual-design-foundations` (readability, contrast)
2. `game-ui-design` (immersion, performance, input method)
3. `accessibility-and-inclusive-design` (colorblind-safe indicators)


## Cross-Faction Integration

### Lyra + Muna (Technical Writer)

**When designing documentation UX**:
- `lyra/ux-designer/information-architecture` (organize docs)
- `muna/technical-writer/documentation-structure` (content structure)
- `muna/technical-writer/clarity-and-style` (microcopy, UI text)

**Example**: "Design documentation site navigation" → Load IA + documentation-structure


### Lyra + Ordis (Security Architect)

**When designing secure interfaces**:
- `lyra/ux-designer/visual-design-foundations` (secure feedback, error states)
- `ordis/security-architect/threat-modeling` (authentication UX threats)

**Example**: "Design login with MFA" → Load interaction-patterns + threat-modeling


## Decision Tree

```
User Request
    |
    ├─ "What is...?" / "Explain..." → ux-fundamentals
    |
    ├─ "Review this design"
    |   ├─ General → visual-design + IA + accessibility
    |   └─ Specific concern → Relevant competency skill
    |       └─ Add platform extension if platform-specific
    |
    ├─ "Design a [feature]"
    |   ├─ Research phase? → user-research-and-validation first
    |   └─ Design phase
    |       ├─ Identify competencies needed (visual, IA, interaction)
    |       ├─ Detect platform (mobile, web, desktop, game)
    |       └─ Load competency + platform extension
    |
    └─ Specific domain
        ├─ Visual → visual-design-foundations
        ├─ Navigation → information-architecture
        ├─ Interaction → interaction-design-patterns
        ├─ Accessibility → accessibility-and-inclusive-design
        └─ Research → user-research-and-validation
```


## Common Patterns

### Pattern 1: "I need general UX advice"
**Load**: `ux-fundamentals` (teaches principles)

### Pattern 2: "Critique my [platform] design"
**Load**: visual-design + IA + accessibility + [platform-extension]

### Pattern 3: "Design [feature] for [platform]"
**Load**: Relevant competencies + [platform-extension]

### Pattern 4: "Is this accessible?"
**Load**: `accessibility-and-inclusive-design` (primary)
**Reference**: visual-design (contrast), interaction-design (keyboard nav)

### Pattern 5: "How do users navigate this?"
**Load**: `information-architecture` (primary)
**Add**: user-research-and-validation (if testing/validation needed)


## Benefits of Routing

**Focused expertise**: Load only what's needed for the task
**Clear boundaries**: Each skill has distinct responsibility
**Composable**: Combine skills for complex scenarios
**Efficient**: Avoid loading all 11 skills at once
**Explicit**: User sees which skills are active


## Related Skills

**All Lyra UX Designer skills**:
- `lyra/ux-designer/ux-fundamentals` - Teaching and principles
- `lyra/ux-designer/visual-design-foundations` - Color, typography, hierarchy
- `lyra/ux-designer/information-architecture` - Navigation, content organization
- `lyra/ux-designer/interaction-design-patterns` - Feedback, micro-interactions
- `lyra/ux-designer/accessibility-and-inclusive-design` - Inclusive design, WCAG
- `lyra/ux-designer/user-research-and-validation` - Research methods, testing
- `lyra/ux-designer/mobile-design-patterns` - iOS/Android patterns
- `lyra/ux-designer/web-application-design` - Web app patterns
- `lyra/ux-designer/desktop-software-design` - Desktop app patterns
- `lyra/ux-designer/game-ui-design` - Game interface patterns

**Cross-faction**:
- `muna/technical-writer/*` - Documentation UX and microcopy
- `ordis/security-architect/*` - Security-aware interface design
