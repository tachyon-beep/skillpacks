
# Accessibility and Inclusive Design

## Overview

This skill provides a systematic framework for creating universally accessible interfaces that work for everyone, regardless of ability, situation, or context.

**Core Principle**: Accessibility constraints make design BETTER for everyone, not just users with disabilities. High contrast helps in sunlight. Large touch targets help when distracted. Clear language helps when stressed.

**Framework**: The Universal Access Model - 6 dimensions ensuring comprehensive accessibility coverage.

## When to Use

Load this skill when:
- User mentions "accessibility", "WCAG", "inclusive design", "screen reader"
- Designing or critiquing any interface (accessibility is universal)
- User asks about contrast ratios, colorblind-safe design, keyboard navigation
- Compliance required (legal, ethical, best practices)
- Evaluating existing designs for accessibility issues

**Don't use for**: Pure visual aesthetics (use visual-design-foundations first, then verify with this skill)

**Critical Note**: This skill should be referenced by ALL other Lyra skills. Accessibility is not optional or separate - it's integrated into every design decision.


## The Universal Access Model

A systematic 6-dimension framework for evaluating and ensuring universal access:

1. **Visual Accessibility** - Contrast, zooming, color-independent information
2. **Motor Accessibility** - Touch targets, keyboard navigation, no precision required
3. **Cognitive Accessibility** - Clear language, reduced mental load, forgiving
4. **Screen Reader Compatibility** - Semantic structure, labels, logical order
5. **Temporal Accessibility** - No timeouts, pauseable content, user-paced
6. **Situational Accessibility** - Works in varied contexts (noise, sunlight, one-handed)

Each dimension has specific WCAG criteria, testing methods, and patterns.


### Dimension 1: VISUAL ACCESSIBILITY

**Purpose**: Users with vision differences (low vision, colorblindness, blindness) can access content

#### Evaluation Questions

**Contrast:**
- Does text meet 4.5:1 contrast ratio (WCAG AA)?
- Does large text (18pt+) meet 3:1 ratio?
- Do UI components meet 3:1 contrast against backgrounds?
- Is contrast even higher for critical actions (7:1 for AAA)?

**Zoom & Resize:**
- Can users zoom to 200% without horizontal scrolling?
- Does text remain readable when zoomed?
- Do layouts reflow gracefully at high zoom levels?
- Are font sizes relative (em, rem) not fixed (px)?

**Color Independence:**
- Is information conveyed with more than color alone?
- Can colorblind users distinguish states (red/green)?
- Are links distinguishable without relying on color?
- Do charts/graphs use patterns in addition to color?

**Readability:**
- Is body text at least 16px on mobile, 14px on desktop?
- Is line height comfortable (1.5x for body text)?
- Is line length appropriate (45-75 characters)?
- Are fonts clear and legible (avoid decorative for body)?

#### WCAG 2.1 AA Requirements

**1.4.3 Contrast (Minimum) - Level AA:**
- Normal text: 4.5:1 minimum
- Large text (18pt/14pt bold+): 3:1 minimum
- UI components and graphics: 3:1 minimum

**1.4.4 Resize Text - Level AA:**
- Text can be resized up to 200% without loss of content or functionality

**1.4.11 Non-text Contrast - Level AA:**
- UI components and graphical objects: 3:1 minimum against adjacent colors

**1.4.1 Use of Color - Level A:**
- Color is not the only visual means of conveying information

#### Patterns (Good)

**High Contrast Text:**
```
Body text: #000000 on #FFFFFF (21:1) ✓
Links: #0066CC on #FFFFFF (8.6:1) ✓
Large headings: #333333 on #FFFFFF (12.6:1) ✓
```

**Color + Icon/Text Indicators:**
```
Error: Red background + "X" icon + "Error" text
Success: Green background + checkmark icon + "Success" text
Warning: Yellow background + "!" icon + "Warning" text
(Colorblind users see icon/text, not just color)
```

**Link Differentiation:**
```
Links: Blue (#0066CC) + underline (always visible)
Or: Different font weight + underline on hover
Not: Just color change (insufficient for colorblind)
```

**Zoom-Friendly Layouts:**
```
Relative units: font-size: 1rem (16px base)
Viewport units: max-width: 90vw
Flexible grids: display: flex with wrap
Not: Fixed pixel widths everywhere
```

#### Anti-Patterns (Problematic)

**Low Contrast:**
```
Light gray on white: #999999 on #FFFFFF (2.8:1) ✗
Fails WCAG AA (needs 4.5:1)
```

**Color as Sole Indicator:**
```
"Red items are errors, green items are valid"
Colorblind users can't distinguish ✗
Add icons/text to each state ✓
```

**Fixed Font Sizes:**
```
font-size: 12px; (user can't resize)
Use: font-size: 0.75rem; (scales with user preference)
```

**Tiny Text:**
```
Body text at 12px on mobile (too small) ✗
Minimum 16px on mobile, 14px on desktop ✓
```

**Insufficient Line Height:**
```
line-height: 1.0; (cramped, hard to read) ✗
line-height: 1.5; (comfortable spacing) ✓
```

#### Testing Methods

**Contrast Tools:**
- WebAIM Contrast Checker: https://webaim.org/resources/contrastchecker/
- Stark plugin (Figma/Sketch): Real-time contrast checking
- Chrome DevTools: Built-in contrast ratio in color picker
- Accessible Colors: Suggests accessible alternatives

**Colorblindness Simulation:**
- Stark (Figma): Protanopia, Deuteranopia, Tritanopia filters
- Colorblinding Chrome extension
- Sim Daltonism (macOS): System-wide colorblind filters
- Test with Protanopia (red-blind) and Deuteranopia (green-blind) at minimum

**Zoom Testing:**
- Browser zoom to 200% (Cmd/Ctrl + Plus)
- Check for horizontal scrolling (bad)
- Check for overlapping content (bad)
- Check for cut-off text (bad)

**Visual Testing Checklist:**
- [ ] All text meets 4.5:1 contrast (WebAIM)
- [ ] UI components meet 3:1 contrast
- [ ] Links distinguishable without color
- [ ] Zoom to 200% works without horizontal scroll
- [ ] Colorblind simulation passes (Stark)
- [ ] Body text minimum 16px mobile, 14px desktop


### Dimension 2: MOTOR ACCESSIBILITY

**Purpose**: Users with motor control differences (tremors, limited dexterity, paralysis) can interact

#### Evaluation Questions

**Touch Targets:**
- Are all interactive elements at least 44x44px (iOS) or 48x48dp (Android)?
- Is spacing adequate between tap targets (8px minimum)?
- Can users with tremors hit targets reliably?
- Are critical actions away from screen edges (accidental activation)?

**Keyboard Navigation:**
- Can users navigate entire interface with keyboard only?
- Is tab order logical (top-to-bottom, left-to-right)?
- Are focus indicators visible (2px outline minimum)?
- Can users activate all functions via keyboard?
- Are keyboard shortcuts provided for common actions?

**No Precision Required:**
- Can users interact without fine motor control?
- Are drag-and-drop operations optional (alternative method)?
- Are hover-only interactions avoided?
- Can users pause/cancel actions (no irreversible quick gestures)?

#### WCAG 2.1 AA Requirements

**2.1.1 Keyboard - Level A:**
- All functionality available via keyboard interface

**2.1.2 No Keyboard Trap - Level A:**
- Focus can move away from component using keyboard alone

**2.4.7 Focus Visible - Level AA:**
- Keyboard focus indicator is visible

**2.5.5 Target Size - Level AAA (Best Practice):**
- Touch targets at least 44x44 CSS pixels (iOS guideline, WCAG AAA)

#### Patterns (Good)

**Large Touch Targets:**
```
Buttons: min-height: 48px, min-width: 48px
Icons: 44x44px clickable area (icon may be smaller, hit area large)
Form inputs: height: 48px
List items: min-height: 56dp (Android Material)
```

**Adequate Spacing:**
```
Margin between buttons: 8px minimum
Padding inside buttons: 12px vertical, 24px horizontal
Space between form fields: 16px minimum
```

**Visible Focus Indicators:**
```
button:focus {
  outline: 2px solid #0066CC;
  outline-offset: 2px;
}

Do not: outline: none; (removes accessibility) ✗
```

**Keyboard Shortcuts:**
```
Common actions:
- Tab: Next element
- Shift+Tab: Previous element
- Enter/Space: Activate button/link
- Esc: Close modal/dismiss
- Arrow keys: Navigate menus/lists

Custom shortcuts:
- Cmd/Ctrl+S: Save
- Cmd/Ctrl+Z: Undo
- /: Focus search (common web pattern)
```

**Skip Links:**
```
<a href="#main-content" class="skip-link">
  Skip to main content
</a>

Allows keyboard users to skip repetitive navigation
```

#### Anti-Patterns (Problematic)

**Tiny Touch Targets:**
```
Buttons: 30x30px (too small, frustrating) ✗
Minimum 44x44px (iOS), 48x48dp (Android) ✓
```

**Crowded Tap Targets:**
```
Buttons with 2px spacing (accidental taps) ✗
Minimum 8px spacing between interactive elements ✓
```

**No Focus Indicator:**
```
*:focus { outline: none; } ✗
Keyboard users can't see where they are
Keep default or enhance: outline: 2px solid blue; ✓
```

**Hover-Only Interactions:**
```
Dropdown menu appears only on hover (keyboard inaccessible) ✗
Add keyboard trigger: click or Enter key ✓
```

**Illogical Tab Order:**
```
HTML order: Header → Sidebar → Content
Visual order: Header → Content → Sidebar
Tab order follows HTML (confusing) ✗
Reorder HTML or use tabindex carefully ✓
```

**Mouse-Only Actions:**
```
Right-click context menus only (no keyboard equivalent) ✗
Provide button or keyboard shortcut alternative ✓
```

#### Testing Methods

**Keyboard Navigation Test:**
1. Unplug/hide your mouse
2. Navigate entire interface with Tab, Shift+Tab, Enter, Space, Esc, Arrows
3. Check:
   - Can you reach all interactive elements?
   - Is focus indicator always visible?
   - Is tab order logical?
   - Can you activate all functions?
   - Can you escape modals/menus?

**Touch Target Testing:**
- Use browser DevTools to measure elements (should be 44x44px minimum)
- Test on actual mobile device (finger test)
- Try tapping with thumb while holding phone (reachability)

**Motor Accessibility Checklist:**
- [ ] All touch targets 44x44px minimum
- [ ] 8px spacing between interactive elements
- [ ] Entire interface navigable via keyboard only
- [ ] Focus indicators visible on all interactive elements
- [ ] Tab order is logical
- [ ] No hover-only critical interactions
- [ ] Keyboard shortcuts for common actions


### Dimension 3: COGNITIVE ACCESSIBILITY

**Purpose**: Users with cognitive differences (dyslexia, ADHD, anxiety, memory issues) can understand

#### Evaluation Questions

**Language Clarity:**
- Is language simple and clear (8th grade reading level)?
- Are sentences short (20 words or less)?
- Is jargon avoided or explained?
- Are instructions direct and actionable?

**Cognitive Load:**
- Is information chunked (5-7 items per group)?
- Are headings and lists used to organize content?
- Is progressive disclosure used (simple first, advanced later)?
- Are users required to remember information across screens?

**Error Prevention & Recovery:**
- Are errors prevented with constraints and validation?
- Are error messages clear and helpful?
- Can users easily undo mistakes?
- Are confirmations provided for destructive actions?

**Visual Clarity:**
- Is layout consistent across screens?
- Are visual cues provided (icons, color, spacing)?
- Is there adequate white space (not overwhelming)?
- Are animations purposeful (not distracting)?

#### WCAG 2.1 AA Requirements

**3.1.5 Reading Level - Level AAA (Best Practice):**
- Text does not require reading ability more advanced than lower secondary education

**3.2.3 Consistent Navigation - Level AA:**
- Navigational mechanisms that are repeated occur in same order

**3.2.4 Consistent Identification - Level AA:**
- Components with same functionality are identified consistently

**3.3.1 Error Identification - Level A:**
- Errors are identified and described to user in text

**3.3.2 Labels or Instructions - Level A:**
- Labels or instructions provided when content requires user input

**3.3.3 Error Suggestion - Level AA:**
- Suggestions provided when input error is automatically detected

**3.3.4 Error Prevention (Legal, Financial, Data) - Level AA:**
- Submissions are reversible, checked, or confirmed

#### Patterns (Good)

**Clear Language:**
```
Good: "Enter your email address"
Bad: "Input your electronic mail identifier"

Good: "Delete this file?"
Bad: "Permanently remove this resource from the filesystem?"

Target 8th grade reading level (Hemingway Editor)
```

**Chunked Information:**
```
Group related fields:
┌─────────────────────┐
│ Contact Information │
│ - Name              │
│ - Email             │
│ - Phone             │
└─────────────────────┘

Not: 15 fields in one long list
```

**Progressive Disclosure:**
```
Default view: Basic settings (3-5 options)
"Advanced" button reveals: Power-user settings
Not: All 30 settings visible at once (overwhelming)
```

**Helpful Error Messages:**
```
Good: "Email format incorrect. Example: name@example.com"
Bad: "Invalid input in field 3"

Good: "Password must contain: 8 characters, 1 number, 1 uppercase"
Bad: "Password requirements not met"
```

**Confirmation for Destructive Actions:**
```
User clicks "Delete Account"
Modal: "Are you sure? This will permanently delete your account and all data. Type 'DELETE' to confirm."
Prevents accidental data loss
```

**Visual Hierarchy:**
```
Use headings, spacing, color to create structure:
- H1: Page title (32px, bold)
- H2: Section headers (24px, bold)
- H3: Subsections (18px, bold)
- Body: Content (16px, regular)
- Spacing: 24px between sections
```

#### Anti-Patterns (Problematic)

**Complex Jargon:**
```
"Utilize the aforementioned methodology" ✗
"Use this method" ✓
```

**Wall of Text:**
```
Paragraph with 200 words, no headings or lists ✗
Break into sections with headings, use bullet lists ✓
```

**Overwhelming Choices:**
```
15 buttons on one screen (decision paralysis) ✗
1 primary action, 1-2 secondary, hide rest in menu ✓
```

**Cryptic Errors:**
```
"Error 4032" (user has no idea what to do) ✗
"Email already in use. Try logging in instead." ✓
```

**No Confirmation for Destructive Actions:**
```
Single click deletes account (disaster) ✗
Confirmation modal with type-to-confirm ✓
```

**Inconsistent Patterns:**
```
"Save" button top-right on page 1, bottom-left on page 2 ✗
Consistent placement across all screens ✓
```

#### Testing Methods

**Readability Testing:**
- Hemingway Editor: Checks reading level (aim for grade 8 or lower)
- Readable.io: Analyzes content complexity
- Read text aloud: If it sounds awkward, rewrite

**Cognitive Load Testing:**
- Count choices per screen (5-7 max before chunking)
- Measure form fields (chunk into logical groups)
- Check memory requirements (can user see info they need to reference?)

**User Testing:**
- Watch users complete tasks (where do they hesitate?)
- Ask users to explain what they think will happen (mental model check)
- Test with users with cognitive disabilities if possible

**Cognitive Accessibility Checklist:**
- [ ] Language at 8th grade level or lower (Hemingway)
- [ ] Sentences short (20 words or less)
- [ ] Information chunked (5-7 items per group)
- [ ] Error messages clear and actionable
- [ ] Destructive actions require confirmation
- [ ] Consistent navigation and patterns
- [ ] Adequate white space and visual structure


### Dimension 4: SCREEN READER COMPATIBILITY

**Purpose**: Users with screen readers (NVDA, JAWS, VoiceOver, TalkBack) can navigate and understand

#### Evaluation Questions

**Semantic HTML:**
- Are proper HTML elements used (button, nav, main, article)?
- Is heading hierarchy logical (h1 → h2 → h3, no skipping)?
- Are landmarks used (header, nav, main, aside, footer)?
- Are lists used for lists (ul, ol, not divs)?

**Alternative Text:**
- Do images have descriptive alt text?
- Are decorative images marked as decorative (alt="")?
- Do icon buttons have text labels or aria-label?
- Are complex images explained (charts, diagrams)?

**Focus Management:**
- Is focus order logical?
- Does focus move to opened modals?
- Is focus trapped in modals (can't escape to background)?
- Does focus return to trigger element when modal closes?

**ARIA Labels:**
- Are ARIA labels used when semantic HTML insufficient?
- Are form inputs properly labeled?
- Are dynamic updates announced (aria-live)?
- Are expanded/collapsed states indicated (aria-expanded)?

#### WCAG 2.1 AA Requirements

**1.1.1 Non-text Content - Level A:**
- Images have text alternatives

**1.3.1 Info and Relationships - Level A:**
- Information, structure, relationships conveyed through markup

**2.4.1 Bypass Blocks - Level A:**
- Mechanism to skip repeated blocks of content

**2.4.6 Headings and Labels - Level AA:**
- Headings and labels describe topic or purpose

**4.1.2 Name, Role, Value - Level A:**
- For all UI components, name and role can be programmatically determined

**4.1.3 Status Messages - Level AA:**
- Status messages can be programmatically determined and announced

#### Patterns (Good)

**Semantic HTML:**
```html
<header>
  <nav aria-label="Main navigation">
    <ul>
      <li><a href="/">Home</a></li>
      <li><a href="/about">About</a></li>
    </ul>
  </nav>
</header>

<main>
  <h1>Page Title</h1>
  <article>
    <h2>Section Heading</h2>
    <p>Content here...</p>
  </article>
</main>

<footer>
  <p>&copy; 2025 Company</p>
</footer>

Not: <div class="header">, <div class="nav">, <div class="button">
```

**Descriptive Alt Text:**
```html
Good:
<img src="chart.png" alt="Line chart showing revenue growth from $1M in 2020 to $5M in 2025">

Bad:
<img src="chart.png" alt="chart">
<img src="chart.png" alt=""> (if informative, needs description)

Decorative:
<img src="decorative-line.png" alt=""> (empty alt for decorative)
```

**Heading Hierarchy:**
```html
<h1>Page Title</h1>
  <h2>Section 1</h2>
    <h3>Subsection 1.1</h3>
    <h3>Subsection 1.2</h3>
  <h2>Section 2</h2>
    <h3>Subsection 2.1</h3>

Not: <h1> → <h3> → <h2> (illogical)
```

**Form Labels:**
```html
<label for="email">Email Address</label>
<input type="email" id="email" name="email" required>

Or with ARIA:
<input type="email" aria-label="Email Address" required>

Not: Placeholder as label (disappears when typing)
```

**Icon Button Labels:**
```html
<button aria-label="Close modal">
  <svg><!-- X icon --></svg>
</button>

Or:
<button>
  <svg aria-hidden="true"><!-- X icon --></svg>
  <span class="visually-hidden">Close modal</span>
</button>
```

**Skip Links:**
```html
<a href="#main-content" class="skip-link">Skip to main content</a>

<!-- Allows screen reader users to skip navigation -->

.skip-link {
  position: absolute;
  left: -9999px;
}
.skip-link:focus {
  left: 0;
  top: 0;
  z-index: 9999;
}
```

**Live Regions:**
```html
<div role="status" aria-live="polite">
  File uploaded successfully
</div>

aria-live="polite": Announces when screen reader finishes current task
aria-live="assertive": Interrupts to announce immediately
```

#### Anti-Patterns (Problematic)

**Divs for Everything:**
```html
<div class="button" onclick="doSomething()">Click me</div> ✗

Use: <button onclick="doSomething()">Click me</button> ✓
```

**Missing Alt Text:**
```html
<img src="important-chart.png"> ✗
<img src="important-chart.png" alt="Revenue chart"> ✓
```

**Skipping Heading Levels:**
```html
<h1>Title</h1>
<h3>Subsection</h3> ✗ (skipped h2)

Use logical hierarchy: h1 → h2 → h3 ✓
```

**Placeholder as Label:**
```html
<input type="text" placeholder="Email"> ✗
(Placeholder disappears when typing, screen reader may not announce)

Use: <label for="email">Email</label> ✓
     <input type="email" id="email">
```

**Icon Without Label:**
```html
<button><i class="icon-trash"></i></button> ✗
(Screen reader announces "button" - user has no idea what it does)

<button aria-label="Delete"><i class="icon-trash"></i></button> ✓
```

**No Focus Management in Modals:**
```
Modal opens, focus stays on background element ✗
Focus moves to modal, traps within modal, returns on close ✓
```

#### Testing Methods

**Screen Reader Testing:**

**Windows - NVDA (Free):**
1. Download NVDA: https://www.nvaccess.org/download/
2. Install and launch
3. Navigate with:
   - Down arrow: Next item
   - H: Next heading
   - Tab: Next interactive element
   - Enter: Activate link/button
   - Insert+Down: Read all
4. Check: Can you navigate? Do elements announce correctly?

**Windows - JAWS (Commercial, most popular):**
- Similar to NVDA, industry standard for Windows
- Free 40-minute trial sessions

**macOS - VoiceOver (Built-in):**
1. Enable: Cmd+F5
2. Navigate with:
   - VO+Right Arrow: Next item (VO = Ctrl+Option)
   - VO+H: Next heading
   - Tab: Next interactive element
   - VO+Space: Activate
3. Check: Does content make sense? Are labels clear?

**Mobile - TalkBack (Android) / VoiceOver (iOS):**
- Enable in Accessibility settings
- Swipe right to next element
- Double-tap to activate
- Test: Can you complete core tasks?

**Automated Testing:**
- Axe DevTools (Chrome): Catches 30-40% of issues automatically
- Lighthouse (Chrome): Accessibility score and specific issues
- WAVE (browser extension): Visual feedback on accessibility issues

**Manual Testing Checklist:**
- [ ] Navigate entire site with screen reader (NVDA/VoiceOver)
- [ ] Check heading hierarchy (use headings list in screen reader)
- [ ] Verify all images have alt text
- [ ] Test forms (labels announced correctly?)
- [ ] Test modals (focus management works?)
- [ ] Run automated tests (Axe, Lighthouse, WAVE)
- [ ] No elements announced as "clickable" or "button" without context


### Dimension 5: TEMPORAL ACCESSIBILITY

**Purpose**: Users have adequate time to read, interact, and complete tasks

#### Evaluation Questions

**Timeouts:**
- Are timeouts avoidable or adjustable?
- Are users warned before session expires?
- Can users extend timeouts easily?
- Are timeouts necessary or arbitrary?

**Moving Content:**
- Can users pause, stop, or hide animations?
- Is auto-playing content controllable?
- Are carousels pausable?
- Do animations respect prefers-reduced-motion?

**Time Limits:**
- Are time limits user-adjustable (extends, disables)?
- Are real-time events (auctions) the only exception?
- Are users given 20 seconds warning minimum before timeout?
- Can users save work if timeout occurs?

#### WCAG 2.1 AA Requirements

**2.2.1 Timing Adjustable - Level A:**
- User can turn off, adjust, or extend time limits (except real-time events)

**2.2.2 Pause, Stop, Hide - Level A:**
- User can pause, stop, or hide moving, blinking, or scrolling content

**2.2.6 Timeouts - Level AAA (Best Practice):**
- Users are warned of the duration of inactivity that could cause data loss

#### Patterns (Good)

**No Auto-Timeout on Critical Flows:**
```
Form filling: No timeout (or 20+ minutes with warning)
Shopping cart: No timeout (save for later)
Authentication: 30-minute timeout with 5-minute warning

Not: 2-minute timeout on complex form (anxiety-inducing)
```

**Timeout Warnings:**
```
Modal appears 5 minutes before timeout:
"Your session will expire in 5 minutes due to inactivity.
[Extend Session] [Log Out]"

Clicking "Extend Session" resets timer
```

**Pausable Animations:**
```
Carousel with controls:
[Previous] [Pause/Play] [Next]

Autoplay pauses on hover or focus
User can disable autoplay entirely
```

**Respect prefers-reduced-motion:**
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}

Users with motion sensitivity can disable animations
```

**User-Controlled Content:**
```
Video/Audio:
- Autoplay disabled (or muted with clear unmute button)
- Play/Pause controls always visible
- Volume controls available
```

#### Anti-Patterns (Problematic)

**Tight Timeouts:**
```
2-minute timeout on multi-page form ✗
Users with cognitive disabilities need more time
Minimum 20 minutes, with warning and extend option ✓
```

**No Timeout Warning:**
```
Session expires suddenly, work lost ✗
Warning 5+ minutes before, with extend option ✓
```

**Auto-Playing Video:**
```
Video plays on page load with sound ✗
(Disorienting, accessibility violation)
Require user interaction to play ✓
```

**Unstoppable Animations:**
```
Constant animation with no pause button ✗
Motion sickness, distraction
Provide pause control ✓
```

**Carousel Auto-Advance:**
```
Carousel advances every 3 seconds, can't pause ✗
Users with reading difficulties can't keep up
Pause on hover, manual controls, disable autoplay ✓
```

#### Testing Methods

**Timeout Testing:**
- Leave interface idle for timeout period
- Check: Is warning shown before timeout?
- Check: Can user extend session?
- Check: Is work saved if timeout occurs?

**Animation Testing:**
- Enable prefers-reduced-motion in OS settings (macOS: Accessibility > Display > Reduce Motion)
- Check: Do animations disable or significantly reduce?
- Check: Is functionality still accessible without animations?

**Auto-Play Testing:**
- Load pages with video/audio/carousels
- Check: Does anything auto-play without user action?
- Check: Are pause controls visible and accessible?
- Check: Can user disable autoplay permanently?

**Temporal Accessibility Checklist:**
- [ ] No critical timeouts under 20 minutes
- [ ] Timeout warnings shown 5+ minutes before expiration
- [ ] Users can extend timeouts easily
- [ ] All animations pausable/stoppable
- [ ] Respect prefers-reduced-motion setting
- [ ] No auto-playing video/audio
- [ ] Carousels have pause controls


### Dimension 6: SITUATIONAL ACCESSIBILITY

**Purpose**: Design works in varied real-world contexts and situations

#### Evaluation Questions

**Environmental Contexts:**
- Does it work in bright sunlight (mobile)?
- Does it work in low light (desktop at night)?
- Does it work with one hand (mobile, carrying things)?
- Does it work in noisy environments (captions available)?
- Does it work in quiet environments (visual feedback, not just audio)?

**Connection Contexts:**
- Does it work on slow connections (3G, rural)?
- Does it work offline (progressive web app)?
- Are large assets lazy-loaded?
- Is there feedback during loading?

**Device Contexts:**
- Does it work on small screens (iPhone SE)?
- Does it work on large screens (4K monitors)?
- Does it work on old devices (performance)?
- Does it work on different input methods (touch, mouse, keyboard, voice)?

**User Contexts:**
- Does it work when distracted (errors prevented)?
- Does it work when stressed (clear language)?
- Does it work when tired (high contrast, large text)?
- Does it work for temporary disabilities (broken arm, eye dilation)?

#### Patterns (Good)

**High Contrast Mode:**
```css
@media (prefers-contrast: high) {
  body {
    background: #000000;
    color: #FFFFFF;
  }
  a {
    color: #FFFF00; /* Yellow on black */
  }
}

System-level high contrast settings respected
```

**Dual Feedback (Visual + Audio):**
```
Notification arrives:
- Visual: Toast notification on screen
- Audio: Subtle sound (optional, user can disable)
- Vibration: Haptic feedback (mobile)

Works in noisy (visual) and quiet (visual) environments
```

**Progressive Enhancement:**
```
Core content: Loads first, readable without JS/CSS
Styling: CSS loads, visual hierarchy added
Interactivity: JS loads, enhanced interactions

Works on slow connections (core content first)
Works with JS disabled (basic functionality remains)
```

**Offline Functionality:**
```
Service Worker caches essential assets
Offline page explains situation: "You're offline. Some features unavailable."
Work continues (drafts saved locally, sync when back online)
```

**Responsive Images:**
```html
<picture>
  <source media="(max-width: 600px)" srcset="image-small.jpg">
  <source media="(max-width: 1200px)" srcset="image-medium.jpg">
  <img src="image-large.jpg" alt="Description">
</picture>

Appropriate size loaded based on screen (saves bandwidth)
```

**One-Handed Use (Mobile):**
```
Primary actions in thumb zone (bottom 50% of screen)
Swipe gestures accessible from edges
Important actions reachable without shifting grip
```

#### Anti-Patterns (Problematic)

**Low Contrast in Sunlight:**
```
Light gray on white (#CCCCCC on #FFFFFF) ✗
Invisible in bright sunlight on mobile
Use high contrast: #000000 on #FFFFFF ✓
```

**Audio-Only Feedback:**
```
Error announced via sound only ✗
Fails in quiet environments (library, meetings)
Visual indicator required (toast, highlight) ✓
```

**Heavy Assets on Slow Connections:**
```
5MB images load before content ✗
Lazy load images, optimize sizes, show content first ✓
```

**Requires Two Hands (Mobile):**
```
Critical action in top-left corner (requires shifting grip) ✗
Primary actions in thumb zone (bottom half) ✓
```

**No Offline Fallback:**
```
Blank white screen when offline ✗
Offline page explaining situation + cached content ✓
```

**Fixed Design (No Responsive):**
```
Desktop-only design (breaks on mobile) ✗
Responsive design (adapts to all screens) ✓
```

#### Testing Methods

**Sunlight Testing:**
- Test mobile app outdoors in bright sunlight
- Check: Can you read text?
- Check: Can you see buttons?
- Solution: Increase contrast, larger text

**One-Handed Testing:**
- Use phone with one hand only
- Check: Can you reach primary actions?
- Check: Can you navigate comfortably?

**Slow Connection Testing:**
- Chrome DevTools > Network > Throttle to "Slow 3G"
- Check: Does core content load first?
- Check: Is there loading feedback?
- Check: Can user accomplish tasks on slow connection?

**Offline Testing:**
- Chrome DevTools > Network > Offline
- Reload page
- Check: Is there offline fallback?
- Check: Is user informed?

**Noise Testing:**
- Disable device sound
- Check: Can you still use all features?
- Check: Is visual feedback provided?

**Situational Accessibility Checklist:**
- [ ] High contrast mode supported
- [ ] Works in bright sunlight (high contrast)
- [ ] Works in noisy and quiet environments (dual feedback)
- [ ] Works on slow connections (progressive enhancement)
- [ ] Works offline (service worker, cached content)
- [ ] Works with one hand (mobile)
- [ ] Works on small and large screens (responsive)


## Practical Application

### Step-by-Step: Accessibility Audit

**Step 1: Automated Scan**
1. Install Axe DevTools (Chrome extension)
2. Open your design/site in browser
3. Run Axe scan
4. Note critical and serious issues (contrast, missing labels, etc.)

**Step 2: Visual Accessibility**
1. Check contrast ratios (WebAIM Contrast Checker)
2. Test colorblind simulation (Stark or Colorblinding extension)
3. Zoom to 200% (check for horizontal scroll, overlapping)
4. Verify text sizes (16px mobile minimum)

**Step 3: Motor Accessibility**
1. Unplug mouse, navigate with keyboard only
2. Check focus indicators visible
3. Measure touch targets (44x44px minimum)
4. Verify spacing between interactive elements (8px)

**Step 4: Cognitive Accessibility**
1. Read content aloud (does it make sense?)
2. Check reading level (Hemingway Editor - grade 8 or lower)
3. Count choices per screen (5-7 max before chunking)
4. Test error messages (clear and actionable?)

**Step 5: Screen Reader**
1. Enable screen reader (NVDA on Windows, VoiceOver on macOS)
2. Navigate with keyboard shortcuts (arrows, H for headings, Tab for links)
3. Check: Are labels clear? Is order logical? Are images described?
4. Test forms (are fields labeled correctly?)

**Step 6: Temporal Accessibility**
1. Check for timeouts (are they adjustable?)
2. Test animations (can user pause/stop?)
3. Enable prefers-reduced-motion (do animations disable?)
4. Check auto-playing content (is it user-controlled?)

**Step 7: Situational Accessibility**
1. Test in bright light (can you read it?)
2. Test on slow connection (DevTools throttling)
3. Test offline (is there fallback?)
4. Test one-handed (mobile - can you reach actions?)

**Step 8: Document & Prioritize**
1. List all issues by WCAG severity (A > AA > AAA)
2. Prioritize: Critical (blocking) > Major (difficult) > Minor (annoying)
3. Create fix plan with ownership and deadlines
4. Re-test after fixes


## WCAG 2.1 AA Compliance Quick Reference

### Level A (Must Have)

**1.1.1** Non-text Content - Images have alt text
**1.3.1** Info and Relationships - Semantic HTML structure
**2.1.1** Keyboard - All functionality keyboard accessible
**2.1.2** No Keyboard Trap - Can move focus away
**2.4.1** Bypass Blocks - Skip links for navigation
**3.3.1** Error Identification - Errors described in text
**3.3.2** Labels or Instructions - Form fields labeled

### Level AA (Should Have)

**1.4.3** Contrast (Minimum) - 4.5:1 for text, 3:1 for large text
**1.4.4** Resize Text - 200% zoom without loss
**1.4.11** Non-text Contrast - 3:1 for UI components
**2.4.6** Headings and Labels - Describe purpose
**2.4.7** Focus Visible - Keyboard focus indicator visible
**3.2.3** Consistent Navigation - Same order across pages
**3.2.4** Consistent Identification - Same function, same label
**3.3.3** Error Suggestion - Suggestions for fixing errors
**3.3.4** Error Prevention - Reversible, checked, or confirmed

### Level AAA (Nice to Have)

**1.4.6** Contrast (Enhanced) - 7:1 for text, 4.5:1 for large text
**2.5.5** Target Size - 44x44px touch targets
**3.1.5** Reading Level - Lower secondary education level

**Note**: WCAG AA is the legal standard in most jurisdictions. AAA is best practice.


## Accessibility Tools Reference

### Contrast Checkers
- **WebAIM Contrast Checker**: https://webaim.org/resources/contrastchecker/
- **Stark** (Figma/Sketch): Real-time contrast checking in design tools
- **Chrome DevTools**: Color picker shows contrast ratio

### Colorblind Simulation
- **Stark**: Protanopia, Deuteranopia, Tritanopia filters (Figma)
- **Colorblinding**: Chrome extension
- **Sim Daltonism**: macOS system-wide filters

### Screen Readers
- **NVDA** (Windows): Free, open-source - https://www.nvaccess.org/
- **JAWS** (Windows): Commercial, most popular - https://www.freedomscientific.com/
- **VoiceOver** (macOS/iOS): Built-in (Cmd+F5 to enable)
- **TalkBack** (Android): Built-in (Settings > Accessibility)

### Automated Testing
- **Axe DevTools**: Chrome/Firefox extension, catches 30-40% of issues
- **Lighthouse**: Chrome built-in, accessibility score + specific issues
- **WAVE**: Browser extension, visual feedback on issues

### Readability
- **Hemingway Editor**: Checks reading level (aim for grade 8 or lower)
- **Readable.io**: Analyzes content complexity

### Performance & Loading
- **Chrome DevTools Network Throttling**: Test slow connections
- **WebPageTest**: Real-world performance testing


## Cross-Platform Accessibility Considerations

### Mobile (iOS/Android)

**Voice Control:**
- iOS: Voice Control for hands-free operation
- Android: Voice Access for voice navigation

**Magnification:**
- iOS: Zoom (triple-tap with three fingers)
- Android: Magnification gestures (triple-tap)

**Screen Readers:**
- iOS: VoiceOver (swipe right to next element, double-tap to activate)
- Android: TalkBack (swipe right to next element, double-tap to activate)

**Touch Targets:**
- iOS: 44x44pt minimum
- Android: 48x48dp minimum

**References:**
- iOS: Accessibility > lyra/ux-designer/mobile-design-patterns
- Android: Accessibility > lyra/ux-designer/mobile-design-patterns


### Web Applications

**Keyboard Shortcuts:**
- Essential: Tab, Shift+Tab, Enter, Esc, Arrow keys
- Optional: Cmd/Ctrl+S (save), Cmd/Ctrl+F (find), / (search)

**Focus Management:**
- Modals: Trap focus, return to trigger on close
- Single-page apps: Manage focus on route changes

**ARIA Live Regions:**
- Dynamic updates announced to screen readers
- Form validation errors announced

**References:**
- Web accessibility > lyra/ux-designer/web-application-design


### Desktop Software

**Keyboard-First:**
- All functions accessible via keyboard
- Customizable shortcuts for power users

**Zoom:**
- System-level zoom support (macOS: Cmd+Plus, Windows: Ctrl+Plus)
- Application-level zoom for specific content

**High Contrast Mode:**
- Windows: High Contrast themes
- macOS: Increase Contrast setting

**References:**
- Desktop accessibility > lyra/ux-designer/desktop-software-design


### Game UI

**Colorblind Modes:**
- Deuteranopia (red-green)
- Protanopia (red-green)
- Tritanopia (blue-yellow)

**Subtitles & Captions:**
- Dialogue subtitles
- Sound effect indicators (footsteps, gunfire direction)

**Difficulty Adjustments:**
- Easier difficulty for cognitive accessibility
- Assist modes for motor accessibility

**References:**
- Game UI accessibility > lyra/ux-designer/game-ui-design


## Common Accessibility Mistakes & Fixes

### Mistake 1: Low Contrast Text
**Problem**: Light gray text on white background (#999 on #fff = 2.8:1)
**Fix**: Use darker gray (#595959 on #fff = 7:1) or black (#000 on #fff = 21:1)
**Tool**: WebAIM Contrast Checker

### Mistake 2: Color as Sole Indicator
**Problem**: "Red items are errors" (colorblind users can't see red)
**Fix**: Red + "X" icon + "Error" text
**Tool**: Stark colorblind simulation

### Mistake 3: Missing Alt Text
**Problem**: `<img src="chart.png">` (screen reader says "image")
**Fix**: `<img src="chart.png" alt="Revenue growth from $1M to $5M over 5 years">`
**Tool**: Screen reader testing (NVDA/VoiceOver)

### Mistake 4: Divs for Buttons
**Problem**: `<div onclick="submit()">Submit</div>` (not keyboard accessible)
**Fix**: `<button onclick="submit()">Submit</button>`
**Tool**: Keyboard navigation test (unplug mouse)

### Mistake 5: No Focus Indicators
**Problem**: `*:focus { outline: none; }` (keyboard users lost)
**Fix**: `button:focus { outline: 2px solid blue; }`
**Tool**: Keyboard navigation test

### Mistake 6: Tiny Touch Targets
**Problem**: 30x30px buttons (frustrating on mobile)
**Fix**: 44x44pt (iOS) or 48x48dp (Android) minimum
**Tool**: DevTools measure, finger test on device

### Mistake 7: Auto-Playing Video
**Problem**: Video plays on page load with sound (disorienting)
**Fix**: Require user interaction to play, or autoplay muted with clear controls
**Tool**: Load page with sound on

### Mistake 8: Illogical Heading Hierarchy
**Problem**: `<h1>` → `<h4>` → `<h2>` (screen reader users confused)
**Fix**: Logical hierarchy: `<h1>` → `<h2>` → `<h3>`
**Tool**: Screen reader headings list (NVDA: Insert+F7)

### Mistake 9: Placeholder as Label
**Problem**: `<input placeholder="Email">` (disappears when typing)
**Fix**: `<label for="email">Email</label><input id="email">`
**Tool**: Screen reader test (is label announced?)

### Mistake 10: Timeout Without Warning
**Problem**: Session expires at 5 minutes, work lost
**Fix**: 20+ minute timeout with 5-minute warning and extend option
**Tool**: Leave interface idle, observe timeout behavior


## Related Skills

**Core UX Skills:**
- `lyra/ux-designer/visual-design-foundations` - Color contrast, typography readability (Dimension 1, 5)
- `lyra/ux-designer/interaction-design-patterns` - Keyboard navigation, focus states, touch targets (Dimension 2, 4)
- `lyra/ux-designer/information-architecture` - Logical structure, clear navigation (Dimension 3, 4)
- `lyra/ux-designer/ux-fundamentals` - Accessibility principles and terminology

**Platform Skills:**
- `lyra/ux-designer/mobile-design-patterns` - Touch target sizing, platform accessibility APIs
- `lyra/ux-designer/web-application-design` - ARIA, semantic HTML, keyboard shortcuts
- `lyra/ux-designer/desktop-software-design` - Keyboard-first design, system accessibility settings
- `lyra/ux-designer/game-ui-design` - Colorblind modes, subtitle/caption systems

**Cross-Faction:**
- `muna/technical-writer/clarity-and-style` - Plain language, 8th grade reading level (Dimension 3)


## Further Resources

**WCAG Guidelines:**
- WCAG 2.1: https://www.w3.org/WAI/WCAG21/quickref/
- WCAG 2.2 (latest): https://www.w3.org/WAI/WCAG22/quickref/

**Testing Tools:**
- Axe DevTools: https://www.deque.com/axe/devtools/
- WAVE: https://wave.webaim.org/
- Lighthouse: Built into Chrome DevTools

**Screen Readers:**
- NVDA (Windows): https://www.nvaccess.org/
- JAWS (Windows): https://www.freedomscientific.com/
- VoiceOver (macOS): Built-in (Cmd+F5)

**Learning:**
- WebAIM: https://webaim.org/ (excellent articles and guides)
- A11y Project: https://www.a11yproject.com/ (community-driven resources)
- Inclusive Components: https://inclusive-components.design/

**Remember**: Accessibility is not optional. It's a legal requirement in most jurisdictions and an ethical imperative. Design for everyone from the start - retrofitting accessibility is expensive and incomplete.
