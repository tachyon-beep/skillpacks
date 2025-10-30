---
name: interaction-design-patterns
description: Use when designing interaction patterns, button states, feedback systems, micro-interactions, or touch targets - systematic framework evaluating affordances, feedback timing, animations, state changes, and gesture comfort
---

# Interaction Design Patterns

## Overview

This skill provides **The Interaction Clarity Framework**, a systematic 5-dimension methodology for designing clear, responsive, and comfortable user interactions. Use this when evaluating or designing interactive elements, feedback systems, button states, animations, or touch-based interfaces.

**Core Principle**: Users should never wonder if something is interactive, whether their action registered, or what state the system is in. Clear interactions build trust and efficiency.

## When to Use

Load this skill when:
- Designing button states, touch targets, or interactive elements
- User mentions: "feedback", "loading states", "micro-interactions", "animations"
- Evaluating interaction clarity ("Does this feel responsive?")
- Specifying interaction behaviors (hover, active, disabled states)
- Designing gesture-based interfaces (swipe, tap, long-press)
- Touch target sizing issues (too small, accidental taps)
- Creating animations or transitions
- Defining system feedback patterns

**Don't use for**: Visual hierarchy alone (use visual-design-foundations), navigation structure (use information-architecture)

---

## The Interaction Clarity Framework

A systematic 5-dimension evaluation model for interaction design:

1. **AFFORDANCES** - Users understand what's interactive
2. **FEEDBACK** - System responds to actions (<100ms)
3. **MICRO-INTERACTIONS** - Purposeful animations (150-300ms)
4. **STATE CHANGES** - Current state is visually distinct
5. **TOUCH TARGETS & GESTURES** - Physically comfortable (44x44pt+)

---

## Dimension 1: AFFORDANCES

**Purpose:** Make interactive elements obvious and discoverable

### Evaluation Questions

- Can users immediately identify what's clickable/tappable?
- Do interactive elements look different from static content?
- Are platform conventions followed (links colored, buttons raised)?
- Does the cursor change on hover (desktop)?
- Do elements have visual weight appropriate to their interactivity?

### Patterns (Good Examples)

**Primary Buttons:**
- Visual weight (filled background, contrasting color)
- Subtle shadow or border for depth
- Clear label with action verb ("Save", "Submit", "Continue")
- Adequate padding (12-16px horizontal, 8-12px vertical)
- Hover state shows interactivity (brightness change, shadow increase)

**Secondary Buttons:**
- Less visual weight than primary (outlined vs filled)
- Still clearly interactive (not flat text)
- Same size targets as primary (even if visually lighter)

**Links:**
- Colored text (blue standard, or brand color)
- Underline on hover or persistent
- Cursor changes to pointer
- Distinct from body text (color and/or weight)

**Interactive Icons:**
- Consistent sizing (24px minimum for touch)
- Hover/focus states (background circle, color change)
- Tooltips on hover (desktop) or long-press (mobile)
- Clear visual boundary (not floating in white space)

**Form Fields:**
- Border or background distinguishes from static text
- Focus state with border color change
- Placeholder text lighter than input text
- Label positioned clearly (above or floating)

### Anti-Patterns (Problematic Examples)

**Flat Text Buttons:**
- Plain text with no visual weight
- No hover state or indication
- Looks like static label
- Users miss the interactive opportunity

**False Affordances:**
- Static elements styled like buttons (shadows, borders)
- Decorative elements that look clickable
- Creates frustration when clicks don't work

**Mystery Interactions:**
- Icons without labels or tooltips
- No indication of hover/focus
- Users must guess what happens

**Inconsistent Affordances:**
- Some buttons filled, some text-only with no pattern
- Links sometimes underlined, sometimes not
- Creates cognitive load as users relearn each screen

### Platform-Specific Affordances

**Web (Desktop):**
- Cursor changes (pointer for interactive, default for static)
- Hover states reveal interactivity
- Underlined links standard

**Mobile (iOS/Android):**
- No hover states (touch-only)
- Visual affordances must be permanent (not hover-dependent)
- Platform conventions (iOS segmented controls, Android FABs)
- Haptic feedback option (tactile confirmation)

**Desktop Software:**
- Menu items highlight on hover
- Keyboard focus indicators (2px outline)
- Right-click context menus (subtle cue on hover)

**Games:**
- Cursor changes for interactive objects
- Outline glow on hover (world objects)
- Button prompts for gamepad ("Press A to interact")

---

## Dimension 2: FEEDBACK

**Purpose:** Confirm user actions with immediate, appropriate responses

### Evaluation Questions

- Does the user see feedback within 100ms of action?
- Is loading state shown for operations >1 second?
- Are errors explained clearly with actionable fixes?
- Is success confirmed explicitly?
- Does feedback match action magnitude (subtle for minor, prominent for major)?

### Feedback Timing Guidelines

**Immediate (<100ms):**
- Button press (depress visual)
- Toggle switch flip
- Checkbox check/uncheck
- Text input (character appears)
- Drag start (element lifts)

**Short Delay (100ms-1s):**
- Form validation (after blur)
- Search autocomplete
- Hover tooltips (300ms delay prevents flicker)
- Menu open/close animations

**Long Operations (>1s):**
- Loading spinner or progress bar
- Percentage complete (if calculable)
- Optimistic UI (assume success, show immediately)
- Background processing indicator

**Error Timing:**
- Inline validation (after field blur, not on every keystroke)
- Form submission errors (immediate, at top of form)
- Toast notifications (appear immediately, auto-dismiss 3-5s)

### Patterns (Good Examples)

**Button Press Feedback:**
- Visual depress (shadow reduces, brightness changes)
- Disabled state immediately on click (prevent double-submit)
- Loading spinner replaces button text
- Success state (checkmark icon, green color, 2s duration)

**Form Validation:**
- Error state: Red border, error icon, error message below field
- Error message explains problem: "Email must include @"
- Success state: Green border, checkmark icon (optional)
- Real-time hints: "Password strength: Weak → Strong"

**Loading States:**
- Spinner for indeterminate waits (<10s expected)
- Progress bar for determinate (shows percentage)
- Skeleton screens (show content structure while loading)
- Optimistic UI (show result immediately, rollback if fails)

**Success Confirmation:**
- Toast notification: "Settings saved successfully" (3s auto-dismiss)
- Inline message: Green banner with checkmark
- Visual transition: Item moves to "completed" section
- Subtle animation: Brief green flash on save icon

**Error Messages:**
- Clear problem statement: "Unable to connect to server"
- Actionable fix: "Check your internet connection and try again"
- Retry button or manual dismiss
- Not technical jargon: "Error 500" → "Something went wrong on our end"

**Drag & Drop Feedback:**
- Item lifts on grab (shadow increases, opacity 90%)
- Drop zones highlight when dragging over (border, background color)
- Snap animation when dropped (ease-out 200ms)
- Cancel if dropped outside zone (return to original position)

### Anti-Patterns (Problematic Examples)

**No Feedback:**
- Button doesn't depress on click
- User clicks again thinking it didn't work
- Results in duplicate submissions, frustration

**Generic Errors:**
- "Something went wrong" (no explanation)
- "Error 500" (technical jargon)
- No guidance on how to fix

**Slow Feedback:**
- Delayed button press visual (>200ms)
- Feels sluggish, unresponsive
- Users lose confidence in system

**No Loading Indicator:**
- Blank screen during load
- User wonders if system is frozen
- May leave or refresh, interrupting process

**Success Without Confirmation:**
- Form submits but no feedback
- User unsure if it worked
- May submit again or leave uncertain

### Platform-Specific Feedback

**Web:**
- Toast notifications (top-right or bottom-center)
- Inline validation messages
- Loading spinners (CSS animations)
- Browser native feedback (form validation bubbles)

**Mobile:**
- Haptic feedback (vibration on button press)
- Bottom sheets for confirmations
- Snackbars (Android Material Design)
- Pull-to-refresh feedback (iOS)

**Desktop:**
- System notifications (OS-level)
- Status bar updates (bottom of window)
- Progress in dock/taskbar icon
- Modal dialogs for critical confirmations

**Games:**
- HUD updates (health, score, ammo)
- Visual effects (hit sparks, damage numbers)
- Audio feedback (essential in fast action)
- Screen shake for major impacts

---

## Dimension 3: MICRO-INTERACTIONS

**Purpose:** Delight users and clarify relationships through purposeful animations

### Evaluation Questions

- Are animations purposeful (not gratuitous decoration)?
- Is timing appropriate (150-300ms for most UI)?
- Do transitions feel natural (easing curves)?
- Do animations hint at spatial relationships?
- Can users disable animations (accessibility)?

### Animation Timing Guidelines

**UI Transitions (150-300ms):**
- Modal open/close: 200ms
- Dropdown menu: 150ms
- Tooltip appear: 200ms
- Page transitions: 250ms
- Accordion expand: 300ms

**Micro-Interactions (100-200ms):**
- Button hover: 100ms
- Toggle switch: 150ms
- Checkbox check: 100ms
- Icon morph: 200ms

**Too Fast (<100ms):**
- Feels abrupt, jarring
- Users miss the transition
- No sense of continuity

**Too Slow (>500ms):**
- Feels sluggish
- Slows users down
- Creates impatience

**Long Animations (>500ms):**
- Only for skeleton loading screens
- Or onboarding/tutorial animations
- User must be able to skip or dismiss

### Easing Curves

**Ease-Out (Deceleration):**
- Use for: Entering animations (modals, tooltips)
- Effect: Fast start, slow finish
- Feels: Natural, settling into place
- CSS: `cubic-bezier(0.0, 0.0, 0.2, 1)`

**Ease-In (Acceleration):**
- Use for: Exiting animations (closing, dismissing)
- Effect: Slow start, fast finish
- Feels: Quick departure
- CSS: `cubic-bezier(0.4, 0.0, 1, 1)`

**Ease-In-Out (Acceleration then Deceleration):**
- Use for: Movement between states (repositioning)
- Effect: Smooth throughout
- Feels: Graceful, continuous
- CSS: `cubic-bezier(0.4, 0.0, 0.2, 1)`

**Linear (No Easing):**
- Use for: Loading spinners, progress bars
- Effect: Constant speed
- Feels: Mechanical (intentionally)
- CSS: `linear`

### Patterns (Good Examples)

**Modal Entry:**
- Fade-in background overlay (200ms ease-out)
- Scale up modal from 90% to 100% (200ms ease-out)
- Slight upward movement (20px)
- Combined effect: Smooth, attention-grabbing

**Dropdown Menu:**
- Expand from button (origin point clear)
- Fade + scale (150ms ease-out)
- Stagger items (20ms delay each) for cascade effect

**Loading Skeleton:**
- Shimmer effect passes over grey boxes
- Continuous animation (1500ms linear loop)
- Indicates "still loading" without blocking

**Toggle Switch:**
- Circle slides across track (150ms ease-out)
- Background color changes (150ms)
- Both animated simultaneously

**Accordion Expand:**
- Content height animates (300ms ease-out)
- Icon rotates 90° (300ms ease-out)
- Smooth reveal, not abrupt pop

**Pull to Refresh:**
- Spinner rotates continuously
- Page slides down as user pulls
- Elastic bounce when released (iOS)

**Hover Lift:**
- Card elevates on hover (shadow increases, slight Y translation)
- 200ms ease-out
- Subtle but clear interactivity

**Icon Morph:**
- Hamburger → X (menu icon)
- SVG path animation (200ms ease-out)
- Clear transformation, not abrupt swap

### Anti-Patterns (Problematic Examples)

**Gratuitous Animation:**
- Every element bounces on page load
- Slows users, adds no value
- Creates distraction, not delight

**Overly Long Animations:**
- Modal takes 1 second to open
- Users feel impatient
- Slows workflow significantly

**Animation Without Purpose:**
- Random spinning or bouncing
- Doesn't clarify relationships
- Pure decoration that distracts

**Inconsistent Timing:**
- Some modals 200ms, others 500ms
- Creates uneven, sloppy feel
- Breaks user expectations

**No Ability to Disable:**
- Critical for motion sensitivity
- WCAG requirement (prefers-reduced-motion)
- Can cause vestibular disorders

**Animations That Block:**
- Must watch animation complete before interacting
- Users can't skip or dismiss
- Creates forced waiting

### Accessibility Considerations

**Respect prefers-reduced-motion:**
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

**Avoid:**
- Parallax scrolling (motion sickness)
- Constant background motion
- Flashing/strobing (seizure risk)
- Rapid color changes

**Provide:**
- Setting to disable animations
- Pause controls for auto-playing content
- Skip buttons for long animations

---

## Dimension 4: STATE CHANGES

**Purpose:** Make current state immediately obvious and visually distinct

### Evaluation Questions

- Can users identify the current state at a glance?
- Is selected/active state high contrast?
- Are disabled states obvious (grayed out)?
- Is progress shown for multi-step flows?
- Do states persist across navigation?

### Standard State Specifications

**Button States:**

**Default (Resting):**
- Normal color and styling
- Cursor: pointer (desktop)
- No special indication

**Hover (Desktop Only):**
- Brightness +10% or shadow increase
- Cursor: pointer
- Transition: 100ms ease-out

**Focus (Keyboard Navigation):**
- 2px outline (browser default or custom color)
- Persistent until blur
- High contrast (visible on all backgrounds)
- Never remove outline without replacement

**Active (Pressed):**
- Brightness -10% or shadow decrease
- Slight scale down (98%)
- Immediate feedback (no transition delay)

**Disabled:**
- Opacity 40-50% or gray color
- Cursor: not-allowed
- No hover/active states
- Clearly non-interactive

**Loading:**
- Disabled state + spinner
- Button text hidden or replaced
- Prevents multiple clicks
- Spinner animates continuously

**Success (Temporary):**
- Green color + checkmark icon
- Duration: 2 seconds
- Then return to default or disabled
- Confirms completion

**Error (Temporary):**
- Red color + error icon
- Duration: Persistent until user acts
- Error message nearby
- Remains interactive (user can retry)

### Form Input States

**Default:**
- Border: 1px solid #ccc
- Background: white or light gray
- Placeholder: #999
- Cursor: text

**Focus:**
- Border: 2px solid brand color
- Shadow: 0 0 0 3px brand-color-20%
- Placeholder fades or disappears
- Cursor: text blinking

**Filled (Valid):**
- Border: 1px solid #ccc or green
- Optional: Green checkmark icon
- Input text: #000

**Error:**
- Border: 2px solid red
- Error message below (red text)
- Error icon inside or next to field
- Focus: Red border remains

**Disabled:**
- Background: #f5f5f5
- Border: #ddd
- Text: #999
- Cursor: not-allowed

### Navigation States

**Default (Unselected):**
- Normal text color
- No background highlight

**Hover:**
- Background: Light gray
- Text color: Slightly darker

**Active/Selected:**
- Background: Brand color or darker gray
- Text: White or high contrast
- Bold weight or indicator (left border, icon)

**Visited (Links Only):**
- Purple or muted brand color
- Indicates "already been here"

### Toggle/Checkbox States

**Unchecked:**
- Empty box or gray background
- Border visible

**Checked:**
- Checkmark icon or filled circle
- Brand color background
- Smooth transition (100ms)

**Indeterminate (Tri-state):**
- Dash icon (some children checked)
- Used in hierarchical lists

### Progress Indicators

**Multi-Step Forms:**
- Stepper: "Step 2 of 5"
- Visual progress: Filled circles or bar
- Current step highlighted
- Completed steps: Checkmark
- Future steps: Gray or outline only

**Loading Progress:**
- Percentage: "Loading 45%"
- Progress bar fills left-to-right
- Smooth animation (not jumpy)

**Status Badges:**
- "Draft" (gray), "Published" (green), "Archived" (blue)
- Color-coded with icon
- Consistent placement

### Patterns (Good Examples)

**Tab Navigation:**
- Default: Gray text, no underline
- Hover: Darker text
- Active: Brand color, bottom border (3px), bold weight
- Smooth underline animation (200ms)

**Toggle Switch:**
- Off: Gray background, circle left
- On: Brand color background, circle right
- Smooth slide animation (150ms)
- Clear on/off state at glance

**Radio Button Group:**
- Selected: Filled circle, brand color
- Unselected: Empty circle, gray border
- Only one selected at a time

**Accordion:**
- Collapsed: Down arrow icon
- Expanded: Up arrow icon, content visible
- Smooth height animation (300ms)

**Drag & Drop Items:**
- Default: Normal state
- Grabbing: Cursor changes, item lifts (shadow)
- Dragging: Opacity 90%, follows cursor
- Drop zone: Highlighted border/background
- Dropped: Snap into place (200ms ease-out)

### Anti-Patterns (Problematic Examples)

**Subtle State Differences:**
- Active tab only 5% darker (not noticeable)
- Users can't tell which tab is current

**No Disabled State:**
- Grayed text but still interactive
- Users click and get error
- Should be cursor: not-allowed

**Lost Progress:**
- Multi-step form refreshes, progress lost
- No indication of current step
- Users lose place and confidence

**No Visual Feedback:**
- Button clicked but no active state
- User unsure if click registered

**Inconsistent States:**
- Some buttons show loading, others don't
- Creates uneven experience

### Platform-Specific States

**iOS:**
- Segmented controls (grouped buttons)
- Large titles (bold 34pt)
- Haptic feedback on state change

**Android:**
- Ripple effect on tap (Material Design)
- FAB (Floating Action Button) for primary action
- Bottom sheets for contextual actions

**Web:**
- Focus visible for keyboard nav
- Hover states (desktop only)
- Active states on click

**Desktop Software:**
- Selected items in lists (full row highlight)
- Menu items (hover background)
- Focus indicators (keyboard nav essential)

---

## Dimension 5: TOUCH TARGETS & GESTURES

**Purpose:** Ensure interactions are physically comfortable and discoverable

### Evaluation Questions

- Are touch targets at least 44x44px (iOS) or 48x48dp (Android)?
- Is adequate spacing between tappable elements (8px minimum)?
- Are gestures discoverable and teachable?
- Do gestures follow platform conventions?
- Can users complete tasks with one hand (mobile)?

### Touch Target Sizing

**Minimum Sizes:**
- **iOS (Apple HIG):** 44x44 points
- **Android (Material Design):** 48x48 dp
- **Web (Touch):** 48x48 pixels
- **Desktop (Mouse):** 32x32 pixels acceptable (more precise)

**Optimal Sizes:**
- **Primary buttons:** 48-56px height
- **Secondary buttons:** 40-48px height
- **Icon buttons:** 48x48px minimum (larger if alone)
- **List items:** 48-72px height (more space = easier tap)
- **Form inputs:** 48px height minimum

**Spacing Between Targets:**
- **Minimum:** 8px between tappable elements
- **Optimal:** 16px spacing (reduces accidental taps)
- **Dense UIs:** 8px acceptable if targets are distinct
- **Forms:** 16-24px spacing between fields

### Thumb Zones (Mobile)

**Most Reachable (One-Handed):**
- Bottom third of screen
- Center of screen
- Thumb's natural arc (bottom-center to middle-right for right hand)

**Hard to Reach:**
- Top corners (especially top-left for right hand)
- Opposite side (top-right for left hand)

**Design Implications:**
- **Primary actions:** Bottom navigation, bottom buttons
- **Navigation:** Bottom tabs (iOS), bottom nav (Android)
- **Secondary actions:** Top (back button, settings)
- **Content:** Top and middle (scrollable)

### Standard Gestures

**Universal Gestures:**
- **Tap:** Select, activate (like mouse click)
- **Double-tap:** Zoom in (maps, images)
- **Long-press:** Context menu, additional options
- **Swipe:** Navigate (swipe back), scroll, pan
- **Pinch:** Zoom in/out (two-finger spread/pinch)
- **Drag:** Move items, reorder lists

**Platform-Specific Gestures:**

**iOS:**
- **Swipe back:** Left edge swipe = back navigation (system-wide)
- **Swipe actions:** Swipe left/right on list items = actions
- **Pull to refresh:** Pull down on scrollable content
- **3D Touch/Haptic Touch:** Force press for quick actions (older devices)

**Android:**
- **Swipe back:** Swipe from left edge or back button
- **Swipe to dismiss:** Notifications, dialogs
- **Pull to refresh:** Pull down (Material Design)
- **Edge swipe:** Navigation drawer (left edge)

**Web (Touch):**
- **Scroll:** Vertical swipe
- **Zoom:** Pinch or double-tap (if enabled)
- **Context menu:** Long-press (may trigger text selection)

### Patterns (Good Examples)

**Bottom Navigation (Mobile):**
- 3-5 tabs at bottom
- 56-72px height
- Icons + labels (or icon-only if 3 items)
- Easy thumb reach

**Floating Action Button (Android):**
- 56dp diameter (large touch target)
- Bottom-right corner
- Primary action for screen
- Elevated (shadow), branded color

**Swipe Actions (List Items):**
- Swipe left: Delete (red)
- Swipe right: Archive (blue)
- Icons reveal as user swipes
- Can swipe fully to auto-trigger or partial to show buttons

**Pull to Refresh:**
- Pull down on scrollable content
- Spinner appears at top
- Elastic bounce on release
- Standard convention (users expect it)

**Large Tap Targets:**
- Full-width list items (not just icon)
- Entire card tappable (not just title)
- Generous padding (16-24px)
- Clear tap feedback (ripple, highlight)

**Gesture Hints:**
- Faint swipe icon on first use
- Tutorial overlay: "Swipe right to see more"
- Visual affordances (visible edge of next screen)

### Anti-Patterns (Problematic Examples)

**Tiny Touch Targets:**
- 30x30px buttons (too small)
- Causes accidental taps, frustration
- Especially bad for users with motor difficulties

**Crowded Interfaces:**
- Buttons 2px apart
- "Fat finger" problem (tap wrong button)
- Requires precision users can't provide

**Undiscoverable Gestures:**
- Custom gestures with no hint
- Users never discover features
- Essential actions hidden behind gestures

**Conflicting Gestures:**
- Swipe right does different things on different screens
- Inconsistency creates confusion

**Top-Heavy Mobile Design:**
- All actions at top of screen
- Requires two-handed use or thumb stretching
- Ignores ergonomics

**No Gesture Alternative:**
- Only way to delete is swipe (no button)
- Some users may not discover gesture
- Always provide visible alternative

### Gesture Discovery Patterns

**Progressive Disclosure:**
1. First use: Show tutorial overlay
2. Subsequent uses: Subtle hint (fade after 2-3 uses)
3. Experienced users: No hints, fast workflow

**Visual Cues:**
- Visible edge of next screen (hints swipe)
- Icon handles on draggable items
- Bounce animation suggests "pull here"

**Empty States:**
- "Swipe left on items to delete"
- Teaches gesture when no content to interfere

**Settings/Help:**
- Gesture guide in settings
- Helps users discover advanced gestures

### Accessibility Considerations

**Motor Accessibility:**
- Large targets (48px+) help everyone
- Adequate spacing (16px) reduces errors
- Avoid hover-only actions (mobile has no hover)

**Gesture Alternatives:**
- Always provide button alternative to gestures
- Some users can't perform complex gestures
- Voice control users need tappable targets

**Timing:**
- No tight timing requirements (no "double-tap within 300ms")
- Allow slow, deliberate interactions
- Long-press timeout: 500ms (not shorter)

---

## Cross-Platform Interaction Patterns

### Web Application Interactions

**Hover States:**
- Desktop only (no hover on touch)
- Reveal additional actions on hover
- Tooltips after 300ms delay
- Cursor changes (pointer, grab, not-allowed)

**Keyboard Shortcuts:**
- Cmd/Ctrl+S: Save
- Cmd/Ctrl+Z: Undo
- Tab: Next field
- Enter: Submit form
- Esc: Close modal

**Form Interactions:**
- Inline validation (after blur)
- Autocomplete for common inputs
- Clear/reset buttons
- Keyboard navigation (Tab, Shift+Tab)

### Mobile Application Interactions

**Touch-First Design:**
- No hover states (all affordances visible)
- Large touch targets (48px+)
- Platform gestures (swipe back, pull to refresh)
- Bottom-heavy layout (thumb zone)

**Haptic Feedback:**
- Button press (light tap)
- Toggle switch (light tap)
- Error (notification haptic)
- Success (success haptic)

**Native Patterns:**
- iOS: Segmented controls, action sheets, large titles
- Android: FABs, bottom sheets, ripple effects, snackbars

### Desktop Software Interactions

**Keyboard-First:**
- All actions accessible via keyboard
- Visible focus indicators
- Keyboard shortcuts (Cmd+N, Cmd+O, etc.)
- Mnemonics (Alt+F for File menu)

**Mouse Precision:**
- Smaller targets acceptable (32px)
- Right-click context menus
- Drag-and-drop between windows
- Hover tooltips (immediate)

**Multi-Window:**
- Window snapping (drag to edge)
- Panel management (dock, resize, collapse)
- Cross-window drag-and-drop

### Game Interactions

**Input Method:**
- Gamepad: Radial menus, D-pad navigation
- Keyboard+Mouse: Click, hotkeys (1-9), WASD movement
- Touch: Virtual joysticks, large buttons

**Feedback:**
- Visual (hit sparks, damage numbers)
- Audio (critical in games)
- Haptic (controller vibration)
- Screen effects (shake, flash)

**Responsiveness:**
- <16ms for competitive games (60fps)
- <33ms for casual games (30fps)
- Immediate feedback even if network delayed
- Optimistic UI (client-side prediction)

---

## Practical Application

### Step-by-Step Usage

**For Critique Mode:**
1. Identify all interactive elements in design
2. Evaluate against all 5 dimensions systematically
3. Note issues: unclear affordances, missing feedback, tiny targets
4. Prioritize by user impact (critical vs nice-to-have)
5. Provide specific recommendations with examples

**For Specification Mode:**
1. Define all interaction states (default, hover, focus, active, disabled, loading)
2. Specify timing for animations and transitions (150-300ms)
3. Detail feedback mechanisms (spinners, messages, state changes)
4. Size touch targets (48x48px minimum)
5. Document gestures and keyboard shortcuts

### Example Evaluation

**Scenario:** Reviewing a mobile checkout button

**Dimension 1 - Affordances:** ✓ Pass
- Green filled button, clear label "Complete Purchase"
- High visual weight, obvious primary action

**Dimension 2 - Feedback:** ✗ Fail
- No loading state on tap
- No success confirmation
- **Recommendation:** Add loading spinner on tap, disable button, show success message

**Dimension 3 - Micro-interactions:** ⚠ Warning
- No press animation
- **Recommendation:** Add subtle scale-down (98%) on active state, 100ms

**Dimension 4 - State Changes:** ✗ Fail
- Disabled state not visually distinct (same color)
- **Recommendation:** Change to opacity 50% when disabled, cursor: not-allowed

**Dimension 5 - Touch Targets:** ✓ Pass
- 56px height, 48px meets minimum
- Full-width button, easy to tap

**Priority Fixes:**
1. Add loading/disabled states (critical - prevents double-submit)
2. Add success confirmation (high - user needs confidence)
3. Add press animation (low - polish, not critical)

---

## Interaction Pattern Library

### Button Interaction Specs

**Primary Button (Full Spec):**
```
Default:
  - Background: Brand color (#0066CC)
  - Text: White, 16px, 600 weight
  - Padding: 12px 24px
  - Height: 48px
  - Border-radius: 8px
  - Shadow: 0 2px 4px rgba(0,0,0,0.1)

Hover (desktop):
  - Background: Lighter brand color (#0077DD)
  - Shadow: 0 4px 8px rgba(0,0,0,0.15)
  - Transition: all 100ms ease-out
  - Cursor: pointer

Focus (keyboard):
  - Outline: 2px solid #0066CC
  - Outline-offset: 2px

Active (pressed):
  - Background: Darker brand color (#0055BB)
  - Shadow: 0 1px 2px rgba(0,0,0,0.1)
  - Transform: scale(0.98)
  - Transition: none (immediate)

Disabled:
  - Background: #CCCCCC
  - Text: #666666
  - Shadow: none
  - Cursor: not-allowed
  - Opacity: 0.6

Loading:
  - Same as disabled
  - Plus: Spinner icon (16px, white, centered)
  - Text: "Processing..." or hidden
```

### Form Input Interaction Specs

**Text Input (Full Spec):**
```
Default:
  - Border: 1px solid #CCCCCC
  - Background: White
  - Padding: 12px 16px
  - Height: 48px
  - Font: 16px (prevents zoom on iOS)
  - Placeholder: #999999

Focus:
  - Border: 2px solid #0066CC
  - Box-shadow: 0 0 0 3px rgba(0,102,204,0.1)
  - Placeholder: fade out or move up (floating label)

Filled (valid):
  - Border: 1px solid #CCCCCC
  - Optional: 1px solid #00CC66 (green)
  - Checkmark icon: inside right (16px)

Error:
  - Border: 2px solid #CC0000
  - Box-shadow: 0 0 0 3px rgba(204,0,0,0.1)
  - Error message: Below field, #CC0000, 14px
  - Error icon: inside right (16px)

Disabled:
  - Background: #F5F5F5
  - Border: 1px solid #DDDDDD
  - Text: #999999
  - Cursor: not-allowed
```

### Modal Interaction Specs

**Modal Dialog (Full Spec):**
```
Entry Animation:
  - Backdrop: Fade in (opacity 0 → 0.5, 200ms ease-out)
  - Modal: Scale up (90% → 100%, 200ms ease-out)
  - Modal: Translate up (0 → -20px, 200ms ease-out)
  - Modal: Fade in (opacity 0 → 1, 200ms ease-out)

Exit Animation:
  - Backdrop: Fade out (200ms ease-in)
  - Modal: Scale down (100% → 95%, 200ms ease-in)
  - Modal: Fade out (200ms ease-in)

Interactions:
  - Backdrop click: Close modal
  - ESC key: Close modal
  - Focus trap: Tab cycles within modal
  - Close button: Top-right, 32x32px minimum

Positioning:
  - Centered horizontally and vertically
  - Max-width: 600px (desktop)
  - Full-width minus 32px margin (mobile)
  - Scrollable if content exceeds viewport
```

---

## Red Flags & Anti-Patterns

### Critical Issues (Fix Immediately)

1. **Touch targets <40px** - Unusable on mobile, accessibility fail
2. **No loading feedback** - Users will click multiple times
3. **Generic error messages** - Users can't fix the problem
4. **Disabled without visual indication** - Users confused why click doesn't work
5. **No keyboard navigation** - Excludes keyboard-only users, WCAG fail

### Major Issues (Fix Soon)

6. **Slow animations (>500ms)** - Users feel impatience
7. **No success confirmation** - Users uncertain if action completed
8. **Inconsistent state styling** - Creates cognitive load
9. **Hover-only critical actions** - Fails on mobile/touch devices
10. **Tiny spacing between targets** - Accidental taps, frustration

### Minor Issues (Nice to Fix)

11. **No micro-interactions** - Feels less polished
12. **Linear easing** - Feels mechanical, not natural
13. **Missing focus indicators** - Keyboard nav unclear (but browser defaults exist)
14. **No haptic feedback** - Missed opportunity for polish (mobile)
15. **Generic spinner** - Missed branding opportunity

---

## Related Skills

**Core UX Skills:**
- `lyra/ux-designer/visual-design-foundations` - Visual feedback, button styling, state colors
- `lyra/ux-designer/accessibility-and-inclusive-design` - Keyboard navigation, focus indicators, touch target sizing
- `lyra/ux-designer/ux-fundamentals` - Interaction design principles, affordances, feedback concepts

**Platform Extensions:**
- `lyra/ux-designer/mobile-design-patterns` - Touch targets (44x44pt iOS, 48x48dp Android), gestures, platform interactions
- `lyra/ux-designer/web-application-design` - Keyboard shortcuts, hover states, responsive interactions
- `lyra/ux-designer/desktop-software-design` - Keyboard-first workflows, focus management, window interactions
- `lyra/ux-designer/game-ui-design` - Gamepad navigation, immediate feedback, performance-optimized interactions

**Cross-Faction:**
- `ordis/security-architect/threat-modeling` - Secure interaction patterns (prevent double-submit, rate limiting UI)

---

## Additional Resources

**Animation Timing References:**
- Material Design Motion: 150-300ms standard
- iOS Human Interface Guidelines: Fast animations (<300ms)
- Google Material: Ease-out for entrances, ease-in for exits

**Touch Target Standards:**
- Apple HIG: 44x44pt minimum
- Material Design: 48x48dp minimum
- WCAG 2.1 AA: 44x44px minimum (Level AAA: larger)

**Gesture Standards:**
- iOS Human Interface Guidelines: Gestures chapter
- Material Design: Gestures documentation
- Platform conventions trump custom gestures

**Accessibility Standards:**
- WCAG 2.1 SC 2.5.5: Target Size (44x44px minimum, Level AAA)
- WCAG 2.1 SC 2.2.4: Interruptions (users can control timing)
- WCAG 2.1 SC 2.3.3: Animation from Interactions (can be disabled)
