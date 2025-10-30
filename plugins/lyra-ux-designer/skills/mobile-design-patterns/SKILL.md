---
name: mobile-design-patterns
description: Use when designing iOS/Android apps - iOS HIG and Material Design patterns, touch interactions (44x44pt iOS, 48x48dp Android), gesture conventions, reachability zones, and platform-specific visual consistency
---

# Mobile Design Patterns

## Overview

This skill provides **The Mobile Interaction Evaluation Model**: a systematic 4-dimension framework for designing native mobile applications (iOS and Android). Use this when designing for touch-first interfaces, platform-specific patterns, or optimizing for mobile constraints.

**Core Principle**: Mobile design requires balancing platform conventions with usability under real-world constraints (one-handed use, variable connectivity, outdoor viewing, interruptions). Success means feeling native to the platform while remaining accessible to the human hand and thumb.

**Platform Focus**: iOS (Human Interface Guidelines) and Android (Material Design), with guidance on when to unify or differentiate.

## When to Use This Skill

**Use this skill when:**
- Designing native iOS or Android applications
- Implementing touch-based interactions and gestures
- Ensuring platform consistency (iOS HIG vs Material Design)
- Optimizing for one-handed use and thumb reachability
- Specifying touch target sizes, navigation patterns, or mobile-specific components
- User mentions: "mobile app", "iOS", "Android", "touch targets", "gestures", "native design"

**Don't use this skill for:**
- Desktop software (use `lyra/ux-designer/desktop-software-design`)
- Responsive web design (use `lyra/ux-designer/web-application-design`)
- Game UI (use `lyra/ux-designer/game-ui-design`)
- Visual hierarchy alone (use `lyra/ux-designer/visual-design-foundations`)
- Accessibility evaluation (use `lyra/ux-designer/accessibility-and-inclusive-design`, though it complements this skill)

---

## The Mobile Interaction Evaluation Model

A systematic 4-dimension framework for evaluating and specifying mobile design:

1. **REACHABILITY** - Thumb zones, one-handed use, navigation placement
2. **GESTURE CONVENTIONS** - Platform-specific gestures (iOS swipe back, Android bottom nav)
3. **PLATFORM VISUAL CONSISTENCY** - iOS HIG vs Material Design visual language
4. **PERFORMANCE PERCEPTION** - Optimistic UI, skeleton screens, instant feedback

Evaluate designs by examining each dimension systematically, identifying platform-specific requirements, and ensuring designs feel native while remaining ergonomically sound.

---

## Dimension 1: REACHABILITY

**Purpose:** Design for comfortable one-handed use with thumb-driven navigation

The human thumb reaches comfortably to the bottom and center of a mobile screen. Top corners require hand repositioning or two-handed use. Design should place primary actions in the natural thumb zone (bottom 50-60% of screen) and secondary actions in harder-to-reach areas.

### Evaluation Questions

1. **Are primary actions in the thumb zone?**
   - Bottom 50-60% of screen: Most reachable
   - Center: Comfortable for most grips
   - Top corners: Hardest to reach (avoid critical actions)

2. **Does navigation placement follow platform conventions?**
   - iOS: Tab bar at bottom (5 items max)
   - Android: Bottom navigation (3-5 destinations) or navigation drawer
   - Top navigation only for secondary/browse contexts

3. **Can users complete primary tasks one-handed?**
   - Critical actions within thumb arc
   - No stretching required for frequent actions
   - Two-handed interactions reserved for secondary tasks

4. **Are destructive actions protected?**
   - Delete/dangerous actions not in easy-tap zones
   - Confirmation dialogs for irreversible actions
   - Swipe-to-delete requires intentional gesture

### Thumb Zone Mapping

**Right-Handed Thumb Zones (mirror for left-handed):**

```
┌─────────────────────────┐
│  HARD    │    HARD      │  ← Top corners: Hardest reach
│  (Back)  │  (Settings)  │     Reserve for secondary actions
├──────────┴──────────────┤
│                         │
│   COMFORTABLE STRETCH   │  ← Upper middle: Reachable with stretch
│     (Content area)      │     Place scrollable content here
│                         │
├─────────────────────────┤
│                         │
│    NATURAL THUMB ARC    │  ← Bottom 50%: Most comfortable
│   (Primary actions)     │     Place navigation & primary CTAs
│                         │
└─────────────────────────┘
     ↑ Bottom: Easiest reach
     (Tab bar, bottom nav, primary buttons)
```

**Specific Zone Recommendations:**

- **Easy (Bottom 1/3):** Tab bars, bottom navigation, primary CTAs, FABs
- **Comfortable (Middle 1/3):** Scrollable content, secondary actions, list items
- **Stretch (Top 1/3):** Back button, settings, page titles, status indicators

### iOS Navigation Patterns

**iOS Tab Bar (Bottom Navigation):**
- **Position:** Fixed at bottom of screen
- **Height:** 49pt (without safe area) or 83pt (with iPhone X+ safe area)
- **Items:** 3-5 tabs maximum (5 max per Apple HIG)
- **Icon size:** 25x25pt (image assets at 1x, 2x, 3x)
- **Label:** Optional but recommended (better clarity)
- **State:** Selected tab uses tint color, unselected uses gray
- **Behavior:** Tapping selected tab scrolls to top

```
Example Tab Bar Specs:
- Container height: 49pt + safe area inset
- Item spacing: Equal distribution
- Icon: 25x25pt template image
- Label: 10pt SF Pro font
- Selected color: Brand color (e.g., #007AFF iOS blue)
- Unselected color: #8E8E93 (iOS gray)
```

**iOS Navigation Bar (Top):**
- **Height:** 44pt (standard) or 96pt (large title)
- **Back button:** Top-left (platform convention)
- **Title:** Center or large title (scrolls to center on scroll)
- **Actions:** Top-right (1-2 buttons max)
- **Translucent:** Blur effect behind (iOS standard)

**When to Use:**
- Tab bar: 3-5 top-level sections, flat hierarchy
- Navigation bar: Hierarchical navigation, drill-down flows

### Android Navigation Patterns

**Bottom Navigation (Material Design):**
- **Position:** Fixed at bottom
- **Height:** 56dp
- **Items:** 3-5 destinations (Material Design spec)
- **Icon size:** 24x24dp
- **Label:** Always show for 3 items, optional for 4-5 items
- **Elevation:** 8dp shadow
- **Behavior:** No scroll-to-top on re-tap (unlike iOS)

```
Example Bottom Nav Specs:
- Container height: 56dp
- Item icon: 24x24dp
- Label: 12sp Roboto Medium
- Active color: Primary color (#6200EE Material purple)
- Inactive color: #757575 (Material gray)
- Elevation: 8dp
```

**Navigation Drawer (Side Menu):**
- **Width:** 256dp (mobile), 320dp (tablet)
- **Activation:** Left edge swipe or hamburger menu button
- **Content:** App-level navigation, settings, account
- **Header:** 64dp height with branding/account info
- **Items:** 48-56dp height, 16dp left padding for icon

**Top App Bar:**
- **Height:** 56dp (mobile), 64dp (tablet/desktop)
- **Navigation icon:** Left (24x24dp hamburger or back arrow)
- **Title:** 20sp Roboto Medium
- **Action icons:** Right (24x24dp, 48dp touch target)

**When to Use:**
- Bottom navigation: 3-5 top-level destinations, primary navigation
- Navigation drawer: 6+ destinations, or mixed hierarchy
- Top app bar: Always (provides context and actions)

### Floating Action Button (FAB)

**Material Design FAB Specs:**
- **Size:** 56dp diameter (default) or 40dp (mini)
- **Position:** Bottom-right, 16dp from edges
- **Elevation:** 6dp (resting), 12dp (pressed)
- **Icon:** 24dp, centered
- **Color:** Primary or secondary brand color
- **Purpose:** Single primary action for screen

**Usage Guidelines:**
- One FAB per screen maximum
- Primary action only (e.g., "Compose", "Add", "Create")
- Not for navigation or minor actions
- Can extend on scroll to show label

**iOS Alternative:**
- iOS doesn't have FAB convention
- Use toolbar button or prominent button in content
- Or adopt FAB if it's your brand pattern (e.g., Google apps on iOS)

### One-Handed Optimization Checklist

✓ **Primary actions in bottom 50% of screen**
✓ **Navigation at bottom (tab bar or bottom nav)**
✓ **Secondary actions in top (back, settings, filters)**
✓ **Scrollable content in middle (comfortable reading)**
✓ **Large touch targets (44x44pt iOS, 48x48dp Android)**
✓ **Spacing between targets (8dp/pt minimum)**
✓ **Two-handed gestures optional (pinch zoom, not required for core tasks)**

---

## Dimension 2: GESTURE CONVENTIONS

**Purpose:** Follow platform-specific gesture standards for intuitive, discoverable interactions

Mobile users expect platform-standard gestures. Custom gestures create friction and require learning. Always prioritize platform conventions over custom interactions unless you have exceptional reason (and user testing) to deviate.

### Evaluation Questions

1. **Do gestures follow platform conventions?**
   - iOS: Swipe back from left edge, pull to refresh
   - Android: System back button, edge swipe for navigation drawer
   - Cross-platform: Pinch to zoom, swipe to delete

2. **Are gestures discoverable?**
   - Visual hints for non-standard gestures
   - Tutorial on first use (if custom)
   - Alternative button-based actions available

3. **Do gestures have immediate feedback?**
   - Visual response within 100ms
   - Follow gesture with finger (drag, swipe)
   - Haptic feedback on success/completion (optional but nice)

4. **Are conflicts avoided?**
   - Custom gestures don't override system gestures
   - Swipe directions have consistent meanings
   - Long-press doesn't interfere with tap

### iOS Standard Gestures

**Swipe Back Navigation (System-Wide):**
- **Gesture:** Swipe right from left edge of screen
- **Behavior:** Navigate back in hierarchy
- **Feedback:** Page slides right, revealing previous page underneath
- **System-level:** Works in all apps by default (UINavigationController)
- **Design implication:** Don't put critical actions on left edge (conflicts with swipe)

```
Implementation note:
- Enabled by default in UINavigationController
- Can be disabled but shouldn't (user expectation)
- Custom back gestures should augment, not replace
```

**Pull to Refresh:**
- **Gesture:** Pull down on scrollable content
- **Feedback:** Spinner appears at top, elastic bounce
- **Use case:** Refresh lists, feeds, mailboxes
- **System component:** UIRefreshControl
- **Timing:** Trigger after ~60pt pull distance

**Swipe Actions on Lists:**
- **Gesture:** Swipe left or right on list item
- **Feedback:** Action buttons reveal behind item
- **Common patterns:**
  - Swipe left: Delete (red), Archive (blue)
  - Swipe right: Mark as read, Favorite
- **Full swipe:** Can auto-trigger action without showing buttons
- **Color coding:** Red for destructive, blue/green for non-destructive

```
iOS Swipe Actions Specs:
- Button height: Match row height (minimum 44pt)
- Button width: 74pt per button (typical)
- Icon + text or icon-only
- Full swipe threshold: ~50% of row width
```

**Long Press (Context Menu):**
- **Gesture:** Touch and hold for 500ms
- **Feedback:** Haptic tap, menu appears (iOS 13+: blur background)
- **Use case:** Preview (peek), context actions, copy/paste
- **iOS 13+:** Context menu with preview and actions
- **Design:** Provide shortcuts, not primary actions

**3D Touch / Haptic Touch:**
- **Deprecated:** 3D Touch removed in iPhone 11+
- **Replaced by:** Long press (Haptic Touch)
- **Design:** Don't rely on pressure sensitivity

**Pinch to Zoom:**
- **Gesture:** Two-finger pinch (spread to zoom in, pinch to zoom out)
- **Use case:** Images, maps, PDFs, web content
- **Feedback:** Content scales with gesture
- **Alternative:** Double-tap to zoom in (toggles 2x zoom)

### Android Standard Gestures

**System Navigation (Android 10+ Gesture Nav):**
- **Swipe up from bottom:** Go home
- **Swipe up and hold:** Recent apps / multitasking
- **Swipe from left or right edge:** Back navigation
- **Design implication:** Avoid critical actions on screen edges

**Note:** Android supports 3 navigation modes:
1. Gesture navigation (modern, default on Android 10+)
2. 2-button navigation (back button + home button)
3. 3-button navigation (back, home, recents - legacy)

Design should work with all three modes.

**Navigation Drawer (Left Edge Swipe):**
- **Gesture:** Swipe right from left edge
- **Behavior:** Open navigation drawer (side menu)
- **Feedback:** Drawer slides in from left
- **Component:** DrawerLayout (Material Design)
- **Design:** Show hamburger menu icon to hint drawer presence

**Pull to Refresh:**
- **Gesture:** Pull down on scrollable content
- **Feedback:** Circular spinner, Material motion
- **Component:** SwipeRefreshLayout
- **Timing:** Similar to iOS (~60dp pull to trigger)

**Swipe to Dismiss:**
- **Gesture:** Swipe left or right on cards, notifications
- **Feedback:** Item fades and slides out
- **Use case:** Notifications, recent apps, dismissable cards
- **Undo:** Show Snackbar with "Undo" option (Material pattern)

**Long Press:**
- **Gesture:** Touch and hold for 500ms
- **Feedback:** Haptic feedback (if available), context menu appears
- **Use case:** Select mode (multi-select), app shortcuts (home screen)
- **Design:** Similar to iOS long-press

**Bottom Sheet Swipe:**
- **Gesture:** Swipe up on bottom sheet to expand, down to collapse
- **Feedback:** Sheet follows finger, snaps to states
- **States:** Collapsed, half-expanded, fully expanded
- **Use case:** Contextual actions, maps (Google Maps), music players

### Cross-Platform Universal Gestures

**Tap (Primary Interaction):**
- **Minimum target size:** 44x44pt (iOS), 48x48dp (Android)
- **Feedback:** Visual press state, ripple (Android), or opacity change (iOS)
- **Timing:** Immediate (<100ms)

**Double-Tap:**
- **Use case:** Zoom in (images, maps), like (Instagram heart)
- **Feedback:** Zoom animation or visual confirmation
- **Timing window:** Second tap within 300ms

**Scroll (Vertical Swipe):**
- **Gesture:** Swipe up/down in content area
- **Behavior:** Inertial scrolling (continues after release)
- **Edge behavior:** Bounce (iOS), glow (Android)

**Horizontal Swipe:**
- **Use case:** Carousels, image galleries, onboarding flows
- **Feedback:** Content slides horizontally
- **Pagination:** Snap to pages or free-scroll

**Drag:**
- **Use case:** Reorder lists, move items, pan maps
- **Feedback:** Item lifts (shadow increases), follows finger
- **States:** Rest → Lift (shadow) → Drag (follows) → Drop (animates to position)

**Pinch (Two-Finger):**
- **Use case:** Zoom in/out (maps, images, PDFs)
- **Feedback:** Content scales continuously with gesture
- **Alternative:** Provide zoom buttons for accessibility

### Gesture Discovery Patterns

**Problem:** Custom gestures are invisible to users

**Solutions:**

1. **First-Time Hints:**
   - Transparent overlay with animation: "Swipe left to archive"
   - Dismiss after first use or after 3 seconds
   - Don't show again (store in user preferences)

2. **Visual Affordances:**
   - Partially visible next screen (hint swipe)
   - Bounce animation on first load (suggests pull down)
   - Handle or drag indicator (bottom sheets)

3. **Progressive Disclosure:**
   - Animate gesture on empty states
   - Show partial swipe action on first interaction
   - Tutorial screens for complex gestures

4. **Always Provide Alternative:**
   - Button for swipe action (three-dot menu)
   - Long-press for quick actions
   - Visible controls alongside gesture shortcuts

### Gesture Conflicts to Avoid

❌ **Left edge swipe with custom action (iOS)**
- Conflicts with system back gesture
- Users will accidentally trigger back navigation

❌ **Bottom edge swipe (Android 10+)**
- Conflicts with system home gesture
- Reserve bottom 16dp for system gestures

❌ **Horizontal scroll inside vertical scroll**
- Gesture ambiguity (which direction?)
- Use tabs or clear horizontal zones

❌ **Custom pinch gesture with zoom-enabled content**
- Conflicts with expected zoom behavior
- Users expect pinch = zoom universally

### Platform-Specific Gesture Specs

**iOS Gesture Timing:**
- Tap: <100ms to register
- Long press: 500ms hold
- Swipe velocity threshold: 500 points/second
- Scroll deceleration: Exponential (feels natural)

**Android Gesture Timing:**
- Tap: <100ms to register
- Long press: 500ms hold (ViewConfiguration.getLongPressTimeout())
- Swipe velocity threshold: 100-1000 dp/second (ViewConfiguration.getScaledMinimumFlingVelocity())
- Scroll deceleration: Android physics (different from iOS)

**Haptic Feedback:**
- **iOS:** UIImpactFeedbackGenerator (light, medium, heavy)
- **Android:** HapticFeedbackConstants (VIRTUAL_KEY, LONG_PRESS, etc.)
- **Use sparingly:** Confirmation, error, important events only
- **Make optional:** Some users disable haptics in settings

---

## Dimension 3: PLATFORM VISUAL CONSISTENCY

**Purpose:** Match OS design conventions while maintaining brand identity

Users expect apps to feel native to their platform. iOS users expect iOS visual patterns; Android users expect Material Design. Deviating creates cognitive friction and "uncanny valley" feelings of wrongness.

### Evaluation Questions

1. **Does the design follow platform conventions?**
   - iOS: Translucent nav bars, SF Pro font, large titles, rounded cards
   - Android: Elevation/shadows, Roboto font, FABs, Snackbars

2. **Are typography choices platform-appropriate?**
   - iOS: SF Pro (system default), large titles (34pt), dynamic type support
   - Android: Roboto (system default), distinct type scale, Material typography

3. **Do elevation/shadows match platform expectations?**
   - iOS: Subtle shadows, translucency, blur effects
   - Android: Elevation system (2dp, 4dp, 8dp), z-axis hierarchy

4. **Are navigation components platform-standard?**
   - iOS: Tab bar (bottom), navigation bar (top), action sheets
   - Android: Bottom nav or drawer, top app bar, bottom sheets

5. **Do UI components match OS design language?**
   - iOS: Segmented controls, switches, activity indicators
   - Android: Tabs, switches, progress indicators, ripple effects

### iOS Human Interface Guidelines (HIG)

**iOS Typography:**

```
SF Pro (System Font):
- Large Title: 34pt, Bold (Navigation bars, scrolls to 17pt)
- Title 1: 28pt, Regular (Page titles)
- Title 2: 22pt, Regular (Section headers)
- Title 3: 20pt, Regular (Subsections)
- Headline: 17pt, Semibold (Emphasized content)
- Body: 17pt, Regular (Primary content)
- Callout: 16pt, Regular (Secondary content)
- Subhead: 15pt, Regular (Tertiary content)
- Footnote: 13pt, Regular (Timestamps, meta)
- Caption 1: 12pt, Regular (Image captions)
- Caption 2: 11pt, Regular (Smallest readable text)
```

**Dynamic Type:**
- iOS allows users to scale text size (Settings → Display & Brightness → Text Size)
- Use system text styles to support automatic scaling
- Test at largest and smallest sizes
- Don't use fixed pixel sizes for text

**iOS Visual Patterns:**

**Translucency:**
- Navigation bars: Translucent blur effect (shows content beneath)
- Tab bars: Translucent with blur
- Toolbars: Translucent
- System overlays: Heavy blur

**Rounded Corners:**
- Cards: 13pt corner radius (continuous curve, not circular)
- Buttons: 8-10pt corner radius
- Modals: 13pt corner radius (top corners only)
- Alerts: 13pt corner radius

**Shadows:**
- Subtle: iOS uses minimal shadows
- Prefer translucency and separators over heavy shadows
- Cards: 0 2px 8px rgba(0,0,0,0.1) (light shadow)

**Borders and Separators:**
- Hairline separators: 0.5pt (1px at 2x) in light gray
- Minimal use of borders (rely on spacing and hierarchy)
- Inset separators: Indented 16pt from left (list items)

**Color:**
- System colors: Adapt to light/dark mode automatically
- Tint color: Single accent color for interactive elements (default #007AFF)
- Gray scale: #F2F2F7 (light gray) to #1C1C1E (dark gray)
- Support both light and dark modes (required as of iOS 13)

**iOS Components:**

**Tab Bar:**
- Height: 49pt + safe area
- Items: 3-5 maximum
- Icons: 25x25pt, line style (not filled unless selected)
- Selected: Filled icon + tint color
- Labels: 10pt, always visible

**Navigation Bar:**
- Height: 44pt (standard) or 96pt (large title)
- Large title: 34pt Bold, scrolls to 17pt Regular center title
- Back button: Chevron + previous screen title (or "Back")
- Actions: Text buttons or icons (top-right)

**Segmented Control:**
- Grouped buttons with single selection
- Background: Light gray rounded rectangle
- Selected: White background (iOS 13+)
- Height: 28-32pt

**Action Sheet:**
- Modal from bottom with actions
- iOS style: Rounded corners, translucent background
- Destructive action: Red text
- Cancel: Separate at bottom

**iOS Motion:**
- Spring animations (UISpringTimingParameters)
- Natural, physics-based easing
- Respects Reduce Motion accessibility setting
- Duration: 200-300ms typical

### Android Material Design

**Android Typography:**

```
Roboto (System Font):
- H1: 96sp, Light, -1.5sp letter spacing
- H2: 60sp, Light, -0.5sp letter spacing
- H3: 48sp, Regular, 0sp letter spacing
- H4: 34sp, Regular, 0.25sp letter spacing
- H5: 24sp, Regular, 0sp letter spacing (Page titles)
- H6: 20sp, Medium, 0.15sp letter spacing (Section headers)
- Subtitle 1: 16sp, Regular, 0.15sp letter spacing
- Subtitle 2: 14sp, Medium, 0.1sp letter spacing
- Body 1: 16sp, Regular, 0.5sp letter spacing (Primary content)
- Body 2: 14sp, Regular, 0.25sp letter spacing
- Button: 14sp, Medium, 1.25sp letter spacing, ALL CAPS
- Caption: 12sp, Regular, 0.4sp letter spacing (Meta info)
- Overline: 10sp, Regular, 1.5sp letter spacing, ALL CAPS
```

**Note:** Android uses "sp" (scale-independent pixels) for text, which respects user font size settings.

**Material Elevation System:**

Material Design uses elevation (z-axis) to create hierarchy and show relationships:

```
Elevation (shadows):
- 0dp: Flat on surface (text, backgrounds)
- 1dp: Cards (resting state)
- 2dp: Buttons (resting state), app bar (scrolled)
- 4dp: App bar (resting state)
- 6dp: FAB (resting state), snackbar
- 8dp: Bottom navigation, drawer, modal side sheet
- 9dp: FAB (pressed), dialog, time picker
- 12dp: FAB (pressed, alternative spec)
- 16dp: Navigation drawer (while opening)
- 24dp: Modal bottom sheet
```

**Elevation creates shadows:**
- Higher elevation = larger, softer shadow
- Shadow color: rgba(0,0,0,0.14) for ambient, rgba(0,0,0,0.20) for key light
- Elevation changes on interaction (press = +6dp to +12dp for FAB)

**Material Visual Patterns:**

**Rounded Corners:**
- Cards: 4dp corner radius
- Buttons: 4dp corner radius
- Dialogs: 4dp corner radius
- Bottom sheets: 8dp corner radius (top corners only)
- More pronounced than iOS (4-8dp vs 8-13pt)

**Color:**
- Primary color: Brand color for main UI elements
- Secondary color: Accent color for FABs, highlights
- Surface colors: White (light theme), #121212 (dark theme)
- Error: #B00020 (Material red)
- Background elevation: Lighter overlays on dark theme (8% white per elevation step)

**Ripple Effect:**
- Touch feedback: Circular ripple emanating from touch point
- Color: 12% opacity of foreground color
- Duration: 300ms fade out
- All interactive elements should have ripple (Material standard)

**Android Components:**

**Bottom Navigation:**
- Height: 56dp
- Items: 3-5 destinations
- Icons: 24dp, active/inactive states
- Labels: 12sp, show for 3 items, optional for 4-5
- Ripple on tap
- Elevation: 8dp

**Top App Bar:**
- Height: 56dp (mobile), 64dp (tablet/desktop)
- Title: 20sp Medium
- Icons: 24dp (48dp touch target)
- Elevation: 4dp (0dp if scrolled to top, optional)

**FAB (Floating Action Button):**
- Size: 56dp (default) or 40dp (mini)
- Icon: 24dp centered
- Elevation: 6dp (resting), 12dp (pressed)
- Ripple: Circular outward
- Position: 16dp from edges (bottom-right typical)

**Snackbar:**
- Height: 48-80dp (single/multi-line)
- Position: Bottom of screen (above bottom nav if present)
- Duration: 4-10 seconds or indefinite
- Action: Optional text button (right side)
- Elevation: 6dp

**Bottom Sheet:**
- Swipe up to expand, down to collapse
- Corner radius: 8dp (top corners)
- States: Collapsed, half-expanded, fully expanded
- Handle: Optional 32x4dp rounded indicator at top
- Elevation: 16dp (modal) or 1dp (persistent)

**Material Motion:**
- Easing curves: Material easing (cubic-bezier(0.4, 0.0, 0.2, 1))
- Duration: 150-300ms for most transitions
- Choreography: Elements move together in meaningful ways
- Shared element transitions: Smooth transitions between screens

### When to Deviate from Platform Conventions

**Strong Brand Identity:**
- **Example:** Instagram, Spotify, Netflix use custom UI across platforms
- **Rationale:** Brand consistency trumps platform consistency
- **Risk:** Slightly steeper learning curve, but offset by familiarity with brand
- **Rule:** If your brand is strong enough, users will adapt

**Cross-Platform Consistency:**
- **Example:** Google apps on iOS use Material Design (Gmail, Drive, Photos)
- **Rationale:** Consistent experience across devices for multi-platform users
- **Risk:** Feels "wrong" to platform purists
- **Rule:** If your users switch platforms frequently, consistency helps

**Unique Use Case:**
- **Example:** Creative tools (Procreate, Figma), games, AR/VR apps
- **Rationale:** Standard patterns don't fit the interaction model
- **Risk:** Requires custom patterns, user education
- **Rule:** Deviate only if platform patterns genuinely don't support your use case

**Careful Deviation:**
- **Test extensively:** Custom patterns require more usability testing
- **Provide onboarding:** Teach custom interactions
- **Maintain accessibility:** Platform conventions are accessible by default
- **Respect system gestures:** Don't override system back, home, etc.

**Never Deviate On:**
- System gestures (back, home, app switcher)
- Accessibility features (VoiceOver, TalkBack, Dynamic Type, large text)
- Status bar, safe areas, notches (use system APIs)
- System permissions and dialogs

### Supporting Both Light and Dark Modes

**iOS:**
- **Required:** All apps must support dark mode (iOS 13+)
- **System colors:** Use UIColor.label, .systemBackground, etc. (adapt automatically)
- **Custom colors:** Define light and dark variants in asset catalog
- **Test:** Switch in Settings → Developer → Dark Appearance

**Android:**
- **Recommended:** Support dark theme (Material Design standard, Android 10+)
- **System colors:** Use ?attr/colorPrimary, ?attr/colorSurface (adapt automatically)
- **Dark theme elevation:** Lighter overlays on elevated surfaces (8% white per step)
- **Test:** Settings → Display → Dark theme

**Dark Mode Best Practices:**
- Don't use pure black (#000000) for backgrounds (use #121212 on Android, system background on iOS)
- Reduce saturation of bright colors (lower luminance)
- Use elevation to create hierarchy (not just borders)
- Test contrast ratios (still need 4.5:1 for text on dark backgrounds)
- Avoid pure white text (use off-white for less eye strain)

---

## Dimension 4: PERFORMANCE PERCEPTION

**Purpose:** Create the perception of speed even when actual processing takes time

Users judge app quality by perceived performance, not actual milliseconds. An app that feels instant (even if loading in background) beats an app that's fast but provides no feedback. Optimistic UI, skeleton screens, and instant feedback are critical for mobile where network is unreliable.

### Evaluation Questions

1. **Does the app provide instant feedback (<100ms)?**
   - Button press visual change
   - Tap acknowledgment
   - Action initiated indicator

2. **Are skeleton screens used during content loading?**
   - Show content structure while loading
   - Better than blank screens or spinners
   - Smooth transition to real content

3. **Is optimistic UI used for actions?**
   - Assume success, show result immediately
   - Rollback if action fails
   - Reduces perceived latency

4. **Is offline functionality supported?**
   - Cached content available offline
   - Queue actions for later sync
   - Clear indicators of offline state

5. **Are loading states appropriate for duration?**
   - <1 second: No indicator (instant)
   - 1-10 seconds: Spinner or progress indicator
   - >10 seconds: Progress bar with percentage (if calculable)

### Optimistic UI

**Concept:** Assume user actions will succeed and update UI immediately, rollback if failure occurs.

**When to Use:**
- Actions with high success rate (>95%)
- Network-dependent actions (post, like, save)
- Actions with visible state (checkbox, toggle, star)

**When NOT to Use:**
- Destructive actions (delete account, permanent delete)
- Financial transactions (payment processing)
- Actions with complex server-side validation

**Pattern Example: Like Button**

```
User taps "Like":
1. Immediate: Change icon from outline to filled (0ms)
2. Immediate: Increment like count (0ms)
3. Background: Send API request
4. On success (500ms later): No change, already updated
5. On failure: Revert icon and count, show error toast

Result: Feels instant, even with 500ms network latency
```

**Pattern Example: Post Creation**

```
User taps "Post":
1. Immediate: Add post to feed with "Posting..." indicator (0ms)
2. Immediate: Disable edit/delete for this post
3. Background: Upload to server
4. On success (2s later): Remove "Posting..." indicator, enable actions
5. On failure: Show "Failed to post" with Retry button, mark post in error state

Result: Feed updates instantly, user can continue browsing
```

**Rollback Strategy:**
- Visual indicator during sync ("Sending...", gray outline)
- Error state if fails (red outline, "Failed to send")
- Retry button or automatic retry (exponential backoff)
- Don't silently fail (user needs to know)

**Implementation Considerations:**
- Maintain optimistic state locally (local-first architecture)
- Track pending actions (queue)
- Handle conflicts (what if another user edited same item?)
- Provide manual refresh ("Pull to refresh") for paranoid users

### Skeleton Screens

**Concept:** Show content structure (gray boxes) while loading actual content.

**Better Than:**
- Blank white screen (looks broken)
- Spinner alone (no context, feels slow)
- Old content with spinner (confusing if content changes)

**Pattern Structure:**

```
Skeleton for News Feed:
┌─────────────────────────────┐
│ [Avatar] [Text line────]    │ ← User info placeholder
│          [Text line──]       │
│                              │
│ [Gray rectangle────────]     │ ← Image placeholder
│ [─────────────────────]      │
│                              │
│ [Text line──────────]        │ ← Content placeholder
│ [Text line────────]          │
│ [Text line──────]            │
└─────────────────────────────┘

Shimmer effect: Light sweep left-to-right (1.5s loop)
```

**Skeleton Guidelines:**
- Match actual content layout (same card size, spacing)
- Use neutral gray (#E0E0E0 light theme, #2C2C2C dark theme)
- Shimmer animation (optional): Light gradient sweeps across (suggests loading)
- Transition: Fade out skeleton, fade in real content (200ms)
- Number of skeletons: Match expected content (3-5 cards typical)

**Skeleton Components:**
- Avatar: Circle (40-48dp/pt)
- Text line: Rounded rectangle (12-16dp/pt height)
- Image: Rectangle (match aspect ratio of real images)
- Button: Rounded rectangle (48dp/pt height)

**When to Use:**
- Initial screen load (first 1-3 seconds)
- Infinite scroll (next page loading)
- Content refresh (pull-to-refresh)

**iOS Example:** Apple News, App Store (show skeleton cards while loading)
**Android Example:** YouTube, LinkedIn (show skeleton feed)

### Progressive Loading

**Concept:** Load content in stages, showing most important content first.

**Priority Layers:**

1. **Layout shell (0ms):** Navigation, screen structure, skeleton
2. **Critical content (500ms):** Headlines, titles, primary text
3. **Images (1-2s):** Load low-res placeholder, then high-res
4. **Non-critical (2-5s):** Ads, recommendations, analytics

**Pattern Example: Article Screen**

```
Stage 1 (0ms): Show nav bar, skeleton article layout
Stage 2 (200ms): Load and show article title, author, date (text is fast)
Stage 3 (500ms): Load article body text (streaming if long)
Stage 4 (1s): Load hero image (show low-res blur first)
Stage 5 (2s): Load high-res image, comments, related articles

User can start reading at Stage 2, well before images load.
```

**Image Loading Strategy:**

1. **Placeholder:** Solid color (average color of image) or blur hash
2. **Low-res:** 10-20px width, blurred, loads in <100ms
3. **High-res:** Full size, replaces low-res with fade transition (200ms)
4. **Lazy loading:** Images below fold load when scrolled into view

**iOS:** Use UIImage progressive loading or SDWebImage library
**Android:** Use Glide or Coil libraries (built-in progressive loading)

### Instant Feedback

**100ms Rule:** User must see visual response within 100ms of touch.

**Feedback Types:**

**Button Press (0-50ms):**
- Visual: Brightness change, scale down (98%), or shadow reduction
- iOS: Opacity 60% or slight scale
- Android: Ripple effect starts immediately

**Toggle Switch (0-50ms):**
- Visual: Circle starts sliding animation
- Haptic: Light tap feedback
- State: Assume toggled (optimistic)

**Text Input (0ms):**
- Visual: Character appears immediately (no perceptible lag)
- Feedback: Cursor blinks, keyboard haptic (optional)

**Action Completion (100-500ms):**
- Success: Checkmark, green color (brief animation)
- Error: Shake animation, red color, error message
- Processing: Spinner, "Saving..." text

**Timing Guidelines:**

- **<100ms:** Instant (no spinner needed)
- **100ms-1s:** Short wait (spinner or progress indicator)
- **1-10s:** Loading state (spinner with message: "Loading your feed...")
- **>10s:** Progress bar (show percentage if calculable)
- **>30s:** Consider background task with notification when complete

**Long Operations:**
- Show estimated time ("About 2 minutes remaining...")
- Allow cancellation (Cancel button)
- Allow background processing (minimize, app continues in background)
- Notify on completion (push notification)

### Offline-First Architecture

**Concept:** App works offline by default, syncs when connected.

**Core Patterns:**

**Cache Strategy:**
- Cache recently viewed content locally
- Serve from cache first, update from network in background (stale-while-revalidate)
- Show cache age: "Updated 5 minutes ago"

**Action Queue:**
- Queue user actions (like, post, save) when offline
- Auto-sync when connection restored
- Show pending actions: "3 items waiting to sync"

**Sync Indicators:**
- Offline mode: Banner at top ("No internet connection")
- Syncing: Small indicator ("Syncing..." in status area)
- Synced: Brief confirmation ("All changes saved")

**Conflict Resolution:**
- Last-write-wins (simple, works for most cases)
- Server-wins (important data, user must re-apply)
- Manual merge (complex, show both versions, let user choose)

**iOS:** NSURLSession background tasks, Core Data for local storage
**Android:** WorkManager for background sync, Room database for local storage

### Performance Perception Checklist

✓ **Instant feedback on all taps (<100ms visual response)**
✓ **Optimistic UI for common actions (like, save, post)**
✓ **Skeleton screens for initial load (not blank or spinner-only)**
✓ **Progressive image loading (low-res → high-res)**
✓ **Offline support (cache content, queue actions)**
✓ **Appropriate loading indicators (spinner for 1-10s, progress for >10s)**
✓ **Background sync (don't block user while syncing)**
✓ **Pull-to-refresh for manual updates (user control)**

---

## Mobile-Specific Constraints

### Screen Size Considerations

**Design for Smallest First:**
- **iOS:** iPhone SE (375x667pt at 2x) = 750x1334 physical pixels
- **Android:** Small phone (~360x640dp) = ~720x1280 physical pixels
- Test on smallest supported device, scale up for larger

**Common iOS Sizes:**
- iPhone SE: 375x667pt (4.7" diagonal)
- iPhone 13/14: 390x844pt (6.1")
- iPhone 14 Pro Max: 430x932pt (6.7")
- iPad Mini: 768x1024pt (8.3")
- iPad Pro: 1024x1366pt (12.9")

**Common Android Sizes:**
- Small: 360x640dp (5" phone)
- Medium: 411x731dp (6" phone)
- Large: 480x853dp (6.5" phone)
- Tablet: 600x960dp (7" tablet) and up

**Responsive Strategy:**
- Single column on phones (< 600dp wide)
- Two column on tablets (>= 600dp wide)
- Three column on large tablets (>= 960dp wide)
- Use safe areas for notches, rounded corners, home indicators

**Safe Areas:**
- **iOS:** Top inset (status bar, notch), bottom inset (home indicator)
- **Android:** Status bar, navigation bar (if 3-button), notches/camera cutouts
- **Always use system APIs:** UIView.safeAreaInsets (iOS), WindowInsets (Android)

### Touch Target Sizing

**Minimum Touch Target Sizes:**

```
iOS (Apple HIG):
- Minimum: 44x44pt (points, not pixels)
- Recommended: 48pt for primary actions
- Spacing: 8pt minimum between targets

Android (Material Design):
- Minimum: 48x48dp (density-independent pixels)
- Recommended: 56dp for primary actions (FABs, buttons)
- Spacing: 8dp minimum between targets

Web Mobile:
- Minimum: 48x48px (physical pixels)
- Recommended: 56px for primary actions
- Spacing: 8px minimum between targets
```

**Why 44/48 Minimum?**
- Average adult fingertip: 10mm (roughly 40-50px)
- Allows for imprecision, reduces accidental taps
- WCAG 2.1 Level AAA: 44x44px minimum

**Target Size Calculation:**

Even if visible button is smaller, ensure touch target is 44/48:

```
Example: Icon button with 24pt icon
- Visible icon: 24x24pt
- Touch target: 48x48pt (add 12pt padding on all sides)
- Implementation: Increase padding or use transparent hit area

iOS (SwiftUI):
.frame(width: 48, height: 48) // Touch target
.contentShape(Rectangle()) // Make entire frame tappable

Android (XML):
android:minWidth="48dp"
android:minHeight="48dp"
android:padding="12dp" // Icon is 24dp, padding makes it 48dp
```

**Spacing Between Targets:**

```
Minimum spacing: 8dp/pt (prevents accidental taps)
Comfortable spacing: 16dp/pt (easier, more forgiving)

Example: Three buttons in a row
[Button] <8dp> [Button] <8dp> [Button]
Each button 48x48dp, spacing 8dp = total 168dp width
```

**High-Density Interfaces:**
- Lists with 48dp height items: 8dp spacing acceptable (dividers help)
- Toolbars with multiple icons: 8dp spacing, 48dp targets (icons smaller, padding larger)
- Keyboards: Slightly smaller targets acceptable (40dp) due to muscle memory and two-handed use

### Context of Use

**One-Handed Operation:**
- Primary actions in thumb zone (bottom 50%)
- Bottom navigation (not top tabs)
- Avoid top-left corner for critical actions
- Test: Hold phone one-handed, reach with thumb

**Environmental Factors:**

**Sunlight (Outdoor Use):**
- High contrast critical (text: 7:1 or higher)
- Avoid subtle grays, low-contrast colors
- Bright UI backgrounds easier to read in sun
- Dark mode can be harder to read outdoors (less contrast with environment)

**Movement (Walking, Transit):**
- Larger touch targets (harder to be precise while moving)
- Less reliance on small text (harder to read while bouncing)
- Confirmation dialogs for destructive actions (prevent accidental taps)

**Interruptions:**
- Save state frequently (app can be backgrounded anytime)
- Auto-save form inputs (don't lose data on interruption)
- Resume where user left off (deep linking, state restoration)
- Handle calls, notifications, alarms gracefully

**Variable Connectivity:**

```
Connection Quality:
- WiFi: Fast, reliable (10+ Mbps typical)
- 4G LTE: Fast, mostly reliable (5-50 Mbps)
- 3G: Slow, variable (0.5-5 Mbps)
- Offline: No connection

Design Implications:
- Don't assume always online (offline-first architecture)
- Optimize images (WebP format, responsive sizes)
- Show loading indicators >1 second load time
- Allow manual refresh (pull-to-refresh)
- Cache aggressively (reduce network dependency)
```

**Low-Connectivity Strategy:**
- Serve cached content first (instant, even if stale)
- Update in background when connected
- Show sync status ("Updated 5 min ago")
- Defer non-critical content (ads, analytics, recommendations)
- Compress data (gzip, Brotli, protocol buffers)

---

## Platform Detection and Adaptation

### Detecting Platform at Runtime

**iOS (Swift):**
```swift
#if os(iOS)
  // iOS-specific code
  let tabBar = UITabBar() // iOS tab bar
#elseif os(macOS)
  // macOS-specific code
#endif

// Detect device type
if UIDevice.current.userInterfaceIdiom == .phone {
  // iPhone
} else if UIDevice.current.userInterfaceIdiom == .pad {
  // iPad
}
```

**Android (Kotlin):**
```kotlin
// Screen size category
val screenSize = resources.configuration.screenLayout and
                 Configuration.SCREENLAYOUT_SIZE_MASK

when (screenSize) {
    Configuration.SCREENLAYOUT_SIZE_SMALL -> // Small phone
    Configuration.SCREENLAYOUT_SIZE_NORMAL -> // Normal phone
    Configuration.SCREENLAYOUT_SIZE_LARGE -> // Large phone/small tablet
    Configuration.SCREENLAYOUT_SIZE_XLARGE -> // Tablet
}

// Width breakpoints
val widthDp = resources.configuration.screenWidthDp
if (widthDp >= 600) {
  // Tablet layout
}
```

**React Native (Cross-Platform):**
```javascript
import { Platform } from 'react-native';

if (Platform.OS === 'ios') {
  // iOS-specific code
  return <IOSTabBar />;
} else if (Platform.OS === 'android') {
  // Android-specific code
  return <AndroidBottomNav />;
}
```

### Responsive Layout Strategies

**Breakpoint-Based:**

```
iOS (Size Classes):
- Compact width: iPhone portrait, narrow iPhone landscape
- Regular width: iPad, wide iPhone landscape
- Compact height: iPhone landscape
- Regular height: iPhone portrait, iPad

Android (Width Breakpoints):
- < 600dp: Phone portrait
- 600-839dp: Phone landscape, small tablet portrait
- 840-1279dp: Large tablet portrait, medium tablet landscape
- >= 1280dp: Large tablet landscape, desktop
```

**Layout Adaptation:**

```
Phone (< 600dp):
- Single column
- Bottom navigation
- Full-width buttons
- Stacked content

Tablet (>= 600dp):
- Two columns (master-detail)
- Side navigation (drawer always visible or tabs)
- Floating modals (not full-screen)
- Multi-pane layouts
```

---

## Cross-Platform Design Strategy

### Option 1: Platform-Specific Design

**Approach:** Design separately for iOS and Android, following each platform's conventions.

**Pros:**
- Feels native to each platform
- Leverages platform-specific components
- Meets user expectations (iOS users expect iOS, Android users expect Android)

**Cons:**
- Double design effort (two designs, two specs)
- Double development effort (two implementations)
- Inconsistent across platforms (brand less recognizable)

**When to Use:**
- Native development (Swift/SwiftUI for iOS, Kotlin/Jetpack Compose for Android)
- Platform-first companies (most iOS users OR most Android users, not both)
- OS-level integration (widgets, Siri shortcuts, Android sharing)

**Example:** Apollo Reddit client (iOS-first, feels perfectly iOS-native)

### Option 2: Unified Design Language

**Approach:** Create single design system, apply to both platforms (ignore platform conventions).

**Pros:**
- Consistent experience across platforms
- Single design effort (one design, one spec)
- Easier to maintain (changes apply everywhere)
- Strong brand identity

**Cons:**
- Feels foreign on both platforms
- More custom development (can't use platform components)
- Steeper learning curve (users must learn your patterns)

**When to Use:**
- Strong brand identity (Instagram, Spotify, Netflix)
- Cross-platform tools (React Native, Flutter)
- Frequent platform switchers (users have both iOS and Android)

**Example:** Instagram, Spotify (nearly identical on iOS and Android)

### Option 3: Hybrid Approach (Recommended)

**Approach:** Unified brand/design language, but adapt navigation and key interactions to platform conventions.

**Platform-Specific:**
- Navigation (iOS tab bar, Android bottom nav or drawer)
- System gestures (iOS swipe back, Android back button)
- Typography (SF Pro on iOS, Roboto on Android)
- Modals/sheets (iOS action sheet, Android bottom sheet)

**Unified:**
- Brand colors, logos, iconography
- Content layout, information hierarchy
- Core feature design (posts, feeds, detail screens)
- Illustration style, photography treatment

**Pros:**
- Feels native (navigation, gestures)
- Consistent brand (colors, layout, features)
- Balanced effort (shared design, minor platform tweaks)

**Cons:**
- Requires design flexibility (can't pixel-perfect match screenshots)
- Need platform-aware components (tabs vs bottom nav)

**When to Use:**
- Most apps (best balance of native feel and consistency)
- Cross-platform frameworks with platform components (React Native, Flutter)
- Large user base on both iOS and Android

**Example:** Twitter, Gmail, YouTube (unified brand, platform-adapted navigation)

**Implementation Tips:**

```
Design System Structure:
1. Core brand (colors, typography scale, spacing)
2. Components (buttons, cards, forms - unified)
3. Navigation (platform-specific templates)
4. Patterns (platform-specific gestures, modals)

Result: 80% shared, 20% platform-specific
```

---

## Anti-Patterns

### Priority 0 (Critical - Never Do)

**1. Touch targets smaller than 44pt/48dp:**
- **Problem:** Users can't tap accurately, frustration, accessibility failure
- **Impact:** Critical usability issue, WCAG fail
- **Fix:** Increase touch target to minimum 44x44pt (iOS) or 48x48dp (Android)

**2. Critical actions in top corners:**
- **Problem:** Hardest reach, especially one-handed
- **Impact:** Users must reposition hand or use two hands
- **Fix:** Move primary actions to bottom 50% of screen (thumb zone)

**3. No offline error handling:**
- **Problem:** App breaks when offline, no feedback
- **Impact:** Users think app is broken, not just offline
- **Fix:** Cache content, queue actions, show offline indicator

**4. Ignoring safe areas (notches, home indicators):**
- **Problem:** Content hidden behind notch or home indicator
- **Impact:** Critical content unreadable, unprofessional
- **Fix:** Use system safe area APIs (safeAreaInsets, WindowInsets)

**5. Overriding system back gesture (iOS) or back button (Android):**
- **Problem:** Breaks universal navigation pattern, confuses users
- **Impact:** Users trapped in screens, can't navigate out
- **Fix:** Always support system back navigation

### Priority 1 (High - Avoid)

**6. Blank screens during loading (no skeleton):**
- **Problem:** Looks broken, feels slow
- **Impact:** Perceived performance is poor, higher bounce rate
- **Fix:** Show skeleton screen or cached content while loading

**7. Platform-inappropriate gestures:**
- **Problem:** Android swipe-back when platform uses back button, or vice versa
- **Impact:** Inconsistent with user expectations, confusion
- **Fix:** Follow platform gesture conventions (iOS swipe back, Android back button)

**8. Desktop-first responsive design (shrunk desktop UI):**
- **Problem:** Small text, tiny touch targets, cramped layout
- **Impact:** Unusable on mobile, requires zooming
- **Fix:** Design mobile-first, touch targets 44-48pt minimum, readable text 16sp+

**9. No loading feedback (>1 second operations):**
- **Problem:** Users don't know if action registered, may tap again
- **Impact:** Double-submits, confusion, frustration
- **Fix:** Show spinner or progress indicator for operations >1 second

**10. Modal dialogs for everything:**
- **Problem:** Overuse of modals interrupts flow, feels heavy
- **Impact:** Annoying, slows users down
- **Fix:** Use inline editing, bottom sheets (Android), or action sheets (iOS) for lighter interactions

### Priority 2 (Medium - Be Cautious)

**11. Custom UI that breaks platform expectations:**
- **Problem:** Non-standard navigation, unusual gestures
- **Impact:** Learning curve, feels foreign
- **Fix:** Use platform conventions unless you have strong brand reason to deviate

**12. Animations longer than 300ms:**
- **Problem:** Slows users down, feels sluggish
- **Impact:** Perceived performance suffers
- **Fix:** Keep animations 150-300ms, use interruption (user can tap to skip)

**13. Hidden navigation patterns (hamburger menu overuse):**
- **Problem:** Navigation hidden behind hamburger, low discoverability
- **Impact:** Users don't explore features, lower engagement
- **Fix:** Use bottom navigation (3-5 items) for primary navigation, drawer for secondary

**14. Too many onboarding screens (>3):**
- **Problem:** Users skip, don't read, or abandon
- **Impact:** Low completion rate, doesn't teach effectively
- **Fix:** Progressive onboarding (teach in context when feature is needed)

**15. No pull-to-refresh (for feed-based apps):**
- **Problem:** Users expect pull-to-refresh, can't manually update
- **Impact:** Frustration, feels outdated
- **Fix:** Implement pull-to-refresh for feeds, lists, content screens

**16. Inconsistent touch target sizes:**
- **Problem:** Some buttons 44pt, others 32pt, creates uneven feel
- **Impact:** Harder to develop muscle memory, feels sloppy
- **Fix:** Standardize to 48pt (primary), 44pt (secondary), never below 44pt

---

## Practical Application

### Workflow 1: New iOS App Design

**Step 1: Define Information Architecture**
- Map top-level sections (3-5 for tab bar)
- Plan navigation hierarchy (push/pop stack)
- Identify modal flows (login, creation, settings)

**Step 2: Design for Smallest iOS Device (iPhone SE 375x667pt)**
- Single column layout
- Bottom tab bar (49pt + safe area)
- Top navigation bar (44pt or 96pt large title)
- Safe area insets (status bar, home indicator)

**Step 3: Apply iOS Visual Patterns**
- Typography: SF Pro, large titles (34pt), body (17pt)
- Components: Tab bar, navigation bar, segmented controls
- Colors: System colors for light/dark mode support
- Corners: 13pt radius for cards, 8-10pt for buttons

**Step 4: Specify Touch Targets and Spacing**
- Minimum: 44x44pt for all interactive elements
- Spacing: 8pt minimum between targets
- Primary buttons: 48pt height
- Form inputs: 44pt height

**Step 5: Define Gestures**
- Swipe back: System gesture (enabled by default)
- Pull to refresh: UIRefreshControl for lists
- Swipe actions: Trailing (delete) and leading (archive) on list items
- Long press: Context menu for previews

**Step 6: Design Loading and Error States**
- Skeleton screens: Show content structure while loading
- Pull-to-refresh: Manual refresh option
- Error states: Clear message + retry button
- Empty states: Illustration + message + CTA

**Step 7: Test on Multiple iOS Sizes**
- iPhone SE (375pt): Smallest, most constrained
- iPhone 13 (390pt): Most common
- iPhone 14 Pro Max (430pt): Largest, test scaling
- iPad (768pt+): Adapt to two-column if supported

**Step 8: Accessibility Check**
- VoiceOver labels for all interactive elements
- Dynamic Type support (text scales with user setting)
- High contrast colors (check in accessibility settings)
- Reduce motion support (disable animations)

### Workflow 2: New Android App Design

**Step 1: Define Information Architecture**
- Map top-level destinations (3-5 for bottom nav, 6+ for drawer)
- Plan navigation hierarchy (fragments, back stack)
- Identify modals and bottom sheets

**Step 2: Design for Smallest Android Device (360x640dp)**
- Single column layout
- Bottom navigation (56dp) or navigation drawer
- Top app bar (56dp)
- System navigation bar spacing (gesture or button nav)

**Step 3: Apply Material Design Patterns**
- Typography: Roboto, H6 (20sp), Body1 (16sp)
- Components: Bottom nav, FAB, top app bar, bottom sheets
- Elevation: Cards 1dp, app bar 4dp, FAB 6dp, bottom nav 8dp
- Colors: Primary (brand), secondary (accent), surface, error
- Corners: 4dp for cards/buttons, 8dp for bottom sheets

**Step 4: Specify Touch Targets and Spacing**
- Minimum: 48x48dp for all interactive elements
- Spacing: 8dp minimum between targets
- Primary buttons: 56dp height
- FAB: 56dp diameter

**Step 5: Define Gestures**
- System back: Support back button and gesture (edge swipe)
- Pull to refresh: SwipeRefreshLayout for lists
- Swipe to dismiss: Notifications, cards (optional)
- Bottom sheet swipe: Expand/collapse gestures
- Navigation drawer: Left edge swipe to open (if using drawer)

**Step 6: Design Loading and Error States**
- Skeleton screens: Show content structure while loading
- Pull-to-refresh: Manual refresh option
- Snackbars: Brief messages (4s), optional action
- Error states: Clear message + retry button

**Step 7: Test on Multiple Android Sizes**
- Small phone (360dp): Smallest, most constrained
- Medium phone (411dp): Common (Pixel, Samsung)
- Large phone (480dp): Test scaling
- Tablet (600dp+): Adapt to two-column layout

**Step 8: Accessibility Check**
- TalkBack labels for all interactive elements
- Large text support (test at 200% font size)
- High contrast colors (check in accessibility settings)
- Touch target size (48dp minimum, test with accessibility scanner)

### Workflow 3: Cross-Platform App (React Native/Flutter)

**Step 1: Define Unified Information Architecture**
- Core features same on both platforms
- Navigation adapts to platform (iOS tab bar, Android bottom nav)
- Content screens identical (articles, profiles, detail pages)

**Step 2: Create Unified Design System**
- Brand colors (apply to both platforms)
- Typography scale (adapt font family per platform: SF Pro/Roboto)
- Component library (buttons, cards, forms - unified styling)
- Iconography (custom icon set or cross-platform library)

**Step 3: Design Platform-Specific Navigation**
- **iOS:** Tab bar (bottom), navigation bar (top), large titles
- **Android:** Bottom navigation or drawer, top app bar, FAB

**Step 4: Specify Touch Targets (Use Larger Minimum)**
- Minimum: 48x48pt/dp (largest of the two platforms)
- Works for both iOS (44pt min) and Android (48dp min)
- Reduces platform-specific specs

**Step 5: Define Gestures (Support Both Platform Conventions)**
- iOS: Swipe back (from left edge)
- Android: Back button and gesture
- Both: Pull to refresh, swipe actions (if used), pinch to zoom
- Use platform-detection for gesture implementation

**Step 6: Implement Platform-Aware Components**

```
Example: Navigation Component (React Native)
import { Platform } from 'react-native';

const Navigation = () => {
  if (Platform.OS === 'ios') {
    return <TabBar items={topLevelScreens} />;
  } else {
    return <BottomNavigation items={topLevelScreens} />;
  }
};
```

**Step 7: Test on Both Platforms**
- iOS: iPhone SE, iPhone 13, iPad (if supported)
- Android: Small phone (360dp), medium phone (411dp), tablet (if supported)
- Verify navigation feels native to each platform

**Step 8: Accessibility on Both Platforms**
- iOS: VoiceOver, Dynamic Type
- Android: TalkBack, large text
- Cross-platform: Ensure accessibility props work on both

### Workflow 4: Mobile-Responsive Web App

**Step 1: Mobile-First Design (Start at 375px Width)**
- Single column layout
- Touch-friendly targets (48x48px minimum)
- Readable text (16px+ body text)
- Thumb-reachable navigation (bottom or sticky header)

**Step 2: Navigation Pattern (Web Mobile)**
- **Bottom navigation:** 3-5 items, fixed position (mimics native app)
- **Hamburger menu:** Overlay menu (left or right slide-in)
- **Sticky header:** Fixed top navigation (can collapse on scroll)

**Step 3: Touch Target Sizing (48x48px Minimum)**
- Web mobile uses physical pixels (px)
- Minimum: 48x48px (slightly larger than iOS 44pt at 2x = 88px physical)
- Spacing: 8px minimum between targets

**Step 4: Typography (Web Mobile)**
- Body text: 16px minimum (prevents iOS auto-zoom on focus)
- Line height: 1.5 (24px for 16px text)
- Headings: 24px+, line height 1.2-1.3
- Max line length: 70ch (~600px at 16px)

**Step 5: Gestures (Web Mobile)**
- **Tap:** Standard (click events work)
- **Scroll:** Vertical swipe (native browser)
- **Pinch zoom:** Disabled by default on web (viewport meta tag), enable for images/maps
- **Pull to refresh:** Not standard on web, can implement with JS library
- **No swipe back:** Web doesn't have system swipe back (use back button in nav)

**Step 6: Responsive Breakpoints**
```
Mobile: < 600px (single column, bottom nav)
Tablet: 600-1024px (two column, side nav optional)
Desktop: > 1024px (multi-column, side nav)
```

**Step 7: Progressive Web App Features (Optional)**
- Add to home screen (PWA manifest)
- Service worker (offline support)
- Push notifications (web push)
- Feels like native app on mobile

**Step 8: Test on Real Devices**
- iOS Safari: Quirks with viewport, safe areas
- Android Chrome: Default browser
- Test touch targets, scrolling, gestures

---

## Related Skills

**Core Lyra UX Skills:**
- **`lyra/ux-designer/visual-design-foundations`**: Visual hierarchy, contrast, typography, color (applies to mobile, but mobile has specific constraints like high outdoor contrast)
- **`lyra/ux-designer/interaction-design-patterns`**: Touch targets (44x44pt iOS, 48x48dp Android), button states, feedback, animations (mobile is touch-first subset of general interactions)
- **`lyra/ux-designer/information-architecture`**: Navigation structure, hierarchy (mobile requires simpler, flatter IA due to smaller screen)
- **`lyra/ux-designer/accessibility-and-inclusive-design`**: Touch target sizing (WCAG 44px min), VoiceOver/TalkBack, high contrast, large text support (critical for mobile)
- **`lyra/ux-designer/user-research-and-validation`**: Usability testing on mobile devices, thumb zone validation, one-handed use testing

**Other Platform Skills:**
- **`lyra/ux-designer/web-application-design`**: Responsive web differs from native mobile (no native gestures, different navigation patterns)
- **`lyra/ux-designer/desktop-software-design`**: Desktop has opposite constraints (mouse precision, large screen, keyboard-first)
- **`lyra/ux-designer/game-ui-design`**: Games on mobile have unique constraints (performance, gamepad or touch controls)

**Meta-Skill:**
- **`lyra/ux-designer/using-ux-designer`**: Routes to appropriate UX skill based on user's question (mentions "mobile", "iOS", "Android" → routes here)

**Cross-Faction:**
- **`muna/technical-writer/clarity-and-style`**: Clear UI copy, error messages, microcopy (especially important on mobile where space is limited)
- **`ordis/security-architect/secure-authentication-patterns`**: Biometric authentication (Face ID, Touch ID, Android Biometric), secure mobile auth flows

---

## Additional Resources

**iOS Human Interface Guidelines:**
- **Official documentation:** https://developer.apple.com/design/human-interface-guidelines/ios
- **Key sections:** Layout (safe areas, size classes), Visual Design (typography, color), Interaction (gestures, touch)
- **Updated regularly:** Check for latest iOS version guidelines

**Android Material Design:**
- **Official documentation:** https://m3.material.io/ (Material 3, latest)
- **Key sections:** Foundations (layout, typography, color), Components (bottom nav, FAB, app bar), Motion
- **Material 3 vs Material 2:** Material 3 is newest (dynamic color, updated components), but Material 2 still widely used

**Touch Target Standards:**
- Apple HIG: 44x44pt minimum
- Material Design: 48x48dp minimum
- WCAG 2.1 Level AAA (SC 2.5.5): 44x44px minimum
- Microsoft: 34x34px minimum (less strict)

**Platform-Specific Components:**
- **iOS:** UITabBar, UINavigationBar, UISegmentedControl, UIActionSheet
- **Android:** BottomNavigationView, FloatingActionButton, AppBarLayout, BottomSheetDialog
- **Cross-platform:** React Native, Flutter have platform-aware components

**Testing Tools:**
- **iOS:** Xcode Simulator (size classes, safe areas, dark mode)
- **Android:** Android Studio Emulator (screen sizes, system navigation modes)
- **Accessibility:** iOS Accessibility Inspector, Android Accessibility Scanner
- **Performance:** Xcode Instruments, Android Profiler

**Design Tools:**
- **Figma:** iOS and Android design kits (official and community)
- **Sketch:** iOS and Android UI kits
- **Adobe XD:** Mobile design templates
- **Framer:** Interactive prototypes with gestures

**Recommended Reading:**
- "Mobile Design Pattern Gallery" by Theresa Neil
- "Designing Mobile Interfaces" by Steven Hoober & Eric Berkman
- iOS HIG (updated with each iOS release)
- Material Design Guidelines (updated regularly)
- "Don't Make Me Think" by Steve Krug (general usability, applies to mobile)
