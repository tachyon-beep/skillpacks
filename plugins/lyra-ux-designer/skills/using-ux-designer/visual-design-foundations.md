
# Visual Design Foundations

## Overview

This skill provides **The Visual Hierarchy Analysis Framework**: a systematic 6-dimension evaluation model for visual design decisions. Use this when working with color, typography, spacing, contrast, or overall visual hierarchy.

**Core Principle**: Visual hierarchy guides user attention through intentional design choices. Every visual element competes for attention—systematic evaluation ensures priority matches importance.

**Dual-Mode Support**:
- **Critique Mode**: "Your dashboard has contrast issues—the primary CTA blends with secondary actions."
- **Specification Mode**: "Primary button: #0066CC background, #FFFFFF text (7:1 contrast), 16px padding, 48px min height."

## When to Use

Load this skill when:
- User mentions: "color", "typography", "hierarchy", "spacing", "layout", "contrast", "visual design"
- Reviewing visual aspects of a design (mockups, screenshots, prototypes)
- Creating new visual specifications for interfaces
- Troubleshooting readability or attention-flow issues
- Establishing visual design systems

**Don't use for**: Navigation structure (use information-architecture), interaction patterns (use interaction-design-patterns), pure content strategy


## The Visual Hierarchy Analysis Framework

A systematic 6-dimension evaluation model. Evaluate designs by examining each dimension sequentially, identifying friction points where dimensions conflict, then prioritizing fixes by user impact.


### Dimension 1: CONTRAST

**Purpose:** Guide attention through visual weight differences

Contrast creates hierarchy by making important elements stand out from their surroundings. High contrast draws attention; low contrast recedes into the background.

#### Evaluation Questions

1. **What's the most important element on this screen?**
   - Does it have the highest visual contrast?
   - Can users identify it within 3 seconds?

2. **Does contrast match priority?**
   - Primary actions: High contrast
   - Secondary actions: Medium contrast
   - Tertiary/destructive: Low contrast or subdued

3. **Are there competing focal points?**
   - If everything screams for attention, nothing gets it
   - Only 1-2 high-contrast elements per screen

4. **Is text readable against backgrounds?**
   - Light text on dark: sufficient contrast?
   - Dark text on light: sufficient contrast?
   - Meets WCAG contrast ratios? (see accessibility-and-inclusive-design)

#### Patterns (Good Examples)

**Primary Action Emphasis**:
- Submit button: Dark blue (#0066CC) on white background = 7:1 contrast
- Secondary "Cancel": Gray (#666666) on white = 3:1 contrast
- Result: Eye goes to Submit first

**Card Hierarchy**:
- Card background: White
- Body text: Dark gray (#333333) = 12:1 contrast (high readability)
- Meta text (timestamps): Light gray (#999999) = 2.8:1 contrast (de-emphasized)
- Result: Important content stands out, metadata recedes

**Call-to-Action Contrast**:
- Page content: Black text on white background (standard)
- CTA button: Bright color (#FF6B35) with white text = vibrant, attention-grabbing
- Result: Action is unmissable

#### Anti-Patterns (Problematic Examples)

**Everything High Contrast** ❌:
- All buttons same bright color
- All text same dark black
- No hierarchy = user doesn't know where to look first
- **Fix**: Vary contrast to match importance

**Low Contrast on Critical Actions** ❌:
- Light gray button (#CCCCCC) on white background (#FFFFFF)
- Contrast ratio: 1.4:1 (fails WCAG AA 3:1 minimum for UI components)
- Users miss the action entirely
- **Fix**: Increase contrast to at least 3:1, preferably 4.5:1+

**Insufficient Text Contrast** ❌:
- Gray text (#999999) on white (#FFFFFF) = 2.8:1
- Fails WCAG AA (requires 4.5:1 for normal text)
- Strains eyes, inaccessible to low-vision users
- **Fix**: Darken to #767676 or darker for 4.5:1+ contrast

**False Affordances via Contrast** ❌:
- Static text styled with high contrast and color
- Looks clickable but isn't interactive
- **Fix**: Reserve high-contrast color for interactive elements

#### Practical Application

**Critique Mode Example**:
> "Your login form has contrast hierarchy issues (Dimension 1). The 'Forgot Password' link (#0066CC) has higher visual weight than the 'Sign In' button (#EEEEEE on white). Users' eyes are drawn to the secondary action first. **Recommendation**: Increase Sign In button contrast (dark blue background, white text = 7:1 ratio) and reduce Forgot Password to standard link styling."

**Specification Mode Example**:
> **Primary Button (Sign In)**:
> - Background: #0066CC (dark blue)
> - Text: #FFFFFF (white)
> - Contrast ratio: 7:1 (WCAG AAA)
> - Visual weight: High (draws attention)
>
> **Secondary Link (Forgot Password)**:
> - Text: #0066CC (blue)
> - Background: Transparent
> - Contrast ratio: 4.8:1 against white
> - Visual weight: Medium (visible but not primary)

#### Cross-Platform Considerations

- **Mobile**: Higher contrast needed (viewed in sunlight, smaller screens)
- **Web**: Standard contrast works, but test with browser zoom (200%)
- **Desktop**: Larger screens allow subtler contrast variations
- **Game**: High contrast essential during action (HUD must be readable)


### Dimension 2: SCALE

**Purpose:** Size communicates importance and functionality

Larger elements signal importance. Smaller elements recede. Scale also affects usability—too small = hard to interact with.

#### Evaluation Questions

1. **Does size match information hierarchy?**
   - Headings > subheadings > body text?
   - Primary actions > secondary actions?

2. **Are touch targets large enough?**
   - Mobile: 44x44pt minimum (iOS), 48x48dp (Android)
   - Web: 40x40px minimum for mouse, 48x48px for touch
   - Desktop: 24x24px minimum (mouse precision)

3. **Is text readable at intended viewing distance?**
   - Mobile: 16px+ body text (arm's length, small screen)
   - Desktop: 14px+ body text (farther viewing distance, larger screen)
   - Presentations: 24px+ body text (across room)

4. **Is there clear visual distinction between levels?**
   - At least 2px difference in font sizes
   - At least 20% difference in element sizes

#### Patterns (Good Examples)

**Typographic Scale**:
- H1: 32px (page title)
- H2: 24px (section headers)
- H3: 20px (subsection headers)
- Body: 16px (paragraphs)
- Caption: 14px (meta information)
- **Result**: Clear hierarchy, each level distinct

**Button Sizing**:
- Primary CTA: 48px height (prominent, easy to tap)
- Secondary actions: 40px height (smaller but still comfortable)
- Tertiary links: Standard text size with padding
- **Result**: Importance reflected in size

**Touch Target Sizing (Mobile)**:
- Interactive icons: 44x44pt minimum
- Buttons: 48pt height minimum
- Form inputs: 44pt height minimum
- Spacing between targets: 8pt minimum
- **Result**: No accidental taps, comfortable for thumbs

**Icon Sizing**:
- Interactive icons: 24px minimum (recognizable, tappable)
- Decorative icons: 16-20px (visual accent)
- Large feature icons: 48-64px (hero sections)
- **Result**: Function matches size

#### Anti-Patterns (Problematic Examples)

**Tiny Touch Targets** ❌:
- 30x30px buttons on mobile
- Users miss taps, frustration
- **Fix**: Increase to 44x44pt minimum, add spacing

**All Elements Same Size** ❌:
- H1 = H2 = H3 = 18px
- No visual hierarchy, can't scan page
- **Fix**: Use modular scale (1.2x or 1.5x multiplier per level)

**Unreadable Small Text** ❌:
- 12px body text on mobile
- Requires zooming, poor accessibility
- **Fix**: 16px minimum for body text on mobile

**Insufficient Size Distinction** ❌:
- H1: 18px, Body: 16px (only 2px difference)
- Hierarchy unclear, looks like emphasis not heading
- **Fix**: H1 at least 1.5x body size (24px+)

**Button Size vs Importance Mismatch** ❌:
- Large "Cancel" button, small "Submit" button
- Visual weight suggests Cancel is primary action
- **Fix**: Primary action should be largest

#### Practical Application

**Critique Mode Example**:
> "Your mobile form has touch target issues (Dimension 2). The checkbox inputs are 28x28px—below iOS minimum of 44x44pt. Users will struggle to tap accurately. **Recommendation**: Increase touch target to 48x48px (includes padding around 24px checkbox icon). Also, the primary 'Submit' button at 36px height is smaller than the 'Cancel' button at 44px—this inverts the importance hierarchy."

**Specification Mode Example**:
> **Typographic Scale**:
> - H1 (Page Title): 32px / 2rem / 700 weight
> - H2 (Section): 24px / 1.5rem / 600 weight
> - H3 (Subsection): 20px / 1.25rem / 600 weight
> - Body: 16px / 1rem / 400 weight
> - Caption: 14px / 0.875rem / 400 weight
>
> **Button Sizing (Mobile)**:
> - Primary: 48px height, 16px horizontal padding, full-width on mobile
> - Secondary: 44px height, 16px horizontal padding
> - Min touch target: 44x44pt with 8pt spacing

#### Cross-Platform Considerations

- **Mobile**: Larger touch targets (44-48pt), larger body text (16px+)
- **Web**: Flexible sizing, responsive scaling (use rem units)
- **Desktop**: Can use smaller targets (mouse precision), but don't go below 24px
- **Game**: Depends on input method (gamepad = larger, mouse = smaller)


### Dimension 3: SPACING

**Purpose:** White space creates relationships and breathing room

Spacing (margins, padding, line height) groups related elements and separates distinct sections. White space isn't wasted space—it's essential for readability and comprehension.

#### Evaluation Questions

1. **Are related items grouped with less space between them?**
   - Gestalt principle of proximity
   - Tight spacing = "these belong together"

2. **Is there adequate breathing room around content?**
   - Cramped layouts feel overwhelming
   - Generous padding improves focus

3. **Is spacing consistent?**
   - 8px or 4px base unit recommended
   - Avoid random values (13px here, 18px there)

4. **Is line height comfortable for reading?**
   - Body text: 1.4-1.6x font size
   - Headings: 1.2-1.3x font size
   - Tight line height (<1.3) = cramped

5. **Is line length appropriate?**
   - 45-75 characters per line (optimal readability)
   - Wider = eyes lose place, narrower = choppy

#### Patterns (Good Examples)

**8px Grid System**:
- All spacing in multiples of 8px (8, 16, 24, 32, 40, 48...)
- Or 4px for tighter control (4, 8, 12, 16, 20, 24...)
- **Result**: Consistent rhythm, easier to maintain

**Grouping Related Content**:
- Form label + input: 4px gap
- Input + helper text: 4px gap
- Form field groups: 16px gap
- Form sections: 32px gap
- **Result**: Clear relationships, scannable structure

**Padding for Breathing Room**:
- Card padding: 24px all sides
- Button padding: 12px vertical, 24px horizontal
- Container padding: 16-24px (mobile), 32-48px (desktop)
- **Result**: Content doesn't feel cramped

**Line Height for Readability**:
- Body text (16px): line-height 1.5 = 24px (comfortable reading)
- Headings (32px): line-height 1.2 = 38px (tighter, more impact)
- Code blocks: line-height 1.6 = 25.6px (extra space for legibility)
- **Result**: Easy to read, eyes don't jump lines

**Line Length**:
- Desktop: max-width 70ch (70 characters, ~600px for 16px text)
- Mobile: Full width okay (narrow screen limits length naturally)
- **Result**: Comfortable reading pace

#### Anti-Patterns (Problematic Examples)

**Cramming Content** ❌:
- No padding inside cards
- Text touches edges
- Feels suffocating, hard to focus
- **Fix**: Add 16-24px padding

**Inconsistent Spacing** ❌:
- Some sections: 13px gap
- Other sections: 18px gap
- Others: 22px gap
- No pattern = feels chaotic
- **Fix**: Use 8px or 4px grid, stick to multiples

**Insufficient Line Height** ❌:
- Body text with line-height: 1.0
- Lines touch, hard to read, feels cramped
- **Fix**: 1.4-1.6 for body text

**Too Much Space** ❌:
- Excessive padding (64px on mobile cards)
- Wastes screen real estate
- Requires excessive scrolling
- **Fix**: 16-24px padding on mobile, 24-32px on desktop

**Wrong Grouping** ❌:
- Form label far from input (24px gap)
- Input close to next label (8px gap)
- User can't tell which label belongs to which input
- **Fix**: Label + input tight (4-8px), input + next label loose (24px+)

**Line Length Too Wide** ❌:
- Full-width paragraphs on 1920px desktop
- ~200 characters per line
- Eyes lose place, skipping lines
- **Fix**: max-width 70ch or ~600-800px

#### Practical Application

**Critique Mode Example**:
> "Your article layout has spacing issues (Dimension 3). The body text has line-height: 1.2, which makes multi-paragraph reading uncomfortable—lines feel cramped. Also, the paragraph spacing (8px) is less than the line spacing within paragraphs, violating proximity principles. **Recommendation**: Increase line-height to 1.5 (24px for 16px text) and paragraph spacing to 24px (1.5x line height). Additionally, limit line length to 70ch (~600px) to prevent overly wide reading."

**Specification Mode Example**:
> **Spacing System (8px base unit)**:
> - XS: 8px (tight grouping, label + input)
> - S: 16px (related elements, list items)
> - M: 24px (section spacing, card padding)
> - L: 32px (major sections)
> - XL: 48px (page sections)
> - XXL: 64px (hero sections)
>
> **Typography Spacing**:
> - Body text: 16px font, 24px line-height (1.5)
> - Paragraphs: 24px bottom margin
> - Headings: 32px top margin (section break), 16px bottom margin
> - Max line length: 70ch (~600px at 16px)
>
> **Component Padding**:
> - Cards: 24px all sides (mobile), 32px (desktop)
> - Buttons: 12px vertical, 24px horizontal
> - Form inputs: 12px vertical, 16px horizontal

#### Cross-Platform Considerations

- **Mobile**: Tighter spacing (16-24px padding), less margin (maximize content)
- **Web**: Responsive spacing (increase padding on larger screens)
- **Desktop**: More generous spacing (32-48px padding), wider margins
- **Game**: Depends on genre (minimalist HUD vs rich RPG menus)


### Dimension 4: COLOR

**Purpose:** Meaning, emotion, brand identity, and accessibility

Color conveys semantic meaning (red = error, green = success), sets emotional tone, reinforces brand, and must be accessible to all users.

#### Evaluation Questions

1. **Does color have semantic meaning?**
   - Red = error, danger, stop
   - Green = success, go, safe
   - Yellow/orange = warning, caution
   - Blue = information, trust

2. **Does text meet contrast ratios?**
   - WCAG AA: 4.5:1 for normal text, 3:1 for large text (18px+ or 14px+ bold)
   - WCAG AAA: 7:1 for normal text, 4.5:1 for large text
   - Use tools: WebAIM Contrast Checker, Stark, Figma plugins

3. **Is design colorblind-safe?**
   - Don't rely on color alone (use icons, labels, patterns)
   - Test with colorblind simulators (Chromatic Vision Simulator)
   - Most common: Red-green colorblindness (8% of men)

4. **Is palette intentional and limited?**
   - 3-5 main colors (not counting neutrals)
   - Too many colors = visual chaos
   - Each color has purpose

5. **Does color support brand identity?**
   - Primary brand color used consistently
   - Secondary colors complement brand
   - Color reinforces brand recognition

#### Patterns (Good Examples)

**Semantic Color System**:
- Error: #D32F2F (red) with ⚠️ icon
- Success: #388E3C (green) with ✓ checkmark
- Warning: #F57C00 (orange) with ⚠️ icon
- Info: #1976D2 (blue) with ℹ️ icon
- **Result**: Color + icon = accessible, clear meaning

**Limited Intentional Palette**:
- Primary: #0066CC (brand blue, CTAs)
- Secondary: #6C757D (neutral gray, secondary actions)
- Success: #28A745 (green, confirmations)
- Error: #DC3545 (red, errors)
- Neutrals: #000, #333, #666, #999, #CCC, #F5F5F5, #FFF
- **Result**: Cohesive, not overwhelming

**High Contrast Text**:
- Dark text on light: #212121 on #FFFFFF = 16:1 contrast (excellent)
- Light text on dark: #FFFFFF on #1A1A1A = 15:1 contrast (excellent)
- Muted text: #666666 on #FFFFFF = 5.7:1 (passes AA, good for secondary text)
- **Result**: Readable, accessible

**Colorblind-Safe Design**:
- Status indicators: Green ✓ + "Success", Red ✗ + "Error"
- Charts: Use patterns + colors (stripes, dots, solids)
- Links: Underlined + colored (not color alone)
- **Result**: Works for 100% of users

**Brand-Consistent Color**:
- Primary brand color (#FF6B35 orange) on all CTAs
- Secondary brand color (#004E89 blue) on headers
- Neutrals for body content
- **Result**: Strong brand recognition

#### Anti-Patterns (Problematic Examples)

**Low Contrast Text** ❌:
- Light gray text (#CCCCCC) on white (#FFFFFF)
- Contrast ratio: 1.6:1 (fails WCAG, unreadable)
- **Fix**: Darken to #757575 for 4.5:1 ratio

**Color as Sole Indicator** ❌:
- Red items = errors, green items = success (no icons/labels)
- Colorblind users can't distinguish
- **Fix**: Add icons and text labels

**Rainbow Explosion** ❌:
- 10+ colors in interface
- No clear system, visually chaotic
- **Fix**: Limit to 3-5 main colors + neutrals

**Inaccessible Button Colors** ❌:
- Light blue (#6CB4EE) button with white text (#FFFFFF)
- Contrast: 2.4:1 (fails WCAG 3:1 minimum for UI components)
- **Fix**: Darken button to #0066CC for 4.5:1+ contrast

**Conflicting Color Meanings** ❌:
- Red for both "Delete" and "Primary CTA"
- Confuses semantic meaning
- **Fix**: Reserve red for destructive/error, use brand color for CTAs

**No Visual Hierarchy** ❌:
- All text same color (pure black)
- No distinction between headers, body, meta
- **Fix**: Use color variations (black headers, dark gray body, light gray meta)

#### Practical Application

**Critique Mode Example**:
> "Your form design has color accessibility issues (Dimension 4). The error messages use red text (#FF0000) without any accompanying icon—colorblind users can't distinguish errors from normal text. Also, the placeholder text (#BBBBBB) on white has only 2.5:1 contrast, failing WCAG AA. **Recommendation**: Add ⚠️ icons to error messages and change error text to #D32F2F with an icon. Darken placeholder text to #757575 for 4.5:1 contrast. Additionally, the 'Submit' button (#8BC34A light green) with white text has only 2.7:1 contrast—darken to #388E3C for 4.5:1."

**Specification Mode Example**:
> **Color System**:
> - Primary (CTA): #0066CC (brand blue, 4.8:1 with white text)
> - Secondary (actions): #6C757D (neutral gray)
> - Success: #388E3C (green, 4.5:1 with white text) + ✓ icon
> - Error: #D32F2F (red, 5.5:1 with white text) + ⚠️ icon
> - Warning: #F57C00 (orange, 4.5:1 with white text) + ⚠️ icon
> - Info: #1976D2 (blue, 4.6:1 with white text) + ℹ️ icon
>
> **Text Colors**:
> - Primary text: #212121 on #FFFFFF (16:1 contrast)
> - Secondary text: #666666 on #FFFFFF (5.7:1 contrast)
> - Disabled text: #999999 on #FFFFFF (2.8:1, intentionally low)
> - Link text: #0066CC (4.8:1, underlined)
>
> **Colorblind-Safe**:
> - All status indicators: Color + icon + text label
> - Links: Color + underline
> - Charts: Patterns (stripes, dots) + color
> - Tested with Chromatic Vision Simulator

#### Cross-Platform Considerations

- **Mobile**: High contrast critical (outdoor viewing, sunlight glare)
- **Web**: Test with browser high-contrast modes
- **Desktop**: Color-calibrated displays vary—design for worst case
- **Game**: Art style determines palette, but HUD must be high contrast


### Dimension 5: TYPOGRAPHY

**Purpose:** Readability, hierarchy, and tone through type choices

Typography affects readability (can users read it?), hierarchy (what's important?), and tone (formal vs casual, modern vs traditional).

#### Evaluation Questions

1. **Are font sizes distinguishable?**
   - At least 2px difference between levels
   - Clear visual distinction (not 16px vs 17px)

2. **Is line height comfortable?**
   - Body text: 1.4-1.6x font size
   - Headings: 1.2-1.3x font size
   - Code/monospace: 1.5-1.6x

3. **Is line length appropriate?**
   - 45-75 characters per line (optimal)
   - Use max-width or container constraints

4. **Are font weights used intentionally?**
   - Regular (400) for body text
   - Bold (600-700) for headings and emphasis
   - Light (300) sparingly (can reduce readability)

5. **Is font pairing harmonious?**
   - Limit to 1-2 font families
   - Pair serif + sans-serif, or use single family with varied weights

6. **Is text readable at intended size?**
   - Mobile: 16px+ body text (prevents auto-zoom on iOS)
   - Desktop: 14px+ body text
   - Avoid text below 12px (inaccessible)

#### Patterns (Good Examples)

**Modular Type Scale**:
- H1: 32px / 2rem / 700 weight / 1.2 line-height
- H2: 24px / 1.5rem / 600 weight / 1.3 line-height
- H3: 20px / 1.25rem / 600 weight / 1.3 line-height
- Body: 16px / 1rem / 400 weight / 1.5 line-height
- Caption: 14px / 0.875rem / 400 weight / 1.4 line-height
- **Result**: Clear hierarchy, each level distinct

**Comfortable Line Height**:
- Body text (16px): line-height 1.5 = 24px
- Headings (32px): line-height 1.2 = 38.4px
- **Result**: Easy to read, no line jumping

**Font Pairing**:
- Headings: Montserrat (sans-serif, geometric, modern)
- Body: Open Sans (sans-serif, neutral, readable)
- **Result**: Harmonious, not competing

**Or Single Family**:
- All text: Inter (variable font)
- Headings: 600-700 weight
- Body: 400 weight
- Meta: 400 weight, 14px size
- **Result**: Unified, consistent

**Appropriate Weights**:
- Headers: 600-700 (bold, attention-grabbing)
- Body: 400 (regular, readable)
- Emphasis: 600 or italic
- De-emphasis: 400 with lighter color
- **Result**: Weight reinforces hierarchy

#### Anti-Patterns (Problematic Examples)

**Tiny Text** ❌:
- 12px body text on mobile
- Forces zooming, poor accessibility
- **Fix**: 16px minimum on mobile, 14px minimum on desktop

**Insufficient Line Height** ❌:
- 16px text with line-height: 1.0 = 16px
- Lines touch, cramped, hard to read
- **Fix**: 1.4-1.6 for body text (22-26px for 16px font)

**Too Many Fonts** ❌:
- Headings: Playfair Display (serif)
- Body: Roboto (sans-serif)
- Buttons: Raleway (sans-serif)
- Captions: Lato (sans-serif)
- **Result**: Visual chaos, no cohesion
- **Fix**: 1-2 font families maximum

**Insufficient Size Distinction** ❌:
- H2: 17px, Body: 16px (only 1px difference)
- H2 looks like emphasized body text, not heading
- **Fix**: At least 2px difference, preferably 1.5x multiplier (24px for H2)

**Inappropriate Weights** ❌:
- Light weight (300) for body text
- Harder to read, especially at small sizes
- **Fix**: Regular (400) for body, reserve light for large display text

**Line Length Too Wide** ❌:
- 100+ characters per line on desktop
- Eyes lose place, re-reading lines
- **Fix**: max-width: 70ch (~600-700px)

#### Practical Application

**Critique Mode Example**:
> "Your article typography has readability issues (Dimension 5). The body text is 14px with line-height 1.2 (16.8px), which feels cramped for long-form reading. Additionally, the line length exceeds 100 characters at desktop widths—users will lose their place. The H2 headings at 18px are only 4px larger than body text, making the hierarchy weak. **Recommendation**: Increase body text to 16px with 1.5 line-height (24px), constrain line length to max-width: 70ch, and increase H2 to 24px (1.5x body size) with 600 weight."

**Specification Mode Example**:
> **Typography System**:
> - Font Family: Inter (variable font, sans-serif)
> - Base Size: 16px (1rem)
>
> **Type Scale**:
> - H1: 32px / 2rem / 700 weight / 1.2 line-height / 32px bottom margin
> - H2: 24px / 1.5rem / 600 weight / 1.3 line-height / 24px bottom margin
> - H3: 20px / 1.25rem / 600 weight / 1.3 line-height / 16px bottom margin
> - Body: 16px / 1rem / 400 weight / 1.5 line-height / 24px paragraph margin
> - Caption: 14px / 0.875rem / 400 weight / 1.4 line-height
> - Code: 14px / 0.875rem / 400 weight / 1.6 line-height / monospace font
>
> **Constraints**:
> - Max line length: 70ch (~600px at 16px)
> - Min body size: 16px mobile, 14px desktop
> - Line height: 1.5 for body, 1.2-1.3 for headings

#### Cross-Platform Considerations

- **Mobile**: 16px+ body text (prevents iOS auto-zoom), tighter line height okay (1.4)
- **Web**: Responsive type scaling, use rem units
- **Desktop**: Can use slightly smaller text (14px), but test readability
- **Game**: Depends on viewing distance (couch = larger text, desk = standard)


### Dimension 6: LAYOUT FLOW

**Purpose:** Guide visual path through content

Layout flow determines how users scan the interface. Good flow guides attention to important elements in logical order. Poor flow creates confusion and missed information.

#### Evaluation Questions

1. **What's the natural reading pattern?**
   - F-pattern (content-heavy, scanning): Left-to-right, top-to-bottom, focusing on left edge
   - Z-pattern (action-focused): Top-left → top-right → diagonal → bottom-left → bottom-right
   - Gutenberg diagram: Top-left (primary) → top-right (strong) → bottom-left (weak) → bottom-right (terminal)

2. **Is there a clear entry point?**
   - Where do users' eyes land first?
   - Is it intentional?

3. **Does eye naturally flow to important elements?**
   - Primary action in natural eye path?
   - Supporting content where users expect it?

4. **Is reading order logical?**
   - Top-to-bottom, left-to-right (in LTR languages)
   - Related items close together
   - Sequential steps in order

5. **Are there visual roadblocks?**
   - Large images blocking flow
   - Competing focal points disrupting path

#### Patterns (Good Examples)

**F-Pattern for Content Pages**:
```
[Header Navigation Bar]
[H1 Title____________]
[Body paragraph______]
[H2 Subheading_______]
[Body paragraph______]
[Image] [Caption_____]
```
- Users scan left edge (headings)
- Scan top horizontal (title, nav)
- Primary content on left, supporting on right
- **Result**: Efficient scanning, easy to find information

**Z-Pattern for Landing Pages**:
```
[Logo]           [Nav Links] [CTA Button]
         ↘
    [Hero Headline]
         ↘
[Feature]  [Feature]  [Feature]
    ↘
[Secondary CTA]
```
- Top-left (logo) → top-right (CTA)
- Diagonal to hero message
- Bottom CTA as final action
- **Result**: Guided path to conversion

**Gutenberg Diagram for Forms**:
```
[Primary Optical Area]     [Strong Fallow Area]
 Field Labels               Helper Text
 Input Fields               Validation Messages

[Weak Fallow Area]         [Terminal Area]
 Optional Fields            Submit Button
 Secondary Actions          Cancel Link
```
- Top-left: Essential fields (name, email)
- Top-right: Helper text, instructions
- Bottom-left: Optional fields
- Bottom-right: Submit (terminal action)
- **Result**: Natural progression, clear end point

**Card Grid Layout**:
```
[Card] [Card] [Card]
[Card] [Card] [Card]
```
- Equal visual weight
- Left-to-right, top-to-bottom reading
- Clear grid structure
- **Result**: Scannable, organized

**Single-Column Mobile Layout**:
```
[Image]
[Heading]
[Body text]
[CTA Button]
[Supporting Info]
```
- Top-to-bottom flow
- Most important content first
- Primary action before secondary
- **Result**: Clear priority, minimal cognitive load

#### Anti-Patterns (Problematic Examples)

**Competing Focal Points** ❌:
- Large hero image on left
- Large form on right
- User doesn't know where to look first
- **Fix**: Establish clear primary element (size, contrast, position)

**Burying Primary Action** ❌:
- Critical "Submit" button at bottom of long form
- Users might not scroll to find it
- **Fix**: Sticky footer with submit button, or repeat at top and bottom

**Ignoring Natural Reading Patterns** ❌:
- Important content in bottom-left (weak fallow area)
- Users scan top-left first, miss key information
- **Fix**: Move critical content to primary optical area (top-left)

**Visual Roadblocks** ❌:
- Full-width banner image mid-page
- Breaks reading flow, users think page ended
- **Fix**: Constrain image width, or ensure clear continuation cues below

**Illogical Order** ❌:
- Step 1, Step 3, Step 2 in visual order
- Confusing, breaks mental model
- **Fix**: Visual order = logical order (top-to-bottom = 1, 2, 3)

**No Clear Entry Point** ❌:
- Homepage with 10 equal-sized elements
- User doesn't know where to start
- **Fix**: Create clear hierarchy (one large hero, smaller supporting elements)

#### Practical Application

**Critique Mode Example**:
> "Your landing page has layout flow issues (Dimension 6). The primary CTA button ('Start Free Trial') is positioned in the bottom-left corner, which is the weak fallow area in the Gutenberg diagram—users' eyes naturally skip this region. Additionally, the large testimonial image in the center creates a visual roadblock, breaking the natural Z-pattern flow from hero to CTA. **Recommendation**: Move the primary CTA to the top-right (Z-pattern terminus) or bottom-right (terminal area). Reduce testimonial image size and position it as supporting content on the left or right rail to avoid blocking the main flow."

**Specification Mode Example**:
> **Layout Flow (Z-Pattern Landing Page)**:
> - **Top-left (Primary Optical Area)**: Logo + product name
> - **Top-right (Strong Fallow Area)**: Navigation links + primary CTA button
> - **Center (Diagonal)**: Hero headline (48px bold) + subheading (20px) + supporting image
> - **Middle**: Three feature cards in horizontal row (equal visual weight)
> - **Bottom-right (Terminal Area)**: Secondary CTA ("Learn More" button)
>
> **Visual Flow Path**:
> 1. Logo (brand recognition)
> 2. Primary CTA (top-right, high contrast)
> 3. Hero headline (large scale, center)
> 4. Feature cards (scan left-to-right)
> 5. Secondary CTA (final action)
>
> **Grid Structure**:
> - Desktop: 12-column grid, hero spans 8 columns, image 4 columns
> - Mobile: Single column, top-to-bottom (logo → headline → image → CTA)

#### Cross-Platform Considerations

- **Mobile**: Single-column layout (natural top-to-bottom flow), critical actions in thumb zone
- **Web**: F-pattern for content, Z-pattern for landing pages, responsive reflow
- **Desktop**: Multi-column layouts okay (larger screens), still honor reading patterns
- **Game**: Depends on genre—HUD elements follow expected positions (health top-left, map corner)


## Cross-Platform Visual Considerations

### Mobile-Specific

- **Contrast**: Higher contrast needed (outdoor viewing, sunlight)
- **Scale**: Larger touch targets (44x44pt iOS, 48x48dp Android), 16px+ body text
- **Spacing**: Tighter spacing okay (maximize content), 16-24px padding
- **Color**: High contrast critical, test in bright sunlight
- **Typography**: 16px+ to prevent iOS auto-zoom
- **Layout**: Single-column flow, thumb-friendly placement

**Reference**: lyra/ux-designer/mobile-design-patterns for platform-specific details

### Web Application

- **Contrast**: Standard ratios work, test with browser zoom (200%)
- **Scale**: Flexible sizing (rem units), responsive scaling
- **Spacing**: Responsive (increase padding on larger screens)
- **Color**: Test with high-contrast mode, colorblind simulators
- **Typography**: Fluid type scaling, max-width for readability
- **Layout**: Multi-column on desktop, single-column mobile

**Reference**: lyra/ux-designer/web-application-design for responsive patterns

### Desktop Software

- **Contrast**: Standard, but test with OS dark modes
- **Scale**: Smaller targets okay (mouse precision), but 24px+ minimum
- **Spacing**: More generous (32-48px padding), wider margins
- **Color**: Support light/dark themes
- **Typography**: 14px+ body text, allow user font size adjustment
- **Layout**: Multi-column, panels, customizable workspaces

**Reference**: lyra/ux-designer/desktop-software-design for desktop patterns

### Game UI

- **Contrast**: Extremely high contrast (readability during action)
- **Scale**: Depends on input (gamepad = larger, mouse = smaller)
- **Spacing**: Minimal (maximize game view), but don't cram
- **Color**: Must work with game palette, colorblind modes essential
- **Typography**: Bold, clear fonts (readable at distance, during action)
- **Layout**: Corners for HUD, center for critical alerts

**Reference**: lyra/ux-designer/game-ui-design for game-specific patterns


## Practical Application Steps

### Step 1: Identify Primary User Goal

Before evaluating visual design, understand what users are trying to accomplish on this screen.

**Questions**:
- What's the primary task? (e.g., sign in, read article, complete purchase)
- What's the most important information? (e.g., price, error message, next step)
- What action should user take? (e.g., click CTA, read content, enter data)

### Step 2: Evaluate All 6 Dimensions Systematically

Work through each dimension in order, taking notes:

1. **Contrast**: Does visual weight match importance?
2. **Scale**: Are sizes appropriate for hierarchy and usability?
3. **Spacing**: Does white space group related items and provide breathing room?
4. **Color**: Is palette accessible, meaningful, and intentional?
5. **Typography**: Is text readable and hierarchical?
6. **Layout Flow**: Does visual path guide users to important elements?

### Step 3: Identify Friction Points

Note where dimensions conflict:
- High contrast but poor color contrast (dimension 1 vs 4)
- Large scale but cramped spacing (dimension 2 vs 3)
- Good hierarchy but poor layout flow (dimension 1 vs 6)

### Step 4: Prioritize Fixes by User Impact

**Critical (P0)**: Accessibility failures, unusable interfaces
- Text contrast below 4.5:1
- Touch targets below 44px on mobile
- Color as sole indicator (colorblind fail)

**High (P1)**: Usability issues, confusing hierarchy
- Primary action not prominent
- Inconsistent spacing
- Poor typography (cramped, tiny)

**Medium (P2)**: Suboptimal but functional
- Could use more breathing room
- Minor inconsistencies
- Aesthetic improvements

### Step 5: Provide Actionable Recommendations

**Critique Mode**:
- Identify specific issue with dimension reference
- Explain user impact
- Suggest concrete fix with rationale

**Specification Mode**:
- Provide exact values (colors, sizes, spacing)
- Include contrast ratios, sizing units
- Specify responsive behavior


## Visual Design Pattern Library

### Button Design

**Primary Button**:
- Background: High-contrast brand color (#0066CC)
- Text: White (#FFFFFF, 7:1 contrast)
- Size: 48px height mobile, 44px desktop
- Padding: 12px vertical, 24px horizontal
- Border-radius: 4-8px
- Font: 16px, 600 weight

**Secondary Button**:
- Background: Transparent or light gray (#F5F5F5)
- Text: Brand color (#0066CC, 4.8:1 contrast)
- Border: 1px solid #0066CC
- Size: 44px height
- Padding: 12px vertical, 24px horizontal

**States**: Default, hover (+shadow/darken), active (depressed), disabled (gray, cursor not-allowed)

### Card Design

- Background: White (#FFFFFF)
- Border: 1px solid #E0E0E0 or subtle shadow
- Padding: 24px all sides (mobile), 32px (desktop)
- Border-radius: 8px
- Spacing: 16px between cards (mobile), 24px (desktop)

### Form Design

- Label: 14px, 600 weight, 4px above input
- Input: 16px text, 44px height, 12px padding, 4px border-radius
- Helper text: 14px, gray (#666666), 4px below input
- Error: Red (#D32F2F) with icon, 14px, 4px below input
- Spacing: 16px between fields, 32px between sections

### Typography Hierarchy

- H1: 32px / 700 weight / 1.2 line-height (page title)
- H2: 24px / 600 weight / 1.3 line-height (sections)
- H3: 20px / 600 weight / 1.3 line-height (subsections)
- Body: 16px / 400 weight / 1.5 line-height
- Caption: 14px / 400 weight / 1.4 line-height (meta)


## Red Flags & Anti-Patterns Catalog

### Critical Issues (Fix Immediately)

1. **Text contrast below 4.5:1**: Accessibility failure, many users can't read
2. **Color as sole indicator**: Colorblind users miss information
3. **Touch targets below 44px**: Frustrating on mobile, accessibility issue
4. **Body text below 14px**: Unreadable, especially mobile/older users
5. **Line height below 1.3**: Cramped, hard to read
6. **Primary action not prominent**: Users can't find key task

### High-Priority Issues

7. **Inconsistent spacing**: Feels chaotic, unprofessional
8. **Too many colors**: Visual overwhelm, no hierarchy
9. **Competing focal points**: User doesn't know where to look
10. **Poor font pairing**: Visual discord, amateurish
11. **Line length over 100 characters**: Users lose place, re-read
12. **Tiny secondary text (<12px)**: Hard to read, especially disclaimers

### Medium-Priority Issues

13. **Insufficient padding**: Feels cramped, less breathing room
14. **Weak hierarchy**: All elements similar size/weight
15. **Inappropriate button size**: Size doesn't match importance
16. **Ignoring platform conventions**: Feels foreign to users
17. **Low contrast on non-text**: UI components below 3:1
18. **Excessive white space**: Wastes screen real estate


## Related Skills

**Core UX Skills**:
- **lyra/ux-designer/accessibility-and-inclusive-design**: Color contrast, typography readability, visual accessibility requirements (WCAG 2.1 AA)
- **lyra/ux-designer/interaction-design-patterns**: Visual feedback for interactions, button states, visual affordances
- **lyra/ux-designer/ux-fundamentals**: Core principles (progressive disclosure, aesthetic & minimalist design, visual hierarchy concepts)

**Platform Extensions**:
- **lyra/ux-designer/mobile-design-patterns**: Platform-specific visual conventions (iOS HIG vs Material Design), mobile constraints
- **lyra/ux-designer/web-application-design**: Responsive design patterns, complex data visualization
- **lyra/ux-designer/desktop-software-design**: Desktop visual conventions, themes (light/dark mode)
- **lyra/ux-designer/game-ui-design**: Visual coherence with game art style, HUD design for readability

**Cross-Faction**:
- **muna/technical-writer/clarity-and-style**: Writing clear UI copy and microcopy that supports visual hierarchy
