
# Information Architecture

## Overview

This skill provides a systematic approach to organizing content and designing navigation systems. Use when users can't find features, navigation feels confusing, or content needs structural organization.

**Core Principle**: Good IA matches users' mental models and provides multiple paths to content with clear information scent.

## When to Use

Load this skill when:
- Users complain "I can't find [feature]"
- Navigation structure needs design or reorganization
- Content organization is unclear or inconsistent
- Designing site maps, menus, or navigation systems
- User requests mention: "navigation", "menu", "structure", "organize", "findability"

**Don't use for**: Visual styling (use visual-design-foundations), interaction mechanics (use interaction-design-patterns)


## The Navigation & Discoverability Model

A systematic 4-layer framework for evaluating and designing information architecture:

**Layer 1: MENTAL MODELS** - Does structure match how users think?
**Layer 2: NAVIGATION SYSTEMS** - Are there multiple paths to content?
**Layer 3: INFORMATION SCENT** - Do labels lead users to their goal?
**Layer 4: DISCOVERABILITY** - Can users find features beyond the basics?

Evaluate each layer systematically to identify IA gaps.


## Layer 1: Mental Models

**Purpose:** Align information structure with users' existing mental models

### Evaluation Questions

- Does the IA match how users think about this domain?
- Are we using terminology users understand?
- Would users expect to find X under category Y?
- Does the structure match users' tasks and goals?
- Are we organizing by internal company structure (BAD) or user needs (GOOD)?

### Research Methods

**Card Sorting (Open)**
- Give users cards with content/features
- Ask them to group cards into categories they create
- Reveals natural mental model organization
- Use for: New IA, understanding user categories

**Card Sorting (Closed)**
- Give users cards and predefined categories
- Ask them to place cards in categories
- Validates existing IA structure
- Use for: Testing proposed navigation

**Tree Testing**
- Give users IA tree (text-only, no visual design)
- Ask them to find specific items
- Measures findability without visual interference
- Use for: Validating IA before visual design

**Mental Model Diagrams**
- Map user's understanding of domain (Indi Young method)
- Compare to organization's mental model
- Identify gaps between user expectations and implementation
- Use for: Deep understanding, strategy

### Patterns

**Task-Based Organization**
- Structure around what users do, not what you sell
- Example: "Send Money" not "Products > Wire Transfer"
- Users think in goals, not features

**Domain-Appropriate Terminology**
- Use industry terms users know (not internal jargon)
- Example: "Shopping Cart" not "Basket" (e-commerce convention)
- Test terminology with card sorting

**Flat-ish Hierarchy**
- Aim for 3-5 levels deep maximum
- Broad (5-9 top-level categories) > Deep (3 categories, 10 levels)
- Users struggle with navigation deeper than 3-4 clicks

**Mutually Exclusive Categories**
- Each item belongs in one obvious place
- Avoid: "Accessories" and "Cables" (cables are accessories)
- Reduces confusion, speeds navigation

### Anti-Patterns

**Organized by Internal Structure**
- "About Us > Divisions > Division 3 > Services"
- Users don't care about org chart
- Organize by user needs, not company hierarchy

**Jargon Users Don't Understand**
- Internal product names ("Project Phoenix")
- Acronyms without context ("SDLC Dashboard")
- Test labels with users unfamiliar with domain

**Categories That Overlap**
- Item could reasonably go in 2+ categories
- Users waste time checking multiple places
- Forces arbitrary decisions

**Too Many Top-Level Categories**
- 15+ menu items = overwhelming
- Users can't hold that many options in memory
- Chunk into 5-9 logical groups

### Practical Application

**Step 1:** Conduct card sorting with 15-20 users
- Open card sort to discover categories
- Closed card sort to validate proposed structure

**Step 2:** Analyze sorting patterns
- What categories did users create?
- What labels did they use?
- Where was agreement? Disagreement?

**Step 3:** Map to mental model
- Create IA structure matching user categories
- Use user terminology for labels
- Keep hierarchy flat (3-4 levels max)

**Step 4:** Validate with tree testing
- Test findability before implementing
- Iterate based on task success rates


## Layer 2: Navigation Systems

**Purpose:** Provide multiple paths to content so users can navigate their preferred way

### Evaluation Questions

- Can users reach content via global navigation?
- Is search available and functional?
- Are contextual links provided within content?
- Can users get back to where they were (breadcrumbs)?
- Are there shortcuts for common tasks?
- Is navigation consistent across screens?

### Navigation Types

**Global Navigation**
- Persistent across entire site/app (top bar, sidebar)
- Contains primary categories (5-7 items)
- Always visible or easily accessible
- Example: Main menu, app navigation drawer

**Local Navigation**
- Specific to current section (sub-menu)
- Shows related pages within category
- Example: Sidebar in documentation, tabs within settings

**Contextual Navigation**
- Links embedded in content
- "See also", "Related articles", inline links
- Helps users discover related content

**Search**
- Direct path to specific content
- Essential when >50 pages/items
- Requires: autocomplete, filters, clear results

**Utility Navigation**
- Secondary functions (login, settings, help)
- Usually top-right corner or under menu
- Less prominent than global nav

**Breadcrumbs**
- Show path from home to current page
- Allow users to backtrack easily
- Example: Home > Products > Electronics > Laptops

### Patterns

**3-Click Rule (Modified)**
- Any content reachable within 3 clicks
- More accurately: 3 CLEAR clicks (with good information scent)
- Better to have 5 obvious clicks than 2 confusing ones

**Consistent Navigation Placement**
- Same location across all pages
- Same labels, same order
- Users learn once, applies everywhere

**Primary + Secondary Actions**
- One primary path (main menu, prominent)
- Secondary paths (search, contextual links)
- Redundancy helps different user styles

**Mobile Navigation Patterns**
- Bottom tab bar (iOS standard, 3-5 items)
- Hamburger menu (for 6+ items)
- Bottom sheet menus (Android)
- Prioritize: Most important actions at bottom (thumb zone)

**Web Navigation Patterns**
- Horizontal top nav (desktop)
- Sticky header (nav always accessible)
- Mega-menus (for complex sites, show 2+ levels)
- Sidebar nav (for deep hierarchies)

**Desktop Navigation Patterns**
- Menu bar (File, Edit, View, Help)
- Toolbar (icon buttons for common actions)
- Keyboard shortcuts (Cmd+N, Cmd+O, Cmd+S)

### Anti-Patterns

**Single Path to Content**
- Only one way to reach feature (menu > submenu > page)
- Users with different mental models get lost
- Provide: Global nav + search + contextual links

**Mega-Menu Overload**
- Showing 50+ links in dropdown
- Overwhelming, hard to scan
- Chunk into logical groups, use hierarchy

**Mystery Meat Navigation**
- Icons without labels
- Users guess what icon means
- Use: Icon + label (or tooltip on hover)

**Inconsistent Navigation**
- Menu moves or changes on different pages
- Different labels for same destination
- Forces users to relearn on each page

**Buried Search**
- Search hidden in menu or obscure icon
- Essential for large sites (>50 pages)
- Make prominent: Top-right is convention

**No Way Back**
- Dead-end pages with no navigation
- Modal with no close/back button
- Always provide: Back, breadcrumbs, or close

### Practical Application

**Step 1:** Audit all navigation paths
- List all ways to reach each major page/feature
- Identify single-path content (problem)

**Step 2:** Add redundant paths
- Global nav for primary content
- Search for direct access
- Contextual links within content
- Breadcrumbs for backtracking

**Step 3:** Test with first-click tests
- Show users page/screen
- Ask "Where would you click to [task]?"
- Measure if first click is correct path

**Step 4:** Verify consistency
- Same navigation on all pages
- Same labels across contexts
- Test on 10+ different pages


## Layer 3: Information Scent

**Purpose:** Labels and clues accurately predict what users will find

### Evaluation Questions

- Do link labels accurately describe destination?
- Are category names specific (not vague)?
- Would users know where to look for their task?
- Do labels use user terminology?
- Are categories mutually exclusive?
- Does "scent" get stronger as user gets closer (progressive refinement)?

### Core Concept: Scent Strength

**Strong Scent**
- Label clearly predicts content
- Example: "Download Receipt" → PDF receipt downloads
- Users confident they're on right path

**Weak Scent**
- Label vague, ambiguous
- Example: "Resources" → Could be anything
- Users unsure if this is right path

**False Scent**
- Label misleads users
- Example: "Settings" → Only shows account info, not all settings
- Users frustrated, waste time

### Patterns

**Specific, Action-Oriented Labels**
- Good: "Download Invoice", "Change Password", "View Order History"
- Bad: "Account", "Options", "More"
- Users scan for action verbs matching their goal

**Front-Loading Keywords**
- Good: "Projects > Project Alpha > Documents"
- Bad: "All > Alpha > Files and Documents for Projects"
- Users scan left-to-right, see important words first

**Descriptive Category Names**
- Good: "User Settings", "Billing Information", "Security Preferences"
- Bad: "Preferences", "Options", "More"
- Specific > generic

**Progressive Refinement**
- Each level narrows scope
- Example: Electronics > Laptops > Gaming Laptops > 17-inch Gaming Laptops
- Scent gets stronger at each level

**Preview/Descriptions for Ambiguous Items**
- When label can't be specific, add description
- Example: "Advanced" (label) + "For power users: API keys, webhooks, custom scripts" (description)
- Clarifies before user clicks

**Avoid Overlapping Categories**
- Bad: "Accessories" + "Cables" (cables are accessories)
- Users check both, waste time
- Make mutually exclusive or nest hierarchically

### Anti-Patterns

**Vague Labels**
- "Stuff", "Other", "More", "Miscellaneous"
- Tells users nothing about content
- If necessary, add description

**Marketing Speak**
- "Solutions", "Innovative Tools", "Next-Gen Platform"
- Sounds impressive, means nothing
- Use plain language users understand

**Icon-Only Labels (No Text)**
- Users guess what icon means
- Different cultures interpret icons differently
- Use icon + text (or tooltip)

**Jargon Users Don't Know**
- Internal product names ("Project Phoenix Dashboard")
- Industry acronyms ("SDLC", "KPIs") without context
- Test with users unfamiliar with domain

**Same Label for Different Things**
- "Settings" in menu → All settings
- "Settings" in modal → Only this feature's settings
- Creates confusion, breaks trust

**Label Doesn't Match Content**
- Link says "View Profile", page shows "Account Settings"
- Breaks trust, wastes time
- Ensure label = page title/content

### Practical Application

**Step 1:** Audit all labels
- List every menu item, button, link
- Ask: "Does this label clearly predict destination?"

**Step 2:** Test with users (5-second test)
- Show label for 5 seconds
- Ask: "What do you expect to find here?"
- Compare to actual content

**Step 3:** Refine labels
- Replace vague with specific
- Front-load keywords
- Use user terminology (from card sorting)

**Step 4:** Validate with tree testing
- Users navigate text-only IA tree
- Measure task success rate
- Iterate on failing labels


## Layer 4: Discoverability

**Purpose:** Users can find features beyond the basics, progressively

### Evaluation Questions

- Can users discover advanced features?
- Is there progressive disclosure (simple first, advanced later)?
- Are there tooltips/contextual help?
- Do features have logical placement (near related functionality)?
- Is there onboarding for core features?
- Can users learn without documentation?

### Patterns

**Progressive Disclosure**
- Show basics by default
- Hide advanced features behind "Advanced" button/section
- Example: Simple form with 3 fields + "More options" reveals 10 more
- Prevents overwhelming new users, empowers advanced users

**Onboarding for Core Features**
- First-time users see guided tour
- Highlights 3-5 essential features
- Optional skip (don't force)
- Example: "Here's how to create your first project"

**Contextual Help**
- "?" icon next to complex fields
- Tooltips on hover (desktop) or tap (mobile)
- Inline explanations (not separate docs)
- Example: "What's this?" next to "API Key"

**Logical Feature Placement**
- Related features grouped together
- Example: "Export as PDF" near "Print"
- Users discover by exploring related functionality

**Empty States with Guidance**
- When no content exists, show how to create content
- Example: "No projects yet. Create your first project."
- Provides clear next action

**Search for Hidden Features**
- Power users use search to find advanced features
- Ensure all features indexed by search
- Example: Search for "API" finds API settings

**"Learn More" Links**
- Link to docs/help for complex features
- Non-intrusive (users opt-in)
- Example: "Learn about webhooks" link in webhook settings

**Feature Announcements**
- Toast/modal when new feature ships
- "Dismiss" option (one-time)
- Example: "New: Dark mode in settings"

### Anti-Patterns

**Power Features Buried 5+ Levels Deep**
- Users never discover them
- Balance: Progressive disclosure vs discoverability
- Solution: Provide search, contextual hints

**No Onboarding (Users Never Discover Features)**
- Blank screen with no guidance
- Users churn because they don't understand value
- Solution: Quick tour highlighting core features

**Everything Exposed at Once (Overwhelming)**
- 50 buttons on screen
- Analysis paralysis, cognitive overload
- Solution: Progressive disclosure, hide advanced

**No Contextual Help**
- Complex features with no explanation
- Users trial-and-error or leave
- Solution: Tooltips, inline help, "?" icons

**Inconsistent Placement**
- "Settings" in different places on different screens
- Users can't predict where to look
- Solution: Consistent placement (top-right for settings is convention)

**Dead-End Experiences**
- Users reach feature but don't know what to do
- No placeholder text, no examples, no guidance
- Solution: Empty states with next steps

**Features Without Labels**
- Icon-only buttons (users guess)
- Hover-only discoverability (fails on mobile)
- Solution: Icons + labels or clear tooltips

### Practical Application

**Step 1:** Identify core vs advanced features
- Core: 80% of users need these
- Advanced: 20% of users, power features

**Step 2:** Design progressive disclosure
- Core features: Visible by default
- Advanced features: Behind "Advanced" or "More"
- Ensure advanced still discoverable (search, contextual hints)

**Step 3:** Add contextual help
- Complex fields get tooltips
- First-time experience gets onboarding
- Empty states explain next actions

**Step 4:** Test with new users
- Give users tasks (don't tell them how)
- Observe: Do they discover features?
- Iterate: Add hints where users get stuck


## User Journey Integration

Information architecture must support complete user journeys:

### Journey Mapping for IA

**Step 1: Map Primary User Journeys**
- List top 5-10 tasks users do
- Example: "Find product", "Check order status", "Contact support"

**Step 2: Identify Touchpoints**
- Where does user enter? (Google, home page, deep link)
- What pages do they visit? (in what order)
- Where do they exit? (goal complete or frustrated)

**Step 3: Evaluate IA Against Journeys**
- Can users complete journey with current IA?
- Where do they get stuck? (missing scent, dead ends)
- Are there unnecessary steps? (simplify)

**Step 4: Optimize IA for Common Journeys**
- Put common endpoints in global nav
- Add shortcuts for frequent paths
- Remove obstacles (reduce clicks, clarify labels)

### Example: E-Commerce Journey

**Journey:** Find and purchase product

**Touchpoints:**
1. Enter via Google search → Product page (deep link)
2. View product details
3. Add to cart
4. View cart
5. Checkout
6. Confirmation

**IA Optimizations:**
- Breadcrumbs on product page (Layer 2: Navigation)
- Related products (Layer 2: Contextual nav)
- Clear "Add to Cart" button (Layer 3: Information scent)
- Persistent cart icon (Layer 4: Discoverability)
- Guest checkout option (Layer 1: Mental model - not everyone wants account)


## Cross-Platform Navigation Considerations

Navigation patterns adapt to platform context:

### Mobile Navigation

**Constraints:**
- Small screen (375px width typical)
- Thumb-based interaction
- One screen at a time (no multi-window)

**Patterns:**
- Bottom tab bar (3-5 items, iOS standard)
- Hamburger menu (for 6+ items)
- Bottom sheet menus (Android)
- Swipe gestures (swipe back, swipe between tabs)

**Prioritization:**
- Most important actions at bottom (thumb zone)
- Secondary actions in hamburger menu
- Search prominent (magnifying glass icon)

**Reference:** See `mobile-design-patterns` for platform specifics

### Web Application Navigation

**Affordances:**
- Large screen (1024px+ width)
- Keyboard + mouse
- Multi-tab workflows

**Patterns:**
- Horizontal top nav (persistent)
- Sidebar nav (for deep hierarchies)
- Breadcrumbs (show location in hierarchy)
- Mega-menus (reveal 2+ levels on hover)
- Keyboard shortcuts (Cmd+K command palette)

**Considerations:**
- Responsive: Collapses to mobile patterns on narrow screens
- Dense information: Can show more hierarchy on desktop

**Reference:** See `web-application-design` for responsive patterns

### Desktop Software Navigation

**Affordances:**
- Multi-window support
- Extensive keyboard shortcuts
- Menu bars

**Patterns:**
- Menu bar (File, Edit, View, Help)
- Toolbars (icon buttons, customizable)
- Panels/palettes (floating or docked)
- Context menus (right-click)

**Considerations:**
- Keyboard-first: Every menu item has shortcut
- Customization: Users rearrange toolbars/panels

**Reference:** See `desktop-software-design` for window management

### Game UI Navigation

**Constraints:**
- Gamepad/controller input (limited buttons)
- Immersion (minimize UI)
- Performance (UI can't tank frame rate)

**Patterns:**
- Radial menus (analog stick selection)
- D-pad for menus (up/down/left/right)
- Pause menu (Start button)
- Context-sensitive prompts ("Press X to interact")

**Considerations:**
- Minimize navigation: Players want to play, not navigate menus
- Contextual: Show only relevant options for current situation

**Reference:** See `game-ui-design` for immersion vs visibility tradeoffs


## Navigation Pattern Library

Common navigation solutions with rationale:

### Pattern: Bottom Tab Bar (Mobile)

**Use When:**
- Mobile app with 3-5 primary sections
- Users frequently switch between sections
- Following iOS Human Interface Guidelines

**Structure:**
- 3-5 tabs at bottom
- Always visible
- Icon + label (or icon only if space constrained)
- Active tab highlighted

**Example:** Instagram (Home, Search, Create, Reels, Profile)


### Pattern: Hamburger Menu (Mobile)

**Use When:**
- 6+ navigation items (too many for tabs)
- Secondary functions (not primary navigation)
- Android or web mobile

**Structure:**
- ☰ icon (top-left or top-right)
- Slides out drawer with nav items
- Overlay or push content

**Warning:** Reduces discoverability ("out of sight, out of mind")


### Pattern: Breadcrumbs (Web/Desktop)

**Use When:**
- Deep hierarchies (3+ levels)
- Users need to understand location
- E-commerce, documentation, file systems

**Structure:**
- Home > Category > Subcategory > Current Page
- Each level clickable (except current)
- Shows path from root to current

**Example:** Amazon product pages


### Pattern: Mega-Menu (Web)

**Use When:**
- Complex site with many categories
- Need to show 2-3 levels of hierarchy at once
- Desktop web (requires hover)

**Structure:**
- Hover top nav item
- Reveals large dropdown with subcategories
- Grid or column layout

**Warning:** Doesn't work on mobile (no hover), requires desktop


### Pattern: Sidebar Navigation (Web/Desktop)

**Use When:**
- Deep hierarchies (documentation, admin panels)
- Always need to show navigation context
- Desktop-first or responsive web

**Structure:**
- Left sidebar (persistent or collapsible)
- Hierarchical tree (expand/collapse sections)
- Active page highlighted

**Example:** Documentation sites, admin dashboards


### Pattern: Command Palette (Web/Desktop)

**Use When:**
- Power users need fast access to all commands
- Many features (50+ actions)
- Keyboard-first workflows

**Structure:**
- Keyboard shortcut (Cmd+K common)
- Fuzzy search all actions
- Jump to any page/feature instantly

**Example:** VS Code, Slack, Figma


## Red Flags & Anti-Patterns

### Red Flag: Users Can't Find Primary Features

**Symptom:** Support requests "Where is [feature]?"

**Root Cause:** Poor information scent or buried navigation

**Fix:**
- Audit Layer 3 (Information Scent): Are labels clear?
- Add to global navigation (Layer 2)
- Provide search (Layer 2)


### Red Flag: Navigation Inconsistent Across Screens

**Symptom:** Menu changes or moves on different pages

**Root Cause:** Lack of navigation system standards

**Fix:**
- Define persistent global navigation
- Same placement, same labels everywhere
- Test on 10+ different screens


### Red Flag: One Way to Reach Content

**Symptom:** Users lost if they don't know exact path

**Root Cause:** Single navigation path (Layer 2 failure)

**Fix:**
- Add redundant paths (global nav + search + contextual links)
- Provide breadcrumbs for backtracking
- Test with users who enter via deep links


### Red Flag: Users Complete Task But Say "That Was Confusing"

**Symptom:** Low satisfaction despite task completion

**Root Cause:** Weak information scent (Layer 3) - users succeeded through trial-and-error

**Fix:**
- Improve label specificity
- Add descriptions to ambiguous categories
- Test with 5-second test (label → expected content)


### Red Flag: Power Users Can't Find Advanced Features

**Symptom:** "Does this have [feature]?" (feature exists but hidden)

**Root Cause:** Poor discoverability (Layer 4)

**Fix:**
- Add search (all features indexed)
- Contextual hints ("Looking for advanced options? Click here")
- Feature announcements when new features ship


### Red Flag: New Users Overwhelmed, Churn Quickly

**Symptom:** High bounce rate, low activation

**Root Cause:** No progressive disclosure, everything shown at once

**Fix:**
- Hide advanced features behind "Advanced" section
- Add onboarding (highlight core features)
- Empty states with guidance


### Red Flag: Users Click Wrong Category, Backtrack Repeatedly

**Symptom:** Analytics show high back-button usage, category exploration

**Root Cause:** Labels don't match user mental model (Layer 1) or weak scent (Layer 3)

**Fix:**
- Card sorting to understand user categories
- Relabel based on user terminology
- Tree testing to validate


## Related Skills

**Cross-references to other Lyra skills:**

- `lyra/ux-designer/user-research-and-validation` - Card sorting, tree testing, mental model research methods for IA validation
- `lyra/ux-designer/interaction-design-patterns` - Navigation interactions (menus, dropdowns, tabs) and micro-interactions for nav elements
- `lyra/ux-designer/visual-design-foundations` - Visual hierarchy to emphasize navigation, typography for readable labels
- `lyra/ux-designer/accessibility-and-inclusive-design` - Keyboard navigation, screen reader compatibility for nav systems
- `lyra/ux-designer/ux-fundamentals` - Core IA principles and terminology

**Platform-specific IA:**

- `lyra/ux-designer/mobile-design-patterns` - Mobile navigation patterns (bottom tabs, hamburger menus, gestures)
- `lyra/ux-designer/web-application-design` - Complex web nav systems (mega-menus, breadcrumbs, responsive patterns)
- `lyra/ux-designer/desktop-software-design` - Desktop navigation (menu bars, toolbars, keyboard shortcuts)
- `lyra/ux-designer/game-ui-design` - Game navigation (radial menus, contextual prompts, minimal UI)

**Cross-faction references:**

- `muna/technical-writer/documentation-structure` - IA for documentation sites (organizing docs for findability)
- `muna/technical-writer/clarity-and-style` - Writing clear labels and navigation copy
