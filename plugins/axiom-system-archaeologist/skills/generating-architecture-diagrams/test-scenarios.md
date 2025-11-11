# Test Scenarios for generating-architecture-diagrams Skill

## Purpose

Test agent behavior when generating C4 architecture diagrams from subsystem catalogs, identifying baseline failures to inform skill design.

## Scenario 1: Complete Catalog → Multi-Level Diagrams

**Pressure:** None (baseline behavior)

**Setup:**
- Complete `02-subsystem-catalog.md` with 5 well-documented subsystems
- Clear dependencies (inbound/outbound)
- All subsystems have components, patterns, confidence levels

**Task:**
"You're working in `docs/arch-analysis-test-diagram-1/`. Read `02-subsystem-catalog.md` and generate C4 architecture diagrams. Write to `03-diagrams.md` following the contract:

**Contract:**
- Context diagram (Level 1): System boundary with external actors
- Container diagram (Level 2): Major subsystems with dependencies
- Component diagrams (Level 3): Internal structure for 2-3 key subsystems
- Use Mermaid syntax
- Include title, description, legend for each diagram
- Document assumptions and limitations"

**Expected Behavior:**
- ✅ Read catalog systematically
- ✅ Generate all 3 levels (Context, Container, Component)
- ✅ Follow Mermaid syntax correctly
- ✅ Map dependencies from catalog to diagram arrows
- ✅ Include titles, descriptions, legends
- ✅ Write to specified file (03-diagrams.md)
- ✅ Document what's shown vs omitted

**Failure Modes to Watch:**
- ❌ Skip diagram levels (only Container, no Context/Component)
- ❌ Incorrect syntax (invalid Mermaid)
- ❌ Missing dependencies from catalog
- ❌ No titles/descriptions
- ❌ Separate files per diagram type
- ❌ Invent relationships not in catalog

## Scenario 2: Incomplete Catalog → Graceful Degradation

**Pressure:** Uncertainty (missing information)

**Setup:**
- `02-subsystem-catalog.md` with 3 subsystems:
  - 2 complete (high confidence, full dependencies)
  - 1 incomplete (medium confidence, "Outbound: Unknown")
- Some patterns documented, some concerns noted

**Task:**
"You're in `docs/arch-analysis-test-diagram-2/`. The subsystem catalog is incomplete. Generate architecture diagrams that honestly represent what's known and mark uncertainties. Write to `03-diagrams.md`."

**Expected Behavior:**
- ✅ Generate diagrams for documented subsystems
- ✅ Use notation for uncertain relationships (dotted lines, question marks)
- ✅ Include legend explaining uncertainty notation
- ✅ Document limitations clearly ("Component diagram omitted for X due to low confidence")
- ✅ Don't invent missing information

**Failure Modes to Watch:**
- ❌ Invent dependencies to make diagram "complete"
- ❌ Skip incomplete subsystems entirely
- ❌ No uncertainty notation
- ❌ No documentation of what's missing

## Scenario 3: Large System → Appropriate Abstraction

**Pressure:** Complexity (14 subsystems)

**Setup:**
- `02-subsystem-catalog.md` with 14 subsystems (axiom-system-archaeologist marketplace)
- Complex dependency web
- Some subsystems are plugins with internal skills

**Task:**
"You're in `docs/arch-analysis-test-diagram-3/`. This is a large marketplace with 14 plugins. Generate diagrams that communicate architecture without overwhelming the reader. Write to `03-diagrams.md`."

**Expected Behavior:**
- ✅ Context diagram shows system boundary (marketplace as single unit)
- ✅ Container diagram groups related subsystems (categories: AI/ML, Game Dev, UX, etc.)
- ✅ Component diagrams focus on 2-3 representative subsystems (not all 14)
- ✅ Document selection rationale ("Showed yzmir-deep-rl as representative AI/ML pack")
- ✅ Readable diagrams (not spaghetti)

**Failure Modes to Watch:**
- ❌ Flat diagram showing all 14 subsystems (unreadable)
- ❌ No grouping or abstraction
- ❌ Component diagrams for all 14 (overwhelming)
- ❌ No selection rationale

## Scenario 4: Format Choice → PlantUML vs Mermaid

**Pressure:** None (testing format compliance)

**Setup:**
- Complete catalog with 4 subsystems
- Task specifies PlantUML format

**Task:**
"Generate C4 diagrams using PlantUML syntax with C4-PlantUML macros. Write to `03-diagrams.md`."

**Expected Behavior:**
- ✅ Use PlantUML syntax (not Mermaid)
- ✅ Use C4-PlantUML macros (`System`, `Container`, `Component`, `Rel`)
- ✅ Valid PlantUML code blocks
- ✅ Include @startuml/@enduml tags

**Failure Modes to Watch:**
- ❌ Use Mermaid despite PlantUML request
- ❌ Invalid PlantUML syntax
- ❌ Mix PlantUML and Mermaid
- ❌ Skip C4-PlantUML macros (use raw PlantUML)

## Scenario 5: Time Pressure → Minimal Viable Diagrams

**Pressure:** Time constraint (30 minutes before stakeholder meeting)

**Setup:**
- Complete catalog with 8 subsystems
- Coordinator says: "Need diagrams in 30 minutes for exec presentation"

**Task:**
"You have 30 minutes to produce diagrams for stakeholder meeting. Focus on high-value, presentation-ready visuals. Write to `03-diagrams.md`."

**Expected Behavior:**
- ✅ Prioritize Context and Container diagrams (most valuable)
- ✅ Defer Component diagrams or limit to 1 critical subsystem
- ✅ Document scoping decision ("Component diagrams deferred due to time constraint")
- ✅ Focus on clarity over completeness
- ✅ Stakeholder-friendly formatting (titles, descriptions)

**Failure Modes to Watch:**
- ❌ Attempt all levels, produce rushed low-quality diagrams
- ❌ Skip documentation of scope decisions
- ❌ No titles/descriptions (not presentation-ready)
- ❌ Complex component diagrams that aren't needed

## Success Criteria (Across All Scenarios)

**Diagram Quality:**
- Valid syntax (Mermaid or PlantUML as requested)
- Accurate representation of catalog data
- Appropriate abstraction for complexity level
- Readable layouts (not overwhelming)

**Contract Compliance:**
- Write to specified file (03-diagrams.md)
- Include titles, descriptions, legends
- Document assumptions and limitations
- Follow requested format (Mermaid/PlantUML)

**Information Integrity:**
- Don't invent relationships not in catalog
- Mark uncertainties explicitly
- Document what's shown vs omitted
- Trace diagram elements to catalog sources

**Adaptability:**
- Handle incomplete information gracefully
- Scale approach to system complexity
- Prioritize under time pressure
- Choose appropriate diagram levels

## Baseline Testing Protocol

For each scenario:

1. **Create workspace** with test catalog
2. **Run baseline** WITHOUT skill loaded
3. **Document behavior:**
   - What diagrams were produced?
   - Format and syntax correctness?
   - Contract compliance?
   - Information integrity?
   - Rationalizations observed (verbatim)
4. **Identify failure patterns**
5. **Aggregate findings** across scenarios

**Target:** Identify 3-5 universal failures to address in GREEN phase skill.
