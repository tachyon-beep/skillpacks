
# Analyzing Unknown Codebases

## Purpose

Systematically analyze unfamiliar code to identify subsystems, components, dependencies, and architectural patterns. Produce catalog entries that follow EXACT output contracts.

## When to Use

- Coordinator delegates subsystem analysis task
- Task specifies reading from workspace and appending to `02-subsystem-catalog.md`
- You need to analyze code you haven't seen before
- Output must integrate with downstream tooling (validation, diagram generation)

## Critical Principles

### Principle 1: You MUST Read Implementation Code

**Analysis requires reading actual source files, not just structure.**

You CANNOT produce valid analysis by only:
- Running `wc -l` or `grep -c` to count lines/functions
- Reading file names and directory structure
- Looking at imports without reading implementations
- Using metadata files (package.json, plugin.json) alone

**If you haven't opened and read source files, you haven't analyzed anything.**

### Principle 2: Contract Compliance

**Your analysis quality doesn't matter if you violate the output contract.**

**Common rationalization:** "I'll add helpful extra sections to improve clarity"

**Reality:** Extra sections break downstream tools. The coordinator expects EXACT format for parsing and validation. Your job is to follow the specification, not improve it.

## Output Contract (MANDATORY)

When writing to `02-subsystem-catalog.md`, append EXACTLY this format:

```markdown
## [Subsystem Name]

**Location:** `path/to/subsystem/`

**Responsibility:** [One sentence describing what this subsystem does]

**Key Components:**
- `file1.ext` - [Brief description]
- `file2.ext` - [Brief description]
- `file3.ext` - [Brief description]

**Dependencies:**
- Inbound: [Subsystems that depend on this one]
- Outbound: [Subsystems this one depends on]

**Patterns Observed:**
- [Pattern 1]
- [Pattern 2]

**Concerns:**
- [Any issues, gaps, or technical debt observed]

**Confidence:** [High/Medium/Low] - [Brief reasoning]

```

**If no concerns exist, write:**
```markdown
**Concerns:**
- None observed
```

**CRITICAL COMPLIANCE RULES:**
- ❌ Add extra sections ("Integration Points", "Recommendations", "Files", etc.)
- ❌ Change section names or reorder them
- ❌ Write to separate file (must append to `02-subsystem-catalog.md`)
- ❌ Skip sections (include ALL sections - use "None observed" if empty)
- ✅ Copy the template structure EXACTLY
- ✅ Keep section order: Location → Responsibility → Key Components → Dependencies → Patterns → Concerns → Confidence

**Contract is specification, not minimum. Extra sections break downstream validation.**

### Example: Complete Compliant Entry

Here's what a correctly formatted entry looks like:

```markdown
## Authentication Service

**Location:** `/src/services/auth/`

**Responsibility:** Handles user authentication, session management, and JWT token generation for API access.

**Key Components:**
- `auth_handler.py` - Main authentication logic with login/logout endpoints (342 lines)
- `token_manager.py` - JWT token generation and validation (156 lines)
- `session_store.py` - Redis-backed session storage (98 lines)

**Dependencies:**
- Inbound: API Gateway, User Service
- Outbound: Database Layer, Cache Service, Logging Service

**Patterns Observed:**
- Dependency injection for testability (all external services injected)
- Token refresh pattern with sliding expiration
- Audit logging for all authentication events

**Concerns:**
- None observed

**Confidence:** High - Clear entry points, documented API, test coverage validates behavior

```

**This is EXACTLY what your output should look like.** No more, no less.

## Minimum Analysis Standards (NON-NEGOTIABLE)

**These standards apply to ALL analysis tasks, regardless of codebase size.**

**For EVERY subsystem analyzed, you MUST:**

1. **Read 100% of metadata files** (plugin.json, package.json, setup.py, etc.)
2. **Read 100% of router/index files** (entry points, skill routers, __init__.py)
3. **For main code files:**
   - Files ≤500 lines: Read 100%
   - Files >500 lines: Read entry points + sample 3+ representative functions
4. **Cross-validate all dependency claims** - Check imports match stated dependencies
5. **Document evidence trail** - Every claim in output must cite file/line you read

**STOP SIGNALS for rationalization:**

❌ "It's simple, so I don't need to read everything"
→ Simple codebases still have patterns worth documenting. Read the code.

❌ "The structure is obvious from names"
→ Names ≠ implementation. Obvious structure may hide non-obvious details. Read the code.

❌ "I can infer behavior without reading code"
→ Inference ≠ analysis. If you haven't read it, you don't know it. Read the code.

❌ "There are only 3 files, I'll just list them"
→ 3 files still need implementation analysis. Read each one. Document what you found.

**Example of INSUFFICIENT analysis:**
```markdown
**Key Components:**
- `main.py` - Main code
```

**Example of COMPLIANT analysis:**
```markdown
**Key Components:**
- `main.py` - Plugin entry point with 3 command handlers (lines 45-180), configuration loader (lines 20-43), and error handling wrapper (lines 181-200)
```

## Systematic Analysis Approach

### Step 1: Read Task Specification

Your task file (`temp/task-[name].md`) specifies:
- What to analyze (scope: directories, plugins, services)
- Where to read context (`01-discovery-findings.md`)
- Where to write output (`02-subsystem-catalog.md` - append)
- Expected format (the contract above)

**Read these files FIRST before analyzing code.**

### Step 2: Layered Exploration

Use this proven approach from baseline testing:

1. **Metadata layer** - Read plugin.json, package.json, setup.py
2. **Structure layer** - Examine directory organization
3. **Router layer** - Find and read router/index files (often named "using-X")
4. **Sampling layer** - Read 3-5 representative files
5. **Quantitative layer** - Use line counts as depth indicators

**Why this order works:**
- Metadata gives overview without code diving
- Structure reveals organization philosophy
- Routers often catalog all components
- Sampling verifies patterns
- Quantitative data supports claims

### Step 3: Mark Confidence Explicitly WITH EVIDENCE

**Every output MUST include confidence level with SPECIFIC EVIDENCE.**

**High confidence** - Read all critical files, verified behavior, cross-checked dependencies
```markdown
**Confidence:** High - Read plugin.json + all 3 router skills + sampled 5/12 implementation files. Cross-verified dependencies by checking imports against package.json. Documented patterns from lines 45-120 in auth_handler.py.
```

**Medium confidence** - Read representative sample, clear structure, some gaps
```markdown
**Confidence:** Medium - Read plugin.json + 4/8 skills + directory structure. Dependencies inferred from imports (not verified against manifest). Implementation details sampled from 3 files.
```

**Low confidence** - Incomplete information, unclear organization, significant gaps
```markdown
**Confidence:** Low - Several SKILL.md files missing, test artifacts suggest work-in-progress, no clear entry point identified, dependencies uncertain.
```

**MANDATORY EVIDENCE REQUIREMENTS:**
- Cite specific files read (e.g., "Read plugin.json + auth_handler.py + token_manager.py")
- Cite specific lines for claims (e.g., "lines 45-180 implement handlers")
- Cite verification steps (e.g., "verified imports match package.json dependencies")

**INSUFFICIENT confidence marking:**
```markdown
**Confidence:** High - Small codebase, easy to understand
```

**COMPLIANT confidence marking:**
```markdown
**Confidence:** High - Read all 3 files (plugin.json, README.md, main.py 200 lines). Verified dependencies by checking imports (lines 1-15) against plugin.json. Patterns documented from implementation review.
```

### Step 4: Distinguish States Clearly

When analyzing codebases with mixed completion:

**Complete** - Skill file exists, has content, passes basic read test
```markdown
- `skill-name/SKILL.md` - Complete skill (1,234 lines)
```

**Placeholder** - Skill file exists but is stub/template
```markdown
- `skill-name/SKILL.md` - Placeholder (12 lines, template only)
```

**Planned** - Referenced in router but no file exists
```markdown
- `skill-name` - Planned (referenced in router, not implemented)
```

**TDD artifacts** - Test scenarios, baseline results (these ARE documentation)
```markdown
- `test-scenarios.md` - TDD test scenarios (RED phase)
- `baseline-results.md` - Baseline behavior documentation
```

### Step 5: Write Output (Contract Compliance)

**Before writing:**
1. Prepare your entry in EXACT contract format from the template above
2. Copy the structure - don't paraphrase or reorganize
3. Triple-check you have ALL sections in correct order

**When writing:**
1. **Target file:** `02-subsystem-catalog.md` in workspace directory
2. **Operation:** Append your entry (create file if first entry, append if file exists)
3. **Method:**
   - If file exists: Read current content, then Write with original + your entry
   - If file doesn't exist: Write your entry directly
4. **Format:** Follow contract sections in exact order
5. **Completeness:** Include ALL sections - use "None observed" for empty Concerns

**DO NOT create separate files** (e.g., `subsystem-X-analysis.md`). The coordinator expects all entries in `02-subsystem-catalog.md`.

**After writing:**
1. Re-read `02-subsystem-catalog.md` to verify your entry was added correctly
2. Validate format matches contract exactly using this checklist:

**Self-Validation Checklist:**
```
[ ] Section 1: Subsystem name as H2 heading (## Name)
[ ] Section 2: Location with backticks and absolute path
[ ] Section 3: Responsibility as single sentence
[ ] Section 4: Key Components as bulleted list with descriptions
[ ] Section 5: Dependencies with "Inbound:" and "Outbound:" labels
[ ] Section 6: Patterns Observed as bulleted list
[ ] Section 7: Concerns present (with issues OR "None observed")
[ ] Section 8: Confidence level (High/Medium/Low) with reasoning
[ ] Separator: "---" line after confidence
[ ] NO extra sections added
[ ] Sections in correct order
[ ] Entry in file: 02-subsystem-catalog.md (not separate file)
```

## Mandatory Concern-Finding Checklist

**Before writing "None observed" in Concerns, verify you checked for:**

**Code Quality:**
- [ ] Error handling present for failure cases?
- [ ] Input validation for user-facing functions?
- [ ] Resource cleanup (file handles, connections, etc.)?
- [ ] Logging/observability for debugging?

**Architecture:**
- [ ] Clear separation of concerns?
- [ ] Dependencies properly abstracted?
- [ ] Configuration externalized vs hardcoded?
- [ ] Entry points clearly documented?

**Completeness:**
- [ ] TODOs, FIXMEs, or placeholder comments?
- [ ] Missing test coverage indicators?
- [ ] Incomplete implementations (stubs, NotImplementedError)?
- [ ] Documentation gaps?

**If you checked all items and found nothing, THEN write:**
```markdown
**Concerns:**
- None observed (verified: error handling, validation, architecture, completeness)
```

**If you skipped this checklist, your analysis is INCOMPLETE.**

## Handling Uncertainty

**When architecture is unclear:**

1. **State what you observe** - Don't guess at intent
   ```markdown
   **Patterns Observed:**
   - 3 files with similar structure (analysis.py, parsing.py, validation.py)
   - Unclear if this is deliberate pattern or coincidence
   ```

2. **Mark confidence appropriately** - Low confidence is valid
   ```markdown
   **Confidence:** Low - Directory structure suggests microservices, but no service definitions found
   ```

3. **Use "Concerns" section** - Document gaps
   ```markdown
   **Concerns:**
   - No clear entry point identified
   - Dependencies inferred from imports, not explicit manifest
   ```

**DO NOT:**
- Invent relationships you didn't verify
- Assume "obvious" architecture without evidence
- Skip confidence marking because you're uncertain

## Positive Behaviors to Maintain

From baseline testing, these approaches WORK:

✅ **Read actual files** - Don't infer from names alone
✅ **Use router skills** - They often provide complete catalogs
✅ **Sample strategically** - 3-5 files verifies patterns without exhaustive reading
✅ **Cross-reference** - Verify claims (imports match listed dependencies)
✅ **Document assumptions** - Make reasoning explicit
✅ **Line counts indicate depth** - 1,500-line skill vs 50-line stub matters

## Common Rationalizations (STOP SIGNALS)

If you catch yourself thinking these, STOP:

| Rationalization | Reality |
|-----------------|---------|
| "I'll add Integration Points section for clarity" | Extra sections break downstream parsing |
| "I'll write to separate file for organization" | Coordinator expects append to specified file |
| "I'll improve the contract format" | Contract is specification from coordinator |
| "More information is always helpful" | Your job: follow spec. Coordinator's job: decide what's included |
| "This comprehensive format is better" | "Better" violates contract. Compliance is mandatory. |

## Validation Criteria

Your output will be validated against:

1. **Contract compliance** - All sections present, no extras
2. **File operation** - Appended to `02-subsystem-catalog.md`, not separate file
3. **Confidence marking** - High/Medium/Low with reasoning
4. **Evidence-based claims** - Components you actually read
5. **Bidirectional dependencies** - If A→B, then B must show A as inbound

**If validation returns NEEDS_REVISION:**
- Read the validation report
- Fix specific issues identified
- Re-submit following contract

## Success Criteria

**You succeeded when:**
- Entry appended to `02-subsystem-catalog.md` in exact contract format
- All sections included (none skipped, none added)
- Confidence level marked with reasoning
- Claims supported by files you read
- Validation returns APPROVED

**You failed when:**
- Added "helpful" extra sections
- Wrote to separate file
- Changed contract format
- Skipped sections
- No confidence marking
- Validation returns BLOCK status

## Anti-Patterns

❌ **Add extra sections**
"I'll add Recommendations section" → Violates contract

❌ **Write to new file**
"I'll create subsystem-X-analysis.md" → Should append to `02-subsystem-catalog.md`

❌ **Skip required sections**
"No concerns, so I'll omit that section" → Include section with "None observed"

❌ **Change format**
"I'll use numbered lists instead of bullet points" → Follow contract exactly

❌ **Work without reading task spec**
"I know what to do" → Read `temp/task-*.md` first

❌ **Shallow analysis of "simple" code**
"It's only 3 files, so I'll just list them" → Must analyze implementation details, patterns, dependencies regardless of size

❌ **Skip concern-finding process**
"Everything looks fine" → Must verify using concern-finding checklist before claiming "None observed"

❌ **Confidence without evidence**
"High confidence because it's simple" → Must cite specific files read, lines checked, verification steps performed

❌ **Pattern documentation without verification**
"Uses standard pattern" → Must cite specific code locations demonstrating the pattern

❌ **Structural metrics as analysis**
"File has 200 lines, 5 functions" → Line counts from `wc` are not analysis. Read the actual code.

## Integration with Workflow

This skill is typically invoked as:

1. **Coordinator** creates workspace and holistic assessment
2. **Coordinator** writes task specification in `temp/task-[yourname].md`
3. **YOU** read task spec + `01-discovery-findings.md`
4. **YOU** analyze assigned subsystem systematically
5. **YOU** append entry to `02-subsystem-catalog.md` following contract
6. **Validator** checks your output against contract
7. **Coordinator** proceeds to next phase if validation passes

**Your role:** Analyze systematically, follow contract exactly, mark confidence explicitly.
