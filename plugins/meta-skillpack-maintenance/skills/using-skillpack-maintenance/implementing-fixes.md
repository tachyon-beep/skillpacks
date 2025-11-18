# Implementing Fixes

**Purpose:** Autonomous execution of approved changes with version management and git commit.

## Prerequisites

You should have completed and gotten approval for:
- Pass 1: Structure review (gaps, duplicates, organization)
- Pass 2: Content testing (gauntlet results, fix priorities)
- Pass 3: Coherence validation (cross-skill consistency, faction alignment)
- User discussion and approval of scope

**Do NOT proceed without user approval of the scope of work.**

## Execution Workflow

### 1. Structural Fixes (from Pass 1)

**Add missing skills:**

For each identified gap:
1. Create skill directory: `plugins/[pack-name]/skills/[skill-name]/`
2. Create SKILL.md with:
   - YAML front matter (name, description following CSO guidelines)
   - Overview section
   - Core content based on gap analysis
   - Examples appropriate to domain
   - Common mistakes section
3. Ensure skill follows faction voice/philosophy
4. Add cross-references to related skills

**Remove duplicate skills:**

For skills marked for removal:
1. Identify unique value in skill being removed
2. Merge unique value into kept skill (if any)
3. Delete duplicate SKILL.md and directory
4. Remove references from router (if exists)
5. Update cross-references in other skills

**Merge overlapping skills:**

For partial duplicates:
1. Identify all unique content from both skills
2. Create merged skill with comprehensive coverage
3. Reorganize structure if needed
4. Delete original skills
5. Update router and cross-references
6. Update skill name/description if needed

**Update router skill:**

If pack has using-X router:
1. Update specialist list to reflect adds/removes
2. Update descriptions to match current skills
3. Verify routing logic makes sense
4. Add cross-references as needed

### 2. Content Enhancements (from Pass 2)

**For each skill marked "Fix needed" in gauntlet testing:**

**Fix rationalizations (A-type issues):**
1. Add explicit counter for each identified rationalization
2. Update "Common Rationalizations" table
3. Add to "Red Flags" list if applicable
4. Strengthen "No exceptions" language

**Fill edge case gaps (C-type issues):**
1. Add guidance for identified corner cases
2. Document when/how to adapt core principles
3. Add examples for edge case handling
4. Cross-reference related skills if needed

**Enhance real-world guidance (B-type issues):**
1. Add examples from realistic scenarios
2. Clarify ambiguous instructions
3. Add decision frameworks where needed
4. Update "When to Use" section if unclear

**Add anti-patterns:**
1. Document observed failure modes from testing
2. Add ❌ WRONG / ✅ CORRECT examples
3. Update "Common Mistakes" section
4. Add warnings for subtle pitfalls

**Improve examples:**
1. Replace weak examples with tested scenarios
2. Ensure examples are complete and runnable
3. Add comments explaining WHY, not just WHAT
4. Use realistic domain context

### 3. Coherence Improvements (from Pass 3)

**Cross-reference updates:**

**CRITICAL:** This is post-update hygiene - ensure skills reference new/enhanced skills.

For each skill in pack:
1. Identify related skills (related concepts, prerequisites, follow-ups)
2. Add cross-references where helpful:
   - "See [skill-name] for [related concept]"
   - "**REQUIRED BACKGROUND:** [skill-name]"
   - "After mastering this, see [skill-name]"
3. Update router cross-references
4. Ensure bidirectional links (if A references B, should B reference A?)

**Terminology alignment:**

1. Identify terminology inconsistencies across skills
2. Choose canonical terms (most clear/standard)
3. Update all skills to use canonical terms
4. Add glossary to router if needed

**Faction voice adjustment:**

For skills flagged with faction drift:
1. Read FACTIONS.md for faction principles
2. Adjust language/tone to match faction
3. Realign examples with faction philosophy
4. If severe drift: Flag for potential rehoming

**If rehoming recommended:**
- Document which faction skill should move to
- Note in commit message for manual handling
- Don't move skills automatically (requires marketplace changes)

**Metadata synchronization:**

Update `plugin.json`:
1. Description - ensure it matches enhanced pack content
2. Count skills if tool supports it
3. Verify category is appropriate

### 4. Version Management (Impact-Based)

**Assess impact of all changes:**

**Patch bump (x.y.Z) - Low impact:**
- Typos fixed
- Formatting improvements
- Minor clarifications (< 50 words added)
- Small example corrections
- No new skills, no skills removed

**Minor bump (x.Y.0) - Medium impact (DEFAULT):**
- Enhanced guidance (added sections, better examples)
- New skills added
- Improved existing skills significantly
- Better anti-pattern coverage
- Fixed gauntlet failures
- Updated for current best practices

**Major bump (X.0.0) - High impact (RARE, use sparingly):**
- Skills removed entirely
- Structural reorganization
- Philosophy shifts
- Breaking changes to how skills work
- Deprecated major patterns

**Decision logic:**
1. Any new skills added? → Minor minimum
2. Any skills removed? → Consider major
3. Only fixes/clarifications? → Patch
4. Enhanced multiple skills significantly? → Minor
5. Changed pack philosophy? → Major

**Default for maintenance reviews: Minor bump**

**Update version in plugin.json:**
```json
{
  "version": "[new-version]"
}
```

### 5. Git Commit

**Single commit with all changes:**

**Commit message format:**

```
feat(meta): enhance [pack-name] - [one-line summary]

Structure changes:
- Added [count] new skills: [skill-1], [skill-2], ...
- Removed [count] duplicate skills: [skill-1], [skill-2], ...
- Merged [skill-a] + [skill-b] into [skill-merged]
- Updated router to reflect new structure

Content improvements:
- Enhanced [skill-1]: [specific improvements]
- Enhanced [skill-2]: [specific improvements]
- Fixed gauntlet failures in [skill-3]: [issues addressed]
- Added anti-patterns to [skill-4]

Coherence updates:
- Added cross-references between [count] skills
- Aligned terminology across pack
- Adjusted faction voice in [skill-name]
- Updated plugin.json metadata

Version: [old-version] → [new-version] ([patch/minor/major])
Rationale: [reason for version bump type]
```

**Commit command:**

```bash
git add plugins/[pack-name]/
git commit -m "$(cat <<'EOF'
feat(meta): enhance [pack-name] - [summary]

[Full message body as above]
EOF
)"
```

**Do NOT push** - let user decide when to push.

## Execution Principles

**Autonomous within approved scope:**
- Execute all approved changes without asking again
- Follow user's approved plan exactly
- Make editorial decisions within scope
- Ask only if something unexpected blocks progress

**Quality standards:**
- All new skills follow CSO guidelines (name/description format)
- All code examples are complete and appropriate to domain
- All cross-references are accurate
- Faction voice is maintained

**Verification before commit:**
- Verify YAML front matter syntax in all modified skills
- Check that all cross-references point to existing skills
- Ensure router (if exists) references all current skills
- Verify plugin.json has valid JSON syntax

## Output After Completion

Provide comprehensive summary:

```
# Pack Enhancement Complete: [pack-name]

## Version: [old] → [new] ([type])

## Summary Statistics

- Skills added: [count]
- Skills removed: [count]
- Skills enhanced: [count]
- Skills tested and passed: [count]

## Changes by Category

### Structure
[List of structural changes]

### Content
[List of content improvements]

### Coherence
[List of coherence updates]

## Git Commit

Created commit: [commit-hash if available]
Message: [first line of commit]

Ready to push: [Yes]
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Proceeding without approval | Always get user approval before executing |
| Batch changes across passes | Complete one pass fully before next |
| Inconsistent faction voice | Read FACTIONS.md, maintain voice throughout |
| Broken cross-references | Verify all referenced skills exist |
| Invalid YAML | Check syntax before committing |
| Pushing automatically | Let user decide when to push |
| Vague commit messages | Be specific about what changed and why |
| Wrong version bump | Follow impact-based rules, default to minor |

## Anti-Patterns

**❌ Changing scope during execution:**
- Don't add extra improvements not discussed
- Don't skip approved changes because "not needed"
- Stick to approved scope exactly

**❌ Sub-optimal quality:**
- Don't write quick/dirty skills to fill gaps
- Don't copy-paste without adapting to faction
- Don't skip cross-references to save time

**❌ Incomplete commits:**
- Don't commit partial work
- Don't split into multiple commits
- Single commit with all changes

**❌ No verification:**
- Don't assume syntax is correct
- Don't skip cross-reference checking
- Verify before committing

## The Bottom Line

**Execute approved changes autonomously with high quality standards.**

One commit. Proper versioning. Complete summary.

No shortcuts. No scope creep. Professional execution.
