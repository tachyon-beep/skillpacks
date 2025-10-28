# Quick Checklist - Keep This Open While Working

## Starting New Skill ✨

```
[ ] Use TodoWrite to create RED-GREEN-REFACTOR checklist for this skill
[ ] Announce: "I'm using the writing-skills skill to create [skill-name]"
[ ] Create skill directory: security-architect/[name]/ or technical-writer/[name]/
```

---

## RED Phase: Watch It Fail 🔴

```
[ ] Create test scenario (pressure for discipline, application for technique)
[ ] Dispatch subagent WITHOUT skill loaded
[ ] Document baseline behavior (verbatim rationalizations, what they miss)
[ ] Identify failure patterns
[ ] DO NOT write skill yet (resist the urge!)
```

---

## GREEN Phase: Write Minimal Skill 🟢

```
[ ] NOW write SKILL.md addressing RED failures
[ ] YAML frontmatter: name (hyphens only), description (starts with "Use when...")
[ ] Overview: Core principle (1-2 sentences)
[ ] When to use: Symptoms and triggers
[ ] Core content: Address baseline failures
[ ] One excellent example (concrete, runnable)
[ ] Common mistakes: From RED phase
[ ] Run scenarios WITH skill - verify compliance
```

---

## REFACTOR Phase: Close Loopholes 🔨

```
[ ] Add pressure (time + sunk cost + authority if discipline skill)
[ ] Find new rationalizations
[ ] Build rationalization table
[ ] Add explicit counters
[ ] Re-test until bulletproof
```

---

## Commit & Move On 💾

```
[ ] git add [skill-directory]
[ ] git commit with clear message
[ ] Update README.md status
[ ] Mark skill complete in TodoWrite
[ ] Move to next skill
```

---

## The Iron Law ⚖️

```
█████████████████████████████████████████████
█  NO SKILL WITHOUT FAILING TEST FIRST      █
█  If you wrote it before testing: DELETE   █
█████████████████████████████████████████████
```

---

## Skills to USE 🛠️

```
ALWAYS announce and load:
✓ superpowers:using-superpowers
✓ superpowers:writing-skills
✓ superpowers:testing-skills-with-subagents
```

---

## Red Flags 🚩

Stop and reconsider if you're thinking:

- ❌ "This is obviously clear" → TEST IT
- ❌ "I'll batch 3 skills" → ONE AT A TIME
- ❌ "Academic review is enough" → PRESSURE TEST
- ❌ "Too simple to need testing" → EVERYTHING NEEDS TESTING
- ❌ "I'll test if issues emerge" → TEST FIRST ALWAYS

---

## Phase 1 Skills (Next 4)

```
1. [ ] security-architect/using-security-architect (meta, 2-3h)
2. [ ] security-architect/threat-modeling (core, 3-4h)
3. [ ] technical-writer/using-technical-writer (meta, 2-3h)
4. [ ] technical-writer/documentation-structure (core, 3-4h)

Phase 1 Done When:
[ ] All 4 pass RED-GREEN-REFACTOR
[ ] Meta-skills route correctly
[ ] Cross-references work
[ ] Personal use validated
[ ] git tag v0.1-phase1
```

---

## Emergency Help 🆘

**Stuck?** Read in order:
1. This checklist (quick reminders)
2. GETTING_STARTED.md (detailed workflow)
3. DESIGN.md section for current skill (full context)

**Lost context?**
- README.md → GETTING_STARTED.md → DESIGN.md

**Forgot methodology?**
- Load superpowers:writing-skills again

---

## Success Rhythm 🎵

```
RED (fail) → GREEN (fix) → REFACTOR (harden) → COMMIT (ship)
     ↓           ↓              ↓                  ↓
  Test first  Address it   Close loopholes    One at a time
```

---

## Print This Page 🖨️

Keep it visible while working. Check boxes as you go. Trust the process.

**You've got this! Ship one skill at a time. Follow the workflow. It works.**
