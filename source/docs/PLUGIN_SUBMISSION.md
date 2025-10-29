# Claude Code Plugin Marketplace Submission

This guide provides step-by-step instructions for submitting skillpacks to the Claude Code plugin marketplace.

---

## Prerequisites Checklist

Before submitting, verify all items are complete:

- [ ] **plugin.json created and validated** - All 18 skills listed with correct paths
- [ ] **PLUGIN_README.md finalized** - Concise marketplace-facing README (<500 words)
- [ ] **All skills tested locally** - Each skill loads correctly in Claude Code
- [ ] **Cross-references verified** - Skills reference each other successfully
- [ ] **Version number set** - Confirmed as 1.0.0 (semantic versioning)
- [ ] **License confirmed** - Apache-2.0 declared in plugin.json and LICENSE file
- [ ] **Author/repository details updated** - No placeholder values remain
- [ ] **Keywords optimized** - plugin.json contains discoverable keywords
- [ ] **Description complete** - Clear, concise description of plugin purpose
- [ ] **Repository URL verified** - GitHub repository link is correct and public

---

## Local Testing Before Submission

Test the plugin thoroughly in your local environment before marketplace submission.

### Step 1: Validate plugin.json

Ensure plugin manifest is valid JSON:

```bash
python3 -m json.tool plugin.json > /dev/null && echo "Valid JSON" || echo "Invalid JSON"
```

Expected output: `Valid JSON`

Verify all 18 skills are listed:

```bash
cat plugin.json | python3 -c "import sys, json; data = json.load(sys.stdin); print(f'Skills listed: {len(data[\"skills\"])}')"
```

Expected output: `Skills listed: 18`

### Step 2: Test Meta-Skills Load

Meta-skills route to appropriate core and extension skills. Test both:

```
I'm using ordis/security-architect/using-security-architect to help me
decide which security skill I need for threat analysis.
```

Verify: Meta-skill loads and can recommend threat-modeling skill.

```
I'm using muna/technical-writer/using-technical-writer to help me
decide which documentation skill I need for API documentation.
```

Verify: Meta-skill loads and can recommend documentation-structure skill.

### Step 3: Test Core Skills (Spot Check)

Test that core skills from each faction load and work:

**Security Architecture Core Skill:**
```
I'm using ordis/security-architect/threat-modeling to analyze this REST API
authentication endpoint for security threats.
```

Verify: Skill loads, STRIDE methodology is available, can analyze threats.

**Technical Writer Core Skill:**
```
I'm using muna/technical-writer/documentation-structure to create an API
reference documentation template.
```

Verify: Skill loads, ADR patterns available, can generate documentation templates.

### Step 4: Test Extension Skills (Spot Check)

Test specialized extension skills to ensure they work:

**Security Architecture Extension:**
```
I'm using ordis/security-architect/classified-systems-security to design
a system that handles classified data following Bell-LaPadula model.
```

Verify: Skill loads, Bell-LaPadula MLS patterns available.

**Technical Writer Extension:**
```
I'm using muna/technical-writer/incident-response-documentation to create
runbooks for our incident response process.
```

Verify: Skill loads, 5-phase response template available.

### Step 5: Test Cross-References Between Skills

Verify that skills can reference each other:

```
I'm using ordis/security-architect/threat-modeling to analyze our API,
and I want to know which security controls to design next.
```

Verify: The skill references `security-controls-design` and that reference works.

```
I'm using ordis/security-architect/security-controls-design and I need to
document my controls with traceability to threats.
```

Verify: The skill references `documenting-threats-and-controls` and works.

### Step 6: Verify File Structure

Ensure plugin package has correct structure:

```bash
# Check plugin.json exists in root
ls -la plugin.json

# Check PLUGIN_README.md exists in root
ls -la PLUGIN_README.md

# Verify all skill paths from plugin.json exist
# Example for one skill:
ls -la ordis/security-architect/threat-modeling/

# Should contain SKILL.md with YAML frontmatter:
# ---
# name: threat-modeling
# description: [description]
# ---
```

---

## Submission Process

### Step 1: Prepare Plugin Package

Create a release-ready package with all required files:

**Required Files in Plugin Root:**
- `plugin.json` - Plugin manifest with all 18 skills
- `PLUGIN_README.md` - Marketplace-facing README
- `LICENSE` - Apache 2.0 license text
- All skill directories with complete SKILL.md files

**Verify Content:**
- `plugin.json` - Valid JSON, 18 skills listed, all fields complete
- `PLUGIN_README.md` - Under 500 words, includes use cases and examples
- `LICENSE` - Complete Apache 2.0 text
- Skills - All paths in plugin.json match actual file locations

### Step 2: Create GitHub Release

Make your repository public and create a GitHub release:

```bash
# Ensure you're on main branch
git checkout main

# Create an annotated tag for version 1.0.0
git tag -a v1.0.0 -m "Release version 1.0.0 - Initial public release"

# Push to GitHub (replace username with your GitHub username)
git push origin main --tags
```

Create release notes on GitHub with:
- **Title:** Skillpacks 1.0.0 - Initial Release
- **Description:** 18 expert-level skills for security architecture and technical writing
- **Assets:** Link to this PLUGIN_SUBMISSION.md documentation

### Step 3: Register with Claude Code Plugin Marketplace

When the Claude Code plugin marketplace becomes available:

1. **Visit the Marketplace Registration Portal**
   - URL: [Claude Code Plugin Marketplace - To be announced]

2. **Create Publisher Account**
   - Provide GitHub username/email
   - Verify email address
   - Accept marketplace terms of service

3. **Create Plugin Listing**
   - **Plugin Name:** `security-architect-technical-writer`
   - **Display Name:** `Security Architect & Technical Writer Skills`
   - **Version:** `1.0.0`
   - **Repository URL:** `https://github.com/[username]/skillpacks`
   - **License:** `Apache-2.0`

### Step 4: Submit Plugin Metadata

Complete marketplace submission form:

- **Short Description** (2-3 lines):
  ```
  Professional security architecture and technical writing skills for Claude Code.
  18 battle-tested skills for threat modeling, compliance, documentation, and more.
  ```

- **Long Description:**
  - Copy first 300-400 words from PLUGIN_README.md
  - Include key capabilities (threat modeling, compliance, documentation)
  - Include target audiences (security professionals, technical writers, developers)

- **Keywords:**
  ```
  security, threat-modeling, documentation, technical-writing, compliance,
  architecture, HIPAA, GDPR, SOC2, ATO, incident-response, zero-trust, defense-in-depth
  ```

- **Category:** `Security & Compliance` or `Documentation`

- **Author/Maintainer:** Your name and email

- **Support Contact:** GitHub Issues link

### Step 5: Upload Plugin Package

Package the plugin for marketplace submission:

1. **Create Plugin Archive:**
   ```bash
   # Create a clean export of your repository
   git clone https://github.com/[username]/skillpacks skillpacks-release
   cd skillpacks-release

   # Create archive (marketplace may specify preferred format)
   zip -r skillpacks-1.0.0.zip . -x "*.git/*" ".github/*"
   ```

2. **Upload to Marketplace**
   - Select plugin archive (ZIP file)
   - Upload PLUGIN_README.md separately if required
   - Confirm all files are included

3. **Verify Plugin Installation**
   - Use marketplace preview feature
   - Test that skills load correctly
   - Confirm cross-references work

### Step 6: Submit for Review

Complete submission and request review:

1. **Review Checklist**
   - [ ] All 18 skills listed in plugin.json
   - [ ] Valid JSON in plugin.json
   - [ ] PLUGIN_README.md concise and complete
   - [ ] All skill file paths correct
   - [ ] YAML frontmatter in all skill files
   - [ ] Cross-references between skills work
   - [ ] Apache 2.0 license included

2. **Submit for Marketplace Review**
   - Click "Submit for Review"
   - Provide any additional notes for reviewers
   - Expected review time: 3-7 business days

3. **Respond to Reviewer Feedback**
   - Monitor email for marketplace reviewer messages
   - Address any questions about skills or metadata
   - If changes needed, update plugin and resubmit
   - Reviewers will verify skills work as described

---

## Post-Submission

### Monitor Marketplace Reviews

After plugin is published:

1. **Track Listing Performance**
   - Monitor install/download metrics on marketplace
   - Check user ratings and reviews
   - Note common feedback themes

2. **Respond to User Reviews**
   - Address any 1-star reviews professionally
   - Answer questions about skill usage
   - Provide support via GitHub Issues

3. **Address User Feedback**
   - **User reports bug:** Create GitHub issue, prioritize fix
   - **User requests feature:** Evaluate for next version (1.1.0)
   - **User asks how-to:** Answer via GitHub Discussions or marketplace
   - **Performance issue:** Investigate and document

### Plan Version Updates

**Version 1.1.0 Timeline** (planned for ~2-3 months post-launch):

**Bug Fixes:**
- Address any critical issues found post-launch
- Update skill content based on user feedback
- Fix documentation clarity issues

**Feature Additions:**
- Consider community-requested extensions (1-2 new skills)
- Incorporate best practices discovered in the field
- Enhance examples based on real-world usage

**Version Update Process:**

1. **Update Skills/Docs in GitHub**
   - Make all changes to skills in GitHub repository
   - Test thoroughly with RED-GREEN-REFACTOR methodology
   - Create feature branch: `git checkout -b v1.1.0-updates`

2. **Update plugin.json**
   - Increment version: `1.0.0` → `1.1.0`
   - Add any new skills to skills array
   - Update description if needed

3. **Update PLUGIN_README.md**
   - Reflect new capabilities if added
   - Update version number
   - Keep under 500 words

4. **Test Updated Plugin Locally**
   - Verify all skills load correctly
   - Test cross-references work
   - Validate JSON

5. **Create Release Commit**
   ```bash
   git add plugin.json PLUGIN_README.md
   git commit -m "Release v1.1.0 - Bug fixes and community feedback

   Changes:
   - [Fix #123: skill issue description]
   - [Enhance: skill description improvement]

   New Version: 1.1.0"
   ```

6. **Submit Updated Plugin to Marketplace**
   - Upload new version (1.1.0)
   - Include release notes in submission
   - Provide context for changes made

7. **Announce Update**
   - Create GitHub release notes
   - Post announcement in Discussions
   - Include what's new and why

### Long-Term Maintenance

**Ongoing Responsibilities:**
- Monitor marketplace for issues/ratings
- Review and merge community contributions
- Keep skills accurate and relevant
- Update documentation as needed
- Respond to GitHub issues within 1-2 weeks

**Annual Review:**
- Assess skill effectiveness with real-world usage
- Consider major version update (2.0.0) if significant changes
- Plan for new skills based on community feedback
- Update plugin dependencies if needed

---

## Troubleshooting Common Issues

### Issue: plugin.json validation fails

**Symptoms:** Error when validating JSON

**Solution:**
```bash
# Check for JSON syntax errors
python3 -m json.tool plugin.json

# Common issues:
# - Trailing comma in last array element
# - Missing quotes around string values
# - Tab characters instead of spaces
```

**Fix:** Use a JSON linter to identify and fix syntax errors.

### Issue: Skill doesn't load from marketplace

**Symptoms:** Plugin loads but skill returns "not found" error

**Solution:**
1. Verify path in plugin.json matches actual file location
   ```bash
   # Check path in plugin.json
   cat plugin.json | grep "ordis/security-architect/threat-modeling"

   # Verify file exists
   ls -la ordis/security-architect/threat-modeling/SKILL.md
   ```

2. Confirm file paths use forward slashes (not backslashes)
3. Ensure no extra whitespace in paths

**Fix:** Update plugin.json with correct path, retest, resubmit.

### Issue: Cross-reference between skills doesn't work

**Symptoms:** Skill A references Skill B but reference fails to load

**Solution:**
1. Verify reference uses correct skill path format
   ```markdown
   **Use WITH this skill:**
   - `ordis/security-architect/security-controls-design`
   ```

2. Test reference manually:
   ```
   I'm using ordis/security-architect/threat-modeling
   Can you also explain ordis/security-architect/security-controls-design?
   ```

3. Check that referenced skill exists and loads independently

**Fix:** Correct reference path, test, resubmit plugin.

### Issue: YAML frontmatter not parsed correctly

**Symptoms:** Skill loads but metadata (name, description) missing

**Solution:**
1. Verify frontmatter structure:
   ```yaml
   ---
   name: skill-name
   description: Skill description here
   ---
   ```

2. Check formatting:
   - Frontmatter must start at first line (no blank line before)
   - Must have `---` on separate lines
   - Use spaces, not tabs in YAML
   - Name and description are required fields

3. Test file format:
   ```bash
   head -5 ordis/security-architect/threat-modeling/SKILL.md
   # Should show frontmatter with ---
   ```

**Fix:** Correct YAML frontmatter format, retest, resubmit.

### Issue: Marketplace submission rejected

**Symptoms:** Submission rejected with feedback

**Solutions by Issue Type:**

**"Skills not loading"**
- Test each skill locally first
- Verify all paths in plugin.json are correct
- Ensure SKILL.md files exist with proper frontmatter

**"Description insufficient"**
- Provide more detail about what skills do
- Include use cases and target audience
- Reference key capabilities and frameworks

**"Missing documentation"**
- Ensure PLUGIN_README.md is complete
- Include examples of skill usage
- Provide clear installation instructions

**"License issue"**
- Verify Apache 2.0 is in LICENSE file
- Confirm version number matches marketplace submission
- Ensure copyright year is current

---

## Resources

### Official Documentation
- **Claude Code Documentation:** [Claude Code Docs - To be announced]
- **Plugin Development Guide:** [Plugin Dev Docs - To be announced]
- **Marketplace Requirements:** [Marketplace Docs - To be announced]

### Project Documentation
- **GitHub Repository:** https://github.com/tachyon-beep/skillpacks
- **Main README:** https://github.com/tachyon-beep/skillpacks/blob/main/README.md
- **Contributing Guide:** https://github.com/tachyon-beep/skillpacks/blob/main/CONTRIBUTING.md
- **Issue Tracker:** https://github.com/tachyon-beep/skillpacks/issues
- **Discussions:** https://github.com/tachyon-beep/skillpacks/discussions

### Related Wave 2 Files
- **plugin.json** - Plugin manifest with all 18 skills
- **PLUGIN_README.md** - Marketplace-facing README for users
- **This guide** - PLUGIN_SUBMISSION.md (step-by-step submission process)

---

## Version History

| Version | Date | Status | Notes |
|---------|------|--------|-------|
| 1.0.0 | Oct 2025 | Ready for Submission | Initial release with 18 skills |
| 1.1.0 | Q1 2026 | Planned | Bug fixes and community feedback |
| 2.0.0 | Q3 2026+ | Future | Major enhancements based on feedback |

---

## Checklist: Ready for Submission?

Use this final checklist before submitting to marketplace:

**Plugin Files:**
- [ ] plugin.json exists in root directory
- [ ] plugin.json is valid JSON (tested with python3 -m json.tool)
- [ ] plugin.json contains 18 skills
- [ ] PLUGIN_README.md exists in root directory
- [ ] PLUGIN_README.md is under 500 words
- [ ] LICENSE file contains Apache 2.0 text
- [ ] All skills referenced in plugin.json have corresponding SKILL.md files

**Skill Content:**
- [ ] All SKILL.md files start with YAML frontmatter (---name---description---)
- [ ] All skills have name and description fields
- [ ] All skill paths in plugin.json match actual file locations
- [ ] Cross-references between skills use correct format (`faction/pack/skill-name`)
- [ ] All examples use obviously fake credentials (no real secrets)
- [ ] All examples use example.com domains (RFC 2606)

**Metadata:**
- [ ] Author name and email are correct (no placeholders)
- [ ] Repository URL points to correct GitHub repository
- [ ] Version number is 1.0.0
- [ ] License is Apache-2.0
- [ ] Keywords are relevant and discoverable
- [ ] Description clearly explains what the plugin does

**Testing:**
- [ ] All meta-skills load and can route to appropriate skills
- [ ] All core skills load and demonstrate their core capability
- [ ] Sample of extension skills tested and work
- [ ] Cross-references between skills verified working
- [ ] Plugin tested locally before submission

**Legal/Administrative:**
- [ ] Repository is public on GitHub
- [ ] LICENSE file has current copyright year
- [ ] CONTRIBUTING.md exists for community contributions
- [ ] CODE_OF_CONDUCT.md exists (Contributor Covenant)
- [ ] You have authority to submit under Apache 2.0 license

**Go/No-Go Decision:**

If **ALL** boxes are checked: **READY FOR SUBMISSION**

If any boxes are unchecked:
1. Complete the unchecked items
2. Re-test affected components
3. Return to this checklist
4. Proceed only when all items complete

---

## Next Steps

1. **Complete the checklist above** ✓
2. **Test local installation** ✓
3. **Register with Claude Code Marketplace** (when available)
4. **Submit plugin for review**
5. **Monitor marketplace for review feedback**
6. **Address reviewer comments** (if any)
7. **Plugin published and available to Claude Code users!**
8. **Plan version 1.1.0 updates** (ongoing)
9. **Monitor user feedback and support** (ongoing)

---

## Questions or Issues?

- **GitHub Issues:** [Report issues with the plugin](https://github.com/tachyon-beep/skillpacks/issues)
- **GitHub Discussions:** [Ask questions and discuss](https://github.com/tachyon-beep/skillpacks/discussions)
- **Contributing:** [Learn how to contribute](https://github.com/tachyon-beep/skillpacks/blob/main/CONTRIBUTING.md)
