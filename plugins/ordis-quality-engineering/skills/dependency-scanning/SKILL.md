---
name: dependency-scanning
description: Use when integrating SCA tools (Snyk, Dependabot, OWASP Dependency-Check), automating vulnerability management, handling license compliance, setting up automated dependency updates, or managing security advisories - provides tool selection, PR automation workflows, and false positive management
---

# Dependency Scanning

## Overview

**Core principle:** Third-party dependencies introduce security vulnerabilities and license risks. Automate scanning to catch them early.

**Rule:** Block merges on critical/high vulnerabilities in direct dependencies. Monitor and plan fixes for transitive dependencies.

## Why Dependency Scanning Matters

**Security vulnerabilities:**
- 80% of codebases contain at least one vulnerable dependency
- Log4Shell (CVE-2021-44228) affected millions of applications
- Attackers actively scan GitHub for known vulnerabilities

**License compliance:**
- GPL dependencies in proprietary software = legal risk
- Some licenses require source code disclosure
- Incompatible license combinations

---

## Tool Selection

| Tool | Use Case | Cost | Best For |
|------|----------|------|----------|
| **Dependabot** | Automated PRs for updates | Free (GitHub) | GitHub projects, basic scanning |
| **Snyk** | Comprehensive security + license scanning | Free tier, paid plans | Production apps, detailed remediation |
| **OWASP Dependency-Check** | Security-focused, self-hosted | Free | Privacy-sensitive, custom workflows |
| **npm audit** | JavaScript quick scan | Free | Quick local checks |
| **pip-audit** | Python quick scan | Free | Quick local checks |
| **bundler-audit** | Ruby quick scan | Free | Quick local checks |

**Recommended setup:**
- **GitHub repos:** Dependabot (automated) + Snyk (security focus)
- **Self-hosted:** OWASP Dependency-Check
- **Quick local checks:** npm audit / pip-audit

---

## Dependabot Configuration

### Enable Dependabot (GitHub)

``yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 5
    labels:
      - "dependencies"
    reviewers:
      - "security-team"

  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    target-branch: "develop"
```

**What Dependabot does:**
- Scans dependencies weekly
- Creates PRs for vulnerabilities
- Updates to safe versions
- Provides CVE details

---

## Snyk Integration

### Installation

```bash
npm install -g snyk
snyk auth  # Authenticate with Snyk account
```

---

### Scan Local Project

```bash
# Test for vulnerabilities
snyk test

# Monitor project (continuous scanning)
snyk monitor
```

---

### CI/CD Integration

```yaml
# .github/workflows/snyk.yml
name: Snyk Security Scan

on: [pull_request, push]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Snyk
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high  # Fail on high+ severity
```

**Severity thresholds:**
- **Critical:** Block merge immediately
- **High:** Block merge, fix within 7 days
- **Medium:** Create issue, fix within 30 days
- **Low:** Monitor, fix opportunistically

---

## OWASP Dependency-Check

### Installation

```bash
# Download latest release
wget https://github.com/jeremylong/DependencyCheck/releases/download/v8.0.0/dependency-check-8.0.0-release.zip
unzip dependency-check-8.0.0-release.zip
```

---

### Run Scan

```bash
# Scan project
./dependency-check/bin/dependency-check.sh \
  --scan ./src \
  --format HTML \
  --out ./reports \
  --suppression ./dependency-check-suppressions.xml
```

---

### Suppression File (False Positives)

```xml
<!-- dependency-check-suppressions.xml -->
<?xml version="1.0" encoding="UTF-8"?>
<suppressions xmlns="https://jeremylong.github.io/DependencyCheck/dependency-suppression.1.3.xsd">
    <suppress>
        <notes>False positive - CVE applies to server mode only, we use client mode</notes>
        <cve>CVE-2021-12345</cve>
    </suppress>
</suppressions>
```

---

## License Compliance

### Checking Licenses (npm)

```bash
# List all licenses
npx license-checker

# Filter incompatible licenses
npx license-checker --onlyAllow 'MIT;Apache-2.0;BSD-3-Clause'
```

---

### Blocking Incompatible Licenses

```json
// package.json
{
  "scripts": {
    "license-check": "license-checker --onlyAllow 'MIT;Apache-2.0;BSD-3-Clause;ISC' --production"
  }
}
```

```yaml
# CI: Fail if incompatible licenses detected
- name: Check licenses
  run: npm run license-check
```

**Common license risks:**
- **GPL/AGPL:** Requires source code disclosure
- **SSPL:** Restrictive for SaaS
- **Proprietary:** May prohibit commercial use

---

## Automated Dependency Updates

### Auto-Merge Strategy

**Safe to auto-merge:**
- Patch versions (1.2.3 → 1.2.4)
- No breaking changes
- Passing all tests

```yaml
# .github/workflows/auto-merge-dependabot.yml
name: Auto-merge Dependabot PRs

on: pull_request

jobs:
  auto-merge:
    runs-on: ubuntu-latest
    if: github.actor == 'dependabot[bot]'
    steps:
      - name: Check if patch update
        id: check
        run: |
          # Only auto-merge patch/minor, not major
          if [[ "${{ github.event.pull_request.title }}" =~ ^Bump.*from.*\.[0-9]+$ ]]; then
            echo "auto_merge=true" >> $GITHUB_OUTPUT
          fi

      - name: Enable auto-merge
        if: steps.check.outputs.auto_merge == 'true'
        run: gh pr merge --auto --squash "$PR_URL"
        env:
          PR_URL: ${{ github.event.pull_request.html_url }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## Vulnerability Remediation Workflow

### 1. Triage (Within 24 hours)

**For each vulnerability:**
- **Assess severity:** Critical → immediate, High → 7 days, Medium → 30 days
- **Check exploitability:** Is it reachable in our code?
- **Verify patch availability:** Is there a fixed version?

---

### 2. Remediation Options

| Option | When to Use | Example |
|--------|-------------|---------|
| **Update dependency** | Patch available | `npm update lodash` |
| **Update lockfile only** | Transitive dependency | `npm audit fix` |
| **Replace dependency** | No patch, actively exploited | Replace `request` with `axios` |
| **Apply workaround** | No patch, low risk | Disable vulnerable feature |
| **Accept risk** | False positive, not exploitable | Document in suppression file |

---

### 3. Verification

```bash
# After fix, verify vulnerability is resolved
npm audit
snyk test

# Run full test suite
npm test
```

---

## Anti-Patterns Catalog

### ❌ Ignoring Transitive Dependencies

**Symptom:** "We don't use that library directly, so it's fine"

**Why bad:** Transitive dependencies are still in your app

```
Your App
  └─ express@4.18.0
      └─ body-parser@1.19.0
          └─ qs@6.7.0 (vulnerable!)
```

**Fix:** Update parent dependency or override version

```json
// package.json - force safe version
{
  "overrides": {
    "qs": "^6.11.0"
  }
}
```

---

### ❌ Auto-Merging All Updates

**Symptom:** Dependabot PRs merged without review

**Why bad:**
- Major versions can break functionality
- Updates may introduce new bugs
- No verification tests run

**Fix:** Auto-merge only patch versions, review major/minor

---

### ❌ Suppressing Without Investigation

**Symptom:** Marking all vulnerabilities as false positives

```xml
<!-- ❌ BAD: No justification -->
<suppress>
    <cve>CVE-2021-12345</cve>
</suppress>
```

**Fix:** Document WHY it's suppressed

```xml
<!-- ✅ GOOD: Clear justification -->
<suppress>
    <notes>
        False positive: CVE applies to XML parsing feature.
        We only use JSON parsing (verified in code review).
        Tracking issue: #1234
    </notes>
    <cve>CVE-2021-12345</cve>
</suppress>
```

---

### ❌ No SLA for Fixes

**Symptom:** Vulnerabilities sit unfixed for months

**Fix:** Define SLAs by severity

**Example SLA:**
- **Critical:** Fix within 24 hours
- **High:** Fix within 7 days
- **Medium:** Fix within 30 days
- **Low:** Fix within 90 days or next release

---

## Monitoring & Alerting

### Slack Notifications

```yaml
# .github/workflows/security-alerts.yml
name: Security Alerts

on:
  schedule:
    - cron: '0 9 * * *'  # Daily at 9 AM

jobs:
  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run Snyk
        id: snyk
        run: |
          snyk test --json > snyk-results.json || true

      - name: Send Slack alert
        if: steps.snyk.outcome == 'failure'
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "🚨 Security vulnerabilities detected!",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Critical vulnerabilities found in dependencies*\nView details: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

---

## Bottom Line

**Automate dependency scanning to catch vulnerabilities and license issues early. Block merges on critical issues, monitor and plan fixes for others.**

**Setup:**
- Enable Dependabot (automated PRs)
- Add Snyk or OWASP Dependency-Check (security scanning)
- Check licenses (license-checker)
- Define SLAs (Critical: 24h, High: 7d, Medium: 30d)

**Remediation:**
- Update dependencies to patched versions
- Override transitive dependencies if needed
- Document suppressions with justification
- Verify fixes with tests

**If you're not scanning dependencies, you're shipping known vulnerabilities. Automate it in CI/CD.**
