# Reference Sheet: Configuration Management in Azure DevOps

## Purpose & Context

Implements CMMI **CM** using Azure Repos branch policies, required reviewers, and work item integration.

**Key Advantage**: Tighter integration with work items, policy-based merge enforcement.

---

## Branch Policies (Level 3 Requirements)

**Repos → Branches → main → Branch policies**

```yaml
Require reviewers:
  Minimum: 2 reviewers
  Allow requestors to approve: No
  Reset votes when source branch updated: Yes
  
Build validation:
  Build pipeline: CI-Pipeline
  Trigger: Automatic
  Policy requirement: Required
  Build expiration: Immediately when main is updated
  
Status checks:
  - Code coverage >= 80%
  - Security scan passed
  
Work item linking:
  Require work items: Yes
  Check for linked work items: Required
  
Comment resolution:
  Require all comments resolved: Yes
```

**As code** (ARM template):

```json
{
  "minimumApproverCount": 2,
  "creatorVoteCounts": false,
  "resetOnSourcePush": true,
  "requiredReviewers": [],
  "buildValidation": {
    "buildDefinitionId": 12,
    "validDuration": 0
  },
  "statusChecks": [
    {
      "name": "coverage-check",
      "genre": "quality",
      "isRequired": true
    }
  ]
}
```

---

## Work Item Linking Enforcement

**Policy**: "Check for linked work items" = Required

**Link work item in PR**:

1. Create PR
2. Click "Add link" → "Existing item"
3. Select work item (User Story, Bug, Task)
4. PR cannot complete until work item linked

**In commit message**:

```bash
git commit -m "Fix authentication bug

Related work items: #12345"
```

---

## Merge Strategies

**Repos → Settings → Repository policies**

| Strategy | Use Case | Traceability Impact |
|----------|----------|---------------------|
| **Squash merge** (recommended) | Clean history | One commit per work item |
| **Merge commit** | Preserve full history | Full PR history visible |
| **Rebase and fast-forward** | Linear history | Loses PR merge point |
| **Semi-linear merge** | Balance | Merge commit + linear first-parent |

**Configuration**: Enable "Squash merge" only for Level 3

---

## Release Baseline Management

**Create baseline (Git tag)**:

```bash
git tag -a v1.2.0 -m "Release 1.2.0 baseline"
git push origin v1.2.0
```

**Lock baseline** (prevent branch deletion):

Repos → Branches → main → Security → Deny "Force push" and "Delete branch"

**Associate work items with release**:

Pipelines → Releases → v1.2.0 → Work items (automatically linked via commits)

---

## Emergency Hotfix Workflow

**Azure DevOps hotfix pattern**:

1. Create hotfix branch from release tag
2. Expedited review (1 reviewer, bypass status checks with justification)
3. Deploy via Release pipeline
4. **Post-facto review** documented in work item
5. Cherry-pick to main

**Override branch policy** (requires permissions):

- Click "Override branch policies"
- Provide reason: "Hotfix for critical production issue #12345"
- Requires "Override branch policies" permission

---

## Audit Trail

**Work item revisions**:

Work Item → History tab → View all revisions with field changes

**Branch policy bypasses**:

Repos → Branches → main → Policy → History of overrides

---

## Related Practices

- `./azdo-requirements.md` - Work item integration
- `./azdo-quality-gates.md` - Build validation policies
- `./github-config-mgmt.md` - GitHub comparison

---

**Last Updated**: 2026-01-25
