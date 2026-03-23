# Reference Sheet: Requirements Management in Azure DevOps

## Purpose & Context

Implements CMMI **REQM** and **RD** using Azure DevOps Work Items, Queries, and Backlogs.

**Key Advantage over GitHub**: Built-in hierarchical work items (Epic → Feature → User Story), rich custom fields, advanced queries.

---

## Work Item Type Hierarchy

```
Epic (Business Initiative)
  ↓
Feature (Capability)
  ↓
User Story (Requirement)
  ↓
Task (Implementation)
```

**Traceability**: Parent-child links automatically maintained

---

## Custom Fields for CMMI REQM

**Settings → Process → [Process] → Work Item Types → User Story → Add Field**

| Field Name | Type | Purpose | CMMI Practice |
|------------|------|---------|---------------|
| Requirement ID | Text | Unique identifier (REQ-YYYY-NNN) | REQM SP 1.1 |
| Verification Method | Dropdown | Unit Test, Integration Test, Review | REQM SP 1.4 |
| Source | Text | Stakeholder or regulation | RD SP 1.1 |
| Risk Level | Dropdown | Critical, High, Medium, Low | RSKM integration |
| Baseline | Text | Milestone or release version | CM integration |
| Approval Status | Dropdown | Draft, Under Review, Approved | REQM SP 1.2 |

---

## Bidirectional Traceability

### Requirements → Code

**Link work items to commits**:

```bash
git commit -m "Implement OAuth login

Related work items: #12345"
```

Azure DevOps automatically creates link between commit and work item #12345.

### Requirements → Tests

**Link test cases to requirements**:

1. Create Test Plan
2. Add Test Suites
3. Link Test Cases to User Story work items
4. Execute tests, results automatically linked

**Query for test coverage**:

```
SELECT [System.Id], [System.Title], [System.State]
FROM WorkItemLinks
WHERE ([Source].[Work Item Type] = 'User Story')
  AND ([Target].[Work Item Type] = 'Test Case')
  AND ([System.Links.LinkType] = 'Tests')
MODE (MustContain)
```

---

## Requirement Approval Workflow

**Custom workflow states**:

1. **New** → Requirements created
2. **Under Review** → Stakeholder review in progress
3. **Approved** → Ready for implementation
4. **Committed** → Included in sprint backlog
5. **Done** → Implementation verified
6. **Closed** → Deployment complete

**State transition rules** (enforce workflow):

- New → Under Review: No restrictions
- Under Review → Approved: Requires review comment with approval
- Approved → Committed: Only during sprint planning
- Committed → Done: All linked test cases must pass
- Done → Closed: Deployment to production confirmed

---

## Automated Traceability Reports

**Azure DevOps Analytics**:

Dashboard widget → Work Item Query → Custom Query:

```
SELECT [ID], [Title], [State], [Assigned To]
FROM WorkItems
WHERE [Work Item Type] = 'User Story'
  AND [State] != 'Closed'
  AND [System.Links.LinkCount] = 0  -- No linked PRs/commits
```

**PowerBI Integration**:

```powerquery
// Connect to Azure DevOps Analytics
let
    Source = AzureDevOps.Analytics("https://analytics.dev.azure.com/YOUR_ORG"),
    WorkItems = Source{[Name="WorkItems"]}[Data],
    Requirements = Table.SelectRows(WorkItems, each [WorkItemType] = "User Story"),
    WithLinks = Table.ExpandRecordColumn(Requirements, "Links", {"Count"})
in
    WithLinks
```

---

## Level 2/3/4 Scaling

**Level 2**:
- Use built-in User Story work items
- Manual traceability via linked work items
- Basic queries for reporting

**Level 3**:
- Custom fields for CMMI compliance
- Automated workflow state transitions
- Required approvals before state changes
- Analytics dashboards for traceability coverage

**Level 4**:
- Requirements volatility metrics (changes per requirement per week)
- Defect density by requirement
- Lead time from requirement to deployment
- Predictive models (PowerBI)

---

## Common Anti-Patterns

| Anti-Pattern | Better Approach |
|--------------|-----------------|
| Flat work item structure | Use Epic → Feature → Story hierarchy |
| No custom fields | Add CMMI-specific fields (Requirement ID, Approval Status) |
| Manual traceability | Auto-link commits via commit messages |
| No test plans | Create test plans linked to requirements |

---

## Related Practices

- `./azdo-quality-gates.md` - Test Plans integration
- `./azdo-config-mgmt.md` - Work item baseline management
- `./github-requirements.md` - GitHub comparison

---

**Last Updated**: 2026-01-25
