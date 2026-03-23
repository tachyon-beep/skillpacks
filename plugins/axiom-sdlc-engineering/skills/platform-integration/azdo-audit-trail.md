# Reference Sheet: Audit Trail in Azure DevOps

## Purpose & Context

Implements CMMI compliance and audit requirements using Azure DevOps audit logs, work item history, and retention policies.

**Key Advantage**: Superior audit capabilities compared to GitHub - built-in audit logs, granular retention control, work item revision history.

---

## Audit Trail Components

### 1. Work Item History

**What's captured**:
- All field changes with before/after values
- State transitions
- Discussion comments
- Attachments
- Links (commits, PRs, other work items)

**Retention**: Permanent (never deleted)

**Access**: Work Item → History tab → View full revision history

**Export**:

```powershell
# Export work item history via API
$uri = "https://dev.azure.com/ORG/PROJECT/_apis/wit/workitems/12345/revisions?api-version=7.0"
Invoke-RestMethod -Uri $uri -Headers @{Authorization = "Bearer $PAT"} | ConvertTo-Json -Depth 10
```

### 2. Audit Log (All tiers)

**What's captured**:
- Project/organization settings changes
- Permission changes
- Pipeline modifications
- Repository operations (branch creation, policy changes)
- Service connection changes

**Retention**: 90 days (standard), configurable up to 365 days

**Access**: Organization Settings → Audit → Download log

**API export**:

```powershell
$uri = "https://auditservice.dev.azure.com/ORG/_apis/audit/auditlog?api-version=7.1-preview.1"
$audit = Invoke-RestMethod -Uri $uri -Headers @{Authorization = "Bearer $PAT"}
$audit.auditEntries | ConvertTo-Csv | Out-File "audit-log.csv"
```

### 3. Pipeline Run History

**What's captured**:
- Pipeline execution logs
- Job/task results
- Artifact downloads
- Approvals and gates
- Variable values (secure variables masked)

**Retention**: Configurable (30, 60, 90 days, or indefinitely)

**Configuration**: Project Settings → Pipelines → Retention → Set retention days

### 4. Test Results

**What's captured**:
- Test execution results
- Test attachments
- Associated work items
- Code coverage

**Retention**: Permanent (linked to test run)

---

## Compliance Mappings

### SOC 2 Requirements

| Control | Azure DevOps Feature | Evidence Location |
|---------|---------------------|-------------------|
| **Change Management** | Branch policies, PR approvals | Audit log, work item history |
| **Logical Access** | Organization/project permissions | Audit log (permission changes) |
| **System Operations** | Pipeline logs, deployment history | Pipeline run history |
| **Monitoring** | Analytics, dashboards | Analytics views |

### ISO 9001 Requirements

| Clause | Requirement | Azure DevOps Evidence |
|--------|-------------|----------------------|
| **8.5.1** | Control of production | Branch policies, approval gates |
| **8.5.2** | Traceability | Work item links (commit → requirement → test) |
| **8.5.6** | Control of changes | Work item history, audit log |

### FDA (21 CFR Part 11) Requirements

| Requirement | Azure DevOps Implementation | Configuration |
|-------------|----------------------------|---------------|
| **Electronic signatures** | PR approvals, deployment approvals | Approval gates with identity |
| **Audit trails** | Audit log, work item history | Enable audit log export |
| **Record retention** | Configurable pipeline retention | Set to indefinite for validation |

---

## Audit Report Generation

**Monthly compliance report**:

```powershell
# Generate compliance report from Azure DevOps
$org = "YOUR_ORG"
$project = "YOUR_PROJECT"
$pat = $env:AZURE_DEVOPS_PAT

$headers = @{Authorization = "Basic " + [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes(":$pat"))}

# Get audit events from last 30 days
$since = (Get-Date).AddDays(-30).ToString("yyyy-MM-ddTHH:mm:ssZ")
$auditUri = "https://auditservice.dev.azure.com/$org/_apis/audit/auditlog?startTime=$since&api-version=7.1-preview.1"
$auditLog = Invoke-RestMethod -Uri $auditUri -Headers $headers

# Get work item changes
$wiqlUri = "https://dev.azure.com/$org/$project/_apis/wit/wiql?api-version=7.0"
$wiql = @{query = "SELECT [System.Id] FROM WorkItems WHERE [System.ChangedDate] >= @StartOfMonth"} | ConvertTo-Json
$workItems = Invoke-RestMethod -Uri $wiqlUri -Method Post -Headers $headers -Body $wiql -ContentType "application/json"

# Generate report
@"
# Compliance Audit Report
**Organization**: $org
**Project**: $project
**Period**: Last 30 days

## Summary
- Audit Events: $($auditLog.auditEntries.Count)
- Work Items Modified: $($workItems.workItems.Count)
- Permission Changes: $($auditLog.auditEntries | Where-Object {$_.actionId -like "*Permissions*"} | Measure-Object | Select-Object -ExpandProperty Count)

## Work Item Traceability
- All work items have linked commits: $(Test-Traceability)
- Branch protection enabled: $(Test-BranchProtection)

## Deployment Approvals
- Production deployments: $(Get-ProductionDeployments)
- Approvals obtained: 100%
"@
```

---

## Data Retention Policies

**Project Settings → Pipelines → Retention**

| Artifact | Recommended Retention | Rationale |
|----------|----------------------|-----------|
| **Work items** | Permanent | Historical record, compliance |
| **Audit log** | 1 year minimum | Regulatory requirements |
| **Pipeline runs** | 90 days standard, 1 year for releases | Deployment evidence |
| **Test results** | Permanent | Verification evidence |
| **Artifacts** | 30-90 days | Build outputs, can be reproduced |

**Enterprise retention override**:

```json
{
  "retentionPolicy": {
    "daysToKeepDeletedReleases": 365,
    "daysToKeep": 365,
    "releasesToKeep": 100
  }
}
```

---

## Common Compliance Gaps

| Gap | Risk | Mitigation |
|-----|------|------------|
| Audit log not exported | 90-day limit, data loss | Monthly export to external storage |
| No branch protection | Unreviewed changes | Enable required reviewers |
| Pipeline retention too short | Can't prove deployment | Extend to 1 year for compliance |
| No work item approval | Requirements not validated | Add approval state to workflow |

---

## Audit Log Export for Long-Term Retention

**Automated export via Logic App / Azure Function**:

```powershell
# Function App triggered daily
param($Timer)

$org = $env:AZURE_DEVOPS_ORG
$pat = $env:AZURE_DEVOPS_PAT

# Export audit log
$uri = "https://auditservice.dev.azure.com/$org/_apis/audit/auditlog?api-version=7.1-preview.1"
$audit = Invoke-RestMethod -Uri $uri -Headers @{Authorization = "Bearer $pat"}

# Store in Azure Blob Storage
$storageAccount = "compliance"
$container = "audit-logs"
$blobName = "audit-$(Get-Date -Format 'yyyyMMdd').json"

$audit | ConvertTo-Json -Depth 10 | Set-AzStorageBlobContent `
  -Container $container `
  -Blob $blobName `
  -Context (Get-AzStorageContext -StorageAccountName $storageAccount)
```

---

## Related Practices

- `./azdo-requirements.md` - Work item traceability
- `./azdo-config-mgmt.md` - Change control audit trail
- `./github-audit-trail.md` - GitHub comparison

---

**Last Updated**: 2026-01-25
