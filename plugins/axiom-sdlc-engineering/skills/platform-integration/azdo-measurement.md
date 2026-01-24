# Reference Sheet: Measurement in Azure DevOps

## Purpose & Context

Implements CMMI **MA** using Azure DevOps Analytics, Dashboards, and PowerBI integration.

**Key Advantage**: Built-in analytics engine, rich dashboards, PowerBI direct connect.

---

## Azure DevOps Analytics

**Access**: Analytics → Analytics views

**Pre-built views**:
- Work Items: Current state, historical trends
- Pipelines: Build/release success rates, duration
- Test Results: Pass rate, duration, flaky tests

**OData API endpoint**:
```
https://analytics.dev.azure.com/{organization}/{project}/_odata/v3.0-preview/
```

---

## DORA Metrics in Azure DevOps

### 1. Deployment Frequency

**Dashboard widget**: Deployment status

**Custom query** (via API):

```odata
https://analytics.dev.azure.com/ORG/PROJECT/_odata/v3.0-preview/Deployments
?$filter=Environment eq 'Production'
&$select=CompletedDate
&$orderby=CompletedDate desc
```

Calculate deployments per day/week.

### 2. Lead Time for Changes

**Track commit to deployment**:

Query: Work Items → Deployment time

```odata
WorkItems
?$expand=Links($filter=LinkTypeName eq 'Commit')
&$select=CreatedDate,ClosedDate
```

Lead time = ClosedDate - CreatedDate

### 3. Change Failure Rate

**Track failed deployments**:

```odata
Deployments
?$filter=DeploymentStatus eq 'Failed'
&$select=CompletedDate,Environment
```

Calculate: Failed / Total deployments × 100%

### 4. MTTR (Mean Time to Recovery)

**Track incidents**:

```odata
WorkItems
?$filter=WorkItemType eq 'Bug' and Severity eq 'Critical'
&$select=CreatedDate,ClosedDate
```

MTTR = Average(ClosedDate - CreatedDate)

---

## Dashboard Creation

**Dashboards → New Dashboard → Add widgets**

**Recommended widgets for CMMI Level 3**:

1. **Velocity** - Story points completed per sprint
2. **Burndown** - Sprint progress tracking
3. **Code Coverage** - Trend over time
4. **Build Success Rate** - Last 30 builds
5. **Test Pass Rate** - By test suite
6. **Deployment Frequency** - Releases per week
7. **Lead Time** - Commit to deployment

**Widget configuration example** (Velocity):

```json
{
  "name": "Velocity",
  "position": {"row": 1, "column": 1},
  "size": {"rowSpan": 2, "columnSpan": 2},
  "settings": {
    "iterations": 6,
    "teamId": "TEAM_ID"
  }
}
```

---

## PowerBI Integration

**Connect to Azure DevOps Analytics**:

1. PowerBI Desktop → Get Data → Azure DevOps (Boards only)
2. Enter organization URL
3. Select project and entity (WorkItems, Pipelines, etc.)
4. Transform data in Power Query

**Example PowerBI query**:

```powerquery
let
    Source = AzureDevOps.Analytics("https://analytics.dev.azure.com/YOUR_ORG"),
    WorkItems = Source{[Name="WorkItems"]}[Data],
    FilteredRows = Table.SelectRows(WorkItems, each [WorkItemType] = "Bug"),
    GroupedRows = Table.Group(FilteredRows, {"CreatedDate"}, {{"Count", each Table.RowCount(_), Int64.Type}})
in
    GroupedRows
```

**Publish to PowerBI Service** for shared dashboards.

---

## Statistical Baselines (Level 4)

**Calculate organizational baselines**:

```powerquery
// Lead time baseline from last 6 months
let
    LeadTimes = {...},  // Array of lead times in hours
    Mean = List.Average(LeadTimes),
    StdDev = List.StandardDeviation(LeadTimes),
    UpperControlLimit = Mean + (2 * StdDev),
    LowerControlLimit = Mean - (2 * StdDev)
in
    [Mean = Mean, UCL = UpperControlLimit, LCL = LowerControlLimit]
```

**Create control chart** in PowerBI with UCL/LCL lines.

---

## Automated Metrics Collection

**Pipeline task** to collect metrics:

```yaml
- task: PowerShell@2
  inputs:
    targetType: 'inline'
    script: |
      # Query Analytics API
      $uri = "https://analytics.dev.azure.com/$(System.TeamProject)/_odata/v3.0-preview/WorkItems"
      $result = Invoke-RestMethod -Uri $uri -Headers @{Authorization = "Bearer $(System.AccessToken)"}
      
      # Store metrics
      $metrics = @{
        timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
        workItemCount = $result.value.Count
      }
      
      $metrics | ConvertTo-Json | Out-File "metrics.json"
```

---

## Common Anti-Patterns

| Anti-Pattern | Better Approach |
|--------------|-----------------|
| Not using Analytics | Leverage built-in Analytics instead of manual queries |
| Static dashboards | Auto-refresh dashboards with real-time data |
| No baselines | Establish mean + control limits (Level 3/4) |
| PowerBI only for reporting | Use for predictive analytics, forecasting |

---

## Related Practices

- `../quantitative-management/SKILL.md` - MA process definitions, GQM
- `./azdo-quality-gates.md` - Pipeline metrics
- `./github-measurement.md` - GitHub comparison

---

**Last Updated**: 2026-01-25
