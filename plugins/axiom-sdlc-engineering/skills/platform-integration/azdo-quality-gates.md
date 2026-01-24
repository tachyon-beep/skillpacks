# Reference Sheet: Quality Gates in Azure DevOps

## Purpose & Context

Implements CMMI **VER**, **VAL**, **PI** using Azure Pipelines multi-stage YAML, gates, and Test Plans.

**Key Advantage**: Built-in test management, approval gates, environment protection.

---

## Multi-Stage Pipeline (Level 3)

**File**: `azure-pipelines.yml`

```yaml
trigger:
  branches:
    include:
      - main

stages:
  - stage: Build
    jobs:
      - job: BuildAndTest
        pool:
          vmImage: 'ubuntu-latest'
        steps:
          - task: UsePythonVersion@0
            inputs:
              versionSpec: '3.10'
          
          - script: |
              pip install -r requirements.txt
              pytest tests/unit/ --cov=src --cov-report=xml
            displayName: 'Unit Tests'
          
          - task: PublishCodeCoverageResults@1
            inputs:
              codeCoverageTool: 'Cobertura'
              summaryFileLocation: '$(System.DefaultWorkingDirectory)/coverage.xml'
              failIfCoverageEmpty: true
  
  - stage: IntegrationTest
    dependsOn: Build
    condition: succeeded()
    jobs:
      - job: IntegrationTests
        steps:
          - script: pytest tests/integration/
            displayName: 'Integration Tests'
  
  - stage: DeployStaging
    dependsOn: IntegrationTest
    condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
    jobs:
      - deployment: DeployToStaging
        environment: Staging
        strategy:
          runOnce:
            deploy:
              steps:
                - script: echo "Deploying to staging..."
  
  - stage: DeployProduction
    dependsOn: DeployStaging
    condition: succeeded()
    jobs:
      - deployment: DeployToProduction
        environment: Production
        strategy:
          runOnce:
            deploy:
              steps:
                - script: echo "Deploying to production..."
```

---

## Environment Protection Rules

**Pipelines → Environments → Production → Approvals and checks**

**Approvals**:
- Approvers: 2 people from "Release Approvers" group
- Minimum number: 2
- Timeout: 30 days
- Instructions: "Verify staging deployment before approving"

**Gates**:
- **Invoke Azure Function**: Health check endpoint
- **Query Work Items**: No critical bugs open
- **Azure Monitor**: Alert count = 0

**Configuration**:

```yaml
environment:
  name: Production
  resourceType: None
approvals:
  - approver: approvers-group
    minRequired: 2
gates:
  - gate: AzureFunction
    inputs:
      function: '$(HealthCheckUrl)'
    successCriteria: 'eq(root[''status''], ''healthy'')'
```

---

## Test Plans Integration

**Test Plans → New Test Plan → Link to User Stories**

**Structure**:

```
Test Plan: Sprint 12
  ├─ Test Suite: Authentication
  │    ├─ Test Case: Login with valid credentials (linked to User Story #12345)
  │    └─ Test Case: Login with invalid credentials
  └─ Test Suite: Authorization
       └─ Test Case: Access control verification
```

**Automated test execution**:

```yaml
- task: VSTest@2
  inputs:
    testSelector: 'testAssemblies'
    testAssemblyVer2: |
      **\*test*.dll
      !**\*TestAdapter.dll
      !**\obj\**
    testPlan: $(TestPlanId)
    testSuite: $(TestSuiteId)
    publishRunAttachments: true
```

**Traceability**: Test results automatically linked to work items

---

## Quality Metrics Dashboard

**Dashboards → New Dashboard → Add widgets**

**Recommended widgets**:
- Build success rate (last 30 builds)
- Test pass rate (by test suite)
- Code coverage trend
- Deployment frequency (releases per week)
- Failed deployments

**Query example** (test coverage by requirement):

```
SELECT [ID], [Title], [Test Coverage %]
FROM WorkItems
WHERE [Work Item Type] = 'User Story'
ORDER BY [Test Coverage %] ASC
```

---

## Common Anti-Patterns

| Anti-Pattern | Better Approach |
|--------------|-----------------|
| Single-stage pipeline | Multi-stage with gates between stages |
| No test plans | Create test plans linked to requirements |
| Manual approvals only | Automate with gates (health checks, work item queries) |
| No deployment rollback | Define rollback strategy in release pipeline |

---

## Related Practices

- `./azdo-requirements.md` - Work item integration with tests
- `./azdo-measurement.md` - Pipeline metrics
- `./github-quality-gates.md` - GitHub comparison

---

**Last Updated**: 2026-01-25
