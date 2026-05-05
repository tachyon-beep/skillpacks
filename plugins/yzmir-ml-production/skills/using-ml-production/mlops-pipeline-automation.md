
# MLOps Pipeline Automation Skill

## When to Use This Skill

Use this skill when:
- Building production ML systems requiring automated workflows
- Implementing CI/CD for machine learning models
- Managing data and model versioning at scale
- Ensuring consistent feature engineering across training and serving
- Automating model retraining and deployment
- Orchestrating complex ML pipelines with multiple dependencies
- Operationalizing LLM applications (prompt versioning, RAG-index rebuilds, fine-tune pipelines, eval gates)

**When NOT to use:** One-off experiments, notebook prototypes, or research projects with no deployment requirements.

## Core Principle

**Manual ML workflows don't scale. Automation is mandatory for production.**

Without automation:
- Manual deployment: hours per model with elevated error rate from human steps
- No CI/CD: models reach production without testing and break it
- No data validation: schema/quality issues silently corrupt training and serving
- No feature store: train-serve skew causes large performance drops
- Manual retraining: models go stale and accuracy decays over weeks/months

**Formula:** CI/CD (automated testing + validation gates + deployment) + Feature stores (consistency + point-in-time correctness) + Data validation (schema checks + drift detection) + Model validation (accuracy thresholds + regression tests) + Automated retraining (triggers + orchestration) + Orchestration (DAG scheduling) = Production-ready MLOps.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    1. Git-Based Workflows                    │
│   Code (git) + Data (DVC / lakeFS) + Model registry          │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    2. CI/CD for ML                           │
│   Automated tests + Validation gates + Deployment pipeline   │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    3. Data Validation                        │
│   Schema (Great Expectations / Pandera / Soda)               │
│   + Drift (Evidently / NannyML / Alibi Detect)               │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    4. Feature Store                          │
│   Online + offline + point-in-time correctness               │
│   (Feast / Tecton / Hopsworks / Vertex / SageMaker FS)       │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    5. Model Validation                       │
│   Accuracy thresholds + Bias checks + Regression tests       │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    6. Pipeline Orchestration                 │
│   Airflow / Prefect / Dagster / Flyte / Metaflow / ZenML /   │
│   Argo / Kubeflow / managed (SageMaker / Vertex / Azure ML)  │
└────────────────────────────┬────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    7. Automated Retraining                   │
│   Triggered by drift / SLO breach / schedule / new data      │
└─────────────────────────────────────────────────────────────┘
```


## RED: Manual ML Workflows (The Problems)

### Failure 1: Manual Deployment (Slow and Error-Prone)

**Problem:** Data scientists manually export models, copy files to servers, edit configs, restart services.

**Symptoms:**
- Hours per deployment, elevated failure rate from human steps
- No rollback capability
- Configuration mismatches between environments
- "Works on my machine" syndrome

```python
# Manual deployment script (DON'T DO THIS)
def manual_deploy():
    model = train_model()
    joblib.dump(model, "model.pkl")
    # scp model.pkl user@prod-server:/models/  ← manual SSH, no audit trail
    # edit /etc/ml-service/config.yaml         ← typos break production
    # ssh user@prod-server "sudo systemctl restart ml-service"
    print("Deployment complete. Fingers crossed!")
```

### Failure 2: No CI/CD for ML (Models Not Tested Before Deploy)

```python
def deploy_without_testing():
    model = train_model(data)
    deploy_to_production(model)  # No regression test, no perf test, no schema check
```

Common production breakages traceable to a missing CI/CD pipeline:
- Lower accuracy than the previous version (regression)
- Failure on edge cases not seen in training
- Incompatible input schema (breaks API contract)
- Latency exceeds production SLO
- Higher bias than the prior version
- Missing dependencies on production hosts

### Failure 3: No Data Validation (Garbage In, Models Break)

Schema changes (renamed columns, dtype shifts), out-of-range outliers, and missing-value explosions all silently destroy model performance. A single Great-Expectations / Pandera / Soda suite run before training and before serving prevents most of these.

```python
class DataValidationCheck:
    EXPECTED_SCHEMA = {'user_id':'int64','age':'int64','income':'float64'}
    EXPECTED_RANGES = {'age':(18,100),'income':(0,500000)}
    def validate(self, df):
        assert set(df.columns) >= set(self.EXPECTED_SCHEMA)
        for c, t in self.EXPECTED_SCHEMA.items():
            assert df[c].dtype == t, f"{c} dtype changed"
        for c, (lo, hi) in self.EXPECTED_RANGES.items():
            assert df[c].between(lo, hi).all(), f"{c} out of range"
        assert df.isnull().mean().max() < 0.10
```

### Failure 4: No Feature Store (Train-Serve Skew)

Feature logic duplicated across training pipelines and online inference paths drifts apart over time — column-name mismatches, timestamp-handling inconsistencies, point-in-time correctness violations. Production accuracy diverges from training accuracy by 10–25%, and the gap can take weeks to attribute. A feature store is the architectural fix.

### Failure 5: Manual Retraining (Stale Models)

Without automated triggers, retraining cadence is "whenever someone notices." Performance degrades quietly until users complain, then a multi-week manual retrain cycle starts. Automated retraining triggered by drift, SLO breach, or schedule keeps models inside their performance envelope.


## GREEN: Automated MLOps (The Solutions)

### Solution 1: CI/CD for ML

```python
# CI/CD test framework
import pytest, time, joblib, numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

class MLModelCI:
    def __init__(self, model, X_test, y_test):
        self.model, self.X_test, self.y_test = model, X_test, y_test

    def run_all_tests(self):
        tests = {
            "unit":        self.test_basic,
            "accuracy":    self.test_accuracy,
            "latency":     self.test_latency,
            "regression":  self.test_no_regression,
            "integration": self.test_data_pipeline,
        }
        results, all_passed = {}, True
        for name, fn in tests.items():
            try:
                ok = fn()
                results[name] = "PASS" if ok else "FAIL"
                all_passed &= ok
            except Exception as e:
                results[name] = f"ERROR: {e}"; all_passed = False
        return all_passed, results

    def test_basic(self):
        preds = self.model.predict(self.X_test[:10])
        return preds.shape[0] == 10

    def test_accuracy(self):
        MIN = 0.85
        return accuracy_score(self.y_test, self.model.predict(self.X_test)) >= MIN

    def test_latency(self):
        MAX_MS = 100
        t0 = time.time()
        _ = self.model.predict(self.X_test[:100])
        return ((time.time() - t0) * 1000 / 100) <= MAX_MS

    def test_no_regression(self):
        try:
            prod = joblib.load('models/production_model.pkl')
            return accuracy_score(self.y_test, self.model.predict(self.X_test)) >= \
                   accuracy_score(self.y_test, prod.predict(self.X_test)) - 0.01
        except FileNotFoundError:
            return True

    def test_data_pipeline(self):
        from data_pipeline import load_test_data
        return load_test_data() is not None
```

GitHub Actions reference (works equivalently in GitLab CI / Azure Pipelines / Buildkite):

```yaml
name: ML Model CI/CD
on:
  push: { branches: [ main, develop ] }
  pull_request: { branches: [ main ] }
jobs:
  test-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt pytest
      - run: pytest tests/ -v --cov=src
      - run: python train_model.py && python validate_model.py
      - run: python ci/check_model_metrics.py
      - uses: actions/upload-artifact@v4
        with: { name: trained-model, path: models/model.pkl }
  deploy-model:
    needs: test-model
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with: { name: trained-model, path: models/ }
      - run: python deploy.py --environment staging
      - run: python tests/smoke_tests.py --environment staging
      - run: python deploy.py --environment production
```

GitHub Actions docs: <https://docs.github.com/actions>. For Kubernetes-native CD specifically, **Argo CD** (<https://argo-cd.readthedocs.io>) and **Flux** (<https://fluxcd.io>) are the GitOps standards — pair them with image-tag updaters that promote model containers from staging to prod after evals pass.

### Solution 2: Feature Store (Consistent Features)

```python
# Feature store with Feast (open-source, multi-backend)
from feast import FeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

class MLFeatureStore:
    def __init__(self, repo_path: str = "feature_repo"):
        self.store = FeatureStore(repo_path=repo_path)

    def define_features(self):
        user = Entity(name="user", join_keys=["user_id"])
        user_features = FeatureView(
            name="user_features", entities=[user],
            schema=[
                Field(name="age",                       dtype=Int64),
                Field(name="lifetime_purchases",        dtype=Int64),
                Field(name="avg_purchase_amount",       dtype=Float32),
                Field(name="days_since_last_purchase",  dtype=Int64),
                Field(name="purchase_frequency",        dtype=Float32),
            ],
            source=FileSource(path="data/user_features.parquet",
                              timestamp_field="event_timestamp"),
            ttl=timedelta(days=365),
        )
        return [user_features]

    def get_training_features(self, entity_df, features):
        return self.store.get_historical_features(entity_df=entity_df,
                                                  features=features).to_df()

    def get_online_features(self, entity_rows, features):
        return self.store.get_online_features(entity_rows=entity_rows,
                                              features=features).to_df()
```

**Feature-store landscape:**

- **Feast** — open-source, vendor-neutral; supports Redis, DynamoDB, BigQuery, Snowflake, etc. as online/offline stores. <https://docs.feast.dev>
- **Tecton** — managed enterprise feature platform with stream/batch/realtime transformations. <https://www.tecton.ai>
- **Hopsworks** — open-source feature store + ML platform from Logical Clocks. <https://www.hopsworks.ai/feature-store>
- **Databricks Feature Engineering** (formerly Feature Store) — integrated with Unity Catalog. <https://docs.databricks.com/aws/en/machine-learning/feature-store/>
- **AWS SageMaker Feature Store** — <https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html>
- **GCP Vertex AI Feature Store** — <https://cloud.google.com/vertex-ai/docs/featurestore/latest>
- **Azure ML Feature Store** — <https://learn.microsoft.com/azure/machine-learning/concept-what-is-managed-feature-store>

### Solution 3: Data Validation (Schema and Drift)

**Validation libraries:**

- **Great Expectations** — declarative expectation suites with rich docs/HTML reports. <https://docs.greatexpectations.io>
- **Pandera** — Pythonic dataframe schema validation, integrates with pandas / polars / pyspark. <https://pandera.readthedocs.io>
- **Soda Core** / **Soda Cloud** — SQL-native checks for warehouse data quality. <https://docs.soda.io>
- **dbt tests** — for dbt-managed warehouse pipelines. <https://docs.getdbt.com/docs/build/data-tests>

```python
import great_expectations as ge
import pandas as pd

class DataValidator:
    def create_expectations(self, df: pd.DataFrame):
        ge_df = ge.from_pandas(df, expectation_suite_name="data_validation")
        ge_df.expect_table_columns_to_match_ordered_list(
            ['user_id','age','income','purchase_history','target'])
        ge_df.expect_column_values_to_be_of_type('user_id','int')
        ge_df.expect_column_values_to_be_between('age', 18, 100)
        ge_df.expect_column_values_to_be_between('income', 0, 500_000)
        ge_df.expect_column_values_to_not_be_null('user_id')
        ge_df.expect_column_values_to_be_unique('user_id')
        ge_df.expect_column_mean_to_be_between('age', 25, 65)
        return ge_df

    def validate(self, df):
        results = self.create_expectations(df).validate()
        if not results['success']:
            for failed in [e for e in results['results'] if not e['success']]:
                print("FAIL:", failed['expectation_config']['expectation_type'])
        return results
```

**Drift libraries:** `Evidently`, `NannyML`, `Alibi Detect`, `Deepchecks`, `TorchDrift` — see the production-monitoring sheet's drift section for citations and trade-offs.

### Solution 4: Model Validation

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np, time, joblib

class ModelValidator:
    def __init__(self, model, X_test, y_test, prod_model_path=None):
        self.model, self.X_test, self.y_test = model, X_test, y_test
        self.prod_model_path = prod_model_path

    def validate_all(self):
        checks = {
            'accuracy':    self.check_accuracy,
            'regression':  self.check_no_regression,
            'performance': self.check_performance,
            'robustness':  self.check_robustness,
        }
        results, all_passed = {}, True
        for name, fn in checks.items():
            ok, details = fn()
            results[name] = {'passed': ok, 'details': details}
            all_passed &= ok
        return all_passed, results

    def check_accuracy(self):
        y_pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        details = {'accuracy': acc, 'f1': f1}
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(self.X_test)
            if proba.shape[1] == 2:
                details['auc'] = roc_auc_score(self.y_test, proba[:,1])
        return acc >= 0.85 and f1 >= 0.80, details

    def check_no_regression(self):
        if not self.prod_model_path:
            return True, {'message': 'no prod model'}
        prod = joblib.load(self.prod_model_path)
        new_acc = accuracy_score(self.y_test, self.model.predict(self.X_test))
        prod_acc = accuracy_score(self.y_test, prod.predict(self.X_test))
        return new_acc >= prod_acc - 0.01, {'new_acc': new_acc, 'prod_acc': prod_acc}

    def check_performance(self):
        latencies = []
        for _ in range(100):
            t0 = time.time()
            _ = self.model.predict(self.X_test[:1])
            latencies.append((time.time() - t0) * 1000)
        p95 = float(np.percentile(latencies, 95))
        return p95 < 100, {'p95_ms': p95}

    def check_robustness(self):
        X2 = self.X_test + np.random.normal(0, 0.01, self.X_test.shape)
        agreement = float(np.mean(self.model.predict(self.X_test) == self.model.predict(X2)))
        return agreement >= 0.95, {'agreement': agreement}
```

For fairness checks specifically, prefer dedicated libraries over hand-rolled logic: **Fairlearn** (<https://fairlearn.org>), **AIF360** (<https://aif360.res.ibm.com>). For broader ML quality testing in CI, **Deepchecks** (<https://docs.deepchecks.com>).


### Solution 5: Pipeline Orchestration

The orchestrator landscape has matured significantly — **Airflow** is no longer the default for new ML platforms. Pick based on cloud strategy, language preferences, scale, and how much you value asset-/lineage-based abstractions versus task-based DAGs.

#### Orchestrator Survey

**Apache Airflow** — incumbent, widely deployed, batch-oriented. Strong scheduler, mature operator ecosystem, large community. Weaker fit for ML-specific concerns: DAG definitions are static (less Pythonic), passing data between tasks goes through XComs (small payloads only) or external storage, dynamic task generation is awkward. Best for batch data-engineering DAGs. Reposition as the "good for batch ETL, OK for ML" choice. Docs: <https://airflow.apache.org/docs/>. Apache project page: <https://airflow.apache.org>. Astronomer (managed Airflow): <https://www.astronomer.io>.

**Prefect** (Prefect 2 / Prefect 3) — Pythonic flows decorated with `@flow` / `@task`, native async, dynamic DAGs ("subflows" and runtime-generated tasks), hybrid execution model (orchestration server + agent/worker pools you run anywhere). Prefect Cloud is the managed control plane; Prefect Server is the OSS self-hostable equivalent. Strong choice for teams that want Airflow-style orchestration with a much friendlier Python DX. Docs: <https://docs.prefect.io>. Site: <https://www.prefect.io>.

**Dagster** — software-defined assets (SDAs) paradigm: you declare *the data assets you want to exist* and Dagster figures out which ops to run. Built-in lineage, asset materialization tracking, partitioning, and observability. Especially strong for analytics-engineering and ML data prep pipelines. Integrates with dbt natively. Docs: <https://docs.dagster.io>. Site: <https://dagster.io>.

**Flyte** — Kubernetes-native, strongly-typed Python tasks with versioned, immutable workflows; first-class for ML/data-science workloads at scale (developed at Lyft, now LF AI & Data). Caches outputs on inputs+code, supports dynamic workflows, integrates with Spark / Ray / Dask. Steeper ops cost than Prefect but better fit for large multi-team ML platforms. Docs: <https://docs.flyte.org>. Site: <https://flyte.org>. Managed: **Union.ai** (<https://www.union.ai>).

**Metaflow** — Netflix open-source, scientist-friendly. Decorator-based (`@step`), seamless local-to-cloud (`@batch`, `@kubernetes`), built-in artifact storage and Python-native data passing between steps. Best DX for data scientists who want to focus on the model rather than the platform. Native AWS support; community support for Azure and GCP. Docs: <https://docs.metaflow.org>. Site: <https://metaflow.org>. Managed: **Outerbounds** (<https://outerbounds.com>).

**ZenML** — MLOps framework with backend pluggability: write pipelines once, swap stack components (orchestrator, artifact store, experiment tracker, model deployer) via configuration. Itself runs *on top of* Airflow / Kubeflow / Vertex / SageMaker / Tekton / Argo as the actual orchestrator. Strong for teams that want a uniform pipeline definition while migrating between underlying platforms. Docs: <https://docs.zenml.io>. Site: <https://zenml.io>.

**Argo Workflows** — Kubernetes-native, container-only DAGs (each step is a pod). YAML or Python DSL (Hera). Lower-level than Prefect/Dagster, no Python data-passing abstractions, but rock-solid for Kubernetes shops that want minimal extra moving parts. Argoproj umbrella also includes Argo CD (GitOps deploys) and Argo Events (event-driven triggers). Docs: <https://argo-workflows.readthedocs.io>. Site: <https://argoproj.github.io>.

**Kubeflow Pipelines (KFP) v2** — Kubernetes-native pipeline orchestration tailored for ML, with the KFP v2 SDK and component model (Python decorators that compile to pipeline definitions). Runs on Kubeflow on K8s, and the same pipeline can target **Vertex AI Pipelines** (Google's managed runtime). Better Python DX than v1 (which used `kfp.dsl.ContainerOp` heavily). Docs: <https://www.kubeflow.org/docs/components/pipelines/>. Quickstart: <https://www.kubeflow.org/docs/components/pipelines/getting-started/>.

**Cloud-managed pipelines** (use when you're already deeply in one cloud):
- **AWS SageMaker Pipelines** — Python SDK, integrated with SageMaker model registry, training jobs, batch transform. Docs: <https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html>.
- **GCP Vertex AI Pipelines** — runs KFP v2 pipelines on a managed serverless backend. Docs: <https://cloud.google.com/vertex-ai/docs/pipelines/introduction>.
- **Azure ML Pipelines** — YAML or Python SDK v2, integrated with Azure ML compute, environments, and registry. Docs: <https://learn.microsoft.com/azure/machine-learning/concept-ml-pipelines>.
- **Databricks Workflows / Lakeflow Jobs** — orchestrator for notebooks, Python wheels, dbt, Spark jobs on Databricks. Docs: <https://docs.databricks.com/aws/en/jobs/>.

**Other notable orchestrators:**
- **Mage** — open-source Pythonic / SQL pipeline tool, batch + streaming. <https://www.mage.ai>
- **Kestra** — declarative YAML workflows, strong UI. <https://kestra.io>
- **Temporal** — durable workflows (general-purpose, used by some ML teams for long-running training jobs and human-in-the-loop). <https://temporal.io>

#### Orchestrator Selection Matrix

| Dimension | Airflow | Prefect | Dagster | Flyte | Metaflow | ZenML | Argo Workflows | KFP v2 | Cloud-managed (SM/Vertex/Az ML) |
|---|---|---|---|---|---|---|---|---|---|
| Primary paradigm | Task DAGs | Pythonic flows | Asset graphs | Typed K8s tasks | Decorated steps | Stack-portable | Container DAGs | ML-specific DAGs | Cloud-native ML |
| Language | Python (DAG file) | Python (decorators) | Python (decorators) | Python (decorators) | Python (decorators) | Python (decorators) | YAML / Hera | Python SDK | Python SDK / YAML |
| Data passing | XCom / external | Python objects | Asset materialization | Typed I/O | Python objects | Pluggable | Pod artifacts | Pipeline artifacts | Pipeline artifacts |
| Self-host complexity | High (scheduler+executors+DB) | Low (server + workers) | Medium | High (K8s) | Low–Medium | Inherits backend | Medium (K8s) | High (Kubeflow on K8s) | None (managed) |
| Cloud-managed option | Astronomer / MWAA / Composer | Prefect Cloud | Dagster+ | Union | Outerbounds | ZenML Pro | (none) | Vertex AI Pipelines | Native |
| Native ML focus | Generic | Generic | Generic + analytics | Strong | Strong | Strong (MLOps) | Generic | Strong | Strong |
| Lineage / observability | Add-on (OpenLineage) | Built-in basic | Strong (built-in) | Strong | Built-in artifacts | Pluggable | Minimal | Pipeline-scoped | Cloud-native |
| Best for | Established batch ETL shops | Python teams wanting Airflow-DX-without-pain | Analytics + ML with strong lineage needs | Large multi-team ML at scale on K8s | Data scientists who want to ship fast | Teams migrating between backends | K8s purists | Teams committed to Kubeflow / Vertex | Teams committed to one cloud |

#### Worked Example: Airflow DAG (kept as the canonical batch example)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
}

with DAG(
    'ml_training_pipeline',
    default_args=default_args,
    schedule='0 2 * * *',                  # daily 02:00 (Airflow 2.4+ uses `schedule`)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['ml','training'],
) as dag:

    extract = PythonOperator(task_id='extract_data',     python_callable=extract_data)
    validate = PythonOperator(task_id='validate_data',   python_callable=validate_data)
    drift = PythonOperator(task_id='check_drift',         python_callable=check_drift)
    preprocess = PythonOperator(task_id='preprocess',     python_callable=preprocess_data)
    train = PythonOperator(task_id='train_model',         python_callable=train_model)
    validate_model_t = PythonOperator(task_id='validate_model', python_callable=validate_model)
    deploy = PythonOperator(task_id='deploy_model',       python_callable=deploy_model)
    monitor = PythonOperator(task_id='monitor',           python_callable=monitor_deployment)

    extract >> validate >> drift >> preprocess >> train >> validate_model_t >> deploy >> monitor
```

#### Worked Example: Prefect 2/3 flow

```python
from prefect import flow, task

@task(retries=2, retry_delay_seconds=30)
def extract():        return load_data()
@task
def validate(df):     return run_great_expectations(df)
@task
def train(df):        return fit_model(df)
@task
def evaluate(model, holdout):  return run_validator(model, holdout)
@task
def deploy(model):    return push_to_registry(model)

@flow(name="ml-training")
def ml_training_flow():
    df = extract()
    validated = validate(df)
    model = train(validated)
    metrics = evaluate(model, load_holdout())
    if metrics["accuracy"] >= 0.85:
        deploy(model)
```

Run with `prefect deploy` against Prefect Server / Prefect Cloud and a worker pool (e.g. process / Kubernetes / ECS / Vertex). Docs: <https://docs.prefect.io/latest/deploy/>.

#### Worked Example: Dagster software-defined assets

```python
from dagster import asset, AssetIn, Definitions, ScheduleDefinition

@asset
def raw_transactions():
    return query_warehouse("SELECT * FROM transactions WHERE date > current_date - 90")

@asset
def validated_transactions(raw_transactions):
    return run_great_expectations(raw_transactions)

@asset
def trained_model(validated_transactions):
    return fit_model(validated_transactions)

@asset
def deployment(trained_model):
    return register_and_deploy(trained_model)

defs = Definitions(
    assets=[raw_transactions, validated_transactions, trained_model, deployment],
    schedules=[ScheduleDefinition(cron_schedule="0 2 * * *",
                                  asset_selection=["deployment"])]
)
```

Dagster's UI shows the asset graph, last materialization time per asset, and lineage to upstream sources — invaluable for diagnosing "which input changed and broke the model?"

#### Worked Example: Kubeflow Pipelines v2

```python
from kfp import dsl, compiler

@dsl.component(packages_to_install=['scikit-learn==1.4.*','pandas','joblib'])
def train_op(data_path: str, model_path: dsl.Output[dsl.Model]):
    import pandas as pd, joblib
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_parquet(data_path)
    X, y = df.drop('target', axis=1), df['target']
    m = RandomForestClassifier(n_estimators=100).fit(X, y)
    joblib.dump(m, model_path.path)

@dsl.component
def validate_op(model: dsl.Input[dsl.Model]) -> float:
    import joblib
    return float(joblib.load(model.path).score_self())  # placeholder

@dsl.pipeline(name='ml-training')
def pipeline(data_path: str = "gs://bucket/data.parquet"):
    t = train_op(data_path=data_path)
    validate_op(model=t.outputs['model_path'])

compiler.Compiler().compile(pipeline, 'pipeline.yaml')
# Submit to Kubeflow Pipelines or Vertex AI Pipelines
```

Vertex AI Pipelines runs the same compiled `pipeline.yaml` as a managed serverless service.


### Solution 6: Automated Retraining Triggers

```python
import os, joblib, pandas as pd, numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score

class AutomatedRetrainingSystem:
    def __init__(self, model_path, accuracy_threshold=0.85,
                 drift_threshold=5, schedule_days=7):
        self.model_path = model_path
        self.accuracy_threshold = accuracy_threshold
        self.drift_threshold = drift_threshold
        self.schedule_days = schedule_days
        self.last_retrain_date = self._load_last()

    def should_retrain(self):
        reasons = []
        if self._performance_degraded():    reasons.append("performance_degradation")
        if self._data_drift():              reasons.append("data_drift")
        if self._scheduled_due():           reasons.append("scheduled")
        if self._manual_trigger():          reasons.append("manual")
        return bool(reasons), ", ".join(reasons) or "none"

    def _performance_degraded(self):
        recent = self._load_recent_predictions(days=1)
        if len(recent) < 100: return False
        return accuracy_score(recent['actual'], recent['predicted']) < self.accuracy_threshold

    def _data_drift(self):
        ref = pd.read_parquet('data/reference_data.parquet')
        curr = self._load_recent_features(days=7)
        # Use Evidently / NannyML / Alibi Detect in production; KS as quick check:
        from scipy.stats import ks_2samp
        n_drifted = sum(
            ks_2samp(ref[c].dropna(), curr[c].dropna()).pvalue < 0.05
            for c in ref.columns if ref[c].dtype.kind in 'fi'
        )
        return n_drifted > self.drift_threshold

    def _scheduled_due(self):
        return (datetime.now() - self.last_retrain_date).days >= self.schedule_days

    def _manual_trigger(self):
        flag = '/tmp/manual_retrain_trigger'
        if os.path.exists(flag):
            os.remove(flag); return True
        return False

    def trigger_retraining(self, reason):
        # Trigger your orchestrator. Examples:
        #   Airflow:  airflow dags trigger ml_training_pipeline --conf '{"reason":"..."}'
        #   Prefect:  prefect deployment run ml-training/prod
        #   Dagster:  dagster job execute -j ml_training
        #   KFP:      kfp run create -e <experiment> -f pipeline.yaml
        ...
        self.last_retrain_date = datetime.now()
        self._save_last()
```


## MLOps for LLM Applications

LLM apps shift the MLOps surface area: the "model" is often a third-party API, but the *application* (prompts, retrieval pipelines, eval suites, fine-tunes, agent tools) still needs the same rigor — versioning, testing, deployment, monitoring. Cross-references: see `yzmir-llm-specialist/llm-finetuning-strategies.md` for fine-tune training pipelines, `rag-architecture-patterns.md` for RAG-index design, `llm-evaluation-metrics.md` for eval methodology, and `prompt-engineering-patterns.md` for prompt design.

### Prompt Versioning and Deployment

Treat prompts as first-class artifacts: version-controlled, tested, deployed via CI/CD with rollback. Tools:

- **Langfuse Prompt Management** — versioned prompts with labels (`production`, `staging`, `experiment-A`), retrievable via SDK at runtime; supports A/B and rollback. <https://langfuse.com/docs/prompt-management/overview>
- **PromptLayer** — prompt registry, evals, and observability. <https://www.promptlayer.com>
- **Pezzo** — open-source prompt management and observability. <https://pezzo.ai> (verify maintenance status before adopting — small project)
- **Helicone Prompts** — built into Helicone's proxy. <https://docs.helicone.ai/features/prompts>
- **LangSmith Prompts** — LangChain-native prompt hub. <https://docs.smith.langchain.com/prompt_engineering>
- **Agenta** — open-source prompt-engineering platform. <https://agenta.ai>
- **Plain Git** — for many teams, prompts as `.txt` / `.md` / `.yaml` files in the model repo, deployed via standard CI/CD, is sufficient and simpler.

```python
# Example: pull a labeled prompt at runtime via Langfuse SDK
from langfuse import Langfuse
lf = Langfuse()  # reads env vars
prompt = lf.get_prompt("customer-support-router", label="production")
filled = prompt.compile(user_message=user_input, locale="en-GB")
```

### RAG Index Rebuild Pipelines

A RAG system is a continuously-running pipeline: source documents change, the embedding index must be rebuilt or incrementally updated, retrieval quality must be measured. Pattern:

1. **Source ingestion** — pull from CMS / S3 / docs portal / Confluence on a schedule or via webhook.
2. **Document processing** — chunk, clean, deduplicate. Track chunk-level provenance (source URL + offset) for citations.
3. **Embedding generation** — call the chosen embedding-model tier in batches; respect rate limits.
4. **Index write** — upsert to vector store (Pinecone / Weaviate / Qdrant / Milvus / pgvector / Chroma / Vertex AI Vector Search / OpenSearch k-NN / Elasticsearch).
5. **Eval gate** — run a labeled query set against the new index; require retrieval recall@k and end-to-end faithfulness above thresholds before promotion.
6. **Promotion** — atomically swap the live index alias (most vector stores support namespaces or aliases for blue/green).
7. **Monitoring** — track retrieval-relevance scores in production (cross-ref `production-monitoring-and-alerting.md`).

This entire flow fits naturally into Prefect, Dagster (assets!), Airflow, Argo, or KFP. Dagster's asset model is particularly well-suited because the index is literally an asset whose materialization depends on upstream documents.

### Eval Sets in CI

Keep an eval set under version control (golden Q&A pairs, regression cases, adversarial inputs). Run it in CI on every PR that touches prompts, RAG configuration, model tier, or eval logic. Tools:

- **DeepEval** — pytest-style LLM evals. <https://docs.confident-ai.com>
- **Promptfoo** — declarative YAML eval configs, CLI + CI integration. <https://www.promptfoo.dev>
- **Ragas** — RAG-specific metrics (faithfulness, answer-relevance, context-precision, context-recall). <https://docs.ragas.io>
- **TruLens** — eval + tracing for LLM apps. <https://www.trulens.org>
- **OpenAI Evals** — open-source eval framework. <https://github.com/openai/evals>
- **LangSmith Evaluations** — eval datasets + LLM-as-judge runs tied to traces. <https://docs.smith.langchain.com/evaluation>
- **Braintrust** — eval-and-experiment platform with strong UI. <https://www.braintrust.dev>

CI pattern: PR opens → run small fast eval set (smoke) → if green, run full eval set → require faithfulness ≥ 0.85, answer-relevance ≥ 0.80, regression-set pass rate = 100% before merge.

### Fine-Tune Training Pipelines

When fine-tuning local models or via provider fine-tuning APIs, treat the fine-tune job as a first-class pipeline step (cross-ref `yzmir-llm-specialist/llm-finetuning-strategies.md`):

1. **Data curation pipeline** — generate / clean / deduplicate / split training and eval data; version with DVC or warehouse snapshots.
2. **Training job** — local (Hugging Face Transformers + PEFT/LoRA + Accelerate, Axolotl, Unsloth, or Llama-Factory) or via provider fine-tuning APIs.
3. **Eval job** — run labeled eval set against the fine-tuned checkpoint; compare to base model and previous fine-tune.
4. **Registry** — push to **Hugging Face Hub** (private repo) or your model registry (MLflow, W&B Models — see `experiment-tracking-and-versioning.md`).
5. **Canary deploy** — route a small % of traffic to the new fine-tune; compare production-eval-judge scores; roll forward or back.

Hugging Face Hub: <https://huggingface.co/docs/hub/index>. Axolotl: <https://github.com/axolotl-ai-cloud/axolotl>. Unsloth: <https://github.com/unslothai/unsloth>. Llama-Factory: <https://github.com/hiyouga/LLaMA-Factory>.

### Capability-Tier Selection (No Hardcoded Model IDs)

When orchestrating LLM pipelines, parameterize the model by *capability tier* (e.g. `flagship`, `mid`, `cheap-fast`, `embedding-large`, `embedding-small`) and resolve the actual model ID at deploy time from a config map. This is the same discipline as not hardcoding container image tags — it lets you upgrade tiers across providers without rewriting pipeline code.

```yaml
# llm_models.yaml (versioned config)
tiers:
  flagship:    { provider: anthropic,   model: <resolved-at-deploy> }
  mid:         { provider: openai,      model: <resolved-at-deploy> }
  cheap_fast:  { provider: google,      model: <resolved-at-deploy> }
  embedding:   { provider: openai,      model: <resolved-at-deploy> }
```

The config is consumed by your orchestrator/runtime; bumping a tier is a config-only change with full CI eval coverage before promotion.


## REFACTOR: Pressure Tests

### Pressure Test 1: Scale to 100+ Models

```python
def test_scale_to_100_models():
    # CI/CD scales: 100 model pipelines run in parallel
    # Feature store: shared definitions used by all 100 models
    # Monitoring: dashboards aggregate health across all models
    pass
```

### Pressure Test 2: Deploy 10 Times Per Day

```python
def test_deploy_10_per_day():
    # Per-deploy cost: <10 minutes including tests + smoke + canary
    # Automatic rollback on canary regression
    pass
```

### Pressure Test 3: Detect Bad Data in < 1 Hour

```python
def test_data_quality_incident():
    bad = inject_corruption()
    assert not validator.validate(bad)['success']
    # Pipeline blocks; alert sent; no bad model trained
```

### Pressure Test 4: Auto-Retrain on Degradation

```python
def test_auto_retrain():
    simulate_degradation(target=0.70)
    should, reason = retraining_system.should_retrain()
    assert should and "performance_degradation" in reason
```

### Pressure Test 5: Feature Store at 1000 QPS

```python
def test_fs_qps():
    # Online store P95 latency < 10ms at 1000 QPS
    # No train-serve skew (training and serving features identical)
    pass
```

### Pressure Test 6: Rollback in < 5 Minutes

```python
def test_rollback():
    bad = train_intentionally_bad_model()
    ok, _ = ModelValidator(bad, X_test, y_test).validate_all()
    assert not ok  # validation blocks deployment
```

### Pressure Test 7: Drift-Triggered Retraining < 24h

```python
def test_drift_retrain():
    drifted = simulate_drift(n_features=10)
    assert sum(drift_detector.detect_drift(drifted).values()) >= 10
```

### Pressure Test 8: CI/CD < 10 Minutes

```python
def test_cicd_speed():
    t0 = time.time()
    ok, _ = MLModelCI(model, (X_test, y_test)).run_all_tests()
    assert ok and time.time() - t0 < 600
```

### Pressure Test 9: Train-Serve Feature Consistency

```python
def test_feature_consistency():
    e = pd.DataFrame({'user_id':[1001], 'event_timestamp':[pd.Timestamp('2026-01-01')]})
    train = fs.get_training_features(e, ['user_features:age']).iloc[0]['age']
    serve = fs.get_online_features({'user_id':[1001]}, ['user_features:age']).iloc[0]['age']
    assert train == serve
```

### Pressure Test 10: Eval-Set Regression in LLM CI

```python
def test_llm_eval_gate():
    # On every PR touching prompts/RAG/model-tier:
    #   - Run pinned eval set (golden Q&A + regression set)
    #   - Block merge if faithfulness < 0.85 or any regression case fails
    pass
```


## Summary

**MLOps automation transforms manual ML workflows into production-ready systems.**

Key implementations:

1. **CI/CD for ML** — automated tests, quality gates, automated deploy with rollback.
2. **Feature store** — single source of truth, point-in-time correctness, low-latency serving.
3. **Data validation** — schema (Great Expectations / Pandera / Soda) + drift (Evidently / NannyML / Alibi Detect / Deepchecks).
4. **Model validation** — accuracy thresholds, regression tests, performance, fairness (Fairlearn / AIF360).
5. **Pipeline orchestration** — choose Airflow / Prefect / Dagster / Flyte / Metaflow / ZenML / Argo / KFP / cloud-managed based on the selection matrix; don't default to Airflow without justification.
6. **Automated retraining** — drift / SLO / schedule / manual triggers, orchestrator-agnostic.
7. **MLOps for LLM applications** — prompt versioning, RAG-index rebuild pipelines, eval-set CI gates, fine-tune pipelines, capability-tier abstraction.

**Cross-pack integration:**
- Production signals from `production-monitoring-and-alerting.md` feed retraining triggers.
- `experiment-tracking-and-versioning.md` provides the model registry that CI/CD promotes through.
- `yzmir-llm-specialist/*` sheets supply the LLM-specific evals, prompts, and fine-tune strategies that this pipeline operationalizes.

Tooling and APIs current as of 2026-05; revisit quarterly.
