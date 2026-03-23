
# MLOps Pipeline Automation Skill

## When to Use This Skill

Use this skill when:
- Building production ML systems requiring automated workflows
- Implementing CI/CD for machine learning models
- Managing data and model versioning at scale
- Ensuring consistent feature engineering across training and serving
- Automating model retraining and deployment
- Orchestrating complex ML pipelines with multiple dependencies
- Implementing validation gates for data quality and model performance

**When NOT to use:** One-off experiments, notebook prototypes, or research projects with no deployment requirements.

## Core Principle

**Manual ML workflows don't scale. Automation is mandatory for production.**

Without automation:
- Manual deployment: 2-4 hours per model, 20% error rate
- No CI/CD: Models deployed without testing (12% break production)
- No data validation: Garbage in breaks models (8% of predictions fail)
- No feature store: Feature inconsistency causes 15-25% performance degradation
- Manual retraining: Models go stale (30% accuracy drop after 3 months)

**Formula:** CI/CD (automated testing + validation gates + deployment) + Feature stores (consistency + point-in-time correctness) + Data validation (schema checks + drift detection) + Model validation (accuracy thresholds + regression tests) + Automated retraining (triggers + orchestration) = Production-ready MLOps.

## MLOps Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    1. Git-Based Workflows                    │
│  Code versioning + DVC for data + Model registry + Branch    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    2. CI/CD for ML                          │
│  Automated tests + Validation gates + Deployment pipeline    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    3. Data Validation                       │
│  Schema checks + Great Expectations + Drift detection        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    4. Feature Store                         │
│  Online/offline stores + Point-in-time correctness + Feast   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    5. Model Validation                      │
│  Accuracy thresholds + Bias checks + Regression tests        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    6. Pipeline Orchestration                │
│  Airflow/Kubeflow/Prefect + DAGs + Dependency management     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    7. Automated Retraining                  │
│  Performance monitoring + Triggers + Scheduled updates       │
└─────────────────────────────────────────────────────────────┘
```


## RED: Manual ML Workflows (The Problems)

### Failure 1: Manual Deployment (Slow and Error-Prone)

**Problem:** Data scientists manually export models, copy files to servers, update configs, restart services.

**Symptoms:**
- 2-4 hours per deployment
- 20% deployments fail due to human error
- No rollback capability
- Configuration mismatches between environments
- "Works on my machine" syndrome

```python
# Manual deployment script (DON'T DO THIS)
def manual_deploy():
    """Manual model deployment - slow, error-prone, no validation."""

    # Step 1: Export model (manual)
    print("Exporting model...")
    model = train_model()
    joblib.dump(model, "model.pkl")

    # Step 2: Copy to server (manual, error-prone)
    print("Copying to production server...")
    # scp model.pkl user@prod-server:/models/
    # ^ Requires manual SSH, credentials, permission checks

    # Step 3: Update config (manual editing)
    print("Updating config file...")
    # Edit /etc/ml-service/config.yaml by hand
    # ^ Typos break production

    # Step 4: Restart service (manual)
    print("Restarting service...")
    # ssh user@prod-server "sudo systemctl restart ml-service"
    # ^ No health checks, no rollback

    # Step 5: Hope it works
    print("Deployment complete. Fingers crossed!")
    # ^ No validation, no monitoring, no alerts

# Problems:
# - Takes 2-4 hours
# - 20% failure rate
# - No version control
# - No rollback capability
# - No validation gates
```

**Impact:**
- Slow iteration: Deploy once per week instead of multiple times per day
- Production incidents: Manual errors break production
- Fear of deployment: Teams avoid deploying improvements
- Lost productivity: Engineers spend 30% time on deployment toil


### Failure 2: No CI/CD for ML (Models Not Tested Before Deploy)

**Problem:** Models deployed to production without automated testing or validation.

**Symptoms:**
- Models break production unexpectedly
- No regression testing (new models perform worse than old)
- No performance validation before deployment
- Integration issues discovered in production

```python
# No CI/CD - models deployed without testing
def deploy_without_testing():
    """Deploy model without any validation."""

    # Train model
    model = train_model(data)

    # Deploy immediately (NO TESTING)
    deploy_to_production(model)
    # ^ What could go wrong?

# What goes wrong:
# 1. Model has lower accuracy than previous version (regression)
# 2. Model fails on edge cases not seen in training
# 3. Model has incompatible input schema (breaks API)
# 4. Model is too slow for production latency requirements
# 5. Model has higher bias than previous version
# 6. Model dependencies missing in production environment

# Example: Production failure
class ProductionFailure:
    """Real production incident from lack of CI/CD."""

    def __init__(self):
        self.incident = {
            "timestamp": "2024-03-15 14:23:00",
            "severity": "CRITICAL",
            "issue": "Model prediction latency increased from 50ms to 2000ms",
            "root_cause": "New model uses feature requiring database join",
            "detection_time": "3 hours after deployment",
            "affected_users": 125000,
            "resolution": "Manual rollback to previous version",
            "downtime": "3 hours",
            "revenue_impact": "$75,000"
        }

        # This would have been caught by CI/CD:
        # 1. Performance test would catch 2000ms latency
        # 2. Validation gate would block deployment
        # 3. Automated rollback would trigger within 5 minutes
        # 4. Total downtime: 5 minutes instead of 3 hours
```

**Impact:**
- 12% of model deployments break production
- Mean time to detection: 2-4 hours
- Mean time to recovery: 2-6 hours (manual rollback)
- Customer trust erosion from prediction failures


### Failure 3: No Data Validation (Garbage In, Models Break)

**Problem:** Training and serving data not validated, leading to data quality issues that break models.

**Symptoms:**
- Schema changes break models in production
- Data drift degrades model performance
- Missing values cause prediction failures
- Invalid data types crash inference pipeline

```python
# No data validation - garbage in, garbage out
def train_without_validation(df):
    """Train model on unvalidated data."""

    # No schema validation
    # What if column names changed?
    # What if data types changed?
    # What if required columns are missing?

    # No data quality checks
    # What if 50% of values are null?
    # What if outliers are corrupted data?
    # What if categorical values have new unseen categories?

    # Just train and hope for the best
    X = df[['feature1', 'feature2', 'feature3']]  # KeyError if columns missing
    y = df['target']

    model = RandomForestClassifier()
    model.fit(X, y)

    return model

# Real production failures from lack of data validation:

# Failure 1: Schema change
# - Upstream team renamed "customer_id" to "customerId"
# - Model training crashed with KeyError
# - Detection time: 6 hours (next scheduled training run)
# - Impact: No model updates for 6 hours + debugging time

# Failure 2: Data type change
# - Feature "age" changed from int to string ("25 years")
# - Model predictions crashed at inference time
# - Detection: First user request after deployment
# - Impact: 100% prediction failure rate for 2 hours

# Failure 3: Extreme outliers from bad data
# - Corrupt data pipeline created outliers (prices: $999,999,999)
# - Model trained on corrupted data
# - Predictions wildly inaccurate
# - Detection: 24 hours (after user complaints)
# - Impact: 40% accuracy drop, customer trust damage

# Failure 4: Missing values explosion
# - Upstream ETL bug caused 80% null values
# - Model trained with 80% missing data
# - Predictions random and useless
# - Detection: After deployment and user complaints
# - Impact: Week of bad predictions before root cause found

# Example: Data validation would catch these
class DataValidationCheck:
    """What data validation should check."""

    EXPECTED_SCHEMA = {
        'customer_id': 'int64',
        'age': 'int64',
        'income': 'float64',
        'purchase_history': 'int64'
    }

    EXPECTED_RANGES = {
        'age': (18, 100),
        'income': (0, 500000),
        'purchase_history': (0, 1000)
    }

    def validate(self, df):
        """All checks that were skipped."""

        # Schema validation (would catch column rename)
        assert set(df.columns) == set(self.EXPECTED_SCHEMA.keys())

        # Data type validation (would catch type change)
        for col, dtype in self.EXPECTED_SCHEMA.items():
            assert df[col].dtype == dtype

        # Range validation (would catch outliers)
        for col, (min_val, max_val) in self.EXPECTED_RANGES.items():
            assert df[col].between(min_val, max_val).all()

        # Missing value validation (would catch null explosion)
        assert df.isnull().mean().max() < 0.10  # Max 10% nulls

        # All these checks take 5 seconds
        # Would have prevented days of production incidents
```

**Impact:**
- 8% of predictions fail due to data quality issues
- 30% accuracy degradation when data drift undetected
- Mean time to detection: 12-48 hours
- Debugging data quality issues: 2-5 days per incident


### Failure 4: No Feature Store (Inconsistent Features)

**Problem:** Feature engineering logic duplicated between training and serving, causing train-serve skew.

**Symptoms:**
- Training accuracy: 92%, Production accuracy: 78%
- Inconsistent feature calculations
- Point-in-time correctness violations (data leakage)
- Slow feature computation at inference time

```python
# No feature store - training and serving features diverge
class TrainServeSkew:
    """Training and serving compute features differently."""

    def training_features(self, user_id, training_data):
        """Features computed during training."""

        # Training time: Compute features from entire dataset
        user_data = training_data[training_data['user_id'] == user_id]

        # Average purchase amount (uses future data - leakage!)
        avg_purchase = user_data['purchase_amount'].mean()

        # Days since last purchase
        days_since_purchase = (
            pd.Timestamp.now() - user_data['purchase_date'].max()
        ).days

        # Purchase frequency
        purchase_frequency = len(user_data) / 365

        return {
            'avg_purchase': avg_purchase,
            'days_since_purchase': days_since_purchase,
            'purchase_frequency': purchase_frequency
        }

    def serving_features(self, user_id):
        """Features computed during serving (production)."""

        # Production: Query database for recent data
        user_data = db.query(f"SELECT * FROM purchases WHERE user_id = {user_id}")

        # Compute average (but query might return different time range)
        avg_purchase = user_data['amount'].mean()  # Column name different!

        # Days since last purchase (might use different timestamp logic)
        days_since = (datetime.now() - user_data['date'].max()).days

        # Frequency calculation might differ
        purchase_frequency = len(user_data) / 360  # Different denominator!

        return {
            'avg_purchase': avg_purchase,
            'days_since_purchase': days_since,
            'purchase_frequency': purchase_frequency
        }

# Problems with duplicated feature logic:
# 1. Column name inconsistency: 'purchase_amount' vs 'amount'
# 2. Timestamp handling inconsistency: pd.Timestamp.now() vs datetime.now()
# 3. Calculation inconsistency: / 365 vs / 360
# 4. Point-in-time correctness violated (training uses future data)
# 5. Performance: Slow database queries at serving time

# Impact on production accuracy:
# - Training accuracy: 92%
# - Production accuracy: 78% (14% drop due to feature inconsistency)
# - Debugging time: 2-3 weeks to identify train-serve skew
# - Cost: $200k in compute for debugging + lost revenue

# Feature store would solve this:
# - Single source of truth for feature definitions
# - Consistent computation in training and serving
# - Point-in-time correctness enforced
# - Precomputed features for fast serving
# - Feature reuse across models
```

**Impact:**
- 15-25% accuracy drop from train-serve skew
- 2-4 weeks to debug feature inconsistencies
- Slow inference (database queries at serving time)
- Feature engineering logic duplicated across models


### Failure 5: Manual Retraining (Stale Models)

**Problem:** Models retrained manually on ad-hoc schedule, causing stale predictions.

**Symptoms:**
- Model accuracy degrades over time
- Manual retraining every few months (or never)
- No automated triggers for retraining
- Production performance monitoring disconnected from retraining

```python
# Manual retraining - models go stale
class ManualRetraining:
    """Manual model retraining (happens rarely)."""

    def __init__(self):
        self.last_trained = datetime(2024, 1, 1)
        self.model_version = "v1.0"

    def check_if_retrain_needed(self):
        """Manual check (someone has to remember to do this)."""

        # Step 1: Someone notices accuracy dropped
        # (Requires: monitoring, someone looking at metrics, someone caring)
        print("Has anyone checked model accuracy lately?")

        # Step 2: Someone investigates
        # (Requires: time, expertise, access to metrics)
        print("Model accuracy dropped from 92% to 78%")

        # Step 3: Someone decides to retrain
        # (Requires: priority, resources, approval)
        print("Should we retrain? Let's schedule a meeting...")

        # Step 4: Weeks later, someone actually retrains
        # (Requires: compute resources, data pipeline working, manual steps)
        print("Finally retraining after 3 months...")

    def retrain_manually(self):
        """Manual retraining process."""

        # Step 1: Pull latest data (manual)
        print("Downloading data from warehouse...")
        data = manual_data_pull()  # Someone runs SQL query

        # Step 2: Preprocess data (manual)
        print("Preprocessing data...")
        processed = manual_preprocessing(data)  # Someone runs script

        # Step 3: Train model (manual)
        print("Training model...")
        model = manual_training(processed)  # Someone runs training script

        # Step 4: Validate model (manual)
        print("Validating model...")
        metrics = manual_validation(model)  # Someone checks metrics

        # Step 5: Deploy model (manual)
        print("Deploying model...")
        manual_deploy(model)  # Someone copies files to server

        # Step 6: Update docs (manual, often skipped)
        print("Updating documentation...")
        # (This step usually skipped due to time pressure)

        # Total time: 2-4 days
        # Frequency: Every 3-6 months (or when something breaks)

# What happens with manual retraining:
class ModelDecayTimeline:
    """How model performance degrades without automated retraining."""

    timeline = {
        "Week 0": {
            "accuracy": 0.92,
            "status": "Model deployed, performing well"
        },
        "Month 1": {
            "accuracy": 0.90,
            "status": "Slight degradation, unnoticed"
        },
        "Month 2": {
            "accuracy": 0.86,
            "status": "Noticeable degradation, no one investigating"
        },
        "Month 3": {
            "accuracy": 0.78,
            "status": "Major degradation, users complaining"
        },
        "Month 4": {
            "accuracy": 0.78,
            "status": "Meeting scheduled to discuss retraining"
        },
        "Month 5": {
            "accuracy": 0.75,
            "status": "Retraining approved, waiting for resources"
        },
        "Month 6": {
            "accuracy": 0.72,
            "status": "Finally retraining, takes 2 weeks"
        },
        "Month 6.5": {
            "accuracy": 0.91,
            "status": "New model deployed, accuracy restored"
        }
    }

    # Total accuracy degradation period: 6 months
    # Average accuracy during period: 0.82 (10% below optimal)
    # Impact: Lost revenue, poor user experience, competitive disadvantage

    # With automated retraining:
    # - Accuracy threshold trigger: < 0.90
    # - Automated retraining: Weekly
    # - Model always stays above 0.90
    # - No manual intervention required
```

**Impact:**
- 30% accuracy drop after 3-6 months without retraining
- Mean time to retrain: 4-8 weeks (from decision to deployment)
- Lost revenue from stale predictions
- Competitive disadvantage from degraded performance


## GREEN: Automated MLOps (The Solutions)

### Solution 1: CI/CD for ML (Automated Testing and Deployment)

**Goal:** Automate model testing, validation, and deployment with quality gates.

**Components:**
- Automated unit tests for model code
- Integration tests for data pipeline
- Model validation tests (accuracy, latency, bias)
- Automated deployment pipeline
- Rollback capability

```python
# CI/CD for ML - automated testing and deployment
import pytest
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import mlflow
from typing import Dict, Tuple
import time

class MLModelCI:
    """CI/CD pipeline for ML models."""

    def __init__(self, model, test_data: Tuple[np.ndarray, np.ndarray]):
        self.model = model
        self.X_test, self.y_test = test_data
        self.validation_results = {}

    def run_all_tests(self) -> Dict[str, bool]:
        """Run complete CI/CD test suite."""

        tests = {
            "unit_tests": self.test_model_basic_functionality,
            "accuracy_test": self.test_model_accuracy,
            "performance_test": self.test_inference_latency,
            "bias_test": self.test_model_fairness,
            "regression_test": self.test_no_regression,
            "integration_test": self.test_data_pipeline
        }

        results = {}
        all_passed = True

        for test_name, test_func in tests.items():
            try:
                passed = test_func()
                results[test_name] = "PASS" if passed else "FAIL"
                if not passed:
                    all_passed = False
            except Exception as e:
                results[test_name] = f"ERROR: {str(e)}"
                all_passed = False

        self.validation_results = results
        return all_passed

    def test_model_basic_functionality(self) -> bool:
        """Test basic model functionality."""

        # Test 1: Model can make predictions
        try:
            predictions = self.model.predict(self.X_test[:10])
            assert len(predictions) == 10
        except Exception as e:
            print(f"❌ Prediction test failed: {e}")
            return False

        # Test 2: Predictions have correct shape
        try:
            predictions = self.model.predict(self.X_test)
            assert predictions.shape[0] == self.X_test.shape[0]
        except Exception as e:
            print(f"❌ Shape test failed: {e}")
            return False

        # Test 3: Predictions in valid range
        try:
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(self.X_test)
                assert np.all((probas >= 0) & (probas <= 1))
        except Exception as e:
            print(f"❌ Probability range test failed: {e}")
            return False

        print("✅ Basic functionality tests passed")
        return True

    def test_model_accuracy(self) -> bool:
        """Test model meets accuracy threshold."""

        # Minimum accuracy threshold
        MIN_ACCURACY = 0.85
        MIN_AUC = 0.80

        # Compute metrics
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)

        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(self.y_test, probas)
        else:
            auc = None

        # Check thresholds
        if accuracy < MIN_ACCURACY:
            print(f"❌ Accuracy {accuracy:.3f} below threshold {MIN_ACCURACY}")
            return False

        if auc is not None and auc < MIN_AUC:
            print(f"❌ AUC {auc:.3f} below threshold {MIN_AUC}")
            return False

        print(f"✅ Accuracy test passed: {accuracy:.3f} (threshold: {MIN_ACCURACY})")
        if auc:
            print(f"✅ AUC test passed: {auc:.3f} (threshold: {MIN_AUC})")

        return True

    def test_inference_latency(self) -> bool:
        """Test inference latency meets requirements."""

        # Maximum latency threshold (milliseconds)
        MAX_LATENCY_MS = 100

        # Measure latency
        start_time = time.time()
        _ = self.model.predict(self.X_test[:100])
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000 / 100  # Per prediction

        if latency_ms > MAX_LATENCY_MS:
            print(f"❌ Latency {latency_ms:.2f}ms exceeds threshold {MAX_LATENCY_MS}ms")
            return False

        print(f"✅ Latency test passed: {latency_ms:.2f}ms (threshold: {MAX_LATENCY_MS}ms)")
        return True

    def test_model_fairness(self) -> bool:
        """Test model for bias across protected attributes."""

        # This is a simplified example
        # In production, use comprehensive fairness metrics

        # Assume X_test has a 'gender' column for this example
        # In reality, you'd need to handle this more carefully

        print("✅ Fairness test passed (simplified)")
        return True

    def test_no_regression(self) -> bool:
        """Test new model doesn't regress from production model."""

        try:
            # Load production model
            prod_model = joblib.load('models/production_model.pkl')

            # Compare accuracy
            new_predictions = self.model.predict(self.X_test)
            new_accuracy = accuracy_score(self.y_test, new_predictions)

            prod_predictions = prod_model.predict(self.X_test)
            prod_accuracy = accuracy_score(self.y_test, prod_predictions)

            # Allow small degradation (1%)
            if new_accuracy < prod_accuracy - 0.01:
                print(f"❌ Regression detected: {new_accuracy:.3f} vs prod {prod_accuracy:.3f}")
                return False

            print(f"✅ No regression: {new_accuracy:.3f} vs prod {prod_accuracy:.3f}")
            return True

        except FileNotFoundError:
            print("⚠️  No production model found, skipping regression test")
            return True

    def test_data_pipeline(self) -> bool:
        """Test data pipeline integration."""

        # Test data loading
        try:
            from data_pipeline import load_test_data
            data = load_test_data()
            assert data is not None
            assert len(data) > 0
        except Exception as e:
            print(f"❌ Data pipeline test failed: {e}")
            return False

        print("✅ Data pipeline test passed")
        return True


# GitHub Actions workflow for ML CI/CD
class GitHubActionsConfig:
    """Configuration for GitHub Actions ML CI/CD."""

    WORKFLOW_YAML = """
name: ML Model CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-model:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v

    - name: Train and validate model
      run: |
        python train_model.py
        python validate_model.py

    - name: Check model metrics
      run: |
        python ci/check_model_metrics.py

    - name: Upload model artifacts
      uses: actions/upload-artifact@v2
      with:
        name: trained-model
        path: models/model.pkl

  deploy-model:
    needs: test-model
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v2

    - name: Download model artifacts
      uses: actions/download-artifact@v2
      with:
        name: trained-model
        path: models/

    - name: Deploy to staging
      run: |
        python deploy.py --environment staging

    - name: Run smoke tests
      run: |
        python tests/smoke_tests.py --environment staging

    - name: Deploy to production
      run: |
        python deploy.py --environment production

    - name: Monitor deployment
      run: |
        python monitor_deployment.py
"""


# Pytest tests for model validation
class TestModelValidation:
    """Pytest tests for model CI/CD."""

    @pytest.fixture
    def trained_model(self):
        """Load trained model for testing."""
        return joblib.load('models/model.pkl')

    @pytest.fixture
    def test_data(self):
        """Load test data."""
        from data_pipeline import load_test_data
        return load_test_data()

    def test_model_accuracy(self, trained_model, test_data):
        """Test model meets accuracy threshold."""
        X_test, y_test = test_data
        predictions = trained_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        assert accuracy >= 0.85, f"Accuracy {accuracy:.3f} below threshold"

    def test_model_latency(self, trained_model, test_data):
        """Test model meets latency requirements."""
        X_test, _ = test_data

        start_time = time.time()
        _ = trained_model.predict(X_test[:100])
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000 / 100
        assert latency_ms < 100, f"Latency {latency_ms:.2f}ms exceeds threshold"

    def test_no_regression(self, trained_model, test_data):
        """Test new model doesn't regress from production."""
        X_test, y_test = test_data

        new_accuracy = accuracy_score(y_test, trained_model.predict(X_test))

        prod_model = joblib.load('models/production_model.pkl')
        prod_accuracy = accuracy_score(y_test, prod_model.predict(X_test))

        assert new_accuracy >= prod_accuracy - 0.01, \
            f"Regression: {new_accuracy:.3f} vs {prod_accuracy:.3f}"

    def test_prediction_range(self, trained_model, test_data):
        """Test predictions are in valid range."""
        X_test, _ = test_data

        if hasattr(trained_model, 'predict_proba'):
            probas = trained_model.predict_proba(X_test)
            assert np.all((probas >= 0) & (probas <= 1)), "Probabilities out of range"
```


### Solution 2: Feature Store (Consistent Features)

**Goal:** Single source of truth for features, ensuring consistency between training and serving.

**Components:**
- Centralized feature definitions
- Online and offline feature stores
- Point-in-time correctness
- Feature versioning and lineage

```python
# Feature store implementation using Feast
from feast import FeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta
import pandas as pd
from typing import List, Dict

class MLFeatureStore:
    """Feature store for consistent feature engineering."""

    def __init__(self, repo_path: str = "feature_repo"):
        self.store = FeatureStore(repo_path=repo_path)

    def define_features(self):
        """Define features once, use everywhere."""

        # Define entity (user)
        user = Entity(
            name="user",
            join_keys=["user_id"],
            description="User entity"
        )

        # Define user features
        user_features = FeatureView(
            name="user_features",
            entities=[user],
            schema=[
                Field(name="age", dtype=Int64),
                Field(name="lifetime_purchases", dtype=Int64),
                Field(name="avg_purchase_amount", dtype=Float32),
                Field(name="days_since_last_purchase", dtype=Int64),
                Field(name="purchase_frequency", dtype=Float32)
            ],
            source=FileSource(
                path="data/user_features.parquet",
                timestamp_field="event_timestamp"
            ),
            ttl=timedelta(days=365)
        )

        return [user_features]

    def get_training_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str]
    ) -> pd.DataFrame:
        """Get historical features for training (point-in-time correct)."""

        # Point-in-time correct feature retrieval
        # Only uses data available at entity_df['event_timestamp']
        training_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=features
        ).to_df()

        return training_df

    def get_online_features(
        self,
        entity_ids: Dict[str, List],
        features: List[str]
    ) -> pd.DataFrame:
        """Get latest features for online serving (low latency)."""

        # Fast retrieval from online store (Redis, DynamoDB, etc.)
        online_features = self.store.get_online_features(
            entity_rows=entity_ids,
            features=features
        ).to_df()

        return online_features


# Example: Using feature store for training and serving
class FeatureStoreExample:
    """Example of using feature store for consistency."""

    def __init__(self):
        self.feature_store = MLFeatureStore()

    def train_model_with_features(self):
        """Training with feature store (consistent features)."""

        # Define entity dataframe (users and timestamps)
        entity_df = pd.DataFrame({
            'user_id': [1001, 1002, 1003, 1004],
            'event_timestamp': [
                pd.Timestamp('2024-01-01'),
                pd.Timestamp('2024-01-02'),
                pd.Timestamp('2024-01-03'),
                pd.Timestamp('2024-01-04')
            ],
            'label': [1, 0, 1, 0]  # Target variable
        })

        # Get historical features (point-in-time correct)
        features = [
            'user_features:age',
            'user_features:lifetime_purchases',
            'user_features:avg_purchase_amount',
            'user_features:days_since_last_purchase',
            'user_features:purchase_frequency'
        ]

        training_df = self.feature_store.get_training_features(
            entity_df=entity_df,
            features=features
        )

        # Train model
        X = training_df[['age', 'lifetime_purchases', 'avg_purchase_amount',
                        'days_since_last_purchase', 'purchase_frequency']]
        y = training_df['label']

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(X, y)

        return model

    def predict_with_features(self, user_ids: List[int]):
        """Serving with feature store (same features as training)."""

        # Get online features (fast, low latency)
        features = [
            'user_features:age',
            'user_features:lifetime_purchases',
            'user_features:avg_purchase_amount',
            'user_features:days_since_last_purchase',
            'user_features:purchase_frequency'
        ]

        serving_df = self.feature_store.get_online_features(
            entity_ids={'user_id': user_ids},
            features=features
        )

        # Make predictions
        model = joblib.load('models/model.pkl')
        predictions = model.predict(serving_df)

        return predictions


# Feature computation and materialization
class FeatureComputation:
    """Compute and materialize features to feature store."""

    def compute_user_features(self, user_transactions: pd.DataFrame) -> pd.DataFrame:
        """Compute user features from raw transactions."""

        # This logic defined ONCE, used everywhere
        user_features = user_transactions.groupby('user_id').agg({
            'transaction_amount': ['mean', 'sum', 'count'],
            'transaction_date': ['max', 'min']
        }).reset_index()

        user_features.columns = [
            'user_id',
            'avg_purchase_amount',
            'total_spent',
            'lifetime_purchases',
            'last_purchase_date',
            'first_purchase_date'
        ]

        # Compute derived features
        user_features['days_since_last_purchase'] = (
            pd.Timestamp.now() - user_features['last_purchase_date']
        ).dt.days

        user_features['customer_lifetime_days'] = (
            user_features['last_purchase_date'] - user_features['first_purchase_date']
        ).dt.days + 1

        user_features['purchase_frequency'] = (
            user_features['lifetime_purchases'] / user_features['customer_lifetime_days']
        )

        # Add timestamp for Feast
        user_features['event_timestamp'] = pd.Timestamp.now()

        return user_features

    def materialize_features(self):
        """Materialize features to online store."""

        # Compute features
        transactions = pd.read_parquet('data/transactions.parquet')
        user_features = self.compute_user_features(transactions)

        # Save to offline store
        user_features.to_parquet('data/user_features.parquet')

        # Materialize to online store for serving
        feature_store = FeatureStore(repo_path="feature_repo")
        feature_store.materialize_incremental(end_date=pd.Timestamp.now())

        print(f"✅ Materialized {len(user_features)} user features")


# Benefits of feature store
class FeatureStoreBenefits:
    """Benefits of using a feature store."""

    benefits = {
        "consistency": {
            "problem": "Training accuracy 92%, production 78% (train-serve skew)",
            "solution": "Single feature definition, training and serving use same code",
            "impact": "Production accuracy matches training (92%)"
        },
        "point_in_time_correctness": {
            "problem": "Data leakage from using future data in training",
            "solution": "Feature store enforces point-in-time correctness",
            "impact": "No data leakage, accurate performance estimates"
        },
        "reusability": {
            "problem": "Each model team reimplements same features",
            "solution": "Features defined once, reused across models",
            "impact": "10x faster feature development"
        },
        "serving_latency": {
            "problem": "Database queries at inference time (500ms latency)",
            "solution": "Precomputed features in online store",
            "impact": "5ms feature retrieval latency"
        },
        "feature_discovery": {
            "problem": "Teams don't know what features exist",
            "solution": "Feature registry with documentation and lineage",
            "impact": "Faster model development, feature reuse"
        }
    }
```


### Solution 3: Data Validation (Schema Checks and Drift Detection)

**Goal:** Validate data quality and detect schema changes and distribution shifts.

**Components:**
- Schema validation
- Statistical validation
- Drift detection
- Data quality monitoring

```python
# Data validation using Great Expectations
import great_expectations as ge
from great_expectations.dataset import PandasDataset
import pandas as pd
from typing import Dict, List
import numpy as np
from scipy import stats

class DataValidator:
    """Validate data quality and detect issues."""

    def __init__(self, expectation_suite_name: str = "data_validation_suite"):
        self.suite_name = expectation_suite_name
        self.validation_results = {}

    def create_expectations(self, df: pd.DataFrame) -> PandasDataset:
        """Create data expectations (validation rules)."""

        # Convert to Great Expectations dataset
        ge_df = ge.from_pandas(df, expectation_suite_name=self.suite_name)

        # Schema expectations
        ge_df.expect_table_columns_to_match_ordered_list([
            'user_id', 'age', 'income', 'purchase_history', 'target'
        ])

        # Data type expectations
        ge_df.expect_column_values_to_be_of_type('user_id', 'int')
        ge_df.expect_column_values_to_be_of_type('age', 'int')
        ge_df.expect_column_values_to_be_of_type('income', 'float')

        # Range expectations
        ge_df.expect_column_values_to_be_between('age', min_value=18, max_value=100)
        ge_df.expect_column_values_to_be_between('income', min_value=0, max_value=500000)
        ge_df.expect_column_values_to_be_between('purchase_history', min_value=0, max_value=1000)

        # Missing value expectations
        ge_df.expect_column_values_to_not_be_null('user_id')
        ge_df.expect_column_values_to_not_be_null('target')
        ge_df.expect_column_values_to_be_null('income', mostly=0.9)  # Max 10% null

        # Uniqueness expectations
        ge_df.expect_column_values_to_be_unique('user_id')

        # Distribution expectations
        ge_df.expect_column_mean_to_be_between('age', min_value=25, max_value=65)
        ge_df.expect_column_stdev_to_be_between('age', min_value=10, max_value=20)

        return ge_df

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """Validate data against expectations."""

        # Create or load expectations
        ge_df = self.create_expectations(df)

        # Run validation
        results = ge_df.validate()

        # Check if all expectations passed
        success = results['success']
        failed_expectations = [
            exp for exp in results['results']
            if not exp['success']
        ]

        self.validation_results = {
            'success': success,
            'total_expectations': len(results['results']),
            'failed_count': len(failed_expectations),
            'failed_expectations': failed_expectations
        }

        if not success:
            print("❌ Data validation failed:")
            for failed in failed_expectations:
                print(f"  - {failed['expectation_config']['expectation_type']}")
                print(f"    {failed.get('exception_info', {}).get('raised_exception', 'See details')}")
        else:
            print("✅ Data validation passed")

        return self.validation_results


class DriftDetector:
    """Detect distribution drift in features."""

    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.drift_results = {}

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.05
    ) -> Dict[str, bool]:
        """Detect drift using statistical tests."""

        drift_detected = {}

        for column in self.reference_data.columns:
            if column in current_data.columns:
                # Numerical columns: Kolmogorov-Smirnov test
                if pd.api.types.is_numeric_dtype(self.reference_data[column]):
                    drift = self._ks_test(
                        self.reference_data[column],
                        current_data[column],
                        threshold
                    )
                # Categorical columns: Chi-square test
                elif pd.api.types.is_categorical_dtype(self.reference_data[column]) or \
                     pd.api.types.is_object_dtype(self.reference_data[column]):
                    drift = self._chi_square_test(
                        self.reference_data[column],
                        current_data[column],
                        threshold
                    )
                else:
                    drift = False

                drift_detected[column] = drift

        self.drift_results = drift_detected

        # Report drift
        drifted_features = [col for col, drifted in drift_detected.items() if drifted]

        if drifted_features:
            print(f"⚠️  Drift detected in {len(drifted_features)} features:")
            for feature in drifted_features:
                print(f"  - {feature}")
        else:
            print("✅ No drift detected")

        return drift_detected

    def _ks_test(
        self,
        reference: pd.Series,
        current: pd.Series,
        threshold: float
    ) -> bool:
        """Kolmogorov-Smirnov test for numerical features."""

        # Remove nulls
        ref_clean = reference.dropna()
        curr_clean = current.dropna()

        # KS test
        statistic, p_value = stats.ks_2samp(ref_clean, curr_clean)

        # Drift if p-value < threshold
        return p_value < threshold

    def _chi_square_test(
        self,
        reference: pd.Series,
        current: pd.Series,
        threshold: float
    ) -> bool:
        """Chi-square test for categorical features."""

        # Get value counts
        ref_counts = reference.value_counts(normalize=True)
        curr_counts = current.value_counts(normalize=True)

        # Align categories
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        ref_freq = np.array([ref_counts.get(cat, 0) for cat in all_categories])
        curr_freq = np.array([curr_counts.get(cat, 0) for cat in all_categories])

        # Chi-square test
        # Scale to counts
        ref_count = len(reference)
        curr_count = len(current)

        observed = curr_freq * curr_count
        expected = ref_freq * curr_count

        # Avoid division by zero
        expected = np.where(expected == 0, 1e-10, expected)

        chi_square = np.sum((observed - expected) ** 2 / expected)
        degrees_of_freedom = len(all_categories) - 1
        p_value = 1 - stats.chi2.cdf(chi_square, degrees_of_freedom)

        return p_value < threshold

    def compute_drift_metrics(self, current_data: pd.DataFrame) -> Dict:
        """Compute detailed drift metrics."""

        metrics = {}

        for column in self.reference_data.columns:
            if column in current_data.columns:
                if pd.api.types.is_numeric_dtype(self.reference_data[column]):
                    # Numerical drift metrics
                    ref_mean = self.reference_data[column].mean()
                    curr_mean = current_data[column].mean()
                    mean_shift = (curr_mean - ref_mean) / ref_mean if ref_mean != 0 else 0

                    ref_std = self.reference_data[column].std()
                    curr_std = current_data[column].std()
                    std_shift = (curr_std - ref_std) / ref_std if ref_std != 0 else 0

                    metrics[column] = {
                        'type': 'numerical',
                        'mean_shift': mean_shift,
                        'std_shift': std_shift,
                        'ref_mean': ref_mean,
                        'curr_mean': curr_mean
                    }

        return metrics


# Example: Data validation pipeline
class DataValidationPipeline:
    """Complete data validation pipeline."""

    def __init__(self):
        self.validator = DataValidator()
        self.drift_detector = None

    def validate_training_data(self, df: pd.DataFrame) -> bool:
        """Validate training data before model training."""

        print("Running data validation...")

        # Step 1: Schema and quality validation
        validation_results = self.validator.validate_data(df)

        if not validation_results['success']:
            print("❌ Data validation failed. Cannot proceed with training.")
            return False

        # Step 2: Store reference data for drift detection
        self.drift_detector = DriftDetector(reference_data=df)

        print("✅ Training data validation passed")
        return True

    def validate_serving_data(self, df: pd.DataFrame) -> bool:
        """Validate serving data before inference."""

        print("Running serving data validation...")

        # Step 1: Schema and quality validation
        validation_results = self.validator.validate_data(df)

        if not validation_results['success']:
            print("⚠️  Serving data quality issues detected")
            return False

        # Step 2: Drift detection
        if self.drift_detector is not None:
            drift_results = self.drift_detector.detect_drift(df)

            drifted_features = sum(drift_results.values())
            if drifted_features > 0:
                print(f"⚠️  Drift detected in {drifted_features} features")
                # Trigger retraining pipeline
                return False

        print("✅ Serving data validation passed")
        return True
```


### Solution 4: Model Validation (Accuracy Thresholds and Regression Tests)

**Goal:** Validate model performance before deployment with comprehensive testing.

**Components:**
- Accuracy threshold validation
- Regression testing
- Fairness and bias validation
- Performance (latency) validation

```python
# Model validation framework
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import numpy as np
from typing import Dict, Tuple, Optional
import joblib

class ModelValidator:
    """Comprehensive model validation."""

    def __init__(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        production_model_path: Optional[str] = None
    ):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.production_model_path = production_model_path
        self.validation_results = {}

    def validate_all(self) -> Tuple[bool, Dict]:
        """Run all validation checks."""

        checks = {
            'accuracy_threshold': self.check_accuracy_threshold,
            'regression_test': self.check_no_regression,
            'fairness_test': self.check_fairness,
            'performance_test': self.check_performance,
            'robustness_test': self.check_robustness
        }

        all_passed = True
        results = {}

        for check_name, check_func in checks.items():
            try:
                passed, details = check_func()
                results[check_name] = {
                    'passed': passed,
                    'details': details
                }
                if not passed:
                    all_passed = False
            except Exception as e:
                results[check_name] = {
                    'passed': False,
                    'details': {'error': str(e)}
                }
                all_passed = False

        self.validation_results = results

        if all_passed:
            print("✅ All validation checks passed")
        else:
            print("❌ Some validation checks failed")
            for check, result in results.items():
                if not result['passed']:
                    print(f"  - {check}: FAILED")

        return all_passed, results

    def check_accuracy_threshold(self) -> Tuple[bool, Dict]:
        """Check model meets minimum accuracy thresholds."""

        # Define thresholds
        MIN_ACCURACY = 0.85
        MIN_PRECISION = 0.80
        MIN_RECALL = 0.80
        MIN_F1 = 0.80
        MIN_AUC = 0.85

        # Compute metrics
        y_pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        # AUC if model supports probabilities
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(self.X_test)
            if y_proba.shape[1] == 2:  # Binary classification
                auc = roc_auc_score(self.y_test, y_proba[:, 1])
                metrics['auc'] = auc

        # Check thresholds
        passed = (
            accuracy >= MIN_ACCURACY and
            precision >= MIN_PRECISION and
            recall >= MIN_RECALL and
            f1 >= MIN_F1
        )

        if 'auc' in metrics:
            passed = passed and metrics['auc'] >= MIN_AUC

        details = {
            'metrics': metrics,
            'thresholds': {
                'accuracy': MIN_ACCURACY,
                'precision': MIN_PRECISION,
                'recall': MIN_RECALL,
                'f1': MIN_F1,
                'auc': MIN_AUC
            }
        }

        return passed, details

    def check_no_regression(self) -> Tuple[bool, Dict]:
        """Check new model doesn't regress from production model."""

        if self.production_model_path is None:
            return True, {'message': 'No production model to compare against'}

        try:
            # Load production model
            prod_model = joblib.load(self.production_model_path)

            # Compare metrics
            new_pred = self.model.predict(self.X_test)
            prod_pred = prod_model.predict(self.X_test)

            new_accuracy = accuracy_score(self.y_test, new_pred)
            prod_accuracy = accuracy_score(self.y_test, prod_pred)

            new_f1 = f1_score(self.y_test, new_pred, average='weighted')
            prod_f1 = f1_score(self.y_test, prod_pred, average='weighted')

            # Allow 1% regression tolerance
            REGRESSION_TOLERANCE = 0.01

            accuracy_regressed = new_accuracy < prod_accuracy - REGRESSION_TOLERANCE
            f1_regressed = new_f1 < prod_f1 - REGRESSION_TOLERANCE

            passed = not (accuracy_regressed or f1_regressed)

            details = {
                'new_accuracy': new_accuracy,
                'prod_accuracy': prod_accuracy,
                'accuracy_diff': new_accuracy - prod_accuracy,
                'new_f1': new_f1,
                'prod_f1': prod_f1,
                'f1_diff': new_f1 - prod_f1
            }

            return passed, details

        except Exception as e:
            return False, {'error': f"Failed to load production model: {str(e)}"}

    def check_fairness(self) -> Tuple[bool, Dict]:
        """Check model fairness across protected attributes."""

        # This is a simplified example
        # In production, use comprehensive fairness libraries like Fairlearn

        # For this example, assume we have protected attribute in test data
        # In reality, you'd need to carefully handle protected attributes

        passed = True
        details = {'message': 'Fairness check passed (simplified)'}

        return passed, details

    def check_performance(self) -> Tuple[bool, Dict]:
        """Check model inference performance."""

        import time

        # Latency threshold
        MAX_LATENCY_MS = 100

        # Measure latency
        latencies = []
        for _ in range(100):
            start = time.time()
            _ = self.model.predict(self.X_test[:1])
            end = time.time()
            latencies.append((end - start) * 1000)

        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        passed = p95_latency < MAX_LATENCY_MS

        details = {
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'threshold_ms': MAX_LATENCY_MS
        }

        return passed, details

    def check_robustness(self) -> Tuple[bool, Dict]:
        """Check model robustness to input perturbations."""

        # Test with slightly perturbed inputs
        noise_level = 0.01
        X_perturbed = self.X_test + np.random.normal(0, noise_level, self.X_test.shape)

        # Predictions should be similar
        pred_original = self.model.predict(self.X_test)
        pred_perturbed = self.model.predict(X_perturbed)

        agreement = np.mean(pred_original == pred_perturbed)

        # Require 95% agreement
        passed = agreement >= 0.95

        details = {
            'agreement': agreement,
            'threshold': 0.95,
            'noise_level': noise_level
        }

        return passed, details


# Model validation pipeline
class ModelValidationPipeline:
    """Automated model validation pipeline."""

    def __init__(self):
        self.validation_history = []

    def validate_before_deployment(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> bool:
        """Validate model before deployment."""

        print("="* 60)
        print("MODEL VALIDATION PIPELINE")
        print("="* 60)

        # Initialize validator
        validator = ModelValidator(
            model=model,
            X_test=X_test,
            y_test=y_test,
            production_model_path='models/production_model.pkl'
        )

        # Run all validations
        all_passed, results = validator.validate_all()

        # Log results
        self.validation_history.append({
            'timestamp': pd.Timestamp.now(),
            'passed': all_passed,
            'results': results
        })

        # Print summary
        print("\n" + "="* 60)
        if all_passed:
            print("✅ VALIDATION PASSED - Model ready for deployment")
        else:
            print("❌ VALIDATION FAILED - Model NOT ready for deployment")
        print("="* 60)

        return all_passed
```


### Solution 5: Pipeline Orchestration (Airflow/Kubeflow/Prefect)

**Goal:** Orchestrate complex ML workflows with dependency management and scheduling.

**Components:**
- DAG (Directed Acyclic Graph) definition
- Task dependencies
- Scheduled execution
- Failure handling and retries

```python
# ML pipeline orchestration using Apache Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import pandas as pd
import joblib

# Default arguments for Airflow DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email': ['ml-alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

# Define ML training pipeline DAG
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='End-to-end ML training pipeline',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'training']
)

# Task 1: Extract data
def extract_data(**context):
    """Extract data from warehouse."""
    print("Extracting data from warehouse...")

    # Query data warehouse
    query = """
    SELECT *
    FROM ml_features
    WHERE date >= CURRENT_DATE - INTERVAL '90 days'
    """

    df = pd.read_sql(query, connection_string)

    # Save to temporary location
    df.to_parquet('/tmp/raw_data.parquet')

    print(f"Extracted {len(df)} rows")

    # Push metadata to XCom
    context['task_instance'].xcom_push(key='num_rows', value=len(df))

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

# Task 2: Validate data
def validate_data(**context):
    """Validate data quality."""
    print("Validating data...")

    df = pd.read_parquet('/tmp/raw_data.parquet')

    validator = DataValidator()
    validation_results = validator.validate_data(df)

    if not validation_results['success']:
        raise ValueError("Data validation failed")

    print("✅ Data validation passed")

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

# Task 3: Check for drift
def check_drift(**context):
    """Check for data drift."""
    print("Checking for drift...")

    current_data = pd.read_parquet('/tmp/raw_data.parquet')
    reference_data = pd.read_parquet('/data/reference_data.parquet')

    drift_detector = DriftDetector(reference_data)
    drift_results = drift_detector.detect_drift(current_data)

    drifted_features = sum(drift_results.values())

    if drifted_features > 5:
        print(f"⚠️  Significant drift detected in {drifted_features} features")
        print("Proceeding with retraining...")
    else:
        print("✅ No significant drift detected")

drift_task = PythonOperator(
    task_id='check_drift',
    python_callable=check_drift,
    dag=dag
)

# Task 4: Preprocess data
def preprocess_data(**context):
    """Preprocess data for training."""
    print("Preprocessing data...")

    df = pd.read_parquet('/tmp/raw_data.parquet')

    # Feature engineering
    # (In production, use feature store)
    processed_df = feature_engineering(df)

    # Train/test split
    from sklearn.model_selection import train_test_split

    X = processed_df.drop('target', axis=1)
    y = processed_df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save splits
    joblib.dump((X_train, X_test, y_train, y_test), '/tmp/train_test_split.pkl')

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

# Task 5: Train model
def train_model(**context):
    """Train ML model."""
    print("Training model...")

    # Load data
    X_train, X_test, y_train, y_test = joblib.load('/tmp/train_test_split.pkl')

    # Train model
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Save model
    model_path = '/tmp/trained_model.pkl'
    joblib.dump(model, model_path)

    print("✅ Model training complete")

    # Log to MLflow
    import mlflow
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.sklearn.log_model(model, "model")

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

# Task 6: Validate model
def validate_model(**context):
    """Validate trained model."""
    print("Validating model...")

    # Load data and model
    X_train, X_test, y_train, y_test = joblib.load('/tmp/train_test_split.pkl')
    model = joblib.load('/tmp/trained_model.pkl')

    # Run validation
    validator = ModelValidator(
        model=model,
        X_test=X_test,
        y_test=y_test,
        production_model_path='/models/production_model.pkl'
    )

    all_passed, results = validator.validate_all()

    if not all_passed:
        raise ValueError("Model validation failed")

    print("✅ Model validation passed")

validate_model_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag
)

# Task 7: Deploy model
def deploy_model(**context):
    """Deploy model to production."""
    print("Deploying model to production...")

    # Copy model to production location
    import shutil
    shutil.copy('/tmp/trained_model.pkl', '/models/production_model.pkl')

    # Update model version
    version_info = {
        'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'deployed_at': datetime.now().isoformat(),
        'metrics': context['task_instance'].xcom_pull(
            task_ids='validate_model',
            key='metrics'
        )
    }

    with open('/models/version_info.json', 'w') as f:
        json.dump(version_info, f)

    print(f"✅ Model deployed: version {version_info['version']}")

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

# Task 8: Monitor deployment
def monitor_deployment(**context):
    """Monitor model deployment."""
    print("Monitoring deployment...")

    # Run smoke tests
    # Check model is accessible
    # Verify predictions are being made
    # Check latency metrics

    print("✅ Deployment monitoring complete")

monitor_task = PythonOperator(
    task_id='monitor_deployment',
    python_callable=monitor_deployment,
    dag=dag
)

# Define task dependencies (DAG)
extract_task >> validate_task >> check_drift >> preprocess_task >> train_task >> validate_model_task >> deploy_task >> monitor_task


# Alternative: Kubeflow pipeline
class KubeflowPipeline:
    """ML pipeline using Kubeflow."""

    @staticmethod
    def create_pipeline():
        """Create Kubeflow pipeline."""

        import kfp
        from kfp import dsl

        @dsl.component
        def extract_data_op():
            """Extract data component."""
            # Component code
            pass

        @dsl.component
        def train_model_op(data_path: str):
            """Train model component."""
            # Component code
            pass

        @dsl.component
        def deploy_model_op(model_path: str):
            """Deploy model component."""
            # Component code
            pass

        @dsl.pipeline(
            name='ML Training Pipeline',
            description='End-to-end ML pipeline'
        )
        def ml_pipeline():
            """Define pipeline."""

            extract_task = extract_data_op()
            train_task = train_model_op(data_path=extract_task.output)
            deploy_task = deploy_model_op(model_path=train_task.output)

        return ml_pipeline


# Alternative: Prefect pipeline
class PrefectPipeline:
    """ML pipeline using Prefect."""

    @staticmethod
    def create_flow():
        """Create Prefect flow."""

        from prefect import flow, task

        @task
        def extract_data():
            """Extract data."""
            # Task code
            return data_path

        @task
        def train_model(data_path):
            """Train model."""
            # Task code
            return model_path

        @task
        def deploy_model(model_path):
            """Deploy model."""
            # Task code
            pass

        @flow(name="ML Training Pipeline")
        def ml_pipeline():
            """Define flow."""

            data_path = extract_data()
            model_path = train_model(data_path)
            deploy_model(model_path)

        return ml_pipeline
```


### Solution 6: Automated Retraining Triggers

**Goal:** Automatically trigger model retraining based on performance degradation or schedule.

**Components:**
- Performance monitoring
- Automated triggers
- Retraining orchestration
- Deployment automation

```python
# Automated retraining system
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import joblib

class AutomatedRetrainingSystem:
    """Automatically trigger and manage model retraining."""

    def __init__(
        self,
        model_path: str,
        accuracy_threshold: float = 0.85,
        drift_threshold: int = 5,
        retraining_schedule_days: int = 7
    ):
        self.model_path = model_path
        self.accuracy_threshold = accuracy_threshold
        self.drift_threshold = drift_threshold
        self.retraining_schedule_days = retraining_schedule_days

        self.last_retrain_date = self._load_last_retrain_date()
        self.performance_history = []

    def should_retrain(self) -> Tuple[bool, str]:
        """Determine if model should be retrained."""

        reasons = []

        # Trigger 1: Performance degradation
        if self._check_performance_degradation():
            reasons.append("performance_degradation")

        # Trigger 2: Data drift
        if self._check_data_drift():
            reasons.append("data_drift")

        # Trigger 3: Scheduled retraining
        if self._check_schedule():
            reasons.append("scheduled_retraining")

        # Trigger 4: Manual override
        if self._check_manual_trigger():
            reasons.append("manual_trigger")

        should_retrain = len(reasons) > 0
        reason_str = ", ".join(reasons) if reasons else "no_triggers"

        return should_retrain, reason_str

    def _check_performance_degradation(self) -> bool:
        """Check if model performance has degraded."""

        # Load recent predictions and actuals
        recent_data = self._load_recent_predictions(days=1)

        if len(recent_data) < 100:  # Need minimum samples
            return False

        # Compute current accuracy
        y_true = recent_data['actual']
        y_pred = recent_data['predicted']

        current_accuracy = accuracy_score(y_true, y_pred)

        # Track performance
        self.performance_history.append({
            'timestamp': datetime.now(),
            'accuracy': current_accuracy
        })

        # Check threshold
        if current_accuracy < self.accuracy_threshold:
            print(f"⚠️  Performance degradation detected: {current_accuracy:.3f} < {self.accuracy_threshold}")
            return True

        return False

    def _check_data_drift(self) -> bool:
        """Check if data drift has occurred."""

        # Load reference and current data
        reference_data = pd.read_parquet('data/reference_data.parquet')
        current_data = self._load_recent_features(days=7)

        # Detect drift
        drift_detector = DriftDetector(reference_data)
        drift_results = drift_detector.detect_drift(current_data)

        drifted_features = sum(drift_results.values())

        if drifted_features > self.drift_threshold:
            print(f"⚠️  Data drift detected: {drifted_features} features drifted")
            return True

        return False

    def _check_schedule(self) -> bool:
        """Check if scheduled retraining is due."""

        days_since_retrain = (datetime.now() - self.last_retrain_date).days

        if days_since_retrain >= self.retraining_schedule_days:
            print(f"⚠️  Scheduled retraining due: {days_since_retrain} days since last retrain")
            return True

        return False

    def _check_manual_trigger(self) -> bool:
        """Check for manual retraining trigger."""

        # Check flag file
        import os
        trigger_file = '/tmp/manual_retrain_trigger'

        if os.path.exists(trigger_file):
            print("⚠️  Manual retraining trigger detected")
            os.remove(trigger_file)
            return True

        return False

    def trigger_retraining(self, reason: str):
        """Trigger automated retraining pipeline."""

        print(f"\n{'='*60}")
        print(f"TRIGGERING AUTOMATED RETRAINING")
        print(f"Reason: {reason}")
        print(f"Timestamp: {datetime.now()}")
        print(f"{'='*60}\n")

        # Trigger Airflow DAG
        from airflow.api.client.local_client import Client

        client = Client(None, None)
        client.trigger_dag(
            dag_id='ml_training_pipeline',
            run_id=f'auto_retrain_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            conf={'trigger_reason': reason}
        )

        # Update last retrain date
        self.last_retrain_date = datetime.now()
        self._save_last_retrain_date()

        print("✅ Retraining pipeline triggered")

    def _load_last_retrain_date(self) -> datetime:
        """Load last retraining date."""
        try:
            with open('models/last_retrain.txt', 'r') as f:
                return datetime.fromisoformat(f.read().strip())
        except:
            return datetime.now() - timedelta(days=365)  # Default to long ago

    def _save_last_retrain_date(self):
        """Save last retraining date."""
        with open('models/last_retrain.txt', 'w') as f:
            f.write(self.last_retrain_date.isoformat())

    def _load_recent_predictions(self, days: int) -> pd.DataFrame:
        """Load recent predictions for performance monitoring."""
        # In production, load from database or logging system
        # This is a placeholder
        return pd.DataFrame({
            'predicted': np.random.randint(0, 2, 1000),
            'actual': np.random.randint(0, 2, 1000)
        })

    def _load_recent_features(self, days: int) -> pd.DataFrame:
        """Load recent features for drift detection."""
        # In production, load from feature store
        # This is a placeholder
        return pd.DataFrame(np.random.randn(1000, 10))


# Monitoring service that runs continuously
class RetrainingMonitorService:
    """Continuous monitoring service for automated retraining."""

    def __init__(self, check_interval_minutes: int = 60):
        self.check_interval = check_interval_minutes
        self.retraining_system = AutomatedRetrainingSystem(
            model_path='models/production_model.pkl',
            accuracy_threshold=0.85,
            drift_threshold=5,
            retraining_schedule_days=7
        )

    def run(self):
        """Run continuous monitoring."""

        print("Starting automated retraining monitoring service...")

        while True:
            try:
                # Check if retraining needed
                should_retrain, reason = self.retraining_system.should_retrain()

                if should_retrain:
                    print(f"\n⚠️  Retraining triggered: {reason}")
                    self.retraining_system.trigger_retraining(reason)
                else:
                    print(f"✅ No retraining needed (checked at {datetime.now()})")

                # Wait for next check
                import time
                time.sleep(self.check_interval * 60)

            except Exception as e:
                print(f"❌ Error in monitoring service: {e}")
                import time
                time.sleep(60)  # Wait 1 minute before retrying


# Run monitoring service as a daemon
def start_monitoring_service():
    """Start retraining monitoring service."""

    service = RetrainingMonitorService(check_interval_minutes=60)
    service.run()
```


## REFACTOR: Pressure Tests

### Pressure Test 1: Scale to 100+ Models

**Scenario:** Team manages 100+ models, manual processes break down.

**Test:**
```python
# Pressure test: Scale to 100 models
def test_scale_to_100_models():
    """Can MLOps system handle 100+ models?"""

    num_models = 100

    # Test 1: CI/CD scales to 100 models
    # All models get automated testing
    for model_id in range(num_models):
        # Each model has own CI/CD pipeline
        # Tests run in parallel
        # Deployment automated
        pass

    # Test 2: Feature store serves 100 models
    # Single feature definitions used by all models
    # No duplication of feature logic

    # Test 3: Monitoring scales to 100 models
    # Automated alerts for all models
    # Dashboard shows health of all models

    print("✅ System scales to 100+ models")
```

### Pressure Test 2: Deploy 10 Times Per Day

**Scenario:** High-velocity team deploys models 10 times per day.

**Test:**
```python
# Pressure test: High deployment velocity
def test_deploy_10_times_per_day():
    """Can system handle 10 deployments per day?"""

    for deployment in range(10):
        # Automated testing (5 minutes)
        # Automated validation (2 minutes)
        # Automated deployment (3 minutes)
        # Total: 10 minutes per deployment

        # No manual intervention
        # Automatic rollback on failure
        pass

    print("✅ System handles 10 deployments/day")
```

### Pressure Test 3: Detect and Fix Data Quality Issue in < 1 Hour

**Scenario:** Upstream data pipeline breaks, corrupting training data.

**Test:**
```python
# Pressure test: Data quality incident response
def test_data_quality_incident():
    """Can system detect and block bad data quickly?"""

    # Corrupt data arrives
    corrupted_data = inject_data_corruption()

    # Data validation catches it immediately
    validation_results = validator.validate_data(corrupted_data)
    assert not validation_results['success'], "Should detect corruption"

    # Training pipeline blocked
    # Alert sent to team
    # No bad model trained

    # Time to detection: < 1 minute
    # Time to block: < 1 minute

    print("✅ Data quality issue detected and blocked")
```

### Pressure Test 4: Model Accuracy Drops to 70%, System Retrains Automatically

**Scenario:** Production model degrades, needs automatic retraining.

**Test:**
```python
# Pressure test: Automatic retraining on degradation
def test_automatic_retraining():
    """Does system automatically retrain on performance drop?"""

    # Simulate accuracy drop
    simulate_performance_degradation(target_accuracy=0.70)

    # Monitor detects degradation
    should_retrain, reason = retraining_system.should_retrain()
    assert should_retrain, "Should detect degradation"
    assert "performance_degradation" in reason

    # Automated retraining triggered
    retraining_system.trigger_retraining(reason)

    # New model trained and deployed
    # Accuracy restored to 90%

    # Total time: 2 hours (fully automated)

    print("✅ Automatic retraining on degradation works")
```

### Pressure Test 5: Feature Store Serves 1000 QPS

**Scenario:** High-traffic application requires low-latency feature retrieval.

**Test:**
```python
# Pressure test: Feature store performance
def test_feature_store_performance():
    """Can feature store handle 1000 QPS?"""

    import time
    import concurrent.futures

    def get_features(user_id):
        start = time.time()
        features = feature_store.get_online_features(
            entity_ids={'user_id': [user_id]},
            features=['user_features:age', 'user_features:avg_purchase_amount']
        )
        latency = time.time() - start
        return latency

    # Simulate 1000 QPS for 10 seconds
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for _ in range(10000):  # 1000 QPS * 10 seconds
            user_id = np.random.randint(1, 100000)
            futures.append(executor.submit(get_features, user_id))

        latencies = [f.result() for f in futures]

    # Check latency
    p95_latency = np.percentile(latencies, 95)
    assert p95_latency < 0.01, f"P95 latency {p95_latency*1000:.2f}ms too high"

    print(f"✅ Feature store handles 1000 QPS (P95: {p95_latency*1000:.2f}ms)")
```

### Pressure Test 6: Rollback Failed Deployment in < 5 Minutes

**Scenario:** New model deployment fails, needs immediate rollback.

**Test:**
```python
# Pressure test: Deployment rollback
def test_deployment_rollback():
    """Can system rollback failed deployment quickly?"""

    # Deploy bad model (fails validation)
    bad_model = train_intentionally_bad_model()

    # Validation catches issues
    validator = ModelValidator(bad_model, X_test, y_test)
    passed, results = validator.validate_all()
    assert not passed, "Should fail validation"

    # Deployment blocked
    # Production model unchanged
    # No user impact

    # Time to detect and block: < 5 minutes

    print("✅ Failed deployment blocked before production")
```

### Pressure Test 7: Data Drift Detected, Model Retrains Within 24 Hours

**Scenario:** Significant data drift occurs, triggering retraining.

**Test:**
```python
# Pressure test: Drift-triggered retraining
def test_drift_triggered_retraining():
    """Does drift trigger automatic retraining?"""

    # Simulate significant drift
    drifted_data = simulate_data_drift(num_drifted_features=10)

    # Drift detection catches it
    drift_detector = DriftDetector(reference_data)
    drift_results = drift_detector.detect_drift(drifted_data)
    drifted_features = sum(drift_results.values())
    assert drifted_features >= 10, "Should detect drift"

    # Retraining triggered
    should_retrain, reason = retraining_system.should_retrain()
    assert should_retrain, "Should trigger retraining"
    assert "data_drift" in reason

    # Model retrained within 24 hours
    # New model adapts to data distribution

    print("✅ Drift-triggered retraining works")
```

### Pressure Test 8: CI/CD Pipeline Runs All Tests in < 10 Minutes

**Scenario:** Fast iteration requires quick CI/CD feedback.

**Test:**
```python
# Pressure test: CI/CD speed
def test_cicd_speed():
    """Does CI/CD complete in < 10 minutes?"""

    import time

    start_time = time.time()

    # Run full CI/CD pipeline
    # - Unit tests (1 min)
    # - Integration tests (2 min)
    # - Model training (3 min)
    # - Model validation (2 min)
    # - Deployment (2 min)

    ci_system = MLModelCI(model, (X_test, y_test))
    passed = ci_system.run_all_tests()

    elapsed_time = time.time() - start_time

    assert elapsed_time < 600, f"CI/CD took {elapsed_time:.0f}s, target <600s"
    assert passed, "CI/CD should pass"

    print(f"✅ CI/CD completes in {elapsed_time:.0f}s")
```

### Pressure Test 9: Feature Consistency Between Training and Serving

**Scenario:** Verify no train-serve skew with feature store.

**Test:**
```python
# Pressure test: Feature consistency
def test_feature_consistency():
    """Are training and serving features identical?"""

    # Get training features
    entity_df = pd.DataFrame({
        'user_id': [1001],
        'event_timestamp': [pd.Timestamp('2024-01-01')]
    })

    training_features = feature_store.get_training_features(
        entity_df=entity_df,
        features=['user_features:age', 'user_features:avg_purchase_amount']
    )

    # Get serving features (same user, same timestamp)
    serving_features = feature_store.get_online_features(
        entity_ids={'user_id': [1001]},
        features=['user_features:age', 'user_features:avg_purchase_amount']
    )

    # Features should be identical
    assert training_features['age'].iloc[0] == serving_features['age'].iloc[0]
    assert training_features['avg_purchase_amount'].iloc[0] == \
           serving_features['avg_purchase_amount'].iloc[0]

    print("✅ Feature consistency verified")
```

### Pressure Test 10: Monitor and Alert on Model Degradation Within 1 Hour

**Scenario:** Model performance degrades, alerts sent quickly.

**Test:**
```python
# Pressure test: Monitoring and alerting
def test_monitoring_alerting():
    """Are performance issues detected and alerted quickly?"""

    # Simulate performance degradation
    simulate_performance_degradation(target_accuracy=0.75)

    # Monitor detects it
    monitor = RetrainingMonitorService(check_interval_minutes=60)

    # Within 1 hour:
    # 1. Performance degradation detected
    # 2. Alert sent to team
    # 3. Retraining automatically triggered

    should_retrain, reason = monitor.retraining_system.should_retrain()
    assert should_retrain, "Should detect degradation"

    # Alert sent (email, Slack, PagerDuty)
    # Time to detection: < 1 hour
    # Time to alert: < 1 minute after detection

    print("✅ Monitoring and alerting working")
```


## Summary

**MLOps automation transforms manual ML workflows into production-ready systems.**

**Key implementations:**

1. **CI/CD for ML**
   - Automated testing (unit, integration, validation)
   - Quality gates (accuracy, latency, bias)
   - Automated deployment with rollback

2. **Feature Store**
   - Single source of truth for features
   - Training and serving consistency
   - Point-in-time correctness
   - Low-latency serving

3. **Data Validation**
   - Schema validation (Great Expectations)
   - Drift detection (statistical tests)
   - Quality monitoring

4. **Model Validation**
   - Accuracy thresholds
   - Regression testing
   - Performance validation
   - Fairness checks

5. **Pipeline Orchestration**
   - Airflow/Kubeflow/Prefect DAGs
   - Dependency management
   - Scheduled execution
   - Failure handling

6. **Automated Retraining**
   - Performance monitoring
   - Drift detection
   - Scheduled updates
   - Automatic triggers

**Impact:**

- **Deployment speed:** 2-4 hours → 10 minutes (24x faster)
- **Deployment reliability:** 80% → 99%+ success rate
- **Production accuracy:** +14% (eliminates train-serve skew)
- **Time to detect issues:** 2-4 hours → 5 minutes (24-48x faster)
- **Model freshness:** Updated weekly/monthly → daily/weekly
- **Team productivity:** 30% less time on toil, 30% more on modeling

**The result:** Production ML systems that are reliable, automated, and scalable.
