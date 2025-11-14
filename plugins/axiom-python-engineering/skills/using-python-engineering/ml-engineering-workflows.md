
# ML Engineering Workflows

## Overview

**Core Principle:** Experiments must be reproducible. Track everything: code, data, parameters, metrics, environment. Without reproducibility, ML experiments are just random number generation.

ML engineering is about systematic experimentation and production deployment. Track experiments with MLflow/Weights & Biases, manage configuration with Hydra, ensure reproducible data splits, monitor models in production. The biggest mistake: running experiments without tracking parameters or random seeds.

## When to Use

**Use this skill when:**
- "Track ML experiments"
- "MLflow setup"
- "Reproducible ML"
- "Model lifecycle"
- "Hyperparameter management"
- "ML monitoring"
- "ML project structure"
- "Experiment comparison"

**Don't use when:**
- Setting up Python project (use project-structure-and-tooling first)
- NumPy/pandas optimization (use scientific-computing-foundations)
- Profiling ML code (use debugging-and-profiling)

**Symptoms triggering this skill:**
- Can't reproduce results
- Lost track of which parameters produced which metrics
- Need to compare many experiments
- Deploying model to production


## Experiment Tracking with MLflow

### Basic MLflow Setup

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ❌ WRONG: Not tracking experiments
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {accuracy}")  # Lost forever after terminal closes

# ✅ CORRECT: Track with MLflow
mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    # Log parameters
    params = {"n_estimators": 100, "max_depth": 10}
    mlflow.log_params(params)

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Log metrics
    y_pred = model.predict(X_test)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='weighted'))

    # Log model
    mlflow.sklearn.log_model(model, "model")

# ✅ CORRECT: Log artifacts (plots, confusion matrix)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

with mlflow.start_run():
    mlflow.log_params(params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Create and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average='weighted')
    })
```

**Why this matters**: MLflow tracks all experiments with parameters and metrics. Can compare runs, reproduce results, and deploy best model.

### Nested Runs for Cross-Validation

```python
# ❌ WRONG: CV results not tracked properly
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f"Mean: {scores.mean()}")  # Lost context

# ✅ CORRECT: Track CV with nested runs
from sklearn.model_selection import KFold

with mlflow.start_run(run_name="rf_cv_experiment") as parent_run:
    mlflow.log_params(params)

    kf = KFold(n_splits=5, shuffle=True, random_seed=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train_fold, y_train_fold)

            score = accuracy_score(y_val_fold, model.predict(X_val_fold))
            cv_scores.append(score)

            mlflow.log_metric("accuracy", score)
            mlflow.log_metric("fold", fold)

    # Log aggregate metrics in parent run
    mlflow.log_metric("cv_mean_accuracy", np.mean(cv_scores))
    mlflow.log_metric("cv_std_accuracy", np.std(cv_scores))
```

### Hyperparameter Tuning with Tracking

```python
from sklearn.model_selection import GridSearchCV

# ❌ WRONG: GridSearchCV without tracking
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)  # Only get best, lose all other trials

# ✅ CORRECT: Track all hyperparameter trials
with mlflow.start_run(run_name="grid_search"):
    for n_est in [50, 100, 200]:
        for max_d in [5, 10, 20]:
            with mlflow.start_run(nested=True):
                params = {"n_estimators": n_est, "max_depth": max_d}
                mlflow.log_params(params)

                model = RandomForestClassifier(**params, random_state=42)
                model.fit(X_train, y_train)

                score = accuracy_score(y_test, model.predict(X_test))
                mlflow.log_metric("accuracy", score)

# ✅ BETTER: Use MLflow with Optuna for Bayesian optimization
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
    }

    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

mlflc = MLflowCallback(tracking_uri="mlruns", metric_name="accuracy")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, callbacks=[mlflc])
```

**Why this matters**: Hyperparameter tuning generates many experiments. Tracking all trials enables comparison and understanding of parameter importance.


## Configuration Management with Hydra

### Basic Hydra Configuration

```python
# ❌ WRONG: Hardcoded parameters
def train():
    learning_rate = 0.001
    batch_size = 32
    epochs = 100
    # What if we want to try different values? Edit code each time?

# ✅ CORRECT: Hydra configuration
# File: config.yaml
"""
model:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100

data:
  train_path: data/train.csv
  test_path: data/test.csv
"""

# File: train.py
import hydra
from omegaconf import DictConfig

@hydra.main(config_path=".", config_name="config", version_base=None)
def train(cfg: DictConfig):
    print(f"Learning rate: {cfg.model.learning_rate}")
    print(f"Batch size: {cfg.model.batch_size}")

    # Access config values
    model = create_model(
        lr=cfg.model.learning_rate,
        batch_size=cfg.model.batch_size
    )

if __name__ == "__main__":
    train()

# Run with overrides:
# python train.py model.learning_rate=0.01 model.batch_size=64
```

### Structured Configs with Dataclasses

```python
# ✅ CORRECT: Type-safe configs with dataclasses
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    hidden_dim: int = 256

@dataclass
class DataConfig:
    train_path: str = "data/train.csv"
    test_path: str = "data/test.csv"
    val_split: float = 0.2

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()

cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(config_path=None, config_name="config", version_base=None)
def train(cfg: Config):
    # Type hints work!
    lr: float = cfg.model.learning_rate
    batch_size: int = cfg.model.batch_size

if __name__ == "__main__":
    train()
```

**Why this matters**: Hydra enables command-line overrides without code changes. Structured configs provide type safety and IDE autocomplete.

### Multi-Run Sweeps

```python
# ✅ CORRECT: Hydra multirun for hyperparameter sweeps
# config.yaml
"""
defaults:
  - override hydra/launcher: basic

model:
  learning_rate: 0.001
  batch_size: 32
"""

# Run multiple experiments:
# python train.py -m model.learning_rate=0.001,0.01,0.1 model.batch_size=32,64,128
# Creates 9 runs (3 x 3)

@hydra.main(config_path=".", config_name="config", version_base=None)
def train(cfg: DictConfig):
    with mlflow.start_run():
        mlflow.log_params({
            "learning_rate": cfg.model.learning_rate,
            "batch_size": cfg.model.batch_size
        })

        model = train_model(cfg)
        metrics = evaluate_model(model, test_data)
        mlflow.log_metrics(metrics)
```


## Reproducibility Best Practices

### Random Seed Management

```python
import random
import numpy as np
import torch

# ❌ WRONG: No random seed
model = create_model()
model.fit(X_train, y_train)
# Different results every run!

# ✅ CORRECT: Set all random seeds
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full reproducibility with CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# In training script
set_seed(42)
model = create_model()
model.fit(X_train, y_train)

# ✅ CORRECT: Track seed in MLflow
with mlflow.start_run():
    seed = 42
    mlflow.log_param("random_seed", seed)
    set_seed(seed)
    # ... training code ...
```

### Reproducible Data Splits

```python
# ❌ WRONG: Non-reproducible split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Different split every time!

# ✅ CORRECT: Fixed random seed for splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ BETTER: Hash-based deterministic split (stable across runs)
import hashlib

def hash_split(df, test_size=0.2, id_column='id'):
    """Deterministic split based on ID hash."""
    def test_set_check(identifier, test_size):
        hash_val = int(hashlib.md5(str(identifier).encode()).hexdigest(), 16)
        return hash_val % 100 < test_size * 100

    is_test = df[id_column].apply(lambda x: test_set_check(x, test_size))
    return df[~is_test], df[is_test]

train_df, test_df = hash_split(df, test_size=0.2, id_column='user_id')
# Same split even if data order changes or new rows added
```

### Environment Reproducibility

```python
# ✅ CORRECT: Log environment info
import mlflow
import sys
import platform

with mlflow.start_run():
    # Log Python version
    mlflow.log_param("python_version", sys.version)

    # Log package versions
    import sklearn
    import pandas
    import numpy
    mlflow.log_params({
        "sklearn_version": sklearn.__version__,
        "pandas_version": pandas.__version__,
        "numpy_version": numpy.__version__,
    })

    # Log system info
    mlflow.log_params({
        "platform": platform.platform(),
        "cpu_count": os.cpu_count()
    })

# ✅ BETTER: Use conda/docker for full reproducibility
# conda env export > environment.yml
# Log environment file as artifact
with mlflow.start_run():
    mlflow.log_artifact("environment.yml")
```

**Why this matters**: Reproducibility requires controlling all randomness sources. Different package versions or Python versions can produce different results.


## Data Versioning and Lineage

### Data Versioning with DVC

```bash
# Initialize DVC
dvc init

# Track large data files
dvc add data/train.csv
git add data/train.csv.dvc data/.gitignore
git commit -m "Track training data"

# Configure remote storage (S3, GCS, Azure, etc.)
dvc remote add -d myremote s3://mybucket/dvcstore
dvc push

# Retrieve specific version
git checkout v1.0
dvc pull
```

### Logging Data Info in MLflow

```python
# ✅ CORRECT: Log data characteristics
import pandas as pd
import mlflow

with mlflow.start_run():
    # Load data
    df = pd.read_csv("data/train.csv")

    # Log data info
    mlflow.log_params({
        "n_samples": len(df),
        "n_features": len(df.columns),
        "class_balance": df['target'].value_counts().to_dict(),
        "data_version": "v1.0",  # Track data version
        "data_hash": hashlib.md5(df.to_csv(index=False).encode()).hexdigest()
    })

    # Log sample of data
    df.head(100).to_csv("data_sample.csv", index=False)
    mlflow.log_artifact("data_sample.csv")
```

### Feature Engineering Pipeline Tracking

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ✅ CORRECT: Track entire preprocessing pipeline
with mlflow.start_run():
    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)),
        ('classifier', RandomForestClassifier(n_estimators=100))
    ])

    # Log pipeline parameters
    mlflow.log_params({
        "scaler": "StandardScaler",
        "pca_components": 50,
        "classifier": "RandomForestClassifier",
        "n_estimators": 100
    })

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Log entire pipeline
    mlflow.sklearn.log_model(pipeline, "model_pipeline")

    # Evaluate
    score = pipeline.score(X_test, y_test)
    mlflow.log_metric("accuracy", score)
```


## Model Lifecycle Management

### Model Registry

```python
# ✅ CORRECT: Register model in MLflow
with mlflow.start_run() as run:
    model = train_model(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")

    # Register model
    model_uri = f"runs:/{run.info.run_id}/model"
    mlflow.register_model(model_uri, "my_model")

# ✅ CORRECT: Promote model to production
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get latest version
latest_version = client.get_latest_versions("my_model", stages=["None"])[0]

# Transition to staging
client.transition_model_version_stage(
    name="my_model",
    version=latest_version.version,
    stage="Staging"
)

# After testing, promote to production
client.transition_model_version_stage(
    name="my_model",
    version=latest_version.version,
    stage="Production"
)

# ✅ CORRECT: Load production model
model = mlflow.pyfunc.load_model(
    model_uri="models:/my_model/Production"
)
predictions = model.predict(X_new)
```

### Model Metadata and Tags

```python
# ✅ CORRECT: Add tags for searchability
with mlflow.start_run() as run:
    mlflow.set_tag("model_type", "random_forest")
    mlflow.set_tag("task", "classification")
    mlflow.set_tag("dataset", "customer_churn")
    mlflow.set_tag("owner", "data_science_team")

    # Train and log model
    model = train_model(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")

    # Add version tag
    mlflow.set_tag("version", "v2.1.0")

# Search for runs with tags
from mlflow.tracking import MlflowClient
client = MlflowClient()
runs = client.search_runs(
    experiment_ids=["0"],
    filter_string="tags.model_type = 'random_forest' AND metrics.accuracy > 0.85"
)
```


## Metrics and Logging

### Structured Logging

```python
import logging
import mlflow

# ✅ CORRECT: Structured logging with MLflow
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

with mlflow.start_run():
    logger.info("Starting training")
    mlflow.log_param("learning_rate", 0.001)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader)
        val_loss = validate(model, val_loader)

        # Log metrics per epoch
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)

        logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Early stopping check
        if val_loss > best_val_loss:
            patience_counter += 1
            if patience_counter > patience:
                logger.info(f"Early stopping at epoch {epoch}")
                mlflow.set_tag("early_stopped", "true")
                mlflow.log_param("stopped_epoch", epoch)
                break
```

### Custom Metrics

```python
from sklearn.metrics import make_scorer

# ✅ CORRECT: Define and log custom metrics
def business_metric(y_true, y_pred):
    """Custom metric: cost of false positives vs false negatives."""
    fp_cost = 10  # Cost of false positive
    fn_cost = 100  # Cost of false negative

    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    return fp * fp_cost + fn * fn_cost

with mlflow.start_run():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Log standard metrics
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    })

    # Log custom business metric
    cost = business_metric(y_test, y_pred)
    mlflow.log_metric("business_cost", cost)
```

### Metric Visualization

```python
# ✅ CORRECT: Log plots and visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

with mlflow.start_run():
    # Train model
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")
    plt.close()

    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    mlflow.log_metric("roc_auc", roc_auc)
```


## Production Monitoring

### Model Performance Monitoring

```python
# ✅ CORRECT: Monitor model performance in production
import mlflow
from datetime import datetime

class ModelMonitor:
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.mlflow_client = MlflowClient()

    def log_prediction(self, features, prediction, actual=None):
        """Log prediction for monitoring."""
        with mlflow.start_run(run_name=f"prediction_{datetime.now().isoformat()}"):
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("model_version", self.model_version)

            # Log feature statistics
            mlflow.log_params({
                f"feature_{i}_mean": float(features[:, i].mean())
                for i in range(features.shape[1])
            })

            # Log prediction
            mlflow.log_metric("prediction", float(prediction))

            # If actual available (for online evaluation)
            if actual is not None:
                mlflow.log_metric("actual", float(actual))
                mlflow.log_metric("error", abs(float(prediction - actual)))

    def check_data_drift(self, current_data, reference_data):
        """Detect data drift using KS test."""
        from scipy.stats import ks_2samp

        drift_detected = False
        drift_features = []

        with mlflow.start_run(run_name="drift_check"):
            for i in range(current_data.shape[1]):
                stat, p_value = ks_2samp(
                    reference_data[:, i],
                    current_data[:, i]
                )

                mlflow.log_metric(f"feature_{i}_ks_stat", stat)
                mlflow.log_metric(f"feature_{i}_p_value", p_value)

                if p_value < 0.05:  # Significant drift
                    drift_detected = True
                    drift_features.append(i)

            mlflow.log_param("drift_detected", drift_detected)
            mlflow.log_param("drift_features", drift_features)

        return drift_detected, drift_features
```

### Alerting and Anomaly Detection

```python
# ✅ CORRECT: Monitor for anomalies in predictions
class PredictionMonitor:
    def __init__(self, threshold_std: float = 3.0):
        self.threshold_std = threshold_std
        self.recent_predictions = []
        self.window_size = 1000

    def check_anomaly(self, prediction: float) -> bool:
        """Check if prediction is anomalous."""
        self.recent_predictions.append(prediction)

        if len(self.recent_predictions) > self.window_size:
            self.recent_predictions.pop(0)

        if len(self.recent_predictions) < 100:
            return False  # Not enough data

        mean = np.mean(self.recent_predictions)
        std = np.std(self.recent_predictions)

        z_score = abs(prediction - mean) / std

        is_anomaly = z_score > self.threshold_std

        # Log to MLflow
        mlflow.log_metrics({
            "prediction": prediction,
            "rolling_mean": mean,
            "rolling_std": std,
            "z_score": z_score,
            "is_anomaly": int(is_anomaly)
        })

        return is_anomaly
```


## ML Project Structure

### Standard Project Layout

```
ml_project/
├── data/
│   ├── raw/              # Original immutable data
│   ├── processed/        # Cleaned, transformed data
│   └── features/         # Engineered features
├── notebooks/            # Exploratory notebooks
│   └── eda.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py       # Data loading
│   │   └── preprocess.py # Preprocessing
│   ├── features/
│   │   ├── __init__.py
│   │   └── build.py      # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py      # Training
│   │   ├── predict.py    # Inference
│   │   └── evaluate.py   # Evaluation
│   └── utils/
│       ├── __init__.py
│       └── config.py     # Configuration
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── configs/              # Hydra configs
│   ├── config.yaml
│   ├── model/
│   │   ├── rf.yaml
│   │   └── xgboost.yaml
│   └── data/
│       └── default.yaml
├── mlruns/              # MLflow tracking
├── outputs/             # Hydra outputs
├── requirements.txt
├── setup.py
└── README.md
```

### Makefile for Common Tasks

```makefile
# ✅ CORRECT: Makefile for reproducible workflows
.PHONY: data features train evaluate

data:
	python src/data/load.py
	python src/data/preprocess.py

features: data
	python src/features/build.py

train: features
	python src/models/train.py

evaluate: train
	python src/models/evaluate.py

clean:
	rm -rf data/processed/*
	rm -rf mlruns/*

test:
	pytest tests/

lint:
	ruff check src/
	mypy src/
```


## Integration Patterns

### MLflow + Hydra Integration

```python
# ✅ CORRECT: Combine MLflow tracking with Hydra config
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    # Set MLflow experiment
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run():
        # Log all Hydra config as parameters
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        # Log Hydra config file as artifact
        config_path = ".hydra/config.yaml"
        mlflow.log_artifact(config_path)

        # Train model
        model = create_model(cfg.model)
        model.fit(X_train, y_train)

        # Log metrics
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    train()
```

### Continuous Training Pipeline

```python
# ✅ CORRECT: Automated retraining pipeline
from datetime import datetime
import mlflow

def continuous_training_pipeline():
    """Retrain model if performance degrades."""
    # Load production model
    prod_model = mlflow.pyfunc.load_model("models:/my_model/Production")

    # Load recent data
    recent_data = load_recent_data()

    # Evaluate production model on recent data
    prod_metrics = evaluate_model(prod_model, recent_data)

    # Check if retraining needed
    if prod_metrics['accuracy'] < 0.85:  # Threshold
        print("Performance degraded, retraining...")

        with mlflow.start_run(run_name=f"retrain_{datetime.now().isoformat()}"):
            # Log why retraining
            mlflow.set_tag("retrain_reason", "accuracy_below_threshold")
            mlflow.log_metric("prod_accuracy", prod_metrics['accuracy'])

            # Train new model
            new_model = train_model(load_training_data())

            # Evaluate new model
            new_metrics = evaluate_model(new_model, recent_data)
            mlflow.log_metrics(new_metrics)

            # If better, register and promote
            if new_metrics['accuracy'] > prod_metrics['accuracy']:
                mlflow.sklearn.log_model(new_model, "model")

                # Register new version
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                model_version = mlflow.register_model(model_uri, "my_model")

                # Promote to production
                client = MlflowClient()
                client.transition_model_version_stage(
                    name="my_model",
                    version=model_version.version,
                    stage="Production",
                    archive_existing_versions=True
                )
```


## Anti-Patterns

### Not Tracking Experiments

```python
# ❌ WRONG: No tracking
for lr in [0.001, 0.01, 0.1]:
    model = train_model(lr)
    print(f"LR={lr}, Accuracy={evaluate(model)}")
# Which LR was best? Lost after terminal closes.

# ✅ CORRECT: Track everything
for lr in [0.001, 0.01, 0.1]:
    with mlflow.start_run():
        mlflow.log_param("learning_rate", lr)
        model = train_model(lr)
        acc = evaluate(model)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
```

### Non-Reproducible Data Splits

```python
# ❌ WRONG: Random split without seed
X_train, X_test = train_test_split(X, y, test_size=0.2)
# Different split every run!

# ✅ CORRECT: Fixed seed
X_train, X_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Hardcoded Paths and Parameters

```python
# ❌ WRONG: Hardcoded values
data = pd.read_csv("/home/user/data/train.csv")
model = RandomForestClassifier(n_estimators=100, max_depth=10)

# ✅ CORRECT: Config-driven
@hydra.main(config_path=".", config_name="config", version_base=None)
def train(cfg: DictConfig):
    data = pd.read_csv(cfg.data.train_path)
    model = RandomForestClassifier(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth
    )
```


## Decision Trees

### Should I Track This Experiment?

```
Is this a throwaway experiment?
├─ Yes (just testing) → Maybe skip tracking
└─ No → ALWAYS TRACK
    ├─ Comparing models → Track
    ├─ Tuning hyperparameters → Track
    ├─ Production candidate → Track
    └─ Debugging → Track (helps identify issues)
```

### When to Register a Model?

```
Is model for production use?
├─ Yes → Register in model registry
│   ├─ Test in staging first
│   └─ Promote to production after validation
└─ No (experiment only) → Log but don't register
```


## Integration with Other Skills

**After using this skill:**
- If profiling ML code → See @debugging-and-profiling
- If optimizing data processing → See @scientific-computing-foundations
- If setting up CI/CD → See @project-structure-and-tooling

**Before using this skill:**
- If setting up project → Use @project-structure-and-tooling first
- If data processing slow → Use @scientific-computing-foundations to optimize


## Quick Reference

### MLflow Essential Commands

```python
# Start run
with mlflow.start_run():
    mlflow.log_param("param_name", value)
    mlflow.log_metric("metric_name", value)
    mlflow.log_artifact("file.png")
    mlflow.sklearn.log_model(model, "model")

# Register model
mlflow.register_model("runs:/<run_id>/model", "model_name")

# Load model
model = mlflow.pyfunc.load_model("models:/model_name/Production")
```

### Hydra Essential Patterns

```python
# Basic config
@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(cfg.param)

# Override from CLI
# python script.py param=value

# Multirun
# python script.py -m param=1,2,3
```

### Reproducibility Checklist

- [ ] Set random seeds (Python, NumPy, PyTorch)
- [ ] Use fixed random_state in train_test_split
- [ ] Track data version/hash
- [ ] Log package versions
- [ ] Track preprocessing steps
- [ ] Version control code
- [ ] Use config files (don't hardcode)
