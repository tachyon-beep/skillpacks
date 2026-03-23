
# Experiment Tracking and Versioning Skill

## When to Use This Skill

Use this skill when you observe these symptoms:

**Reproducibility Symptoms:**
- Cannot reproduce a good result from last week (which hyperparameters?)
- Someone asks "which model is in production?" and you do not know
- Lost track of which data version produced which model
- Experiments tracked in spreadsheets or text files (manual, error-prone)

**Collaboration Symptoms:**
- Multiple people running experiments, no central tracking
- Cannot compare runs across team members
- Lost experiments when someone leaves the team
- No visibility into what others are trying

**Production Symptoms:**
- Cannot trace predictions back to model version and training data
- Need to roll back model but do not know which previous version was good
- Compliance requires audit trail (data → model → predictions)
- Cannot A/B test models because tracking unclear

**When NOT to use this skill:**
- Single experiment, one-off analysis (no need for tracking infrastructure)
- Prototyping where reproducibility not yet important
- Already have robust experiment tracking system working well

## Core Principle

**If you cannot reproduce it, it does not exist.**

Experiment tracking captures everything needed to reproduce a result:
- **Code version** (git commit hash)
- **Data version** (dataset hash, version tag)
- **Hyperparameters** (learning rate, batch size, etc.)
- **Environment** (Python version, library versions)
- **Random seeds** (for deterministic results)
- **Metrics** (accuracy, loss over time)
- **Artifacts** (model checkpoints, predictions)

**Formula:** Good tracking = Code + Data + Config + Environment + Seeds + Metrics + Artifacts

The skill is building a system where **every experiment is automatically reproducible**.

## Experiment Tracking Framework

```
┌────────────────────────────────────────────┐
│   1. Recognize Tracking Need               │
│   "Cannot reproduce" OR "Which model?"     │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│   2. Choose Tracking Tool                  │
│   MLflow (local) vs W&B (cloud+collab)     │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│   3. Instrument Training Code              │
│   Log params, metrics, artifacts           │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│   4. Version Models + Data                 │
│   Model registry + data versioning         │
└──────────────┬─────────────────────────────┘
               │
               ▼
┌────────────────────────────────────────────┐
│   5. Ensure Reproducibility                │
│   Validate can recreate any experiment     │
└────────────────────────────────────────────┘
```

## Part 1: RED - Without Experiment Tracking (5 Failures)

### Failure 1: Cannot Reproduce Best Run

**Scenario:** Training image classifier, got 94.2% accuracy last week. Today, best is 91.8%. Cannot figure out what changed.

**Without tracking:**

```python
# train_model.py - NO TRACKING VERSION

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def train_model():
    """
    Train model with no experiment tracking.

    FAILURE MODE: Cannot reproduce results.
    - Hyperparameters hardcoded or in head
    - No record of what produced 94.2% accuracy
    - Changed learning rate? batch size? data augmentation?
    - Lost forever if not documented manually
    """
    # Load data (which version? unknown)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Did we use this before?
        transforms.RandomCrop(32, padding=4),  # What padding last time?
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # Hyperparameters (were these the good ones?)
    batch_size = 128  # Or was it 64? 256?
    learning_rate = 0.001  # Or 0.01? 0.0001?
    epochs = 50  # Or 100?

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model (which architecture exactly?)
    model = models.resnet18(pretrained=True)  # Was it pretrained=True or False?
    model.fc = nn.Linear(model.fc.in_features, 10)

    # Optimizer (which one? what momentum?)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9  # Or 0.95?
    )

    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Print loss to terminal (lost after terminal closes)
        print(f"Epoch {epoch}: Loss = {loss.item()}")

    # Save model (no version information)
    torch.save(model.state_dict(), 'model_best.pth')  # Overwrites previous best!

    print("Training complete")

if __name__ == '__main__':
    train_model()
```

**Problems:**
- No record of hyperparameters that produced 94.2% accuracy
- Cannot compare current run to previous runs
- Terminal output lost after closing
- Manual notes (if any) error-prone and incomplete
- model_best.pth gets overwritten (lost previous version)

**Impact:** Wasted hours trying to reproduce good result. May never find it again.


### Failure 2: No Model Versioning (Which Model in Production?)

**Scenario:** Bug report from production. Customer service asks "which model is deployed?" No clear answer.

**Without versioning:**

```python
# deploy_model.py - NO VERSIONING

import torch
import shutil
from datetime import datetime

def deploy_model():
    """
    Deploy model without versioning.

    FAILURE MODE: Cannot identify which model is where.
    - No semantic versioning (v1.0.0, v1.1.0)
    - No metadata (when trained, on what data, by whom)
    - Cannot tell if production model is same as local model
    - Cannot roll back to previous version
    """
    # Which model file? (multiple candidates)
    model_candidates = [
        'model_best.pth',       # Best on validation set
        'model_final.pth',      # Last epoch
        'model_checkpoint.pth', # Some checkpoint
        'model_old.pth',        # Backup?
    ]

    # Pick one (guess which is best?)
    model_path = 'model_best.pth'

    # Copy to production (no version tag)
    production_path = '/models/production/model.pth'
    shutil.copy(model_path, production_path)

    # No metadata
    # - When was this model trained?
    # - What accuracy does it have?
    # - What data was it trained on?
    # - Can we roll back?

    print(f"Deployed {model_path} to production")
    # But wait... which exact version is this?
    # If there's a bug, how do we identify it?

def rollback_model():
    """
    Try to roll back model.

    FAILURE MODE: No previous versions saved.
    """
    # There's only one production model file
    # Previous version was overwritten
    # Cannot roll back!

    print("ERROR: No previous version to roll back to")

# Questions we cannot answer:
# 1. Which model version is in production?
# 2. When was it deployed?
# 3. What accuracy does it have?
# 4. What data was it trained on?
# 5. Can we compare to previous versions?
# 6. Can we roll back if needed?
```

**Problems:**
- No way to identify which model version is deployed
- Cannot roll back to previous version (overwritten)
- No metadata (accuracy, training date, data version)
- Audit trail missing (compliance issue)
- Cannot A/B test (need multiple tagged versions)

**Impact:** Production incident, cannot identify or rollback problematic model. Hours of debugging.


### Failure 3: Manual Artifact Management (Files Everywhere)

**Scenario:** Running multiple experiments, artifacts scattered across directories. Cannot find the model checkpoint from experiment 42.

**Without artifact management:**

```python
# experiment_runner.py - NO ARTIFACT MANAGEMENT

import os
import torch
from datetime import datetime

class ExperimentRunner:
    """
    Run experiments with manual file management.

    FAILURE MODE: Files scattered everywhere.
    - No consistent naming convention
    - No organization by experiment
    - Cannot find specific checkpoint
    - Disk space wasted on duplicates
    """

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

        # Where do files go? (inconsistent across experiments)
        self.save_dir = f"./experiments/{experiment_name}/"
        os.makedirs(self.save_dir, exist_ok=True)

    def save_checkpoint(self, model, epoch, metrics):
        """
        Save checkpoint with unclear naming.

        PROBLEMS:
        - Filename not descriptive (which epoch? which metrics?)
        - No metadata saved with checkpoint
        - Hard to find best checkpoint later
        """
        # Bad filename (not descriptive)
        filename = f"checkpoint_{epoch}.pth"
        path = os.path.join(self.save_dir, filename)

        torch.save(model.state_dict(), path)

        # Metrics not saved! (only in terminal)
        print(f"Saved checkpoint to {path}")

    def save_predictions(self, predictions, split):
        """Save predictions (no link to model version)."""
        filename = f"predictions_{split}.npy"
        path = os.path.join(self.save_dir, filename)

        import numpy as np
        np.save(path, predictions)

        # PROBLEM: Cannot trace predictions back to model version
        # If we have 10 prediction files, which model generated which?

# Result after 50 experiments:
# ./experiments/
#   exp1/
#     checkpoint_10.pth
#     checkpoint_20.pth
#     predictions_val.npy
#   exp2/
#     checkpoint_50.pth  (wait, is this epoch 50 or checkpoint index 50?)
#     model_best.pth
#     predictions.npy     (which split? val or test?)
#   exp3/
#     model_final.pth
#     model_best.pth      (which is actually best?)
#     checkpoint_100.pth
#     old_checkpoint.pth  (what is this?)
#   ...
#   exp50/
#     (where is exp42? deleted by accident?)

# Questions we cannot answer:
# 1. Which checkpoint has best validation accuracy?
# 2. Which experiment produced predictions_val.npy?
# 3. How much disk space are we wasting?
# 4. Can we safely delete old checkpoints?
# 5. How do we reproduce experiment 42?
```

**Problems:**
- Inconsistent file naming across experiments
- No metadata linking artifacts to runs
- Cannot find specific checkpoint without manual search
- Disk space wasted (no automatic cleanup)
- Artifacts lost when directories deleted

**Impact:** Hours wasted searching for files, confusion about which artifact is which.


### Failure 4: No Lineage Tracking (Data → Model → Predictions)

**Scenario:** Model performance degraded in production. Need to trace back: which data was used? Cannot reconstruct lineage.

**Without lineage tracking:**

```python
# production_pipeline.py - NO LINEAGE TRACKING

import torch
import pandas as pd
from datetime import datetime

def production_pipeline():
    """
    Run production pipeline without lineage tracking.

    FAILURE MODE: Cannot trace predictions to source.
    - Which data version was used?
    - Which model version made predictions?
    - Can we reproduce these predictions?
    - What if data or model changed?
    """
    # Load data (which version?)
    data = pd.read_csv('data/production_data.csv')  # File gets overwritten daily!

    # Preprocess (what transformations?)
    # ... (transformations not logged)

    # Load model (which version?)
    model = torch.load('models/production/model.pth')  # No version info!

    # Make predictions
    predictions = model(data)

    # Save predictions (no lineage)
    predictions.to_csv('predictions/output.csv')  # Overwrites previous!

    # No record of:
    # - Which data file (version, hash, timestamp)
    # - Which model version (training run, accuracy, date)
    # - Which preprocessing (code version, parameters)
    # - Link between predictions and inputs

# Questions we cannot answer when predictions are wrong:
# 1. Which data was used? (data/production_data.csv changes daily)
# 2. Which model version? (models/production/model.pth changes weekly)
# 3. Can we reproduce? (no record of inputs, model, or preprocessing)
# 4. When did model last change? (no audit log)
# 5. What was prediction quality? (no metrics logged)

class DataVersionTracker:
    """
    Attempt to track data versions manually.

    FAILURE MODE: Manual tracking is incomplete and error-prone.
    """

    def __init__(self):
        self.versions = {}  # In-memory only (lost on restart)

    def track_data(self, data_path):
        """Track data version manually."""
        import hashlib

        # Compute hash (expensive for large files)
        with open(data_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        # Store in memory (lost when process ends)
        self.versions[data_path] = {
            'hash': file_hash,
            'timestamp': datetime.now()
        }

        # NOT PERSISTED! Lost on restart.
        # NOT LINKED to model or predictions

        return file_hash

# Manual tracking fails because:
# - Easy to forget to call track_data()
# - Data not automatically linked to models
# - Metadata lost when process ends
# - No visualization or query interface
```

**Problems:**
- Cannot trace predictions back to data and model versions
- No automatic lineage capture (manual = unreliable)
- Compliance issues (cannot prove which data produced which predictions)
- Cannot reproduce production results
- Debugging production issues requires guesswork

**Impact:** Production debugging nightmare. May violate compliance requirements. Cannot reproduce issues.


### Failure 5: Cannot Compare Runs

**Scenario:** Tried 20 different hyperparameter settings. Which one was best? Need to manually check 20 log files.

**Without run comparison:**

```python
# hyperparameter_search.py - NO RUN COMPARISON

import torch
import itertools
from datetime import datetime

def hyperparameter_search():
    """
    Search hyperparameters without tracking.

    FAILURE MODE: Cannot compare runs systematically.
    - Results printed to terminal (lost)
    - No structured storage of metrics
    - Cannot sort by metric
    - Cannot visualize trends
    """
    # Hyperparameter grid
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]
    optimizers = ['sgd', 'adam']

    # Try all combinations
    for lr, bs, opt in itertools.product(learning_rates, batch_sizes, optimizers):
        print(f"\n{'='*50}")
        print(f"Running: LR={lr}, BS={bs}, Optimizer={opt}")
        print(f"{'='*50}")

        # Train model
        model = train_model_with_params(lr, bs, opt)

        # Evaluate
        accuracy = evaluate_model(model)

        # Print results (LOST after terminal closes)
        print(f"Accuracy: {accuracy:.4f}")

        # Maybe save to file? (manual, error-prone)
        with open('results.txt', 'a') as f:  # Append to text file
            f.write(f"{lr},{bs},{opt},{accuracy}\n")

    print("\nSearch complete! Now manually parse results.txt to find best...")

# After 20 runs, results.txt looks like:
# 0.001,32,sgd,0.8234
# 0.001,32,adam,0.8456
# 0.001,64,sgd,0.8312
# ...
#
# Questions we cannot answer easily:
# 1. Which hyperparameters gave best accuracy?
#    (need to manually parse and sort)
# 2. How did accuracy change over epochs?
#    (not logged per-epoch)
# 3. Which optimizer works best for each learning rate?
#    (need to write custom analysis script)
# 4. Any correlation between batch size and accuracy?
#    (need to create scatter plots manually)
# 5. Can we visualize learning curves?
#    (would need to log per-epoch, plot manually)

def analyze_results_manually():
    """Manually analyze results from text file."""
    import pandas as pd

    # Parse text file (fragile)
    df = pd.read_csv('results.txt',
                     names=['lr', 'bs', 'opt', 'acc'],
                     header=None)

    # Find best (simple)
    best = df.loc[df['acc'].idxmax()]
    print(f"Best: LR={best['lr']}, BS={best['bs']}, Opt={best['opt']}, Acc={best['acc']}")

    # But cannot:
    # - See learning curves (not logged)
    # - Compare training time (not logged)
    # - Check if overfit (no validation metrics)
    # - Reproduce best run (hyperparameters not complete)
    # - Share results with team (no web UI)
```

**Problems:**
- Results scattered across terminal output and text files
- Cannot easily compare metrics across runs
- No visualization (learning curves, distributions)
- Cannot filter/sort runs by metric
- No structured query interface
- Results not shareable (text files, not web UI)

**Impact:** Wasted time manually parsing logs. Cannot make data-driven decisions. Poor hyperparameter selection.


### RED Summary: The Cost of No Tracking

**Time wasted per week:**
- Searching for lost experiments: 2-4 hours
- Manually comparing runs: 1-2 hours
- Reproducing previous results: 3-5 hours
- Identifying production models: 30-60 minutes
- Finding and organizing files: 1-2 hours

**Total:** 7.5-13.5 hours per week wasted on manual tracking

**Risks:**
- Cannot reproduce published results (reputation damage)
- Cannot identify production models (compliance violation)
- Lost best experiments (wasted training cost)
- Team confusion (duplicated effort)
- Production incidents (no rollback capability)

**The problem:** Without systematic tracking, ML development becomes archaeology. You spend more time searching for what you did than doing new work.

**Example failure cascade:**

```python
# Day 1: Train model, get 96.5% accuracy
# "I'll remember these hyperparameters"

# Day 7: Try to reproduce
# "Was it learning_rate=0.01 or 0.001?"
# "Did I use pretrained=True?"
# "Which data augmentation did I use?"

# Day 30: Someone asks "what's your best accuracy?"
# "I think it was 96.5%... or was it 94.5%?"
# "Let me search my terminal history..."
# "Oops, I cleared it last week"

# Day 90: Paper deadline
# "Claimed 96.5% in abstract"
# "Cannot reproduce, best now is 93.2%"
# "Withdraw paper or publish unreproducible results?"

# Day 180: Production incident
# "Which model version is deployed?"
# "model_best.pth was overwritten 5 times"
# "No idea which one is in production"
# "Cannot roll back, previous version lost"

# Total cost: Wasted weeks, damaged credibility, compliance risk
```

**Common excuses and why they fail:**

1. **"I'll write it down in a notebook"**
   - Reality: Notebooks get lost, incomplete, not searchable
   - What's missing: Automatic tracking, artifact links

2. **"I'll use descriptive filenames"**
   - Reality: model_lr0.01_bs128_acc94.2.pth grows to 50+ files
   - What's missing: Metadata, comparison UI, version history

3. **"I'll commit to git"**
   - Reality: Git not designed for large model files
   - What's missing: Model versioning, metric tracking, visualization

4. **"I'll remember important experiments"**
   - Reality: Memory fades, especially after 100+ experiments
   - What's missing: Durable, searchable record

5. **"It's just me, don't need formal tracking"**
   - Reality: Future you is a different person who forgot past you's decisions
   - What's missing: Documentation for future self

**The solution:** Systematic experiment tracking (MLflow, W&B) makes reproducibility automatic instead of manual.


### Bonus RED Example: The Compliance Nightmare

**Scenario:** Regulated industry (healthcare, finance) requires full audit trail. Auditor asks: "For prediction made on patient X on 2025-10-15, prove which model version and data were used."

**Without tracking (compliance failure):**

```python
# production_inference.py - NO AUDIT TRAIL

def make_prediction(patient_data):
    """
    Make prediction without audit trail.

    COMPLIANCE VIOLATION:
    - Cannot prove which model was used
    - Cannot prove which data was used
    - Cannot reproduce prediction
    - No timestamp, no version, no lineage
    """
    # Load model (which version? when trained? by whom?)
    model = torch.load('production_model.pth')

    # Make prediction
    prediction = model(patient_data)

    # Save result (no audit metadata)
    save_to_database(patient_id, prediction)

    # MISSING:
    # - Model version ID
    # - Model training date
    # - Model accuracy on validation set
    # - Data preprocessing version
    # - Prediction timestamp
    # - Link to training run
    # - Link to data version

    return prediction

# Auditor questions we CANNOT answer:
# 1. "Which model version made this prediction?"
#    Answer: "We have model.pth but no version info"
#
# 2. "What was this model's validation accuracy?"
#    Answer: "Not sure, we didn't save that"
#
# 3. "Can you reproduce this exact prediction?"
#    Answer: "Maybe, if we still have the same model file"
#
# 4. "When was this model trained and by whom?"
#    Answer: "We'd have to check git logs and emails..."
#
# 5. "What data was this model trained on?"
#    Answer: "Probably the data in the data/ folder?"
#
# 6. "Has the model been updated since this prediction?"
#    Answer: "Yes, several times, but we overwrote the file"
#
# 7. "Show me the full lineage from training data to this prediction"
#    Answer: "We don't have that information"

# RESULT: Compliance violation, potential regulatory fine, project shutdown

def audit_trail_attempt():
    """
    Attempt to create audit trail manually.

    FAILURE: Manual tracking is incomplete and unreliable.
    """
    # Try to piece together audit trail after the fact
    audit_log = {
        'model_file': 'production_model.pth',
        'file_size': os.path.getsize('production_model.pth'),
        'file_modified': os.path.getmtime('production_model.pth'),
        # But:
        # - File has been overwritten 5 times (lost history)
        # - No link to training run
        # - No validation metrics
        # - No data version
        # - No code version
        # - Timestamps are file system timestamps (unreliable)
    }

    # This audit trail is insufficient for compliance
    return audit_log

# Cost of compliance failure:
# - Regulatory fines: $100,000+
# - Project shutdown until compliant
# - Reputation damage
# - Legal liability
# - Cannot deploy to production in regulated industry

# The problem: Compliance requires proof, not promises.
# Manual tracking = no proof = compliance failure
```

**Problems:**
- No model version tracking (which exact model made prediction?)
- No lineage tracking (cannot trace back to training data)
- No audit timestamps (when was model trained, deployed, used?)
- No metadata (accuracy, training details, responsible party)
- Cannot reproduce predictions (no saved inputs, model version unclear)
- File overwrites destroy history (cannot recover previous versions)

**Impact:** Regulatory non-compliance, potential fines, project cancellation, legal liability.

**The solution:** Experiment tracking with full lineage provides automatic compliance audit trail.


## Part 2: GREEN - With Experiment Tracking (Solutions)

### Solution 1: MLflow for Local Tracking

**When to use MLflow:**
- Single user or small team
- Want to run tracking server locally
- Need model registry
- Self-hosted infrastructure
- Open source requirement

**MLflow setup:**

```python
# mlflow_setup.py - Install and start MLflow

"""
MLflow installation and basic setup.

WHY MLflow:
- Open source, self-hosted
- Good model registry
- Integrates with PyTorch, TensorFlow, scikit-learn
- Simple API
- Can run locally or on server

Installation:
    pip install mlflow

Start server:
    mlflow server --host 0.0.0.0 --port 5000

UI: http://localhost:5000
"""

import mlflow
import mlflow.pytorch

# Set tracking URI (local or remote)
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment name (organizes runs)
mlflow.set_experiment("image-classification")

print("MLflow configured. Access UI at http://localhost:5000")
```

**MLflow-instrumented training:**

```python
# train_with_mlflow.py - PROPER TRACKING VERSION

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import mlflow
import mlflow.pytorch
from pathlib import Path
import hashlib

def compute_data_hash(dataset_path: Path) -> str:
    """
    Compute hash of dataset for versioning.

    WHY: Ensures we know exactly which data was used.
    Different data = different results.
    """
    # Hash dataset directory or file
    import hashlib
    hash_md5 = hashlib.md5()

    for file_path in sorted(dataset_path.rglob('*')):
        if file_path.is_file():
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)

    return hash_md5.hexdigest()

def train_model():
    """
    Train model with MLflow tracking.

    SOLUTION: All experiment data logged automatically.
    - Hyperparameters
    - Metrics (loss, accuracy per epoch)
    - Artifacts (model checkpoints, plots)
    - Code version (git commit)
    - Data version (hash)
    - Environment (Python, library versions)
    """
    # Configure MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("cifar10-classification")

    # Start MLflow run
    with mlflow.start_run(run_name="resnet18-experiment") as run:

        # 1. LOG HYPERPARAMETERS
        # WHY: Need to reproduce later
        hyperparams = {
            'batch_size': 128,
            'learning_rate': 0.001,
            'epochs': 50,
            'optimizer': 'sgd',
            'momentum': 0.9,
            'model_arch': 'resnet18',
            'pretrained': True,
            'image_size': 32,
        }

        mlflow.log_params(hyperparams)

        # 2. LOG CODE VERSION
        # WHY: Need to know which code produced these results
        import subprocess
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']
            ).decode('ascii').strip()
            mlflow.log_param('git_commit', git_commit)
        except:
            mlflow.log_param('git_commit', 'unknown')

        # 3. LOG DATA VERSION
        # WHY: Different data = different results
        data_path = Path('./data/cifar10')
        data_hash = compute_data_hash(data_path)
        mlflow.log_param('data_hash', data_hash)
        mlflow.log_param('data_path', str(data_path))

        # 4. LOG ENVIRONMENT
        # WHY: Library versions affect results
        import torch
        mlflow.log_param('pytorch_version', torch.__version__)
        mlflow.log_param('cuda_available', torch.cuda.is_available())

        # 5. SET RANDOM SEEDS (REPRODUCIBILITY)
        # WHY: Makes training deterministic
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        mlflow.log_param('random_seed', seed)

        # Load data
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )

        val_dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=hyperparams['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=hyperparams['batch_size'],
            shuffle=False
        )

        # Model
        model = models.resnet18(pretrained=hyperparams['pretrained'])
        model.fc = nn.Linear(model.fc.in_features, 10)

        # Optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hyperparams['learning_rate'],
            momentum=hyperparams['momentum']
        )

        criterion = nn.CrossEntropyLoss()

        # 6. TRAINING LOOP WITH METRIC LOGGING
        best_val_acc = 0.0

        for epoch in range(hyperparams['epochs']):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()

            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    loss = criterion(output, target)

                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total

            # 7. LOG METRICS PER EPOCH
            # WHY: Can plot learning curves, detect overfitting
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, step=epoch)

            print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

            # 8. SAVE CHECKPOINTS AS ARTIFACTS
            # WHY: Can resume training, compare different checkpoints
            if val_acc > best_val_acc:
                best_val_acc = val_acc

                # Save model checkpoint
                checkpoint_path = f"checkpoints/best_model_epoch_{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'hyperparams': hyperparams,
                }, checkpoint_path)

                # Log to MLflow
                mlflow.log_artifact(checkpoint_path)
                mlflow.log_metric('best_val_acc', best_val_acc)

        # 9. LOG FINAL MODEL
        # WHY: Easy to load and deploy
        mlflow.pytorch.log_model(model, "model")

        # 10. LOG MODEL TO REGISTRY
        # WHY: Versioning, staging (dev/staging/production)
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "cifar10-resnet18")

        print(f"\n{'='*60}")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"View results: http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        print(f"{'='*60}")

        return run.info.run_id

# NOW WE CAN:
# 1. Reproduce any run (all hyperparams, code version, data version logged)
# 2. Compare runs in UI (sort by accuracy, visualize learning curves)
# 3. Download model artifacts (checkpoints, final model)
# 4. Track which model is in production (model registry)
# 5. Roll back to previous version (registry has all versions)

if __name__ == '__main__':
    train_model()
```

**Benefits of MLflow tracking:**
- All hyperparameters logged automatically
- Metrics logged per-epoch (can plot learning curves)
- Artifacts saved (model checkpoints, plots)
- Code version captured (git commit)
- Data version captured (hash)
- Environment captured (Python, PyTorch versions)
- Can reproduce any experiment
- Web UI for browsing and comparing runs
- Model registry for versioning and deployment


### Solution 2: Weights & Biases for Collaboration

**When to use W&B:**
- Team collaboration (multiple people)
- Want hosted solution (no server management)
- Need advanced visualization
- Real-time monitoring during training
- Want to share results with stakeholders

**W&B setup:**

```python
# wandb_setup.py - Install and configure W&B

"""
Weights & Biases installation and setup.

WHY W&B:
- Cloud-hosted (no server management)
- Beautiful visualizations
- Real-time monitoring
- Team collaboration
- Easy sharing (send link to stakeholders)
- Free tier for individuals

Installation:
    pip install wandb

Login:
    wandb login
    (Enter API key from https://wandb.ai/authorize)
"""

import wandb

# Login (do once)
# wandb.login()

# Initialize run
wandb.init(
    project="cifar10-classification",
    name="resnet18-baseline",
    config={
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 128,
    }
)

print("W&B configured. View runs at https://wandb.ai")
```

**W&B-instrumented training:**

```python
# train_with_wandb.py - W&B TRACKING VERSION

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import wandb
from pathlib import Path

def train_model():
    """
    Train model with W&B tracking.

    SOLUTION: Real-time monitoring and team collaboration.
    - Live training visualization (see metrics update in real-time)
    - Automatic system metrics (GPU usage, memory)
    - Beautiful dashboards (compare runs visually)
    - Easy sharing (send link to team)
    """
    # 1. INITIALIZE WANDB
    config = {
        'batch_size': 128,
        'learning_rate': 0.001,
        'epochs': 50,
        'optimizer': 'sgd',
        'momentum': 0.9,
        'model_arch': 'resnet18',
        'pretrained': True,
        'random_seed': 42,
    }

    run = wandb.init(
        project="cifar10-classification",
        name="resnet18-baseline",
        config=config,
        tags=['resnet', 'baseline', 'cifar10'],  # For filtering
    )

    # 2. SET RANDOM SEEDS
    torch.manual_seed(config['random_seed'])
    torch.cuda.manual_seed_all(config['random_seed'])

    # 3. DATA
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    val_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_val
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # 4. MODEL
    model = models.resnet18(pretrained=config['pretrained'])
    model.fc = nn.Linear(model.fc.in_features, 10)

    # 5. WATCH MODEL (logs gradients and parameters)
    # WHY: Can detect gradient explosion, vanishing gradients
    wandb.watch(model, log='all', log_freq=100)

    # 6. OPTIMIZER
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum']
    )

    criterion = nn.CrossEntropyLoss()

    # 7. TRAINING LOOP
    best_val_acc = 0.0

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

            # Log per-batch metrics (optional, for detailed monitoring)
            if batch_idx % 50 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch_idx': batch_idx + epoch * len(train_loader),
                })

        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        # 8. LOG METRICS (appears in real-time on W&B dashboard)
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })

        print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")

        # 9. SAVE BEST MODEL
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # Save checkpoint
            checkpoint_path = f"checkpoints/best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)

            # 10. LOG ARTIFACT TO W&B
            # WHY: Linked to run, can download later
            artifact = wandb.Artifact(
                name=f"model-{run.id}",
                type='model',
                description=f"Best model from run {run.name}",
                metadata={
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'architecture': config['model_arch'],
                }
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)

            wandb.log({'best_val_acc': best_val_acc})

    # 11. SAVE FINAL MODEL
    final_model_path = "checkpoints/final_model.pth"
    torch.save(model.state_dict(), final_model_path)

    final_artifact = wandb.Artifact(
        name=f"final-model-{run.id}",
        type='model',
        description="Final model after all epochs"
    )
    final_artifact.add_file(final_model_path)
    wandb.log_artifact(final_artifact)

    # 12. CREATE SUMMARY METRICS
    # WHY: Shown in run table, easy to compare
    wandb.summary['best_val_acc'] = best_val_acc
    wandb.summary['final_train_acc'] = train_acc
    wandb.summary['total_params'] = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"W&B Run: {run.url}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}")

    wandb.finish()

# NOW WE CAN:
# 1. See training progress in real-time (no waiting for training to finish)
# 2. Compare runs visually (parallel coordinates, scatter plots)
# 3. Share results with team (send W&B link)
# 4. Track system metrics (GPU usage, memory)
# 5. Download model artifacts from any run
# 6. Filter runs by tags, hyperparameters, metrics

if __name__ == '__main__':
    train_model()
```

**Benefits of W&B:**
- Real-time visualization (see training progress live)
- Automatic system monitoring (GPU usage, memory, CPU)
- Beautiful dashboards (compare runs visually)
- Easy collaboration (share link with team)
- Hosted solution (no server management)
- Advanced features (hyperparameter sweeps, reports)


### Solution 3: Model Versioning with Model Registry

**Model registry solves:**
- Which model is in production?
- What are all previous versions?
- Can we roll back?
- What metadata for each version?

**MLflow Model Registry:**

```python
# model_registry.py - Model versioning with MLflow

import mlflow
from mlflow.tracking import MlflowClient

class ModelRegistry:
    """
    Manage model versions with MLflow Model Registry.

    SOLUTION: Clear model versioning and lifecycle.
    - Semantic versioning (v1, v2, v3)
    - Staging labels (dev, staging, production)
    - Metadata (accuracy, training date, data version)
    - Rollback capability
    """

    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_model(self, run_id: str, model_name: str, description: str = ""):
        """
        Register model from training run.

        WHY: Creates versioned model in registry.
        Each registration creates new version (v1, v2, v3).
        """
        model_uri = f"runs:/{run_id}/model"

        # Register model (creates new version)
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags={
                'run_id': run_id,
                'description': description,
            }
        )

        print(f"Registered {model_name} version {model_version.version}")
        return model_version

    def add_model_metadata(self, model_name: str, version: int, metadata: dict):
        """
        Add metadata to model version.

        WHY: Track accuracy, data version, training details.
        """
        for key, value in metadata.items():
            self.client.set_model_version_tag(
                name=model_name,
                version=str(version),
                key=key,
                value=str(value)
            )

        print(f"Added metadata to {model_name} v{version}")

    def transition_to_staging(self, model_name: str, version: int):
        """
        Move model to staging.

        WHY: Indicates model ready for testing in staging environment.
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False  # Keep old staging versions
        )

        print(f"Transitioned {model_name} v{version} to Staging")

    def transition_to_production(self, model_name: str, version: int):
        """
        Move model to production.

        WHY: Indicates model deployed to production.
        Archives previous production version (can roll back).
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True  # Archive old production version
        )

        print(f"Transitioned {model_name} v{version} to Production")

    def get_production_model(self, model_name: str):
        """
        Get current production model.

        WHY: Load model for serving.
        """
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pytorch.load_model(model_uri)

        # Get version info
        versions = self.client.search_model_versions(f"name='{model_name}'")
        prod_version = [v for v in versions if v.current_stage == 'Production'][0]

        print(f"Loaded {model_name} v{prod_version.version} (Production)")
        return model, prod_version

    def rollback_production(self, model_name: str, target_version: int):
        """
        Roll back production to previous version.

        WHY: Quick recovery from bad deployment.
        """
        # Move target version to production
        self.transition_to_production(model_name, target_version)

        print(f"Rolled back {model_name} to v{target_version}")

    def list_model_versions(self, model_name: str):
        """
        List all versions of a model.

        WHY: See history, compare versions.
        """
        versions = self.client.search_model_versions(f"name='{model_name}'")

        for v in versions:
            print(f"Version {v.version}: {v.current_stage}")
            print(f"  Created: {v.creation_timestamp}")
            print(f"  Tags: {v.tags}")
            print()

        return versions

# Usage example
if __name__ == '__main__':
    registry = ModelRegistry()

    # After training, register model
    run_id = "abc123..."  # From MLflow training run
    model_version = registry.register_model(
        run_id=run_id,
        model_name="cifar10-resnet18",
        description="Baseline ResNet18 model"
    )

    # Add metadata
    registry.add_model_metadata(
        model_name="cifar10-resnet18",
        version=model_version.version,
        metadata={
            'val_acc': 94.2,
            'data_version': 'v1.0',
            'training_date': '2025-10-30',
            'trained_by': 'john',
        }
    )

    # Transition through stages
    registry.transition_to_staging("cifar10-resnet18", model_version.version)

    # After testing in staging, promote to production
    registry.transition_to_production("cifar10-resnet18", model_version.version)

    # Load production model for serving
    model, version_info = registry.get_production_model("cifar10-resnet18")

    # If production model has issues, roll back
    # registry.rollback_production("cifar10-resnet18", target_version=2)
```

**Model registry benefits:**
- Clear versioning (v1, v2, v3)
- Staging workflow (dev → staging → production)
- Metadata tracking (accuracy, data version, etc.)
- Rollback capability (revert to previous version)
- Audit trail (who deployed what when)


### Solution 4: Data Versioning

**When to version data:**
- Dataset changes over time
- Need to reproduce experiments with exact data
- Large datasets (cannot commit to git)

**Option A: DVC (Data Version Control)**

```bash
# Install and setup DVC
pip install dvc

# Initialize DVC in project
dvc init

# Add dataset to DVC
dvc add data/cifar10

# Commit DVC metadata to git
git add data/cifar10.dvc .gitignore
git commit -m "Add CIFAR-10 v1.0"
git tag data-v1.0

# Push data to remote storage (S3, GCS, etc.)
dvc remote add storage s3://my-bucket/dvc-store
dvc push

# Team members pull data
dvc pull

# Checkout specific version
git checkout data-v1.0
dvc checkout
```

**Option B: Hash-Based Versioning**

```python
# Simpler: Just compute and log data hash
import hashlib
from pathlib import Path

def compute_data_hash(data_path: Path) -> str:
    """Compute dataset hash for versioning."""
    hash_md5 = hashlib.md5()
    for file_path in sorted(data_path.rglob('*')):
        if file_path.is_file():
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()

# In training code
data_hash = compute_data_hash(Path('./data/cifar10'))
mlflow.log_param('data_hash', data_hash)  # Track data version
```

**Data versioning benefits:**
- Reproduce experiments with exact data
- Detect when data changes affect metrics
- Sync datasets across team
- Compliance audit trail


### Solution 5: Lineage Tracking (Data → Model → Predictions)

**Lineage tracking solves:**
- Which data produced which model?
- Which model made which predictions?
- Can we reproduce production predictions?
- Compliance and audit trail

**Lineage tracking implementation:**

```python
# lineage_tracking.py - Track full pipeline lineage

import mlflow
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class LineageTracker:
    """
    Track lineage from data to predictions.

    SOLUTION: Full traceability of ML pipeline.
    - Data version → Training run → Model version → Predictions
    - Can reproduce any step
    - Compliance-ready audit trail
    """

    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()

    def track_data_ingestion(self, data_path: Path, run_id: str) -> str:
        """
        Track data ingestion with hash and metadata.

        WHY: Links training run to specific data version.
        """
        # Compute data hash
        data_hash = self._compute_hash(data_path)

        # Log to MLflow
        with mlflow.start_run(run_id=run_id):
            mlflow.log_param('data_path', str(data_path))
            mlflow.log_param('data_hash', data_hash)
            mlflow.log_param('data_timestamp', datetime.now().isoformat())

        print(f"Tracked data: {data_path} (hash: {data_hash[:8]}...)")
        return data_hash

    def track_training(
        self,
        data_hash: str,
        hyperparams: Dict[str, Any],
        metrics: Dict[str, float],
        model_path: Path,
    ) -> str:
        """
        Track training run with lineage to data.

        WHY: Links model to training config and data version.
        """
        with mlflow.start_run() as run:
            # Link to data
            mlflow.log_param('data_hash', data_hash)

            # Log hyperparameters
            mlflow.log_params(hyperparams)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.log_artifact(str(model_path))

            # Compute model hash
            model_hash = self._compute_hash(model_path)
            mlflow.log_param('model_hash', model_hash)

            print(f"Tracked training run: {run.info.run_id}")
            return run.info.run_id

    def track_inference(
        self,
        model_version: str,
        input_data_hash: str,
        predictions_path: Path,
    ) -> str:
        """
        Track inference with lineage to model and data.

        WHY: Links predictions to model version and input data.
        Can reproduce predictions.
        """
        with mlflow.start_run(run_name=f"inference-{datetime.now().strftime('%Y%m%d-%H%M%S')}") as run:
            # Link to model
            mlflow.log_param('model_version', model_version)

            # Link to input data
            mlflow.log_param('input_data_hash', input_data_hash)

            # Log predictions
            mlflow.log_artifact(str(predictions_path))

            # Compute predictions hash
            predictions_hash = self._compute_hash(predictions_path)
            mlflow.log_param('predictions_hash', predictions_hash)
            mlflow.log_param('inference_timestamp', datetime.now().isoformat())

            print(f"Tracked inference: {run.info.run_id}")
            return run.info.run_id

    def get_lineage(self, run_id: str) -> Dict[str, Any]:
        """
        Get full lineage for a run.

        WHY: Trace back from predictions to source data.
        """
        run = self.client.get_run(run_id)

        lineage = {
            'run_id': run_id,
            'run_name': run.info.run_name,
            'start_time': datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
            'params': run.data.params,
            'metrics': run.data.metrics,
            'tags': run.data.tags,
        }

        # If this is inference run, trace back to training
        if 'model_version' in run.data.params:
            model_version = run.data.params['model_version']
            # Get training run that produced this model
            # (implementation depends on your model registry setup)
            lineage['model_lineage'] = {
                'model_version': model_version,
                # Add training run details here
            }

        return lineage

    def _compute_hash(self, path: Path) -> str:
        """Compute MD5 hash of file or directory."""
        hash_md5 = hashlib.md5()

        if path.is_file():
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        else:
            for file_path in sorted(path.rglob('*')):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)

        return hash_md5.hexdigest()

# Usage: Full pipeline with lineage tracking
def production_pipeline_with_lineage():
    """
    Production pipeline with full lineage tracking.

    SOLUTION: Every step tracked, fully reproducible.
    """
    tracker = LineageTracker()

    # 1. DATA INGESTION
    print("Step 1: Data Ingestion")
    data_path = Path('./data/production_data_2025-10-30.csv')
    data_hash = tracker.track_data_ingestion(data_path, run_id=None)

    # 2. TRAINING
    print("Step 2: Training")
    hyperparams = {
        'learning_rate': 0.001,
        'batch_size': 128,
        'epochs': 50,
    }
    metrics = {
        'val_acc': 94.2,
        'val_loss': 0.234,
    }
    model_path = Path('./models/model_20251030.pth')

    training_run_id = tracker.track_training(
        data_hash=data_hash,
        hyperparams=hyperparams,
        metrics=metrics,
        model_path=model_path,
    )

    # 3. INFERENCE
    print("Step 3: Inference")
    input_data = Path('./data/production_input_20251030.csv')
    input_hash = tracker._compute_hash(input_data)
    predictions_path = Path('./predictions/output_20251030.csv')

    inference_run_id = tracker.track_inference(
        model_version=training_run_id,
        input_data_hash=input_hash,
        predictions_path=predictions_path,
    )

    # 4. QUERY LINEAGE
    print("\nStep 4: Query Lineage")
    lineage = tracker.get_lineage(inference_run_id)

    print("\nLineage:")
    print(json.dumps(lineage, indent=2))

    print("\nNOW WE CAN:")
    print("1. Trace predictions back to model and data")
    print("2. Reproduce any step in pipeline")
    print("3. Satisfy compliance requirements")
    print("4. Debug production issues with full context")

if __name__ == '__main__':
    production_pipeline_with_lineage()
```

**Lineage tracking benefits:**
- Full traceability (data → model → predictions)
- Can reproduce any pipeline step
- Compliance-ready audit trail
- Debug production issues with context
- Link predictions to source


### Solution 6: MLflow vs W&B Decision Matrix

**When to use MLflow:**
- Self-hosted infrastructure (data privacy, compliance)
- Single user or small team (< 5 people)
- Want open source solution (no vendor lock-in)
- Simple experiment tracking needs
- Have DevOps resources (can run server)

**When to use W&B:**
- Team collaboration (> 5 people)
- Want hosted solution (no server management)
- Need advanced visualization (parallel coordinates, 3D plots)
- Real-time monitoring during training
- Easy sharing with stakeholders (send link)

**When to use both:**
- MLflow for model registry (staging/production workflow)
- W&B for experiment tracking (better visualization)
- Best of both worlds (local registry + cloud tracking)

**Quick integration:**

```python
# Use both MLflow and W&B together
with mlflow.start_run() as run:
    wandb.init(project="my-project", config=config)

    # Log to both
    mlflow.log_params(config)
    wandb.config.update(config)

    for epoch in range(epochs):
        metrics = train_epoch()
        mlflow.log_metrics(metrics, step=epoch)
        wandb.log(metrics)

    # W&B for visualization, MLflow for registry
    mlflow.register_model(model_uri, "model-name")
    wandb.log_artifact("model.pth")
```


### Solution 7: Reproducibility Checklist

**What to track for reproducibility:**

```python
# reproducibility.py - Ensure full reproducibility

import torch
import numpy as np
import random
import os
import subprocess
import mlflow

def ensure_reproducibility(config: dict):
    """
    Set up environment for reproducible experiments.

    SOLUTION: Eliminates non-determinism.
    """
    # 1. SET RANDOM SEEDS
    # WHY: Makes training deterministic
    seed = config.get('random_seed', 42)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 2. CUDNN DETERMINISTIC (slower but reproducible)
    # WHY: CuDNN has non-deterministic algorithms by default
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 3. LOG ENVIRONMENT
    # WHY: Library versions affect results
    env_info = {
        'python_version': subprocess.check_output(
            ['python', '--version']
        ).decode().strip(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'numpy_version': np.__version__,
    }

    mlflow.log_params(env_info)

    # 4. LOG CODE VERSION
    # WHY: Different code = different results
    try:
        git_commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode().strip()
        git_branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        ).decode().strip()

        mlflow.log_param('git_commit', git_commit)
        mlflow.log_param('git_branch', git_branch)
    except:
        pass

    # 5. LOG DATA VERSION
    # WHY: Different data = different results
    if 'data_hash' in config:
        mlflow.log_param('data_hash', config['data_hash'])

    # 6. LOG HYPERPARAMETERS
    # WHY: Core of experiment configuration
    mlflow.log_params(config)

    print("Reproducibility configured:")
    print(f"  Random seed: {seed}")
    print(f"  CuDNN deterministic: True")
    print(f"  Environment logged")
    print(f"  Code version logged")

# Reproducibility checklist:
# ✅ Random seeds set (torch, numpy, random)
# ✅ CuDNN deterministic mode enabled
# ✅ Environment versions logged (Python, PyTorch, CUDA)
# ✅ Code version logged (git commit hash)
# ✅ Data version logged (dataset hash)
# ✅ Hyperparameters logged (all config)
# ✅ Model architecture logged
# ✅ Training procedure documented

# NOW ANY EXPERIMENT CAN BE REPRODUCED EXACTLY!
```


## Part 3: REFACTOR - Pressure Tests (10 Scenarios)

### Pressure Test 1: Lost Experiment

**Scenario:** Your best model (96.5% accuracy) was trained 2 weeks ago. You did not track it. Can you reproduce it?

**Expected behavior:**
- WITHOUT tracking: Cannot reproduce (hyperparameters lost)
- WITH tracking: Load exact hyperparameters, data version, code version from MLflow/W&B, reproduce exactly

**Validation:**
```python
def test_lost_experiment_recovery():
    """
    Test ability to recover lost experiment.

    SUCCESS CRITERIA:
    - Can find experiment from 2 weeks ago
    - Can see exact hyperparameters used
    - Can download model checkpoint
    - Can reproduce training with same results
    """
    # Search for best run
    runs = mlflow.search_runs(
        experiment_ids=["1"],
        filter_string="metrics.val_acc > 96.0",
        order_by=["metrics.val_acc DESC"],
        max_results=1
    )

    assert len(runs) > 0, "Cannot find best run!"

    best_run = runs.iloc[0]

    # Verify we have everything needed
    required_params = [
        'learning_rate', 'batch_size', 'optimizer',
        'git_commit', 'data_hash', 'random_seed'
    ]

    for param in required_params:
        assert param in best_run['params'], f"Missing {param}!"

    print("✅ Can recover lost experiment")
    print(f"   Val Acc: {best_run['metrics.val_acc']:.2f}%")
    print(f"   LR: {best_run['params.learning_rate']}")
    print(f"   Batch Size: {best_run['params.batch_size']}")
    print(f"   Git Commit: {best_run['params.git_commit']}")
```


### Pressure Test 2: Production Model Unknown

**Scenario:** Production has a bug. Someone asks "which model version is deployed?" Can you answer?

**Expected behavior:**
- WITHOUT versioning: "I think it's model_best.pth from last week?"
- WITH versioning: "Production is v4 (run ID abc123, trained 2025-10-25, 94.2% val acc)"

**Validation:**
```python
def test_production_model_identification():
    """
    Test ability to identify production model.

    SUCCESS CRITERIA:
    - Can query model registry for production model
    - Can get version number, training date, metrics
    - Can download exact model weights
    """
    client = mlflow.tracking.MlflowClient()

    # Get production model
    model_name = "cifar10-resnet18"
    versions = client.search_model_versions(f"name='{model_name}'")

    prod_versions = [v for v in versions if v.current_stage == 'Production']

    assert len(prod_versions) > 0, "No production model found!"

    prod_model = prod_versions[0]

    # Verify we have metadata
    assert prod_model.version is not None
    assert prod_model.creation_timestamp is not None
    assert 'val_acc' in prod_model.tags

    print("✅ Production model identified")
    print(f"   Version: {prod_model.version}")
    print(f"   Stage: {prod_model.current_stage}")
    print(f"   Accuracy: {prod_model.tags.get('val_acc')}")
    print(f"   Created: {prod_model.creation_timestamp}")
```


### Pressure Test 3: Multiple Team Members

**Scenario:** 3 people training models. Can you compare all runs and find the best?

**Expected behavior:**
- WITHOUT tracking: Each person has own files, manual comparison
- WITH tracking: All runs in shared MLflow/W&B, sort by metric, see best instantly

**Validation:**
```python
def test_multi_user_comparison():
    """
    Test ability to compare runs across team members.

    SUCCESS CRITERIA:
    - All team members' runs visible
    - Can filter by user
    - Can sort by metric
    - Can see who achieved best result
    """
    # Search all runs from last week
    runs = mlflow.search_runs(
        experiment_ids=["1"],
        order_by=["metrics.val_acc DESC"],
    )

    # Verify we have runs from multiple users
    users = runs['tags.mlflow.user'].unique()
    assert len(users) >= 2, "Only one user's runs found"

    # Find best run
    best_run = runs.iloc[0]
    best_user = best_run['tags.mlflow.user']
    best_acc = best_run['metrics.val_acc']

    print("✅ Can compare team members' runs")
    print(f"   Total runs: {len(runs)}")
    print(f"   Team members: {list(users)}")
    print(f"   Best run: {best_user} ({best_acc:.2f}%)")
```


### Pressure Test 4: Data Changed

**Scenario:** Dataset updated yesterday. Model performance dropped. Was it code change or data change?

**Expected behavior:**
- WITHOUT data versioning: "Not sure, maybe data changed?"
- WITH data versioning: "Data hash changed from abc123 to def456, that's the cause"

**Validation:**
```python
def test_data_change_detection():
    """
    Test ability to detect data changes.

    SUCCESS CRITERIA:
    - Can see data hash for each run
    - Can identify when data changed
    - Can correlate data change with metric change
    """
    # Get recent runs
    runs = mlflow.search_runs(
        experiment_ids=["1"],
        order_by=["start_time DESC"],
        max_results=10
    )

    # Check if data hash is tracked
    assert 'params.data_hash' in runs.columns, "Data hash not tracked!"

    # Find runs with different data
    data_hashes = runs['params.data_hash'].unique()

    if len(data_hashes) > 1:
        print("✅ Data change detected")
        print(f"   Different data versions: {len(data_hashes)}")

        # Compare metrics across data versions
        for data_hash in data_hashes:
            runs_with_hash = runs[runs['params.data_hash'] == data_hash]
            avg_acc = runs_with_hash['metrics.val_acc'].mean()
            print(f"   Data {data_hash[:8]}: Avg acc = {avg_acc:.2f}%")
    else:
        print("✅ Data hash tracked (no changes detected)")
```


### Pressure Test 5: Rollback Required

**Scenario:** New model deployed to production. It's worse. Need to roll back to previous version immediately.

**Expected behavior:**
- WITHOUT versioning: "We overwrote the old model, cannot roll back"
- WITH versioning: "Rolling back to v3 (previous production)... done!"

**Validation:**
```python
def test_model_rollback():
    """
    Test ability to roll back production model.

    SUCCESS CRITERIA:
    - Can identify previous production version
    - Can transition back to that version
    - Model weights downloadable
    - <5 minutes to roll back
    """
    client = mlflow.tracking.MlflowClient()
    model_name = "cifar10-resnet18"

    # Get all versions
    versions = client.search_model_versions(f"name='{model_name}'")
    versions = sorted(versions, key=lambda v: v.version, reverse=True)

    assert len(versions) >= 2, "Need at least 2 versions to test rollback"

    # Current production
    current_prod = [v for v in versions if v.current_stage == 'Production']

    # Find previous production (in Archived)
    archived = [v for v in versions if v.current_stage == 'Archived']

    if len(archived) > 0:
        # Roll back to archived version
        target_version = archived[0].version

        client.transition_model_version_stage(
            name=model_name,
            version=target_version,
            stage="Production"
        )

        print("✅ Rollback successful")
        print(f"   Rolled back to version {target_version}")
    else:
        print("✅ Rollback capability available (no archived versions yet)")
```


### Pressure Test 6: Prediction Audit

**Scenario:** Compliance asks: "For prediction ID 12345, which model and data produced it?"

**Expected behavior:**
- WITHOUT lineage: "Not sure, let me check logs... (hours later) cannot determine"
- WITH lineage: "Prediction 12345: Model v3, Input data hash abc123, Timestamp 2025-10-30 14:23"

**Validation:**
```python
def test_prediction_audit_trail():
    """
    Test ability to audit predictions.

    SUCCESS CRITERIA:
    - Can trace prediction to model version
    - Can trace prediction to input data
    - Can get timestamp
    - Full audit trail available
    """
    # Search inference runs
    runs = mlflow.search_runs(
        experiment_ids=["1"],
        filter_string="tags.mlflow.runName LIKE 'inference-%'",
        order_by=["start_time DESC"],
    )

    assert len(runs) > 0, "No inference runs found!"

    # Check audit trail for first inference
    inference_run = runs.iloc[0]

    required_metadata = [
        'params.model_version',
        'params.input_data_hash',
        'params.predictions_hash',
        'params.inference_timestamp',
    ]

    for field in required_metadata:
        assert field in inference_run, f"Missing {field}!"

    print("✅ Prediction audit trail complete")
    print(f"   Model version: {inference_run['params.model_version']}")
    print(f"   Input data: {inference_run['params.input_data_hash'][:8]}...")
    print(f"   Timestamp: {inference_run['params.inference_timestamp']}")
```


### Pressure Test 7: Hyperparameter Search

**Scenario:** Ran 100 experiments with different hyperparameters. Which combination is best?

**Expected behavior:**
- WITHOUT tracking: Parse 100 log files manually, create spreadsheet
- WITH tracking: Sort by metric in UI, see best instantly, download config

**Validation:**
```python
def test_hyperparameter_search_analysis():
    """
    Test ability to analyze hyperparameter search.

    SUCCESS CRITERIA:
    - Can query all search runs
    - Can sort by metric
    - Can visualize hyperparameter impact
    - Can download best config
    """
    # Search all runs
    runs = mlflow.search_runs(
        experiment_ids=["1"],
        order_by=["metrics.val_acc DESC"],
    )

    assert len(runs) >= 10, "Need multiple runs for search analysis"

    # Get best run
    best_run = runs.iloc[0]

    # Extract hyperparameters
    hyperparam_columns = [col for col in runs.columns if col.startswith('params.')]

    assert len(hyperparam_columns) > 0, "No hyperparameters logged!"

    best_config = {
        col.replace('params.', ''): best_run[col]
        for col in hyperparam_columns
    }

    print("✅ Hyperparameter search analyzable")
    print(f"   Total runs: {len(runs)}")
    print(f"   Best accuracy: {best_run['metrics.val_acc']:.2f}%")
    print(f"   Best config: {best_config}")
```


### Pressure Test 8: Reproduce Paper Results

**Scenario:** Colleague published paper with "96.8% accuracy". Can they reproduce it 6 months later?

**Expected behavior:**
- WITHOUT tracking: "I think I used learning_rate=0.01? Not sure..."
- WITH tracking: Load exact run from MLflow, all details preserved, reproduce exactly

**Validation:**
```python
def test_long_term_reproducibility():
    """
    Test ability to reproduce results long-term.

    SUCCESS CRITERIA:
    - Can find run from 6 months ago
    - All configuration preserved
    - Model checkpoint available
    - Can re-run with same config
    """
    # Search runs older than 30 days
    import time
    thirty_days_ago = int((time.time() - 30*24*3600) * 1000)

    runs = mlflow.search_runs(
        experiment_ids=["1"],
        filter_string=f"attributes.start_time < {thirty_days_ago}",
        order_by=["start_time ASC"],
        max_results=1
    )

    if len(runs) > 0:
        old_run = runs.iloc[0]

        # Check configuration is complete
        required_fields = [
            'params.learning_rate',
            'params.batch_size',
            'params.random_seed',
            'params.git_commit',
            'params.data_hash',
        ]

        missing = [f for f in required_fields if f not in old_run or pd.isna(old_run[f])]

        if len(missing) == 0:
            print("✅ Long-term reproducibility verified")
            print(f"   Run age: {old_run['start_time']}")
            print(f"   All config preserved")
        else:
            print(f"⚠️  Missing fields: {missing}")
    else:
        print("✅ Tracking system ready for long-term reproducibility")
```


### Pressure Test 9: Artifact Management

**Scenario:** 50 experiments, each saves 5 checkpoints. Running out of disk space. Which can be deleted?

**Expected behavior:**
- WITHOUT artifact tracking: Manually check each file, guess which are safe to delete
- WITH artifact tracking: Query MLflow for artifacts, delete all except top-5 runs

**Validation:**
```python
def test_artifact_cleanup():
    """
    Test ability to manage artifacts efficiently.

    SUCCESS CRITERIA:
    - Can list all artifacts
    - Can identify artifacts from low-performing runs
    - Can safely delete artifacts
    - Keep top-N runs automatically
    """
    # Get all runs
    runs = mlflow.search_runs(
        experiment_ids=["1"],
        order_by=["metrics.val_acc DESC"],
    )

    # Identify top runs to keep
    top_n = 5
    top_runs = runs.head(top_n)
    deletable_runs = runs.tail(len(runs) - top_n)

    print("✅ Artifact management possible")
    print(f"   Total runs: {len(runs)}")
    print(f"   Keeping top {top_n} runs")
    print(f"   Can delete {len(deletable_runs)} runs")

    # In production, would delete artifacts from deletable_runs:
    # for run_id in deletable_runs['run_id']:
    #     client.delete_run(run_id)
```


### Pressure Test 10: Team Onboarding

**Scenario:** New team member joins. Can they see all past experiments and understand what was tried?

**Expected behavior:**
- WITHOUT tracking: Read scattered docs, ask questions, incomplete picture
- WITH tracking: Browse MLflow/W&B UI, see all experiments, metrics, configs, get up to speed in hours

**Validation:**
```python
def test_team_onboarding():
    """
    Test ability to onboard new team members.

    SUCCESS CRITERIA:
    - Can browse all past experiments
    - Can see what was tried (hyperparameters)
    - Can see what worked (metrics)
    - Can download models and configs
    - Documentation in one place
    """
    # Get all experiments
    experiments = mlflow.search_experiments()

    total_runs = 0
    for exp in experiments:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        total_runs += len(runs)

    print("✅ Team onboarding enabled")
    print(f"   Experiments: {len(experiments)}")
    print(f"   Total runs: {total_runs}")
    print(f"   UI: http://localhost:5000")
    print(f"   New members can browse all past work")
```


### REFACTOR Summary: Stress Testing Your Tracking

**All 10 pressure tests must pass:**

1. **Lost Experiment Recovery** - Find and reproduce best run from weeks ago
2. **Production Model ID** - Instantly identify which model is deployed
3. **Multi-User Comparison** - Compare runs across team members
4. **Data Change Detection** - Trace performance changes to data versions
5. **Model Rollback** - Revert production to previous version in <5 minutes
6. **Prediction Audit** - Full lineage from predictions to source
7. **Hyperparameter Search** - Analyze 100+ runs efficiently
8. **Long-term Reproducibility** - Reproduce results from 6+ months ago
9. **Artifact Cleanup** - Safely delete artifacts without losing important runs
10. **Team Onboarding** - New members understand past work in hours

**Common tracking failures that pressure tests catch:**

```python
# Failure 1: Incomplete logging
# SYMPTOM: Can find run but missing key parameters
mlflow.log_params({
    'learning_rate': 0.001,
    # MISSING: batch_size, optimizer, random_seed
})
# RESULT: Pressure Test 1 fails (cannot fully reproduce)

# Failure 2: No model registry
# SYMPTOM: Cannot identify production model
torch.save(model, 'production_model.pth')  # No versioning!
# RESULT: Pressure Test 2 fails (which version is this?)

# Failure 3: No data versioning
# SYMPTOM: Cannot correlate metric changes to data changes
mlflow.log_param('data_path', './data')  # Path, not version!
# RESULT: Pressure Test 4 fails (data changed, how to know?)

# Failure 4: No lineage tracking
# SYMPTOM: Cannot trace predictions to model/data
model.predict(data)
save_predictions('output.csv')  # No link to model version!
# RESULT: Pressure Test 6 fails (which model made these predictions?)

# Failure 5: No artifact retention policy
# SYMPTOM: Disk fills up, unclear what to delete
for i in range(100):
    mlflow.log_artifact(f'checkpoint_{i}.pth')  # All saved forever!
# RESULT: Pressure Test 9 fails (200GB of checkpoints, which are important?)
```

**Pressure test frequency:**

- **During development:** Run tests 1, 3, 7 daily (experiment recovery, comparison, search)
- **Before production deploy:** Run tests 2, 5, 6 (model ID, rollback, audit)
- **Monthly:** Run tests 4, 8, 9 (data changes, long-term repro, cleanup)
- **New hire:** Run test 10 (onboarding)

**Failure recovery:**

If pressure tests fail, fix tracking systematically:

```python
# Step 1: Add missing parameters
REQUIRED_PARAMS = [
    'learning_rate', 'batch_size', 'optimizer', 'random_seed',
    'git_commit', 'data_hash', 'model_architecture',
]

for param in REQUIRED_PARAMS:
    assert param in config, f"Missing required param: {param}"
    mlflow.log_param(param, config[param])

# Step 2: Enable model registry
mlflow.register_model(model_uri, model_name)

# Step 3: Version data with hash
data_hash = compute_hash(data_path)
mlflow.log_param('data_hash', data_hash)

# Step 4: Track lineage
mlflow.log_param('parent_run_id', training_run_id)  # Link inference to training

# Step 5: Implement artifact retention
if run_metric < top_5_threshold:
    # Don't log large artifacts for low-performing runs
    pass
```

**Success metrics:**

Your tracking is production-ready when:

- ✅ All 10 pressure tests pass
- ✅ Can reproduce any experiment from last 6 months in <10 minutes
- ✅ Can identify production model version in <30 seconds
- ✅ New team members productive in <4 hours (not <2 days)
- ✅ Disk usage under control (automatic cleanup)
- ✅ Zero compliance violations (full audit trail)
- ✅ Zero lost experiments (everything tracked)

**The test:** Can you go on vacation for 2 weeks and have team reproduce your best result? If no, tracking is incomplete.


## Part 4: Integration Patterns

### Pattern 1: MLflow in Training Script

```python
# Minimal MLflow integration
with mlflow.start_run():
    mlflow.log_params(config)

    for epoch in range(epochs):
        train_loss, val_loss = train_epoch()
        mlflow.log_metrics({'train_loss': train_loss, 'val_loss': val_loss}, step=epoch)

    mlflow.pytorch.log_model(model, "model")
```

### Pattern 2: W&B in Training Loop

```python
# Minimal W&B integration
wandb.init(project="my-project", config=config)

for epoch in range(epochs):
    train_loss, val_loss = train_epoch()
    wandb.log({'train_loss': train_loss, 'val_loss': val_loss})

wandb.finish()
```

### Pattern 3: Hyperparameter Sweep

```python
# W&B hyperparameter sweep
sweep_config = {
    'method': 'random',
    'parameters': {
        'learning_rate': {'values': [0.001, 0.01, 0.1]},
        'batch_size': {'values': [32, 64, 128]},
    }
}

sweep_id = wandb.sweep(sweep_config, project="my-project")
wandb.agent(sweep_id, function=train_model, count=20)
```


## Skill Mastery Checklist

You have mastered experiment tracking when you can:

- [ ] Recognize when tracking is needed (cannot reproduce, lost experiments)
- [ ] Set up MLflow tracking server and UI
- [ ] Set up W&B account and project
- [ ] Instrument training code to log hyperparameters, metrics, artifacts
- [ ] Version models in model registry with staging labels
- [ ] Version datasets with DVC or hash-based tracking
- [ ] Implement lineage tracking (data → model → predictions)
- [ ] Ensure reproducibility (seeds, environment, code version)
- [ ] Choose between MLflow and W&B based on requirements
- [ ] Query tracking system to find best experiments
- [ ] Roll back production models when needed
- [ ] Audit predictions for compliance
- [ ] Onboard new team members using tracked experiments

**Key insight:** Without tracking, experiments are lost. With tracking, every experiment is reproducible and queryable. The skill is building systems where reproducibility is automatic, not manual.
