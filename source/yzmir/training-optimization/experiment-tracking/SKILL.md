---
name: experiment-tracking
description: Master experiment tracking for ML training - what to track (hyperparameters, metrics, artifacts), tool selection (TensorBoard, Weights & Biases, MLflow), reproducibility, experiment organization, comparison, and team collaboration
---

# Experiment Tracking Skill

## When to Use This Skill

Use this skill when:
- User starts training a model and asks "should I track this experiment?"
- User wants to reproduce a previous result but doesn't remember settings
- Training runs overnight and user needs persistent logs
- User asks "which tool should I use: TensorBoard, W&B, or MLflow?"
- Multiple experiments running and user can't compare results
- User wants to share results with teammates or collaborators
- Model checkpoints accumulating with no organization or versioning
- User asks "what should I track?" or "how do I make experiments reproducible?"
- Debugging training issues and needs historical data (metrics, gradients)
- User wants to visualize training curves or compare hyperparameters
- Working on a research project that requires tracking many experiments
- User lost their best result and can't reproduce it

Do NOT use when:
- User is doing quick prototyping with throwaway code (<5 minutes)
- Only running inference on pre-trained models (no training)
- Single experiment that's already tracked and working
- User is asking about hyperparameter tuning strategy (not tracking)
- Discussing model architecture design (not experiment management)

---

## Core Principles

### 1. Track Before You Need It (Can't Add Retroactively)

The BIGGEST mistake: waiting to track until results are worth saving.

**The Reality**:
- The best result is ALWAYS the one you didn't track
- Can't add tracking after the experiment completes
- Human memory fails within hours (let alone days/weeks)
- Print statements disappear when terminal closes
- Code changes between experiments (git state matters)

**When Tracking Matters**:
```
Experiment value curve:
    ^
    |                    ╱─  Peak result (untracked = lost forever)
    |                  ╱
    |                ╱
    |              ╱
    |            ╱
    |          ╱
    |        ╱
    |      ╱
    |____╱________________________________>
         Start                        Time

If you wait to track "important" experiments, you've already lost them.
```

**Track From Day 1**:
- First experiment (even if "just testing")
- Every hyperparameter change
- Every model architecture variation
- Every data preprocessing change

**Decision Rule**: If you're running `python train.py`, you should be tracking. No exceptions.

---

### 2. Complete Tracking = Hyperparameters + Metrics + Artifacts + Environment

Reproducibility requires tracking EVERYTHING that affects the result.

**The Five Categories**:

```
┌─────────────────────────────────────────────────────────┐
│ 1. HYPERPARAMETERS (what you're tuning)                │
├─────────────────────────────────────────────────────────┤
│ • Learning rate, batch size, optimizer type             │
│ • Model architecture (width, depth, activation)         │
│ • Regularization (weight decay, dropout)                │
│ • Training length (epochs, steps)                       │
│ • Data augmentation settings                            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 2. METRICS (how you're doing)                           │
├─────────────────────────────────────────────────────────┤
│ • Training loss (every step or epoch)                   │
│ • Validation loss (every epoch)                         │
│ • Evaluation metrics (accuracy, F1, mAP, etc.)          │
│ • Learning rate schedule (actual LR each step)          │
│ • Gradient norms (for debugging)                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 3. ARTIFACTS (what you're saving)                       │
├─────────────────────────────────────────────────────────┤
│ • Model checkpoints (with epoch/step metadata)          │
│ • Training plots (loss curves, confusion matrices)      │
│ • Predictions on validation set                         │
│ • Logs (stdout, stderr)                                 │
│ • Config files (for reproducibility)                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 4. CODE VERSION (what you're running)                   │
├─────────────────────────────────────────────────────────┤
│ • Git commit hash                                        │
│ • Git branch name                                        │
│ • Dirty status (uncommitted changes)                    │
│ • Code diff (if uncommitted)                            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 5. ENVIRONMENT (where you're running)                   │
├─────────────────────────────────────────────────────────┤
│ • Python version, PyTorch version                       │
│ • CUDA version, GPU type                                │
│ • Random seeds (Python, NumPy, PyTorch, CUDA)           │
│ • Data version (if dataset changes)                     │
│ • Hardware (CPU, RAM, GPU count)                        │
└─────────────────────────────────────────────────────────┘
```

**Reproducibility Test**:
> Can someone else (or future you) reproduce the result with ONLY the tracked information?

If NO, you're not tracking enough.

---

### 3. Tool Selection: Local vs Team vs Production

Different tools for different use cases. Choose based on your needs.

**Tool Comparison**:

| Feature | TensorBoard | Weights & Biases | MLflow | Custom |
|---------|-------------|------------------|--------|--------|
| **Setup Complexity** | Low | Low | Medium | High |
| **Local Only** | Yes | No (cloud) | Yes | Yes |
| **Team Collaboration** | Limited | Excellent | Good | Custom |
| **Cost** | Free | Free tier + paid | Free | Free |
| **Scalability** | Medium | High | High | Low |
| **Visualization** | Good | Excellent | Good | Custom |
| **Integration** | PyTorch, TF | Everything | Everything | Manual |
| **Best For** | Solo projects | Team research | Production | Specific needs |

**Decision Tree**:
```
Do you need team collaboration?
├─ YES → Need to share results with teammates?
│   ├─ YES → Weights & Biases (best team features)
│   └─ NO → MLflow (self-hosted, more control)
│
└─ NO → Solo project?
    ├─ YES → TensorBoard (simplest, local)
    └─ NO → MLflow (scales to production)

Budget constraints?
├─ FREE only → TensorBoard or MLflow
└─ Can pay → W&B (worth it for teams)

Production deployment?
├─ YES → MLflow (production-ready)
└─ NO → TensorBoard or W&B (research)
```

**Recommendation**:
- **Starting out / learning**: TensorBoard (easiest, free, local)
- **Research team / collaboration**: Weights & Biases (best UX, sharing)
- **Production ML / enterprise**: MLflow (self-hosted, model registry)
- **Specific needs / customization**: Custom logging (CSV + Git)

---

### 4. Minimal Overhead, Maximum Value

Tracking should cost 1-5% overhead, not 50%.

**What to Track at Different Frequencies**:

```python
# Every step (high frequency, small data):
log_every_step = {
    "train_loss": loss.item(),
    "learning_rate": optimizer.param_groups[0]['lr'],
    "step": global_step,
}

# Every epoch (medium frequency, medium data):
log_every_epoch = {
    "train_loss_avg": train_losses.mean(),
    "val_loss": val_loss,
    "val_accuracy": val_acc,
    "epoch": epoch,
}

# Once per experiment (low frequency, large data):
log_once = {
    "hyperparameters": config,
    "git_commit": get_git_hash(),
    "environment": {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    },
}

# Only on improvement (conditional):
if val_loss < best_val_loss:
    save_checkpoint(model, optimizer, epoch, val_loss)
    log_artifact("best_model.pt")
```

**Overhead Guidelines**:
- Logging scalars (loss, accuracy): <0.1% overhead (always do)
- Logging images/plots: 1-2% overhead (do every epoch)
- Logging checkpoints: 5-10% overhead (do only on improvement)
- Logging gradients: 10-20% overhead (do only for debugging)

**Don't Track**:
- Raw training data (too large, use data versioning instead)
- Every intermediate activation (use profiling tools instead)
- Full model weights every step (only on improvement)

---

### 5. Experiment Organization: Naming, Tagging, Grouping

With 100+ experiments, organization is survival.

**Naming Convention**:
```python
# GOOD: Descriptive, sortable, parseable
experiment_name = f"{model}_{dataset}_{timestamp}_{hyperparams}"
# Examples:
# "resnet18_cifar10_20241030_lr0.01_bs128"
# "bert_squad_20241030_lr3e-5_warmup1000"
# "gpt2_wikitext_20241030_ctx512_layers12"

# BAD: Uninformative
experiment_name = "test"
experiment_name = "final"
experiment_name = "model_v2"
experiment_name = "test_again_actually_final"
```

**Tagging Strategy**:
```python
# Tags for filtering and grouping
tags = {
    "model": "resnet18",
    "dataset": "cifar10",
    "experiment_type": "hyperparameter_search",
    "status": "completed",
    "goal": "beat_baseline",
    "author": "john",
}

# Can filter later:
# - Show me all "hyperparameter_search" experiments
# - Show me all "resnet18" on "cifar10"
# - Show me experiments by "john"
```

**Grouping Related Experiments**:
```python
# Group by goal/project
project = "cifar10_sota"
group = "learning_rate_search"
experiment_name = f"{project}/{group}/lr_{lr}"

# Hierarchy:
# cifar10_sota/
#   ├─ learning_rate_search/
#   │   ├─ lr_0.001
#   │   ├─ lr_0.01
#   │   └─ lr_0.1
#   ├─ architecture_search/
#   │   ├─ resnet18
#   │   ├─ resnet34
#   │   └─ resnet50
#   └─ regularization_search/
#       ├─ dropout_0.1
#       ├─ dropout_0.3
#       └─ dropout_0.5
```

---

## Tool-Specific Integration

### TensorBoard (Local, Simple)

**Setup**:
```python
from torch.utils.tensorboard import SummaryWriter

# Create writer
writer = SummaryWriter(f"runs/{experiment_name}")

# Log hyperparameters
hparams = {
    "learning_rate": 0.01,
    "batch_size": 128,
    "optimizer": "adam",
}
metrics = {
    "best_val_acc": 0.0,
}
writer.add_hparams(hparams, metrics)
```

**During Training**:
```python
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Training step
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Log every N steps
        global_step = epoch * len(train_loader) + batch_idx
        if global_step % log_interval == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)

    # Validation
    val_loss, val_acc = evaluate(model, val_loader)
    writer.add_scalar("val/loss", val_loss, epoch)
    writer.add_scalar("val/accuracy", val_acc, epoch)

    # Log images (confusion matrix, etc.)
    if epoch % 10 == 0:
        fig = plot_confusion_matrix(model, val_loader)
        writer.add_figure("val/confusion_matrix", fig, epoch)

writer.close()
```

**View Results**:
```bash
tensorboard --logdir=runs
# Opens web UI at http://localhost:6006
```

**Pros**:
- Simple setup (2 lines of code)
- Local (no cloud dependency)
- Good visualizations (scalars, images, graphs)
- Integrated with PyTorch

**Cons**:
- No hyperparameter comparison table
- Limited team collaboration
- No artifact storage (checkpoints)
- Manual experiment management

---

### Weights & Biases (Team, Cloud)

**Setup**:
```python
import wandb

# Initialize experiment
wandb.init(
    project="cifar10-sota",
    name=experiment_name,
    config={
        "learning_rate": 0.01,
        "batch_size": 128,
        "optimizer": "adam",
        "model": "resnet18",
        "dataset": "cifar10",
    },
    tags=["hyperparameter_search", "resnet"],
)

# Config is automatically tracked
config = wandb.config
```

**During Training**:
```python
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Log metrics
        wandb.log({
            "train/loss": loss.item(),
            "train/lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch,
        })

    # Validation
    val_loss, val_acc = evaluate(model, val_loader)
    wandb.log({
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "epoch": epoch,
    })

    # Save checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        wandb.save("best_model.pt")  # Upload to cloud

# Log final results
wandb.log({"best_val_accuracy": best_val_acc})
wandb.finish()
```

**Advanced Features**:
```python
# Log images
wandb.log({"examples": [wandb.Image(img, caption=f"Pred: {pred}") for img, pred in samples]})

# Log plots
fig = plot_confusion_matrix(model, val_loader)
wandb.log({"confusion_matrix": wandb.Image(fig)})

# Log tables (for result analysis)
table = wandb.Table(columns=["epoch", "train_loss", "val_loss", "val_acc"])
for epoch, tl, vl, va in zip(epochs, train_losses, val_losses, val_accs):
    table.add_data(epoch, tl, vl, va)
wandb.log({"results": table})

# Log model architecture
wandb.watch(model, log="all", log_freq=100)  # Logs gradients + weights
```

**View Results**:
- Web UI: https://wandb.ai/your-username/cifar10-sota
- Compare experiments side-by-side
- Share links with teammates
- Filter by tags, hyperparameters

**Pros**:
- Excellent team collaboration (share links)
- Beautiful visualizations
- Hyperparameter comparison (parallel coordinates)
- Artifact versioning (models, data)
- Integration with everything (PyTorch, TF, JAX)

**Cons**:
- Cloud-based (requires internet)
- Free tier limits (100GB storage)
- Data leaves your machine (privacy concern)

---

### MLflow (Production, Self-Hosted)

**Setup**:
```python
import mlflow
import mlflow.pytorch

# Start experiment
mlflow.set_experiment("cifar10-sota")

# Start run
with mlflow.start_run(run_name=experiment_name):
    # Log hyperparameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 128)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("model", "resnet18")

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)

        # Log metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

    # Log final metrics
    mlflow.log_metric("best_val_accuracy", best_val_acc)

    # Log model
    mlflow.pytorch.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("config.yaml")
    mlflow.log_artifact("best_model.pt")
```

**View Results**:
```bash
mlflow ui
# Opens web UI at http://localhost:5000
```

**Model Registry** (for production):
```python
# Register model
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "cifar10-resnet18")

# Load registered model
model = mlflow.pytorch.load_model("models:/cifar10-resnet18/production")
```

**Pros**:
- Self-hosted (full control, privacy)
- Model registry (production deployment)
- Scales to large teams
- Integration with deployment tools

**Cons**:
- More complex setup (need server)
- Visualization not as good as W&B
- Less intuitive UI

---

## Reproducibility Patterns

### 1. Seed Everything

```python
import random
import numpy as np
import torch

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# At start of training
set_seed(42)

# Log seed
config = {"seed": 42}
```

**Warning**: Deterministic mode can be 10-20% slower. Trade-off between speed and reproducibility.

---

### 2. Capture Git State

```python
import subprocess

def get_git_info():
    """Capture current git state."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()

        # Check for uncommitted changes
        status = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()
        is_dirty = len(status) > 0

        # Get diff if dirty
        diff = None
        if is_dirty:
            diff = subprocess.check_output(['git', 'diff']).decode('ascii')

        return {
            "commit": commit,
            "branch": branch,
            "is_dirty": is_dirty,
            "diff": diff,
        }
    except Exception as e:
        return {"error": str(e)}

# Log git info
git_info = get_git_info()
if git_info.get("is_dirty"):
    print("WARNING: Uncommitted changes detected!")
    print("Experiment may not be reproducible without the diff.")
```

---

### 3. Environment Capture

```python
import sys
import torch

def get_environment_info():
    """Capture environment details."""
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
    }

# Save requirements.txt
# pip freeze > requirements.txt

# Or use pip-tools
# pip-compile requirements.in
```

---

### 4. Config Files for Reproducibility

```python
# config.yaml
model:
  name: resnet18
  num_classes: 10

training:
  learning_rate: 0.01
  batch_size: 128
  num_epochs: 100
  optimizer: adam
  weight_decay: 0.0001

data:
  dataset: cifar10
  augmentation: true
  normalize: true

# Load config
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Save config alongside results
import shutil
shutil.copy("config.yaml", f"results/{experiment_name}/config.yaml")
```

---

## Experiment Comparison

### 1. Comparing Metrics

```python
# TensorBoard: Compare multiple runs
# tensorboard --logdir=runs --port=6006
# Select multiple runs in UI

# W&B: Filter and compare
# Go to project page, select runs, click "Compare"

# MLflow: Query experiments
import mlflow

# Get all runs from an experiment
experiment = mlflow.get_experiment_by_name("cifar10-sota")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Filter by metric
best_runs = runs[runs["metrics.val_accuracy"] > 0.85]

# Sort by metric
best_runs = runs.sort_values("metrics.val_accuracy", ascending=False)

# Analyze hyperparameter impact
import pandas as pd
import seaborn as sns

# Plot learning rate vs accuracy
sns.scatterplot(data=runs, x="params.learning_rate", y="metrics.val_accuracy")
```

---

### 2. Hyperparameter Analysis

```python
# W&B: Parallel coordinates plot
# Shows which hyperparameter combinations lead to best results
# UI: Click "Parallel Coordinates" in project view

# MLflow: Custom analysis
import matplotlib.pyplot as plt

# Group by hyperparameter
for lr in [0.001, 0.01, 0.1]:
    lr_runs = runs[runs["params.learning_rate"] == str(lr)]
    accuracies = lr_runs["metrics.val_accuracy"]
    plt.scatter([lr] * len(accuracies), accuracies, alpha=0.5, label=f"LR={lr}")

plt.xlabel("Learning Rate")
plt.ylabel("Validation Accuracy")
plt.xscale("log")
plt.legend()
plt.title("Learning Rate vs Accuracy")
plt.show()
```

---

### 3. Comparing Artifacts

```python
# Compare model checkpoints
from torchvision.models import resnet18

# Load two models
model_a = resnet18()
model_a.load_state_dict(torch.load("experiments/exp_a/best_model.pt"))

model_b = resnet18()
model_b.load_state_dict(torch.load("experiments/exp_b/best_model.pt"))

# Compare on validation set
acc_a = evaluate(model_a, val_loader)
acc_b = evaluate(model_b, val_loader)

print(f"Model A: {acc_a:.2%}")
print(f"Model B: {acc_b:.2%}")

# Compare predictions
preds_a = model_a(val_data)
preds_b = model_b(val_data)
agreement = (preds_a.argmax(1) == preds_b.argmax(1)).float().mean()
print(f"Prediction agreement: {agreement:.2%}")
```

---

## Collaboration Workflows

### 1. Sharing Results (W&B)

```python
# Share experiment link
# https://wandb.ai/your-username/cifar10-sota/runs/run-id

# Create report
# W&B UI: Click "Create Report" → Add charts, text, code

# Export results
# W&B UI: Click "Export" → CSV, JSON, or API

# API access for programmatic sharing
import wandb
api = wandb.Api()
runs = api.runs("your-username/cifar10-sota")

for run in runs:
    print(f"{run.name}: {run.summary['val_accuracy']}")
```

---

### 2. Team Experiment Dashboard

```python
# MLflow: Shared tracking server
# Server machine:
mlflow server --host 0.0.0.0 --port 5000

# Team members:
import mlflow
mlflow.set_tracking_uri("http://shared-server:5000")

# Everyone logs to same server
with mlflow.start_run():
    mlflow.log_metric("val_accuracy", 0.87)
```

---

### 3. Experiment Handoff

```python
# Package experiment for reproducibility
experiment_package = {
    "code": "git_commit_hash",
    "config": "config.yaml",
    "model": "best_model.pt",
    "results": "results.csv",
    "logs": "training.log",
    "environment": "requirements.txt",
}

# Create reproducibility script
# reproduce.sh
"""
#!/bin/bash
git checkout <commit-hash>
pip install -r requirements.txt
python train.py --config config.yaml
"""
```

---

## Complete Tracking Example

Here's a production-ready tracking setup:

```python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import wandb
import yaml
import subprocess
from pathlib import Path
from datetime import datetime

class ExperimentTracker:
    """Complete experiment tracking wrapper."""

    def __init__(self, config, experiment_name=None, use_wandb=True, use_tensorboard=True):
        self.config = config
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard

        # Generate experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{config['model']}_{config['dataset']}_{timestamp}"
        self.experiment_name = experiment_name

        # Create experiment directory
        self.exp_dir = Path(f"experiments/{experiment_name}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking tools
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(self.exp_dir / "tensorboard")

        if self.use_wandb:
            wandb.init(
                project=config.get("project", "default"),
                name=experiment_name,
                config=config,
                dir=self.exp_dir,
            )

        # Save config
        with open(self.exp_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        # Capture environment
        self._log_environment()

        # Capture git state
        self._log_git_state()

        # Setup logging
        self._setup_logging()

        self.global_step = 0
        self.best_metric = float('-inf')

    def _log_environment(self):
        """Log environment information."""
        import sys
        env_info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count(),
        }

        # Save to file
        with open(self.exp_dir / "environment.yaml", "w") as f:
            yaml.dump(env_info, f)

        # Log to W&B
        if self.use_wandb:
            wandb.config.update({"environment": env_info})

    def _log_git_state(self):
        """Log git commit and status."""
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
            status = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()
            is_dirty = len(status) > 0

            git_info = {
                "commit": commit,
                "branch": branch,
                "is_dirty": is_dirty,
            }

            # Save to file
            with open(self.exp_dir / "git_info.yaml", "w") as f:
                yaml.dump(git_info, f)

            # Save diff if dirty
            if is_dirty:
                diff = subprocess.check_output(['git', 'diff']).decode('ascii')
                with open(self.exp_dir / "git_diff.patch", "w") as f:
                    f.write(diff)
                print("WARNING: Uncommitted changes detected! Saved to git_diff.patch")

            # Log to W&B
            if self.use_wandb:
                wandb.config.update({"git": git_info})

        except Exception as e:
            print(f"Failed to capture git state: {e}")

    def _setup_logging(self):
        """Setup file logging."""
        import logging
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.exp_dir / "training.log")
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_metrics(self, metrics, step=None):
        """Log metrics to all tracking backends."""
        if step is None:
            step = self.global_step

        # TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, step)

        # W&B
        if self.use_wandb:
            wandb.log(metrics, step=step)

        # File
        self.logger.info(f"Step {step}: {metrics}")

        self.global_step = step + 1

    def save_checkpoint(self, model, optimizer, epoch, metric_value, metric_name="val_accuracy"):
        """Save model checkpoint with metadata."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            metric_name: metric_value,
            "config": self.config,
        }

        # Save latest checkpoint
        checkpoint_path = self.exp_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if metric_value > self.best_metric:
            self.best_metric = metric_value
            best_path = self.exp_dir / "checkpoints" / "best_model.pt"
            torch.save(checkpoint, best_path)

            self.logger.info(f"New best model saved: {metric_name}={metric_value:.4f}")

            # Log to W&B
            if self.use_wandb:
                wandb.log({f"best_{metric_name}": metric_value})
                wandb.save(str(best_path))

        return checkpoint_path

    def log_figure(self, name, figure, step=None):
        """Log matplotlib figure."""
        if step is None:
            step = self.global_step

        # TensorBoard
        if self.use_tensorboard:
            self.tb_writer.add_figure(name, figure, step)

        # W&B
        if self.use_wandb:
            wandb.log({name: wandb.Image(figure)}, step=step)

        # Save to disk
        fig_path = self.exp_dir / "figures" / f"{name}_step_{step}.png"
        fig_path.parent.mkdir(exist_ok=True)
        figure.savefig(fig_path)

    def finish(self):
        """Clean up and close tracking backends."""
        if self.use_tensorboard:
            self.tb_writer.close()

        if self.use_wandb:
            wandb.finish()

        self.logger.info("Experiment tracking finished.")


# Usage example
if __name__ == "__main__":
    config = {
        "project": "cifar10-sota",
        "model": "resnet18",
        "dataset": "cifar10",
        "learning_rate": 0.01,
        "batch_size": 128,
        "num_epochs": 100,
        "optimizer": "adam",
        "seed": 42,
    }

    # Initialize tracker
    tracker = ExperimentTracker(config)

    # Training loop
    model = create_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["num_epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_acc = evaluate(model, val_loader)

        # Log metrics
        tracker.log_metrics({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "epoch": epoch,
        })

        # Save checkpoint
        tracker.save_checkpoint(model, optimizer, epoch, val_acc)

        # Log figure (every 10 epochs)
        if epoch % 10 == 0:
            fig = plot_confusion_matrix(model, val_loader)
            tracker.log_figure("confusion_matrix", fig)

    # Finish
    tracker.finish()
```

---

## Pitfalls and Anti-Patterns

### Pitfall 1: Tracking Metrics But Not Config

**Symptom**: Have CSV with 50 experiments' metrics, but no idea what hyperparameters produced them.

**Why It Happens**:
- User focuses on "what matters" (the metric)
- Assumes they'll remember settings
- Doesn't realize metrics without context are useless

**Fix**:
```python
# WRONG: Only metrics
with open("results.csv", "a") as f:
    f.write(f"{epoch},{train_loss},{val_loss}\n")

# RIGHT: Metrics + config
experiment_id = f"exp_{timestamp}"
with open(f"{experiment_id}_config.yaml", "w") as f:
    yaml.dump(config, f)
with open(f"{experiment_id}_results.csv", "w") as f:
    f.write(f"{epoch},{train_loss},{val_loss}\n")
```

---

### Pitfall 2: Overwriting Checkpoints Without Versioning

**Symptom**: Always saving to "best_model.pt", can't recover earlier checkpoints.

**Why It Happens**:
- Disk space concerns (misguided)
- Only care about "best" model
- Don't anticipate evaluation bugs

**Fix**:
```python
# WRONG: Overwriting
torch.save(model.state_dict(), "best_model.pt")

# RIGHT: Versioned checkpoints
torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pt")
torch.save(model.state_dict(), f"checkpoints/best_model_val_acc_{val_acc:.4f}.pt")
```

---

### Pitfall 3: Using Print Instead of Logging

**Symptom**: Training crashes, all print output lost, can't debug.

**Why It Happens**:
- Print is simpler than logging
- Works for short scripts
- Doesn't anticipate crashes

**Fix**:
```python
# WRONG: Print statements
print(f"Epoch {epoch}: loss={loss}")

# RIGHT: Proper logging
import logging
logging.basicConfig(filename="training.log", level=logging.INFO)
logging.info(f"Epoch {epoch}: loss={loss}")
```

---

### Pitfall 4: No Git Tracking for Code Changes

**Symptom**: Can't reproduce result because code changed between experiments.

**Why It Happens**:
- Rapid iteration (uncommitted changes)
- "I'll commit later"
- Don't realize code version matters

**Fix**:
```python
# Log git commit at start of training
git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
config["git_commit"] = git_hash

# Better: Require clean git state
status = subprocess.check_output(['git', 'status', '--porcelain']).decode('ascii').strip()
if status:
    print("ERROR: Uncommitted changes detected!")
    print("Commit your changes before running experiments.")
    sys.exit(1)
```

---

### Pitfall 5: Not Tracking Random Seeds

**Symptom**: Same code, same hyperparameters, different results every time.

**Why It Happens**:
- Forget to set seed
- Set seed in one place but not others (PyTorch, NumPy, CUDA)
- Don't log seed value

**Fix**:
```python
# Set all seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Use seed from config
set_seed(config["seed"])

# Log seed
tracker.log_metrics({"seed": config["seed"]})
```

---

### Pitfall 6: Tracking Too Much Data (Storage Bloat)

**Symptom**: 100GB of logs for 50 experiments, can't store more.

**Why It Happens**:
- Logging every step (not just epoch)
- Saving all checkpoints (not just best)
- Logging high-resolution images

**Fix**:
```python
# Log at appropriate frequency
if global_step % 100 == 0:  # Every 100 steps, not every step
    tracker.log_metrics({"train/loss": loss})

# Save only best checkpoints
if val_acc > best_val_acc:  # Only when improving
    tracker.save_checkpoint(model, optimizer, epoch, val_acc)

# Downsample images
img_low_res = F.interpolate(img, size=(64, 64))  # Don't log 224x224
```

---

### Pitfall 7: No Experiment Naming Convention

**Symptom**: experiments/test, experiments/test2, experiments/final, experiments/final_final

**Why It Happens**:
- No planning for multiple experiments
- Naming feels unimportant
- "I'll organize later"

**Fix**:
```python
# Good naming convention
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"{config['model']}_{config['dataset']}_{timestamp}_lr{config['lr']}"
# Example: "resnet18_cifar10_20241030_120000_lr0.01"
```

---

### Pitfall 8: Not Tracking Evaluation Metrics

**Symptom**: Saved best model by training loss, but validation loss was actually increasing (overfitting).

**Why It Happens**:
- Only tracking training metrics
- Assuming training loss = model quality
- Not validating frequently enough

**Fix**:
```python
# Track both training and validation
tracker.log_metrics({
    "train/loss": train_loss,
    "val/loss": val_loss,  # Don't forget validation!
    "val/accuracy": val_acc,
})

# Save best model by validation metric, not training
if val_acc > best_val_acc:
    tracker.save_checkpoint(model, optimizer, epoch, val_acc)
```

---

### Pitfall 9: Local-Only Tracking for Team Projects

**Symptom**: Team members can't see each other's experiments, duplicate work.

**Why It Happens**:
- TensorBoard is local by default
- Don't realize collaboration tools exist
- Privacy concerns (unfounded)

**Fix**:
```python
# Use team-friendly tool
wandb.init(project="team-project")  # Everyone can see

# Or: Share TensorBoard logs
# scp -r runs/ shared-server:/path/
# tensorboard --logdir=/path/runs --host=0.0.0.0
```

---

### Pitfall 10: No Tracking Until "Important" Experiment

**Symptom**: First 20 experiments untracked, realize they had valuable insights.

**Why It Happens**:
- "Just testing" mentality
- Tracking feels like overhead
- Don't realize importance until later

**Fix**:
```python
# Track from experiment 1
# Even if "just testing", it takes 30 seconds to set up tracking
tracker = ExperimentTracker(config)

# Future you will thank past you
```

---

## Rationalization vs Reality Table

| User Rationalization | Reality | Recommendation |
|----------------------|---------|----------------|
| "I'll remember what I tried" | You won't (memory fails in hours) | Track from day 1, always |
| "Print statements are enough" | Lost on crash or terminal close | Use proper logging to file |
| "Only track final metrics" | Can't debug without intermediate data | Track every epoch minimum |
| "Just save best model" | Need checkpoints for analysis | Version all important checkpoints |
| "Tracking adds too much overhead" | <1% overhead for scalars | Log metrics, not raw data |
| "I only need the model file" | Need hyperparameters to understand it | Save config + model + metrics |
| "TensorBoard is too complex" | 2 lines of code to set up | Start simple, expand later |
| "I'll organize experiments later" | Never happens, chaos ensues | Use naming convention from start |
| "Git commits slow me down" | Uncommitted code = irreproducible | Commit before experiments |
| "Cloud tracking costs money" | Free tiers are generous | W&B free: 100GB, unlimited experiments |
| "I don't need reproducibility" | Your future self will | Track environment + seed + git |
| "Tracking is for production, not research" | Research needs it more (exploration) | Research = more experiments = more tracking |

---

## Red Flags (Likely to Fail)

1. **"I'll track it later"**
   - Reality: Later = never; best results are always untracked
   - Action: Track from experiment 1

2. **"Just using print statements"**
   - Reality: Lost on crash/close; can't analyze later
   - Action: Use logging framework or tracking tool

3. **"Only tracking the final metric"**
   - Reality: Can't debug convergence issues; no training curves
   - Action: Track every epoch at minimum

4. **"Saving to best_model.pt (overwriting)"**
   - Reality: Can't recover earlier checkpoints; evaluation bugs = disaster
   - Action: Version checkpoints with epoch/metric

5. **"Don't need to track hyperparameters"**
   - Reality: Metrics without config are meaningless
   - Action: Log config alongside metrics

6. **"Not tracking git commit"**
   - Reality: Code changes = irreproducible
   - Action: Log git hash, check for uncommitted changes

7. **"Random seed doesn't matter"**
   - Reality: Can cause 5%+ variance in results
   - Action: Set and log all seeds

8. **"TensorBoard/W&B is overkill for me"**
   - Reality: Setup takes 2 minutes, saves hours later
   - Action: Use simplest tool (TensorBoard), expand if needed

9. **"I'm just testing, don't need tracking"**
   - Reality: Best results come from "tests"
   - Action: Track everything, including tests

10. **"Team doesn't need to see my experiments"**
    - Reality: Collaboration requires transparency
    - Action: Use shared tracking (W&B, MLflow server)

---

## When This Skill Applies

**Strong Signals** (definitely use):
- Starting a new ML project (even "quick prototype")
- User asks "should I track this?"
- User lost their best result and can't reproduce
- Multiple experiments running (need comparison)
- Team collaboration (need to share results)
- User asks about TensorBoard, W&B, or MLflow
- Training crashes and user needs debugging data

**Weak Signals** (maybe use):
- User has tracking but it's incomplete
- Asking about reproducibility
- Discussing hyperparameter tuning (needs tracking)
- Long-running training (overnight, multi-day)

**Not Applicable**:
- Pure inference (no training)
- Single experiment already tracked
- Discussing model architecture only
- Data preprocessing questions (pre-training)

---

## Success Criteria

You've successfully applied this skill when:

1. **Complete Tracking**: Hyperparameters + metrics + artifacts + git + environment all logged
2. **Reproducibility**: Someone else (or future you) can reproduce the result from tracked info
3. **Tool Choice**: Selected appropriate tool (TensorBoard, W&B, MLflow) for use case
4. **Organization**: Experiments have clear naming, tagging, grouping
5. **Comparison**: Can compare experiments side-by-side, analyze hyperparameter impact
6. **Collaboration**: Team can see and discuss results (if team project)
7. **Minimal Overhead**: Tracking adds <5% runtime overhead
8. **Persistence**: Logs survive crashes, terminal closes, reboots
9. **Historical Analysis**: Can go back to any experiment and understand what was done
10. **Best Practices**: Git commits before experiments, seeds set, evaluation bugs impossible

**Final Test**: Can you reproduce the best result from 6 months ago using only the tracked information?

If YES: Excellent tracking! If NO: Gaps remain.

---

## Advanced Tracking Patterns

### 1. Multi-Run Experiments (Hyperparameter Sweeps)

When running many experiments systematically:

```python
# W&B Sweeps
sweep_config = {
    "method": "random",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"values": [0.001, 0.01, 0.1]},
        "batch_size": {"values": [32, 64, 128]},
        "optimizer": {"values": ["adam", "sgd"]},
    },
}

sweep_id = wandb.sweep(sweep_config, project="cifar10-sweep")

def train():
    run = wandb.init()
    config = wandb.config

    model = create_model(config)
    # ... training code ...
    wandb.log({"val_accuracy": val_acc})

wandb.agent(sweep_id, train, count=10)

# MLflow with Optuna
import optuna
import mlflow

def objective(trial):
    with mlflow.start_run(nested=True):
        lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        mlflow.log_params({"learning_rate": lr, "batch_size": batch_size})

        val_acc = train_and_evaluate(lr, batch_size)
        mlflow.log_metric("val_accuracy", val_acc)

        return val_acc

with mlflow.start_run():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    mlflow.log_param("best_params", study.best_params)
    mlflow.log_metric("best_accuracy", study.best_value)
```

---

### 2. Distributed Training Tracking

When training on multiple GPUs or machines:

```python
import torch.distributed as dist

def setup_distributed_tracking(rank, world_size):
    """Setup tracking for distributed training."""

    # Only rank 0 logs to avoid duplicates
    if rank == 0:
        tracker = ExperimentTracker(config)
    else:
        tracker = None

    return tracker

def train_distributed(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    tracker = setup_distributed_tracking(rank, world_size)

    model = DistributedDataParallel(model, device_ids=[rank])

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer)

        # Gather metrics from all ranks
        train_loss_tensor = torch.tensor(train_loss).cuda()
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = train_loss_tensor.item() / world_size

        # Only rank 0 logs
        if rank == 0 and tracker:
            tracker.log_metrics({
                "train/loss": avg_train_loss,
                "epoch": epoch,
            })

    if rank == 0 and tracker:
        tracker.finish()

    dist.destroy_process_group()
```

---

### 3. Experiment Resumption

Tracking setup for resumable experiments:

```python
class ResumableExperimentTracker(ExperimentTracker):
    """Experiment tracker with resume support."""

    def __init__(self, config, checkpoint_path=None):
        super().__init__(config)

        self.checkpoint_path = checkpoint_path

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.resume_from_checkpoint()

    def resume_from_checkpoint(self):
        """Resume tracking from saved checkpoint."""
        checkpoint = torch.load(self.checkpoint_path)

        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", float('-inf'))

        self.logger.info(f"Resumed from checkpoint: step={self.global_step}")

    def save_checkpoint(self, model, optimizer, epoch, metric_value, metric_name="val_accuracy"):
        """Save checkpoint with tracker state."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            metric_name: metric_value,
            "config": self.config,
        }

        checkpoint_path = self.exp_dir / "checkpoints" / "latest.pt"
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

        # Also save best
        if metric_value > self.best_metric:
            self.best_metric = metric_value
            best_path = self.exp_dir / "checkpoints" / "best.pt"
            torch.save(checkpoint, best_path)

        return checkpoint_path

# Usage
tracker = ResumableExperimentTracker(config, checkpoint_path="checkpoints/latest.pt")

# Training continues from where it left off
for epoch in range(start_epoch, num_epochs):
    # ... training ...
    tracker.save_checkpoint(model, optimizer, epoch, val_acc)
```

---

### 4. Experiment Comparison and Analysis

Programmatic experiment analysis:

```python
def analyze_experiments(project_name):
    """Analyze all experiments in a project."""

    # W&B
    import wandb
    api = wandb.Api()
    runs = api.runs(project_name)

    # Extract data
    data = []
    for run in runs:
        data.append({
            "name": run.name,
            "learning_rate": run.config.get("learning_rate"),
            "batch_size": run.config.get("batch_size"),
            "val_accuracy": run.summary.get("val_accuracy"),
            "train_time": run.summary.get("_runtime"),
        })

    df = pd.DataFrame(data)

    # Analysis
    print("Top 5 experiments by accuracy:")
    print(df.nlargest(5, "val_accuracy"))

    # Hyperparameter impact
    print("\nAverage accuracy by learning rate:")
    print(df.groupby("learning_rate")["val_accuracy"].mean())

    # Visualization
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Learning rate vs accuracy
    axes[0].scatter(df["learning_rate"], df["val_accuracy"])
    axes[0].set_xlabel("Learning Rate")
    axes[0].set_ylabel("Validation Accuracy")
    axes[0].set_xscale("log")

    # Batch size vs accuracy
    axes[1].scatter(df["batch_size"], df["val_accuracy"])
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Validation Accuracy")

    plt.tight_layout()
    plt.savefig("experiment_analysis.png")

    return df

# Run analysis
df = analyze_experiments("team/cifar10-sota")
```

---

### 5. Data Versioning Integration

Tracking data versions alongside experiments:

```python
import hashlib

def hash_dataset(dataset_path):
    """Compute hash of dataset for versioning."""
    hasher = hashlib.sha256()

    # Hash dataset files
    for file in sorted(Path(dataset_path).rglob("*")):
        if file.is_file():
            with open(file, "rb") as f:
                hasher.update(f.read())

    return hasher.hexdigest()

# Track data version
data_version = hash_dataset("data/cifar10")
config["data_version"] = data_version

tracker = ExperimentTracker(config)

# Or use DVC
"""
# Initialize DVC
dvc init

# Track data
dvc add data/cifar10
git add data/cifar10.dvc

# Log DVC hash in experiment
with open("data/cifar10.dvc") as f:
    dvc_config = yaml.safe_load(f)
    data_hash = dvc_config["outs"][0]["md5"]
    config["data_hash"] = data_hash
"""
```

---

### 6. Artifact Management Best Practices

Organizing and managing experiment artifacts:

```python
class ArtifactManager:
    """Manages experiment artifacts (models, plots, logs)."""

    def __init__(self, experiment_dir):
        self.exp_dir = Path(experiment_dir)

        # Create subdirectories
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.figures_dir = self.exp_dir / "figures"
        self.logs_dir = self.exp_dir / "logs"

        for d in [self.checkpoints_dir, self.figures_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, checkpoint, name):
        """Save checkpoint with automatic cleanup."""
        path = self.checkpoints_dir / f"{name}.pt"
        torch.save(checkpoint, path)

        # Keep only last N checkpoints (except best)
        self._cleanup_checkpoints(keep_n=5)

        return path

    def _cleanup_checkpoints(self, keep_n=5):
        """Keep only recent checkpoints to save space."""
        checkpoints = sorted(
            self.checkpoints_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Delete old checkpoints (keep best + last N)
        for ckpt in checkpoints[keep_n:]:
            if "best" not in ckpt.name:
                ckpt.unlink()

    def save_figure(self, fig, name, step=None):
        """Save matplotlib figure with metadata."""
        if step is not None:
            filename = f"{name}_step_{step}.png"
        else:
            filename = f"{name}.png"

        path = self.figures_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")

        return path

    def get_artifact_summary(self):
        """Get summary of stored artifacts."""
        summary = {
            "num_checkpoints": len(list(self.checkpoints_dir.glob("*.pt"))),
            "num_figures": len(list(self.figures_dir.glob("*.png"))),
            "total_size_mb": sum(
                f.stat().st_size for f in self.exp_dir.rglob("*") if f.is_file()
            ) / (1024 * 1024),
        }
        return summary

# Usage
artifacts = ArtifactManager(experiment_dir)
artifacts.save_checkpoint(checkpoint, f"checkpoint_epoch_{epoch}")
artifacts.save_figure(fig, "training_curve")
print(artifacts.get_artifact_summary())
```

---

### 7. Real-Time Monitoring and Alerts

Setup alerts for experiment issues:

```python
# W&B Alerts
import wandb

wandb.init(project="cifar10")

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer)

    wandb.log({"train/loss": train_loss, "epoch": epoch})

    # Alert on divergence
    if math.isnan(train_loss) or train_loss > 10:
        wandb.alert(
            title="Training Diverged",
            text=f"Loss is {train_loss} at epoch {epoch}",
            level=wandb.AlertLevel.ERROR,
        )
        break

    # Alert on milestone
    if val_acc > 0.90:
        wandb.alert(
            title="90% Accuracy Reached!",
            text=f"Validation accuracy: {val_acc:.2%}",
            level=wandb.AlertLevel.INFO,
        )

# Slack integration
def send_slack_alert(message, webhook_url):
    """Send alert to Slack."""
    import requests
    requests.post(webhook_url, json={"text": message})

# Email alerts
def send_email_alert(subject, body, to_email):
    """Send email alert."""
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["To"] = to_email
    msg.set_content(body)

    # Send via SMTP
    with smtplib.SMTP("localhost") as s:
        s.send_message(msg)
```

---

## Common Integration Patterns

### Pattern 1: Training Script with Complete Tracking

```python
#!/usr/bin/env python3
"""
Complete training script with experiment tracking.
"""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiment_tracker import ExperimentTracker
from models import create_model
from data import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--name", type=str, help="Experiment name")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Initialize tracking
    tracker = ExperimentTracker(
        config=config,
        experiment_name=args.name,
        use_wandb=True,
        use_tensorboard=True,
    )

    # Setup training
    model = create_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader = load_dataset(config)

    # Resume if checkpoint provided
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        tracker.logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_val_acc = 0.0
    for epoch in range(start_epoch, config["num_epochs"]):
        # Train
        model.train()
        train_losses = []
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Log every N batches
            if batch_idx % config.get("log_interval", 100) == 0:
                tracker.log_metrics({
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]['lr'],
                }, step=epoch * len(train_loader) + batch_idx)

        # Validate
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_losses.append(loss.item())

                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        val_acc = correct / total

        # Log epoch metrics
        tracker.log_metrics({
            "train/loss_epoch": train_loss,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "epoch": epoch,
        })

        # Save checkpoint
        tracker.save_checkpoint(model, optimizer, epoch, val_acc)

        # Update best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            tracker.logger.info(f"New best accuracy: {val_acc:.4f}")

        # Early stopping
        if epoch > 50 and val_acc < 0.5:
            tracker.logger.warning("Model not improving, stopping early")
            break

    # Log final results
    tracker.log_metrics({"best_val_accuracy": best_val_acc})
    tracker.logger.info(f"Training completed. Best accuracy: {best_val_acc:.4f}")

    # Cleanup
    tracker.finish()

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
# Train new model
python train.py --config configs/resnet18.yaml --name resnet18_baseline

# Resume training
python train.py --config configs/resnet18.yaml --resume experiments/resnet18_baseline/checkpoints/latest.pt
```

---

## Further Reading

- **Papers**:
  - "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., 2015)
  - "Reproducibility in Machine Learning" (Pineau et al., 2020)
  - "A Step Toward Quantifying Independently Reproducible ML Research" (Dodge et al., 2019)

- **Tool Documentation**:
  - TensorBoard: https://www.tensorflow.org/tensorboard
  - Weights & Biases: https://docs.wandb.ai/
  - MLflow: https://mlflow.org/docs/latest/index.html
  - DVC (Data Version Control): https://dvc.org/doc
  - Hydra (Config Management): https://hydra.cc/docs/intro/

- **Best Practices**:
  - Papers With Code (Reproducibility): https://paperswithcode.com/
  - ML Code Completeness Checklist: https://github.com/paperswithcode/releasing-research-code
  - Experiment Management Guide: https://neptune.ai/blog/experiment-management

- **Books**:
  - "Designing Machine Learning Systems" by Chip Huyen (Chapter on Experiment Tracking)
  - "Machine Learning Engineering" by Andriy Burkov (Chapter on MLOps)

---

**Remember**: Experiment tracking is insurance. It costs 1% overhead but saves 100% when disaster strikes. Track from day 1, track everything, and your future self will thank you.
