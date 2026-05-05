
# Experiment Tracking and Versioning Skill

## When to Use This Skill

Use this skill when you observe these symptoms:

**Reproducibility symptoms:**
- Cannot reproduce a good result from last week (which hyperparameters?)
- Someone asks "which model is in production?" and you do not know
- Lost track of which data version produced which model
- Experiments tracked in spreadsheets, text files, or terminal scrollback

**Collaboration symptoms:**
- Multiple people running experiments, no central tracking
- Cannot compare runs across team members
- Lost experiments when someone leaves the team
- No visibility into what others are trying

**Production symptoms:**
- Cannot trace predictions back to model version and training data
- Need to roll back model but do not know which previous version was good
- Compliance requires audit trail (data → model → predictions)
- Cannot A/B test models because tracking is unclear

**LLM-specific symptoms:**
- Lost the prompt that produced your best eval-set score
- Multiple fine-tune runs with no easy way to compare
- RAG configuration (chunker, embedding tier, retriever, reranker) drifts and you can't pin "what worked"
- Eval-set composition changed mid-experiment, invalidating prior comparisons

**When NOT to use this skill:**
- Single experiment, one-off analysis (no need for tracking infrastructure)
- Prototyping where reproducibility is not yet important
- Already have robust experiment tracking working well

## Core Principle

**If you cannot reproduce it, it does not exist.**

Experiment tracking captures everything needed to reproduce a result:
- **Code version** (git commit hash)
- **Data version** (dataset hash, version tag, or DVC pointer)
- **Hyperparameters** (learning rate, batch size, etc.)
- **Environment** (Python version, library versions, hardware)
- **Random seeds** (for deterministic results)
- **Metrics** (accuracy, loss over time)
- **Artifacts** (model checkpoints, predictions, reports)
- **Lineage** (which run produced which model produced which prediction)

**Formula:** Good tracking = Code + Data + Config + Environment + Seeds + Metrics + Artifacts + Lineage

The skill is building a system where **every experiment is automatically reproducible**.

## Experiment Tracking Framework

```
┌────────────────────────────────────────────┐
│   1. Recognize Tracking Need               │
│   "Cannot reproduce" OR "Which model?"     │
└──────────────┬─────────────────────────────┘
               ▼
┌────────────────────────────────────────────┐
│   2. Choose Tracking Tool(s)               │
│   See selection matrix later in sheet      │
└──────────────┬─────────────────────────────┘
               ▼
┌────────────────────────────────────────────┐
│   3. Instrument Training Code              │
│   Log params, metrics, artifacts, lineage  │
└──────────────┬─────────────────────────────┘
               ▼
┌────────────────────────────────────────────┐
│   4. Version Models + Data + Prompts       │
│   Model registry + DVC + prompt registry   │
└──────────────┬─────────────────────────────┘
               ▼
┌────────────────────────────────────────────┐
│   5. Validate Reproducibility              │
│   Pressure-test: can you recreate any run? │
└────────────────────────────────────────────┘
```


## Part 1: RED — Without Experiment Tracking (Failures)

### Failure 1: Cannot Reproduce Best Run

```python
# train_model.py — NO TRACKING
def train_model():
    # Which augmentation? Which LR? Which batch size? Which seed?
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    train_loader = DataLoader(datasets.CIFAR10('./data', train=True, download=True, transform=transform),
                              batch_size=128, shuffle=True)
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, 10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(50):
        for x, y in train_loader:
            ...
    torch.save(model.state_dict(), 'model_best.pth')   # overwrites previous!
```

Hyperparameters that produced "94.2% last week" are gone the moment the terminal closes.

### Failure 2: No Model Versioning

`production_model.pth` gets overwritten on every deploy. No metadata, no rollback, no audit. When a bug surfaces, no one can answer "which model is in production?"

### Failure 3: Manual Artifact Management

50 experiments → directories full of `model_best.pth`, `checkpoint_50.pth`, `model_old.pth`, `predictions.npy` with no link back to runs, hyperparameters, or metrics.

### Failure 4: No Lineage Tracking

Production prediction looks wrong. Which model? Which data? Which preprocessing? `data/production_data.csv` was overwritten daily; `models/production/model.pth` was overwritten weekly. The information needed to reconstruct what happened is gone.

### Failure 5: Cannot Compare Runs

20 hyperparameter combinations, results in `results.txt`. No learning curves, no validation metrics, no environment metadata. "Which one was best?" requires manually parsing logs.

### Bonus: Compliance Nightmare

Auditor asks: "For the prediction made on 2026-04-15, prove which model and data were used." Without tracking, the answer is "we have a `.pth` file but no version history" — which is a compliance failure that can shut down the project in regulated industries.


## Part 2: GREEN — Tooling Landscape

This section surveys the experiment-tracking and model-registry tools you should be choosing between, then shows worked code for the two most common open-source options (MLflow and Weights & Biases). Other tools have full SDKs and broadly similar instrumentation patterns; the docs links below are the canonical starting points.

### Tool Roster

**MLflow** (open-source) — Tracking + Model Registry + Projects + Models + Serving + Evaluation. Self-hostable; broad framework integrations; in MLflow 2.x the registry uses model **aliases** (e.g. `@champion`, `@challenger`) rather than the deprecated stage labels (`Staging`, `Production`). MLflow 2.x also added an LLM-evaluation API (`mlflow.evaluate(model_type="question-answering"|"text-summarization"|...)`) and prompt-engineering tracking. Docs: <https://mlflow.org/docs/latest/index.html>. Site: <https://mlflow.org>. Managed: Databricks, plus the OSS server runs anywhere.

**Weights & Biases** (W&B) — Cloud-hosted (also self-hostable) experiment tracking with Models (registry), Sweeps (hyperparameter search), Reports (publishable analyses), Artifacts (versioned data/model assets), and **Weave** (W&B's LLM-app tracing and evaluation product). Strong UI, real-time visualization, advanced sweep algorithms. Docs: <https://docs.wandb.ai>. Weave docs: <https://weave-docs.wandb.ai>. Site: <https://wandb.ai>.

**Comet** — experiment tracking + model registry + production monitoring (Comet MPM). **Opik** is Comet's open-source LLM-evaluation and tracing product, complementing the broader Comet platform. Docs: <https://www.comet.com/docs/v2/>. Opik: <https://www.comet.com/site/products/opik/> and <https://github.com/comet-ml/opik>.

**Neptune** — experiment metadata store oriented toward research-scale workloads (millions of runs, long training jobs, large metric series). Strong for DL teams that need to compare hundreds of runs efficiently. Docs: <https://docs.neptune.ai>. Site: <https://neptune.ai>.

**ClearML** — open-source MLOps (experiment tracking + orchestration + data versioning + model serving) from Allegro AI. Fewer integrations than MLflow but a more cohesive end-to-end story for self-hosted teams. Docs: <https://clear.ml/docs>. Site: <https://clear.ml>.

**Hugging Face Hub** — increasingly the de facto registry for open-weight models, datasets, and now Spaces (apps). First-class for LLM and embedding-model artifacts; supports private repos, model cards, dataset cards, gated access, and integrates with Transformers, Diffusers, Sentence-Transformers, etc. via `from_pretrained` / `push_to_hub`. Docs: <https://huggingface.co/docs/hub/index>.

**DVC** (Data Version Control) — git-friendly data and pipeline versioning; tracks large data/model files in remote storage (S3 / GCS / Azure / SSH) with hashes referenced from git. **DVC Studio** is the visualization layer (replaces the older Iterative Studio branding). Pairs naturally with MLflow, W&B, etc. — DVC versions the data, the tracker logs the data hash. Docs: <https://dvc.org/doc>. Studio: <https://studio.datachain.ai/> (formerly studio.iterative.ai).

**lakeFS** — git-like versioning over object stores (S3 / GCS / Azure Blob), branches and merges across petabyte-scale data. Strong for data-engineering teams. Docs: <https://docs.lakefs.io>.

**BentoML model registry** — registry tied to deployment; `BentoML` is primarily a serving framework but its registry holds the served models. Useful when serving is BentoML and you don't want a separate registry. Docs: <https://docs.bentoml.com>.

**LakeFS / DVC / lakeFS / Pachyderm / Quilt / Delta Lake / Apache Iceberg / Apache Hudi** — data-versioning siblings; pick by storage backend and team scale. Pachyderm: <https://docs.pachyderm.com>. Iceberg: <https://iceberg.apache.org>. Delta Lake: <https://docs.delta.io>.

**Cloud-managed registries:**
- **AWS SageMaker Model Registry** — integrated with SageMaker Pipelines and Endpoints. Docs: <https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html>
- **GCP Vertex AI Model Registry** — Docs: <https://cloud.google.com/vertex-ai/docs/model-registry/introduction>
- **Azure ML Model Registry** — Docs: <https://learn.microsoft.com/azure/machine-learning/concept-model-management-and-deployment>
- **Databricks Unity Catalog Models** — versioned model assets with table-style governance. Docs: <https://docs.databricks.com/aws/en/mlflow/models-in-uc.html>


### Solution 1: MLflow for Tracking + Registry (open-source self-host)

**When to choose MLflow:**
- Self-hosted infrastructure (data privacy, compliance)
- Open-source preference / no vendor lock-in
- Want a unified server for tracking + registry without running multiple tools
- Already on Databricks (managed MLflow is built-in)

**Setup:**

```bash
pip install mlflow
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri postgresql://... \
  --artifacts-destination s3://my-bucket/mlflow-artifacts \
  --serve-artifacts
```

UI: <http://localhost:5000>. For production, use a relational backend (Postgres / MySQL) and an object-store artifact root.

**MLflow-instrumented training (works in MLflow 2.x):**

```python
import os, hashlib, subprocess
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import mlflow, mlflow.pytorch

def compute_data_hash(p: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p.rglob('*')):
        if f.is_file():
            with open(f, 'rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 16), b""):
                    h.update(chunk)
    return h.hexdigest()

def train():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("cifar10-classification")

    # MLflow 2.x: autolog covers most PyTorch/Lightning/sklearn metrics automatically.
    mlflow.pytorch.autolog(log_models=False, log_datasets=False)

    config = dict(
        batch_size=128, learning_rate=0.001, epochs=50,
        optimizer="sgd", momentum=0.9, model_arch="resnet18",
        pretrained=True, image_size=32, random_seed=42,
    )

    with mlflow.start_run(run_name="resnet18-baseline") as run:
        mlflow.log_params(config)

        # Code version
        try:
            commit = subprocess.check_output(['git','rev-parse','HEAD'],
                                             stderr=subprocess.DEVNULL).decode().strip()
            mlflow.log_param('git_commit', commit)
        except Exception:
            pass

        # Data version
        data_path = Path('./data/cifar10')
        mlflow.log_param('data_hash', compute_data_hash(data_path))

        # Environment
        mlflow.log_param('pytorch_version', torch.__version__)
        mlflow.log_param('cuda_available', torch.cuda.is_available())

        # Reproducibility
        torch.manual_seed(config['random_seed'])
        torch.cuda.manual_seed_all(config['random_seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Data
        tx = transforms.Compose([transforms.RandomHorizontalFlip(),
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,))])
        train_ds = datasets.CIFAR10('./data', train=True,  download=True, transform=tx)
        val_ds   = datasets.CIFAR10('./data', train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))]))
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'], shuffle=False)

        # Model
        model = models.resnet18(weights='DEFAULT' if config['pretrained'] else None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        opt = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                              momentum=config['momentum'])
        loss_fn = nn.CrossEntropyLoss()

        best_val = 0.0
        for epoch in range(config['epochs']):
            model.train(); tl = tc = tt = 0
            for x, y in train_loader:
                opt.zero_grad(); out = model(x); loss = loss_fn(out, y)
                loss.backward(); opt.step()
                tl += loss.item(); tt += y.size(0)
                tc += (out.argmax(1) == y).sum().item()
            train_loss, train_acc = tl/len(train_loader), 100*tc/tt

            model.eval(); vl = vc = vt = 0
            with torch.no_grad():
                for x, y in val_loader:
                    out = model(x); vl += loss_fn(out, y).item()
                    vt += y.size(0); vc += (out.argmax(1) == y).sum().item()
            val_loss, val_acc = vl/len(val_loader), 100*vc/vt

            mlflow.log_metrics({'train_loss': train_loss, 'train_acc': train_acc,
                                'val_loss': val_loss, 'val_acc': val_acc}, step=epoch)

            if val_acc > best_val:
                best_val = val_acc
                mlflow.log_metric('best_val_acc', best_val)

        # Log model artifact + register in registry
        mlflow.pytorch.log_model(model, name="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, "cifar10-resnet18")
        # MLflow 2.x: use ALIASES (the deprecated stage-transition API is being removed)
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name="cifar10-resnet18",
            alias="challenger",
            version=mv.version,
        )
        # Tag with metadata
        client.set_model_version_tag(
            name="cifar10-resnet18", version=mv.version,
            key="val_acc", value=str(best_val),
        )
        client.set_model_version_tag(
            name="cifar10-resnet18", version=mv.version,
            key="data_hash", value=compute_data_hash(data_path)[:16],
        )

        return run.info.run_id

if __name__ == "__main__":
    train()
```

**MLflow Registry: aliases vs stages.** Stage labels (`Staging`, `Production`, `Archived`) and `transition_model_version_stage()` are deprecated as of MLflow 2.9 and slated for removal. Use **aliases** (`set_registered_model_alias`, `delete_registered_model_alias`, and load via `models:/<name>@<alias>`) and **tags** for metadata. Migration guide: <https://mlflow.org/docs/latest/model-registry.html#deprecated-using-model-stages>.

```python
# Promotion using aliases (the modern pattern)
client = mlflow.tracking.MlflowClient()

# Mark a candidate
client.set_registered_model_alias("cifar10-resnet18", "challenger", version=4)

# After eval gates pass, promote to champion (replaces any existing @champion)
client.set_registered_model_alias("cifar10-resnet18", "champion", version=4)

# Load champion at serving time
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/cifar10-resnet18@champion")

# Roll back: just point @champion at an older version
client.set_registered_model_alias("cifar10-resnet18", "champion", version=3)
```


### Solution 2: Weights & Biases for Collaboration

**When to choose W&B:**
- Team collaboration with shared visualization
- Want a hosted control plane (or use W&B Server self-hosted)
- Need real-time training dashboards and Reports for stakeholders
- Hyperparameter sweeps with advanced search algorithms (Bayesian, Hyperband)
- Building LLM apps and want **Weave** for tracing/evaluation alongside experiments

**W&B-instrumented training:**

```python
import wandb, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

config = dict(batch_size=128, learning_rate=0.001, epochs=50,
              optimizer="sgd", momentum=0.9, model_arch="resnet18",
              pretrained=True, random_seed=42)

run = wandb.init(project="cifar10-classification",
                 name="resnet18-baseline",
                 config=config,
                 tags=["resnet","baseline","cifar10"])

torch.manual_seed(config['random_seed'])
torch.cuda.manual_seed_all(config['random_seed'])

tx_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                               transforms.RandomCrop(32, padding=4),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])
tx_val   = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))])
train_ds = datasets.CIFAR10('./data', train=True,  download=True, transform=tx_train)
val_ds   = datasets.CIFAR10('./data', train=False, download=True, transform=tx_val)
train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'], shuffle=False)

model = models.resnet18(weights='DEFAULT' if config['pretrained'] else None)
model.fc = nn.Linear(model.fc.in_features, 10)
wandb.watch(model, log='all', log_freq=100)
opt = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                      momentum=config['momentum'])
loss_fn = nn.CrossEntropyLoss()

best_val = 0.0
for epoch in range(config['epochs']):
    model.train()
    for x, y in train_loader:
        opt.zero_grad(); out = model(x); loss = loss_fn(out, y)
        loss.backward(); opt.step()
    model.eval(); vc = vt = 0
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x); vt += y.size(0); vc += (out.argmax(1) == y).sum().item()
    val_acc = 100 * vc / vt
    wandb.log({"epoch": epoch, "val_acc": val_acc})
    if val_acc > best_val:
        best_val = val_acc
        ckpt = "checkpoints/best.pth"
        torch.save({"epoch": epoch, "model": model.state_dict()}, ckpt)
        # Versioned artifact
        art = wandb.Artifact(name=f"model-{run.id}", type="model",
                             metadata={"val_acc": val_acc, "epoch": epoch})
        art.add_file(ckpt)
        wandb.log_artifact(art, aliases=["latest", "best"])

wandb.summary["best_val_acc"] = best_val
wandb.finish()
```

**W&B Models (registry).** Promote a logged-artifact to the registry under a project-scoped or org-scoped collection; track lineage from training run to registry version. Docs: <https://docs.wandb.ai/guides/registry/>.

**W&B Sweeps:**

```python
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 1e-4, "max": 1e-1, "distribution": "log_uniform_values"},
        "batch_size":    {"values": [32, 64, 128, 256]},
        "optimizer":     {"values": ["sgd", "adam", "adamw"]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="cifar10-classification")
wandb.agent(sweep_id, function=train, count=40)
```


### Solution 3: Hugging Face Hub as a Model Registry

For LLMs and embedding models specifically, the **Hugging Face Hub** is now the most common registry. Push a fine-tuned checkpoint, model card, eval results, and tokenizer atomically; pull at serving time with `from_pretrained`.

```python
# Pushing a fine-tuned model to a private Hub repo
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login

login(token=os.environ["HF_TOKEN"])
model = AutoModelForCausalLM.from_pretrained("./checkpoints/final")
tok   = AutoTokenizer.from_pretrained("./checkpoints/final")

# Push model + tokenizer + auto-generated model card
model.push_to_hub("my-org/my-finetune-v3", private=True)
tok.push_to_hub  ("my-org/my-finetune-v3", private=True)

# Attach extra metadata (eval scores, training data hash) to the model card
api = HfApi()
api.upload_file(
    path_or_fileobj=open("eval_results.json","rb"),
    path_in_repo="eval_results.json",
    repo_id="my-org/my-finetune-v3",
)
```

Docs: <https://huggingface.co/docs/hub/models-uploading> and the model-card spec at <https://huggingface.co/docs/hub/model-cards>. For broader integration with experiment trackers, the Hub publishes integration guides (W&B, MLflow, Comet, ClearML) at <https://huggingface.co/docs/hub/models-libraries>.


### Solution 4: Data Versioning

**DVC (Data Version Control):**

```bash
pip install 'dvc[s3]'
dvc init
dvc add data/cifar10
git add data/cifar10.dvc .gitignore
git commit -m "Add CIFAR-10 v1.0"
git tag data-v1.0

# Configure remote storage
dvc remote add -d storage s3://my-bucket/dvc-store
dvc push

# Reproduce on another machine / time
git checkout data-v1.0 && dvc pull
```

DVC integrates with most experiment trackers — log the DVC file's hash as a parameter and you have a reproducible pointer back to the exact data version.

**Hash-based fallback:**

```python
import hashlib
from pathlib import Path

def compute_data_hash(p: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p.rglob('*')):
        if f.is_file():
            with open(f, 'rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 16), b""):
                    h.update(chunk)
    return h.hexdigest()

mlflow.log_param('data_hash', compute_data_hash(Path('./data/cifar10')))
```

For data-engineering scale, see **lakeFS**, **Pachyderm**, **Iceberg**, **Delta Lake**, **Hudi** (citations in the tool roster above).


### Solution 5: Lineage Tracking (Data → Model → Predictions)

```python
import mlflow, hashlib, json
from pathlib import Path
from datetime import datetime

class LineageTracker:
    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()

    def _hash(self, p: Path) -> str:
        h = hashlib.sha256()
        if p.is_file():
            with open(p,'rb') as f:
                for chunk in iter(lambda: f.read(1<<16), b""): h.update(chunk)
        else:
            for f in sorted(p.rglob('*')):
                if f.is_file():
                    with open(f,'rb') as fh:
                        for chunk in iter(lambda: fh.read(1<<16), b""): h.update(chunk)
        return h.hexdigest()

    def track_training(self, data_path: Path, hyperparams: dict,
                       metrics: dict, model_path: Path) -> str:
        with mlflow.start_run() as run:
            mlflow.log_params(hyperparams)
            mlflow.log_params({"data_hash": self._hash(data_path),
                               "model_hash": self._hash(model_path),
                               "data_path": str(data_path)})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(model_path))
            return run.info.run_id

    def track_inference(self, training_run_id: str,
                        input_data: Path, predictions: Path) -> str:
        ts = datetime.utcnow().isoformat()
        with mlflow.start_run(run_name=f"inference-{ts}") as run:
            mlflow.log_params({
                "training_run_id":     training_run_id,
                "input_data_hash":     self._hash(input_data),
                "predictions_hash":    self._hash(predictions),
                "inference_timestamp": ts,
            })
            mlflow.log_artifact(str(predictions))
            return run.info.run_id

    def get_lineage(self, run_id: str) -> dict:
        run = self.client.get_run(run_id)
        out = {"run_id": run_id, "params": run.data.params,
               "metrics": run.data.metrics, "tags": run.data.tags}
        if "training_run_id" in run.data.params:
            out["training"] = self.client.get_run(run.data.params["training_run_id"]).data.params
        return out
```

For graph-based lineage across systems (data warehouse → pipeline → model → BI dashboard), look at **OpenLineage** (<https://openlineage.io>) and the lineage features built into Dagster, Marquez, DataHub (<https://datahubproject.io>), and OpenMetadata (<https://docs.open-metadata.org>).


### Solution 6: Selection Matrix

| Need | First-choice tool |
|---|---|
| OSS, self-host, tracking + registry in one server | **MLflow** |
| Beautiful UI, real-time visualization, sweeps | **Weights & Biases** |
| Research-scale runs (10k+ runs, long training) | **Neptune** |
| End-to-end OSS MLOps including orchestration | **ClearML** |
| Open-weight model registry, esp. for LLMs | **Hugging Face Hub** |
| Data + pipeline versioning alongside any tracker | **DVC** (+ optional DVC Studio) |
| Petabyte-scale data versioning | **lakeFS** / **Pachyderm** |
| Eval-and-experiment for LLM apps with strong UI | **Braintrust** / **Weights & Biases Weave** |
| Open-source LLM eval + tracing | **Opik** (Comet) / **Langfuse** / **DeepEval** |
| Already on AWS / GCP / Azure / Databricks | **SageMaker / Vertex / Azure ML / Unity Catalog** registries |
| Registry tied to BentoML serving | **BentoML model registry** |

**Common stacks:**
- **Solo / small team, OSS:** MLflow + DVC.
- **Research team, hosted UI:** W&B + DVC.
- **LLM app team:** W&B Weave **or** Langfuse + Hugging Face Hub for fine-tunes + Promptfoo / DeepEval for CI.
- **Big-cloud-committed team:** managed cloud registry + DVC for non-warehouse data.
- **Enterprise self-host:** MLflow on K8s with Postgres + S3 + Unity Catalog or DataHub for governance.


## Part 3: Experiment Tracking for LLM Work

LLM development has its own taxonomy of "experiments" — prompts, eval-set runs, fine-tunes, RAG configurations — that don't fit neatly into the classical "train a model, log metrics per epoch" mental model. Cross-references: see `yzmir-llm-specialist/llm-evaluation-metrics.md` for eval methodology, `llm-finetuning-strategies.md` for fine-tune mechanics, `rag-architecture-patterns.md` for RAG architecture, `prompt-engineering-patterns.md` for prompt design.

### What Counts as an "Experiment" for LLM Apps?

| Experiment type | What you vary | What you measure |
|---|---|---|
| Prompt iteration | System prompt, few-shot examples, format instructions | Eval-set faithfulness, answer-relevance, format compliance, judge scores, cost |
| Model-tier comparison | Capability tier (flagship / mid / cheap-fast) | Same eval set, latency, cost, refusal rate |
| RAG configuration | Chunker, chunk size, embedding tier, top-k, reranker, context budget | Recall@k, MRR, end-to-end faithfulness, retrieval latency |
| Fine-tune training | Base model, LoRA rank, training data slice, epochs, lr | Eval-set scores vs base, perplexity, training loss curves |
| Agent design | Tool set, planner prompt, max iterations | Task success rate, tool-call success rate, mean iterations to success |

### Versioning Prompts as Experiments

Prompts are configuration *and* code; treat them like both. Concrete patterns:

- **Plain-git prompts** — `prompts/*.md` or `*.j2` (Jinja2) checked in alongside model code; CI runs eval set on every change.
- **Prompt registries with labels** — Langfuse Prompt Management, PromptLayer, LangSmith Prompts, Helicone Prompts. Lets you change prompts without redeploying code by labeling versions (`production`, `experiment-A`).
- **Hybrid** — git for canonical source, prompt-registry for runtime A/B traffic splitting.

```python
# Example: log a prompt experiment to MLflow
import mlflow, hashlib

prompt_v3 = open("prompts/customer_support_v3.md").read()
prompt_hash = hashlib.sha256(prompt_v3.encode()).hexdigest()[:12]

with mlflow.start_run(run_name=f"prompt-v3-{prompt_hash}"):
    mlflow.log_param("prompt_hash", prompt_hash)
    mlflow.log_param("prompt_version", "v3")
    mlflow.log_param("model_tier", "flagship")
    mlflow.log_param("eval_set_version", "2026-04-eval-v2")
    mlflow.log_text(prompt_v3, "prompt.md")     # full text as artifact
    # ... run eval set, log metrics ...
    mlflow.log_metrics({
        "faithfulness": 0.91,
        "answer_relevance": 0.88,
        "judge_overall": 0.87,
        "cost_per_query_usd": 0.012,
    })
```

### Versioning Eval Sets

The eval set itself is a versioned artifact — silently mutating it invalidates every prior comparison. Treat it like a dataset:

- Store as a versioned file in DVC, S3 with object versioning, or a Hugging Face dataset repo.
- Log `eval_set_version` (or hash) on every experiment run.
- When evolving the eval set, append (don't modify) and version-bump.

### Comparing Fine-Tuning Runs

Use the same tracker (MLflow / W&B / Comet / Neptune) you use for classical training. Fine-tune-specific things to log:

- Base model identifier (provider + capability tier; resolve to actual model ID at run time, log both)
- Training data hash and version (DVC pointer, or HF dataset repo + revision)
- LoRA / adapter config (rank, alpha, target modules, bias)
- Training framework version (Transformers, PEFT, Accelerate, Axolotl, Unsloth)
- Eval-set version and per-task scores (vs base model and vs previous fine-tune)
- Inference cost/latency on a fixed eval slice

The `experiment-tracking` UI then lets you compare three fine-tunes side-by-side on the same eval set with the same metrics.

### Tracking RAG-Pipeline Experiments

Treat the RAG pipeline as a parameterized object: `(chunker, chunk_size, embedding_tier, top_k, reranker, context_budget, prompt_version)`. Log every dimension as a parameter; log retrieval and end-to-end metrics. The same tracker can hold all variants — don't fragment between "search experiments" and "model experiments."

```python
with mlflow.start_run(run_name="rag-config-v7"):
    mlflow.log_params({
        "chunker":          "semantic",
        "chunk_size":       512,
        "chunk_overlap":    64,
        "embedding_tier":   "embedding-large",
        "vector_store":     "qdrant",
        "top_k":            8,
        "reranker":         "cross-encoder",
        "context_budget":   8000,
        "prompt_version":   "v3",
        "eval_set_version": "2026-04-eval-v2",
        "index_size":       142_000,
    })
    mlflow.log_metrics({
        "recall_at_k":     0.89,
        "mrr":             0.71,
        "faithfulness":    0.93,
        "answer_relev":    0.86,
        "p95_latency_ms":  840,
        "cost_per_query":  0.014,
    })
```

### LLM-Specialized Tools

For teams whose primary workload is LLM experimentation, dedicated tools may serve better than (or alongside) a classical tracker:

- **W&B Weave** — LLM app tracing + evaluation, integrated with W&B experiments and Models. <https://weave-docs.wandb.ai>
- **Langfuse** — open-source LLM observability + prompt management + datasets/evals. <https://langfuse.com/docs>
- **Opik** (Comet) — open-source LLM evals + tracing. <https://www.comet.com/site/products/opik/>
- **LangSmith** — eval datasets + LLM-as-judge, deepest LangChain/LangGraph integration. <https://docs.smith.langchain.com>
- **Braintrust** — eval-and-experiment platform with strong UI. <https://www.braintrust.dev>
- **Promptfoo** — declarative YAML eval configs in CI. <https://www.promptfoo.dev>
- **DeepEval** — pytest-style LLM evals. <https://docs.confident-ai.com>
- **Ragas** — RAG-specific metrics. <https://docs.ragas.io>

A reasonable hybrid: **MLflow or W&B for fine-tune training runs and the model registry, plus Langfuse or LangSmith for prompt and RAG experimentation and production tracing.**

### Capability-Tier Discipline

When tracking LLM experiments, log the **capability tier** (`flagship`, `mid`, `cheap-fast`, `embedding-large`, `embedding-small`) as a first-class parameter, plus the resolved model ID at the time of the run. Comparing across tiers is meaningful; comparing across model IDs that have been silently rotated by a provider is not. This mirrors the "no hardcoded model IDs" discipline applied throughout the LLM specialist sheets.


## Part 4: Reproducibility Checklist

```python
import torch, numpy as np, random, subprocess, mlflow

def ensure_reproducibility(config: dict):
    seed = config.get('random_seed', 42)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed);    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    env = {
        "pytorch_version": torch.__version__,
        "cuda_version":    torch.version.cuda,
        "cudnn_version":   torch.backends.cudnn.version(),
        "numpy_version":   np.__version__,
    }
    mlflow.log_params(env)

    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'],
                                         stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output(['git','rev-parse','--abbrev-ref','HEAD'],
                                         stderr=subprocess.DEVNULL).decode().strip()
        mlflow.log_params({"git_commit": commit, "git_branch": branch})
    except Exception:
        pass

    if "data_hash" in config:
        mlflow.log_param("data_hash", config["data_hash"])
    mlflow.log_params(config)
```

Reproducibility checklist:

- [x] Random seeds set (torch, numpy, random)
- [x] CuDNN deterministic mode enabled (or `torch.use_deterministic_algorithms(True)` for full strict mode)
- [x] Environment versions logged (Python, framework, CUDA / cuDNN)
- [x] Code version logged (git commit + branch + dirty-tree flag if not clean)
- [x] Data version logged (DVC pointer or content hash)
- [x] Hyperparameters logged (full config dict)
- [x] Model architecture logged (or class name + init args)
- [x] For LLM work: prompt hash, eval-set version, model tier, resolved model ID


## Part 5: REFACTOR — Pressure Tests

### Pressure Test 1: Lost Experiment Recovery

```python
def test_lost_experiment_recovery():
    runs = mlflow.search_runs(
        experiment_names=["cifar10-classification"],
        filter_string="metrics.val_acc > 96.0",
        order_by=["metrics.val_acc DESC"], max_results=1,
    )
    assert len(runs) > 0
    best = runs.iloc[0]
    for p in ['params.learning_rate','params.batch_size','params.git_commit',
              'params.data_hash','params.random_seed']:
        assert p in best and not pd.isna(best[p]), f"Missing {p}"
```

### Pressure Test 2: Production Model Identification

```python
def test_production_model_id():
    client = mlflow.tracking.MlflowClient()
    # Modern: use alias instead of stage
    mv = client.get_model_version_by_alias("cifar10-resnet18", "champion")
    assert mv is not None
    assert "val_acc" in mv.tags
```

### Pressure Test 3: Multi-User Comparison

```python
def test_multi_user_comparison():
    runs = mlflow.search_runs(experiment_names=["cifar10-classification"],
                              order_by=["metrics.val_acc DESC"])
    users = runs['tags.mlflow.user'].dropna().unique()
    assert len(users) >= 2
```

### Pressure Test 4: Data-Change Detection

```python
def test_data_change_detection():
    runs = mlflow.search_runs(experiment_names=["cifar10-classification"],
                              order_by=["start_time DESC"], max_results=10)
    assert 'params.data_hash' in runs.columns
    hashes = runs['params.data_hash'].dropna().unique()
    # If experiments span data versions, we can correlate accuracy changes:
    if len(hashes) > 1:
        for h in hashes:
            sl = runs[runs['params.data_hash'] == h]
            print(f"data {h[:8]}: mean val_acc = {sl['metrics.val_acc'].mean():.2f}")
```

### Pressure Test 5: Rollback (Alias-Based)

```python
def test_alias_rollback():
    client = mlflow.tracking.MlflowClient()
    versions = sorted(client.search_model_versions("name='cifar10-resnet18'"),
                      key=lambda v: int(v.version), reverse=True)
    assert len(versions) >= 2
    target = versions[1].version  # previous version
    client.set_registered_model_alias("cifar10-resnet18", "champion", target)
    assert client.get_model_version_by_alias("cifar10-resnet18", "champion").version == target
```

### Pressure Test 6: Prediction Audit

```python
def test_prediction_audit():
    runs = mlflow.search_runs(experiment_names=["cifar10-classification"],
                              filter_string="tags.mlflow.runName LIKE 'inference-%'",
                              order_by=["start_time DESC"], max_results=1)
    assert len(runs) > 0
    inf = runs.iloc[0]
    for f in ['params.training_run_id','params.input_data_hash',
              'params.predictions_hash','params.inference_timestamp']:
        assert f in inf and not pd.isna(inf[f]), f"Missing {f}"
```

### Pressure Test 7: Hyperparameter Search Analysis

```python
def test_search_analysis():
    runs = mlflow.search_runs(experiment_names=["cifar10-classification"],
                              order_by=["metrics.val_acc DESC"])
    assert len(runs) >= 10
    best = runs.iloc[0]
    hp_cols = [c for c in runs.columns if c.startswith('params.')]
    assert hp_cols, "No hyperparameters logged"
```

### Pressure Test 8: Long-Term Reproducibility

```python
def test_long_term_repro():
    import time
    cutoff = int((time.time() - 30*24*3600) * 1000)
    runs = mlflow.search_runs(experiment_names=["cifar10-classification"],
                              filter_string=f"attributes.start_time < {cutoff}",
                              order_by=["start_time ASC"], max_results=1)
    if not len(runs):
        return
    old = runs.iloc[0]
    for f in ['params.learning_rate','params.batch_size','params.random_seed',
              'params.git_commit','params.data_hash']:
        assert f in old and not pd.isna(old[f]), f"Missing {f}"
```

### Pressure Test 9: Artifact Cleanup

```python
def test_artifact_cleanup():
    runs = mlflow.search_runs(experiment_names=["cifar10-classification"],
                              order_by=["metrics.val_acc DESC"])
    keep = set(runs.head(5)['run_id'])
    delete = set(runs['run_id']) - keep
    # In production: client.delete_run(run_id) for each in delete (after backup)
    assert len(keep) == 5
```

### Pressure Test 10: Team Onboarding

```python
def test_onboarding():
    experiments = mlflow.search_experiments()
    total = sum(len(mlflow.search_runs(experiment_ids=[e.experiment_id]))
                for e in experiments)
    assert total > 0  # New hires can browse this history
```

### Pressure Test 11 (LLM): Reproduce a Prompt-Eval Comparison

```python
def test_reproduce_prompt_eval():
    runs = mlflow.search_runs(experiment_names=["llm-prompt-iter"],
                              order_by=["metrics.judge_overall DESC"])
    best = runs.iloc[0]
    for p in ['params.prompt_hash','params.eval_set_version',
              'params.model_tier','params.model_id_resolved']:
        assert p in best and not pd.isna(best[p]), f"Missing {p}"
```


## Part 6: Integration Patterns (Cheat Sheet)

```python
# MLflow minimal
with mlflow.start_run():
    mlflow.log_params(config)
    for epoch in range(epochs):
        mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)
    mlflow.pytorch.log_model(model, name="model")

# W&B minimal
wandb.init(project="my-project", config=config)
for epoch in range(epochs):
    wandb.log({"train_loss": tl, "val_loss": vl})
wandb.finish()

# Hugging Face Hub minimal
model.push_to_hub("my-org/my-model", private=True)

# DVC minimal
# (CLI) dvc add data/ ; git commit ... ; dvc push
mlflow.log_param("data_hash", compute_data_hash(Path("./data")))

# Hyperparameter sweep (W&B)
sweep_id = wandb.sweep(sweep_config, project="my-project")
wandb.agent(sweep_id, function=train, count=20)
```


## Skill Mastery Checklist

You have mastered experiment tracking when you can:

- [ ] Recognize when tracking is needed (cannot reproduce, lost experiments, audit failure)
- [ ] Set up MLflow tracking server with Postgres backend and S3-compatible artifact root
- [ ] Set up W&B (cloud or self-hosted) and instrument a training run end-to-end
- [ ] Use the **alias-based** MLflow Model Registry workflow (the modern API; stages are deprecated)
- [ ] Push fine-tuned LLM checkpoints to Hugging Face Hub with model cards and eval results
- [ ] Version datasets with DVC (or hashes if you're not ready for DVC)
- [ ] Implement lineage (data → training run → registered model → inference run)
- [ ] Ensure reproducibility (seeds, environment, code, data, prompt where relevant)
- [ ] Choose between MLflow / W&B / Comet / Neptune / ClearML / cloud-managed based on the matrix
- [ ] Track LLM experiments correctly (prompt hash + eval-set version + capability tier + resolved model ID)
- [ ] Roll back production models via alias re-pointing in <5 minutes
- [ ] Audit predictions for compliance using lineage queries

**Key insight:** Without tracking, experiments are lost. With tracking, every experiment is reproducible and queryable. The skill is building systems where reproducibility is automatic, not manual — and where the discipline extends naturally to prompts, eval sets, and fine-tunes for LLM workloads.

Tooling and APIs current as of 2026-05; revisit quarterly.
