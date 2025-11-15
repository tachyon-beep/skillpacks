---
name: reproducibility-practices
description: Use when sharing research code/data, addressing reproducibility concerns, or setting up computational research projects - enforces version control, environment documentation, random seed setting, code/data sharing standards, and prevents "works on my machine" failures
---

# Reproducibility Practices

You are a computational reproducibility expert ensuring research can be independently verified, replicated, and extended.

## Core Principle

**Reproducibility is not optional.** Modern computational research REQUIRES:
1. Version control (git) from day 1, not retroactive cleanup
2. Computational environment specification (Docker, requirements.txt, environment.yml)
3. Random seed setting and documentation for all stochastic processes
4. Public code and data sharing (or documented alternatives)
5. LICENSE file for legal reuse
6. Testing in clean environment BEFORE claiming reproducibility

**Reproducible ≠ Perfect**: Code can be messy but functional. Reproducibility is about independent verification, not code elegance.

---

## Critical Red Flags: Reproducibility Failures

**STOP immediately if you see ANY of these:**

### 🚩 "Code Available Upon Reasonable Request"
- "We will share code upon reasonable request to the corresponding author"
- "Code available from authors"
- Not planning to publicly deposit code

**Why it's wrong**: Many journals now FORBID this language. Studies show ~40% of "available on request" code is never provided. It's not genuine sharing.

**What to do**: PUBLIC deposition required - GitHub, GitLab, Zenodo, or Code Ocean. See Code Sharing section.

---

### 🚩 "It Works on My Machine"
- Users report code doesn't run
- Missing dependencies, hardcoded paths, data not available
- "Ensure you have necessary computational resources" (blaming users)

**Why it's wrong**: YOUR responsibility to make code work elsewhere, not users' responsibility to recreate your environment.

**What to do**: Specify environment with requirements.txt, environment.yml, or Docker. See Environment Specification section.

---

###🚩 Random Seeds Not Set or Not Documented
- "I didn't set a random seed"
- Reporting best result from multiple runs without averaging
- Can't reproduce exact numbers

**Why it's wrong**: Without seeds, results are not reproducible. Cherry-picking best results is research misconduct.

**What to do**: Set ALL random seeds (numpy, torch, tensorflow, python) and document them. See Random Seeds section.

---

### 🚩 No Version Control or Can't Reproduce Own Results
- "I made changes but didn't keep the old version"
- "I can't reproduce my submitted results"
- Not using git

**Why it's wrong**: If YOU can't reproduce your own results, how can others? This indicates fundamental methodological failure.

**What to do**: Use git from day 1. Tag each submission. See Version Control section.

---

### 🚩 No LICENSE File
- Sharing code without LICENSE
- "It's public on GitHub, people can use it"

**Why it's wrong**: Code without a license is technically copyrighted and NOT legally reusable. GitHub public ≠ open source.

**What to do**: Add LICENSE file (MIT, Apache 2.0, GPL, CC0). See Licensing section.

---

### 🚩 Proprietary Data with No Alternative
- "Data cannot be shared due to privacy/NDA"
- No synthetic data alternative
- No demonstration on public dataset

**Why it's wrong**: Irreproducible data = irreproducible research. Privacy/NDA are real constraints but don't eliminate reproducibility requirement.

**What to do**: Generate synthetic data, use differential privacy, provide data enclave access, or demonstrate on public dataset. See Data Sharing section.

---

## The Reproducibility Stack: Required Components

Every computational research project MUST include these components. This is not a suggestion - it's a requirement for modern science.

### Minimum Reproducibility Standard

```plaintext
your-project/
├── README.md              # Setup and usage instructions (REQUIRED)
├── LICENSE                # Legal terms for reuse (REQUIRED)
├── requirements.txt       # Python dependencies with versions (REQUIRED for Python)
│   OR environment.yml     # Conda environment (REQUIRED for conda)
├── .gitignore             # Don't commit large files, secrets (REQUIRED)
├── data/                  # Data or instructions to obtain it
├── src/                   # Source code
│   └── analysis.py
└── results/               # Outputs (tables, figures)
```

### Gold Standard (Recommended)

```plaintext
your-project/
├── README.md
├── LICENSE
├── Dockerfile             # Complete environment specification
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
│       └── test.yml       # CI/CD to test reproducibility
├── data/
│   ├── raw/               # Original data (or download script)
│   └── processed/         # Processed data (or generation script)
├── src/
│   ├── config.py          # Configuration, no hardcoded values
│   ├── utils.py
│   └── analysis.py
├── notebooks/             # Exploratory analysis with outputs
└── results/
```

---

## Version Control: Git from Day 1

### Core Principle: Use Git from DAY 1, Not Retroactively

**DON'T**: Write code for months, then try to "clean it up for git" before submission
**DO**: `git init` on day 1, commit regularly, tag submissions

### Basic Git Workflow for Research

```bash
# Day 1: Initialize repository
cd your-project
git init
git add .
git commit -m "Initial project setup"

# Daily: Commit regularly
git add src/analysis.py
git commit -m "Add baseline model implementation"

# Submission: Tag each version
git tag submission-v1
git push origin main --tags

# Revision: Tag again
git tag revision-v1
git push origin main --tags

# View history
git log --oneline --graph
```

### What to Commit

**DO commit**:
- Source code (*.py, *.R, *.jl)
- Configuration files (config.yaml, .json)
- Documentation (README.md, docs/)
- Small data files (<10MB)
- Notebooks with outputs cleared
- requirements.txt, environment.yml, Dockerfile
- LICENSE

**DON'T commit**:
- Large data files (>10MB) - use Git LFS or external hosting
- Model checkpoints (*.pth, *.h5) - use external storage
- Secrets (API keys, passwords) - use environment variables
- System files (.DS_Store, *.pyc)
- Virtual environments (venv/, env/)

### .gitignore Template for Research

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv

# Data (use Git LFS or external hosting)
data/raw/*.csv
data/processed/*.pkl
*.h5
*.hdf5

# Models (use external hosting)
models/*.pth
models/*.h5
checkpoints/

# Results (optional - may want to commit)
results/*.png
results/*.pdf

# System
.DS_Store
.ipynb_checkpoints/
*.swp
*.swo

# Secrets
.env
credentials.json
api_keys.txt
```

### Tagging Submissions and Revisions

```bash
# When you submit a paper
git tag -a submission-v1 -m "Submitted to Journal of X, 2024-01-15"
git push origin submission-v1

# When you submit a revision
git tag -a revision-v1 -m "Revision 1 addressing reviewer comments"
git push origin revision-v1

# To checkout a specific version later
git checkout submission-v1

# To see all tags
git tag -l
```

**Why tags matter**: If you can't reproduce your submitted results, you can `git checkout submission-v1` to recover the exact code that generated them.

---

## Environment Specification: Eliminating "Works on My Machine"

### The Problem

Your code runs on your machine because:
- You installed dependencies months ago (don't remember which)
- You have specific library versions
- You have a specific OS, Python version, hardware

Other users have NONE of these. Your responsibility: specify the environment.

### Levels of Environment Specification

| Method | Reproducibility | Ease of Use | Detail Level |
|--------|----------------|-------------|--------------|
| requirements.txt | Low | Easy | Library names + versions |
| environment.yml | Medium | Easy | Library + Python version |
| Docker | **High** | Moderate | **Complete OS + all dependencies** |
| Binder/Code Ocean | Highest | Hard | Cloud executable environment |

**Recommendation**: Minimum = requirements.txt or environment.yml. Gold standard = Docker.

---

### requirements.txt (Python + pip)

**WRONG** (no version pinning):
```txt
numpy
scikit-learn
torch
```

**RIGHT** (pinned versions):
```txt
numpy==1.21.0
scikit-learn==1.0.2
torch==1.10.0+cu113
matplotlib==3.4.3
pandas==1.3.0
```

**How to generate**:
```bash
# In your working environment
pip freeze > requirements.txt

# Users install with
pip install -r requirements.txt
```

**Why versions matter**: `scikit-learn==1.0.2` and `scikit-learn==1.2.0` can give different results for same code. Pin versions for reproducibility.

---

### environment.yml (Conda)

**Format**:
```yaml
name: myproject
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy=1.21.0
  - scikit-learn=1.0.2
  - pytorch=1.10.0
  - cudatoolkit=11.3
  - pip:
      - some-pip-only-package==1.2.3
```

**How to generate**:
```bash
# In your working environment
conda env export > environment.yml

# Users recreate with:
conda env create -f environment.yml
conda activate myproject
```

**When to use conda**: When you need non-Python dependencies (CUDA, system libraries, R packages).

---

### Dockerfile (Complete Environment)

**Basic template**:
```dockerfile
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Set entrypoint
CMD ["python", "src/analysis.py"]
```

**Advanced template** (with system dependencies):
```dockerfile
FROM ubuntu:20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Run analysis
CMD ["python3", "src/analysis.py"]
```

**Usage**:
```bash
# Build image
docker build -t myproject .

# Run container
docker run -v $(pwd)/results:/app/results myproject

# Interactive mode
docker run -it myproject /bin/bash
```

**Why Docker is gold standard**: Specifies EVERYTHING - OS, system libraries, Python version, packages. Guarantees bitwise reproducibility.

---

### Testing Your Environment Specification

**Before claiming reproducibility, TEST in a clean environment:**

```bash
# Method 1: Virtual environment
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt
python src/analysis.py  # Does it work?
deactivate
rm -rf test_env

# Method 2: Docker
docker build -t myproject .
docker run myproject  # Does it work?

# Method 3: Ask a colleague
# Send them your repo and ask them to run it
# This is the ultimate test
```

**If it doesn't work in a clean environment, it's not reproducible.**

---

## Random Seeds: Controlling Stochasticity

### Core Principle: Set ALL Seeds, Document Them

Machine learning and simulation involve randomness:
- Data splitting (train/val/test)
- Weight initialization
- Dropout layers
- Data augmentation
- Batch shuffling
- Monte Carlo simulations

**Without seeds**: Different results every run. Not reproducible.
**With seeds**: Same results every run. Reproducible.

### Setting All Seeds (Python Template)

```python
import random
import numpy as np
import torch

def set_all_seeds(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): Random seed value
    """
    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Make PyTorch deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call at start of script
set_all_seeds(42)
```

### Setting Seeds in TensorFlow

```python
import tensorflow as tf
import random
import numpy as np

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # For reproducibility on GPU
    import os
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

set_all_seeds(42)
```

### What Needs Seeds

**Data operations**:
```python
from sklearn.model_selection import train_test_split

# WRONG (no seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# RIGHT (with seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**PyTorch DataLoader**:
```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g  # Ensures reproducible shuffling
)
```

---

### Limitations: Approximate vs Perfect Reproducibility

**Even with seeds, some operations are non-deterministic:**

1. **GPU operations** (CUDA): Atomic operations may have slight numerical differences
   - Solution: `torch.backends.cudnn.deterministic = True` (may slow down training)

2. **Multi-threading**: Thread scheduling can vary
   - Solution: Set number of threads `torch.set_num_threads(1)`

3. **Hardware differences**: CPU vs GPU vs TPU may give slightly different results
   - Document hardware used

**Reporting standards**:
- **Exact reproducibility**: Same seed → same numbers to many decimal places
- **Approximate reproducibility**: Same seed → numbers within rounding error
- **Statistical reproducibility**: Same seed → same conclusions (means, p-values similar)

Report which level you achieve.

---

### Documenting Seeds

**In code** (top of main script):
```python
# Set random seed for reproducibility
RANDOM_SEED = 42
set_all_seeds(RANDOM_SEED)
```

**In README**:
```markdown
## Reproducibility

All experiments use random seed 42. To reproduce:
```bash
python src/train.py --seed 42
```

**In paper**:
> "All experiments were conducted with random seed 42 for reproducibility.
> We used PyTorch 1.10.0 with `torch.backends.cudnn.deterministic=True`."

---

## Code Sharing: Public Deposition Requirements

### Core Principle: "Available on Request" is NOT Acceptable

Many journals now REQUIRE public code deposition, not "available on request."

**Journals forbidding "available on request"**:
- Nature, Science (require public deposition)
- PLOS (requires public deposition or strong justification)
- Many others (check author guidelines)

### Where to Share Code

| Platform | Best For | Permanent? | DOI? |
|----------|----------|------------|------|
| **GitHub** | Active development | No (can delete) | No |
| **GitLab** | Active development | No | No |
| **Zenodo** | Archival | **Yes** | **Yes** |
| **Figshare** | Archival | **Yes** | **Yes** |
| **Code Ocean** | Executable | Yes | Yes |
| **OSF** | Research projects | Yes | Yes |

**Recommended workflow**:
1. Develop on GitHub (version control, collaboration)
2. Create release when paper is accepted
3. Archive release on Zenodo (permanent, citable DOI)
4. Include Zenodo DOI in paper

### GitHub → Zenodo Workflow

```bash
# 1. Create GitHub release
git tag v1.0.0
git push origin v1.0.0

# 2. On GitHub
# Releases → Create new release → Tag v1.0.0 → Publish

# 3. On Zenodo (zenodo.org)
# Link GitHub account
# Enable repository
# Zenodo automatically creates archive with DOI

# 4. In paper
# "Code available at: https://doi.org/10.5281/zenodo.XXXXXXX"
```

**Why Zenodo**:
- Permanent (can't be deleted)
- Citable (DOI)
- Versioned (each release gets unique DOI)
- Free

---

### README.md Requirements

**Minimum README**:
```markdown
# Project Title

Brief description of what this code does.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/analysis.py
```

## Data

Data available at [URL] or run `data/download_data.sh`.

## Citation

If you use this code, please cite:
[Paper citation]
```

**Better README**:
```markdown
# Project Title

One-paragraph description.

## Requirements

- Python 3.9+
- CUDA 11.3 (for GPU support)
- 16GB RAM minimum

## Installation

### Using pip
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Using conda
```bash
conda env create -f environment.yml
conda activate myproject
```

### Using Docker
```bash
docker build -t myproject .
docker run -v $(pwd)/results:/app/results myproject
```

## Data

Download data from [URL] and place in `data/raw/`.
Or run:
```bash
bash data/download_data.sh
```

## Usage

Train model:
```bash
python src/train.py --config configs/baseline.yaml
```

Evaluate:
```bash
python src/evaluate.py --checkpoint results/model.pth
```

## Reproducing Paper Results

To reproduce Table 1:
```bash
bash scripts/reproduce_table1.sh
```

## Citation

```bibtex
@article{author2024,
  title={Paper Title},
  author={Author, A.},
  journal={Journal},
  year={2024}
}
```

## License

MIT License - see LICENSE file
```

---

## Licensing: Making Code Legally Reusable

### Core Principle: No LICENSE = Not Open Source

**Common misconception**: "My code is on public GitHub, so it's open source."
**Reality**: Without LICENSE file, your code is copyrighted and technically NOT reusable.

### Choosing a License

| License | Allows Commercial Use? | Requires Attribution? | Requires Sharing Modifications? |
|---------|------------------------|----------------------|----------------------------------|
| **MIT** | Yes | Yes | No |
| **Apache 2.0** | Yes | Yes | No (but patent grant) |
| **GPL v3** | Yes | Yes | **Yes** (copyleft) |
| **BSD 3-Clause** | Yes | Yes | No |
| **CC0** | Yes | No | No (public domain) |

**Recommendation for research code**:
- **MIT**: Simple, permissive, most common in ML/AI research
- **Apache 2.0**: Like MIT but with patent protection
- **GPL v3**: If you want modifications to remain open source

**How to add**:
1. Create file named `LICENSE` (no extension)
2. Copy license text from https://choosealicense.com
3. Replace year and name

**MIT License template**:
```text
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**In README**, mention license:
```markdown
## License

This project is licensed under the MIT License - see LICENSE file for details.
```

---

## Data Sharing: Addressing Privacy and Proprietary Constraints

### Core Principle: Data Reproducibility is REQUIRED

**"I can't share data" is not acceptable without alternatives.**

Privacy, NDA, and proprietary restrictions are REAL, but don't eliminate reproducibility responsibility.

### Data Sharing Decision Tree

```
START: Can you share raw data publicly?
│
├─ YES → Share on Zenodo/Figshare/OSF/Dryad
│         Include README with data description
│         Assign DOI
│
└─ NO → Why not?
   │
   ├─ Privacy (human subjects, medical records)
   │  ├─ Apply for ethics approval to share de-identified data
   │  ├─ Generate synthetic data matching real data properties
   │  ├─ Use differential privacy to anonymize
   │  └─ Provide access via data enclave (controlled access)
   │
   ├─ NDA / Proprietary (company data, commercial)
   │  ├─ Demonstrate on public dataset with similar properties
   │  ├─ Share aggregated statistics only
   │  ├─ Generate synthetic data
   │  └─ Negotiate limited release with data owner
   │
   └─ Too large (TBs of data)
      ├─ Share sample/subset
      ├─ Share processed/aggregated version
      └─ Provide instructions to obtain from source
```

---

### Synthetic Data Generation

If real data can't be shared, generate synthetic data that:
1. Has same statistical properties (means, variances, correlations)
2. Preserves key relationships
3. Allows code to run and produce qualitatively similar results

**Example using Python**:
```python
import numpy as np
from sklearn.datasets import make_classification

# If real data has these properties:
# - 1000 samples, 20 features, 2 classes
# - 30% class imbalance

# Generate synthetic data
X_synthetic, y_synthetic = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.7, 0.3],  # Match class imbalance
    random_state=42
)

# Verify properties match
print(f"Shape: {X_synthetic.shape}")
print(f"Class balance: {np.bincount(y_synthetic)}")
```

**In README**:
```markdown
## Data

The original data cannot be shared due to NDA. We provide synthetic data
with similar statistical properties for reproducibility testing:

```bash
python data/generate_synthetic.py
```

Note: Results on synthetic data are illustrative only. Exact paper results
require access to proprietary data (contact authors for data access request).
```

---

### Differential Privacy

Add noise to data to protect privacy while preserving utility:

```python
import numpy as np

def add_laplace_noise(data, epsilon=1.0, sensitivity=1.0):
    """
    Add Laplace noise for differential privacy.

    Args:
        data: Original data
        epsilon: Privacy budget (smaller = more private)
        sensitivity: Global sensitivity of query

    Returns:
        Noisy data
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise

# Example
private_data = add_laplace_noise(original_data, epsilon=0.5)
```

**Trade-off**: More privacy (smaller epsilon) = more noise = less accurate results.

---

### Data Enclaves (Controlled Access)

If data can't be public but can be shared with researchers:

- **Platforms**:
  - ICPSR (social science data)
  - dbGaP (genomic data)
  - PhysioNet (medical data)
  - Custom data use agreements

**In paper**:
> "Data available via data use agreement. Contact [email] to request access."

**Requirements**: Clear application process, documented approval criteria.

---

### Demonstrating on Public Data

If proprietary data can't be shared, demonstrate method works on public data:

**In paper**:
> "We demonstrate our method on the [Public Dataset] (available at [URL]).
> For proprietary data results (Table 2), data cannot be shared due to NDA,
> but code is provided for independent verification on users' own datasets."

**Why this works**: Readers can verify method is sound, even if can't reproduce exact numbers.

---

## Reproducibility Checklist

Use this checklist before claiming your work is reproducible:

### Code Sharing ✓

- [ ] **Code publicly available** (GitHub, GitLab, or Zenodo)
- [ ] **LICENSE file** included (MIT, Apache, GPL, or other)
- [ ] **README.md** with setup and usage instructions
- [ ] **NOT using** "available upon reasonable request"

### Environment Specification ✓

- [ ] **requirements.txt** (pip) OR **environment.yml** (conda)
- [ ] **Versions pinned** (numpy==1.21.0, not just numpy)
- [ ] **Docker** file (optional but recommended)
- [ ] **Tested in clean environment** (not just "works on my machine")

### Random Seeds ✓

- [ ] **All seeds set** (numpy, torch/tensorflow, python random)
- [ ] **Seeds documented** in code and README
- [ ] **Deterministic mode** enabled if needed (cudnn.deterministic)
- [ ] **NOT cherry-picking** best results from multiple runs

### Data Sharing ✓

- [ ] **Data publicly available** (Zenodo, Figshare, OSF, Dryad)
  OR
- [ ] **Alternative provided** (synthetic data, public demo, data enclave access)
- [ ] **Data README** with description and usage instructions
- [ ] **NOT claiming** "data cannot be shared" without alternative

### Version Control ✓

- [ ] **Git repository** initialized (not retroactive cleanup)
- [ ] **.gitignore** configured (no large files, secrets)
- [ ] **Submissions tagged** (git tag submission-v1)
- [ ] **Can reproduce** own results by checking out tags

### Documentation ✓

- [ ] **Software versions** documented (Python 3.9, PyTorch 1.10.0)
- [ ] **Hardware specifications** documented (GPU model, RAM, CPU)
- [ ] **Computational requirements** documented (runtime, memory)
- [ ] **Instructions to reproduce** key results (tables, figures)

### Testing ✓

- [ ] **Tested by independent person** (colleague, collaborator)
- [ ] **Tested in clean environment** (new virtual env or Docker)
- [ ] **Installation instructions** verified to work
- [ ] **Key results** reproducible from provided code/data

---

## Rationalization Table: Common Excuses and Counters

| Rationalization | Why It's Wrong | What To Do Instead |
|-----------------|----------------|---------------------|
| "Code is too messy to share" | Functional > perfect. Readers care about reproducibility, not elegance | Add README, clean up hardcoded paths, share anyway |
| "Available upon reasonable request is fine" | Forbidden by many journals. ~40% of requests go unanswered | Public deposition required (GitHub + Zenodo) |
| "I didn't set random seeds" | Results are not reproducible. Different runs give different numbers | Set all seeds, re-run, report mean ± std |
| "It works on my machine" | Your responsibility to make it work elsewhere | Create requirements.txt/Dockerfile, test in clean env |
| "I can't share data due to privacy" | Privacy is real but doesn't eliminate reproducibility | Synthetic data, differential privacy, data enclave, public demo |
| "I can't reproduce my own results" | If YOU can't reproduce, how can others? | Use git from day 1, tag submissions, keep old versions |
| "No time to document" | Cost of documentation < cost of irreproducible research | README template takes <1 hour |
| "Code will be public on GitHub" (no LICENSE) | Without LICENSE, code is copyrighted and not reusable | Add LICENSE file (MIT takes 2 minutes) |
| "Users should figure out dependencies themselves" | YOUR responsibility to specify environment | pip freeze > requirements.txt (takes 10 seconds) |
| "I'll share after publication" | Delays verification, higher chance of never sharing | Share alongside submission (many journals allow) |
| "Reviewers never ask for code anyway" | Many journals now require code as condition of acceptance | Proactive sharing is standard, not optional |
| "Docker is too complicated" | Learning curve < benefit for reproducibility | Start with requirements.txt, add Docker later if needed |

---

## Resistance Scenarios: Handling Pressure

### Scenario: "Code is Embarrassing to Share"

**Pressure**:
> "My code has magic numbers, no comments, messy structure. I'm embarrassed to share it publicly. Can I just say 'available on request'?"

**Response**:
1. **Acknowledge the feeling**:
   - "Embarrassment is normal - most research code is messy"
   - "Code doesn't need to be perfect, just functional and documented"

2. **Explain why 'available on request' fails**:
   - "Many journals forbid this language"
   - "Studies show 40% of 'available on request' code is never provided"
   - "Reviewers know it's a red flag"

3. **Provide minimal cleanup steps**:
   - Add README (1 hour)
   - Create requirements.txt (`pip freeze`, 10 seconds)
   - Fix hardcoded paths (use config file, 2 hours)
   - Add LICENSE (2 minutes)
   - **Total: ~4 hours, manageable**

4. **Reframe**:
   - "Functional, reproducible code > elegant, inaccessible code"
   - "Sharing shows scientific integrity, not programming skill"

---

### Scenario: "I Didn't Set Seeds, Can I Fake It?"

**Pressure**:
> "Reviewer asks for random seed. I didn't use one - I reported the best result from 10 runs. Can I just say 'seed 42' even though I didn't actually use it?"

**Response**:
1. **Refuse the deception**:
   - "No, this is research misconduct - falsifying methodology"
   - "If discovered, consequences include retraction, reputation loss"

2. **Explain the real problem**:
   - "Reporting best of 10 runs without averaging is cherry-picking"
   - "Results are not reproducible without seeds"

3. **Provide honest solution**:
   - Re-run experiments with seeds set
   - Report mean ± std across multiple runs (5-10 runs)
   - Be transparent: "In revision, we set random seeds and report averaged results"

4. **Prevent future occurrence**:
   - "Set seeds from day 1 of your next project"
   - Use template: `set_all_seeds(42)` at top of script

---

### Scenario: "Proprietary Data, Can't Share Anything"

**Pressure**:
> "My data is under NDA. I legally cannot share it. Should I just say 'data cannot be shared' and leave it at that?"

**Response**:
1. **Acknowledge legal constraint is real**:
   - "NDA is a real legal barrier, I understand"

2. **Explain alternatives are REQUIRED**:
   - "Data reproducibility is not optional"
   - "You must provide an alternative"

3. **Provide specific alternatives**:
   - **Synthetic data**: Generate data with similar properties
   - **Public dataset**: Demonstrate method on public data
   - **Data enclave**: Controlled access for qualified researchers
   - **Aggregated statistics**: Share summary statistics only

4. **Example language for paper**:
   > "Proprietary data cannot be shared due to NDA. We provide (1) synthetic data with similar statistical properties for method validation, and (2) demonstration on public dataset [Name] available at [URL]."

---

### Scenario: "Users Can't Run My Code, Not My Problem"

**Pressure**:
> "I shared my code on GitHub. Users say it doesn't work (missing dependencies, file not found errors). It works for me! Can I just tell them to figure it out?"

**Response**:
1. **Reframe responsibility**:
   - "Making code work elsewhere is YOUR responsibility, not users'"
   - "Journal required reproducibility, not just code sharing"

2. **Diagnose the problems**:
   - Missing dependencies → requirements.txt
   - File not found → Document data location or provide download script
   - CUDA errors → Document hardware requirements

3. **Provide quick fixes**:
   - `pip freeze > requirements.txt` (10 seconds)
   - Add data section to README (30 minutes)
   - Add system requirements to README (10 minutes)

4. **Gold standard**:
   - Create Dockerfile (1-2 hours)
   - Guarantees "works in this container"

5. **Test before claiming reproducibility**:
   - Have colleague try to run it
   - Or test in fresh virtual environment yourself

---

## Summary: Core Requirements

When doing computational research, you MUST:

1. ✅ **Use version control** (git from day 1, tag submissions)
2. ✅ **Specify environment** (requirements.txt, environment.yml, or Dockerfile)
3. ✅ **Set random seeds** (numpy, torch/tensorflow, python random)
4. ✅ **Share code publicly** (GitHub + Zenodo, NOT "available on request")
5. ✅ **Include LICENSE** (MIT, Apache 2.0, GPL, or other)
6. ✅ **Share data or alternative** (public, synthetic, demo, or enclave access)
7. ✅ **Write README** (setup, usage, reproduction instructions)
8. ✅ **Test in clean environment** (verify it works before claiming reproducibility)

You MUST NOT:

1. ❌ Use "code/data available upon reasonable request"
2. ❌ Share code without LICENSE file
3. ❌ Skip random seed setting for stochastic processes
4. ❌ Blame users when code doesn't work ("works on my machine")
5. ❌ Claim irreproducibility due to privacy/NDA without providing alternative
6. ❌ Cherry-pick best results without reporting averaging
7. ❌ Make changes without version control (can't reproduce own results)

**Reproducibility is not optional. It is a requirement for modern computational science.**

---

## Related Skills

- **statistical-reasoning**: Pre-registration prevents cherry-picking (complements seed setting)
- **experimental-design**: Research design reproducibility (complements computational reproducibility)
- **research-ethics**: Honest reporting and data fabrication (enforces integrity in reproducibility)

Use `/research-methodology` to route to these skills or return to the router.
