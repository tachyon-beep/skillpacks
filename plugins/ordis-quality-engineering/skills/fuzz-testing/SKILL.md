---
name: fuzz-testing
description: Use when testing input validation, discovering edge cases, finding security vulnerabilities, testing parsers/APIs with random inputs, or integrating fuzzing tools (AFL, libFuzzer, Atheris) - provides fuzzing strategies, tool selection, and crash triage workflows
---

# Fuzz Testing

## Overview

**Core principle:** Fuzz testing feeds random/malformed inputs to find crashes, hangs, and security vulnerabilities that manual tests miss.

**Rule:** Fuzzing finds bugs you didn't know to test for. Use it for security-critical code (parsers, validators, APIs).

## Fuzz Testing vs Other Testing

| Test Type | Input | Goal |
|-----------|-------|------|
| **Unit Testing** | Known valid/invalid inputs | Verify expected behavior |
| **Property-Based Testing** | Generated valid inputs | Verify invariants hold |
| **Fuzz Testing** | Random/malformed inputs | Find crashes, hangs, memory issues |

**Fuzzing finds:** Buffer overflows, null pointer dereferences, infinite loops, unhandled exceptions

**Fuzzing does NOT find:** Logic bugs, performance issues

---

## When to Use Fuzz Testing

**Good candidates:**
- Input parsers (JSON, XML, CSV, binary formats)
- Network protocol handlers
- Image/video codecs
- Cryptographic functions
- User input validators (file uploads, form data)
- APIs accepting untrusted data

**Poor candidates:**
- Business logic (use property-based testing)
- UI interactions (use E2E tests)
- Database queries (use integration tests)

---

## Tool Selection

| Tool | Language | Type | When to Use |
|------|----------|------|-------------|
| **Atheris** | Python | Coverage-guided | Python applications, libraries |
| **AFL (American Fuzzy Lop)** | C/C++ | Coverage-guided | Native code, high performance |
| **libFuzzer** | C/C++/Rust | Coverage-guided | Integrated with LLVM/Clang |
| **Jazzer** | Java/JVM | Coverage-guided | Java applications |
| **go-fuzz** | Go | Coverage-guided | Go applications |

**Coverage-guided:** Tracks which code paths are executed, generates inputs to explore new paths

---

## Basic Fuzzing Example (Python + Atheris)

### Installation

```bash
pip install atheris
```

---

### Simple Fuzz Test

```python
import atheris
import sys

def parse_email(data):
    """Function to fuzz - finds bugs we didn't know about."""
    if "@" not in data:
        raise ValueError("Invalid email")

    local, domain = data.split("@", 1)

    if "." not in domain:
        raise ValueError("Invalid domain")

    # BUG: Crashes on multiple @ symbols!
    # "user@@example.com" → crashes with ValueError

    return (local, domain)

@atheris.instrument_func
def TestOneInput(data):
    """Fuzz harness - called repeatedly with random inputs."""
    try:
        parse_email(data.decode('utf-8', errors='ignore'))
    except (ValueError, UnicodeDecodeError):
        # Expected exceptions - not crashes
        pass
    # Any other exception = crash found!

atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()
```

**Run:**
```bash
python fuzz_email.py
```

**Output:**
```
INFO: Seed: 1234567890
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
#1: NEW coverage: 10 exec/s: 1000
#100: NEW coverage: 15 exec/s: 5000
CRASH: input was 'user@@example.com'
```

---

## Advanced Fuzzing Patterns

### Structured Fuzzing (JSON)

**Problem:** Random bytes rarely form valid JSON

```python
import atheris
import json

@atheris.instrument_func
def TestOneInput(data):
    try:
        # Parse as JSON
        obj = json.loads(data.decode('utf-8', errors='ignore'))

        # Fuzz your JSON handler
        process_user_data(obj)
    except (json.JSONDecodeError, ValueError, KeyError):
        pass  # Expected for invalid JSON

def process_user_data(data):
    """Crashes on: {"name": "", "age": -1}"""
    if len(data["name"]) == 0:
        raise ValueError("Name cannot be empty")
    if data["age"] < 0:
        raise ValueError("Age cannot be negative")
```

---

### Fuzzing with Corpus (Seed Inputs)

**Corpus:** Collection of valid inputs to start from

```python
import atheris
import sys
import os

# Seed corpus: Valid examples
CORPUS_DIR = "./corpus"
os.makedirs(CORPUS_DIR, exist_ok=True)

# Create seed files
with open(f"{CORPUS_DIR}/valid1.txt", "wb") as f:
    f.write(b"user@example.com")
with open(f"{CORPUS_DIR}/valid2.txt", "wb") as f:
    f.write(b"alice+tag@subdomain.example.org")

@atheris.instrument_func
def TestOneInput(data):
    try:
        parse_email(data.decode('utf-8'))
    except ValueError:
        pass

atheris.Setup(sys.argv, TestOneInput, corpus_dir=CORPUS_DIR)
atheris.Fuzz()
```

**Benefits:** Faster convergence to interesting inputs

---

## Crash Triage Workflow

### 1. Reproduce Crash

```bash
# Atheris outputs crash input
CRASH: input was b'user@@example.com'

# Save to file
echo "user@@example.com" > crash.txt
```

---

### 2. Minimize Input

**Find smallest input that triggers crash:**

```python
# Original: "user@@example.com" (19 bytes)
# Minimized: "@@" (2 bytes)

# Atheris does this automatically
python fuzz_email.py crash.txt
```

---

### 3. Root Cause Analysis

```python
def parse_email(data):
    # Crash: data = "@@"
    local, domain = data.split("@", 1)
    # local = "", domain = "@"

    if "." not in domain:
        # domain = "@" → no "." → raises ValueError
        raise ValueError("Invalid domain")

    # FIX: Validate before splitting
    if data.count("@") != 1:
        raise ValueError("Email must have exactly one @")
```

---

### 4. Write Regression Test

```python
def test_email_multiple_at_symbols():
    """Regression test for fuzz-found bug."""
    with pytest.raises(ValueError, match="exactly one @"):
        parse_email("user@@example.com")
```

---

## Integration with CI/CD

### Continuous Fuzzing (GitHub Actions)

```yaml
# .github/workflows/fuzz.yml
name: Fuzz Testing

on:
  schedule:
    - cron: '0 2 * * *'  # Nightly at 2 AM
  workflow_dispatch:

jobs:
  fuzz:
    runs-on: ubuntu-latest
    timeout-minutes: 60  # Run for 1 hour
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install atheris

      - name: Run fuzzing
        run: |
          timeout 3600 python fuzz_email.py || true

      - name: Upload crashes
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: fuzz-crashes
          path: crash-*
```

**Why nightly:** Fuzzing is CPU-intensive, not suitable for every PR

---

## AFL (C/C++) Example

### Installation

```bash
# Ubuntu/Debian
sudo apt-get install afl++

# macOS
brew install afl++
```

---

### Fuzz Target

```c
// fuzz_target.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void parse_command(const char *input) {
    char buffer[64];

    // BUG: Buffer overflow if input > 64 bytes!
    strcpy(buffer, input);

    if (strcmp(buffer, "exit") == 0) {
        exit(0);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;

    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;

    char buffer[1024];
    size_t len = fread(buffer, 1, sizeof(buffer), f);
    fclose(f);

    buffer[len] = '\0';
    parse_command(buffer);

    return 0;
}
```

---

### Compile and Run

```bash
# Compile with AFL instrumentation
afl-gcc fuzz_target.c -o fuzz_target

# Create corpus directory
mkdir -p corpus
echo "exit" > corpus/input1.txt

# Run fuzzer
afl-fuzz -i corpus -o findings -- ./fuzz_target @@
```

**Output:**
```
american fuzzy lop 4.00a
  path : findings/queue
  crashes : 1
  hangs : 0
  execs done : 1000000
```

**Crashes found in:** `findings/crashes/`

---

## Anti-Patterns Catalog

### ❌ Fuzzing Without Sanitizers

**Symptom:** Memory bugs don't crash, just corrupt silently

**Fix:** Compile with AddressSanitizer (ASan)

```bash
# C/C++: Compile with ASan
afl-gcc -fsanitize=address fuzz_target.c -o fuzz_target

# Python: Use PyASan (if available)
```

**What ASan catches:** Buffer overflows, use-after-free, memory leaks

---

### ❌ Ignoring Hangs

**Symptom:** Fuzzer reports hangs, not investigated

**What hangs mean:** Infinite loops, algorithmic complexity attacks

**Fix:** Investigate and add timeout checks

```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

@atheris.instrument_func
def TestOneInput(data):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(1)  # 1-second timeout

    try:
        parse_data(data.decode('utf-8'))
    except (ValueError, TimeoutError):
        pass
    finally:
        signal.alarm(0)
```

---

### ❌ No Regression Tests

**Symptom:** Same bugs found repeatedly

**Fix:** Add regression test for every crash

```python
# After fuzzing finds crash on input "@@"
def test_regression_double_at():
    with pytest.raises(ValueError):
        parse_email("@@")
```

---

## Bottom Line

**Fuzz testing finds crashes and security vulnerabilities by feeding random/malformed inputs. Use it for security-critical code (parsers, validators, APIs).**

**Setup:**
- Use Atheris (Python), AFL (C/C++), or language-specific fuzzer
- Start with corpus (valid examples)
- Run nightly in CI (1-24 hours)

**Workflow:**
1. Fuzzer finds crash
2. Minimize crashing input
3. Root cause analysis
4. Fix bug
5. Add regression test

**If your code accepts untrusted input (files, network data, user input), you should be fuzzing it. Fuzzing finds bugs that manual testing misses.**
