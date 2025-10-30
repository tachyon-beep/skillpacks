---
name: production-debugging-techniques
description: Debug production systems with profiling, error analysis, A/B testing, and incident post-mortems.
---

# Production Debugging Techniques Skill

## When to Use This Skill

Use this skill when:
- Investigating production incidents or outages
- Debugging performance bottlenecks or latency spikes
- Analyzing model quality issues (wrong predictions, hallucinations)
- Investigating A/B test anomalies or statistical issues
- Performing post-incident analysis and root cause investigation
- Debugging edge cases or unexpected behavior
- Analyzing production logs, traces, and metrics

**When NOT to use:** Development debugging (use IDE debugger), unit test failures (use TDD), or pre-production validation.

## Core Principle

**Production debugging is forensic investigation, not random guessing.**

Without systematic debugging:
- You make random changes hoping to fix issues (doesn't address root cause)
- You guess bottlenecks without data (optimize the wrong things)
- You can't diagnose issues from logs (missing critical information)
- You panic and rollback without learning (incidents repeat)
- You skip post-mortems (no prevention, just reaction)

**Formula:** Reproduce → Profile → Diagnose → Fix → Verify → Document = Systematic resolution.

## Production Debugging Framework

```
                    ┌─────────────────────────────────┐
                    │   Incident Detection/Report     │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │  Systematic Reproduction        │
                    │  Minimal repro, not speculation  │
                    └──────────┬──────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
        ┌───────▼───────┐ ┌───▼──────┐ ┌────▼────────┐
        │ Performance   │ │  Error   │ │   Model     │
        │  Profiling    │ │ Analysis │ │  Debugging  │
        └───────┬───────┘ └───┬──────┘ └────┬────────┘
                │              │             │
                └──────────────┼─────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │    Root Cause Identification    │
                │  Not symptoms, actual cause     │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │      Fix Implementation         │
                │  Targeted, verified fix         │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │         Verification            │
                │  Prove fix works                │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │    Post-Mortem & Prevention     │
                │  Blameless, actionable          │
                └──────────────────────────────────┘
```

---

## RED Phase: Common Debugging Anti-Patterns

### Anti-Pattern 1: Random Changes (No Systematic Debugging)

**Symptom:** "Let me try changing this parameter and see if it helps."

**Why it fails:**
- No reproduction of the issue (can't verify fix)
- No understanding of root cause (might fix symptom, not cause)
- No measurement of impact (did it actually help?)
- Creates more problems (unintended side effects)

**Example:**

```python
# WRONG: Random parameter changes without investigation
def fix_slow_inference():
    # User reported slow inference, let's just try stuff
    model.batch_size = 32  # Maybe this helps?
    model.num_threads = 8  # Or this?
    model.use_cache = True  # Definitely cache!
    # Did any of this help? Who knows!
```

**Consequences:**
- Issue not actually fixed (root cause still present)
- New issues introduced (different batch size breaks memory)
- Can't explain what fixed it (no learning)
- Incident repeats (no prevention)

### Anti-Pattern 2: No Profiling (Guess Bottlenecks)

**Symptom:** "The database is probably slow, let's add caching everywhere."

**Why it fails:**
- Optimize based on intuition, not data
- Miss actual bottleneck (CPU, not DB)
- Waste time on irrelevant optimizations
- No measurable improvement

**Example:**

```python
# WRONG: Adding caching without profiling
def optimize_without_profiling():
    # Guess: Database is slow
    @cache  # Add caching everywhere
    def get_user_data(user_id):
        return db.query(user_id)

    # Actual bottleneck: JSON serialization (not DB)
    # Caching doesn't help!
```

**Consequences:**
- Latency still high (actual bottleneck not addressed)
- Increased complexity (caching layer adds bugs)
- Wasted optimization effort (wrong target)
- No improvement in metrics

### Anti-Pattern 3: Bad Logging (Can't Diagnose Issues)

**Symptom:** "An error occurred but I can't figure out what caused it."

**Why it fails:**
- Missing context (no user ID, request ID, timestamp)
- No structured logging (can't query or aggregate)
- Too much noise (logs everything, signal buried)
- No trace IDs (can't follow request across services)

**Example:**

```python
# WRONG: Useless logging
def process_request(request):
    print("Processing request")  # What request? When? By whom?

    try:
        result = model.predict(request.data)
    except Exception as e:
        print(f"Error: {e}")  # No context, can't debug

    print("Done")  # Success or failure?
```

**Consequences:**
- Can't reproduce issues (missing critical context)
- Can't trace distributed requests (no correlation)
- Can't analyze patterns (unstructured data)
- Slow investigation (manual log digging)

### Anti-Pattern 4: Panic Rollback (Don't Learn from Incidents)

**Symptom:** "There's an error! Rollback immediately! Now!"

**Why it fails:**
- No evidence collection (can't do post-mortem)
- No root cause analysis (will happen again)
- Lose opportunity to learn (panic mode)
- No distinction between minor and critical issues

**Example:**

```python
# WRONG: Immediate rollback without investigation
def handle_incident():
    if error_rate > 0.1:  # Any errors = panic
        # Rollback immediately!
        deploy_previous_version()
        # Wait, what was the error? We'll never know now...
```

**Consequences:**
- Issue repeats (root cause not fixed)
- Lost learning opportunity (no forensics)
- Unnecessary rollbacks (minor issues treated as critical)
- Team doesn't improve (no post-mortem)

### Anti-Pattern 5: No Post-Mortems

**Symptom:** "Incident resolved, let's move on to the next task."

**Why it fails:**
- No prevention (same incident repeats)
- No learning (team doesn't improve)
- No action items (nothing changes)
- Culture of blame (fear of investigation)

**Example:**

```python
# WRONG: No post-mortem process
def resolve_incident(incident):
    fix_issue(incident)
    close_ticket(incident)
    # Done! What incident? Already forgot...
    # No documentation, no prevention, no learning
```

**Consequences:**
- Incidents repeat (no prevention mechanisms)
- No improvement (same mistakes over and over)
- Low bus factor (knowledge not shared)
- Reactive culture (firefighting, not prevention)

---

## GREEN Phase: Systematic Debugging Methodology

### Part 1: Systematic Debugging Framework

**Core principle:** Reproduce → Diagnose → Fix → Verify

**Step-by-step process:**

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

@dataclass
class DebuggingSession:
    """
    Structured debugging session with systematic methodology.
    """
    incident_id: str
    reported_by: str
    reported_at: datetime
    description: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW

    # Reproduction
    reproduction_steps: List[str] = None
    minimal_repro: str = None
    reproduction_rate: float = 0.0  # 0.0 to 1.0

    # Diagnosis
    hypothesis: str = None
    evidence: Dict[str, Any] = None
    root_cause: str = None

    # Fix
    fix_description: str = None
    fix_verification: str = None

    # Prevention
    prevention_measures: List[str] = None

    def __post_init__(self):
        self.reproduction_steps = []
        self.evidence = {}
        self.prevention_measures = []


class SystematicDebugger:
    """
    Systematic debugging methodology for production issues.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sessions: Dict[str, DebuggingSession] = {}

    def start_session(
        self,
        incident_id: str,
        reported_by: str,
        description: str,
        severity: str
    ) -> DebuggingSession:
        """
        Start a new debugging session.

        Args:
            incident_id: Unique incident identifier
            reported_by: Who reported the issue
            description: What is the problem
            severity: CRITICAL, HIGH, MEDIUM, LOW

        Returns:
            DebuggingSession object
        """
        session = DebuggingSession(
            incident_id=incident_id,
            reported_by=reported_by,
            reported_at=datetime.now(),
            description=description,
            severity=severity
        )

        self.sessions[incident_id] = session
        self.logger.info(
            f"Started debugging session",
            extra={
                "incident_id": incident_id,
                "severity": severity,
                "description": description
            }
        )

        return session

    def reproduce_issue(
        self,
        session: DebuggingSession,
        reproduction_steps: List[str]
    ) -> bool:
        """
        Step 1: Reproduce the issue with minimal test case.

        Goal: Create minimal, deterministic reproduction.

        Args:
            session: Debugging session
            reproduction_steps: Steps to reproduce

        Returns:
            True if successfully reproduced
        """
        session.reproduction_steps = reproduction_steps

        # Try to reproduce
        for attempt in range(10):
            if self._attempt_reproduction(reproduction_steps):
                session.reproduction_rate += 0.1

        session.reproduction_rate = session.reproduction_rate

        reproduced = session.reproduction_rate > 0.5

        self.logger.info(
            f"Reproduction attempt",
            extra={
                "incident_id": session.incident_id,
                "reproduced": reproduced,
                "reproduction_rate": session.reproduction_rate
            }
        )

        return reproduced

    def _attempt_reproduction(self, steps: List[str]) -> bool:
        """
        Attempt to reproduce issue.
        Implementation depends on issue type.
        """
        # Override in subclass
        return False

    def collect_evidence(
        self,
        session: DebuggingSession,
        evidence_type: str,
        evidence_data: Any
    ):
        """
        Step 2: Collect evidence from multiple sources.

        Evidence types:
        - logs: Application logs
        - traces: Distributed traces
        - metrics: Performance metrics
        - profiles: CPU/memory profiles
        - requests: Failed request data
        """
        if evidence_type not in session.evidence:
            session.evidence[evidence_type] = []

        session.evidence[evidence_type].append({
            "timestamp": datetime.now(),
            "data": evidence_data
        })

        self.logger.info(
            f"Collected evidence",
            extra={
                "incident_id": session.incident_id,
                "evidence_type": evidence_type
            }
        )

    def form_hypothesis(
        self,
        session: DebuggingSession,
        hypothesis: str
    ):
        """
        Step 3: Form hypothesis based on evidence.

        Good hypothesis:
        - Specific and testable
        - Based on evidence, not intuition
        - Explains all symptoms
        """
        session.hypothesis = hypothesis

        self.logger.info(
            f"Formed hypothesis",
            extra={
                "incident_id": session.incident_id,
                "hypothesis": hypothesis
            }
        )

    def verify_hypothesis(
        self,
        session: DebuggingSession,
        verification_test: str,
        result: bool
    ) -> bool:
        """
        Step 4: Verify hypothesis with targeted test.

        Args:
            session: Debugging session
            verification_test: What test was run
            result: Did hypothesis hold?

        Returns:
            True if hypothesis verified
        """
        self.collect_evidence(
            session,
            "hypothesis_verification",
            {
                "test": verification_test,
                "result": result,
                "hypothesis": session.hypothesis
            }
        )

        return result

    def identify_root_cause(
        self,
        session: DebuggingSession,
        root_cause: str
    ):
        """
        Step 5: Identify root cause (not just symptoms).

        Root cause vs symptom:
        - Symptom: "API returns 500 errors"
        - Root cause: "Connection pool exhausted due to connection leak"
        """
        session.root_cause = root_cause

        self.logger.info(
            f"Identified root cause",
            extra={
                "incident_id": session.incident_id,
                "root_cause": root_cause
            }
        )

    def implement_fix(
        self,
        session: DebuggingSession,
        fix_description: str,
        fix_code: str = None
    ):
        """
        Step 6: Implement targeted fix.

        Good fix:
        - Addresses root cause, not symptom
        - Minimal changes (surgical fix)
        - Includes verification test
        """
        session.fix_description = fix_description

        self.logger.info(
            f"Implemented fix",
            extra={
                "incident_id": session.incident_id,
                "fix_description": fix_description
            }
        )

    def verify_fix(
        self,
        session: DebuggingSession,
        verification_method: str,
        verified: bool
    ) -> bool:
        """
        Step 7: Verify fix resolves the issue.

        Verification methods:
        - Reproduction test no longer fails
        - Metrics return to normal
        - No new errors in logs
        - A/B test shows improvement
        """
        session.fix_verification = verification_method

        self.logger.info(
            f"Verified fix",
            extra={
                "incident_id": session.incident_id,
                "verified": verified,
                "verification_method": verification_method
            }
        )

        return verified

    def add_prevention_measure(
        self,
        session: DebuggingSession,
        measure: str
    ):
        """
        Step 8: Add prevention measures.

        Prevention types:
        - Monitoring: Alert on similar patterns
        - Testing: Add regression test
        - Validation: Input validation to prevent
        - Documentation: Runbook for similar issues
        """
        session.prevention_measures.append(measure)

        self.logger.info(
            f"Added prevention measure",
            extra={
                "incident_id": session.incident_id,
                "measure": measure
            }
        )


# Example usage
debugger = SystematicDebugger()

# Start debugging session
session = debugger.start_session(
    incident_id="INC-2025-001",
    reported_by="oncall-engineer",
    description="API latency spike from 200ms to 2000ms",
    severity="HIGH"
)

# Step 1: Reproduce
reproduced = debugger.reproduce_issue(
    session,
    reproduction_steps=[
        "Send 100 concurrent requests to /api/predict",
        "Observe latency increase after 50 requests",
        "Check connection pool metrics"
    ]
)

if reproduced:
    # Step 2: Collect evidence
    debugger.collect_evidence(session, "metrics", {
        "latency_p50": 2000,
        "latency_p95": 5000,
        "connection_pool_size": 10,
        "active_connections": 10,
        "waiting_requests": 90
    })

    # Step 3: Form hypothesis
    debugger.form_hypothesis(
        session,
        "Connection pool exhausted. Pool size (10) too small for load (100 concurrent)."
    )

    # Step 4: Verify hypothesis
    verified = debugger.verify_hypothesis(
        session,
        "Increased pool size to 50, latency returned to normal",
        True
    )

    if verified:
        # Step 5: Root cause
        debugger.identify_root_cause(
            session,
            "Connection pool size not scaled with traffic increase"
        )

        # Step 6: Implement fix
        debugger.implement_fix(
            session,
            "Increase connection pool size to 50 and add auto-scaling"
        )

        # Step 7: Verify fix
        debugger.verify_fix(
            session,
            "A/B test: latency p95 < 300ms for 1 hour",
            True
        )

        # Step 8: Prevention
        debugger.add_prevention_measure(
            session,
            "Alert when connection pool utilization > 80%"
        )
        debugger.add_prevention_measure(
            session,
            "Load test before deploying to production"
        )
```

**Key principles:**

1. **Reproduce first:** Can't debug what you can't reproduce
2. **Evidence-based:** Collect data before forming hypothesis
3. **Root cause, not symptom:** Fix the actual cause
4. **Verify fix:** Prove it works before closing
5. **Prevent recurrence:** Add monitoring and tests

---

### Part 2: Performance Profiling

**When to profile:**
- Latency spikes or slow responses
- High CPU or memory usage
- Resource exhaustion (connections, threads)
- Optimization opportunities

#### CPU Profiling with py-spy

```python
import subprocess
import signal
import time
from pathlib import Path

class ProductionProfiler:
    """
    Non-intrusive profiling for production systems.
    """

    def __init__(self, output_dir: str = "./profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def profile_cpu(
        self,
        pid: int,
        duration: int = 60,
        rate: int = 100
    ) -> str:
        """
        Profile CPU usage with py-spy (no code changes needed).

        Args:
            pid: Process ID to profile
            duration: How long to profile (seconds)
            rate: Sampling rate (samples/second)

        Returns:
            Path to flamegraph SVG

        Usage:
            # Install: pip install py-spy
            # Run: sudo py-spy record -o profile.svg --pid 12345 --duration 60
        """
        output_file = self.output_dir / f"cpu_profile_{pid}_{int(time.time())}.svg"

        cmd = [
            "py-spy", "record",
            "-o", str(output_file),
            "--pid", str(pid),
            "--duration", str(duration),
            "--rate", str(rate),
            "--format", "flamegraph"
        ]

        print(f"Profiling PID {pid} for {duration} seconds...")
        subprocess.run(cmd, check=True)

        print(f"Profile saved to: {output_file}")
        return str(output_file)

    def profile_memory(
        self,
        pid: int,
        duration: int = 60
    ) -> str:
        """
        Profile memory usage with memory_profiler.

        Returns:
            Path to memory profile
        """
        output_file = self.output_dir / f"memory_profile_{pid}_{int(time.time())}.txt"

        # Use memory_profiler for line-by-line analysis
        cmd = [
            "python", "-m", "memory_profiler",
            "--backend", "psutil",
            str(pid)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=duration
        )

        output_file.write_text(result.stdout)
        print(f"Memory profile saved to: {output_file}")

        return str(output_file)


# Example: Profile production inference
profiler = ProductionProfiler()

# Get PID of running process
import os
pid = os.getpid()

# Profile for 60 seconds
flamegraph = profiler.profile_cpu(pid, duration=60)
print(f"View flamegraph: {flamegraph}")

# Analyze flamegraph:
# - Wide bars = most time spent (bottleneck)
# - Look for unexpected functions
# - Check for excessive I/O waits
```

#### PyTorch Model Profiling

```python
import torch
import torch.profiler as profiler
from typing import Dict, List
import json

class ModelProfiler:
    """
    Profile PyTorch model performance.
    """

    def profile_model(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        num_steps: int = 100
    ) -> Dict[str, any]:
        """
        Profile model inference with PyTorch profiler.

        Args:
            model: PyTorch model
            sample_input: Sample input tensor
            num_steps: Number of profiling steps

        Returns:
            Profiling results
        """
        model.eval()

        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with profiler.record_function("model_inference"):
                for _ in range(num_steps):
                    with torch.no_grad():
                        _ = model(sample_input)

        # Print report
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=10
        ))

        # Save trace
        prof.export_chrome_trace("model_trace.json")

        # Analyze results
        results = self._analyze_profile(prof)
        return results

    def _analyze_profile(self, prof) -> Dict[str, any]:
        """
        Analyze profiling results.
        """
        events = prof.key_averages()

        # Find bottlenecks
        cpu_events = sorted(
            [e for e in events if e.device_type == profiler.DeviceType.CPU],
            key=lambda e: e.self_cpu_time_total,
            reverse=True
        )

        cuda_events = sorted(
            [e for e in events if e.device_type == profiler.DeviceType.CUDA],
            key=lambda e: e.self_cuda_time_total,
            reverse=True
        )

        results = {
            "top_cpu_ops": [
                {
                    "name": e.key,
                    "cpu_time_ms": e.self_cpu_time_total / 1000,
                    "calls": e.count
                }
                for e in cpu_events[:10]
            ],
            "top_cuda_ops": [
                {
                    "name": e.key,
                    "cuda_time_ms": e.self_cuda_time_total / 1000,
                    "calls": e.count
                }
                for e in cuda_events[:10]
            ],
            "total_cpu_time_ms": sum(e.self_cpu_time_total for e in events) / 1000,
            "total_cuda_time_ms": sum(e.self_cuda_time_total for e in events) / 1000,
        }

        return results


# Example usage
import torch.nn as nn

model = nn.Transformer(
    d_model=512,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6
)

sample_input = torch.randn(10, 32, 512)  # (seq_len, batch, d_model)

profiler = ModelProfiler()
results = profiler.profile_model(model, sample_input)

print(json.dumps(results, indent=2))

# Identify bottlenecks:
# - Which operations take most time?
# - CPU vs GPU time (data transfer overhead?)
# - Memory usage patterns
```

#### Database Query Profiling

```python
import time
from contextlib import contextmanager
from typing import Dict, List
import logging

class QueryProfiler:
    """
    Profile database query performance.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.query_stats: List[Dict] = []

    @contextmanager
    def profile_query(self, query_name: str):
        """
        Context manager to profile a query.

        Usage:
            with profiler.profile_query("get_user"):
                user = db.query(User).filter_by(id=user_id).first()
        """
        start = time.perf_counter()

        try:
            yield
        finally:
            duration = (time.perf_counter() - start) * 1000  # ms

            self.query_stats.append({
                "query": query_name,
                "duration_ms": duration,
                "timestamp": time.time()
            })

            if duration > 100:  # Slow query threshold
                self.logger.warning(
                    f"Slow query detected",
                    extra={
                        "query": query_name,
                        "duration_ms": duration
                    }
                )

    def get_slow_queries(self, threshold_ms: float = 100) -> List[Dict]:
        """
        Get queries slower than threshold.
        """
        return [
            q for q in self.query_stats
            if q["duration_ms"] > threshold_ms
        ]

    def get_query_stats(self) -> Dict[str, Dict]:
        """
        Get aggregate statistics per query.
        """
        from collections import defaultdict
        import statistics

        stats_by_query = defaultdict(list)

        for q in self.query_stats:
            stats_by_query[q["query"]].append(q["duration_ms"])

        result = {}
        for query, durations in stats_by_query.items():
            result[query] = {
                "count": len(durations),
                "mean_ms": statistics.mean(durations),
                "median_ms": statistics.median(durations),
                "p95_ms": sorted(durations)[int(len(durations) * 0.95)]
                    if len(durations) > 0 else 0,
                "max_ms": max(durations)
            }

        return result


# Example usage
profiler = QueryProfiler()

# Profile queries
for user_id in range(100):
    with profiler.profile_query("get_user"):
        # user = db.query(User).filter_by(id=user_id).first()
        time.sleep(0.05)  # Simulate query

    with profiler.profile_query("get_posts"):
        # posts = db.query(Post).filter_by(user_id=user_id).all()
        time.sleep(0.15)  # Simulate slow query

# Analyze
slow_queries = profiler.get_slow_queries(threshold_ms=100)
print(f"Found {len(slow_queries)} slow queries")

stats = profiler.get_query_stats()
for query, metrics in stats.items():
    print(f"{query}: {metrics}")
```

**Key profiling insights:**

| Profile Type | Tool | What to Look For |
|--------------|------|------------------|
| CPU | py-spy | Wide bars in flamegraph (bottlenecks) |
| Memory | memory_profiler | Memory leaks, large allocations |
| Model | torch.profiler | Slow operations, CPU-GPU transfer |
| Database | Query profiler | Slow queries, N+1 queries |
| Network | distributed tracing | High latency services, cascading failures |

---

### Part 3: Error Analysis and Root Cause Investigation

**Goal:** Categorize errors, find patterns, identify root cause (not symptoms).

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import re

@dataclass
class ErrorEvent:
    """
    Structured error event.
    """
    timestamp: datetime
    error_type: str
    error_message: str
    stack_trace: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    severity: str = "ERROR"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

    # Context
    input_data: Optional[Dict] = None
    system_state: Optional[Dict] = None


class ErrorAnalyzer:
    """
    Analyze error patterns and identify root causes.
    """

    def __init__(self):
        self.errors: List[ErrorEvent] = []

    def add_error(self, error: ErrorEvent):
        """Add error to analysis."""
        self.errors.append(error)

    def categorize_errors(self) -> Dict[str, List[ErrorEvent]]:
        """
        Categorize errors by type.

        Categories:
        - Input validation errors
        - Model inference errors
        - Infrastructure errors (DB, network)
        - Third-party API errors
        - Resource exhaustion errors
        """
        categories = defaultdict(list)

        for error in self.errors:
            category = self._categorize_single_error(error)
            categories[category].append(error)

        return dict(categories)

    def _categorize_single_error(self, error: ErrorEvent) -> str:
        """
        Categorize single error based on error message and type.
        """
        msg = error.error_message.lower()

        # Input validation
        if any(keyword in msg for keyword in ["invalid", "validation", "schema"]):
            return "input_validation"

        # Model errors
        if any(keyword in msg for keyword in ["model", "inference", "prediction"]):
            return "model_inference"

        # Infrastructure
        if any(keyword in msg for keyword in ["connection", "timeout", "database"]):
            return "infrastructure"

        # Resource exhaustion
        if any(keyword in msg for keyword in ["memory", "cpu", "quota", "limit"]):
            return "resource_exhaustion"

        # Third-party
        if any(keyword in msg for keyword in ["api", "external", "service"]):
            return "third_party"

        return "unknown"

    def find_error_patterns(self) -> List[Dict]:
        """
        Find patterns in errors (temporal, user, endpoint).
        """
        patterns = []

        # Temporal clustering (errors spike at certain times?)
        temporal = self._analyze_temporal_patterns()
        if temporal:
            patterns.append({
                "type": "temporal",
                "description": f"Error spike detected",
                "details": temporal
            })

        # User clustering (errors for specific users?)
        user_errors = defaultdict(int)
        for error in self.errors:
            if error.user_id:
                user_errors[error.user_id] += 1

        # Top 5 users with most errors
        top_users = sorted(
            user_errors.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        if top_users and top_users[0][1] > 10:
            patterns.append({
                "type": "user_specific",
                "description": f"High error rate for specific users",
                "details": {"top_users": top_users}
            })

        # Endpoint clustering
        endpoint_errors = defaultdict(int)
        for error in self.errors:
            if error.endpoint:
                endpoint_errors[error.endpoint] += 1

        top_endpoints = sorted(
            endpoint_errors.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        if top_endpoints:
            patterns.append({
                "type": "endpoint_specific",
                "description": f"Errors concentrated in specific endpoints",
                "details": {"top_endpoints": top_endpoints}
            })

        return patterns

    def _analyze_temporal_patterns(self) -> Optional[Dict]:
        """
        Detect temporal error patterns (spikes, periodicity).
        """
        if len(self.errors) < 10:
            return None

        # Group by hour
        errors_by_hour = defaultdict(int)
        for error in self.errors:
            hour_key = error.timestamp.replace(minute=0, second=0, microsecond=0)
            errors_by_hour[hour_key] += 1

        # Calculate average and detect spikes
        error_counts = list(errors_by_hour.values())
        avg_errors = sum(error_counts) / len(error_counts)
        max_errors = max(error_counts)

        if max_errors > avg_errors * 3:  # 3x spike
            spike_hour = max(errors_by_hour, key=errors_by_hour.get)
            return {
                "avg_errors_per_hour": avg_errors,
                "max_errors_per_hour": max_errors,
                "spike_time": spike_hour.isoformat(),
                "spike_magnitude": max_errors / avg_errors
            }

        return None

    def identify_root_cause(
        self,
        error_category: str,
        errors: List[ErrorEvent]
    ) -> Dict:
        """
        Identify root cause for category of errors.

        Analysis steps:
        1. Find common patterns in error messages
        2. Analyze system state at error time
        3. Check for external factors (deployment, traffic spike)
        4. Identify root cause vs symptoms
        """
        analysis = {
            "category": error_category,
            "total_errors": len(errors),
            "time_range": {
                "start": min(e.timestamp for e in errors).isoformat(),
                "end": max(e.timestamp for e in errors).isoformat()
            }
        }

        # Common error messages
        error_messages = [e.error_message for e in errors]
        message_counts = Counter(error_messages)
        analysis["most_common_errors"] = message_counts.most_common(5)

        # Stack trace analysis (find common frames)
        common_frames = self._find_common_stack_frames(errors)
        analysis["common_stack_frames"] = common_frames

        # Hypothesis based on category
        if error_category == "input_validation":
            analysis["hypothesis"] = "Client sending invalid data. Check API contract."
            analysis["action_items"] = [
                "Add input validation at API layer",
                "Return clear error messages to client",
                "Add monitoring for validation failures"
            ]

        elif error_category == "model_inference":
            analysis["hypothesis"] = "Model failing on specific inputs. Check edge cases."
            analysis["action_items"] = [
                "Analyze failed inputs for patterns",
                "Add input sanitization before inference",
                "Add fallback for model failures",
                "Retrain model with failed examples"
            ]

        elif error_category == "infrastructure":
            analysis["hypothesis"] = "Infrastructure issue (DB, network). Check external dependencies."
            analysis["action_items"] = [
                "Check database connection pool size",
                "Check network connectivity to services",
                "Add retry logic with exponential backoff",
                "Add circuit breaker for failing services"
            ]

        elif error_category == "resource_exhaustion":
            analysis["hypothesis"] = "Resource limits exceeded. Scale up or optimize."
            analysis["action_items"] = [
                "Profile memory/CPU usage",
                "Increase resource limits",
                "Optimize hot paths",
                "Add auto-scaling"
            ]

        return analysis

    def _find_common_stack_frames(
        self,
        errors: List[ErrorEvent],
        min_frequency: float = 0.5
    ) -> List[str]:
        """
        Find stack frames common to most errors.
        """
        frame_counts = Counter()

        for error in errors:
            # Extract function names from stack trace
            frames = re.findall(r'File ".*", line \d+, in (\w+)', error.stack_trace)
            frame_counts.update(frames)

        # Find frames in at least 50% of errors
        threshold = len(errors) * min_frequency
        common_frames = [
            frame for frame, count in frame_counts.items()
            if count >= threshold
        ]

        return common_frames


# Example usage
analyzer = ErrorAnalyzer()

# Simulate errors
for i in range(100):
    if i % 10 == 0:  # Pattern: every 10th request fails
        analyzer.add_error(ErrorEvent(
            timestamp=datetime.now() + timedelta(seconds=i),
            error_type="ValueError",
            error_message="Invalid input shape: expected (batch, 512), got (batch, 256)",
            stack_trace='File "model.py", line 42, in predict\n  result = self.model(input_tensor)',
            user_id=f"user_{i % 5}",  # Pattern: 5 users with issues
            endpoint="/api/predict"
        ))

# Categorize errors
categories = analyzer.categorize_errors()
print(f"Error categories: {list(categories.keys())}")

# Find patterns
patterns = analyzer.find_error_patterns()
for pattern in patterns:
    print(f"\nPattern: {pattern['type']}")
    print(f"  {pattern['description']}")
    print(f"  Details: {pattern['details']}")

# Root cause analysis
for category, errors in categories.items():
    print(f"\n{'='*60}")
    print(f"Root cause analysis: {category}")
    print(f"{'='*60}")

    analysis = analyzer.identify_root_cause(category, errors)

    print(f"\nHypothesis: {analysis['hypothesis']}")
    print(f"\nAction items:")
    for item in analysis['action_items']:
        print(f"  - {item}")
```

**Root cause analysis checklist:**

- [ ] Reproduce error consistently
- [ ] Categorize error type (input, model, infrastructure, resource)
- [ ] Find error patterns (temporal, user, endpoint)
- [ ] Analyze system state at error time
- [ ] Check for external factors (deployment, traffic, dependencies)
- [ ] Distinguish root cause from symptoms
- [ ] Verify fix resolves root cause

---

### Part 4: A/B Test Debugging

**Common A/B test issues:**
- No statistical significance (insufficient sample size)
- Confounding factors (unbalanced segments)
- Simpson's paradox (aggregate vs segment differences)
- Selection bias (non-random assignment)
- Novelty effect (temporary impact)

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from scipy import stats

@dataclass
class ABTestResult:
    """
    A/B test variant result.
    """
    variant: str
    sample_size: int
    success_count: int
    metric_values: List[float]

    @property
    def success_rate(self) -> float:
        return self.success_count / self.sample_size if self.sample_size > 0 else 0.0

    @property
    def mean_metric(self) -> float:
        return np.mean(self.metric_values) if self.metric_values else 0.0


class ABTestDebugger:
    """
    Debug A/B test issues and validate statistical significance.
    """

    def validate_test_design(
        self,
        control: ABTestResult,
        treatment: ABTestResult,
        min_sample_size: int = 200
    ) -> Dict:
        """
        Validate A/B test design and detect issues.

        Returns:
            Validation results with warnings
        """
        issues = []

        # Check 1: Sufficient sample size
        if control.sample_size < min_sample_size:
            issues.append({
                "type": "insufficient_sample_size",
                "severity": "CRITICAL",
                "message": f"Control sample size ({control.sample_size}) < minimum ({min_sample_size})"
            })

        if treatment.sample_size < min_sample_size:
            issues.append({
                "type": "insufficient_sample_size",
                "severity": "CRITICAL",
                "message": f"Treatment sample size ({treatment.sample_size}) < minimum ({min_sample_size})"
            })

        # Check 2: Balanced sample sizes
        ratio = control.sample_size / treatment.sample_size
        if ratio < 0.8 or ratio > 1.25:  # More than 20% imbalance
            issues.append({
                "type": "imbalanced_samples",
                "severity": "WARNING",
                "message": f"Sample size ratio {ratio:.2f} indicates imbalanced assignment"
            })

        # Check 3: Variance analysis
        control_std = np.std(control.metric_values)
        treatment_std = np.std(treatment.metric_values)

        if control_std == 0 or treatment_std == 0:
            issues.append({
                "type": "no_variance",
                "severity": "CRITICAL",
                "message": "One variant has zero variance. Check data collection."
            })

        return {
            "valid": len([i for i in issues if i["severity"] == "CRITICAL"]) == 0,
            "issues": issues
        }

    def test_statistical_significance(
        self,
        control: ABTestResult,
        treatment: ABTestResult,
        alpha: float = 0.05
    ) -> Dict:
        """
        Test statistical significance between variants.

        Args:
            control: Control variant results
            treatment: Treatment variant results
            alpha: Significance level (default 0.05)

        Returns:
            Statistical test results
        """
        # Two-proportion z-test for success rates
        n1, n2 = control.sample_size, treatment.sample_size
        p1, p2 = control.success_rate, treatment.success_rate

        # Pooled proportion
        p_pool = (control.success_count + treatment.success_count) / (n1 + n2)

        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))

        # Z-score
        z_score = (p2 - p1) / se if se > 0 else 0

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        # Effect size (relative lift)
        relative_lift = ((p2 - p1) / p1 * 100) if p1 > 0 else 0

        # Confidence interval
        ci_margin = stats.norm.ppf(1 - alpha/2) * se
        ci_lower = (p2 - p1) - ci_margin
        ci_upper = (p2 - p1) + ci_margin

        return {
            "statistically_significant": p_value < alpha,
            "p_value": p_value,
            "z_score": z_score,
            "alpha": alpha,
            "control_rate": p1,
            "treatment_rate": p2,
            "absolute_lift": p2 - p1,
            "relative_lift_percent": relative_lift,
            "confidence_interval": (ci_lower, ci_upper),
            "interpretation": self._interpret_results(p_value, alpha, relative_lift)
        }

    def _interpret_results(
        self,
        p_value: float,
        alpha: float,
        relative_lift: float
    ) -> str:
        """
        Interpret statistical test results.
        """
        if p_value < alpha:
            direction = "better" if relative_lift > 0 else "worse"
            return f"Treatment is statistically significantly {direction} than control ({relative_lift:+.1f}% lift)"
        else:
            return f"No statistical significance detected (p={p_value:.3f} > {alpha}). Need more data or larger effect size."

    def detect_simpsons_paradox(
        self,
        control_segments: Dict[str, ABTestResult],
        treatment_segments: Dict[str, ABTestResult]
    ) -> Dict:
        """
        Detect Simpson's Paradox in segmented data.

        Simpson's Paradox: Treatment better in each segment but worse overall,
        or vice versa. Caused by confounding variables.

        Args:
            control_segments: Control results per segment (e.g., by country, device)
            treatment_segments: Treatment results per segment

        Returns:
            Detection results
        """
        # Overall results
        total_control = ABTestResult(
            variant="control_total",
            sample_size=sum(s.sample_size for s in control_segments.values()),
            success_count=sum(s.success_count for s in control_segments.values()),
            metric_values=[]
        )

        total_treatment = ABTestResult(
            variant="treatment_total",
            sample_size=sum(s.sample_size for s in treatment_segments.values()),
            success_count=sum(s.success_count for s in treatment_segments.values()),
            metric_values=[]
        )

        overall_direction = "treatment_better" if total_treatment.success_rate > total_control.success_rate else "control_better"

        # Check each segment
        segment_directions = {}
        for segment in control_segments.keys():
            ctrl = control_segments[segment]
            treat = treatment_segments[segment]

            segment_directions[segment] = "treatment_better" if treat.success_rate > ctrl.success_rate else "control_better"

        # Detect paradox: overall direction differs from all segments
        all_segments_agree = all(d == overall_direction for d in segment_directions.values())

        paradox_detected = not all_segments_agree

        return {
            "paradox_detected": paradox_detected,
            "overall_direction": overall_direction,
            "segment_directions": segment_directions,
            "explanation": self._explain_simpsons_paradox(
                paradox_detected,
                overall_direction,
                segment_directions
            )
        }

    def _explain_simpsons_paradox(
        self,
        detected: bool,
        overall: str,
        segments: Dict[str, str]
    ) -> str:
        """
        Explain Simpson's Paradox if detected.
        """
        if not detected:
            return "No Simpson's Paradox detected. Segment and overall results agree."

        return f"Simpson's Paradox detected! Overall: {overall}, but segments show: {segments}. This indicates a confounding variable. Review segment sizes and assignment."

    def calculate_required_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.80
    ) -> int:
        """
        Calculate required sample size per variant.

        Args:
            baseline_rate: Current conversion rate (e.g., 0.10 for 10%)
            minimum_detectable_effect: Minimum relative change to detect (e.g., 0.10 for 10% improvement)
            alpha: Significance level (default 0.05)
            power: Statistical power (default 0.80)

        Returns:
            Required sample size per variant
        """
        treatment_rate = baseline_rate * (1 + minimum_detectable_effect)

        # Effect size (Cohen's h)
        effect_size = 2 * (np.arcsin(np.sqrt(treatment_rate)) - np.arcsin(np.sqrt(baseline_rate)))

        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)

        # Sample size calculation
        n = ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n))


# Example: Debug A/B test
debugger = ABTestDebugger()

# Simulate test results
control = ABTestResult(
    variant="control",
    sample_size=500,
    success_count=50,  # 10% conversion
    metric_values=np.random.normal(100, 20, 500).tolist()
)

treatment = ABTestResult(
    variant="treatment",
    sample_size=520,
    success_count=62,  # 11.9% conversion
    metric_values=np.random.normal(105, 20, 520).tolist()
)

# Validate design
validation = debugger.validate_test_design(control, treatment)
print(f"Test valid: {validation['valid']}")
if validation['issues']:
    for issue in validation['issues']:
        print(f"  [{issue['severity']}] {issue['message']}")

# Test significance
results = debugger.test_statistical_significance(control, treatment)
print(f"\nStatistical significance: {results['statistically_significant']}")
print(f"P-value: {results['p_value']:.4f}")
print(f"Relative lift: {results['relative_lift_percent']:.2f}%")
print(f"Interpretation: {results['interpretation']}")

# Check for Simpson's Paradox
control_segments = {
    "US": ABTestResult("control_US", 300, 40, []),
    "UK": ABTestResult("control_UK", 200, 10, [])
}

treatment_segments = {
    "US": ABTestResult("treatment_US", 400, 48, []),  # Better
    "UK": ABTestResult("treatment_UK", 120, 14, [])   # Better
}

paradox = debugger.detect_simpsons_paradox(control_segments, treatment_segments)
print(f"\nSimpson's Paradox: {paradox['paradox_detected']}")
print(f"Explanation: {paradox['explanation']}")

# Calculate required sample size
required_n = debugger.calculate_required_sample_size(
    baseline_rate=0.10,
    minimum_detectable_effect=0.10  # Detect 10% relative improvement
)
print(f"\nRequired sample size per variant: {required_n}")
```

**A/B test debugging checklist:**

- [ ] Sufficient sample size (use power analysis)
- [ ] Balanced assignment (50/50 or 70/30, not random)
- [ ] Random assignment (no selection bias)
- [ ] Statistical significance (p < 0.05)
- [ ] Practical significance (meaningful effect size)
- [ ] Check for Simpson's Paradox (segment analysis)
- [ ] Monitor for novelty effect (long-term trends)
- [ ] Validate metrics (correct calculation, no bugs)

---

### Part 5: Model Debugging (Wrong Predictions, Edge Cases)

**Common model issues:**
- Wrong predictions on edge cases
- High confidence wrong predictions
- Inconsistent behavior (same input, different output)
- Bias or fairness issues
- Input validation failures

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import torch

@dataclass
class PredictionError:
    """
    Failed prediction for analysis.
    """
    input_data: Any
    true_label: Any
    predicted_label: Any
    confidence: float
    error_type: str  # wrong_class, low_confidence, edge_case, etc.


class ModelDebugger:
    """
    Debug model prediction errors and edge cases.
    """

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.errors: List[PredictionError] = []

    def add_error(self, error: PredictionError):
        """Add prediction error for analysis."""
        self.errors.append(error)

    def find_error_patterns(self) -> Dict[str, List[PredictionError]]:
        """
        Find patterns in prediction errors.

        Patterns:
        - Errors on specific input types (long text, numbers, special chars)
        - Errors on specific classes (class imbalance?)
        - High-confidence errors (model overconfident)
        - Consistent errors (model learned wrong pattern)
        """
        patterns = {
            "high_confidence_errors": [],
            "low_confidence_errors": [],
            "edge_cases": [],
            "class_specific": {}
        }

        for error in self.errors:
            # High confidence but wrong
            if error.confidence > 0.9:
                patterns["high_confidence_errors"].append(error)

            # Low confidence (uncertain)
            elif error.confidence < 0.6:
                patterns["low_confidence_errors"].append(error)

            # Edge cases
            if error.error_type == "edge_case":
                patterns["edge_cases"].append(error)

            # Group by predicted class
            pred_class = str(error.predicted_label)
            if pred_class not in patterns["class_specific"]:
                patterns["class_specific"][pred_class] = []
            patterns["class_specific"][pred_class].append(error)

        return patterns

    def analyze_edge_cases(self) -> List[Dict]:
        """
        Analyze edge cases to understand failure modes.

        Edge case types:
        - Out-of-distribution inputs
        - Extreme values (very long, very short)
        - Special characters or formatting
        - Ambiguous inputs
        """
        edge_cases = [e for e in self.errors if e.error_type == "edge_case"]

        analyses = []
        for error in edge_cases:
            analysis = {
                "input": error.input_data,
                "true_label": error.true_label,
                "predicted_label": error.predicted_label,
                "confidence": error.confidence,
                "characteristics": self._characterize_input(error.input_data)
            }
            analyses.append(analysis)

        return analyses

    def _characterize_input(self, input_data: Any) -> Dict:
        """
        Characterize input to identify unusual features.
        """
        if isinstance(input_data, str):
            return {
                "type": "text",
                "length": len(input_data),
                "has_numbers": any(c.isdigit() for c in input_data),
                "has_special_chars": any(not c.isalnum() and not c.isspace() for c in input_data),
                "all_caps": input_data.isupper(),
                "all_lowercase": input_data.islower()
            }
        elif isinstance(input_data, (list, np.ndarray)):
            return {
                "type": "array",
                "shape": np.array(input_data).shape,
                "min": np.min(input_data),
                "max": np.max(input_data),
                "mean": np.mean(input_data)
            }
        else:
            return {"type": str(type(input_data))}

    def test_input_variations(
        self,
        input_data: Any,
        variations: List[str]
    ) -> Dict[str, Any]:
        """
        Test model on variations of input to check robustness.

        Variations:
        - case_change: Change case (upper/lower)
        - whitespace: Add/remove whitespace
        - typos: Introduce typos
        - paraphrase: Rephrase input

        Args:
            input_data: Original input
            variations: List of variation types to test

        Returns:
            Results for each variation
        """
        results = {}

        # Original prediction
        original_pred = self._predict(input_data)
        results["original"] = {
            "input": input_data,
            "prediction": original_pred
        }

        # Generate and test variations
        for var_type in variations:
            varied_input = self._generate_variation(input_data, var_type)
            varied_pred = self._predict(varied_input)

            results[var_type] = {
                "input": varied_input,
                "prediction": varied_pred,
                "consistent": varied_pred["label"] == original_pred["label"]
            }

        # Check consistency
        all_consistent = all(r.get("consistent", True) for r in results.values() if r != results["original"])

        return {
            "consistent": all_consistent,
            "results": results
        }

    def _generate_variation(self, input_data: str, variation_type: str) -> str:
        """
        Generate input variation.
        """
        if variation_type == "case_change":
            return input_data.upper() if input_data.islower() else input_data.lower()

        elif variation_type == "whitespace":
            return "  ".join(input_data.split())

        elif variation_type == "typos":
            # Simple typo: swap two adjacent characters
            if len(input_data) > 2:
                idx = len(input_data) // 2
                return input_data[:idx] + input_data[idx+1] + input_data[idx] + input_data[idx+2:]
            return input_data

        return input_data

    def _predict(self, input_data: Any) -> Dict:
        """
        Run model prediction.
        """
        # Simplified prediction (adapt to your model)
        # Example for text classification
        if self.tokenizer:
            inputs = self.tokenizer(input_data, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)

            probs = torch.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_class].item()

            return {
                "label": pred_class,
                "confidence": confidence
            }

        return {"label": None, "confidence": 0.0}

    def validate_inputs(self, inputs: List[Any]) -> List[Dict]:
        """
        Validate inputs before inference.

        Validation checks:
        - Type correctness
        - Value ranges
        - Format compliance
        - Size limits
        """
        validation_results = []

        for i, input_data in enumerate(inputs):
            issues = []

            if isinstance(input_data, str):
                # Text validation
                if len(input_data) == 0:
                    issues.append("Empty input")
                elif len(input_data) > 10000:
                    issues.append("Input too long (>10k chars)")

                if not input_data.strip():
                    issues.append("Only whitespace")

            validation_results.append({
                "index": i,
                "valid": len(issues) == 0,
                "issues": issues
            })

        return validation_results


# Example usage
class DummyModel:
    def __call__(self, input_ids, attention_mask):
        # Dummy model for demonstration
        return type('obj', (object,), {
            'logits': torch.randn(1, 3)
        })


class DummyTokenizer:
    def __call__(self, text, return_tensors=None):
        return {
            "input_ids": torch.randint(0, 1000, (1, 10)),
            "attention_mask": torch.ones(1, 10)
        }


model = DummyModel()
tokenizer = DummyTokenizer()
debugger = ModelDebugger(model, tokenizer)

# Add prediction errors
debugger.add_error(PredictionError(
    input_data="This is a test",
    true_label=1,
    predicted_label=2,
    confidence=0.95,
    error_type="high_confidence"
))

debugger.add_error(PredictionError(
    input_data="AAAAAAAAAAA",  # Edge case: all same character
    true_label=0,
    predicted_label=1,
    confidence=0.85,
    error_type="edge_case"
))

# Find error patterns
patterns = debugger.find_error_patterns()
print(f"High confidence errors: {len(patterns['high_confidence_errors'])}")
print(f"Edge cases: {len(patterns['edge_cases'])}")

# Analyze edge cases
edge_analyses = debugger.analyze_edge_cases()
for analysis in edge_analyses:
    print(f"\nEdge case: {analysis['input']}")
    print(f"Characteristics: {analysis['characteristics']}")

# Test input variations
variations_result = debugger.test_input_variations(
    "This is a test",
    ["case_change", "whitespace", "typos"]
)
print(f"\nInput variation consistency: {variations_result['consistent']}")
```

**Model debugging checklist:**

- [ ] Collect failed predictions with context
- [ ] Categorize errors (high confidence, edge cases, class-specific)
- [ ] Analyze input characteristics (what makes them fail?)
- [ ] Test input variations (robustness check)
- [ ] Validate inputs before inference (prevent bad inputs)
- [ ] Check for bias (fairness across groups)
- [ ] Add error cases to training data (improve model)

---

### Part 6: Logging Best Practices

**Good logging enables debugging. Bad logging creates noise.**

```python
import logging
import json
import sys
from datetime import datetime
from contextvars import ContextVar
from typing import Dict, Any, Optional
import traceback

# Context variable for request/trace ID
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class StructuredLogger:
    """
    Structured logging for production systems.

    Best practices:
    - JSON format (machine-readable)
    - Include context (request_id, user_id, etc.)
    - Log at appropriate levels
    - Include timing information
    - Don't log sensitive data
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self.JSONFormatter())
        self.logger.addHandler(handler)

    class JSONFormatter(logging.Formatter):
        """
        Format logs as JSON.
        """
        def format(self, record: logging.LogRecord) -> str:
            log_data = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }

            # Add request ID from context
            request_id = request_id_var.get()
            if request_id:
                log_data["request_id"] = request_id

            # Add extra fields
            if hasattr(record, "extra"):
                log_data.update(record.extra)

            # Add exception info
            if record.exc_info:
                log_data["exception"] = {
                    "type": record.exc_info[0].__name__,
                    "message": str(record.exc_info[1]),
                    "traceback": traceback.format_exception(*record.exc_info)
                }

            return json.dumps(log_data)

    def log(
        self,
        level: str,
        message: str,
        **kwargs
    ):
        """
        Log with structured context.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            **kwargs: Additional context fields
        """
        log_method = getattr(self.logger, level.lower())

        # Create LogRecord with extra fields
        extra = {"extra": kwargs}
        log_method(message, extra=extra)

    def debug(self, message: str, **kwargs):
        self.log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs):
        self.log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self.log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        self.log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs):
        self.log("CRITICAL", message, **kwargs)


class RequestLogger:
    """
    Log HTTP requests with full context.
    """

    def __init__(self):
        self.logger = StructuredLogger("api")

    def log_request(
        self,
        request_id: str,
        method: str,
        path: str,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """
        Log incoming request.
        """
        # Set request ID in context
        request_id_var.set(request_id)

        self.logger.info(
            "Request started",
            request_id=request_id,
            method=method,
            path=path,
            user_id=user_id,
            **kwargs
        )

    def log_response(
        self,
        request_id: str,
        status_code: int,
        duration_ms: float,
        **kwargs
    ):
        """
        Log response with timing.
        """
        level = "INFO" if status_code < 400 else "ERROR"

        self.logger.log(
            level,
            "Request completed",
            request_id=request_id,
            status_code=status_code,
            duration_ms=duration_ms,
            **kwargs
        )

    def log_error(
        self,
        request_id: str,
        error: Exception,
        **kwargs
    ):
        """
        Log request error with full context.
        """
        self.logger.error(
            "Request failed",
            request_id=request_id,
            error_type=type(error).__name__,
            error_message=str(error),
            **kwargs,
            exc_info=True
        )


class ModelInferenceLogger:
    """
    Log model inference with input/output context.
    """

    def __init__(self):
        self.logger = StructuredLogger("model")

    def log_inference(
        self,
        model_name: str,
        model_version: str,
        input_shape: tuple,
        output_shape: tuple,
        duration_ms: float,
        request_id: Optional[str] = None,
        **kwargs
    ):
        """
        Log model inference.
        """
        self.logger.info(
            "Model inference",
            model_name=model_name,
            model_version=model_version,
            input_shape=input_shape,
            output_shape=output_shape,
            duration_ms=duration_ms,
            request_id=request_id,
            **kwargs
        )

    def log_prediction_error(
        self,
        model_name: str,
        error: Exception,
        input_sample: Any,
        request_id: Optional[str] = None,
        **kwargs
    ):
        """
        Log prediction error with input context.

        Note: Be careful not to log sensitive data!
        """
        # Sanitize input (don't log full input if sensitive)
        input_summary = self._summarize_input(input_sample)

        self.logger.error(
            "Prediction failed",
            model_name=model_name,
            error_type=type(error).__name__,
            error_message=str(error),
            input_summary=input_summary,
            request_id=request_id,
            **kwargs,
            exc_info=True
        )

    def _summarize_input(self, input_sample: Any) -> Dict:
        """
        Summarize input without logging sensitive data.
        """
        if isinstance(input_sample, str):
            return {
                "type": "text",
                "length": len(input_sample),
                "preview": input_sample[:50] + "..." if len(input_sample) > 50 else input_sample
            }
        elif isinstance(input_sample, (list, tuple)):
            return {
                "type": "array",
                "length": len(input_sample)
            }
        else:
            return {
                "type": str(type(input_sample))
            }


# Example usage
request_logger = RequestLogger()
model_logger = ModelInferenceLogger()

# Log request
import uuid
import time

request_id = str(uuid.uuid4())
start_time = time.time()

request_logger.log_request(
    request_id=request_id,
    method="POST",
    path="/api/predict",
    user_id="user_123",
    client_ip="192.168.1.100"
)

# Log model inference
model_logger.log_inference(
    model_name="sentiment-classifier",
    model_version="v2.1",
    input_shape=(1, 512),
    output_shape=(1, 3),
    duration_ms=45.2,
    request_id=request_id,
    batch_size=1
)

# Log response
duration_ms = (time.time() - start_time) * 1000
request_logger.log_response(
    request_id=request_id,
    status_code=200,
    duration_ms=duration_ms
)

# Log error (example)
try:
    raise ValueError("Invalid input shape")
except Exception as e:
    request_logger.log_error(request_id, e, endpoint="/api/predict")
```

**What to log:**

| Level | What to Log | Example |
|-------|-------------|---------|
| DEBUG | Detailed diagnostic info | Variable values, function entry/exit |
| INFO | Normal operations | Request started, prediction completed |
| WARNING | Unexpected but handled | Retry attempt, fallback used |
| ERROR | Error conditions | API error, prediction failed |
| CRITICAL | System failure | Database down, out of memory |

**What NOT to log:**
- Passwords, API keys, tokens
- Credit card numbers, SSNs
- Full user data (GDPR violation)
- Large payloads (log summary instead)

**Logging checklist:**

- [ ] Use structured logging (JSON format)
- [ ] Include trace/request IDs (correlation)
- [ ] Log at appropriate levels
- [ ] Include timing information
- [ ] Don't log sensitive data
- [ ] Make logs queryable (structured fields)
- [ ] Include sufficient context for debugging
- [ ] Log errors with full stack traces

---

### Part 7: Rollback Procedures

**When to rollback:**
- Critical error rate spike (>5% errors)
- Significant metric regression (>10% drop)
- Security vulnerability discovered
- Cascading failures affecting downstream

**When NOT to rollback:**
- Minor errors (<1% error rate)
- Single user complaints (investigate first)
- Performance slightly worse (measure first)
- New feature not perfect (iterate instead)

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import subprocess

@dataclass
class DeploymentMetrics:
    """
    Metrics to monitor during deployment.
    """
    error_rate: float
    latency_p95_ms: float
    success_rate: float
    throughput_qps: float
    cpu_usage_percent: float
    memory_usage_percent: float


class RollbackDecider:
    """
    Decide whether to rollback based on metrics.
    """

    def __init__(
        self,
        baseline_metrics: DeploymentMetrics,
        thresholds: Dict[str, float]
    ):
        """
        Args:
            baseline_metrics: Metrics from previous stable version
            thresholds: Rollback thresholds (e.g., {"error_rate": 0.05})
        """
        self.baseline = baseline_metrics
        self.thresholds = thresholds

    def should_rollback(
        self,
        current_metrics: DeploymentMetrics
    ) -> Dict:
        """
        Decide if rollback is needed.

        Returns:
            Decision with reasoning
        """
        violations = []

        # Check error rate
        if current_metrics.error_rate > self.thresholds.get("error_rate", 0.05):
            violations.append({
                "metric": "error_rate",
                "baseline": self.baseline.error_rate,
                "current": current_metrics.error_rate,
                "threshold": self.thresholds["error_rate"],
                "severity": "CRITICAL"
            })

        # Check latency
        latency_increase = (current_metrics.latency_p95_ms - self.baseline.latency_p95_ms) / self.baseline.latency_p95_ms
        if latency_increase > self.thresholds.get("latency_increase", 0.25):  # 25% increase
            violations.append({
                "metric": "latency_p95_ms",
                "baseline": self.baseline.latency_p95_ms,
                "current": current_metrics.latency_p95_ms,
                "increase_percent": latency_increase * 100,
                "threshold": self.thresholds["latency_increase"] * 100,
                "severity": "HIGH"
            })

        # Check success rate
        success_drop = self.baseline.success_rate - current_metrics.success_rate
        if success_drop > self.thresholds.get("success_rate_drop", 0.05):  # 5pp drop
            violations.append({
                "metric": "success_rate",
                "baseline": self.baseline.success_rate,
                "current": current_metrics.success_rate,
                "drop": success_drop,
                "threshold": self.thresholds["success_rate_drop"],
                "severity": "CRITICAL"
            })

        should_rollback = len([v for v in violations if v["severity"] == "CRITICAL"]) > 0

        return {
            "should_rollback": should_rollback,
            "violations": violations,
            "reasoning": self._generate_reasoning(should_rollback, violations)
        }

    def _generate_reasoning(
        self,
        should_rollback: bool,
        violations: List[Dict]
    ) -> str:
        """
        Generate human-readable reasoning.
        """
        if not violations:
            return "All metrics within acceptable thresholds. No rollback needed."

        if should_rollback:
            critical = [v for v in violations if v["severity"] == "CRITICAL"]
            reasons = [f"{v['metric']} violated threshold" for v in critical]
            return f"ROLLBACK RECOMMENDED: {', '.join(reasons)}"
        else:
            return f"Minor issues detected but below rollback threshold. Monitor closely."


class RollbackExecutor:
    """
    Execute rollback procedure.
    """

    def __init__(self, deployment_system: str = "kubernetes"):
        self.deployment_system = deployment_system

    def rollback(
        self,
        service_name: str,
        previous_version: str,
        preserve_evidence: bool = True
    ) -> Dict:
        """
        Execute rollback to previous version.

        Args:
            service_name: Service to rollback
            previous_version: Version to rollback to
            preserve_evidence: Capture logs/metrics before rollback

        Returns:
            Rollback result
        """
        print(f"Starting rollback: {service_name} -> {previous_version}")

        # Step 1: Preserve evidence
        if preserve_evidence:
            evidence = self._preserve_evidence(service_name)
            print(f"Evidence preserved: {evidence}")

        # Step 2: Execute rollback
        if self.deployment_system == "kubernetes":
            result = self._rollback_kubernetes(service_name, previous_version)
        elif self.deployment_system == "docker":
            result = self._rollback_docker(service_name, previous_version)
        else:
            result = {"success": False, "error": "Unknown deployment system"}

        return result

    def _preserve_evidence(self, service_name: str) -> Dict:
        """
        Capture logs and metrics before rollback.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Capture logs (last 1000 lines)
        log_file = f"/tmp/{service_name}_rollback_{timestamp}.log"

        # Simplified: In production, use proper log aggregation
        print(f"Capturing logs to {log_file}")

        # Capture metrics snapshot
        metrics_file = f"/tmp/{service_name}_metrics_{timestamp}.json"
        print(f"Capturing metrics to {metrics_file}")

        return {
            "log_file": log_file,
            "metrics_file": metrics_file,
            "timestamp": timestamp
        }

    def _rollback_kubernetes(
        self,
        service_name: str,
        version: str
    ) -> Dict:
        """
        Rollback Kubernetes deployment.
        """
        try:
            # Option 1: Rollback to previous revision
            cmd = f"kubectl rollout undo deployment/{service_name}"

            # Option 2: Rollback to specific version
            # cmd = f"kubectl rollout undo deployment/{service_name} --to-revision={version}"

            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                check=True
            )

            # Wait for rollout
            wait_cmd = f"kubectl rollout status deployment/{service_name}"
            subprocess.run(
                wait_cmd.split(),
                check=True,
                timeout=300  # 5 min timeout
            )

            return {
                "success": True,
                "service": service_name,
                "version": version,
                "output": result.stdout
            }

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": str(e),
                "output": e.stderr
            }

    def _rollback_docker(
        self,
        service_name: str,
        version: str
    ) -> Dict:
        """
        Rollback Docker service.
        """
        try:
            cmd = f"docker service update --image {service_name}:{version} {service_name}"

            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                check=True
            )

            return {
                "success": True,
                "service": service_name,
                "version": version,
                "output": result.stdout
            }

        except subprocess.CalledProcessError as e:
            return {
                "success": False,
                "error": str(e),
                "output": e.stderr
            }


# Example usage
baseline = DeploymentMetrics(
    error_rate=0.01,
    latency_p95_ms=200,
    success_rate=0.95,
    throughput_qps=100,
    cpu_usage_percent=50,
    memory_usage_percent=60
)

thresholds = {
    "error_rate": 0.05,  # 5% error rate
    "latency_increase": 0.25,  # 25% increase
    "success_rate_drop": 0.05  # 5pp drop
}

decider = RollbackDecider(baseline, thresholds)

# Simulate bad deployment
current = DeploymentMetrics(
    error_rate=0.08,  # High!
    latency_p95_ms=300,  # High!
    success_rate=0.88,  # Low!
    throughput_qps=90,
    cpu_usage_percent=70,
    memory_usage_percent=65
)

decision = decider.should_rollback(current)
print(f"Should rollback: {decision['should_rollback']}")
print(f"Reasoning: {decision['reasoning']}")

if decision['should_rollback']:
    executor = RollbackExecutor(deployment_system="kubernetes")
    result = executor.rollback(
        service_name="ml-api",
        previous_version="v1.2.3",
        preserve_evidence=True
    )
    print(f"Rollback result: {result}")
```

**Rollback checklist:**

- [ ] Preserve evidence (logs, metrics, traces)
- [ ] Document rollback reason
- [ ] Execute rollback (kubectl/docker/terraform)
- [ ] Verify metrics return to normal
- [ ] Notify team and stakeholders
- [ ] Schedule post-mortem
- [ ] Fix issue in development
- [ ] Re-deploy with fix

---

### Part 8: Post-Mortem Process

**Goal:** Learn from incidents to prevent recurrence.

**Post-mortem is blameless:** Focus on systems and processes, not individuals.

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class IncidentTimeline:
    """
    Timeline event during incident.
    """
    timestamp: datetime
    event: str
    actor: str  # Person, system, or automation
    action: str


@dataclass
class ActionItem:
    """
    Post-mortem action item.
    """
    description: str
    owner: str
    due_date: datetime
    priority: str  # CRITICAL, HIGH, MEDIUM, LOW
    status: str = "TODO"  # TODO, IN_PROGRESS, DONE


class PostMortem:
    """
    Structured post-mortem document.
    """

    def __init__(
        self,
        incident_id: str,
        title: str,
        date: datetime,
        severity: str,
        duration_minutes: int
    ):
        self.incident_id = incident_id
        self.title = title
        self.date = date
        self.severity = severity
        self.duration_minutes = duration_minutes

        self.summary: str = ""
        self.impact: Dict = {}
        self.timeline: List[IncidentTimeline] = []
        self.root_cause: str = ""
        self.contributing_factors: List[str] = []
        self.what_went_well: List[str] = []
        self.what_went_wrong: List[str] = []
        self.action_items: List[ActionItem] = []

    def add_timeline_event(
        self,
        timestamp: datetime,
        event: str,
        actor: str,
        action: str
    ):
        """
        Add event to incident timeline.
        """
        self.timeline.append(IncidentTimeline(
            timestamp=timestamp,
            event=event,
            actor=actor,
            action=action
        ))

    def set_root_cause(self, root_cause: str):
        """
        Document root cause.
        """
        self.root_cause = root_cause

    def add_contributing_factor(self, factor: str):
        """
        Add contributing factor (not root cause but made it worse).
        """
        self.contributing_factors.append(factor)

    def add_action_item(
        self,
        description: str,
        owner: str,
        due_date: datetime,
        priority: str = "HIGH"
    ):
        """
        Add action item for prevention.
        """
        self.action_items.append(ActionItem(
            description=description,
            owner=owner,
            due_date=due_date,
            priority=priority
        ))

    def generate_report(self) -> str:
        """
        Generate post-mortem report.
        """
        report = f"""
# Post-Mortem: {self.title}

**Incident ID:** {self.incident_id}
**Date:** {self.date.strftime('%Y-%m-%d %H:%M UTC')}
**Severity:** {self.severity}
**Duration:** {self.duration_minutes} minutes

## Summary

{self.summary}

## Impact

{self._format_impact()}

## Timeline

{self._format_timeline()}

## Root Cause

{self.root_cause}

## Contributing Factors

{self._format_list(self.contributing_factors)}

## What Went Well

{self._format_list(self.what_went_well)}

## What Went Wrong

{self._format_list(self.what_went_wrong)}

## Action Items

{self._format_action_items()}

---

**Review:** This post-mortem should be reviewed by the team and approved by engineering leadership.

**Follow-up:** Track action items to completion. Schedule follow-up review in 30 days.
"""
        return report

    def _format_impact(self) -> str:
        """Format impact section."""
        lines = []
        for key, value in self.impact.items():
            lines.append(f"- **{key}:** {value}")
        return "\n".join(lines) if lines else "No impact documented."

    def _format_timeline(self) -> str:
        """Format timeline section."""
        lines = []
        for event in sorted(self.timeline, key=lambda e: e.timestamp):
            time_str = event.timestamp.strftime('%H:%M:%S')
            lines.append(f"- **{time_str}** [{event.actor}] {event.event} → {event.action}")
        return "\n".join(lines) if lines else "No timeline documented."

    def _format_list(self, items: List[str]) -> str:
        """Format list of items."""
        return "\n".join(f"- {item}" for item in items) if items else "None."

    def _format_action_items(self) -> str:
        """Format action items."""
        if not self.action_items:
            return "No action items."

        lines = []
        for item in sorted(self.action_items, key=lambda x: x.priority):
            due = item.due_date.strftime('%Y-%m-%d')
            lines.append(f"- [{item.priority}] {item.description} (Owner: {item.owner}, Due: {due})")

        return "\n".join(lines)


# Example post-mortem
from datetime import timedelta

pm = PostMortem(
    incident_id="INC-2025-042",
    title="API Latency Spike Causing Timeouts",
    date=datetime(2025, 1, 15, 14, 30),
    severity="HIGH",
    duration_minutes=45
)

pm.summary = """
At 14:30 UTC, API latency spiked from 200ms to 5000ms, causing widespread timeouts.
Error rate increased from 0.5% to 15%. Incident was resolved by scaling up database
connection pool and restarting API servers. No data loss occurred.
"""

pm.impact = {
    "Users affected": "~5,000 users (10% of active users)",
    "Requests failed": "~15,000 requests",
    "Revenue impact": "$2,500 (estimated)",
    "Customer complaints": "23 support tickets"
}

# Timeline
pm.add_timeline_event(
    datetime(2025, 1, 15, 14, 30),
    "Latency spike detected",
    "Monitoring System",
    "Alert sent to on-call"
)

pm.add_timeline_event(
    datetime(2025, 1, 15, 14, 32),
    "On-call engineer acknowledged",
    "Engineer A",
    "Started investigation"
)

pm.add_timeline_event(
    datetime(2025, 1, 15, 14, 40),
    "Root cause identified: DB connection pool exhausted",
    "Engineer A",
    "Scaled connection pool from 10 to 50"
)

pm.add_timeline_event(
    datetime(2025, 1, 15, 14, 45),
    "Restarted API servers",
    "Engineer A",
    "Latency returned to normal"
)

pm.add_timeline_event(
    datetime(2025, 1, 15, 15, 15),
    "Incident resolved",
    "Engineer A",
    "Monitoring confirmed stability"
)

# Root cause and factors
pm.set_root_cause(
    "Database connection pool size (10) was too small for peak traffic (100 concurrent requests). "
    "Connection pool exhaustion caused requests to queue, leading to timeouts."
)

pm.add_contributing_factor("No monitoring for connection pool utilization")
pm.add_contributing_factor("Connection pool size not load tested")
pm.add_contributing_factor("No auto-scaling for database connections")

# What went well/wrong
pm.what_went_well = [
    "Monitoring detected issue within 2 minutes",
    "On-call responded quickly (2 min to acknowledgment)",
    "Root cause identified in 10 minutes",
    "No data loss or corruption"
]

pm.what_went_wrong = [
    "Connection pool not sized for peak traffic",
    "No monitoring for connection pool metrics",
    "Load testing didn't include database connection limits",
    "Incident affected 10% of users for 45 minutes"
]

# Action items
pm.add_action_item(
    "Add monitoring and alerting for DB connection pool utilization (alert at 80%)",
    "Engineer B",
    datetime.now() + timedelta(days=3),
    "CRITICAL"
)

pm.add_action_item(
    "Implement auto-scaling for DB connection pool based on traffic",
    "Engineer C",
    datetime.now() + timedelta(days=7),
    "HIGH"
)

pm.add_action_item(
    "Update load testing to include DB connection limits",
    "Engineer A",
    datetime.now() + timedelta(days=7),
    "HIGH"
)

pm.add_action_item(
    "Document connection pool sizing guidelines for future services",
    "Engineer D",
    datetime.now() + timedelta(days=14),
    "MEDIUM"
)

# Generate report
report = pm.generate_report()
print(report)
```

**Post-mortem checklist:**

- [ ] Schedule post-mortem meeting (within 48 hours)
- [ ] Invite all involved parties
- [ ] Document timeline (facts, not speculation)
- [ ] Identify root cause (not symptoms)
- [ ] List contributing factors
- [ ] What went well / what went wrong
- [ ] Create action items (owner, due date, priority)
- [ ] Review and approve report
- [ ] Track action items to completion
- [ ] Follow-up review in 30 days

**Key principles:**
- **Blameless:** Focus on systems, not people
- **Fact-based:** Use evidence, not opinions
- **Actionable:** Create concrete prevention measures
- **Timely:** Complete within 1 week of incident
- **Shared:** Distribute to entire team

---

### Part 9: Production Forensics (Traces, Logs, Metrics Correlation)

**Goal:** Correlate traces, logs, and metrics to understand incident.

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

@dataclass
class Trace:
    """
    Distributed trace span.
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    service_name: str
    operation_name: str
    start_time: datetime
    duration_ms: float
    status: str  # OK, ERROR
    tags: Dict[str, str]


@dataclass
class LogEntry:
    """
    Structured log entry.
    """
    timestamp: datetime
    level: str
    service: str
    message: str
    trace_id: Optional[str]
    metadata: Dict


@dataclass
class MetricDataPoint:
    """
    Time-series metric data point.
    """
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str]


class ProductionForensics:
    """
    Correlate traces, logs, and metrics for incident investigation.
    """

    def __init__(self):
        self.traces: List[Trace] = []
        self.logs: List[LogEntry] = []
        self.metrics: List[MetricDataPoint] = []

    def add_trace(self, trace: Trace):
        self.traces.append(trace)

    def add_log(self, log: LogEntry):
        self.logs.append(log)

    def add_metric(self, metric: MetricDataPoint):
        self.metrics.append(metric)

    def investigate_slow_request(
        self,
        trace_id: str
    ) -> Dict:
        """
        Investigate slow request using trace, logs, and metrics.

        Args:
            trace_id: Trace ID of slow request

        Returns:
            Investigation results
        """
        # Get trace spans
        trace_spans = [t for t in self.traces if t.trace_id == trace_id]

        if not trace_spans:
            return {"error": "Trace not found"}

        # Sort by start time
        trace_spans.sort(key=lambda s: s.start_time)

        # Calculate total duration
        total_duration = sum(s.duration_ms for s in trace_spans if not s.parent_span_id)

        # Find slowest span
        slowest_span = max(trace_spans, key=lambda s: s.duration_ms)

        # Get logs for this trace
        trace_logs = [l for l in self.logs if l.trace_id == trace_id]
        trace_logs.sort(key=lambda l: l.timestamp)

        # Check for errors
        error_logs = [l for l in trace_logs if l.level == "ERROR"]

        # Get metrics during request time
        start_time = trace_spans[0].start_time
        end_time = start_time + timedelta(milliseconds=total_duration)

        relevant_metrics = [
            m for m in self.metrics
            if start_time <= m.timestamp <= end_time
        ]

        return {
            "trace_id": trace_id,
            "total_duration_ms": total_duration,
            "num_spans": len(trace_spans),
            "slowest_span": {
                "service": slowest_span.service_name,
                "operation": slowest_span.operation_name,
                "duration_ms": slowest_span.duration_ms,
                "percentage": (slowest_span.duration_ms / total_duration * 100) if total_duration > 0 else 0
            },
            "error_count": len(error_logs),
            "errors": [
                {"timestamp": l.timestamp.isoformat(), "message": l.message}
                for l in error_logs
            ],
            "trace_breakdown": [
                {
                    "service": s.service_name,
                    "operation": s.operation_name,
                    "duration_ms": s.duration_ms,
                    "percentage": (s.duration_ms / total_duration * 100) if total_duration > 0 else 0
                }
                for s in trace_spans
            ],
            "metrics_during_request": [
                {
                    "metric": m.metric_name,
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in relevant_metrics
            ]
        }

    def find_correlated_errors(
        self,
        time_window_minutes: int = 10
    ) -> List[Dict]:
        """
        Find errors that occurred around the same time.

        Args:
            time_window_minutes: Time window for correlation

        Returns:
            Clusters of correlated errors
        """
        error_logs = [l for l in self.logs if l.level == "ERROR"]
        error_logs.sort(key=lambda l: l.timestamp)

        if not error_logs:
            return []

        # Cluster errors by time
        clusters = []
        current_cluster = [error_logs[0]]

        for log in error_logs[1:]:
            time_diff = (log.timestamp - current_cluster[-1].timestamp).total_seconds() / 60

            if time_diff <= time_window_minutes:
                current_cluster.append(log)
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [log]

        if len(current_cluster) > 1:
            clusters.append(current_cluster)

        # Analyze each cluster
        results = []
        for cluster in clusters:
            services = set(l.service for l in cluster)
            messages = set(l.message for l in cluster)

            results.append({
                "start_time": cluster[0].timestamp.isoformat(),
                "end_time": cluster[-1].timestamp.isoformat(),
                "error_count": len(cluster),
                "services_affected": list(services),
                "unique_errors": list(messages)
            })

        return results

    def analyze_metric_anomaly(
        self,
        metric_name: str,
        anomaly_time: datetime,
        window_minutes: int = 5
    ) -> Dict:
        """
        Analyze what happened around metric anomaly.

        Args:
            metric_name: Metric that had anomaly
            anomaly_time: When anomaly occurred
            window_minutes: Time window to analyze

        Returns:
            Analysis results
        """
        start_time = anomaly_time - timedelta(minutes=window_minutes)
        end_time = anomaly_time + timedelta(minutes=window_minutes)

        # Get metric values
        metric_values = [
            m for m in self.metrics
            if m.metric_name == metric_name and start_time <= m.timestamp <= end_time
        ]

        # Get logs during this time
        logs_during = [
            l for l in self.logs
            if start_time <= l.timestamp <= end_time
        ]

        # Get traces during this time
        traces_during = [
            t for t in self.traces
            if start_time <= t.start_time <= end_time
        ]

        # Count errors
        error_count = len([l for l in logs_during if l.level == "ERROR"])
        failed_traces = len([t for t in traces_during if t.status == "ERROR"])

        return {
            "metric_name": metric_name,
            "anomaly_time": anomaly_time.isoformat(),
            "window_minutes": window_minutes,
            "metric_values": [
                {"timestamp": m.timestamp.isoformat(), "value": m.value}
                for m in metric_values
            ],
            "error_count_during_window": error_count,
            "failed_traces_during_window": failed_traces,
            "top_errors": self._get_top_errors(logs_during, limit=5),
            "services_involved": list(set(t.service_name for t in traces_during))
        }

    def _get_top_errors(self, logs: List[LogEntry], limit: int = 5) -> List[Dict]:
        """
        Get most common error messages.
        """
        from collections import Counter

        error_logs = [l for l in logs if l.level == "ERROR"]
        error_messages = [l.message for l in error_logs]

        counter = Counter(error_messages)

        return [
            {"message": msg, "count": count}
            for msg, count in counter.most_common(limit)
        ]


# Example usage
forensics = ProductionForensics()

# Simulate data
trace_id = "trace-123"

# Add trace spans
forensics.add_trace(Trace(
    trace_id=trace_id,
    span_id="span-1",
    parent_span_id=None,
    service_name="api-gateway",
    operation_name="POST /predict",
    start_time=datetime(2025, 1, 15, 14, 30, 0),
    duration_ms=5000,
    status="OK",
    tags={"user_id": "user_123"}
))

forensics.add_trace(Trace(
    trace_id=trace_id,
    span_id="span-2",
    parent_span_id="span-1",
    service_name="ml-service",
    operation_name="model_inference",
    start_time=datetime(2025, 1, 15, 14, 30, 0, 500000),
    duration_ms=4500,  # Slow!
    status="OK",
    tags={"model": "sentiment-classifier"}
))

# Add logs
forensics.add_log(LogEntry(
    timestamp=datetime(2025, 1, 15, 14, 30, 3),
    level="WARNING",
    service="ml-service",
    message="High inference latency detected",
    trace_id=trace_id,
    metadata={"latency_ms": 4500}
))

# Add metrics
forensics.add_metric(MetricDataPoint(
    timestamp=datetime(2025, 1, 15, 14, 30, 0),
    metric_name="api_latency_ms",
    value=5000,
    tags={"service": "api-gateway"}
))

# Investigate slow request
investigation = forensics.investigate_slow_request(trace_id)
print(json.dumps(investigation, indent=2))
```

**Forensics checklist:**

- [ ] Identify affected time window
- [ ] Collect traces for failed/slow requests
- [ ] Collect logs with matching trace IDs
- [ ] Collect metrics during time window
- [ ] Correlate traces + logs + metrics
- [ ] Identify slowest operations (trace breakdown)
- [ ] Find error patterns (log analysis)
- [ ] Check metric anomalies (spikes/drops)
- [ ] Build timeline of events

---

## Summary

**Production debugging is systematic investigation, not random guessing.**

**Core methodology:**
1. **Reproduce** → Create minimal, deterministic reproduction
2. **Profile** → Use data, not intuition (py-spy, torch.profiler)
3. **Diagnose** → Find root cause, not symptoms
4. **Fix** → Targeted fix verified by tests
5. **Verify** → Prove fix works in production
6. **Document** → Post-mortem for prevention

**Key principles:**
- **Evidence-based:** Collect data before forming hypothesis
- **Systematic:** Follow debugging framework, don't skip steps
- **Root cause:** Fix the cause, not symptoms
- **Verification:** Prove fix works before closing
- **Prevention:** Add monitoring, tests, and documentation

**Production debugging toolkit:**
- Performance profiling: py-spy, torch.profiler, cProfile
- Error analysis: Categorize, find patterns, identify root cause
- A/B test debugging: Statistical significance, Simpson's paradox
- Model debugging: Edge cases, input variations, robustness
- Logging: Structured, with trace IDs and context
- Rollback: Preserve evidence, rollback quickly, fix properly
- Post-mortems: Blameless, actionable, prevent recurrence
- Forensics: Correlate traces, logs, metrics

**Common pitfalls to avoid:**
- Random changes without reproduction
- Guessing bottlenecks without profiling
- Bad logging (no context, unstructured)
- Panic rollback without learning
- Skipping post-mortems

Without systematic debugging, you fight the same fires repeatedly. With systematic debugging, you prevent fires from starting.

---

## REFACTOR Phase: Pressure Tests

### Pressure Test 1: Random Changes Without Investigation

**Scenario:** Model latency spiked from 100ms to 500ms. Engineer makes random changes hoping to fix it.

**Test:** Verify skill prevents random changes and enforces systematic investigation.

**Expected behavior:**
- ✅ Refuse to make changes without reproduction
- ✅ Require profiling data before optimization
- ✅ Collect evidence (metrics, logs, traces)
- ✅ Form hypothesis based on data
- ✅ Verify hypothesis before implementing fix

**Failure mode:** Makes parameter changes without profiling or understanding root cause.

---

### Pressure Test 2: No Profiling Before Optimization

**Scenario:** API is slow. Engineer says "Database is probably the bottleneck, let's add caching."

**Test:** Verify skill requires profiling data before optimization.

**Expected behavior:**
- ✅ Demand profiling data (py-spy flamegraph, query profiler)
- ✅ Identify actual bottleneck from profile
- ✅ Verify bottleneck hypothesis
- ✅ Optimize proven bottleneck, not guessed one

**Failure mode:** Optimizes based on intuition without profiling data.

---

### Pressure Test 3: Useless Logging

**Scenario:** Production error occurred but logs don't have enough context to debug.

**Test:** Verify skill enforces structured logging with context.

**Expected behavior:**
- ✅ Use structured logging (JSON format)
- ✅ Include trace/request IDs for correlation
- ✅ Log sufficient context (user_id, endpoint, input summary)
- ✅ Don't log sensitive data (passwords, PII)

**Failure mode:** Logs "Error occurred" with no context, making debugging impossible.

---

### Pressure Test 4: Immediate Rollback Without Evidence

**Scenario:** Error rate increased to 2%. Engineer wants to rollback immediately.

**Test:** Verify skill preserves evidence before rollback.

**Expected behavior:**
- ✅ Assess severity (2% error rate = investigate, not immediate rollback)
- ✅ Preserve evidence (logs, metrics, traces)
- ✅ Investigate root cause while monitoring
- ✅ Only rollback if critical threshold (>5% errors, cascading failures)

**Failure mode:** Rollbacks immediately without preserving evidence or assessing severity.

---

### Pressure Test 5: No Root Cause Analysis

**Scenario:** API returns 500 errors. Engineer fixes symptom (restart service) but not root cause.

**Test:** Verify skill identifies and fixes root cause.

**Expected behavior:**
- ✅ Distinguish symptom ("500 errors") from root cause ("connection pool exhausted")
- ✅ Investigate why symptom occurred
- ✅ Fix root cause (increase pool size, add monitoring)
- ✅ Verify fix addresses root cause

**Failure mode:** Fixes symptom (restart) but root cause remains, issue repeats.

---

### Pressure Test 6: A/B Test Without Statistical Significance

**Scenario:** A/B test with 50 samples per variant shows 5% improvement. Engineer wants to ship.

**Test:** Verify skill requires statistical significance.

**Expected behavior:**
- ✅ Calculate required sample size (power analysis)
- ✅ Check statistical significance (p-value < 0.05)
- ✅ Reject if insufficient samples or not significant
- ✅ Check for Simpson's Paradox (segment analysis)

**Failure mode:** Ships based on insufficient data or non-significant results.

---

### Pressure Test 7: Model Edge Case Ignored

**Scenario:** Model fails on all-caps input but works on normal case. Engineer ignores edge case.

**Test:** Verify skill investigates and handles edge cases.

**Expected behavior:**
- ✅ Collect edge case examples
- ✅ Categorize edge cases (all caps, special chars, long inputs)
- ✅ Add input validation or preprocessing
- ✅ Add edge cases to test suite

**Failure mode:** Ignores edge cases as "not important" without investigation.

---

### Pressure Test 8: Skip Post-Mortem

**Scenario:** Incident resolved. Engineer closes ticket and moves on without post-mortem.

**Test:** Verify skill enforces post-mortem process.

**Expected behavior:**
- ✅ Require post-mortem for all incidents (severity HIGH or above)
- ✅ Document timeline, root cause, action items
- ✅ Make post-mortem blameless (systems, not people)
- ✅ Track action items to completion

**Failure mode:** Skips post-mortem, incident repeats, no learning.

---

### Pressure Test 9: No Metrics Correlation

**Scenario:** Latency spike at 2pm. Engineer looks at logs but not metrics or traces.

**Test:** Verify skill correlates traces, logs, and metrics.

**Expected behavior:**
- ✅ Collect traces for affected requests
- ✅ Collect logs with matching trace IDs
- ✅ Collect metrics during time window
- ✅ Correlate all three to find root cause

**Failure mode:** Only looks at logs, misses critical information in traces/metrics.

---

### Pressure Test 10: High Confidence Wrong Predictions Ignored

**Scenario:** Model makes high-confidence (>95%) wrong predictions. Engineer says "accuracy is good overall."

**Test:** Verify skill investigates high-confidence errors.

**Expected behavior:**
- ✅ Separate high-confidence errors from low-confidence
- ✅ Analyze input characteristics causing high-confidence errors
- ✅ Test input variations for robustness
- ✅ Add error cases to training data or add validation

**Failure mode:** Ignores high-confidence errors because "overall accuracy is fine."
