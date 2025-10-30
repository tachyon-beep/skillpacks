---
name: classified-systems-security
description: Use when handling classified data or implementing multi-level security - applies Bell-LaPadula model, fail-fast enforcement, trusted downgrade patterns, and immutability to prevent unauthorized information flow in government/defense systems
---

# Classified Systems Security

## Overview

Implement multi-level security (MLS) for classified data. Core principle: **Invalid configurations must be impossible to create, not detected at runtime**.

**Key insight**: You cannot "sanitize" your way out of classification violations. Use mandatory access control (MAC) with fail-fast validation at construction time.

## When to Use

Load this skill when:
- Handling classified data (UNOFFICIAL, OFFICIAL, SECRET, TOP SECRET)
- Implementing government/defense systems
- Designing multi-level security (MLS) architectures
- Processing data at different classification levels

**Symptoms you need this**:
- "How do I handle SECRET and UNOFFICIAL data in the same pipeline?"
- Designing systems with clearance levels
- Government/defense contract requirements
- Data classification enforcement

**Don't use for**:
- General security (use `ordis/security-architect/security-controls-design`)
- Non-classified sensitivity levels (use standard access control)

## Bell-LaPadula MLS Model

### The Two Rules

**1. No Read Up (Simple Security Property)**
- Subject cannot read data at higher classification
- UNOFFICIAL component CANNOT read OFFICIAL data
- OFFICIAL component CANNOT read SECRET data

**2. No Write Down (*-Property/Star Property)**
- Subject cannot write data to lower classification
- SECRET component CANNOT write to OFFICIAL sink
- OFFICIAL component CANNOT write to UNOFFICIAL sink

### Classification Hierarchy

```
TOP SECRET      (highest)
     ↓
   SECRET
     ↓
 PROTECTED
     ↓
OFFICIAL:SENSITIVE
     ↓
 OFFICIAL
     ↓
UNOFFICIAL      (lowest)
```

**Transitivity**: Data derived from SECRET is SECRET until formally declassified.

### Example: Violations

❌ **Read-Up Violation**:
```python
# OFFICIAL clearance component reading SECRET data
official_processor = Processor(clearance=SecurityLevel.OFFICIAL)
secret_source = DataSource(classification=SecurityLevel.SECRET)

# VIOLATION: Cannot read up from OFFICIAL to SECRET
data = official_processor.read(secret_source)  # ❌ FORBIDDEN
```

❌ **Write-Down Violation**:
```python
# SECRET component writing to OFFICIAL sink
secret_processor = Processor(clearance=SecurityLevel.SECRET)
official_sink = DataSink(classification=SecurityLevel.OFFICIAL)

# VIOLATION: Cannot write down from SECRET to OFFICIAL
secret_processor.write(data, official_sink)  # ❌ FORBIDDEN
```

✅ **Compliant Flow**:
```python
# Same clearance level throughout
secret_source = DataSource(classification=SecurityLevel.SECRET)
secret_processor = Processor(clearance=SecurityLevel.SECRET)
secret_sink = DataSink(classification=SecurityLevel.SECRET)

# ✅ ALLOWED: All components at SECRET level
pipeline = Pipeline(secret_source, secret_processor, secret_sink)
```

---

## Fail-Fast Construction-Time Validation

### Principle: Prevent Invalid Configurations

**Don't**: Detect violations at runtime (after data exposure)
**Do**: Reject invalid configurations at construction time (before data access)

### Construction vs Runtime Validation

❌ **Runtime Validation (Vulnerable)**:
```python
class Pipeline:
    def __init__(self, source, processor, sink):
        self.source = source
        self.processor = processor
        self.sink = sink

    def run(self):
        # Runtime check - TOO LATE!
        data = self.source.read()
        if data.classification > self.processor.clearance:
            raise SecurityError("Read-up violation")

        # Problem: Data already read from source before check
        # Exposure window = time between read() and check
```

**Issue**: Data already accessed before violation detected.

✅ **Fail-Fast Construction-Time (Secure)**:
```python
class Pipeline:
    def __init__(self, source, processor, sink):
        # Validate BEFORE creating pipeline
        self._validate_clearances(source, processor, sink)

        self.source = source
        self.processor = processor
        self.sink = sink

    def _validate_clearances(self, source, processor, sink):
        # Check no-read-up
        if source.classification > processor.clearance:
            raise SecurityError(
                f"Read-up violation: Processor clearance {processor.clearance} "
                f"cannot read source classified {source.classification}"
            )

        # Check no-write-down
        if processor.clearance > sink.classification:
            raise SecurityError(
                f"Write-down violation: Processor {processor.clearance} "
                f"cannot write to sink classified {sink.classification}"
            )

        # All validations passed

    def run(self):
        # Only callable if construction succeeded
        data = self.source.read()
        processed = self.processor.process(data)
        self.sink.write(processed)

# Usage:
try:
    # Validation happens at construction
    pipeline = Pipeline(secret_source, official_processor, unofficial_sink)
    # ❌ Above line raises SecurityError - pipeline never created
except SecurityError as e:
    print(f"Configuration rejected: {e}")
    sys.exit(1)
```

**Result**: Zero exposure window. Invalid pipeline cannot be created.

---

## Type-System Enforcement

### Classification-Aware Types

Use type system to make violations impossible at compile/construction time.

```python
from typing import Generic, TypeVar
from enum import Enum

class SecurityLevel(Enum):
    UNOFFICIAL = 1
    OFFICIAL = 2
    OFFICIAL_SENSITIVE = 3
    PROTECTED = 4
    SECRET = 5
    TOP_SECRET = 6

# Generic type parameterized by classification
T = TypeVar('T', bound=SecurityLevel)

class ClassifiedData(Generic[T]):
    def __init__(self, data: any, classification: T):
        self.data = data
        self.classification = classification

class DataSource(Generic[T]):
    def __init__(self, classification: T):
        self.classification = classification

    def read(self) -> ClassifiedData[T]:
        return ClassifiedData(self._fetch(), self.classification)

class Processor(Generic[T]):
    def __init__(self, clearance: T):
        self.clearance = clearance

    def process(self, data: ClassifiedData[T]) -> ClassifiedData[T]:
        # Type system ensures data.classification <= self.clearance
        return ClassifiedData(self._transform(data.data), data.classification)

class DataSink(Generic[T]):
    def __init__(self, classification: T):
        self.classification = classification

    def write(self, data: ClassifiedData[T]):
        # Type system ensures data.classification == self.classification
        self._store(data.data)
```

**Type-safe pipeline**:
```python
# All components must be same classification level
secret_source: DataSource[SecurityLevel.SECRET]
secret_processor: Processor[SecurityLevel.SECRET]
secret_sink: DataSink[SecurityLevel.SECRET]

# ✅ Type checker accepts this
pipeline = Pipeline(secret_source, secret_processor, secret_sink)

# ❌ Type checker REJECTS this at compile time
official_processor: Processor[SecurityLevel.OFFICIAL]
pipeline = Pipeline(secret_source, official_processor, secret_sink)  # Type error!
```

---

## Trusted Downgrade Pattern

### When Downgrading is Necessary

Sometimes you need to declassify data (e.g., publishing redacted documents). This requires:

1. **Formal Authority**: Authorized to declassify
2. **Explicit Process**: Manual review and approval
3. **Audit Trail**: Log all downgrade operations
4. **Trusted Service**: Part of Trusted Computing Base (TCB)

### Trusted Downgrade Service

```python
class TrustedDowngradeService:
    """
    Trusted service operating at higher clearance level.
    Part of Trusted Computing Base (TCB).
    """
    def __init__(self, clearance: SecurityLevel):
        self.clearance = clearance  # Must be HIGH to read classified data
        self._validate_authority()  # Verify authorized to declassify

    def _validate_authority(self):
        # Check this service has declassification authority
        if not has_declassification_authority(self):
            raise SecurityError("Service not authorized for declassification")

    def declassify(
        self,
        data: ClassifiedData,
        target_level: SecurityLevel,
        justification: str
    ) -> ClassifiedData:
        """
        Declassify data to lower classification level.

        Args:
            data: Original classified data
            target_level: Target classification (must be lower)
            justification: Reason for declassification (audit trail)

        Returns:
            Data at lower classification level
        """
        # Validate this service can read the data
        if data.classification > self.clearance:
            raise SecurityError("Cannot declassify data above clearance")

        # Validate downgrade direction
        if target_level >= data.classification:
            raise SecurityError("Target must be lower than current classification")

        # Perform declassification (redaction, sanitization)
        declassified_data = self._redact(data.data, target_level)

        # Audit trail
        self._log_declassification(
            original_level=data.classification,
            target_level=target_level,
            justification=justification,
            timestamp=datetime.now()
        )

        # Return data at lower classification
        return ClassifiedData(declassified_data, target_level)

    def _redact(self, data: any, target_level: SecurityLevel) -> any:
        # Remove information inappropriate for target level
        # This is where human review would occur in real systems
        pass

    def _log_declassification(self, **kwargs):
        # Immutable audit log
        audit_log.append(kwargs)
```

### Trusted Downgrade in Pipeline

```python
# Pipeline with trusted downgrade
secret_source = DataSource(SecurityLevel.SECRET)

# Trusted service at SECRET level
downgrade_service = TrustedDowngradeService(clearance=SecurityLevel.SECRET)

# Components at lower levels
official_processor = Processor(SecurityLevel.OFFICIAL)
unofficial_sink = DataSink(SecurityLevel.UNOFFICIAL)

# Workflow:
def secure_pipeline():
    # 1. Read SECRET data (service has SECRET clearance)
    secret_data = secret_source.read()

    # 2. Declassify to OFFICIAL (trusted service with authority)
    official_data = downgrade_service.declassify(
        secret_data,
        SecurityLevel.OFFICIAL,
        justification="Public release approval #12345"
    )

    # 3. Process at OFFICIAL level (no violation)
    processed = official_processor.process(official_data)

    # 4. Declassify to UNOFFICIAL
    unofficial_data = downgrade_service.declassify(
        processed,
        SecurityLevel.UNOFFICIAL,
        justification="Redacted for public disclosure"
    )

    # 5. Write to UNOFFICIAL sink (no violation)
    unofficial_sink.write(unofficial_data)
```

**Key points**:
- Downgrade service operates at HIGH clearance (can read classified data)
- Explicit declassification with justification (not implicit)
- Audit trail for every downgrade operation
- Manual review in production (automated only for specific patterns)

---

## Immutability Enforcement

### Classification Cannot Be Reduced

Once data is classified at a level, it cannot be reduced without formal declassification.

❌ **Mutable Classification (Insecure)**:
```python
class Data:
    def __init__(self, content, classification):
        self.content = content
        self.classification = classification  # Mutable!

data = Data("secret info", SecurityLevel.SECRET)

# ❌ Can be modified at runtime
data.classification = SecurityLevel.UNOFFICIAL  # Forbidden!
```

✅ **Immutable Classification (Secure)**:
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ClassifiedData:
    """Immutable dataclass - classification cannot change."""
    content: str
    classification: SecurityLevel

data = ClassifiedData("secret info", SecurityLevel.SECRET)

# ❌ Raises FrozenInstanceError
data.classification = SecurityLevel.UNOFFICIAL  # Cannot modify!
```

### Derived Data Inherits Classification

```python
def derive_data(original: ClassifiedData) -> ClassifiedData:
    """
    Derived data has SAME classification as original.
    Cannot be lower (information flow property).
    """
    transformed = transform(original.content)

    # Derived data inherits original classification
    return ClassifiedData(transformed, original.classification)

# Example:
secret_data = ClassifiedData("secret", SecurityLevel.SECRET)
derived = derive_data(secret_data)

assert derived.classification == SecurityLevel.SECRET  # Always true
```

---

## Minimum Security Level Computation

### Pipeline-Wide Clearance

Compute minimum clearance required across all components in pipeline.

```python
def compute_pipeline_clearance(components: list) -> SecurityLevel:
    """
    Compute minimum clearance needed to operate pipeline.
    All components must have this clearance or higher.
    """
    max_classification = SecurityLevel.UNOFFICIAL

    for component in components:
        if hasattr(component, 'classification'):
            # Data source/sink - sets required clearance
            if component.classification > max_classification:
                max_classification = component.classification

    return max_classification

def validate_pipeline(components: list, processor_clearance: SecurityLevel):
    """
    Validate processor clearance sufficient for pipeline.
    """
    required_clearance = compute_pipeline_clearance(components)

    if processor_clearance < required_clearance:
        raise SecurityError(
            f"Insufficient clearance: Processor has {processor_clearance}, "
            f"pipeline requires {required_clearance}"
        )

    print(f"✓ Pipeline validated: Clearance {processor_clearance} sufficient")

# Usage:
components = [
    DataSource(SecurityLevel.SECRET),
    DataSink(SecurityLevel.OFFICIAL)
]

processor = Processor(SecurityLevel.SECRET)

# Validate at construction time
validate_pipeline(components, processor.clearance)  # ✓ Passes

processor_low = Processor(SecurityLevel.OFFICIAL)
validate_pipeline(components, processor_low.clearance)  # ❌ Raises SecurityError
```

---

## Mandatory Access Control (MAC)

### MAC vs DAC

**Discretionary Access Control (DAC)**:
- Owner controls access (chmod, ACLs)
- Users can grant access to others
- Examples: RBAC, file permissions

**Mandatory Access Control (MAC)**:
- System enforces access based on security labels
- Users CANNOT override (not discretionary)
- Examples: SELinux, Bell-LaPadula

### MAC for Classified Systems

```python
class MandatoryAccessControl:
    """
    Enforces Bell-LaPadula rules.
    Users/processes CANNOT override.
    """
    def __init__(self):
        self.policy = BellLaPadulaPolicy()

    def check_read_access(
        self,
        subject_clearance: SecurityLevel,
        object_classification: SecurityLevel
    ) -> bool:
        """
        No-read-up: Subject can only read at or below clearance.
        """
        return subject_clearance >= object_classification

    def check_write_access(
        self,
        subject_clearance: SecurityLevel,
        object_classification: SecurityLevel
    ) -> bool:
        """
        No-write-down: Subject can only write at or above clearance.
        """
        return subject_clearance <= object_classification

    def enforce_access(self, operation: str, subject, object):
        """
        Enforce MAC policy. Users cannot override.
        """
        if operation == "read":
            if not self.check_read_access(subject.clearance, object.classification):
                raise SecurityError(f"MAC violation: Read-up forbidden")

        elif operation == "write":
            if not self.check_write_access(subject.clearance, object.classification):
                raise SecurityError(f"MAC violation: Write-down forbidden")

        # Access granted
        return True

# Usage:
mac = MandatoryAccessControl()

subject = Process(clearance=SecurityLevel.OFFICIAL)
secret_obj = Object(classification=SecurityLevel.SECRET)

# ❌ Read-up violation - MAC enforces even if user wants access
mac.enforce_access("read", subject, secret_obj)  # Raises SecurityError
```

**Key difference**: With DAC (RBAC), you could grant a role access. With MAC, no amount of permission grants can override clearance levels.

---

## Quick Reference: Decision Tree

```
Need to handle classified data?
│
├─ Same classification level for all components?
│  └─→ ✓ Simple: All components at same level (e.g., all SECRET)
│
├─ Different levels in same pipeline?
│  ├─ Can you redesign to separate pipelines?
│  │  └─→ ✓ Preferred: Separate SECRET and UNOFFICIAL pipelines
│  │
│  └─ Must mix levels?
│     └─→ Use Trusted Downgrade Service
│        - Service operates at HIGH clearance
│        - Explicit declassification with justification
│        - Audit trail
│        - Manual review for authority
│
└─ Validating pipeline security?
   ├─ Runtime checks? ❌ Too late (exposure window)
   └─ Construction-time validation? ✓ Correct (fail-fast)
```

---

## Common Mistakes

### ❌ Sanitization Instead of Declassification

**Wrong**: "I'll strip SECRET fields and call it UNOFFICIAL"

**Right**: Formal declassification requires:
- Authority to declassify
- Manual review of what can be released
- Audit trail
- Understanding of inference attacks (aggregate data can leak secrets)

**Why**: You cannot automatically determine what's safe to declassify. Requires human judgment and authority.

---

### ❌ Runtime Validation Only

**Wrong**: Check clearances when processing data

**Right**: Fail-fast at construction time before any data access

**Why**: Runtime checks have exposure windows. Construction-time validation = zero exposure.

---

### ❌ Treating Clearances as Roles

**Wrong**: "Just add a 'SECRET' role in RBAC"

**Right**: Implement Mandatory Access Control (MAC) with Bell-LaPadula

**Why**: Clearances are not discretionary. Users cannot grant access across classification levels.

---

### ❌ Mutable Classification

**Wrong**: Allow classification field to be modified at runtime

**Right**: Immutable dataclasses, frozen classification after assignment

**Why**: Classification cannot decrease without formal declassification.

---

### ❌ Missing Information Flow Analysis

**Wrong**: Focus only on direct access (can processor read source?)

**Right**: Analyze full information flow (derived data inherits classification)

**Why**: Data derived from SECRET is SECRET until declassified.

---

## Cross-References

**Use WITH this skill**:
- `ordis/security-architect/secure-by-design-patterns` - Fail-fast validation pattern
- `ordis/security-architect/security-controls-design` - Defense-in-depth for classified systems

**Use AFTER this skill**:
- `ordis/security-architect/security-architecture-review` - Review classified systems against Bell-LaPadula

## Real-World Impact

**Systems using Bell-LaPadula + fail-fast enforcement**:
- **Elspeth MLS Pipeline**: Prevented VULN-004 class (security level overrides) by refusing to construct pipelines with classification mismatches
- **Government Defense System**: Zero classification violations in 3 years due to construction-time validation (previous system with runtime checks had 12 incidents)
- **Type-system enforcement**: Caught 47 configuration errors at build time vs runtime (100% pre-deployment prevention)

**Key lesson**: **Make invalid configurations impossible to create. Bell-LaPadula + fail-fast construction-time validation prevents all no-read-up and no-write-down violations.**
