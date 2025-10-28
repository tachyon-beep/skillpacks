# Real-World Test Scenarios from Elspeth

**Purpose**: Concrete test cases derived from actual architectural evolution
**Use**: Pressure test skills against real scenarios where expertise would have prevented issues

---

## ADR-002 Evolution: The "Cascading Discovery" Case

### Timeline: How Gaps Were Discovered

**Initial Implementation (ADR-002)**: Multi-Level Security Enforcement
- Bell-LaPadula MLS with fail-fast validation
- Minimum security level across pipeline components

**Gap 1 → ADR-003**: Central Plugin Type Registry
- **Discovery**: What if plugin isn't registered at all? MLS checks assume plugin exists.
- **Root cause**: No enforcement that expected plugins are actually registered
- **Solution**: Three-layer defense (nominal typing + registry + test enforcement)

**Gap 2 → ADR-004**: Mandatory BasePlugin Inheritance
- **Discovery**: What if someone bypasses BasePlugin and creates plugin directly?
- **Root cause**: Protocol typing allows duck typing, no nominal type enforcement
- **Solution**: ABC (not Protocol) with sealed security methods

**Gap 3 → ADR-005**: Frozen Plugin Capability
- **Discovery**: What if high-clearance plugin is compromised and starts leaking data?
- **Root cause**: Trusted downgrade assumes high-clearance components are always trustworthy
- **Solution**: Strict level enforcement option (no trusted downgrade)

### What a Skilled Architect Would Have Spotted

**Initial design review with security-architect skills would have revealed:**

1. **Threat Modeling Gap** (`threat-modeling` skill):
   - Threat: Plugin registry manipulation (malicious plugin registration)
   - Threat: Type system bypass (duck-typed plugin bypassing security checks)
   - Threat: Compromised high-clearance component (insider threat/supply chain)
   - **All three ADRs address threats that should have been in original threat model**

2. **Architecture Review Gap** (`architecture-security-review` skill):
   - Review question: "How do you verify plugin is registered?" → Would catch ADR-003
   - Review question: "Can plugin bypass BasePlugin enforcement?" → Would catch ADR-004
   - Review question: "What if trusted component is compromised?" → Would catch ADR-005

3. **Secure-by-Design Gap** (`secure-by-design-patterns` skill):
   - Pattern: Defense-in-depth (not just schema, also registry + tests + nominal typing)
   - Pattern: Fail-secure (frozen capability prevents compromise escalation)
   - Pattern: Minimal TCB (don't trust what you don't have to)

### Test Scenario Design

**RED Phase Pressure Test** (for `architecture-security-review` skill):

```markdown
**Scenario**: "Review this MLS enforcement design before implementation:

Design Overview:
- BasePlugin with security_level property
- Pipeline computes minimum security level
- Components with insufficient clearance reject operation
- High-clearance components can operate at lower levels (trusted downgrade)

We're launching in 2 weeks. Quick review - anything we're missing?"

**Testing Approach**: OPEN-ENDED
- Don't provide checklist of expected questions
- Let agent ask whatever questions their training suggests
- Document what questions they naturally ask (or don't ask)

**Expected Baseline Failures** (without skill):
- Agent approves design: "Looks good, fail-fast enforcement is solid"
- Asks superficial questions or no probing questions
- Doesn't question trust assumptions
- Rationalization: "MLS enforcement handles security, that's sufficient"

**Success Criteria** (with skill):
- Agent asks probing questions about enforcement mechanisms
- Agent questions trust boundaries and assumptions
- Agent explores bypass scenarios naturally
- Questions emerge from thinking patterns, not memorized checklist

**The 3 gaps (ADR-003, 004, 005) should emerge organically from:**
- Defense-in-depth thinking (not just one MLS check)
- Trust boundary analysis (what's trusted vs untrusted)
- Attack surface enumeration (how could this be bypassed)
- Insider threat consideration (what if trusted component compromised)

**Example natural questions a skilled architect might ask:**
- "How do you ensure plugins are actually registered?" (→ ADR-003)
- "What stops someone from making their own plugin that bypasses BasePlugin?" (→ ADR-004)
- "What if a high-clearance component is compromised?" (→ ADR-005)

**Note**: These questions aren't pass/fail checklist - they should emerge naturally from applying architecture review thinking patterns.
```

**This scenario is PERFECT because**:
1. It's real (Elspeth actually discovered these gaps post-implementation)
2. It's measurable (3 specific gaps a skilled architect would catch)
3. It's pressure (2-week deadline encourages "good enough" thinking)
4. It proves skill value (prevents 3 follow-up ADRs and rework)

---

## VULN-004: Configuration Override Attack

### What Happened

**Vulnerability**: Attacker could override `security_level` via configuration YAML
```yaml
plugins:
  datasource:
    type: local_csv
    security_level: SECRET  # ATTACKER INJECTED THIS
    options:
      path: /data/classified.csv
```

**Impact**: UNOFFICIAL plugin could be forced to claim SECRET clearance, bypassing MLS

**Fix Required**: Three-layer defense
1. Schema enforcement (`additionalProperties: false`)
2. Registry runtime rejection (`FORBIDDEN_CONFIG_FIELDS`)
3. Factory post-creation verification (declared vs actual match)

### What a Skilled Architect Would Have Spotted

**Threat Modeling** (`threat-modeling` skill):
- STRIDE analysis of configuration system:
  - **Tampering**: Configuration file modification (attacker-controlled YAML)
  - **Elevation of Privilege**: Override security properties to gain access
- Attack tree: Root goal "Access classified data" → Branch "Override clearance" → Leaf "Inject security_level in config"

**Security Controls Design** (`security-controls-design` skill):
- Defense-in-depth principle: Don't rely on single layer (schema only)
- Fail-secure pattern: Security properties immutable after instantiation
- Separation of duties: Code declares security, config cannot override

### Test Scenario Design

**RED Phase Pressure Test** (for `threat-modeling` skill):

```markdown
**Scenario**: "Quick threat model needed. We have configuration system:

- Plugins configured via YAML
- Config includes type, options
- Schema validates structure
- Plugins declare security_level in code

Launch tomorrow. What threats should we model?"

**Expected Baseline Failures** (without skill):
- Agent focuses on: File access threats, schema validation bypass
- Misses: Configuration override of security properties
- Misses: Immutability requirements for security-critical fields
- Rationalization: "Schema validation handles config threats"

**Success Criteria** (with skill):
- Agent applies STRIDE: Identifies Tampering and Elevation of Privilege
- Agent builds attack tree: Config modification → Security property override
- Agent identifies: Need for immutable security properties
- Agent recommends: Defense-in-depth (schema + runtime + verification)
```

---

## VULN-009: SecureDataFrame Immutability Bypass

### What Happened

**Vulnerability**: `SecureDataFrame` could be mutated via `__dict__` access
```python
df = SecureDataFrame(data=..., security_level=SecurityLevel.SECRET)
df.__dict__['security_level'] = SecurityLevel.UNOFFICIAL  # BYPASS!
```

**Impact**: Classification downgrade violated Bell-LaPadula "no write down"

**Fix Required**: Frozen dataclass + runtime check

### What a Skilled Architect Would Have Spotted

**Architecture Review** (`architecture-security-review` skill):
- Review question: "How is immutability enforced? Python allows __dict__ access."
- Review question: "Can attacker bypass frozen dataclass? Test with __dict__?"
- Red flag: Relying on language feature (frozen) without testing bypass

**Secure-by-Design Patterns** (`secure-by-design-patterns` skill):
- Immutable infrastructure pattern: No runtime modifications to security state
- TCB minimization: SecureDataFrame is in trusted computing base, must be verified
- Fail-secure: Even if code tries to mutate, system refuses

### Test Scenario Design

**RED Phase Pressure Test** (for `architecture-security-review` skill):

```markdown
**Scenario**: "Review this SecureDataFrame implementation:

```python
@dataclass(frozen=True)
class SecureDataFrame:
    data: pd.DataFrame
    security_level: SecurityLevel
```

This prevents modification because frozen=True. Security review?"

**Expected Baseline Failures** (without skill):
- Agent approves: "Frozen dataclass prevents mutation, looks secure"
- Misses: __dict__ bypass possible in Python
- Misses: Need for runtime verification
- Rationalization: "Language feature provides security"

**Success Criteria** (with skill):
- Agent asks: "How does frozen=True work? Can it be bypassed?"
- Agent tests: "What about __dict__ access or __setattr__?"
- Agent identifies: Need for runtime verification, not just language feature
- Agent recommends: Test all bypass attempts, add assertions
```

---

## Documentation Evolution: Scattered Narratives → 14 ADRs

### What Happened

**Early documentation** (pre-ADR practice):
- Long README sections explaining decisions
- Code comments with rationale
- Scattered across multiple files
- No traceability

**Evolution**:
- Realized: Can't find "why we chose X" when reviewing decision
- Solution: ADR format (Context → Decision → Consequences)
- Result: 14 ADRs, clear decision trail

### What a Skilled Technical Writer Would Have Spotted

**Documentation Structure** (`documentation-structure` skill):
- Pattern recognition: Architecture decisions need ADR format
- Symptoms: "We made this decision but I can't remember why"
- Solution: ADR template with clear structure

**Documentation Testing** (`documentation-testing` skill):
- Findability test: "Can engineer find rationale for decision X?"
- Completeness test: "Does doc answer why/what/consequences?"
- Test would fail: Scattered narratives fail findability

### Test Scenario Design

**RED Phase Pressure Test** (for `documentation-structure` skill):

```markdown
**Scenario**: "We decided to use BasePlugin ABC instead of Protocol. Document this decision. We already have code comments explaining it."

**Expected Baseline Failures** (without skill):
- Agent writes: Long README section or expanded code comments
- Misses: ADR format for architecture decision
- Misses: Traceability (status, supersedes, related decisions)
- Rationalization: "Code comments are sufficient documentation"

**Success Criteria** (with skill):
- Agent recognizes: Architecture decision → ADR format
- Agent creates: ADR-004 with Context, Decision, Consequences
- Agent includes: Status (Accepted), related ADRs (002, 003)
- Agent ensures: Findable in docs/architecture/decisions/
```

---

## Test Scenario Summary Table

| Skill | Real Elspeth Case | What Skilled Person Would Catch | Test Scenario |
|-------|------------------|----------------------------------|---------------|
| `threat-modeling` | VULN-004 (config override) | Config tampering threat, elevation of privilege via security property override | "Threat model config system with security properties" under time pressure |
| `architecture-security-review` | ADR-003, 004, 005 gaps | Plugin registration enforcement, type system bypass, compromised component threat | "Review MLS design" before implementation with 3 specific gaps to catch |
| `architecture-security-review` | VULN-009 (immutability bypass) | __dict__ bypass of frozen dataclass, need for runtime verification | "Review SecureDataFrame" with frozen=True, test bypass attempts |
| `security-controls-design` | VULN-004 fix | Defense-in-depth (3 layers), fail-secure pattern, separation of duties | "Design config security" with single-layer validation to improve |
| `secure-by-design-patterns` | ADR-002 → 005 evolution | Defense-in-depth from start, TCB minimization, fail-secure throughout | "Design MLS system" with defense-in-depth requirements |
| `documentation-structure` | README narratives → 14 ADRs | Architecture decision needs ADR format, traceability requirements | "Document BasePlugin decision" with scattered docs to consolidate |

---

## Using These Scenarios

### For Testing Skills (RED-GREEN-REFACTOR)

1. **RED Phase**: Run scenario WITHOUT skill, document baseline failures
   - Use real rationalizations from Elspeth development ("schema is sufficient", "frozen dataclass prevents mutation")
   - Measure specific gaps (3 missing ADR insights, config override missed, etc.)

2. **GREEN Phase**: Write skill addressing baseline failures
   - Teach specific patterns that would have caught issues
   - Include real examples from Elspeth (with permission to share)

3. **REFACTOR Phase**: Add pressure and edge cases
   - Time pressure ("launch tomorrow")
   - Authority pressure ("security team already approved")
   - Sunk cost ("we already implemented this")

### For Validating Skill Quality

**Success criterion**: If you run scenario with skill loaded, agent catches the same gaps that required follow-up ADRs.

**Quality metric**: Skilled agent should identify:
- ADR-002 → 005 evolution: All 3 gaps (registry, typing, frozen capability)
- VULN-004: Configuration override threat
- VULN-009: Immutability bypass

**If skill doesn't catch these real issues, it's not teaching the right thing.**

---

## Meta-Lesson: Cascading Discovery Pattern

**Pattern recognized**: Initial design seems solid → Implementation reveals gap 1 → Fix gap 1 reveals gap 2 → Fix gap 2 reveals gap 3

**Why it happens**:
- Single-layer thinking (one security check feels sufficient)
- Trust in language features (frozen=True, Protocol typing)
- Missing threat modeling upfront
- No architecture review before implementation

**What skills prevent**:
- `threat-modeling`: Identifies threats before implementation
- `architecture-security-review`: Catches gaps during design
- `secure-by-design-patterns`: Designs defense-in-depth from start
- `security-controls-design`: Layers multiple defenses

**Test for this**: Give agent "seemingly solid" design, see if they catch the cascading gaps before implementation.

---

## Gotcha: Don't Over-Learn from Elspeth

**Caution**: These scenarios are Elspeth-specific but skills must be universal.

**Right approach**:
- ✅ Use Elspeth scenarios to TEST skills
- ✅ Teach universal patterns (defense-in-depth, threat modeling, ADR format)
- ❌ Don't make skills Elspeth-specific
- ❌ Don't assume all users have MLS requirements

**Example**:
- ❌ Bad: "Always use Bell-LaPadula MLS"
- ✅ Good: "For classified systems, consider Bell-LaPadula MLS (see classified-systems-security)"

**Test generalizability**: Can skill apply to web app, mobile app, data pipeline? If only applies to security platforms like Elspeth → Too specific.

---

## Contributor Note

**When implementing these test scenarios:**

1. **Anonymize if needed**: Remove any sensitive Elspeth details
2. **Generalize the pattern**: "MLS enforcement design" not "Elspeth's pipeline"
3. **Test transferability**: Would this scenario work for generic project?
4. **Keep the gaps**: The 3 ADR gaps are universal (registration, typing, trust)

**These scenarios are gold** because they're real, measurable, and prove skill value. Use them wisely.

---

**Next Action**: Use ADR-002 → 005 evolution as PRIMARY test case for `architecture-security-review` skill in Phase 1.
