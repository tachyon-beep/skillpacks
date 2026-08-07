---
name: definition-lifecycle
description: Use when a named, versioned artefact governs behaviour across a boundary — a schema version, a policy record, an evaluation rubric, a grammar or profile in a registry — and someone is about to edit it in place, fix a typo in it, promote it to production, delete an old version, reference it as "latest", or ship it before anyone approved it. Also use when the same definition name means different things in two environments, or when an audit cannot say which content was in force when a past record was produced. Covers draft/approved/locked semantics, append-only transition events, content-hash drift detection, consumer pinning, and supersession instead of edit.
---

# Definition Lifecycle

**A named definition — a schema version, a policy record, a rubric, a profile — moves draft → approved → locked through recorded events, and is never silently edited. A definition whose content can change without an event is not a definition; it is a variable that other systems mistakenly treat as a constant.**

## When this earns its cost

Read this sheet when:

- Anything in your system is referenced by name and version — `schema v3`, `pricing-policy 2026-Q1`, `grading-rubric v2`, `profile: strict` — and other records cite that reference as if it fixed a meaning.
- A definition needs a fix and the obvious move is to edit the file and redeploy.
- Someone asks "what did rubric v2 actually say when this record was graded?" and the honest answer is "whatever was in the repo at the time, probably".
- A definition is loaded in production from a path that a human can write to, or referenced as `latest`.
- You are promoting a new schema version from development into production and there is no event anywhere that says who decided it was ready.

The cost is real: three states, an event log, a hash check in CI, and a registry that refuses illegal transitions. What it buys is the ability to answer, years later and from stored records alone, *what was in force* — which is the precondition for every audit, replay, and after-the-fact dispute.

## 1. Three states, and why two of them are frozen

```text
DRAFT ──approve──▶ APPROVED ──lock──▶ LOCKED ──supersede──▶ (new version, DRAFT)
```

**Transitions only move forward. There is no unlock, no un-approve, no "back to draft" — you supersede.**

- **DRAFT** — mutable and iterable. Content may change freely; that is the entire point of the state. Production consumers **refuse** draft definitions, and that refusal is a loader-side check, not a review comment (§4). Nothing durable may cite a draft, because its content is not yet a fact about anything.
- **APPROVED** — usable in production, and **content-frozen**: any content change is a *new version*, which starts back in DRAFT. Approval is an authority event, not a file move — someone with the right to say "this is what we mean" said it, and that is recorded.
- **LOCKED** — immutable forever, because history now references it. Decisions were made under it, records bind it, archived outputs are only interpretable through it.

The question every reader asks: *if APPROVED is already frozen, why does LOCKED exist?* Because frozen and permanent are different guarantees. An APPROVED definition can still be **withdrawn or retired** — nothing binds it yet; if it was approved on Tuesday and nothing has been produced under it, you can retire it and the world loses nothing. A LOCKED definition cannot be withdrawn at any price: withdrawing it would orphan every record that cites it, retroactively making history uninterpretable. Lock is the state that says *this definition has acquired dependents*.

The practical trigger for locking is therefore not "we're confident in it" but **"something durable now references it"** — the first production record produced under it, the first decision graded by it. In most systems that is the first production use, which makes locking a mechanical consequence of use rather than a judgement call:

```python
def on_first_production_use(registry: "DefinitionRegistry", ref: "DefinitionRef") -> None:
    """Locking is caused by dependents appearing, not by anyone feeling ready."""
    if registry.state(ref) is State.APPROVED:
        registry.lock(ref, actor="system:first-binding-record", authority="automatic")
```

Locked definitions never die; they are **superseded**. A superseding version is a new (name, version) that carries a `supersedes` pointer. The old one stays exactly as it was, forever readable, so that a record produced under it still means what it meant.

## 2. Transitions are recorded events, not a mutable field

**The lifecycle state is derived from an append-only event log. A state you can assign is a state you can assign wrongly, silently, and without an author.**

Each event records: who, when, from-state → to-state, the canonical content hash at the moment of transition, and the approval authority under which it happened. Corrections create *new* records; the log is never rewritten.

```python
from dataclasses import dataclass, field
from enum import Enum

class State(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    LOCKED = "locked"

LEGAL = {                                  # closed transition table; nothing else exists
    (State.DRAFT, State.APPROVED),
    (State.APPROVED, State.LOCKED),
}

@dataclass(frozen=True)
class DefinitionRef:
    name: str
    version: int

@dataclass(frozen=True)
class Definition:
    ref: DefinitionRef
    content_hash: str                      # canonical hash; see canonical-identity.md
    supersedes: DefinitionRef | None = None

@dataclass(frozen=True)
class LifecycleEvent:
    ref: DefinitionRef
    from_state: State | None               # None only for the creation event
    to_state: State
    content_hash: str                      # what the content was AT the transition
    actor: str                             # who
    authority: str                         # under what right (closed vocabulary)
    at: str                                # recorded timestamp, supplied by the caller

class LifecycleViolation(Exception):
    """Raised on an illegal transition, an unknown ref, or a content-hash mismatch."""

class DefinitionRegistry:
    def __init__(self) -> None:
        self._defs: dict[DefinitionRef, Definition] = {}
        self._events: list[LifecycleEvent] = []          # append-only; never rewritten

    # --- derivation, not storage -------------------------------------------
    def state(self, ref: DefinitionRef) -> State:
        events = [e for e in self._events if e.ref == ref]
        if not events:
            raise LifecycleViolation(f"unknown definition {ref}")
        return events[-1].to_state

    def _transition(self, ref, to_state, actor, authority, at, expect_hash) -> None:
        current = self.state(ref)
        if (current, to_state) not in LEGAL:
            raise LifecycleViolation(f"{ref}: {current} -> {to_state} is not a transition")
        stored = self._defs[ref].content_hash
        if stored != expect_hash:            # the approver approved *specific bytes*
            raise LifecycleViolation(f"{ref}: content changed since it was reviewed")
        self._events.append(LifecycleEvent(ref, current, to_state, stored,
                                           actor, authority, at))

    def approve(self, ref, actor, authority, at, expect_hash) -> None:
        self._transition(ref, State.APPROVED, actor, authority, at, expect_hash)

    def lock(self, ref, actor, authority, at) -> None:
        self._transition(ref, State.LOCKED, actor, authority, at,
                         expect_hash=self._defs[ref].content_hash)

    def supersede(self, old: DefinitionRef, new_content_hash: str,
                  actor: str, authority: str, at: str) -> DefinitionRef:
        """Supersession creates a NEW draft. It does not touch `old` — locked is locked."""
        if self.state(old) is State.DRAFT:
            raise LifecycleViolation("a draft is edited, not superseded")
        new = DefinitionRef(old.name, old.version + 1)
        self._defs[new] = Definition(new, new_content_hash, supersedes=old)
        self._events.append(LifecycleEvent(new, None, State.DRAFT, new_content_hash,
                                           actor, authority, at))
        return new                            # starts in DRAFT; must be approved again
```

Three things this shape buys, each load-bearing:

- `state()` is a **fold over events**, so there is no field anyone can set. A state with no event that produced it cannot be represented.
- `approve()` takes `expect_hash` — the approver approved *specific content*, and if the file moved underneath the review, approval fails rather than silently blessing different bytes.
- `supersede()` never mutates the old definition. It writes a new one with a `supersedes` pointer. "Marking the old one superseded" by assigning to its state field would contradict the entire meaning of LOCKED; the superseded-ness of v2 is a *fact recorded on v3*, discoverable by following pointers, not a mutation of v2.

## 3. Content-addressing makes tampering detectable

**A frozen state that nothing verifies is a comment. The hash is what converts "someone edited an approved definition in place" from an invisible event into a build failure.**

Every APPROVED or LOCKED definition carries its canonical content hash (`canonical-identity.md` owns what "canonical" means — the hash is over the canonical form, never the raw file bytes, so reformatting is not drift and reordering is not a new definition). CI recomputes every one of them against the registry:

```python
def check_no_drift(registry: DefinitionRegistry, loader) -> list[str]:
    """CI gate. Any content change to a non-draft definition fails the build."""
    failures = []
    for ref, definition in registry.all():
        if registry.state(ref) is State.DRAFT:
            continue                                   # drafts are meant to move
        actual = canonical_hash(loader.load(ref))      # canonical-identity.md
        if actual != definition.content_hash:
            failures.append(
                f"{ref.name} v{ref.version} is {registry.state(ref).value} but its "
                f"content changed: registry {definition.content_hash[:12]} != "
                f"file {actual[:12]}. Frozen definitions are superseded, not edited."
            )
    return failures
```

The message matters as much as the check. A bare hash mismatch invites the fix "update the hash"; naming the discipline points at the correct fix, which is a new version. If your registry treats git as its substrate, this check is what makes that legitimate — see the anti-patterns below.

## 4. Consumers pin what they use

**Production code references definitions by `(name, version)` or by content hash. Never by "latest", and never from a path a human can edit.**

```python
# WRONG — three separate defects in four lines:
rubric = load_yaml("/etc/app/rubrics/latest.yaml")   # editable path + "latest"
schema = registry.get("record-schema")               # unpinned: whatever is newest
apply(rubric, candidate)                             # no state check: a draft may grade

# RIGHT — pinned, state-checked, and the binding is recorded on the output:
ref = DefinitionRef("grading-rubric", 2)             # pinned by version
rubric = registry.load_for_production(ref)           # raises on DRAFT (see below)
result = apply(rubric, candidate)
record = Result(outcome=result,
                rubric_ref=ref,                      # the record carries what governed it
                rubric_hash=registry.hash(ref))      # ...and the exact content
```

```python
def load_for_production(self, ref: DefinitionRef) -> object:
    st = self.state(ref)
    if st is State.DRAFT:
        raise LifecycleViolation(
            f"{ref} is DRAFT — drafts are refused in production; approve it first")
    content = self._loader.load(ref)
    if canonical_hash(content) != self._defs[ref].content_hash:
        raise LifecycleViolation(f"{ref} content does not match its approved hash")
    return content
```

Note that this is a *loader-level* refusal. "Don't ship drafts" as a review convention is enforced by attention, which fails on the day attention is scarce; as a load-time check it fails closed at exactly the moment someone tries.

The producer-side symmetry with `schema-versioning-and-evolution.md` is exact and worth naming: **a schema version is itself a definition.** The reader's version gate refuses versions it does not support; the producer must equally refuse to *emit* under a version still in DRAFT. Both are the same fail-closed rule — nobody acts under a meaning that has not been declared — applied at the two ends of the boundary.

Every derived record carries the versions of everything that governed it: schema version, policy version (`versioned-policy-parameters.md`), rubric or profile version, resolver version (`deterministic-resolution.md`). That is what makes an archived record interpretable under *its own* definitions rather than today's.

## 5. The edit-in-place temptations

Why this defect survives review: the diff is one line and looks obviously correct. Nothing in the diff shows the four thousand decisions it retroactively reinterprets. The reviewer sees a typo fix; the system experiences a silent change in what its history meant.

**"It's just a typo."** A typo in a locked rubric that graded 4,000 decisions is *part of those decisions*. They were made under that text, correctly or not, and rewriting the text rewrites the record of what was applied. Fix forward: a new version with the correction, plus — if the typo materially misled — a recorded **erratum** event that annotates the locked definition without altering it. The erratum is discoverable by anyone reading the old version; the old bytes stay exactly as the decisions saw them.

**"We'll lock it after launch."** Launch is precisely when history starts accruing against it. Every hour of production before the lock is a window in which a definition can change under the records that cite it, and no later lock can retroactively tell you whether it did. Approve before production, lock on first binding record.

**"Delete the old versions to reduce confusion."** Old versions are not clutter; they are the interpretation key for every archived record produced under them. Deleting v1 does not remove v1 from history — it removes your ability to *read* the history. The confusion the deletion is meant to cure is properly cured by making the current version obvious (pointers, a resolved alias) rather than by making the old ones unreadable.

## Anti-patterns

| Anti-pattern | Why it fails |
|---|---|
| Lifecycle state as a mutable enum column with no event log | The state can be set to anything by anyone, with no author, time, or content attached. `state = APPROVED` records that someone approved *nothing in particular*. |
| Git history as the event log, with no approval semantics | A merge is not an approval event unless the process names who may merge and records it as such. Git **can** be the substrate — commits are content-addressed and append-only — but only if the registry maps specific commits to signed transition events with actor and authority. Absent that mapping, "it's in git" means "someone had write access". |
| Definitions loaded from an editable config path in production | Any content-hash guarantee ends at the filesystem boundary. Load from the registry, verify the hash, refuse on mismatch. |
| `latest` as a version reference | "Latest" resolves differently in every environment and at every moment; two hosts running the same build can disagree about what governed a decision, and the record cannot say which was right. |
| Approving a definition without pinning the content reviewed | The approval attaches to a name, so any edit between review and approval ships unreviewed. Approve a hash. |
| Superseding by mutating the old definition's state field | Contradicts LOCKED. Supersession is a relation recorded on the new version, not a change to the old one. |
| An erratum applied by editing the locked text | An erratum that edits is just an edit. It annotates; it never alters. |

## Rationalizations

| Rationalization | Reality |
|---|---|
| "It's a one-character typo in a comment" | Then it costs one new version to fix correctly. If the change genuinely cannot alter meaning, the canonical hash is unchanged and the check never fires — that is exactly what canonicalisation is for. If the hash *does* change, the content changed. |
| "Nobody has consumed v2 yet, so editing it is safe" | If nothing binds it, it should not be LOCKED, and if it is only APPROVED, the correct move is retire-and-supersede — which takes the same two minutes and leaves a record of why v2 was withdrawn. |
| "The approval is in the PR thread" | A thread is not a queryable event with an authority field. In two years the reviewer has left, the thread is paginated behind a service you no longer pay for, and the auditor's question is still "who approved this content". |
| "We version definitions in git, that's the same thing" | Git gives you content-addressing and append-only history — half the mechanism. It does not give you approval authority, a state model, or refusal of drafts in production. Map commits to transition events and you have both. |
| "Locking will slow us down; we iterate fast" | Locking constrains only definitions with durable dependents. If you iterate fast on things nothing references, they are drafts and lock never applies. If they *do* have dependents, the speed you are defending is the speed of changing history. |
| "We keep the old versions in a `deprecated/` folder" | Where they are not loadable, not hash-checked, and not resolvable by the references in archived records. Retired is a lifecycle fact; a folder is not a lifecycle. |
| "Everyone knows the rubric changed in March" | Everyone currently on the team, until they leave. The record either says which version graded it or it does not, and no amount of shared memory closes that gap for an auditor. |

## Red flags

- A definition table with a `status` column and no accompanying events table.
- Any code path that writes to a definition row that is not DRAFT.
- `latest`, `current`, `HEAD`, or an environment-dependent alias appearing in a production reference to a definition.
- Approval represented by a file move, a folder name, a branch merge, or an unrecorded chat message.
- A CI suite with no hash-drift check over approved and locked definitions.
- A record produced under a rubric, policy, or schema that does not carry that definition's version.
- A `supersede`, `retire`, or `errata` function whose implementation mutates the superseded object.
- Old versions absent from the registry — deleted, archived out of reach, or present only as files nothing can resolve.

## Quick reference

| Situation | Correct action |
|---|---|
| Iterating on new content | Keep it DRAFT; production loaders refuse it |
| Ready for production use | `approve()` — records actor, authority, and the exact content hash reviewed |
| First durable record references it | `lock()` — immutable from here; triggered by dependents appearing, not by confidence |
| Content must change after approval | New version, DRAFT, approved on its own merits; `supersedes` pointer to the old |
| Approved but nothing binds it yet, and it is wrong | Retire and supersede — permitted precisely because nothing depends on it |
| Locked and wrong | Fix forward with a new version; add a recorded erratum event if the defect misled |
| Reading an archived record | Load the definition version the record cites, not the current one |
| Production reference | `(name, version)` or content hash, loaded through the registry, state-checked |
| Someone edited an approved definition | CI hash-drift check fails the build and names the correct fix |
| "Can we unlock it?" | No such transition exists. Supersede. |

## Cross-references

- `schema-versioning-and-evolution.md` — a schema version is a definition; it moves draft → approved → locked before any producer emits under it, and the producer-side draft refusal mirrors the reader-side version gate.
- `canonical-identity.md` — the canonical form and hash that content-addressing here depends on; reformatting is not drift.
- `versioned-policy-parameters.md` — policy records as a definition class; binding the version in force onto every record produced under it.
- `deterministic-resolution.md` — resolvers take definitions as recorded inputs and stamp their versions on outputs; replay requires the definition to be recoverable exactly.
- `contract-testing.md` — the CI hash-drift gate, fixtures pinned to definition versions, and tests that a draft cannot be loaded in production.
- `silent-default-elimination.md` — the same fail-closed instinct at field level; an unpinned or defaulted definition reference is the silent-default defect applied to meaning itself.
- `blinding-by-construction.md` — closed vocabularies for the `authority` field; free text in a transition event is a covert channel through the approval record.
- Cross-pack: `axiom-audit-pipelines` — append-only event logs, hash chaining, and integrity verification at whole-system scale; this sheet is the definition-registry special case.
