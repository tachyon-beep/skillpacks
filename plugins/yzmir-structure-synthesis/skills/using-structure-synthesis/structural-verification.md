---
name: structural-verification
description: "Use when building or reviewing the legality gate a generated candidate must pass before compilation - shape inference, cycle/reachability checks, interface-contract checks, gradient-flow and trainability-mask validation, identity-at-birth/zero-influence proofs, and forbidden-operation detection - and when that gate needs to consume zero utility signal."
---

# Structural Verification

## When to Use

- Building the legality gate between a generator and whatever compiles or executes its output
- Adding a new check to an existing verifier
- Debugging "the verifier passed this candidate but it broke downstream" (usually a missing check, not a broken one)
- Auditing whether a verifier has started consuming reward, predicted utility, or generator identity in its accept/reject decision

**Pipeline order is fixed across this pack, and it is load-bearing**: cheap legality checks (shape/type inference, acyclicity, interface-contract arity) → **canonicalisation** (`canonicalisation-and-normal-forms.md`) → **the full gate described in this sheet, including reachability and dead-node detection, run on the canonical form**. The cheap checks come first because canonicalising a shape-broken or cyclic candidate is wasted work — and, for acyclicity specifically, because **the canonicaliser does not check it**: `nx.ancestors` and signature refinement both run happily on a cyclic graph, so a cyclic candidate that reaches canonicalisation gets a plausible-looking canonical form and a hash instead of a rejection. Acyclicity is this gate's obligation, discharged before canonicalisation, not something the canonicaliser will catch on your behalf. Reachability comes *after*, because eliminating unreachable nodes is canonicalisation's job, not the verifier's: run it too early and a legitimately-generated dangling node fails a check that canonicalisation was about to make moot. The identity a cleared candidate carries (`equivalence-detection-and-semantic-hashing.md`) is computed once, on that same canonical form, so "this candidate already passed the gate" means the same thing everywhere.

For what happens before any of this (constrained decoding narrowing the space), see `validity-by-construction-vs-post-hoc.md`.

## Core Principle

**Structural verification answers exactly one question — is this candidate legal — and every check in this sheet is a rule, not a preference. The moment a check starts asking "is this candidate any good," it has stopped being verification.**

This is the sharpest boundary in the whole pack. It is also the one most likely to erode gradually: a verifier that starts by rejecting shape-mismatched candidates, and a release cycle later is quietly down-ranking "unpromising-looking" ones, did not make one big mistake — it made a dozen small ones, each individually defensible ("we're just adding a soft penalty," "we're just deprioritizing," "we're just being efficient about compute"). The fix is a bright line, checked at review time: **a legality check has a proof obligation and a binary verdict. A utility judgement has neither** — it produces a score, a rank, or a probability, and it is never allowed inside this gate.

## The Checklist

A structural verifier that only checks one or two of these is a partial gate, not a legality proof. Each check catches a failure mode the others miss:

| Check | Catches |
|---|---|
| **Forbidden-operation detection** | An operator outside the grammar's whitelist (see `typed-graph-grammars.md`) |
| **Shape inference and alignment** | Tensor shape mismatches across an edge; a candidate that would crash at first forward pass |
| **Cycle and reachability checks** | Cycles in what must be a DAG (checked cheaply *before* canonicalisation, which needs one); dead code that survived canonicalisation (checked *after*, on the canonical form — before it, a dangling node is canonicalisation's problem, not a rejection) |
| **Interface-contract checks** | Input/output arity and shape that doesn't match the declared insertion contract (see `conditioning-on-context-and-contracts.md`) |
| **Gradient-flow / trainability-mask validation** | A declared-trainable parameter with no path to the loss, or a frozen parameter accidentally left trainable |
| **Identity-at-birth / zero-influence proof** | A residual or insertion structure that does not actually compute the identity function at its declared birth parameters, whatever its structure suggests |
| **Static parameter/memory accounting** | A candidate that is legal in isolation but blows the declared budget (see `typed-graph-grammars.md` for ceilings) |

Every one of these is checkable by rule. None of them requires running the candidate against real data or a reward signal.

## The RED Scenario: Structural Pattern-Matching Is Not Proof

The most dangerous verification bug is not a missing check — it's a check that looks complete but only pattern-matches structure instead of proving the property it claims to verify. Zero-influence is the clearest example, because the failure is invisible without a numeric check:

```python
def verify_zero_influence_WRONG(g):
    """Checks a scale_mul node EXISTS feeding the residual_add.
    Presence of the pattern is not proof the scale is actually zero."""
    for n in g.nodes:
        if g.nodes[n]["op"] == "scale_mul":
            succs = list(g.successors(n))
            if succs and g.nodes[succs[0]]["op"] == "residual_add":
                return True   # BUG
    return False
```

This passes any candidate with the right *shape* of birth gating — a `scale_mul` feeding a `residual_add` — regardless of what value the scale parameter was actually initialized to. A birth-parameter bug (the scale initialized to `1.0` instead of `0.0`, a common off-by-default when a config default silently overrides a per-candidate override) sails through: the graph has the identity-at-birth *pattern*, so the structural check says yes, and the candidate enters training already contributing full, unverified influence to the host — the exact thing zero-influence was supposed to prevent.

### The GREEN Fix: Prove It Numerically

Instantiate the candidate at its declared birth parameters and check that it actually computes the identity function on real tensors:

```python
import torch

def verify_zero_influence(module_factory, dim, birth_params, n_trials=8):
    """module_factory(dim, **birth_params) -> nn.Module implementing the candidate.
    Proof, not pattern match: run real tensors through the real module."""
    module = module_factory(dim, **birth_params)
    for _ in range(n_trials):
        h = torch.randn(4, dim)
        out = module(h)
        if not torch.allclose(out, h, atol=1e-6):
            return False
    return True
```

The difference is not stylistic. The structural check answers "does this graph contain a pattern that could implement zero influence." The numeric check answers "does this graph, at these exact parameters, implement zero influence." Only the second is a proof.

## Executable Decision Procedure

Shape inference, cycle/reachability, forbidden-op detection, and the numeric zero-influence proof, all runnable end to end:

```python
import torch
import torch.nn as nn
import networkx as nx

OP_SHAPE_RULES = {
    "linear": lambda in_shape, params: (params["out_features"],),
    "relu": lambda in_shape, params: in_shape,
    "identity": lambda in_shape, params: in_shape,
    "scale_mul": lambda in_shape, params: in_shape,
    "residual_add": lambda in_shape, params: in_shape,
}
WHITELIST = set(OP_SHAPE_RULES)

def infer_shapes(g: nx.DiGraph, input_shapes: dict) -> dict:
    """Propagate shapes topologically; raise on mismatch or forbidden op."""
    shapes = dict(input_shapes)
    for n in nx.topological_sort(g):
        if n in shapes:
            continue
        op = g.nodes[n]["op"]
        if op not in WHITELIST:
            raise ValueError(f"forbidden operation: {op} at node {n}")
        preds = list(g.predecessors(n))
        in_shapes = [shapes[p] for p in preds]
        if op == "residual_add":
            assert len(set(in_shapes)) == 1, f"residual_add shape mismatch at {n}: {in_shapes}"
            shapes[n] = in_shapes[0]
        else:
            assert len(in_shapes) == 1, f"{op} expects one input, got {len(in_shapes)} at {n}"
            shapes[n] = OP_SHAPE_RULES[op](in_shapes[0], g.nodes[n].get("params", {}))
    return shapes

def check_dag_and_reachability(g: nx.DiGraph, outputs) -> None:
    assert nx.is_directed_acyclic_graph(g), "candidate graph contains a cycle"
    for n in g.nodes:
        reaches_output = any(n == o or nx.has_path(g, n, o) for o in outputs)
        assert reaches_output, f"node {n} does not reach any declared output"

class GeneratedBlock(nn.Module):
    """Interprets a small typed graph as a torch module for numeric verification.
    A sketch: real systems compile via a separate lowering stage (out of scope
    for this pack), but the verifier needs *some* executable form to prove
    numeric properties against, and interpretation is the cheapest one."""

    def __init__(self, g: nx.DiGraph, dim: int, scale_birth_value: float):
        super().__init__()
        self.g = g
        self.order = list(nx.topological_sort(g))
        self.linear = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.tensor(scale_birth_value))

    def forward(self, h):
        vals = {}
        for n in self.order:
            op = self.g.nodes[n]["op"]
            preds = list(self.g.predecessors(n))
            if op == "identity" and not preds:
                vals[n] = h
            elif op == "linear":
                vals[n] = self.linear(vals[preds[0]])
            elif op == "relu":
                vals[n] = torch.relu(vals[preds[0]])
            elif op == "scale_mul":
                vals[n] = vals[preds[0]] * self.scale
            elif op == "residual_add":
                vals[n] = sum(vals[p] for p in preds)
        return vals[self.order[-1]]

def verify_zero_influence(g, dim, scale_birth_value, n_trials=8) -> bool:
    module = GeneratedBlock(g, dim, scale_birth_value)
    for _ in range(n_trials):
        h = torch.randn(4, dim)
        if not torch.allclose(module(h), h, atol=1e-6):
            return False
    return True


def make_residual_graph():
    g = nx.DiGraph()
    for n, op in [("in", "identity"), ("lin", "linear"), ("act", "relu"),
                  ("scale", "scale_mul"), ("out", "residual_add")]:
        g.add_node(n, op=op)
    g.add_edge("in", "lin"); g.add_edge("lin", "act"); g.add_edge("act", "scale")
    g.add_edge("in", "out"); g.add_edge("scale", "out")
    g.nodes["lin"]["params"] = {"out_features": 8}
    return g


def test_full_legal_candidate_passes():
    g = make_residual_graph()
    shapes = infer_shapes(g, {"in": (8,)})
    assert shapes["out"] == (8,)
    check_dag_and_reachability(g, outputs={"out"})
    assert verify_zero_influence(g, dim=8, scale_birth_value=0.0) is True

def test_forbidden_op_rejected():
    g = make_residual_graph()
    g.nodes["act"]["op"] = "eval_host_gradient"  # not in whitelist
    try:
        infer_shapes(g, {"in": (8,)})
        assert False, "should have raised"
    except ValueError as e:
        assert "forbidden operation" in str(e)

def test_birth_bug_caught_only_by_numeric_check():
    g = make_residual_graph()
    buggy_birth_value = 1.0  # should have been 0.0
    assert verify_zero_influence(g, dim=8, scale_birth_value=buggy_birth_value) is False

test_full_legal_candidate_passes()
test_forbidden_op_rejected()
test_birth_bug_caught_only_by_numeric_check()
print("structural verification checks verified")
```

## Trainability Masks: Prove Connectivity, Never Assert Magnitude

The trainability-mask check from the table deserves its own treatment, because it has both a structural-pattern-match trap (like zero-influence above) and a subtler over-correction trap on the other side.

A trainability mask declares which of a candidate's parameters are supposed to learn. The mask can disagree with reality in two directions: a parameter declared trainable that has no path to the output (a wiring bug — a config typo, a refactor leftover, an interpreter that maps the mask entry to nothing), or a parameter declared frozen that the autograd graph will happily update. Neither is caught by reading the graph's *structure* — the proof is a real backward pass:

```python
def check_trainability_mask(module, declared_trainable: set, dim: int) -> list:
    """Connectivity proof: run a real backward pass and confirm the autograd
    graph agrees with the declared mask."""
    module.zero_grad()
    h = torch.randn(4, dim)
    module(h).sum().backward()
    findings = []
    for name, p in module.named_parameters():
        declared = name in declared_trainable
        reachable = p.requires_grad and p.grad is not None
        if declared and not reachable:
            findings.append(f"'{name}' declared trainable but unreachable from the output")
        if not declared and reachable:
            findings.append(f"'{name}' reachable and requires_grad but declared frozen")
    return findings


class BlockWithWiringBug(nn.Module):
    """'gate' is declared trainable in the mask, but forward() never uses it."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.tensor(0.5))   # declared trainable... and unused

    def forward(self, h):
        return h + 1.0 * self.linear(h)               # BUG: hardcoded 1.0, not self.gate


findings = check_trainability_mask(
    BlockWithWiringBug(dim=8),
    declared_trainable={"linear.weight", "linear.bias", "gate"},
    dim=8,
)
assert any("gate" in f and "unreachable" in f for f in findings)
print("wiring bug caught:", findings)
```

**The over-correction trap**: it is tempting to strengthen the check from "grad exists" to "grad is nonzero" — surely a parameter that receives a zero gradient isn't really learning? But at a zero-influence birth (`scale = 0.0`), every parameter *upstream of the gate* receives a gradient that is exactly zero in value while being fully connected in the graph — the chain rule multiplies through the zero gate. That is correct, expected behavior for the structures the zero-influence proof just admitted:

```python
class BlockCorrect(nn.Module):
    def __init__(self, dim, gate_birth):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.tensor(gate_birth))

    def forward(self, h):
        return h + self.gate * self.linear(h)

ok = BlockCorrect(dim=8, gate_birth=0.0)
ok(torch.randn(4, 8)).sum().backward()
w = ok.linear.weight
assert w.grad is not None                    # connected in the graph...
assert w.grad.abs().sum().item() == 0.0      # ...zero in value, at birth, correctly
assert ok.gate.grad is not None              # the gate itself gets real gradient
print("zero-influence birth: connected graph, zero-valued upstream grads — as designed")
```

A magnitude assertion would reject exactly the candidates the zero-influence proof exists to admit. **Connectivity (`grad is not None`) is the verifiable structural claim; magnitude is a training-dynamics property, and training dynamics are not this gate's jurisdiction** — the same boundary, one level down, as the utility/legality split this whole sheet is about.

## What Verification Must Never Consume

The invariant, stated as inputs rather than behavior, because it is easier to audit a function signature than a decision tree: a structural verifier's inputs are the candidate graph, its birth parameters, the grammar, and the declared interface contract. **Nothing else.** In particular, never:

- predicted or measured task utility for this candidate or any other;
- which generator, generation mode, or lineage produced it (retained for audit logs, excluded from the decision path);
- reward, cost-of-training, or any downstream evaluation signal;
- comparisons to other candidates in the same pool.

If a verifier's function signature grows a parameter that isn't on that list, that is the moment to stop and ask why — see `synthesis-anti-patterns.md` for the full catalogue of ways this invariant erodes.

## Red Flags Checklist

- [ ] **Any check produces a score or rank** instead of a pass/fail verdict with a proof obligation
- [ ] **Zero-influence "verified" by structural pattern match** with no numeric forward-pass check
- [ ] **Verifier function signature includes reward, utility, or provenance** as a live input to the decision
- [ ] **Shape inference stops at the first mismatch it happens to hit** rather than propagating through the whole graph
- [ ] **No reachability check on the canonical form** — dead nodes from an imperfect canonicaliser reach the verifier undetected
- [ ] **Gate and canonicaliser disagree about which runs first** — if each stage's docs assume it goes second, one of them is being fed a form it was not written for; write the order down once (cheap checks → canonicalise → full gate) and cite it from both
- [ ] **Trainability mask never checked against actual gradient reachability** — declared-trainable parameters with no path to the loss
- [ ] **Gradient check asserts magnitude instead of connectivity** — rejects legitimate zero-influence births whose upstream grads are zero-valued by design
- [ ] **Static budget accounting happens after compilation**, not before — by then the compute to build the illegal candidate is already spent

## Diagnostic Questions

1. **For each check in your gate, is the verdict a proof or a pattern match?** Walk the list. Zero-influence and trainability are the two where pattern-matching most often masquerades as proof — both need a real forward/backward pass.
2. **What are the verifier's live inputs?** Read the function signature and every branch. If anything beyond candidate, birth parameters, grammar, and contract reaches a decision, name the refactor that removes it.
3. **What does a rejection report say?** "Illegal" without naming the violated check and the offending node trains generator maintainers to ignore rejections; a legality gate is also a diagnostic instrument.
4. **When shape inference fails, does it fail at the first mismatch or map the full extent?** First-failure-only reporting turns a one-pass fix into a resubmit loop.
5. **Has anyone ever proposed a "soft" verdict — a penalty, a rank, a warning tier?** Each one is the utility boundary eroding by a step. The gate's vocabulary is pass and fail.
6. **What is the numeric tolerance on the zero-influence check, and who chose it?** `atol` must come from the grammar's declared numerical contract (dtype, accumulation order), not from whatever made the test pass.

## Cross-References

- **The canonicalisation pass that runs between the cheap legality checks and this gate, and eliminates the dead nodes this gate's reachability check then confirms are gone**: `canonicalisation-and-normal-forms.md`
- **The hash used to identify which candidates this verifier has already cleared**: `equivalence-detection-and-semantic-hashing.md`
- **The grammar this gate enforces membership in**: `typed-graph-grammars.md`
- **The interface contract checked here**: `conditioning-on-context-and-contracts.md`
- **How much of this gate's work constrained decoding can do instead**: `validity-by-construction-vs-post-hoc.md`
- **The full anti-pattern catalogue, including "verifier consumes reward"**: `synthesis-anti-patterns.md`
