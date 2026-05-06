---
name: ast-visitation-patterns
description: Use when designing the substrate of a static analyzer — choosing how the engine walks the AST. Covers visitor (double dispatch over node types), walker (depth-first traversal with parent tracking), and transformer (rewrite, in-place or copy-on-write); the lossless-vs-structural AST distinction; what gets visited, what doesn't, and the gotchas (synthetic nodes from desugaring, comment placement, source-position preservation, parent pointers, fixed-point iteration during rewrite). Produces `01-visitation-strategy.md`.
---

# AST Visitation Patterns

## Why Visitation Strategy Is Load-Bearing

The way an analyzer walks the AST determines what its rules can see, in what order, with what context. Get it wrong once and every rule pays — either by re-implementing traversal, by missing nodes, or by silently disagreeing on what "the AST" even is.

Three failure modes are common:

- **Ad-hoc traversal** — every rule writes its own recursion. Three rules later, three slightly different ideas of "visit children" exist. Comments and decorators get visited inconsistently. Rules are not composable.
- **Wrong AST grain** — an analyzer built on a structural AST (Python's `ast`, Babel's `estree`) cannot rewrite source faithfully because formatting and comments were thrown away at parse time. A rule that wants to suggest a fix has nothing to put back.
- **Implicit visitation order** — rules assume "this fires before that" without the visitor stating an order. When the parser changes (a new construct, a new desugaring), order shifts and rules silently break.

`01-visitation-strategy.md` exists to make these decisions visible *before* the first rule is written.

## The Three Patterns

### Visitor (double dispatch over node types)

```python
class TaintVisitor(ast.NodeVisitor):
    def visit_Call(self, node: ast.Call) -> None:
        # called for every Call node; rules attached to Call live here
        ...
        self.generic_visit(node)  # descend into children

    def visit_Name(self, node: ast.Name) -> None:
        ...
        self.generic_visit(node)
```

**Use when:** the analyzer is read-only (no rewrite), rules are attached to specific node types, and you want compile-time-checked rule registration ("if you registered `visit_Call`, the engine guarantees you see every Call").

**Strengths:** dispatch is type-driven; rules don't need to type-test; new node types fail loudly when the visitor lacks a handler.

**Weaknesses:** parent-tracking is awkward (the visitor only knows the current node); cross-cutting state (e.g., "are we inside a class body?") requires manual scope-stack management; cannot easily reorder traversal.

### Walker (depth-first traversal, often with parent stack)

```python
def walk(node: AST, parents: list[AST]) -> Iterator[tuple[AST, list[AST]]]:
    yield node, parents
    for child in ast.iter_child_nodes(node):
        yield from walk(child, parents + [node])
```

**Use when:** rules need parent context (e.g., "an `Assign` to a `Name` whose enclosing scope is a class body"), the analyzer queries the tree more than it dispatches on it, and you want full control over traversal order.

**Strengths:** parents are first-class; rules can express "node X with ancestor Y"; pre-order vs post-order vs reverse is a one-line change.

**Weaknesses:** rules dispatch on node type themselves (more `isinstance` than visitor); easy to forget to descend; no compile-time guarantee that a rule sees every relevant node.

### Transformer (rewrite, in-place or copy-on-write)

```python
class SanitiseTransformer(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        if is_unsafe(node):
            return wrap_in_sanitiser(node)  # returning a new node replaces it
        return node
```

**Use when:** the analyzer produces a fix, a refactoring, or a desugared form for a downstream pass. NodeTransformer is the visitor's rewriting cousin: returning a new node from a `visit_*` replaces the original.

**Strengths:** rewrite is structural and traverses naturally; returning `None` deletes; returning a list inserts.

**Weaknesses:** **fixed-point is on the user** — if your rewrite produces a node that *itself* matches a rule, you must re-run; the transformer does not iterate to convergence by default; lost source positions and comments unless the AST is lossless (see below).

## Lossless vs Structural ASTs

The AST you choose determines what you can do with it. This is the single most consequential decision in `01-visitation-strategy.md`.

| AST kind | Examples | Preserves | Use for |
|----------|----------|-----------|---------|
| **Structural** (semantic) | Python `ast`, Babel `estree`, `rustc_ast` | Program meaning only | Read-only analysis, taint, type checks |
| **Lossless** (syntactic / CST) | `libcst` (Python), `swc`/`@babel` w/ tokens, `tree-sitter`, `ruff_python_parser`, `syn` (Rust, partial) | Whitespace, comments, exact source positions | Rewrite, autofix, formatting-aware analysis |

**The trap:** teams build a read-only analyzer on a structural AST, then discover six months later that they want autofix. The structural AST cannot reproduce the source. Either rebuild on a lossless CST (large rework) or accept that "fix" means "diff suggestion the user has to apply by hand."

**Decision rule for `01-`:** if the analyzer might *ever* need to produce a fix, autofix, or formatted output, choose lossless from day one. The visitation interface is similar; the parse cost is somewhat higher; the cost of switching later is "rewrite everything."

## What Gets Visited (and What Doesn't)

`01-visitation-strategy.md` must enumerate explicitly:

- **Comments and docstrings** — most structural ASTs drop comments entirely. Lossless CSTs keep them, but rules must opt in to seeing them.
- **Synthetic nodes** — desugarings (Python `f-string` → `JoinedStr`, async `for` → state machine in some toolchains, decorator application order) introduce nodes that don't exist in the source. Rules that match source patterns must either run pre-desugar or be aware of the desugared shape.
- **Type annotations** — Python evaluates string annotations lazily (PEP 563); some ASTs deliver the string, some deliver the evaluated form. Pick one and document.
- **Imports** — `from X import *` expands at runtime, not parse time. The AST sees `*`; the analyzer must decide whether to resolve it (requires module loading) or treat it as opaque.
- **Generated code** — code emitted by macros (Rust), code generators (Protobuf, OpenAPI), or build-time templating may or may not be in scope. Decide and document.

## Source-Position Preservation

For diagnostics to be useful, every emitted finding needs (file, start_line, start_col, end_line, end_col). For SARIF emission , you also need byte offsets.

**Rules:**

- **Structural ASTs** typically attach `lineno`, `col_offset`, `end_lineno`, `end_col_offset` to nodes. Verify per language — some node kinds are missing positions in some parsers (synthetic nodes especially).
- **Lossless CSTs** keep ranges by construction, often with byte-accurate spans.
- **Always cite a *range*, not a *point*.** A finding at `line 14` is unactionable in a 200-character line; `line 14, col 8 — line 14, col 27` is.
- **Test `--show-source` early.** A diagnostic that looks fine in tests can point at the wrong column once line continuations or BOMs enter the picture.

## Parent Pointers and Scope

Most ASTs do not carry parent pointers. Rules that need them have three options:

1. **Walker pattern** — pass the parent stack through traversal (cheap, explicit, easy to reason about).
2. **Pre-traversal pass** — annotate every node with `node.parent` (mutates the tree; some ASTs forbid this; thread-unsafe).
3. **Visitor with a scope stack** — the visitor maintains a stack of "current scope" (function, class, module, comprehension) and rules read it (cleaner than full parents; loses non-scope ancestors).

For most analyzers, **walker for queries + visitor for dispatch** is a sound combination: the walker materialises (node, ancestors, scope) tuples; the visitor dispatches by node type; the rule reads scope/ancestors when needed.

## Fixed-Point and Re-Visitation

Transformers must reason about whether their output is itself subject to further rules.

- **Single-pass rewrite** — transformer runs once; nodes produced by the rewrite are *not* re-visited. This is fast and predictable but can leave inconsistent trees if a rewrite produces a node another rule should have caught.
- **Iterate-to-fixpoint** — transformer runs in a loop until no node changes. Termination requires either a measure that decreases (e.g., a complexity metric) or an explicit max-iteration cap with a logged warning. **Without termination reasoning, this loops indefinitely on the wrong rule pair.**
- **Topological-sort rules** — rule A produces nodes consumed by rule B; running A then B, never B before A. This is the most predictable pattern at the cost of declaring rule dependencies explicitly.

`01-` must state the choice. "We re-run until convergence" is a hand-wave unless the convergence argument is in writing.

## Language-Specific Notes

- **Python** — `ast` (structural, stdlib), `libcst` (lossless, comments + whitespace), `tree-sitter-python` (lossless, language-agnostic), `parso` (lossless, used by Jedi). For rewriting source, libcst or tree-sitter; for read-only analysis, `ast` is faster.
- **JavaScript/TypeScript** — `@babel/parser` (structural by default; lossless with tokens option), `@typescript-eslint/parser` (TS-aware), `swc` (Rust-based, fast), `tree-sitter-javascript`. For TS-aware analysis you need the TypeScript compiler API to get inferred types.
- **Rust** — `syn` for procedural macros (token-tree-aware but lossy on whitespace), `rustc_ast`/`rustc_hir`/`rustc_mir` if you write a compiler-internal lint (use `clippy_utils`), `tree-sitter-rust` if you don't want a compiler.
- **Go** — `go/ast` is structural; `go/parser` with `parser.ParseComments` keeps comments; `go/types` provides inferred types; cross-package analysis uses `golang.org/x/tools/go/packages`.
- **Generic** — `tree-sitter` for any language with a grammar; lossless, fast, but no type information.

## The Decision Output (`01-visitation-strategy.md`)

A complete `01-` answers:

1. **AST library** — name and version (e.g., `libcst==1.5.0`, `tree-sitter-python==0.21.0`).
2. **Structural or lossless?** — with the rationale (read-only? autofix planned?).
3. **Visitor / walker / transformer / hybrid?** — with the rationale.
4. **Visitation order** — pre-order, post-order, document-order, reverse-document; whether it matters.
5. **What's in scope** — comments, docstrings, type annotations, imports (resolved? wildcards?), generated code.
6. **Synthetic nodes** — what desugarings happen at parse time and whether rules see source-shape or desugared-shape.
7. **Parent / scope mechanism** — walker stack, scope stack, pre-traversal annotation, or none.
8. **Fixed-point semantics** — if rewriting: single pass, iterate-to-fixpoint with termination argument, or topological with declared dependencies.
9. **Source positions** — what fields, what granularity (byte / col / line), what nodes are missing positions.
10. **Out-of-scope languages or constructs** — record explicitly. ("We do not visit Jinja templates inside Python strings; that is a separate analyzer.")

## Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Build read-only analyzer on structural AST, want autofix six months later | Rewrite to lossless = months of work | Choose lossless from day one if there's any chance of fix output |
| Rules use `isinstance` and write their own recursion | Inconsistent traversal across rules; missed nodes | Pick visitor or walker, declare it, build a base class rules inherit |
| Assume implicit traversal order ("comments before the def fire first") | Order shifts when parser updates | State traversal order in `01-`, write a rule-ordering test |
| Mutate AST in place during read-only analysis | Some AST libraries forbid it; threads break | Use copy-on-write transformer or build a separate annotation map |
| Cite findings at single-point locations | Diagnostics are unactionable | Always cite a range; verify against `--show-source` early |
| Iterate transformer to convergence with no termination argument | Infinite loop on adversarial input | Either a decreasing measure or a max-iteration cap with a warning |
| Assume parent pointers exist | Rule code crashes on root nodes | Pick a parent strategy in `01-` and stick to it |

## Cross-References

- `taint-lattice-design.md` — what the visitor produces (the IR over which the lattice is computed)
- `three-phase-inference.md` — how visited nodes feed the inference worklist
- `plugin-architecture-for-analyzer-rules.md` — how rules attach to visitor entry points
- `decorator-as-assertion.md` — the descriptor pattern for runtime + static dual analysis (visitor sees the decorator; runtime sees the wrapped call)
- `scaling-to-large-codebases.md` — incremental visitation, caching the AST itself
