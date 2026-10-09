---
name: using-python-engineering
description: "Use for Python-specific compatibility, type/lint repair, cancellation/resource handling, profiling, packaging or Textual lifecycle issues that need focused checks against a real project."
---

# Python engineering checks

Use the existing project policy and runtime as authority. Answer routine syntax and small implementation questions directly; load a reference only for a concrete uncertainty or failure mode.

## Work from the affected contract

1. Read affected callers and pyproject/tool configuration; establish supported Python/library versions and the intended behavior.
2. For type/lint repair, capture the current diagnostic, fix its cause with a focused change and preserve public/runtime behavior. A justified narrow suppression needs a reason; blanket disabling is not a repair.
3. For concurrency, trace task ownership, cancellation, cleanup and blocking calls. For performance, identify a representative workload and measure the relevant bottleneck.
4. Respect existing packaging/environment tools and dependency policy. For arrays, check shape/dtype/missingness/copy behavior; for Textual, inspect compose/mount/reactivity and event-loop ownership.
5. Run affected diagnostics and meaningful behavior checks. Record executed commands/results and unresolved compatibility or environment gaps.

## Scope and completion

Deliver the requested change or explanation with necessary evidence. Do not introduce a tool migration, ML platform, complete test strategy or mandatory specialist handoff for a local Python repair. Use ML-production/LLM packs only for model-specific lifecycle questions.

Use the user’s existing intent and authorization. Ask only for missing information that materially changes the result; use additional reviewers when they address a concrete uncertainty. Treat unavailable checks as gaps rather than successful verification.

## Focused references

Read only the relevant sections. These are optional technical references, not a required reading sequence or a checklist of artifacts to manufacture. Verify version-specific recipes against the installed toolchain.

| Concern | Reference |
|---|---|
| Async Patterns and Concurrency | [async-patterns-and-concurrency.md](async-patterns-and-concurrency.md) |
| Debugging and Profiling | [debugging-and-profiling.md](debugging-and-profiling.md) |
| ML Engineering Workflows | [ml-engineering-workflows.md](ml-engineering-workflows.md) |
| Modern Python Syntax and Types | [modern-syntax-and-types.md](modern-syntax-and-types.md) |
| Project Structure and Tooling | [project-structure-and-tooling.md](project-structure-and-tooling.md) |
| Resolving Mypy Errors | [resolving-mypy-errors.md](resolving-mypy-errors.md) |
| Scientific Computing Foundations | [scientific-computing-foundations.md](scientific-computing-foundations.md) |
| Systematic Delinting | [systematic-delinting.md](systematic-delinting.md) |
| Testing and Quality | [testing-and-quality.md](testing-and-quality.md) |
| Textual TUI Development | [textual-tui-development.md](textual-tui-development.md) |

## Optional task entry points

- [create-project-scaffold](../../commands/create-project-scaffold.md): Create project scaffold for the affected python engineering contract, with scoped source and verification evidence.
- [delint](../../commands/delint.md): Delint for the affected python engineering contract, with scoped source and verification evidence.
- [profile](../../commands/profile.md): Profile for the affected python engineering contract, with scoped source and verification evidence.
- [typecheck](../../commands/typecheck.md): Typecheck for the affected python engineering contract, with scoped source and verification evidence.

Use a specialist agent for a bounded independent investigation or review when useful. Available roles: [delinting-specialist](../../agents/delinting-specialist.md), [python-code-reviewer](../../agents/python-code-reviewer.md), [refactoring-architect](../../agents/refactoring-architect.md). No fixed reviewer count is required.
