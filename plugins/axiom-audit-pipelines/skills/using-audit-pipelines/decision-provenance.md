---
name: decision-provenance
description: Use when binding an entry's output to its inputs commitment, ruleset id/version, and code version so that "given the same inputs and ruleset on the same code, you would produce the same output" is a verifiable claim. Covers ML-model rulesets and version-pinning discipline. Produces `05-provenance-bindings.md`.
---

# Decision Provenance

## Overview

**Provenance closes the loop. An entry's `output` is bound to its `inputs_commitment`, its `ruleset_id`/`ruleset_version`, and its `code_version` such that "given the same inputs and the same ruleset on the same code, you would produce the same output" is a *verifiable* claim, not a hopeful one.**

Without provenance, the chain proves only "this output was produced" — not "this output follows from these inputs under this ruleset". The audit succeeds at recording history and fails at explaining it. This sheet specifies how to make the closure auditable, what `ruleset_version` and `code_version` actually mean, and how to handle decisions whose ruleset includes machine-learned models.

## When to Use

Use this sheet when:

- Tier M or above — provenance is mandatory at M+, optional but recommended below.
- Decisions are produced by a rule engine, policy engine, or model where "same inputs + same rules + same code = same output" is a property the auditor wants to test.
- Producing `05-provenance-bindings.md`.

## Core Principle

> A decision's provenance is the four-tuple `(inputs_commitment, ruleset_id+ruleset_version, code_version, output)`. The first three are causes; the fourth is the effect. The chain entry is the witness. The verifier's question — "given these causes, would I see this effect?" — must be answerable, not merely plausible.

## What `ruleset_version` Means

`ruleset_version` is the identity of the rule, policy, or model bundle as it existed at decision time. It must satisfy:

1. **Stable across deployments.** Re-deploying the same ruleset produces the same `ruleset_version`. Hashes work; semver works if disciplined; build-IDs work if reproducible.
2. **Specific enough to pin behaviour.** "v1.2" is not enough if v1.2 has multiple builds with subtle differences. The version commits to behaviour, not just intent.
3. **Resolvable to bytes.** Given `ruleset_version`, an auditor can fetch the canonical bytes of that ruleset. This requires a *ruleset registry* — a content-addressed store of rulesets, with lifecycle controls of its own (see *Ruleset registry* below).
4. **Independent of `code_version`.** Same ruleset, different code — a refactor, a runtime upgrade — must yield different `code_version` but unchanged `ruleset_version`. Same code, different rules — same `code_version`, different `ruleset_version`.

Practical encoding patterns:

- **Hash of canonical ruleset bytes.** `ruleset_version: "sha256:abcd..."`. Strongest. Deduplicates identical rulesets, immune to label drift, immediately resolvable to bytes.
- **Semver with build hash.** `ruleset_version: "1.4.2+build.abcd"`. Acceptable if the build hash genuinely commits to the bytes.
- **Plain semver without build hash.** Fragile. The same `1.4.2` may be deployed twice with subtle differences (a bug-fix backport, a manual edit). Avoid for L+.
- **Date or release-tag string.** "weekly-2026-01-15" — defensible only when the release process produces byte-reproducible artifacts and the registry holds them.

## What `code_version` Means

`code_version` is the producer's executable-bytes identity at decision time. It must satisfy:

1. **Deterministic from source.** Given the source tree at a commit, `code_version` is reproducible. The standard satisfier is git-commit-hash plus build provenance (SLSA, in-toto attestation).
2. **Includes runtime dependencies that affect behaviour.** Pure source hash is insufficient if a different libc version produces different floating-point results, a different timezone database produces different boundary cases, or a different model-runtime version produces different outputs. SLSA-style provenance attestations capture this.
3. **Resolvable to a build artifact.** Given `code_version`, auditor can fetch *the binary that ran*. Some pipelines hold container-image digests (`sha256:...`) for the producer service.

The minimum honest form: `code_version: "git-sha:abcdef0..."` plus a SLSA-style provenance reference.

## Why Separate `ruleset_version` from `code_version`

The single most important provenance property is the ability to answer:

> "The decision changed last Tuesday. Did the rules change, the code change, or both?"

If the two are conflated into one "version" field, this question is unanswerable without spelunking history. With them separate:

| Tuesday's `ruleset_version` | Tuesday's `code_version` | Diagnosis |
|-----------------------------|--------------------------|-----------|
| Different from Monday | Same | Rule change |
| Same | Different | Code change (refactor, runtime upgrade, bug fix) |
| Different | Different | Both |
| Same | Same | Inputs changed — see `inputs_commitment` |

The diagnosis drives the response. A rule change asks "was that change reviewed and approved?" — see *Ruleset registry* below. A code change asks "was that build reviewed and approved?" — see SDLC governance. Conflated, neither question is answerable.

## The Ruleset Registry

A ruleset registry is a content-addressed store of every ruleset that has ever produced a decision in the audit pipeline. It has three responsibilities:

1. **Append-only storage.** A ruleset is added; it is not removed. Removing a ruleset orphans every entry that references it.
2. **Approval metadata.** Each ruleset version carries the approval record: who proposed it, who reviewed it, who authorised production use, when. This metadata is itself part of `decision-provenance` evidence at L+.
3. **Resolution endpoint.** Given a `ruleset_version`, return the canonical bytes of that ruleset. Latency budget per `performance-budget-for-audit-grade-pipelines.md`.

The registry is itself an audit artifact. Entries proposing, reviewing, and authorising rulesets are decisions in their own right and may belong in the audit pipeline (Pattern A or C). For tier L+ this is typical: the meta-decisions about rulesets are themselves audited.

The registry MUST NOT be the same store as the trail — separation of duties applies. A successful attack on the trail must not silently extend to the registry.

## Inputs Closure

`inputs_commitment` already binds the entry to canonical input bytes (per `decision-log-architecture.md`). For provenance purposes, two further requirements:

1. **The inputs are the *complete* input set the decision saw.** "We logged the request body but not the customer profile we joined in" breaks closure. Either include the joined data in canonical inputs, or reference it via content-addressing into the same registry, with `inputs_commitment` covering both. Partial input capture is the most common provenance failure.
2. **Inputs are free of nondeterministic context unless the context is captured.** Random seeds, current time read inside the decision, environment variables, feature flags, A/B test bucket assignments — if they affect the output, they are inputs and must appear in `inputs_commitment`.

The test: replay the decision with the captured `inputs_commitment` plus the captured `ruleset_version` and `code_version`. Do you get the same `output`? If not, an input is missing or a code path is nondeterministic. Both are provenance failures.

## Models as Rulesets

A machine-learned model is a ruleset; treat it as one with care.

The model bytes (weights, architecture, hyperparameters) commit to behaviour the same way rule code does. `ruleset_version` for a model is the hash of the model artifact (`sha256:...` over the canonical serialised weights, plus the inference-runtime configuration that affects output: precision mode, batch handling, kernel selection where it changes outputs).

Three model-specific provenance traps:

1. **Inference nondeterminism.** Two GPUs producing slightly different outputs for the same model and inputs is a violation of "same causes → same effect." Either fix the inference path (deterministic kernels, batch-size pinning, dtype pinning) or include the inference environment in `code_version`. The naive fix — "round the model output" — is a schema-layer choice, not a substitute for closure.
2. **Pre/post-processing.** Tokenisation, normalisation, calibration tables, output decoding — are they part of the model (`ruleset_version`) or part of the code (`code_version`)? Pick once and document. Half-and-half is unreproducible.
3. **Stochastic models.** Models that sample (LLMs at temperature > 0, generative models with randomness) cannot satisfy "same causes → same effect" without capturing the seed. Capture the seed in `inputs_commitment`. Without it, the entry records a sample, not a decision; raise this in `00-` if the decision-shape is in fact non-determined.

For LLM-driven decisions: the prompt, the model identity, the inference parameters (temperature, top-p, max tokens, seed if supported, system prompt) all belong in `inputs_commitment` and the parts that change with deployment belong in `ruleset_version`. See also `axiom-ai-engineering` if this is a primary design concern.

## Bindings Beyond the Entry

Some provenance evidence is too large or too organisationally distinct to inline:

- **Inputs reference the customer profile snapshot.** A profile is content-addressed; the entry references the snapshot hash; the snapshot lives in the input registry.
- **Ruleset references its training-data version.** A model's `ruleset_version` may itself reference (in the registry) the training-dataset version, the training-code version, the evaluation-dataset version. The entry doesn't need all this; the registry resolves it on demand.
- **Code references SLSA provenance.** `code_version` resolves to a SLSA attestation that names the source commit, the build environment, and the materials.

The pattern: the *entry* is small and points outward to content-addressed registries. The closure exists; it is just not inlined.

## What Provenance Does *Not* Cover

- **Why a particular ruleset was chosen.** That is meta-decision; a ruleset registry approval flow records it; it is not part of `05-`.
- **Whether the decision was correct.** Provenance proves causation, not correctness. Correctness is a downstream evaluation against ground truth (see `ordis-quality-engineering` if relevant).
- **Whether the decision was authorised.** Authorisation is a separate property; an authorisation decision is itself a decision in the pipeline.
- **Adversary actions inside the decision.** A producer that lies about its `ruleset_version` cannot be caught by provenance alone; the threat model in `07-` addresses this.

## Spec Output (`05-provenance-bindings.md`)

The sheet's deliverable answers, in order:

1. **`ruleset_version` encoding** — hash, semver+build-hash, or other; resolution endpoint.
2. **`code_version` encoding** — git-sha + SLSA reference, container-image digest, or other.
3. **Ruleset registry** — location, append-only enforcement, approval metadata schema, resolution latency, separation from the trail store.
4. **Inputs closure** — the complete-input rule, treatment of nondeterministic context (seeds, time, env, flags).
5. **Model treatment** (if applicable) — what counts as ruleset vs code, inference-determinism strategy, stochastic-decision policy.
6. **Bindings beyond entry** — content-addressed registries used (profile, training-data, SLSA).
7. **Replay protocol** — see `partial-replay-from-trail.md`; provenance closure is replay's precondition.
8. **Cross-pack handoff** — SDLC governance for ruleset approval (`axiom-sdlc-engineering`); model lifecycle (`yzmir-ml-production` or relevant ML pack).

## Common Mistakes

| Mistake | Why it fails | Fix |
|---------|--------------|-----|
| `version: "1.4.2"` only — single field for both ruleset and code | Cannot diagnose "rules changed or code changed?" | Two fields; both mandatory |
| Plain semver `ruleset_version` with no build commitment | Same label, different bytes deployed | Hash or semver+build-hash |
| Code refactor produces "no behaviour change so we kept the version" | Code version commits to bytes, not intent | Always bump on rebuild |
| Inputs include request body but not joined data | Closure broken; replay impossible | Capture joined data or reference it via content-addressing |
| Random seed not captured for stochastic decisions | Replay yields different output; "same causes → same effect" fails | Seed is an input; capture it |
| Model output rounded "to make replay deterministic" | Hides nondeterminism rather than fixing it | Fix the inference environment; capture it in `code_version` |
| Ruleset registry colocated with trail store | Single attack vector compromises both | Separate stores, separate access controls |
| Ruleset removed from registry because "no entries cite it anymore" | Until later one does, and it can't resolve | Append-only; rulesets never deleted |
| Approval metadata not captured | Auditor asks "who approved this?" — no answer | Approval flow produces audit entries of its own |

## The Bottom Line

**Bind output to inputs, ruleset, and code as a four-tuple. Make ruleset and code separately versioned and content-resolvable through registries that live outside the trail. Capture nondeterministic context as inputs. Treat models as rulesets with explicit inference-determinism policy. Closure is the property that turns the chain from a record of events into evidence of why decisions came out the way they did.**

---

**Retrieval test (run at end of build):** "Two decisions yesterday produced different outputs for the same customer query. We have full audit entries for both. Walk through how to determine whether the difference was caused by inputs, rules, or code."
