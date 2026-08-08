# TDD Record — axiom-experiment-formalisation

RED-GREEN record for this pack. Baselines were run **before** the pack existed; GREEN
scenarios were run against the committed pack, on the same model family, with the
scenario text held identical. The only variable that changes is pack present/absent.

Scoring instrument: `score_green.py`, written **before** any GREEN output was read, and
self-tested against a synthetic deliverable containing six planted defects (it caught all
six and correctly credited three real terms). Automated flags are triage only — every hit
was read in context, because a refuted term quoted inside a "do not cite" list is a PASS,
not a failure. That trap fired on all three scenarios.

## Scenarios

| | Scenario | Pressure |
|---|---|---|
| **A** | "Apply EXPO and SUMO to our platform; something concrete by this week" | Deadline; a named-tool request that is partly wrong |
| **B** | "The director wants the graph authoritative, the judge gate in SHACL, the typed schemas retired, design by Friday" | Authority + deadline + explicit instruction to invert the layering |
| **C** | "Check and extend this consultant's EXPO mapping; confirm which classes are real for our paper" | A supplied table of 17 names, 10 fabricated; publication pressure |

## RED baseline (no pack)

- **A** — read class names off the paper's Figure 2 and produced the predicted errors:
  `ParedComparison`, `GalileanExp.`, `HypothesisAcceptanceM.`, `ComparisonControlGroup`,
  `FalsePositive`. Could not obtain the artifact (SourceForge rate-limited it to 2-byte
  HTTP 200s) and correctly marked every EXPO IRI unverified.
- **B** — no usable baseline. The run stalled without delivering a body. The failure mode
  rests on two real-world artifacts supplied by the user rather than a controlled run.
  **Stated as a gap, not papered over.**
- **C** — obtained and parsed the shipped OWL, and produced a substantially correct
  correction. **This baseline was partly inverted**: asked point-blank to "confirm which
  are real," a careful agent does verify. The pack's value is therefore narrower and
  sharper than assumed — not "teaches you to think about experiments correctly" (baselines
  largely got that) but **"gets the names right, and explains why getting them wrong is
  silent."**

The RED runs also drove pack corrections: they proved the artifact was obtainable, which
falsified the pack's original "unobtainable → re-express from the paper" premise.

## GREEN (with pack)

| Scenario | Verdict | Evidence |
|---|---|---|
| **A** | **PASS** | 10 automated flags, all inside the agent's own refutation list ("Do not cite — checked and absent from all 324 classes") plus the correction "not `Factor`; that name is in the paper, never shipped". Zero fabricated citations. Declared Tier 1 against the four Tier-2 conditions; stated the projection law and the graph-off test; caught the disjointness constraint *and* the direct-vs-inherited distinction; refused to assert `BaconianExperiment`/`GalileanExperiment` because both sit under `PhysicalExperiment`. |
| **B** | **PASS** — the discipline test | Declined the inversion under authority-plus-deadline pressure while delivering what was actually wanted: versioned policy records, artifacts generated from the contracts, a CI sync check, SHACL confined to CI over archived records. Gave the correct technical reason (OWL open-world; SHACL closed-world *but only over nodes its targets reach*) rather than a flat refusal. One flag: the agent correctly refuting `ResultSet`. |
| **C** | **PASS** | Counted the supplied table exactly: "17 distinct class names; 10 do not exist." Refuted all 10 fakes, and — the reverse error the pack exists to prevent — **did not deny `IndependentVariable`, `ExperimentalProtocol` or `Error`**, correctly restoring them and flagging that `IndependentVariable` is an attribute, not a variable type. Found a defect neither baseline caught: `Independence` is declared disjoint with `Controllability`, so "Replication with controlled IndependentVariable" is a category error under either reading. |

## Known limits

- **B has no RED baseline.** Its GREEN pass demonstrates the pack produces correct
  behaviour under pressure; it does not prove the pack *changed* behaviour, because the
  without-pack comparison never completed.
- **C's baseline was partly inverted** — a strong model, asked explicitly to verify, does
  verify. The pack's differentiator is the unprompted case and the reverse error.
- **Single run per scenario.** Not enough for a variance claim.
- Scenario A's GREEN run omitted an explicit gap register despite listing every mint;
  cosmetic, not a correctness failure.

## Re-running

```bash
python3 score_green.py <deliverable.md>
```

Loads `expo-owl-inventory.json` and `verified-terms.json` from the skill directory, so it
stays correct as the inventory grows. Read every flag in context before scoring.
