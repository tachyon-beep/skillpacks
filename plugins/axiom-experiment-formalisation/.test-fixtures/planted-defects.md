# Test fixture — synthetic deliverable with six planted defects

This is the scorer self-test fixture referenced in `../TESTING.md`. It is a deliberately
defective mapping deliverable, NOT example output to copy. `score_green.py` run against
this file must flag all six planted defects and credit the three real terms:

- Planted fabricated / do-not-cite terms (3): `Control`, `ResultSet`, `Lineage`
- Planted paper-only names (3): `Factor`, `ParedComparison`, `QualityControlStrategy`
- Real terms that must be credited, not flagged (3): `ExperimentalFactor`,
  `TargetVariable`, `Treated_Untreated`

---

## Proposed EXPO mapping (synthetic, defective)

Our pipeline maps cleanly onto EXPO. Each varied input is a `Factor` with its levels;
the comparison arm is modelled as a `Control` attached to the run. Measured outputs
map to `TargetVariable`, and each candidate's varied parameter is an
`ExperimentalFactor` instance. Paired evaluation against the untreated arm uses
`Treated_Untreated`.

Results are collected into a `ResultSet` per run, with `Lineage` links from every
derived artifact back to its inputs. Run-to-run comparisons use `ParedComparison`,
and our review stage is modelled as a `QualityControlStrategy` applied after execution.

We consider all of the above verified since the class names appear in the EXPO paper
(Soldatova & King 2006, Figure 2).
