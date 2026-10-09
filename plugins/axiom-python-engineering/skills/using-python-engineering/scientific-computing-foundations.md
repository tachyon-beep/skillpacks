# Array and dataframe failure checks

## Establish the data contract

Record shape, axes, dtype, units, missingness, ordering and expected precision. Inspect boundary inputs rather than inferring these from a successful example. Array broadcasting and dataframe index alignment can return plausible but wrong results.

## Check before optimizing

- Verify broadcasting axis by axis, including singleton and empty dimensions. A shape-compatible expression may still compare the wrong entities.
- Distinguish a view from a copy; inspect mutation/aliasing, strides and contiguity where a downstream library requires them.
- Integer overflow, implicit casts and float accumulation can change semantics. Choose tolerances from the numerical contract rather than widening until tests pass.
- Separate NaN, missing values, sentinels and domain-invalid observations. Check reductions with all-missing and empty inputs.
- Pandas index alignment, joins and grouping can reorder or multiply records. Validate keys/cardinality and preserve an explicit row identity where needed.
- Chunked processing must preserve cross-chunk state, boundary effects and aggregation semantics.
- Vectorization may trade memory for speed. Measure peak memory and representative dimensions; a fast small example can fail at production scale.
- Serialization may lose dtype, timezone, category or index meaning. Validate a real round trip against the consumer contract.

## Evidence

Use edge cases that distinguish plausible wrong implementations: transposed axes, duplicate keys, shuffled rows, non-contiguous views, extremes and missing data. Compare against an independent small oracle when practical. Report numerical and performance evidence separately.
