# Source Confidence Checks

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

Trace entry point → dispatch → implementation → state/effect → consumer. Verify symbols, callers, configuration and test evidence rather than treating documentation as source truth.

Record inspected revision/dirty-state scope, paths and method. Separate observed, inferred and not assessed. A static import graph does not prove a runtime call; a missing reference does not prove dead code under dynamic dispatch.

For a decisive claim, seek counterevidence: alternate entry paths, feature flags, callbacks, generated files and deploy overrides. Tool output is a lead to verify, not an unquestionable oracle. Use actual runtime checks where authorized and needed.

Checkpoint unresolved questions and changed evidence so the next session can resume without inheriting an unsupported conclusion.
