# Context budgets, provenance and caching

Use when long inputs, repeated context, conversation state or cache behavior affects quality/cost. Caching and summarization solve different problems; neither proves that important evidence survived.

## Budget and selection

- Inspect the actual tokenizer/provider accounting, input/output limits and tool/schema overhead. Reserve output and reasoning budget as supported; handle overflow/truncation explicitly.
- Retain task instructions, decisive evidence, provenance, current user intent and pending decisions. Drop duplicate or irrelevant context before compressing important details.
- Select retrieved evidence by relevance, permission, freshness and diversity; preserve source identifiers and enough surrounding context to support the claim.
- Treat summarization as lossy. Test preservation of exact identifiers, exceptions, commitments and uncertainty; keep recoverable source references for details that cannot fit.
- For long conversations, define durable state and a refresh/retrieval mechanism. A summary is not an authoritative replacement for a user's unrecorded authorization.
- Evaluate ordering and long-context retrieval on the actual task; theoretical context capacity does not establish robust use of every position.

## Cache contract

- Verify provider support, cache scope, eligible prefix layout, minimum sizes, TTL and billing from current docs.
- Place compatible stable material before variable suffixes when the provider uses prefix caching. Changing tools/schema/model/auth scope may change cache compatibility.
- Measure cache hit evidence and effective input cost/latency. A repeated string does not itself prove a cache hit.
- Separate cached public/shared instructions from sensitive tenant-specific data; respect isolation/retention rules at the service boundary.
- Compare cached long context, retrieval and simpler prompt changes under task quality and operational limits. Cache availability is not a reason to include irrelevant data.

## Overflow/recovery checks

Test unusually large inputs, mixed-language/token-heavy content, long tool results, summarization failures and resumed sessions. Ensure omitted evidence cannot silently turn a supported answer into a fabricated one. Report unanswerability or fetch the missing source when appropriate.

## Deliverable

Context-selection/cache policy with measured token/cost/quality evidence, provenance/recovery behavior and overflow tests. See [RAG](rag-architecture-patterns.md) and [evaluation](llm-evaluation-metrics.md) for evidence and outcome checks.

## Existing source pointers

Check current applicability/version before using a recipe. These pointers are optional supporting sources.

- <https://platform.claude.com/docs/en/build-with-claude/prompt-caching>
- <https://platform.openai.com/docs/guides/prompt-caching>
