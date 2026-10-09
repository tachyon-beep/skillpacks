# Resources, Prompts and Sampling

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

## Resources

Use addressable context when the host can discover/read/attach it reliably. Specify URI identity, MIME/schema, access, bounds, freshness and subscription behavior if supported. Avoid credentials in content. Read-only tools remain valid where model selection, queries, cost or host support favor them.

Test the actual client workflow, including pagination/list changes, permission failures and large content. A protocol feature's existence does not prove UI attachment or model visibility.

## Prompts

Expose a reusable user-selected workflow only when its arguments and result add useful context. Treat prompt/resource/server text as data under the host's instruction hierarchy; it cannot grant permissions or override user/system authority. Verify the host exposes it as intended.

## Sampling

Use host inference only after capability negotiation and applicable consent. Specify messages/context requested, result bounds, refusal/error/cancellation handling and whether any data leaves a trust boundary. Handle unsupported hosts explicitly.

Server-owned inference is a separate legitimate product option with its own credentials, data/cost and authorization contract. An imported external SDK is not a universal blocker. Compare alternatives from actual requirements and host behavior, not a mandatory primitive slogan.

## Evidence

Record supported host/protocol/SDK versions, observed capability flow, authorization/data constraints and relevant smoke/task results. No universal artifact count or inference architecture applies.
