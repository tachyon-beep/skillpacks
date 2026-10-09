# MCP Primitive Selection

Optional reference: use the parts relevant to the current question and selected policy. Existing records can satisfy the evidence; no fixed artifact count, review duration or reviewer count applies.

## Choose for the actual host and task

| Capability | Good fit | Check |
|---|---|---|
| Tool | Model-selected operation or parameterized/expensive read | Intent, schema, authority, retries and bounded result |
| Resource | Addressable context a client can discover/read/attach | Actual host discovery/attachment, URI stability, access and size |
| Prompt | Reusable user-selected workflow/context template | Host exposure, arguments, trust and permissions |
| Sampling | Server asks the host for model inference | Negotiated support, host consent, data/cost boundary and fallback |

A read-only tool is valid even for stable content when model selection/discovery or host interoperability benefits. A stable URI alone does not prove the host reliably attaches it. Confirm capabilities with a representative client; do not assume protocol availability means product/UI support.

## Inference ownership

A server may implement product functionality using its own inference provider. An external SDK import is not automatically a defect. Record who pays/authorizes it, credentials, data leaving the boundary, model/version behavior, failure/latency and consent obligations. Compare host sampling where supported and appropriate; do not force it when it violates the product/runtime boundary or lacks support.

If borrowing host inference, negotiate capability and handle refusal/cancellation/errors. Neither direct SDK use nor sampling permits exceeding the user's actual authorization or treating model output as trusted enforcement.

## Review

Trace the primitive to intent and actual host behavior. Test representative discovery/attachment/selection, permission and retry paths. Findings need evidence and consequence. First-pass agreement or a clean audit is valid after scoped checks; no disagreement quota applies.

Output: capability, intended caller/host, selection rationale, unsupported cases, authority/data boundary and verification evidence. Use a manifest only when a consumer needs one; a compact design note can carry the same fields.
