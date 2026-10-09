# Testing MCP Servers

Optional recipe: select tests by the surface's risks and the actual host/protocol/SDK versions. A manual demo is useful exploration but does not establish regression coverage.

## Separate the claims

| Layer | What it establishes | What it does not establish |
|---|---|---|
| Protocol/contract test | Initialization, schemas, envelopes, pagination, permissions, retries/races | Model interpretation or tool choice |
| Inspector/client smoke | Server starts and a supported client lists/invokes capabilities | Complete behavior or production usability |
| Frozen call replay | Captured arguments still produce the declared results/state | How a model reads descriptions or chooses calls |
| Real model/host task evaluation | Selection, interpretation, recovery and task outcomes in that configuration | Deterministic correctness for every execution |

A metadata/schema snapshot can flag description edits for review. Frozen calls do not detect changed model interpretation. Evaluate descriptions with actual model/host tasks when selection ambiguity or description changes warrant it; record configuration, task rubric, outcomes, failures and denominator. Choose repetition from uncertainty and cost, not a universal pass quota or per-tool golden count.

## Deterministic contract checks

Drive the real transport/client boundary and assert results and committed state. Cover relevant input/schema errors, denied access, conflict, not-found, cancellation, bounded output and negotiated capabilities. Use the selected protocol/SDK's current official documentation to distinguish protocol errors from tool-execution errors.

For mutations, test the declared guarantee under same-key retry, response loss, parallel callers and crash boundaries. Do not equate two equal responses with one effect: inspect durable state/effect evidence too. Retain independent expected results and realistic fixtures.

## Replay with explicit bindings

The following transport-independent harness compares all scalar values, exact object keys and list order. Dynamic IDs are captured from a result and substituted into later arguments; they are never replayed as a literal `<UUID>`. Use it with a real initialized session adapter whose `call` returns a decoded result envelope.

Only mark declared volatile fields as captures, and separately validate their schema/type/format where relevant. Capture does not make arbitrary output differences acceptable. Expected stable fields and error/state assertions remain explicit.

```python
import copy


def bind_args(value, bindings):
    if isinstance(value, dict):
        if set(value) == {"$ref"}:
            return copy.deepcopy(bindings[value["$ref"]])
        return {k: bind_args(v, bindings) for k, v in value.items()}
    if isinstance(value, list):
        return [bind_args(v, bindings) for v in value]
    return value


def assert_result(actual, expected, bindings):
    if isinstance(expected, dict) and set(expected) == {"$capture"}:
        name = expected["$capture"]
        if name in bindings:
            assert type(actual) is type(bindings[name])
            assert actual == bindings[name]
        else:
            bindings[name] = copy.deepcopy(actual)
        return
    assert type(actual) is type(expected), (actual, expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (actual, expected)
        for key in expected:
            assert_result(actual[key], expected[key], bindings)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), (actual, expected)
        for got, want in zip(actual, expected):
            assert_result(got, want, bindings)
    else:
        assert actual == expected, (actual, expected)


async def replay(call, steps):
    bindings = {}
    for step in steps:
        args = bind_args(step["arguments"], bindings)
        result = await call(step["tool"], args)
        assert_result(result, step["expected"], bindings)
    return bindings


STEPS = [
    {"tool": "claim_issue", "arguments": {"issue_id": 42},
     "expected": {"status": "claimed", "lease_id": {"$capture": "lease"}}},
    {"tool": "resolve_issue",
     "arguments": {"issue_id": 42, "lease_id": {"$ref": "lease"}},
     "expected": {"status": "resolved", "changed": True}},
]
```

Strict comparison intentionally catches added/removed fields; update fixtures deliberately when the contract changes. If order is semantically irrelevant, canonicalize that specific collection under its declared contract rather than normalizing every list. Include full MCP envelopes/schema checks in the transport adapter tests; this helper does not implement the protocol.

## Review output

Name the exercised tools/tasks, environment/baseline, observed assertions/state checks, failures and gaps. A supported clean result is valid. Do not call a frozen transcript an agent evaluation, or claim all tools are covered without checking the actual surface and test inventory.
