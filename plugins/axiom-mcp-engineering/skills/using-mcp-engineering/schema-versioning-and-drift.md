---
name: schema-versioning-and-drift
description: Use when a new model release re-interprets your tool descriptions and previously-correct calls start failing, when you are about to rename or retype a tool parameter and need to know whether it breaks live agents, when a "minor server version bump" silently changed a return field's type or tightened a permission, when you cannot tell which model versions your surface is even validated against, when a deprecated parameter is still being passed and nothing signals the agent to stop, when capability negotiation is missing and the client cannot tell which protocol revision or features your server supports, when JSON Schema dialect mismatches cause structured-output validation to fail, or when you need a version-drift exposure assessment before a model upgrade lands. Covers backward-compatible parameter evolution, additive-vs-breaking change classification, self-signaling deprecation, capability declaration under MCP revisions (2025-06-18, 2025-11-25), outputSchema/structuredContent evolution, and golden-conversation drift regression.
---

# Schema Versioning and Drift

## What this sheet is for

**Your tool surface has two clients you cannot version against: the model and the calendar.** A REST API versions against client code that a human wrote once and changes deliberately. An MCP tool surface is re-read, from scratch, by every model on every turn — and the model you validated against will be silently replaced by a new release that reads the same words differently. This sheet is about surviving that, and about the breaking changes that hide behind a server version bump where neither the agent nor the host can see them.

There are two distinct drift axes, and conflating them is the root error:

- **Surface drift** — *you* change the schema (rename a parameter, retype a return field, remove a tool, tighten a permission). This is under your control and must be classified, signaled, and negotiated.
- **Interpretation drift** — *the model* changes (a new release re-reads an unchanged description and decides a parameter means something else, or stops calling a tool it used to call, or starts passing a value you never expected). This is *not* under your control. You cannot prevent it; you can only detect it and bound your exposure to it.

The MCP Consistency Gate clause this sheet owns: **non-backward-compatible schema changes bump server capability** — they are visible to the agent in the protocol, never invisible behind a server version string. Everything below is in service of that clause and of the architect/critic split that reads it from both sides.

## Two readings of this sheet

- **Architect (constructive):** classify every proposed change as additive or breaking; evolve parameters without breaking live agents; declare capabilities so the host knows what it is talking to; build deprecation that the surface itself signals; pin and grow the set of model versions you validate against.
- **Critic (adversarial):** assume the architect mis-classified at least one change as "minor." Find the breaking change hidden behind a server version bump. Find the renamed parameter with no capability bump. Find the surface that has never been re-validated against the current model release. Every finding carries **severity (blocker/major/minor/nit) + evidence** (the schema diff, the capability declaration, the conversation fragment, the validation matrix cell).

If architect and critic agree the surface is "backward compatible" without a diff in hand, the critic is rubber-stamping.

---

## Currency: what "the protocol" means right now

Write to the **2025-11-25** revision (current); the prior widely-deployed revision is **2025-06-18**. Facts that govern this sheet:

- Revisions are **date-stamped strings**, not semver. The normative artifact is the TS schema in the repo (`schema/2025-11-25/schema.ts`). "Which MCP version" is answered by a date, surfaced during initialization.
- Base wire format is **JSON-RPC 2.0 over a stateful connection with capability negotiation**. Capabilities are exchanged once, in `initialize`, and are the protocol-level place where "what this server can do" is declared.
- **JSON Schema 2020-12** is the default dialect for all MCP schema definitions as of 2025-11-25. A server still emitting draft-07 idioms (e.g. `"definitions"` instead of `"$defs"`, boolean `exclusiveMinimum`) is a dialect-drift defect that bites structured-output validation.
- Tools support **`structuredContent`** plus a declared **`outputSchema`**. Input-validation failures are returned as **Tool Execution Errors** (results with `isError: true`), not protocol errors — so the model can self-correct. The output schema is now part of your contract surface and drifts like any other.
- Non-backward-compatible changes are surfaced through **capability negotiation** and through the `listChanged` notifications (`tools/list_changed`, `resources/list_changed`, `prompts/list_changed`). A surface that changes without a re-`list` and without a capability signal is invisible to the client.

You are versioning against **two** moving things: the MCP revision (you can read it) and the model behind the client (you cannot).

---

## The change-classification table (architect's first move)

Before any edit ships, classify it. The class determines what you must do.

| Change | Class | What it breaks | Required action |
| --- | --- | --- | --- |
| Add a new tool | **Additive** | Nothing | `tools/list_changed`; new golden test |
| Add an **optional** parameter (with default) | **Additive** | Nothing | Update description; new golden test |
| Add a field to a return shape | **Additive** *(if agents ignore unknown fields — they do)* | Nothing | Update `outputSchema`; new golden test |
| Loosen a constraint (widen enum, raise max length) | **Additive** | Nothing | Update schema + description |
| Rename a parameter | **Breaking** | Every live call using the old name | Keep old name as deprecated alias; capability/version signal; deprecation window |
| Add a **required** parameter (no default) | **Breaking** | Every existing call | Make it optional-with-default, or new tool + deprecate old |
| Change a parameter's type (`string`→`object`) | **Breaking** | Every call passing the old type | New tool or new param; never silent retype |
| Change a return field's type or remove it | **Breaking** | Agents that parse it | `outputSchema` bump visible to client; deprecation window |
| Tighten a constraint (narrow enum, lower max) | **Breaking** | Calls at the old boundary | Treat as breaking; signal |
| Remove a tool | **Breaking** | Agents that call it | Deprecate-then-remove with a window; `tools/list_changed` |
| Tighten a permission / scope | **Breaking** | Agents acting under old scope | Capability signal; do not silent-fail |
| Change tool *behavior* under the same schema | **Breaking (silent)** | Everything; invisible | New tool or explicit capability; never reuse the name |

**The rule the table encodes:** *additive is free; breaking must be visible in the protocol.* "Visible in the protocol" means the host can detect it at `initialize`/`tools/list` time — via a capability flag, a changed `outputSchema`, a deprecation annotation, or a new tool name — **not** by reading your server's `package.json` version, which the model never sees.

> The trap this sheet exists to close: a "v2.1.0 → v2.2.0 minor bump" that renamed `repo` to `repository` and retyped `labels` from `string` to `string[]`. Semver said minor. The protocol said nothing. Every live agent broke. The version number lied because it was versioning the *code*, not the *contract the model reads*.

---

## Backward-compatible parameter evolution

The discipline: **never mutate a parameter in place. Add the new shape, accept both, deprecate the old, remove after a window.**

### Worked example — evolving a parameter without breaking live agents

A GitHub-style issue tool starts with a single-label string. You need multi-label support. The wrong move is to retype `label: string` into `labels: string[]` — that is a silent breaking change.

```python
# server.py  — using the Python MCP SDK (mcp.server)
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("issue-tracker")

class CreateIssueArgs(BaseModel):
    title: str = Field(description="Issue title shown in the tracker.")
    body: str = Field(default="", description="Issue body (markdown).")

    # DEPRECATED 2026-04 — single-label form. Accepted for backward
    # compatibility with agents/hosts pinned to the pre-multi-label surface.
    # Removal scheduled no earlier than 2026-10 (>=2 model-release windows).
    label: str | None = Field(
        default=None,
        deprecated=True,  # JSON Schema 2020-12 annotation; surfaced to client
        description=(
            "DEPRECATED: pass `labels` instead. A single label name. "
            "If both `label` and `labels` are given, `labels` wins and "
            "`label` is ignored. This parameter will be removed after 2026-10."
        ),
    )
    # New canonical form.
    labels: list[str] = Field(
        default_factory=list,
        description="Label names to apply. Replaces the deprecated `label`.",
    )

@mcp.tool(
    # Agent-voice intent: the EFFECT, not the implementation.
    description=(
        "Create a tracker issue so it appears in the team's backlog and can "
        "be claimed by another agent. Returns the new issue's id and url. "
        "Idempotency: NOT idempotent — calling twice creates two issues; "
        "deduplicate on `title` before calling if you may retry."
    ),
)
def create_issue(args: CreateIssueArgs) -> dict:
    # Normalize the deprecated form into the canonical one at the boundary.
    labels = list(args.labels)
    if args.label and not labels:
        labels = [args.label]
        # Self-signaling deprecation: emit telemetry every time the old form
        # is used, so you can SEE whether the window can close (see below).
        _record_deprecated_param_use("create_issue.label")
    issue = _backend_create(args.title, args.body, labels)
    return {"id": issue.id, "url": issue.url, "labels": labels}
```

Why each piece is load-bearing:

- **Both names coexist.** No live agent breaks the day you ship multi-label support.
- **`deprecated=True` is a JSON Schema 2020-12 annotation**, surfaced in the tool's input schema to the host. The deprecation is *in the protocol*, not in a changelog the model will never read.
- **The description states the conflict resolution** (`labels` wins). An agent reasoning over both fields needs the rule, not a surprise.
- **The removal date is named and tied to model-release windows**, not to a sprint. You remove when telemetry says nobody passes the old form, not when you feel like it.
- **`_record_deprecated_param_use` is the gate on removal.** You do not get to close the window on optimism — cross-reference `observability-for-tool-calls.md` for the counter; this sheet says *you must have one before you deprecate*.

### The mirror discipline for return shapes (outputSchema drift)

Return-field evolution is the half everyone forgets, because adding a field feels free — and it is, *if and only if* clients ignore unknown fields (they do) and you do not change the meaning or type of an existing field.

```python
class IssueResult(BaseModel):
    id: str
    url: str
    labels: list[str]
    # ADDITIVE (2026-05): new field. Agents ignoring unknown fields are
    # unaffected; agents wanting it opt in. Safe without a capability bump.
    assignee: str | None = None

# BREAKING — do NOT do this in place:
#   id: int          # was str; every agent parsing id as a string breaks
#   url -> html_url  # rename; every agent reading `url` breaks
# Either of these requires a NEW tool name or an explicit capability flag,
# and the changed outputSchema must be visible at tools/list time.
```

Declare the `outputSchema` so the change is machine-visible. A retyped field with an unchanged `outputSchema` is the return-shape equivalent of the version-number lie.

---

## Capability negotiation — making breaking changes visible

When a change genuinely cannot be made additive, the protocol gives you exactly one honest place to surface it: **capabilities, exchanged at `initialize`**, plus per-feature flags. The host reads them and decides what it is talking to. This is the mechanism behind the Gate clause "non-backward-compatible schema changes bump server capability."

### Worked example — declaring an experimental/breaking capability

```python
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.types as types

server = Server("issue-tracker")

def init_options() -> InitializationOptions:
    return InitializationOptions(
        server_name="issue-tracker",
        server_version="3.0.0",  # code version — the model never reads this
        capabilities=server.get_capabilities(
            notification_options=NotificationOptions(
                tools_changed=True,      # we WILL emit tools/list_changed
                resources_changed=True,
            ),
            # The honest channel for "what contract am I": experimental flags
            # the host can branch on. A breaking surface change gets a NEW
            # flag the host must opt into — old hosts simply never see it.
            experimental_capabilities={
                # v3 issue surface: `labels` is array-typed, `label` removed.
                # Hosts that don't recognize this key get the v2-compatible
                # surface; hosts that do can rely on the new shape.
                "issueSurface": {"version": "2026-11", "labelsArrayTyped": True},
            },
        ),
    )
```

The mechanics that matter:

- **The MCP *revision* you speak (`2025-11-25`) is negotiated separately** from your server's feature capabilities. Do not confuse "I support protocol revision X" with "my issue surface is at contract Y." The first is the wire; the second is your domain.
- **`experimental_capabilities` is the place to flag domain-contract changes** the standard capabilities do not model. An old host that does not know the key gets the compatible behavior; a new host opts in. This is how a breaking change becomes *negotiated* rather than *silent*.
- **`tools_changed=True` is a promise you must keep.** If you declare it and then mutate the tool list without emitting `tools/list_changed`, the host's cached tool descriptions go stale and the model calls a surface that no longer exists.
- **Server `server_version` is for humans and logs.** It is structurally incapable of carrying a contract signal to the model. If your only signal that the surface changed is this string, the change is invisible — that is the defect.

---

## Self-signaling deprecation

A deprecation that lives only in your changelog is not a deprecation; it is a plan. The surface must signal it three ways, so that *the agent reading the surface* and *the host caching it* and *you watching telemetry* all see it:

1. **In the schema** — `deprecated: true` (JSON Schema 2020-12) on the parameter/field, surfaced in `tools/list`. Machine-visible to the host.
2. **In the description** — agent-voice text that names the replacement and the resolution rule ("pass `labels`; if both given, `labels` wins"). The model reads this on every turn.
3. **In telemetry** — a counter incremented every time the deprecated form is actually used, so removal is driven by evidence, not by calendar optimism.

```python
def _record_deprecated_param_use(param_path: str) -> None:
    # Emit a structured event so a dashboard can answer:
    # "can we close the deprecation window on create_issue.label yet?"
    metrics.increment(
        "mcp.deprecated_param.used",
        tags={"param": param_path, "server_version": "3.0.0"},
    )
```

**The removal gate:** you may remove a deprecated parameter only when (a) the deprecation has been live for at least your declared window (measured in *model-release cycles*, because interpretation drift is the risk you are managing), and (b) telemetry shows the usage counter at or near zero across that window. Removing on calendar alone is how you break the one pinned host you forgot about.

When you do remove, it is a **breaking change** — `tools/list_changed`, updated `outputSchema`/input schema, and a capability signal if any host could still be relying on the old shape.

---

## Interpretation drift — surviving the model you did not validate against

Surface drift is your fault and your fix. Interpretation drift is nobody's fault and there is no fix — only detection and exposure-bounding. A new model release will, with certainty, re-read some of your unchanged descriptions differently. You cannot stop this. You can:

### 1. Write descriptions that are robust to re-reading

The descriptions most vulnerable to re-interpretation are the ambiguous ones — the ones that worked only because the validated model happened to resolve the ambiguity your way.

- **State the effect, not the implementation** (Gate: agent-voice intent). "Marks the issue in-progress so other agents do not claim it" survives re-reading; "updates the status column" invites a new model to decide what "status" means.
- **Name units, formats, and bounds in the schema, not the prose.** `Field(ge=0, le=100, description="percent")` is harder to re-interpret than "a percentage." Constraints are enforced; prose is interpreted.
- **Enumerate where you can.** A free-text `priority: str` lets each model release invent its own vocabulary; `priority: Literal["low","normal","high"]` does not drift.
- **Do not rely on the model inferring a default.** State it. "If omitted, defaults to the current repository" beats hoping every future model infers the same default.

### 2. Pin and grow a model-validation matrix

You validate against models. New models ship. The matrix is the artifact that turns "we tested it once" into "we know our exposure."

```yaml
# tests/model-validation-matrix.yaml — checked into the repo, updated per release
surface_revision: "2026-11"          # YOUR contract version (the experimental flag)
mcp_protocol_revision: "2025-11-25"
golden_conversations: tests/golden/   # see testing-mcp-servers.md
validated_against:
  - model_family: frontier-general    # capability-tier vocab, not hardcoded ids
    release: "2026-05"
    status: pass
    last_run: "2026-06-10"
  - model_family: frontier-reasoning
    release: "2026-05"
    status: pass
    last_run: "2026-06-10"
  - model_family: fast-cheap
    release: "2026-03"
    status: pass-with-notes           # mis-fills `labels` ~3% — see note
    last_run: "2026-06-10"
  - model_family: frontier-general
    release: "2026-09"                # UPCOMING — exposure UNKNOWN
    status: not-yet-validated         # this cell is your drift exposure
    last_run: null
```

The cell that reads `not-yet-validated` **is** your version-drift exposure. A drift exposure assessment is, concretely: *which golden conversations have not been replayed against the model release that is about to become the default?* If the answer is "all of them," you are shipping a release candidate every time the provider does.

### 3. Replay golden conversations against the new model before it becomes the default

This is the operational counter to interpretation drift and the bridge to `testing-mcp-servers.md`. When a new model release is announced:

1. Replay every golden conversation against the new release.
2. Diff the *tool-call sequences and arguments* the new model produces against the recorded ones.
3. Any tool the new model stops calling, starts calling, or calls with different argument shapes is a drift finding — triage it before the release lands, not after an incident.

A golden conversation that only checks the *final answer* will not catch drift; the model can reach the same answer by calling your tools wrong. Diff the **tool calls**, not just the outcome.

---

## Red flags — STOP

If any of these is true, stop and fix it before shipping. Each is a Gate failure.

- **"It's a minor version bump."** A semver bump on the server *package* carries no signal to the model. If the contract the model reads changed, the version number is lying. STOP and classify the change against the table.
- **You renamed a parameter and the only record is the changelog.** The model does not read changelogs. STOP: keep the old name as a deprecated alias, signal in-schema, set a window.
- **You retyped a return field (`str`→`int`, renamed `url`→`html_url`) "because it's cleaner."** Every agent parsing that field just broke invisibly. STOP: new field additive, or new tool, with `outputSchema` visible at list time.
- **You added a required parameter with no default.** Every existing call is now invalid. STOP: make it optional-with-default or ship a new tool.
- **You changed what a tool *does* but kept its name and schema.** This is the worst drift — invisible to schema diffing entirely. STOP: a behavior change is a breaking change; new tool or explicit capability.
- **You are about to remove a deprecated parameter and have no usage telemetry.** You are removing on hope. STOP: instrument first, watch a full window, then remove.
- **Your surface has never been replayed against the model release about to become default.** Your drift exposure is the entire surface. STOP: run the golden conversations against the new release first.
- **Your server still emits draft-07 schema idioms** (`definitions`, boolean `exclusiveMinimum`) while declaring the 2025-11-25 revision. STOP: dialect mismatch silently breaks structured-output validation under hosts that enforce 2020-12.
- **You bumped the surface but did not emit `tools/list_changed`** (having declared `tools_changed=True`). The host's cache is now stale and the model is calling a surface that no longer exists. STOP and emit the notification.

---

## Common mistakes

- **Versioning the code instead of the contract.** The server's package version is for humans and logs; it cannot reach the model. The contract the model reads is versioned by capabilities, schema annotations, and tool names — nowhere else.
- **Treating additive and breaking as a judgment call.** They are a classification with a table (above). "I think it's backward compatible" is not the same as "it adds an optional field and changes nothing existing." Run the table.
- **Deprecating in prose only.** A `description` that says "deprecated" without the `deprecated: true` annotation and without a usage counter is a deprecation the host cannot detect and you cannot safely close.
- **Removing a deprecated param on a calendar.** Interpretation drift means a pinned or slow-moving host may still use the old form long after you stopped thinking about it. Remove on telemetry, measured across model-release windows.
- **Validating once and assuming forever.** "It worked when we shipped" says nothing about the model release that became the default last week. The validation matrix is a living artifact or it is theatre.
- **Diffing final answers instead of tool calls** when replaying against a new model. Same answer, wrong tool usage, is the drift you most need to catch.
- **Confusing MCP protocol revision with your domain contract version.** "We're on 2025-11-25" answers a wire question, not "is my issue surface v2 or v3." Track both.
- **Letting `outputSchema` rot.** The return shape drifts as silently as the input shape; a retyped return field with an unchanged `outputSchema` is the same lie as the version number.

---

## Counters to the rationalizations used to skip this

- *"It's internal, only our own agent calls it."* Your own agent runs on a model the provider upgrades without asking you. Interpretation drift does not care whose agent it is.
- *"We'll bump the version, that's enough."* The version is invisible to the model. Re-read the Red flags. The version number is the canonical way this discipline fails.
- *"Adding a required field is fine, the agent will figure it out."* The agent cannot figure out a field it does not know exists, and every existing call is now a validation error. Optional-with-default or a new tool.
- *"We can clean up the deprecated param next sprint."* You can — *after* telemetry across a model-release window says zero usage. Without that data, "cleanup" is "break the host we forgot about."
- *"The new model is basically the same, no need to re-validate."* "Basically the same" is exactly the regime where interpretation drift hides: the surface looks unchanged, so nobody checks, so the one re-read description that flipped meaning ships an incident. Replay the goldens.
- *"Renaming for clarity is worth a small break."* Clarity is free if you add the clear name and deprecate the old one. The break is never necessary; it is just faster for you and slower for everyone downstream.

---

## Critic checklist for this sheet's concern

When auditing a surface for version/drift health, each finding gets **severity + evidence**:

- **Blocker:** a breaking change shipped behind a server version bump with no capability/schema signal (cite the diff and the version string). A behavior change under an unchanged name+schema. A removed-or-retyped return field with an unchanged `outputSchema`.
- **Major:** a renamed/retyped parameter with no deprecated alias; a required parameter added without default; a deprecation with no usage telemetry; a surface with no model-validation matrix or with the current default release in `not-yet-validated`.
- **Minor:** deprecation signaled in prose but missing the `deprecated` annotation; ambiguous descriptions (no units/enums/bounds) that invite re-interpretation; draft-07 dialect idioms under a 2025-11-25 declaration.
- **Nit:** changelog and in-schema deprecation wording disagree on the removal date; validation-matrix `last_run` is stale.

Evidence is the schema diff, the capability declaration excerpt, the `outputSchema`, the validation-matrix cell, or the golden-conversation tool-call diff. A "looks backward compatible" verdict with no diff in hand is not a finding — it is the rubber stamp this pack exists to prevent.

## Cross-references

- `error-envelopes-and-recovery.md` — input-validation failures during a deprecation window are Tool Execution Errors with a recovery hint that names the new parameter.
- `observability-for-tool-calls.md` — the deprecated-param usage counter and the per-release tool-call telemetry that drive the removal gate and the drift assessment.
- `testing-mcp-servers.md` — golden-conversation capture and the replay-against-new-model procedure that operationalizes interpretation-drift detection.
- `tool-api-design.md` — descriptions written to survive re-reading start at design time, not at the first drift incident.
- `composition-and-namespaces.md` — capability negotiation across multiple servers in one agent context.
