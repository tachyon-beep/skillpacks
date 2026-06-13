---
name: composition-and-namespaces
description: Use when your MCP server runs alongside other MCP servers in one agent's context and tool names collide, when two servers expose `search`/`get`/`list`/`create` and the model picks the wrong one, when a generic tool name shadows another server's tool, when capability negotiation behaves differently per connection, when adding a server silently changes how the model uses yours, when an aggregating gateway flattens several servers into one tool list, or when you cannot predict which servers will be mounted beside you. Covers namespacing, prefix discipline, name-collision resolution, per-connection capability negotiation, aggregation/proxy gateways, and how your surface interacts with surfaces you do not control.
---

# Composition and Namespaces

## Overview

**Your MCP server is never the only one in the room. The model sees a single flat tool list assembled from every server the host has mounted — your eight tools, plus the filesystem server's `read_file`, plus the GitHub server's `search`, plus whatever the user installed last week. You do not own that namespace; you rent a slice of it, and the rent is paid in disambiguation.**

The protocol gives each server its own connection and its own capability negotiation, but it does **not** give each server its own tool namespace. When the host advertises tools to the model, it concatenates the `tools/list` results of every connected server into one array. The model chooses by `name` and `description` alone. There is no server-of-origin field the model reasons over by default — the host decides whether and how to disambiguate, and many hosts do nothing. So a tool you named `search` is, from the model's seat, indistinguishable from every other `search` in the context except by description text. This sheet is about engineering your surface so it survives that flattening: so it is not shadowed, does not shadow others, does not get mis-selected, and degrades predictably when it is mounted beside servers you have never heard of.

Two readings of the same material. The **architect** asks: what do I name my tools, how do I prefix them, what capabilities do I declare, so that an agent with N other servers mounted still reaches for the right tool? The **critic** asks: given this surface and a realistic co-mounted set, which of these tools will be shadowed, which will shadow someone else, which generic name is a collision waiting for the median install, and where does capability negotiation assume a connection topology the deployment does not guarantee? The architect builds for an unknown neighbourhood. The critic finds the collision the architect assumed away.

## When to Use

Use this sheet when:

- Your server will run in a host (Claude Desktop, an IDE agent, a custom client) that mounts more than one MCP server at a time — which is the default case, not the exception.
- You are about to name a tool `search`, `query`, `get`, `list`, `create`, `update`, `delete`, `run`, `execute`, `status`, or any other word that a dozen other servers also use.
- An agent is calling the wrong server's tool because two tools look identical from the description alone.
- You are building an **aggregating gateway / proxy** that mounts several downstream MCP servers and re-exposes their tools as one surface — now *you* own the namespace and the collision policy.
- Capability negotiation behaves differently across deployments and you suspect it is because the connection topology (one host ↔ one server vs. host ↔ gateway ↔ many servers) is not what you assumed.
- Adding a new server to a working setup silently degraded the agent's use of *your* server, and you need to reason about cross-server interference.

Do not use this sheet for:

- Naming and granularity *within your own server's* surface in isolation — that is [tool-api-design.md](tool-api-design.md). This sheet assumes the single-server naming is already good and asks what happens when it meets other servers.
- Choosing tools vs. resources vs. prompts — that is [mcp-primitive-selection.md](mcp-primitive-selection.md). Collisions affect all named primitives, but the selection decision is upstream of this sheet.
- The wire mechanics of a single connection (framing, reconnect, backpressure) — that is [transport-reliability.md](transport-reliability.md). Composition is about *many* connections; one connection's reliability is its own concern.
- Per-user / per-project authorization — that is [authentication-and-trust.md](authentication-and-trust.md). A gateway that aggregates servers must also reconcile their trust boundaries, but the trust discipline lives there.

## Core Principle

> The tool namespace is shared and flat; capability negotiation is private and per-connection. Design your *names* as if every other server is hostile to disambiguation, and design your *capabilities* as if you are the only server — because that is exactly the asymmetry the protocol gives you.

A name is read by the model in a context you do not control. A capability is negotiated on a connection only you and your host share. Conflating the two — assuming your capability declaration protects your name, or assuming your name is private the way your capability is — is the root error this sheet exists to prevent.

## The Flattening: What the Model Actually Sees

When three servers are mounted, the host runs `tools/list` against each and presents the model with a merged catalog. A concrete merge:

```jsonc
// What the MODEL sees — origin server is NOT a field the model selects on by default.
// (Some hosts inject origin into the description or a display name; many do not.)
[
  { "name": "search",      "description": "Search the codebase by query string." },        // ← your server
  { "name": "search",      "description": "Search GitHub issues and PRs." },                 // ← github server  COLLISION
  { "name": "read_file",   "description": "Read a file from the local filesystem." },        // ← filesystem server
  { "name": "get",         "description": "Fetch a record by id." },                          // ← your server   GENERIC
  { "name": "execute",     "description": "Run a shell command." },                           // ← shell server  DANGEROUS-GENERIC
  { "name": "list_issues", "description": "List issues in the active repository." }           // ← github server
]
```

Three failure modes are visible here, and they are the failures this sheet closes:

1. **Hard collision** — two tools share an exact `name`. Per the current protocol revision, tool names within a single server must be unique, but **nothing in the protocol guarantees uniqueness across servers**. The host must resolve it. Some hosts deduplicate by dropping the second, some by prefixing, some by erroring, some by silently letting the model pick — and *you cannot rely on which*. A dropped tool is your tool that simply does not exist for that session.
2. **Soft collision (mis-selection)** — distinct names, overlapping meaning. `search` (codebase) vs. `search` (GitHub) is a hard collision; but `find_code` vs. `search` vs. `query_repo` is a soft collision: the model has to disambiguate three plausible tools by description alone, and it will sometimes pick wrong. Soft collisions are more common and harder to detect than hard ones.
3. **Generic-name shadowing** — a name so generic (`get`, `run`, `status`) that it attracts calls meant for any server. The model reaches for `get` to fetch a GitHub record because `get` is "close enough", and your server receives a request shaped for a neighbour's data model.

## Namespacing: Prefix Discipline

You do not have a protocol-level namespace. You manufacture one with naming convention. The discipline:

### Rule 1 — Prefix every tool with a stable, server-specific token

Pick a short, unambiguous, lowercase token that identifies *your domain* (not your company, not your product version — your domain as the agent understands it). Prefix every tool name with it.

```jsonc
// BAD — generic names that collide and shadow
{ "name": "search" }
{ "name": "get" }
{ "name": "create" }

// GOOD — namespaced by domain token; collision-resistant, self-describing in the flat list
{ "name": "tracker_search_issues" }
{ "name": "tracker_get_issue" }
{ "name": "tracker_create_issue" }
```

The prefix is not decoration. In the flattened list, `tracker_search_issues` cannot be confused with `github_search_issues` or `docs_search`, and the model reads the prefix as a domain hint when it disambiguates. The cost is verbosity; the benefit is that your surface is legible in a context you do not control. Verbosity is the correct trade — the model has no other disambiguation signal you can rely on.

### Rule 2 — The prefix must be stable across versions and deployments

If the prefix changes between server versions, golden conversations break and the model's learned associations (from in-context examples, from the system prompt) go stale. The prefix is part of your public contract; treat a prefix change as a non-backward-compatible schema change that **bumps server capability** (see [schema-versioning-and-drift.md](schema-versioning-and-drift.md)). Do not derive the prefix from anything mutable (a runtime config value, a tenant id, the connection's project) — derive it from the server's identity.

### Rule 3 — Do not prefix with a word the model treats as a verb

`run_*`, `do_*`, `execute_*` as a *prefix* re-introduces the generic-verb shadowing you were trying to escape. The prefix should be a **noun describing the domain** (`tracker_`, `vault_`, `wiki_`), and the verb lives after it (`tracker_close_issue`). Domain-noun-then-verb reads cleanly in agent-voice and keeps the collision-resistant token first.

### Rule 4 — The intent statement carries the namespace too

The Consistency Gate requires every tool to have an agent-voice intent statement. In a multi-server context, the intent must also state *which world the tool acts on*, so the model can disambiguate two tools whose names are similar. Not "marks an issue in progress" but "marks an issue **in the project tracker** as in-progress so other agents do not claim it" — the bolded scope is the soft-collision defence.

## Capability Negotiation Across Servers

Capability negotiation is **per-connection and private**. Each server independently negotiates capabilities with the host at `initialize`; one server's declared capabilities are invisible to another server and irrelevant to it. This is a feature (isolation) and a trap (false assumptions).

What you may rely on:

- Your `initialize` exchange is yours alone. The host negotiates your tools/resources/prompts capability, your logging capability, your declared protocol revision, independently of every other server.
- The host is responsible for honouring each server's negotiated capabilities separately. A server that declared `tools.listChanged` gets list-change notifications; a co-mounted server that did not, does not.

What you must NOT assume:

- **You cannot see, query, or coordinate with another server's capabilities.** There is no cross-server capability discovery in the protocol. If your tool's behaviour depends on another server being present (e.g., "I emit resource links that the filesystem server resolves"), that dependency is **invisible to the protocol and unenforced**. The filesystem server may be absent, a different version, or a different implementation entirely. Resource links you emit are URIs; whether anything resolves them is the host's business, not a contract you can rely on.
- **You cannot assume the host advertises servers consistently to all consumers.** A host may mount your server for one conversation and not another, may mount it behind a gateway, may apply per-session tool filtering. Your capability negotiation says nothing about *whether the model can currently see your tools* — only about what you offered.
- **Protocol-revision skew is per-connection.** Your server may negotiate revision `2025-11-25` with the host while a co-mounted server negotiated `2025-06-18`. The model sees both surfaces. Features you rely on from the newer revision (`structuredContent` + `outputSchema`, resource links, tool-calling in sampling, Tasks) are available *on your connection* if the host supports them, regardless of what the neighbour negotiated — but do not assume the neighbour's tools behave the way yours do.

The discipline: **declare your capabilities as if you are the only server (because for negotiation purposes you are), and design your tool semantics to be self-contained (because you cannot depend on a neighbour you cannot see).** If your surface genuinely needs another server, that is a deployment contract enforced by the host configuration, not by MCP — document it as an external dependency and fail with a `retry-with-changes` envelope (see [error-envelopes-and-recovery.md](error-envelopes-and-recovery.md)) when the expected neighbour artifact is missing, rather than a fatal crash.

## Aggregating Gateways and Proxies

A common deployment puts a **gateway** between the host and several downstream servers: the host connects to one endpoint, the gateway fans out to N backends and re-exposes their union as one surface. If you build a gateway, **you now own the shared namespace** and inherit the collision policy as a first-class design decision. The host's "do something or nothing" non-determinism becomes *your* explicit choice.

A gateway's collision policy, from worst to best:

| Policy | What it does | Verdict |
| --- | --- | --- |
| **Pass-through** | Forward every downstream tool name unchanged | Hard collisions reach the model; mis-selection guaranteed. Only acceptable if you can prove the downstream sets are disjoint, which you usually cannot. |
| **Drop-on-collision** | Keep the first server's `search`, drop the second's | Silent capability loss — a downstream tool vanishes. Worst kind of bug: works in testing with two servers, breaks in production with three. |
| **Error-on-collision** | Refuse to start if downstream names collide | Honest but brittle; one bad downstream takes the gateway down. Acceptable for a closed, curated backend set. |
| **Prefix-on-mount** | Rewrite each downstream tool to `{server}_{tool}` | **Recommended.** Deterministic, collision-free, self-describing. The cost is that downstream golden conversations referencing bare names break at the gateway — mitigate by making the prefix part of the gateway's published contract. |

The gateway must also **reconcile capability negotiation**: it negotiates one revision with the host but N revisions with backends. It is responsible for *down-converting* — never advertising a capability to the host that some backend cannot honour, and translating newer-revision features (e.g., `structuredContent`) into a form the host's negotiated revision supports. A gateway that advertises `2025-11-25` to the host but proxies a `2025-06-18` backend that returns no `structuredContent` is lying about the merged surface, and the model will be surprised. The gateway's capability set is the **intersection** of what it can faithfully deliver, not the union of what backends offer.

## Common Mistakes

- **Naming a tool `search` / `get` / `list` / `query` with no prefix.** It is fine in isolation and a collision in every real deployment. The unit test passes; the median install fails. (Critic severity: **major** when the name is a known-common token like `search`; **blocker** when it is `execute`/`run` and could capture calls meant for a shell or code-execution server.)
- **Assuming the host disambiguates by server-of-origin.** Some do, most do not, and you cannot detect which from the server side. Design as if the model sees only `name` + `description`.
- **Prefixing with the product name instead of the domain.** `acmecorp_search` is collision-resistant but tells the model nothing about *what world it searches*. The prefix is a disambiguation signal for the model, not branding — make it a domain noun.
- **Letting the prefix vary by tenant, project, or runtime config.** A prefix derived from mutable state means the tool name changes between sessions, which breaks golden conversations and any in-context examples. Derive the prefix from server identity, treat changes as capability bumps.
- **Depending on a co-mounted server's behaviour with no fallback.** "My resource links resolve because the filesystem server is always there" is an unenforced assumption. The neighbour may be absent or a different implementation. Emit self-contained results or fail with a recovery hint, never crash on a missing neighbour.
- **Building a gateway with drop-on-collision (often the default).** It silently deletes a downstream tool the moment two backends collide, and the failure only appears at the scale you did not test. Use prefix-on-mount and make the prefix contractual.
- **A gateway advertising the union of backend capabilities.** It must advertise the *intersection it can faithfully deliver*; over-advertising means the model expects features (structured output, resource links, list-change notifications) that some backend silently does not provide.
- **No golden conversation that mounts a realistic neighbour set.** A regression test that runs your server alone never exercises collision or mis-selection. The Consistency Gate's golden-conversation requirement, in composition context, means *at least one golden conversation runs your server beside the servers it will realistically ship next to* — otherwise the collision is untested by construction.

## Example 1 — Collision-resistant naming and a self-describing surface

A project tracker server mounted beside the official `@modelcontextprotocol/server-github` and `@modelcontextprotocol/server-filesystem`. The github server already exposes `search_issues` and `create_issue`; a naive tracker would collide.

```python
# tracker_server.py — collision-resistant by construction.
# Domain-noun prefix ("tracker_"); intent statement names the world it acts on.

TOOLS = [
    {
        "name": "tracker_search_issues",
        # Agent-voice intent that names the WORLD, defending against soft collision
        # with github's search_issues. The scope phrase is the disambiguator.
        "description": (
            "Search issues in the internal project tracker (NOT GitHub). "
            "Returns a bounded summary list (id, title, state) — at most 50 rows; "
            "use tracker_get_issue for the full record. "
            "Idempotent and read-only; safe to call concurrently."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text query over title and body."},
                "state": {"type": "string", "enum": ["open", "in_progress", "closed"]},
                "cursor": {"type": "string", "description": "Opaque pagination cursor from a prior call."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        # 2025-11-25: declare the structured shape so the host can pass it through faithfully.
        "outputSchema": {
            "type": "object",
            "properties": {
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "state": {"type": "string"},
                        },
                        "required": ["id", "title", "state"],
                    },
                },
                "next_cursor": {"type": ["string", "null"]},
            },
            "required": ["issues", "next_cursor"],
        },
    },
    {
        "name": "tracker_get_issue",
        "description": (
            "Fetch the full record of one issue in the internal project tracker by id. "
            "Read-only, idempotent, concurrency-safe. May be truncated only in the body field, "
            "which is then paginated via body_cursor."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "body_cursor": {"type": ["string", "null"]}},
            "required": ["id"],
            "additionalProperties": False,
        },
    },
]
```

Why this survives the flattened list: `tracker_search_issues` cannot hard-collide with `search_issues` (github) or `read_file` (filesystem); the `(NOT GitHub)` scope phrase defends against the soft collision the model would otherwise make when both surfaces are present; the bounded return shape and idempotency note satisfy the Consistency Gate even when read in a 30-tool merged catalog where the model has little budget per tool description.

## Example 2 — A gateway with explicit prefix-on-mount and capability intersection

A gateway mounts three downstream servers and re-exposes them as one surface. It rewrites names to avoid collisions and down-converts capabilities to what it can faithfully deliver.

```python
# gateway.py — own the namespace; make the collision policy explicit.
import re

class Downstream:
    def __init__(self, mount_token: str, client, negotiated_revision: str):
        self.mount_token = mount_token        # stable, contractual prefix e.g. "gh", "fs", "tracker"
        self.client = client
        self.revision = negotiated_revision   # what THIS backend negotiated

def _sanitize(token: str) -> str:
    # Prefix must be a stable lowercase identifier; reject anything that could be a verb-y collision.
    if not re.fullmatch(r"[a-z][a-z0-9]{1,15}", token):
        raise ValueError(f"unsafe mount token: {token!r}")
    return token

def build_merged_tool_list(downstreams: list[Downstream]) -> tuple[list[dict], dict]:
    """Returns (merged_tools, gateway_capabilities).

    Collision policy: PREFIX-ON-MOUNT. Every downstream tool is renamed
    {mount_token}_{original}. Deterministic, collision-free, self-describing.
    """
    merged: list[dict] = []
    seen: set[str] = set()
    for ds in downstreams:
        prefix = _sanitize(ds.mount_token)
        for tool in ds.client.list_tools():
            new_name = f"{prefix}_{tool['name']}"
            if new_name in seen:
                # Two backends with the same mount token is a config error, not a runtime guess.
                raise RuntimeError(f"mount-token collision on {new_name}: tokens must be unique")
            seen.add(new_name)
            rewritten = dict(tool, name=new_name)
            # Make origin legible in the description too (the model reads this, not a hidden field).
            rewritten["description"] = f"[{prefix}] {tool['description']}"
            merged.append(rewritten)

    # Capability negotiation = INTERSECTION of what every backend can faithfully deliver,
    # never the union. If any backend predates structuredContent, do not advertise it upstream.
    supports_structured = all(ds.revision >= "2025-11-25" for ds in downstreams)
    gateway_caps = {
        "tools": {"listChanged": all(_backend_supports_list_changed(ds) for ds in downstreams)},
        # only claim structured output if EVERY backend can produce it; else down-convert
        "experimental": {"structuredContent": supports_structured},
    }
    return merged, gateway_caps

def route_call(tool_name: str, args: dict, downstreams: list[Downstream]):
    """Reverse the prefix to find the owning backend. Unprefixed/unknown -> fatal-but-recoverable."""
    prefix, _, original = tool_name.partition("_")
    for ds in downstreams:
        if ds.mount_token == prefix:
            return ds.client.call_tool(original, args)
    # Recovery hint, not a stack trace (see error-envelopes-and-recovery.md).
    return {
        "isError": True,
        "error_class": "retry-with-changes",
        "message": (
            f"No mounted server owns prefix '{prefix}'. "
            f"Available prefixes: {sorted(ds.mount_token for ds in downstreams)}."
        ),
    }

def _backend_supports_list_changed(ds: Downstream) -> bool:
    return ds.client.server_capabilities().get("tools", {}).get("listChanged", False)
```

Why this is the correct gateway shape: the namespace is owned and deterministic (prefix-on-mount, never drop-on-collision); mount-token uniqueness is a *config-time* error rather than a runtime guess; the merged descriptions carry the origin so the model can disambiguate; and capabilities are the **intersection** the gateway can honour — the gateway never advertises `structuredContent` upstream unless every backend can produce it, so the model is never surprised by a backend that returns unstructured text where the merged surface promised structure. The routing error is a `retry-with-changes` envelope with the valid prefixes, so an agent that guessed wrong can self-correct instead of giving up.

## Critic Checklist (composition-specific)

When auditing a server *for the neighbourhood it will live in*, every finding carries severity (**blocker / major / minor / nit**) and evidence (the tool name, the colliding neighbour, the merged-list excerpt, the conversation fragment where mis-selection occurred):

- **Hard-collision scan.** Enumerate the realistic co-mounted set (filesystem, github, shell, plus domain-specific). For each of your tool names, is there an exact match? (`search` vs. github `search` → blocker if your tool has side effects, major if read-only.) Evidence: the merged `tools/list` excerpt.
- **Soft-collision scan.** For each pair (your tool, neighbour tool) with overlapping meaning, would the model disambiguate from description alone? Does your intent statement name the world it acts on? Evidence: a conversation fragment where the model picked the neighbour.
- **Generic-name scan.** Any tool whose name is a bare verb or bare noun (`get`, `run`, `status`, `query`)? (Bare `execute`/`run` → blocker: it can capture shell/code-exec calls. Bare `get` → major.)
- **Prefix stability.** Is the prefix derived from server identity (stable) or from mutable runtime state (defect)? Evidence: where the prefix is computed.
- **Neighbour-dependency scan.** Does any tool's documented behaviour depend on a co-mounted server (resource-link resolution, hand-off)? Is that dependency enforced or assumed? Does it degrade with a recovery hint or crash? (Unguarded crash on missing neighbour → major.)
- **Gateway capability honesty.** If aggregating: is the upstream-advertised capability set the intersection backends can deliver, or the over-promising union? (Union → blocker: the model will be surprised mid-conversation.)
- **Composition golden conversation.** Is there at least one regression test that mounts your server beside its realistic neighbours? (Absent → major: collision and mis-selection are untested by construction.)

A clean composition audit that did not enumerate the neighbour set is a skipped check, not a pass — the same rubber-stamp failure the router warns about, applied to the namespace you do not own.
