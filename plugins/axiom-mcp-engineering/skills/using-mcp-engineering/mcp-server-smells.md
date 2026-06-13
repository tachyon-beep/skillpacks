---
name: mcp-server-smells
description: Use when an MCP server's tools confuse or get misused by an agent, when CRUD-named tools mirror database tables, when errors arrive as stack traces or "internal server error", when an agent cannot fill a required parameter, when a tool result blows the context budget or gets truncated, when retries double-execute a side effect, when a schema changed without a capability bump, when tool names collide with another server, or when you cannot decide whether something should be a tool or a resource — the catalogued anti-pattern smell list and critic checklist for MCP tool surfaces.
---

# MCP Server Smells

**A smell is not a bug — it is evidence that an agent will mis-handle this surface, and it demands verification, not a fix-on-sight.**

Every entry below is a structural pattern on an MCP server that correlates with an agent failing in production: calling the wrong tool, retrying into a duplicate side effect, running out of context after one call, giving up because an error told it nothing. None of these are caught by general API review, because general API review assumes a human reads the docs once and writes client code deliberately. An MCP surface is read by a model on every turn, with no human to interpret an ambiguous description or recover from a 500 with a traceback in the body. That asymmetry is what this catalog audits.

This is the **critic checklist** referenced by the router's Consistency Gate. Run it as a structured pass, not a vibe-check. "No smells found" without enumerating all ten is a skipped check, and the router treats a skipped check as a blocking failure. The architect reads the same catalog constructively — as the list of mistakes to design against before shipping a tool. Same corpus, different epistemics: the architect asks "which of these am I about to commit?", the critic asks "which of these is already here, and what is the evidence?"

A surface with three smells that are all justified by context is better than one that looks clean because nobody ran the list.

---

## How to Use This Catalog

Run the catalog as a pass over the tool inventory (and resource/prompt list). For each tool, and for each of the ten smells:

1. **Symptom** — is the structural pattern present? Read the tool description as a model would, with no source access.
2. **Why it hurts** — confirm the failure mode is reachable for *this* surface and *this* workflow, not just theoretically.
3. **False-positive check** — is the pattern explained by a legitimate design choice? Some of these smells have sharp, common false positives.
4. **Fix** — if the smell is real, apply the remediation; if it is a false positive, record *why* and close the finding.

Critic findings carry **severity** (`blocker / major / minor / nit`) and **evidence** (the specific tool name, the parameter, a return-shape excerpt, a conversation fragment, or a retry trace). A finding without evidence is a vibe and does not ship. Use this format when handing findings back to a producer:

```
Smell:          [smell name from this catalog]
Tool:           [tool name, or "surface" if cross-tool]
Severity:       blocker | major | minor | nit
Symptom seen:   [what fired — the description text, the schema, the trace]
Evidence:       [exact excerpt / conversation fragment / retry log]
False-positive: [yes/no — if yes, state why and close]
Fix:            [specific remediation; see entry]
Gate clause:    [which Consistency Gate clause this violates, if any]
```

This catalog is authoritative for smell naming. Severity is contextual — a `return-shape-that-blows-budget` on a tool the agent calls once per session is a `minor`; the same smell on a tool called in a loop is a `blocker`.

---

## The Ten Smells

---

### 1. Overlapping-Tools

**Symptom.** Two or more tools whose descriptions an LLM cannot reliably tell apart, so it picks one by surface cues (name length, keyword overlap with the user's phrasing) rather than by fit. Signal: `search_issues` and `query_issues` and `find_issues` all exist; or `update_issue` and `set_issue_status` both can change status. Tighter signal: you cannot write a one-sentence rule for *when the agent should prefer A over B* that does not itself require reading the source.

**Why it hurts.** Tool selection is the agent's first decision and it is made from descriptions alone. When two tools occupy the same intent-space, the model distributes its calls across them non-deterministically and differently across model versions. You get intermittent bugs that reproduce on one model and vanish on the next, and your golden conversations (Gate: one per tool) drift because the "right" tool keeps changing.

**False-positive check.** Tools may share a *noun* without overlapping if their *verbs* (effects) are genuinely distinct and the descriptions say so in agent-voice: `claim_issue` (takes an exclusive lease) vs `comment_on_issue` (appends, no lease) both touch issues but never compete. Overlap is about competing *intents*, not shared nouns. Also legitimate: a deliberately broad `search` plus a narrow `get_by_id` — different cardinality, clearly separable.

**Fix.** Collapse to one tool with a parameter that selects the variant, *or* sharpen each description to state the discriminating effect and a "use this when / not when" clause. Prefer collapsing when the variants share a return shape; prefer sharpening when they differ. Verify the fix with a selection eval: present the merged/sharpened surface to the model across several phrasings and confirm it picks correctly. Violates the Gate's *agent-voice intent statement* clause when neither tool's intent disambiguates it from its twin.

---

### 2. Tool-As-CRUD-Mirror

**Symptom.** The tool set is a one-to-one projection of the database tables or the REST resource model: `create_X`, `get_X`, `update_X`, `delete_X` for every entity. Signal: the tool inventory has the shape of an ORM, and the verbs are data-model verbs (create/update) rather than workflow verbs (claim/release/close). Tighter signal: to accomplish one thing a user asks for, the agent must orchestrate three or four CRUD calls in a specific order it has to infer.

**Why it hurts.** Agents reason in workflow terms ("mark this in progress so nobody else takes it"), not in data terms ("set the `status` column to `2` and the `assignee_id` to my id"). A CRUD mirror forces the model to reconstruct the workflow from primitives every turn, multiplies the number of calls (each a retry/error opportunity), and exposes invariants the agent can violate — a partial `update_X` that leaves the row in a state no workflow ever intended. It also defeats idempotency: a workflow verb can declare a clean guarantee; a generic `update` cannot.

**Why it hurts (concrete).** With `get_issue` → `update_issue(status=in_progress, assignee=me)`, two agents read the same unassigned issue, both write themselves as assignee, and both "win" — the classic read-modify-write race that a single `claim_issue` with a lease would have made impossible.

**False-positive check.** A CRUD-shaped surface is legitimate when the server is genuinely a thin data layer with no workflow semantics to model (e.g., a notes store the agent treats as scratch memory), or when the entity truly has independent, orthogonal fields each meaningful to set alone. The smell fires when there *is* a workflow and the CRUD surface hides it.

**Fix.** Model the verbs the agent actually wants. Replace `update_issue` with `claim_issue`, `release_claim`, `close_issue`, `reassign_issue` — each with a single intent, a single idempotency guarantee, and a single concurrency contract. Keep `get_issue` for reads. Derive the verb list from the workflow (see `tool-api-design.md`), not from the schema. Violates the Gate's *idempotency guarantee* and *concurrency contract* clauses, because a generic mutator can declare neither.

---

### 3. Error-As-Stack-Trace

**Symptom.** A tool failure returns a Python/Node traceback, a raw exception message, or `{"error": "Internal Server Error"}` in the body. Signal: the error payload contains a file path, a line number, or a class name like `KeyError`. Tighter signal: the error does not tell the agent which of three things to do — retry as-is, retry with changed arguments, or give up and surface to the user.

**Why it hurts.** An MCP error is part of the agent's chain-of-thought. The model reads it and decides what to do next. A traceback says nothing actionable: the agent cannot tell whether the failure is transient (retry), caused by its own bad input (fix and retry), or unrecoverable (stop). So it does the worst thing — retries blindly, often amplifying a side effect (see smell 6), or abandons a task that one corrected argument would have completed. Tracebacks also leak internals (paths, dependency names, SQL) into the model's context, which is both a security smell and a token tax.

**False-positive check.** Logging a full traceback *server-side* is correct and required. The smell is specifically about what crosses the wire to the agent. A terse internal detail in a structured `debug` field that the host strips before the model sees it is acceptable; a traceback in the model-visible `content` is not. Note also that in the current spec, *input-validation* failures should be returned as **Tool Execution Errors** (in the tool result, `isError: true`) so the model can self-correct — not as JSON-RPC protocol errors. Returning a protocol-level error for a fixable bad argument is its own variant of this smell.

**Fix.** Wrap every failure in a structured envelope with a class the agent can branch on — `retry-safe`, `retry-with-changes`, or `fatal` — plus a recovery hint in agent-voice. See `error-envelopes-and-recovery.md`. Violates the Gate's *every error envelope tells the agent what to do* clause; a stack trace is, by the Gate's own words, none of the three classes.

---

### 4. Parameter-The-Agent-Cannot-Fill

**Symptom.** A required parameter whose value the agent has no way to obtain from the conversation, the prior tool results, or the user. Signal: `internal_user_id` (integer), `tenant_uuid`, `partition_key`, `schema_version`, `idempotency_key` (when the server should generate it). Tighter signal: in your own golden conversation, the only way the agent got a valid value was because *you* hand-fed it in the test setup.

**Why it hurts.** The agent will do one of three bad things: hallucinate a plausible-looking value (a UUID it invented), omit it and trip a validation error, or burn turns calling other tools to discover it. All three are failure or latency. Required machine-identifiers are the most common offender — they make sense to the server author who has the database open, and are invisible-to-impossible for a model that only sees natural language.

**False-positive check.** A parameter the agent *can* fill from context is fine even if it looks technical: a `cursor` returned by the previous page of the same tool is fillable (the agent has it in context). A `file_path` the user named is fillable. The smell is about *unobtainable* values, not *technical* ones. Also legitimate: an optional parameter with a sensible default the agent can ignore.

**Fix.** Remove the parameter and derive the value server-side from session/auth context (the `tenant_uuid` belongs to the authenticated connection, not the agent's argument list — see `authentication-and-trust.md`); or make it optional with a server-generated default (the server mints the `idempotency_key`, see `idempotency-and-atomicity.md`); or replace an opaque id with a natural key the agent has (accept `issue_number` not `issue_row_id`). Violates the Gate's *agent-voice intent* spirit: if the agent cannot supply an input, the tool's contract is written for the wrong reader.

---

### 5. Return-Shape-That-Blows-Budget

**Symptom.** A tool whose return size grows with the data — `list_issues` returns every issue with every field, `read_file` returns a 200KB file whole, `search` returns 500 full records. Signal: the return shape has no bound stated in the description and no pagination. Tighter signal: on the *median* realistic input (not the worst case), the result is thousands of tokens.

**Why it hurts.** The result lands in the agent's context window and stays there for the rest of the conversation. One unbounded call can consume the budget the agent needed for the next ten turns; the host may silently truncate the result, leaving the agent reasoning over a fragment it does not know is partial — the most dangerous outcome, because it is invisible. Even when not truncated, every subsequent turn re-pays the token cost of carrying that payload.

**Why it hurts (concrete).** `list_issues()` returning 300 issues × 12 fields each is ~15K tokens. The agent asked "which issues are open?" and needed three. The other 297, and the eleven fields it did not need, now occupy context permanently and may have pushed the system prompt's instructions out of effective attention.

**False-positive check.** A tool with a *constructively bounded* return is fine even without pagination: `get_issue(id)` returns exactly one record; `count_open_issues()` returns a number. The smell needs unbounded *growth*. A tool documented as `may-be-truncated` with a companion detail-tool is an acceptable design, not a smell — the truncation is declared and the agent has a path to the full payload.

**Fix.** Give every growth-shaped return a context-budget profile: *bounded* (return a summary projection — id, title, status — not the full record), *paginated* (first page plus a cursor the agent can follow), or *explicitly-truncated* (return the head plus a `get_detail` pointer). For file reads, take a byte/line range. See `output-shape-and-pagination.md`. Violates the Gate's *context-budget profile* clause directly.

---

### 6. Retry-Amplification

**Symptom.** A side-effecting tool that is not idempotent under retry, on a transport where retries happen by default. Signal: the tool description does not state what a second call with identical arguments does; the implementation just runs again. Tighter signal: a `send_email`, `post_comment`, `charge_card`, or `append_log` with no idempotency key and no dedup.

**Why it hurts.** Agents retry. Hosts retry on timeout. Networks blip. When a slow tool returns after the client has already given up and retried, the side effect fires twice (or N times). The agent has no idea — from its side, the first call "failed" (timed out) and the second "succeeded." You get duplicate emails, double charges, two comments, a counter that is off by the number of retries. This is the single most expensive MCP smell because it is silent, intermittent, and tied to load (it fires exactly when the system is slow, which is exactly when you are least watching).

**Why it hurts (concrete).** `notify_owner(issue_id)` takes 40 seconds under load. The host's tool timeout is 30s; it retries. The owner gets two pages. The agent's transcript shows one "successful" call. The duplicate is invisible until the owner complains.

**False-positive check.** A tool whose effect is *naturally* idempotent is not amplifying: `set_status(id, "closed")` applied twice leaves the same state (`no-op-after-first`). A tool documented as `adds-twice` where the *agent* is contracted to coordinate (rare, but valid for true append-only event streams where duplicates are expected and deduped downstream) is a declared design, not a smell. The smell is *undeclared* non-idempotence on an effect that matters.

**Fix.** Make the effect idempotent: accept (or server-generate) an idempotency key and dedup on it; or use optimistic-locking (`expected_version`) so the second write fails cleanly as `retry-with-changes`; or use a claim-lease for exclusive operations. Declare the resulting guarantee in the description. See `idempotency-and-atomicity.md` and `observability-for-tool-calls.md` (you must also be able to *tell* whether a retry produced one execution or four). Violates the Gate's *idempotency guarantee* clause.

---

### 7. Schema-Drift-Without-Bump

**Symptom.** A non-backward-compatible change to a tool's input or output schema shipped without a corresponding capability/version signal the client can negotiate against. Signal: a parameter was renamed, a return field changed type, a required field was added, a tool was removed — and the server version in `plugin.json` did not change in a way the protocol surfaces, or the change is invisible behind an unchanged capability declaration. Tighter signal: an old golden conversation that used to pass now fails and nothing in the protocol told the client the contract moved.

**Why it hurts.** The agent (and the host that loaded the tool list) cached a contract. A silent breaking change means the model keeps calling the old shape and fails, or — worse — calls succeed but mean something different (a field that used to be a string is now an object; the agent's downstream reasoning is now wrong but does not error). Because MCP tool descriptions are re-interpreted by every model version, drift compounds: you cannot tell whether a regression is *your* schema change or the *model's* re-reading unless your changes are versioned and your conversations are replayable.

**False-positive check.** Backward-compatible additions are not drift: adding an *optional* parameter with a default, adding a new field to a return shape, adding a new tool — none break existing callers and none require a bump (though they may warrant a minor version for discoverability). Under JSON Schema 2020-12 (the current MCP default dialect), widening a constraint is compatible; narrowing it is not. The smell is specifically the *narrowing/renaming/removing* class shipped silently.

**Fix.** Treat non-backward-compatible changes as capability-affecting: bump the server capability so the client negotiates against it, keep the old tool/parameter alive through a deprecation window with the deprecation *visible in the description* (the surface signals its own deprecation), and run the golden-conversation suite across the change to separate your drift from model drift. See `schema-versioning-and-drift.md`. Violates the Gate's *non-backward-compatible changes bump capability* clause directly.

---

### 8. Namespace-Collision

**Symptom.** Your server's tool names collide, or are confusable, with another MCP server's tools in the same agent context. Signal: generic names — `search`, `query`, `get`, `read`, `list`, `create` — with no server-specific prefix; two servers both exposing `search`. Tighter signal: in a host that loads several servers, the agent calls *your* `search` when it meant the *other* server's, or vice versa.

**Why it hurts.** An agent's tool namespace is flat and shared across all connected servers. Identical or near-identical names force the host to disambiguate (often by silently prefixing or by dropping one), and the model — which sees a flat list — cannot tell which `search` hits which backend. You get cross-server mis-routing: a query meant for the issue tracker runs against the file store. This smell is invisible when your server is tested alone (the router's operator track exists precisely because alone-testing hides it) and only appears in real multi-server deployments.

**False-positive check.** A name that is generic but *namespaced by the host's own convention* (some hosts prefix tools with the server id automatically) may be safe — verify how *your* target host renders the combined list before assuming collision. And inside a single server, reusing a verb across distinct nouns (`get_issue`, `get_comment`) is normal, not a collision; collision is across servers or across indistinguishable intents.

**Fix.** Name tools for their domain, not generically: `issuetracker_search` or `search_issues`, never bare `search`. Keep names specific enough that they read unambiguously in a flat list alongside arbitrary other servers. Negotiate capabilities and document the assumption that your server coexists. See `composition-and-namespaces.md`. Relates to smell 1 (overlapping-tools is intra-server; namespace-collision is inter-server) and to the Gate's *agent-voice intent* clause — a name the agent cannot place in context is a mis-described tool.

---

### 9. Resource-That-Should-Be-A-Tool

**Symptom.** Read-only-looking context is exposed as an MCP **resource**, but the agent actually needs to *invoke* it with parameters it chooses, on demand, mid-reasoning. Signal: a resource whose URI encodes query parameters the agent wants to vary (`data://issues?status=open&assignee=me`), or a resource the agent "reads" but whose content depends on arguments only the model knows at call time. Tighter signal: the host attaches the resource once at the start, but the agent needs a *different* slice on turn 7 and has no way to ask for it.

**Why it hurts.** Resources are model-readable context the *host* decides to attach; tools are *model-invoked* with model-chosen arguments. When the thing the agent needs is parameterized-by-the-agent and on-demand, modeling it as a resource means the agent cannot get the variant it needs when it needs it — it is stuck with whatever slice the host attached. The model either works with stale/wrong context or gives up. This is a primitive-selection error, and it is common because "it's just reading data" *feels* like a resource.

**False-positive check.** Genuinely static or host-curated context *is* a resource: a project's README, a config file, a fixed schema doc — content the host attaches and the agent reads without needing to parameterize. Resource *templates* (parameterized URIs the agent can instantiate) legitimately cover some parameterized reads in the current spec — if the host supports templates and the parameter space is small and URI-expressible, a template resource is correct. The smell fires when the agent needs *arbitrary, model-computed* arguments or needs to read *on demand mid-loop*, which exceeds what a template cleanly expresses.

**Fix.** Promote it to a tool (often a read-only tool — a tool with no side effect is fine and idempotent by construction: declare `no-op` / read-only and `safe to call concurrently`). Keep it a resource only if it is host-curated static context or fits a resource template. Apply the decision rule in `mcp-primitive-selection.md` and `resources-prompts-sampling.md`: *who chooses the arguments, and when?* Host-at-attach-time → resource; model-at-call-time → tool.

---

### 10. Tool-That-Should-Be-A-Resource

**Symptom.** Static or slowly-changing context is exposed as a **tool** the agent must remember to call, instead of as a resource the host attaches automatically. Signal: a zero-or-trivial-argument tool like `get_project_readme()`, `get_coding_conventions()`, `get_schema_doc()` that returns the same content every time. Tighter signal: the agent forgets to call it and reasons without context it should always have had.

**Why it hurts.** Making the agent *call* for context it always needs adds a turn, a retry surface, and a forgetting-failure: if the model does not think to call `get_coding_conventions()`, it writes code that violates them. Context the agent should *always* have should be *attached*, not *requested*. It also pollutes the tool namespace (smell 8) with pseudo-tools and inflates the tool list the model has to scan on every turn, which dilutes attention on the tools that actually do things.

**False-positive check.** If the content genuinely varies by model-chosen argument or is expensive/large enough that attaching it always would blow the budget (smell 5), a tool (often paginated) is correct — you do *not* want to auto-attach a 200KB document to every conversation. The smell is specifically *static, always-relevant, attach-cheap* content modeled as a tool. Also: if the target host does not support resources well, a tool may be a pragmatic fallback — record that as the reason, do not pretend it is ideal.

**Fix.** Expose it as a resource (or a small set of resources) the host attaches; reserve tools for actions and for parameterized/on-demand reads (smell 9's legitimate side). This is the mirror image of smell 9; the two together are the primitive-selection axis — run them as a pair. See `mcp-primitive-selection.md`.

---

## Two Worked Examples

### Example A — A CRUD mirror with three co-located smells

A team ships an issue-tracker MCP server. The tool list (abridged), as the agent sees it:

```json
{
  "tools": [
    {
      "name": "get_issue",
      "description": "Returns the issue row.",
      "inputSchema": {
        "type": "object",
        "properties": { "issue_id": { "type": "integer" } },
        "required": ["issue_id"]
      }
    },
    {
      "name": "update_issue",
      "description": "Updates fields on an issue.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "issue_id":    { "type": "integer" },
          "status":      { "type": "integer" },
          "assignee_id": { "type": "integer" }
        },
        "required": ["issue_id"]
      }
    },
    {
      "name": "list_issues",
      "description": "Returns all issues.",
      "inputSchema": { "type": "object", "properties": {} }
    }
  ]
}
```

Critic pass:

```
Smell:          tool-as-CRUD-mirror
Tool:           update_issue
Severity:       major
Symptom seen:   verbs are data-model verbs; the workflow "take an issue so no one else does" is not expressible as one call.
Evidence:       to claim an issue the agent must get_issue then update_issue(status, assignee) — two calls, inferred order.
Fix:            replace with claim_issue / release_claim / close_issue workflow verbs.
Gate clause:    idempotency guarantee; concurrency contract.

Smell:          parameter-the-agent-cannot-fill
Tool:           update_issue
Severity:       blocker
Symptom seen:   status:int and assignee_id:int — the agent has natural language ("close it", "assign to me"), not integer codes.
Evidence:       golden conversation only passed because the test fed status=3 by hand.
Fix:            accept status enum strings ("open"/"in_progress"/"closed"); derive assignee from auth context, drop assignee_id.

Smell:          return-shape-that-blows-budget
Tool:           list_issues
Severity:       blocker
Symptom seen:   "Returns all issues" — unbounded, no pagination, full rows.
Evidence:       prod table has 4,100 issues; one call ≈ 60K tokens; host truncates silently at 25K.
Fix:            paginate (cursor) and return a summary projection {id, title, status}; add get_issue for detail.
Gate clause:    context-budget profile.

Smell:          retry-amplification
Tool:           update_issue (status transition that notifies)
Severity:       blocker
Symptom seen:   status change fires a webhook; no idempotency key; not declared.
Evidence:       retry log: same issue_id, status=closed, two webhook deliveries 31s apart (timeout=30s).
Fix:            optimistic lock via expected_version; declare no-op-after-first; dedup webhook on (issue_id, version).
Gate clause:    idempotency guarantee.
```

The architect's redesign closes all four at once, because they are facets of one mistake — modeling the database instead of the workflow:

```json
{
  "name": "claim_issue",
  "description": "Take an exclusive working claim on an open issue so other agents will not pick it up. No-op if you already hold the claim; fails with retry-with-changes if someone else holds it.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "issue_number":   { "type": "integer", "description": "The issue number shown to users (e.g. 142), not an internal row id." },
      "expected_status": { "type": "string", "enum": ["open"], "description": "Guards against claiming an issue that moved on; the call fails retry-with-changes if the issue is no longer open." }
    },
    "required": ["issue_number"]
  }
}
```

Note what the redesign carries that the CRUD version could not: an **agent-voice intent** ("take a claim so others will not pick it up"), a stated **idempotency guarantee** (no-op-after-first), an **error class** baked into the description (retry-with-changes when held), a **natural key** the agent has (`issue_number`, not row id), and an `assignee` that is *not* a parameter at all because it is derived from the authenticated connection.

### Example B — The primitive-selection pair (smells 9 and 10) in one config

A docs-helper server exposes both of these. Read them as the agent does:

```json
{
  "resources": [
    {
      "uri": "docs://search?q={query}",
      "name": "Documentation search",
      "description": "Search results for a query against the docs."
    }
  ],
  "tools": [
    {
      "name": "get_style_guide",
      "description": "Returns the project's writing style guide.",
      "inputSchema": { "type": "object", "properties": {} }
    }
  ]
}
```

Both are backwards.

```
Smell:          resource-that-should-be-a-tool
Item:           docs://search?q={query}
Severity:       major
Symptom seen:   the agent must search with model-chosen, arbitrary queries on demand mid-reasoning; a resource is host-attached, not model-invoked with free-form args.
Evidence:       conversation turn 6 — agent needs to search "retry backoff" but the host attached only the turn-1 query's results; no way to re-query.
False-positive: no — query is free-form natural language, not a small URI-expressible enum, so a resource template does not cover it either.
Fix:            make search_docs a read-only tool (no-op, safe-concurrent) with a query parameter and a paginated, summary return.

Smell:          tool-that-should-be-a-resource
Tool:           get_style_guide
Severity:       minor
Symptom seen:   zero-argument tool returning the same static content every time; agent must remember to call it.
Evidence:       in 3 of 5 golden conversations the agent wrote prose without calling it and violated the guide.
False-positive: checked — content is ~1,200 tokens, attach-cheap, always relevant. Not a budget concern.
Fix:            expose as a resource the host attaches (docs://style-guide); drop the tool.
```

Corrected surface — the action becomes a tool, the always-on context becomes a resource:

```json
{
  "resources": [
    { "uri": "docs://style-guide", "name": "Writing style guide", "description": "The project's writing style guide. Always-relevant context for any writing task." }
  ],
  "tools": [
    {
      "name": "search_docs",
      "description": "Search the documentation for a natural-language query and get back the top matching sections (title + snippet + section URI). Read-only; safe to call repeatedly. Returns one page; pass the returned cursor to get more.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query":  { "type": "string" },
          "cursor": { "type": "string", "description": "From a prior response's nextCursor; omit for the first page." }
        },
        "required": ["query"]
      }
    }
  ]
}
```

The decision rule that resolves both: **who chooses the arguments, and when?** Host-at-attach-time and static → resource (the style guide). Model-at-call-time with free arguments → tool (the search).

---

## Common Mistakes

- **Running the catalog as a vibe-check, not a list.** "I looked and it seems fine" is not running the catalog. Enumerate all ten against every tool; a smell you did not name is a smell you did not check. The Gate treats "no smells found" without enumeration as a skipped (= failed) check.
- **Treating a smell as a confirmed bug.** A smell is *probable* trouble, not certain trouble. Every entry has a false-positive check for a reason. Record why a smell is acceptable here; do not "fix" a paginated detail-tool that was already correct.
- **Auditing the server in isolation and declaring namespace clean.** Smell 8 is invisible alone. You must audit against the real multi-server deployment, or at minimum against the host's actual rendered tool list.
- **Confusing technical-looking with unfillable.** A `cursor` looks technical and is perfectly fillable (the agent has it from the last page). An `internal_user_id` looks similar and is not. The test is *can the agent obtain this value from context?*, not *does it look like an identifier?*
- **Fixing retry-amplification at the agent layer.** "We told the agent not to retry" is not a fix; hosts and networks retry regardless of the agent's intent. Idempotency lives on the server.
- **Findings without severity or evidence.** A critic finding that does not name the tool, quote the schema/trace, and rate severity is a vibe. The producer cannot act on it and the disagreement record is empty.
- **Zero architect-critic disagreement and calling it a pass.** Per the router: if the critic agrees with everything, the critic is reading the surface the way the architect wrote it. A clean run on a non-trivial surface is evidence of theatre, not health.

---

## Red Flags — STOP

If you catch yourself or a teammate saying any of these, stop and run the relevant smell entry before shipping:

- "It's just CRUD, the agent will figure out the workflow." → Smell 2. It will not figure it out the same way twice or across model versions.
- "The agent can just pass the user id." → Smell 4. Where does the agent get the user id? If the answer is "from auth context," it is not the agent's parameter.
- "It returns everything, but the agent only needs a few." → Smell 5. Everything is now in context forever, and the host may have truncated it without telling anyone.
- "Retries are rare." → Smell 6. They are *correlated with load*, which is when the double-effect costs the most and you are watching the least.
- "It's a tiny rename, no need to version it." → Smell 7. A rename is a non-backward-compatible change by definition; the agent cached the old name.
- "We'll just call it `search`." → Smell 8. In a multi-server context there is no "just"; the namespace is shared and flat.
- "It's read-only so it's a resource." → Smell 9. Read-only does not mean host-attached. If the agent chooses the arguments at call time, it is a tool.
- "Making it a tool is more flexible." → Smell 10. Flexibility the agent must remember to use is context the agent will forget to fetch.
- "We asked Claude and it worked." → Not evidence the smells are absent. One conversation does not exercise the retry path, the rare argument, or the next model version.

---

## Counters to the Rationalizations for Skipping This Pass

- *"The surface is small, I can eyeball it."* — Smells 4, 6, and 7 are invisible to eyeballing because they fire under conditions (impossible arguments, retries, cached contracts) you are not simulating when you read the list. Small surfaces still ship retry-amplification.
- *"We have tests, so the smells would show up."* — Only if the tests are golden conversations that exercise retries, rare arguments, and large inputs. A happy-path smoke test ("it worked once") passes over smells 5 and 6 entirely. See `testing-mcp-servers.md`.
- *"The model is smart enough to handle ambiguity."* — Tool selection (smells 1, 8) and error recovery (smell 3) are exactly the places where "smart enough" varies by model version and by phrasing. Designing for the smart case is designing for a non-deterministic best case you do not control.
- *"This is the same as REST review, our API people already looked at it."* — REST review assumes a human reads the docs and writes client code once. Every smell in this catalog is about an LLM reading the surface on every turn with no human in the loop. REST review structurally cannot catch them; that asymmetry is the whole point of this pack (see the router's pipeline contrast with `/web-backend`).
- *"We'll fix smells when they cause incidents."* — Smells 6 and 7 cause *silent* incidents (duplicate side effects, contract drift) that surface as a customer complaint or a corrupted counter weeks later, with no stack trace pointing back to the cause. The catalog is cheaper than the forensics.
- *"The critic found nothing, so we're clean."* — A no-finding critic pass on a non-trivial surface is a defect of the critic, not a property of the server. Re-run with a fresh frame; the absence of disagreement between architect and critic is itself a red flag.

---

## Cross-References

- `tool-api-design.md` — naming, granularity, the description-as-prompt-fragment discipline (smells 1, 2, 4).
- `mcp-primitive-selection.md` and `resources-prompts-sampling.md` — the tool/resource/prompt/sampling decision rule (smells 9, 10).
- `output-shape-and-pagination.md` — context-budget profiles, summary-vs-detail, truncation policy (smell 5).
- `idempotency-and-atomicity.md` — claim-leases, optimistic locking, exactly-once-under-retry (smell 6).
- `error-envelopes-and-recovery.md` — the three error classes and recovery hints (smell 3).
- `schema-versioning-and-drift.md` — capability bumps, deprecation windows, model-version drift (smell 7).
- `composition-and-namespaces.md` — multi-server namespaces and collision (smell 8).
- `authentication-and-trust.md` — deriving identity from connection context instead of parameters (smell 4).
- `observability-for-tool-calls.md` — how to *detect* retry-amplification in production (smell 6).
- `testing-mcp-servers.md` — golden conversations that actually exercise the smell-prone paths.
