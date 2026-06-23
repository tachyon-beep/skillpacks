# Team Coordination

Multi-agent swarm protocols for filigree 2.0. Load this reference when coordinating
work across multiple agents.

## Atomic Start

### The Race Condition Problem

When multiple agents call `filigree update <issue-id> --status=<wip>`
simultaneously, both think they own the issue. Filigree 2.0 solves this with
`start-work`, which atomically claims the issue *and* transitions it to its
type-specific working status (tasks → `in_progress`, features → `building`,
bugs → `fixing`) in a single DB transaction with optimistic locking on the
assignee.

### Start Protocol

```bash
# Option A: Start a specific issue
filigree start-work <issue-id> --assignee <agent-name>

# Option B: Start the highest-priority ready issue
filigree start-next-work --assignee <agent-name>
```

If another agent already claimed the issue, the call fails with
`code: CONFLICT` (CLI exit 4). No silent overwrite, no half-claimed state —
either both the claim and the transition land, or neither does.

`start-next-work` accepts the work-scoping filters `claim-next` also
takes (`--type`, `--priority-min`, `--priority-max`) so specialised agents
can scope their work. Because `start-next-work` *transitions* (not just
reserves), it additionally accepts `--target-status` to override the wip
target and `--advance` to walk soft transitions to wip — neither of which
`claim-next` has, since `claim-next` only reserves and never changes status.

### Niche: Claim Without Transitioning

If a coordinator wants to reserve an issue without advancing its status
(e.g. earmarking it for a downstream worker), use the atomic primitives:

```bash
filigree claim <issue-id> --assignee <agent-name>
filigree claim-next --assignee <agent-name>
```

These are kept for niche use; `start-work` is the default in 2.0.

### Releasing Claims

If an agent cannot finish the work:

```bash
filigree add-comment <issue-id> "Releasing: blocked on X, needs Y to continue"
filigree release <issue-id>
```

Always add a comment before releasing — the next agent needs context.

## Handoff Protocol

When passing work between agents, follow this sequence:

### Outgoing Agent (Finishing)

1. **Document state**: Add a comment with current progress, decisions made,
   and remaining work
2. **Update status**: Leave in its working status (`in_progress` / `building` /
   `fixing`) if partially done, or close if complete
3. **Flag blockers**: Create blocker issues and add dependencies if needed

```bash
filigree add-comment <issue-id> "Completed: API endpoints for auth.
Remaining: frontend login page needs the /api/token response format.
Decision: used JWT not sessions — see commit abc123.
Blocker: need CORS config before frontend can call API."
```

### Incoming Agent (Picking Up)

1. **Read context**: `filigree show <issue-id>` and `filigree get-comments <issue-id>`
2. **Check dependencies**: Look at `blocked_by` in the show output
3. **Start**: `filigree start-work <issue-id> --assignee <name>`
4. **Continue**: Build on the previous agent's work, don't restart

## Status Update Conventions

### When to Update Status

| Event | Action |
|-------|--------|
| Starting work | `start-work <issue-id> --assignee <name>` (atomic claim + transition) |
| Hit a blocker | Add comment, create blocker issue, add dep |
| Completed the work | `close --reason="..."` |
| Can't finish, releasing | Comment + `release` |
| Found additional work | Create new issues, add deps if needed |

### Comment Conventions

Prefix comments with context markers for quick scanning:

```bash
filigree add-comment <issue-id> "PROGRESS: implemented X and Y, Z remaining"
filigree add-comment <issue-id> "BLOCKED: waiting on <blocker-id> for API schema"
filigree add-comment <issue-id> "DECISION: chose approach A because of B"
filigree add-comment <issue-id> "HANDOFF: releasing, next agent should start at Z"
```

## Swarm Work Distribution

### Leader-Follower Pattern

One agent acts as coordinator:

1. **Leader** runs `filigree ready` and assigns work (or pre-claims via `claim`)
2. **Followers** use `filigree start-work <issue-id> --assignee <name>` to take it on
3. **Followers** report back via comments when done
4. **Leader** monitors `filigree stats` and `filigree list --status=in_progress`

### Self-Organising Pattern

All agents are peers:

1. Each agent runs `filigree start-next-work --assignee <name>`
2. Works on the started issue independently
3. Closes and immediately calls `start-next-work` again
4. No central coordinator needed

This works best when:
- Issues are well-defined and independent
- Dependencies are properly wired (so `start-next-work` only returns unblocked work)
- Priority ordering reflects actual importance

Tie-break ordering for `start-next-work` (and `claim-next`):
1. `priority` ascending (0 = critical first)
2. `created_at` ascending (oldest first within a priority tier)
3. `issue_id` ascending (deterministic tie-break)

### Filtering by Type

Specialised agents can filter their start calls:

```bash
# Backend agent
filigree start-next-work --assignee backend-1 --type task

# Bug-fixing agent
filigree start-next-work --assignee bugfix-1 --type bug --priority-max 1
```

## Conflict Resolution

### Two Agents Modified the Same Code

1. The second agent's commit will show merge conflicts
2. Add a comment on the issue explaining the conflict
3. The agent with the simpler change should rebase
4. Use `filigree add-comment` to document the resolution

### Two Agents Claimed Related Work

If agents discover their tasks overlap:

1. One agent adds a dependency between the tasks
2. The agent with the lower-priority task releases their claim
3. The remaining agent completes the prerequisite first

### Stale Claims

If an agent disappears without completing work:

```bash
filigree list --status=in_progress --assignee <missing-agent>
filigree release <issue-id>                         # free the claim
filigree add-comment <issue-id> "Released: previous agent did not complete"
```

### CONFLICT Responses

A `start-work` (or `claim`) call that loses the race returns
`{error: ..., code: "CONFLICT", details: {current_assignee: "..."}}` and
exits with code 4. This is distinct from operational errors (exit 1) so
automated callers can retry against a different issue without escalating.

## Session Resumption

When an agent starts a new session and needs to resume context:

```bash
# What was I working on?
filigree list --status=in_progress --assignee <name>

# What happened since I last worked?
filigree changes --since <last-session-timestamp>

# What's ready now?
filigree ready
```

The `filigree session-context` hook does this automatically at session start,
but these commands are useful for manual context recovery.
