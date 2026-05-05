---
description: Audit a Rust workspace's dependency hygiene at workspace scope — duplicate dep detection, version drift across members, licence sweep, advisory check, source policy enforcement. Composes `cargo tree --duplicates`, `cargo deny check --workspace`, and `cargo audit` into one pass with structured output. Optionally dispatches the workspace-reviewer agent for findings interpretation.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Task", "Write", "Edit", "AskUserQuestion"]
argument-hint: "[workspace_path]"
---

# Audit Workspace Deps Command

You are running a workspace-scope dependency audit. The output is a structured findings report covering: dep version drift across member crates, licence policy violations against the workspace's allow-list, open security advisories, banned-crate / banned-source policy hits, and the per-`[workspace.dependencies]` inheritance discipline.

## Invocation Path

`/audit-workspace-deps` is a Claude Code slash command that wraps the per-section `cargo deny check`, `cargo audit`, and `cargo tree` invocations into one workflow with consolidated reporting. It composes — does not replace — the per-crate `axiom-rust-engineering:audit` command.

For ongoing supply-chain hygiene, run this on every PR (via the `just deny` recipe per `11-`). For a fresh workspace audit at intake, run with the `--remediate` flag (interactive) to get suggestions for each finding.

## Preconditions

The command takes a single optional argument: a workspace path. Defaults to current working directory.

### Resolve and verify

```bash
WORKSPACE_DIR="${ARGUMENTS:-.}"

if [ ! -f "${WORKSPACE_DIR}/Cargo.toml" ]; then
  echo "ERROR: ${WORKSPACE_DIR}/Cargo.toml not found. Run from a workspace root or pass a path."
  exit 1
fi

# Verify it's a workspace, not a single-crate project
if ! grep -q '^\[workspace\]' "${WORKSPACE_DIR}/Cargo.toml"; then
  echo "ERROR: ${WORKSPACE_DIR}/Cargo.toml has no [workspace] table."
  echo "For single-crate audits, use axiom-rust-engineering:audit instead."
  exit 1
fi
```

### Verify required tooling

```bash
for tool in cargo cargo-deny cargo-audit; do
  if ! command -v "$tool" > /dev/null 2>&1; then
    echo "ERROR: $tool not installed."
    case "$tool" in
      cargo)       echo "  Install: rustup install stable" ;;
      cargo-deny)  echo "  Install: cargo install --locked cargo-deny" ;;
      cargo-audit) echo "  Install: cargo install --locked cargo-audit" ;;
    esac
    exit 1
  fi
done
```

### Verify config files

```bash
test -f "${WORKSPACE_DIR}/deny.toml" || echo "WARN: ${WORKSPACE_DIR}/deny.toml missing — license/bans/sources checks will use cargo-deny defaults"
test -f "${WORKSPACE_DIR}/Cargo.lock" || echo "WARN: Cargo.lock missing — running cargo generate-lockfile first"
```

If `Cargo.lock` is missing, run `cargo generate-lockfile` before audit.

## Workflow

### Step 1 — Dep version drift sweep

```bash
cd "${WORKSPACE_DIR}"
cargo tree --workspace --duplicates --edges normal > /tmp/audit-duplicates.txt
```

Parse the output: every block (separated by blank lines) is a dep that appears at multiple versions. Categorise:

- **Direct duplicates** — the dep is declared in two crates' `[dependencies]` (or `[dev-dependencies]`) at incompatible versions. Workspace-policy violation; should be unified via `[workspace.dependencies]` per `02-`.
- **Transitive duplicates** — the dep is pulled in transitively at multiple versions because two upstream crates depend on it differently. Often unavoidable; record in `04-`'s `[bans] skip` if accepted.

Report:

```text
=== Dep Drift ===
Direct duplicates (workspace-policy violations):
  - serde 0.9.x (in crates/legacy-mod) vs 1.0.x (in crates/myapp-core)
    REMEDIATION: Move to [workspace.dependencies]; upgrade legacy-mod to serde 1.0.

Transitive duplicates (review for [bans].skip waiver):
  - windows-sys 0.48.x (via tokio) vs 0.52.x (via winapi)
    REMEDIATION: Likely unavoidable; if accepted, add to deny.toml [bans].skip.
```

### Step 2 — Verify `[workspace.dependencies]` inheritance

```bash
# For every dep used by ≥ 2 member crates, verify it's in [workspace.dependencies]
# and that members inherit via `dep = { workspace = true }`.

# 1. Extract direct deps per member
for crate_toml in crates/*/Cargo.toml; do
  echo "=== ${crate_toml} ==="
  awk '/^\[(dev-)?dependencies\]/{flag=1; next} /^\[/{flag=0} flag && /^[a-zA-Z]/' "${crate_toml}"
done > /tmp/audit-deps-per-crate.txt

# 2. Extract [workspace.dependencies] names
awk '/^\[workspace\.dependencies\]/{flag=1; next} /^\[/{flag=0} flag && /^[a-zA-Z]/' Cargo.toml \
  | awk -F'=' '{print $1}' | tr -d ' ' | sort -u > /tmp/audit-workspace-deps.txt

# 3. Find deps used by ≥2 crates that are NOT in [workspace.dependencies]
# (manual sweep; report candidates)
```

Report:

```text
=== Workspace Dep Inheritance ===
Deps used by ≥2 member crates but NOT in [workspace.dependencies]:
  - regex (used by crates/myapp-core and crates/myapp-cli)
    REMEDIATION: Move to [workspace.dependencies]; update both to inherit.

Deps in [workspace.dependencies] but not inherited (declared locally instead):
  - serde (in crates/legacy-mod) — declared locally, ignoring workspace version
    REMEDIATION: Change to `serde = { workspace = true }`.
```

### Step 3 — Advisory check

```bash
cargo deny check --workspace advisories 2>&1 | tee /tmp/audit-advisories.txt
echo "---"
cargo audit 2>&1 | tee /tmp/audit-cargo-audit.txt
```

`cargo deny` and `cargo audit` use overlapping advisory databases; running both gives belt-and-braces coverage. Report each finding with: advisory ID, affected crate, severity, the remediation path (upgrade to fixed version, ignore with rationale, or yank-and-wait).

### Step 4 — Licence sweep

```bash
cargo deny check --workspace licenses 2>&1 | tee /tmp/audit-licenses.txt
```

Report:

- Any licence in the dep graph not on the `deny.toml` allow-list.
- Any crate with a `LICENSE-NOT-FOUND` (cargo-deny couldn't determine the licence).
- Confidence-threshold misses (the parser was uncertain; manual review needed).

For each: the offending crate, the offending licence (or absence), the recommended `deny.toml` action (add to `allow`, add as a per-crate exception with rationale, or remove the dep).

### Step 5 — Bans and sources

```bash
cargo deny check --workspace bans 2>&1 | tee /tmp/audit-bans.txt
cargo deny check --workspace sources 2>&1 | tee /tmp/audit-sources.txt
```

`bans` reports:
- Banned-crate hits (per `[bans].deny`).
- Wildcard version requirements (per `[bans].wildcards`).
- Multiple-version warnings (per `[bans].multiple-versions`).

`sources` reports:
- Crates from registries not on the allow-list.
- Crates from git URLs not on the allow-list.

### Step 6 — Cross-check `[workspace.dependencies]` declarations against `deny.toml`

```bash
# For every banned crate in deny.toml, verify it's not in [workspace.dependencies] either
DENIED=$(awk '/^\[\[bans\.deny\]\]/{flag=1} /^\[/{if(NR>1)flag=0} flag && /name *=/' deny.toml \
  | sed -E 's/.*name *= *"([^"]+)".*/\1/')

for crate in $DENIED; do
  if grep -q "^${crate} *=" Cargo.toml; then
    echo "COHERENCE: ${crate} is in [bans].deny but also in [workspace.dependencies]"
  fi
done
```

Report any incoherence between `Cargo.toml` and `deny.toml`. (For deeper coherence checking, dispatch `/validate-workspace-config`.)

### Step 7 — Optional remediation (if `--remediate`)

If invoked with `--remediate`, for each finding use AskUserQuestion to offer:

1. **Apply suggested fix** — write the change (e.g., add the dep to `[workspace.dependencies]`, update the crate's `[dependencies]` to inherit). Stage in git; do not commit.
2. **Add waiver** — write the entry to `deny.toml`'s `ignore` / `skip` / `exceptions` with the user-supplied rationale and re-evaluation trigger.
3. **Skip** — leave the finding open; report it in the summary as "deferred."

### Step 8 — Consolidated report

Emit a structured findings report:

```text
=== Workspace Audit Report ===
Workspace: <path>
Date: <ISO 8601>
Workspace tier (per 00-): <tier>

Summary:
  Direct dep duplicates:           <N>
  Transitive dep duplicates:       <N>
  Workspace.dependencies misses:   <N>
  Open advisories:                 <N>
  Licence violations:              <N>
  Banned-crate hits:               <N>
  Banned-source hits:              <N>
  Coherence issues:                <N>

Findings:
  [1] <severity> <category> <description>
       Affected: <crate or dep>
       Remediation: <suggested fix>
       Cross-ref: <sheet that addresses this class>
  ...

Recommendations:
  - <prioritised next steps>

Status: <PASS | WARN | FAIL>
```

`PASS` = no findings.
`WARN` = findings present but all categorised as "warn" severity in the workspace policy.
`FAIL` = at least one finding at "deny" severity. Block merge.

## Postconditions

After successful audit:

- A consolidated report is on stdout (and optionally to `audit-report-<date>.md` if `--save` is passed).
- No files are modified unless `--remediate` was passed and the user opted to apply fixes.
- The exit code is non-zero if and only if status is `FAIL`.

## Don't Use This Command When

- Auditing a single crate (no `[workspace]` table) — use `axiom-rust-engineering:audit` instead.
- Building from scratch — use `/scaffold-workspace` first; then `/audit-workspace-deps` once member crates have deps.
- Investigating a specific advisory's impact — use `cargo audit fix` directly for guided remediation; this command is the wider sweep.

## Cross-References

- `using-rust-workspaces` skill router — the discipline this command enforces.
- `02-workspace-dependencies-and-resolver.md` — the inheritance and drift rules this command checks.
- `04-workspace-deny-config.md` — the deny-policy this command applies; the waiver lifecycle this command's `--remediate` mode operationalises.
- `05-feature-unification-gotchas.md` — feature-graph issues are NOT covered by this command (use `cargo tree -e features` directly per the sheet); this command covers the dep graph only.
- `13-workspace-anti-patterns.md` — version-drift, deny.toml shadowing, and "we'll consolidate later" are detected here.
- `axiom-rust-engineering:audit` — the per-crate audit; this command composes it at workspace scale.
- `/validate-workspace-config` — the deeper coherence check across `Cargo.toml`, `deny.toml`, `clippy.toml`. Run after this command if findings touch config coherence.
- `workspace-reviewer` agent — for narrative interpretation of findings; dispatched via `Task` if the user wants prose synthesis rather than a structured list.
