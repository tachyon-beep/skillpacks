---
description: Validate cross-file coherence in a Rust workspace's configuration — `Cargo.toml`'s `[workspace]` settings, `deny.toml`, `clippy.toml`, `rust-toolchain.toml`, and per-member `Cargo.toml` files. Detects MSRV mismatches, banned-but-pinned crates, lint policy not inherited, publish-flag drift, resolver omissions, and other "the configs disagree" failure modes. Read-only by default; emits a coherence report.
allowed-tools: ["Read", "Grep", "Glob", "Bash", "Write", "AskUserQuestion"]
argument-hint: "[workspace_path]"
---

# Validate Workspace Config Command

You are running a coherence check across the workspace's configuration files. The output is a structured report listing every disagreement between files — for example, an MSRV declared in `Cargo.toml`'s `package.rust-version` that doesn't match `clippy.toml`'s `msrv`, a crate in `[bans].deny` that `[workspace.dependencies]` still pins, an internal-named crate that lacks `publish = false`, or a per-crate `[lints]` block that doesn't inherit from the workspace.

This command is the static-analysis pass over the workspace's *config*, complementing `/audit-workspace-deps`'s pass over the workspace's *deps*.

## Invocation Path

`/validate-workspace-config` is a Claude Code slash command. It is read-only by default — it diagnoses but does not fix. Pass `--report-only` for emit-and-exit; pass `--interactive` to walk findings with AskUserQuestion and offer suggested edits one at a time.

For greenfield scaffolding, use `/scaffold-workspace` (which already produces coherent config). For dep-graph audits, use `/audit-workspace-deps`. This command checks that the *config files agree with each other*.

## Preconditions

```bash
WORKSPACE_DIR="${ARGUMENTS:-.}"

if [ ! -f "${WORKSPACE_DIR}/Cargo.toml" ]; then
  echo "ERROR: ${WORKSPACE_DIR}/Cargo.toml not found."
  exit 1
fi

if ! grep -q '^\[workspace\]' "${WORKSPACE_DIR}/Cargo.toml"; then
  echo "ERROR: Not a workspace. For single-crate config validation, use axiom-rust-engineering:typecheck."
  exit 1
fi
```

## Workflow

The command runs a series of named **coherence checks**. Each check produces a pass/fail line; failures are collected and reported with the suggested remediation.

### Check 1 — Resolver explicitness

Per `02-`: the resolver must be declared explicitly, not defaulted by edition.

```bash
RESOLVER=$(awk '/^\[workspace\]/{flag=1; next} /^\[/{flag=0} flag && /^resolver/' Cargo.toml)
if [ -z "${RESOLVER}" ]; then
  echo "FAIL: [workspace].resolver not declared explicitly."
  echo "  Remediation: Add \`resolver = \"3\"\` (or \"2\") to [workspace] in Cargo.toml."
fi
```

### Check 2 — MSRV alignment across files

Per `03-`: if the workspace declares an MSRV (`package.rust-version` in any published crate or `[workspace.package].rust-version`), `clippy.toml` must declare the same `msrv`.

```bash
WORKSPACE_MSRV=$(awk '/^\[workspace\.package\]/{flag=1; next} /^\[/{flag=0} flag && /^rust-version/' Cargo.toml \
  | sed -E 's/.*= *"([^"]+)".*/\1/')

CLIPPY_MSRV=$(grep '^msrv' clippy.toml 2>/dev/null | sed -E 's/.*= *"([^"]+)".*/\1/')

if [ -n "${WORKSPACE_MSRV}" ] && [ "${CLIPPY_MSRV}" != "${WORKSPACE_MSRV}" ]; then
  echo "FAIL: Workspace MSRV is ${WORKSPACE_MSRV}; clippy.toml msrv is '${CLIPPY_MSRV:-<unset>}'."
  echo "  Remediation: Set msrv = \"${WORKSPACE_MSRV}\" in clippy.toml."
fi
```

Also check `rust-toolchain.toml`'s channel against the MSRV: if the toolchain channel is *older* than the MSRV, the workspace cannot compile against its declared MSRV.

### Check 3 — Banned crates not pinned

Per `04-` and `13-`: a crate in `[bans].deny` should not appear in `[workspace.dependencies]`.

```bash
BANNED=$(awk '/^\[\[bans\.deny\]\]/{flag=1} /^\[\[/{if(NR>1)flag=0} flag && /^name *=/' deny.toml 2>/dev/null \
  | sed -E 's/.*name *= *"([^"]+)".*/\1/')

for crate in $BANNED; do
  if awk '/^\[workspace\.dependencies\]/{flag=1; next} /^\[/{flag=0} flag' Cargo.toml | grep -q "^${crate} *="; then
    echo "FAIL: ${crate} is in [bans].deny but also in [workspace.dependencies]."
    echo "  Remediation: Either un-ban (remove from deny.toml [bans].deny) or remove from [workspace.dependencies]."
  fi
done
```

### Check 4 — Per-crate `[lints]` inheritance

Per `03-`: every member crate's `Cargo.toml` should have `[lints] workspace = true` unless it's on a documented exception list.

```bash
for crate_toml in crates/*/Cargo.toml; do
  CRATE_NAME=$(awk -F'"' '/^name *=/{print $2; exit}' "${crate_toml}")
  if grep -q '^\[lints\]' "${crate_toml}"; then
    if ! grep -A2 '^\[lints\]' "${crate_toml}" | grep -q 'workspace *= *true'; then
      echo "FAIL: ${CRATE_NAME}: [lints] block does not declare workspace = true."
      echo "  Remediation: Set [lints] workspace = true (and document in 03- if exception)."
    fi
  else
    echo "WARN: ${CRATE_NAME}: no [lints] block. Workspace lints not inherited."
    echo "  Remediation: Add [lints] workspace = true."
  fi
done
```

### Check 5 — `publish` flag declared deliberately

Per `06-`: every member crate has `publish` declared explicitly. Internal crates have `publish = false`; published crates have `publish = ["crates-io"]` (or similar).

```bash
for crate_toml in crates/*/Cargo.toml; do
  CRATE_NAME=$(awk -F'"' '/^name *=/{print $2; exit}' "${crate_toml}")
  PUBLISH_LINE=$(grep '^publish' "${crate_toml}")
  if [ -z "${PUBLISH_LINE}" ]; then
    echo "FAIL: ${CRATE_NAME}: no publish field declared. Defaults to publish=true; risk of accidental publication."
    echo "  Remediation: Add \`publish = false\` (internal) or \`publish = [\"crates-io\"]\` (public; with full metadata)."
  fi
done
```

Cross-check against any publish allowlist mentioned in `06-` or in a CI guard script.

### Check 6 — Per-crate `clippy.toml` shadows

Per `03-` and `13-`: per-crate `clippy.toml` files exist only for the documented exception cases (msrv-shim, generated-code, vendored fork).

```bash
for shadow in crates/*/clippy.toml; do
  if [ -f "${shadow}" ]; then
    CRATE_NAME=$(dirname "${shadow}" | sed 's|.*/||')
    echo "WARN: ${CRATE_NAME} has its own clippy.toml. Shadows workspace policy."
    echo "  Remediation: Delete unless documented in 03- as one of the three legitimate cases."
  fi
done
```

### Check 7 — Per-crate `deny.toml` shadows

Per `04-` and `13-`: per-crate `deny.toml` files are essentially never justified.

```bash
for shadow in crates/*/deny.toml; do
  if [ -f "${shadow}" ]; then
    CRATE_NAME=$(dirname "${shadow}" | sed 's|.*/||')
    echo "FAIL: ${CRATE_NAME} has its own deny.toml. cargo deny check behaviour becomes working-directory-dependent."
    echo "  Remediation: Delete; use the workspace-root deny.toml only."
  fi
done
```

### Check 8 — Path-deps with version on published crates

Per `02-` and `06-`: if any member is published, its path-deps to other members must carry `version = "..."`.

```bash
# For every member declared as published
for crate_toml in crates/*/Cargo.toml; do
  CRATE_NAME=$(awk -F'"' '/^name *=/{print $2; exit}' "${crate_toml}")
  IS_PUBLISHED=$(grep -E '^publish *= *(true|\["crates-io"\])' "${crate_toml}")
  if [ -n "${IS_PUBLISHED}" ]; then
    # Find this crate's path-deps; verify each carries a version field
    awk '/^\[(dependencies|dev-dependencies)\]/{flag=1; next} /^\[/{flag=0} flag && /path *= *".*"/' "${crate_toml}" \
      | while read -r line; do
        if ! echo "${line}" | grep -q 'version *= *"'; then
          DEP_NAME=$(echo "${line}" | awk -F'=' '{print $1}' | tr -d ' ')
          echo "FAIL: ${CRATE_NAME} (published) has path-dep on ${DEP_NAME} without version field."
          echo "  Remediation: Add \`version = \"x.y.z\"\` to the path-dep declaration."
        fi
      done
  fi
done
```

### Check 9 — Workspace.lints `priority` discipline

Per `03-`: every group lint (`pedantic`, `nursery`, `cargo`, `all`) must declare `priority = -1` to allow specific lints to override.

```bash
GROUP_LINTS="pedantic nursery cargo all"
for group in $GROUP_LINTS; do
  if grep -E "^${group} *= *\"" Cargo.toml > /dev/null; then
    # Group lint declared as a string (no priority); will conflict with specific lints
    echo "FAIL: [workspace.lints.clippy].${group} declared without priority field."
    echo "  Remediation: ${group} = { level = \"warn\", priority = -1 }"
  fi
done
```

### Check 10 — Members list integrity

Per `01-`: explicit member list (recommended) or glob-with-excludes; verify every listed member exists and every existing member is either listed or globbed-and-included.

```bash
# Extract the members list (handle both string and glob form)
MEMBERS_BLOCK=$(awk '/^members *=/{flag=1} flag {print; if(/\]/) exit}' Cargo.toml)

# Verify every listed member directory exists with a Cargo.toml
echo "${MEMBERS_BLOCK}" | grep -oE '"[^"]+"' | tr -d '"' | while read -r member; do
  # Skip globs
  if echo "${member}" | grep -q '\*'; then continue; fi
  if [ ! -f "${member}/Cargo.toml" ]; then
    echo "FAIL: Member \`${member}\` listed but ${member}/Cargo.toml does not exist."
    echo "  Remediation: Create the crate or remove from members list."
  fi
done

# Verify every crates/<name>/Cargo.toml is reachable from the members list
# (heuristic; full glob resolution is non-trivial in shell)
for crate_toml in crates/*/Cargo.toml; do
  MEMBER_PATH=$(dirname "${crate_toml}")
  if ! echo "${MEMBERS_BLOCK}" | grep -q "${MEMBER_PATH}\|crates/\*"; then
    CRATE_NAME=$(awk -F'"' '/^name *=/{print $2; exit}' "${crate_toml}")
    echo "WARN: ${CRATE_NAME} (${MEMBER_PATH}) exists but is not in the members list (or matched by glob)."
    echo "  Remediation: Add to members, or move out of crates/ if intentional."
  fi
done
```

### Check 11 — `rust-toolchain.toml` consistency

```bash
if [ -f rust-toolchain.toml ]; then
  CHANNEL=$(awk -F'"' '/^channel *=/{print $2; exit}' rust-toolchain.toml)
  echo "INFO: rust-toolchain.toml pins channel = ${CHANNEL}"
  # If MSRV is declared and channel is older than MSRV, FAIL
  if [ -n "${WORKSPACE_MSRV}" ]; then
    # (Version comparison via sort -V)
    OLDER=$(printf "%s\n%s\n" "${CHANNEL}" "${WORKSPACE_MSRV}" | sort -V | head -1)
    if [ "${OLDER}" = "${CHANNEL}" ] && [ "${CHANNEL}" != "${WORKSPACE_MSRV}" ]; then
      echo "FAIL: rust-toolchain.toml channel (${CHANNEL}) is older than workspace MSRV (${WORKSPACE_MSRV})."
      echo "  Remediation: Bump channel to >= ${WORKSPACE_MSRV}."
    fi
  fi
else
  echo "INFO: No rust-toolchain.toml. Toolchain selection is environment-dependent."
fi
```

### Step — Consolidated coherence report

```text
=== Workspace Config Coherence Report ===
Workspace: <path>
Date: <ISO 8601>

Checks run: 11
PASS: <N>
WARN: <N>
FAIL: <N>

Findings:
  [Check 1: Resolver explicitness] <PASS|FAIL> ...
  [Check 2: MSRV alignment]        <PASS|FAIL> ...
  ...

Recommendations (prioritised):
  1. <highest-severity finding>
  2. ...

Status: <COHERENT | INCOHERENT>
```

`COHERENT` = no FAIL findings.
`INCOHERENT` = at least one FAIL finding. Block release; address before downstream citation.

### Optional `--interactive` mode

If invoked with `--interactive`, for each FAIL/WARN finding use AskUserQuestion to offer:

1. **Show the suggested edit** — display the diff of what the fix would change.
2. **Apply** — write the edit. Stage in git.
3. **Skip** — leave the finding open; record in the report as "deferred."
4. **Document exception** — add a comment to the relevant config file noting the intentional deviation, with rationale.

## Postconditions

After successful run:

- A consolidated coherence report is on stdout (and optionally to `coherence-report-<date>.md` if `--save`).
- No files are modified unless `--interactive` was passed and the user opted to apply.
- Exit code is non-zero if and only if any FAIL is present.

## Don't Use This Command When

- The workspace is greenfield and you're scaffolding from scratch — use `/scaffold-workspace` (which produces coherent config).
- You want a dep-graph audit (drift, advisories, licences) — use `/audit-workspace-deps`.
- You want a deeper structural review (anti-pattern sweep, narrative interpretation) — dispatch the `workspace-reviewer` agent via the Task tool.

## Cross-References

- `using-rust-workspaces` skill router — the discipline this command enforces.
- `/scaffold-workspace` — produces config that passes this validator on first emission.
- `/audit-workspace-deps` — runs in parallel; covers dep-graph concerns; this command covers config-coherence concerns.
- `02-`, `03-`, `04-`, `06-` — the sheets whose policies this command's checks operationalise.
- `13-workspace-anti-patterns.md` — `deny.toml` shadowing, `clippy.toml` shadowing, drift between config files.
- `workspace-reviewer` agent — for narrative findings and prioritisation of the report.
