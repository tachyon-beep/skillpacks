---
description: Run cargo-audit and cargo-deny to check advisories, licenses, bans, and sources
allowed-tools: ["Read", "Edit", "Bash", "Skill"]
argument-hint: "[workspace root] - defaults to current directory"
---

# Audit Command

Run `cargo audit` (RustSec advisories) and `cargo deny check` (advisories, licenses, bans, sources) for supply-chain hygiene.

## Prerequisites

- `cargo audit` and `cargo deny` installed: `cargo install cargo-audit cargo-deny`
- `deny.toml` committed at the workspace root. If missing, bootstrap with `cargo deny init` and tighten the allowlist/bans before running the audit — see `project-structure-and-tooling.md` for a complete template.

## Process

1. **Run `cargo audit`**

   ```bash
   cargo audit --json
   ```

   Each finding shows the RUSTSEC ID, affected crate, version range, severity, and fix.

2. **Run `cargo deny check`**

   ```bash
   cargo deny check advisories licenses bans sources
   ```

   | Check | Catches |
   |-------|---------|
   | advisories | Known CVEs and unmaintained crates |
   | licenses | GPL, proprietary, or disallowed licenses |
   | bans | Crates blacklisted by the policy |
   | sources | Git/path dependencies outside the allowlist |

3. **Triage each finding**
   - **Critical/High advisories**: upgrade (`cargo update <crate>`) or replace the dependency. Re-run step 1.
   - **License violations**: swap to a permissive-licensed alternative, or add an `exceptions` entry with legal-review justification.
   - **Bans**: replace with the approved alternative listed in `deny.toml`.
   - **Unknown sources**: move the dependency to a released crates.io version, or add it to `allow-git` with an explanation.
   - **Unmaintained / yanked** with no fix available: add to `[advisories] ignore` with a `reason` comment explaining why it is acceptable.

## Success Criteria

The audit is complete when:

- `cargo audit --json` exits 0.
- `cargo deny check advisories licenses bans sources` exits 0.
- Every `ignore` entry or `exceptions` entry added to `deny.toml` in this pass carries a comment stating the justification.

**Report back**: list each finding, its severity, and the action taken (upgraded, replaced, waived with reason, or deferred with a ticket reference).

## Load Detailed Guidance

For supply-chain threat patterns, `deny.toml` design, and CI integration:

```
Load skill: axiom-rust-engineering:using-rust-engineering
Then read: project-structure-and-tooling.md
```

References: <https://rustsec.org>, <https://embarkstudios.github.io/cargo-deny/>
