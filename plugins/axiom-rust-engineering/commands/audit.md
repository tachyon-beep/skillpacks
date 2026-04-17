---
description: Run cargo-audit and cargo-deny to check advisories, licenses, bans, and sources
allowed-tools: ["Read", "Edit", "Bash"]
argument-hint: "[workspace root] - defaults to current directory"
---

# Audit Command

Run `cargo audit` (RustSec advisories) + `cargo deny check` (advisories, licenses, bans, sources) for supply-chain hygiene.

## Installation

```bash
cargo install cargo-audit cargo-deny
# Verify: cargo --list | grep -E 'audit|deny'
```

## Process

### 1. Run `cargo audit`

```bash
cargo audit --json
```

Shows vulnerabilities: RUSTSEC-YYYY-XXXX ID, affected crate, version range, severity, fix.

### 2. Run `cargo deny check`

```bash
cargo deny check advisories licenses bans sources
```

Four checks:

| Check | Catches |
|-------|---------|
| advisories | Known CVEs & unmaintained crates |
| licenses | GPL, proprietary, conflicting licenses |
| bans | Internal dependency blacklist |
| sources | Git/path dependencies (block non-crates.io) |

## Minimal `deny.toml` Template

Create at workspace root if missing:

```toml
[advisories]
vulnerability = "deny"
unmaintained = "warn"
yanked = "warn"

[licenses]
allow = ["MIT", "Apache-2.0", "Apache-2.0 OR MIT", "BSD-3-Clause", "ISC"]
deny = ["GPL-2.0", "GPL-3.0", "AGPL-3.0"]
default = "deny"

[bans]
multiple-versions = "warn"
# deny = [{ name = "forbidden-crate" }]

[sources]
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
allow-git = []
```

## Triage Methodology

**Critical/High Advisories**: Upgrade or remove immediately. Use `cargo update <crate>` and re-run.

**License Violations**: 
- MIT/Apache-2.0/BSD: Usually okay
- GPL/AGPL: Requires source distribution (contamination risk)
- Proprietary: Legal review needed

**Bans**: Replace with approved alternative.

**Sources**: Require crates.io releases (git deps are security risk).

### Handling Unmaintained/Yanked Crates

1. Check crate.io for newer versions
2. If no newer version: evaluate risk assessment vs replacement
3. If waiving: document in `deny.toml` with reason:
   ```toml
   [advisories]
   ignore = ["RUSTSEC-YYYY-XXXX"]  # Reason: no patch available, low risk
   ```

## CI Integration

```bash
cargo audit --deny warnings
cargo deny check advisories licenses bans sources
```

Fail on critical/high advisories. Warn on medium. Fail on license violations.

## Load Detailed Guidance

For comprehensive supply-chain patterns and Rust-specific threats:

```
Load skill: axiom-rust-engineering:using-rust-engineering
Then read: project-structure-and-tooling.md
```

References: https://rustsec.org, https://embarkstudios.github.io/cargo-deny/

## Key Principles

- **Fail-fast security**: No critical/high CVEs in production
- **License policy**: Whitelist reflects your legal approval
- **Regular audits**: Weekly or before releases
- **Audit in CI**: Catch transitive-dependency breakage early
