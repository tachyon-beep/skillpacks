---
name: encryption-with-sqlcipher
description: Use when you need to protect a SQLite database file at rest — covers threat-model scoping, SQLCipher 4.x cryptographic construction, PBKDF2 key derivation, key rotation via rekey, building Python wheels, operational hazards, and anti-patterns. The correct mental model: SQLCipher protects the cold artifact, not the running process.
---

# Encryption with SQLCipher

**SQLCipher encrypts the bytes at rest. It does not address an attacker with process memory access, a compromised host, or your own bad key-management. Match the threat model first, then pick the tool.**

## When this earns its cost

Read this sheet when:

- You are building an application that stores sensitive data in a SQLite file and your threat model is **device theft** or **backup-copy exfiltration** — the attacker gets the `.db` file, not a running shell.
- You need to satisfy a compliance requirement that mandates encryption at rest for local data stores (HIPAA, PCI-DSS local storage clauses, GDPR technical measures).
- You are inheriting a codebase that uses SQLCipher and need to understand what PRAGMAs to apply on every connection, what `cipher_page_size` consistency means, and how rekey works.
- You are evaluating SQLCipher against the alternative of filesystem-level encryption (LUKS, FileVault, BitLocker) and need to know when each fits.

Do not read this sheet expecting SQLCipher to solve a runtime-attacker problem. If the attacker has code execution on the host while your application is running, they can read the decrypted pages from the page cache. SQLCipher cannot help with that.

## Threat model — what SQLCipher addresses

SQLCipher's protection is precisely scoped. Be exact when communicating it.

**What SQLCipher addresses:**

- An attacker who physically obtains the `.db` file — stolen laptop, pulled storage device, cloud-snapshot exfiltration — but does not have the key or a running process with the key in memory.
- Backup copies of the database file stored in locations with weaker access controls than the production host (object storage, tape, email attachment sent by mistake).
- Log or forensic tools that dump disk contents without understanding the application layer.

**What SQLCipher does not address:**

- An attacker with a process memory dump of a running application. The decrypted page cache lives in RAM; a memory dump gives them plaintext pages.
- A compromised host where the key store (OS keychain, secrets manager) is also accessible. SQLCipher is only as strong as the weakest link in your key retrieval path.
- Exception traces or log lines that print the key. If `PRAGMA key = 'my-secret'` is logged, the encryption is worthless.
- Side-channel attacks on disk I/O timing or filesystem metadata (file size, access timestamps).
- Column-level or row-level access control. SQLCipher is whole-file — once the file is open to a valid key holder, all rows are readable.
- A threat model where the runtime must be untrusted. That requires hardware enclaves or HSMs; SQLCipher cannot help there.

Stating what it does not address is not a weakness — it is honest threat modelling. A team that deploys SQLCipher without this scoping will over-trust it in exactly the wrong places.

## What SQLCipher actually does

SQLCipher is a SQLite extension that encrypts every database page before it is written to disk and decrypts it after it is read. The cryptographic construction in SQLCipher 4.x:

- **Algorithm**: AES-256-CBC per page.
- **Integrity**: HMAC-SHA512 per page. SQLCipher derives a separate HMAC subkey from the master key material. The construction is encrypt-then-MAC — the HMAC is computed over the ciphertext, not the plaintext. This is correct MAC discipline; it is not AEAD (there is no integrated nonce, and authentication and encryption are handled by separate key material and separate operations).
- **Coverage**: The entire file is encrypted — including the SQLite header, all data pages, the journal (DELETE mode), and the WAL file (WAL mode). SQLCipher handles the WAL and `-shm` files only when the connection is opened through SQLCipher. A plain `sqlite3` open of an SQLCipher file writes an unencrypted WAL.
- **Default page size**: 4096 bytes in SQLCipher 4.x. Every page is independently encrypted; there is no file-wide IV.
- **KDF**: PBKDF2-HMAC-SHA512 with 256,000 iterations by default in SQLCipher 4.x. Earlier major versions had weaker defaults — SQLCipher 1.x used 4,000 iterations; SQLCipher 3.x used 64,000. If you open a database created with an older SQLCipher, the KDF parameters travel with the file. Upgrading the library does not upgrade the KDF of existing files; you must rekey.

**Performance overhead** is real but bounded:

- Read: 5–15% throughput overhead for I/O-bound workloads. The bottleneck is AES and HMAC computation per page, not I/O wait.
- Write: 10–20% overhead. Writes encrypt before flushing; WAL writes see a similar multiplier.
- CPU-bound workloads with hot data in the page cache see negligible overhead — the encryption cost is paid once on the page read; subsequent access to cached pages is free.

## Key derivation: PBKDF2 and beyond

SQLCipher derives the AES key and HMAC key from a passphrase using PBKDF2. There are two calling conventions:

```sql
-- Passphrase (run through PBKDF2, 256K iterations)
PRAGMA key = 'my-application-passphrase';

-- Raw 256-bit key (64 hex chars, bypasses PBKDF2 entirely)
PRAGMA key = "x'2a3f8b1c...64-hex-chars'";
```

The raw-key form is appropriate when your application already holds a high-entropy secret (e.g., a key fetched from an HSM or secrets manager) and you do not want to spend 256K PBKDF2 rounds deriving something equivalent to what you already have.

**Production key sourcing.** The key must come from outside the application binary. Acceptable sources, in order of preference:

1. Hardware security module (HSM) or cloud KMS (AWS KMS, GCP Cloud KMS, Azure Key Vault).
2. OS-managed credential store: macOS Keychain, Windows Credential Manager, Linux Secret Service (libsecret) or DPAPI.
3. A separately-deployed secrets manager (Vault, AWS Secrets Manager) with short-lived leases and audit logging.
4. An environment variable set by an orchestrator (acceptable at low sensitivity; visible via `/proc/<pid>/environ` on Linux — see hazards).

Never hardcode the key, never derive it from a username or hostname, never store it adjacent to the database file.

**Key derivation helper (Python):**

The simplest cross-platform retrieval is `keyring.get_password(service, account)` — backed by macOS Keychain, Windows Credential Manager, or Linux Secret Service. Wrap it in a helper that raises (rather than returning `None`) when the key is absent, so a missing key fails fast at connection time rather than silently producing a wrong-key error one statement later. The worked example below shows the canonical shape.

**Opening a SQLCipher connection (Python, using `sqlcipher3` — community-maintained; `pysqlcipher3` is an equivalent alternative).** The shape of the helper:

1. Open the connection (vanilla `sqlcipher3.connect(path)`).
2. Set `PRAGMA key` as the *first* statement. **`PRAGMA key` does not accept bound parameters** — you must inline the key, and you must escape any embedded `'` characters. See the worked example below for the `_quote_sql_literal` helper and the full PRAGMA block (`cipher_page_size`, `kdf_iter`, `cipher_hmac_algorithm`, `cipher_kdf_algorithm`, plus the standard `journal_mode`/`foreign_keys`/`busy_timeout`).
3. Issue a smoke query (`SELECT count(*) FROM sqlite_master`) to fail fast on a wrong key — SQLCipher reports a wrong key as `SQLITE_NOTADB` on the first real read, not on the `PRAGMA key` call itself.

The four `cipher_*` PRAGMAs are idempotent on SQLCipher 4.x where they match the defaults, but setting them explicitly documents your intent and guards against a library version that ships different defaults. The full helper appears in the worked example.

## Key rotation: rekey

`PRAGMA rekey` re-encrypts every page in the database with a new key. The operation is:

1. Opens the existing database with the current key.
2. Re-encrypts each page in-place using the new key.
3. Truncates and replaces the KDF salt.
4. The new key is active for all subsequent connections.

The operation is proportional to database size and holds an exclusive write lock for its duration. For databases over ~500 MB, plan a maintenance window or perform rotation offline.

**Safe rekey procedure** (four steps; see the full implementation in the worked example below):

1. **Backup first** — `shutil.copy2(db, db.bak)`. If anything goes wrong, you restore from this and the operation was a no-op.
2. **Open with the current key**, apply PRAGMAs, run a smoke query to confirm the current key is correct, then `PRAGMA rekey = '<new_key>'`. The same quoting hazard applies to `rekey` as to `key`: escape `'` in the new passphrase, or use the hex form.
3. **Verify the new key** by closing the connection and reopening with the new key plus a smoke query.
4. **Only after verification**, delete the backup and update the credential store. If verification fails, restore from backup and raise.

**Removing encryption** (converting an encrypted file to plaintext) uses `PRAGMA rekey = ''` — an empty string passphrase, which SQLCipher interprets as "no encryption" and rewrites the file as a vanilla SQLite database.

**Encrypting an existing plaintext SQLite database** is the inverse and uses `sqlcipher_export`. The direction matters and is easy to get backwards — `sqlcipher_export` exports the contents of the *main* connection into the *attached* database, so the *main* connection determines the source and the *attached* database determines the destination:

```sql
-- Source: plaintext main connection. Destination: encrypted attachment.
-- (Opened from a SQLCipher-aware client with NO PRAGMA key on the main.)
ATTACH DATABASE 'encrypted.db' AS encrypted KEY 'new-passphrase';
SELECT sqlcipher_export('encrypted');
DETACH DATABASE encrypted;
```

**Decrypting an encrypted SQLCipher database to a plaintext file** is the mirror — main is encrypted, attachment is plaintext (`KEY ''` means no key on the attached database):

```sql
-- Source: encrypted main connection. Destination: plaintext attachment.
-- (Opened from a SQLCipher-aware client WITH PRAGMA key on the main.)
ATTACH DATABASE 'plain.db' AS plaintext KEY '';
SELECT sqlcipher_export('plaintext');
DETACH DATABASE plaintext;
```

The two snippets differ only in which side has the key — but the operational consequence is opposite. Read the comment header carefully before running either.

## Building SQLCipher wheels

You cannot `pip install sqlcipher3` into an arbitrary environment and expect it to work. Both major Python wrappers — `pysqlcipher3` and `sqlcipher3` — are community-maintained thin wrappers around the SQLCipher C library. They must be built with the SQLCipher source (not the vanilla SQLite amalgamation) and linked against OpenSSL for the cryptographic primitives.

**Why pre-built wheels often fail:**

- The C extension must be compiled with `-DSQLITE_HAS_CODEC`. Wheels built without this flag silently ignore `PRAGMA key` — the database opens without error and without encryption.
- ARM platforms (Apple Silicon, aarch64 Linux), musl libc (Alpine, container base images), and older glibc versions frequently lack matching pre-built wheels.
- OpenSSL version mismatches between the build environment and the runtime environment cause import-time linker errors.

**Build from source on Linux x86_64:**

```bash
# Install system dependencies.
apt-get install -y build-essential libsqlcipher-dev libssl-dev

# Install the Python wrapper.
pip install sqlcipher3
# If libsqlcipher-dev provides the shared library, this links against it.
# Verify the build linked correctly:
python -c "import sqlcipher3; c = sqlcipher3.connect(':memory:'); \
           c.execute(\"PRAGMA key = 'test'\"); \
           c.execute(\"SELECT sqlcipher_version()\").fetchone()"
```

**Build from source on aarch64 (e.g., Raspberry Pi, ARM server):**

```bash
# SQLCipher may not be in the package manager on all aarch64 distros.
# Build SQLCipher from source, then build the Python wrapper against it.
apt-get install -y build-essential libssl-dev tcl-dev

git clone https://github.com/sqlcipher/sqlcipher.git
cd sqlcipher
./configure --enable-tempstore=yes \
            CFLAGS="-DSQLITE_HAS_CODEC" \
            LDFLAGS="-lcrypto"
make && make install

# Now install the Python wrapper pointing at the custom install.
SQLCIPHER_PATH=/usr/local pip install sqlcipher3
```

**Validation after any build:**

```python
import sqlcipher3
conn = sqlcipher3.connect(":memory:")
conn.execute("PRAGMA key = 'smoke-test'")
version = conn.execute("SELECT sqlcipher_version()").fetchone()
print(f"SQLCipher version: {version[0]}")
# Should print "4.x.x ..." — if this raises OperationalError, the build
# did not include SQLCipher; you may have linked vanilla SQLite instead.
```

If `SELECT sqlcipher_version()` raises `OperationalError: no such function: sqlcipher_version`, the library is vanilla SQLite. The `PRAGMA key` silently succeeded — it was accepted as an unknown PRAGMA and ignored. Your data is not encrypted.

## Operational hazards

Five mistakes that undermine SQLCipher in production:

**1. Hardcoded key in source.**
`PRAGMA key = 'hunter2'` checked into a repository. The database is "encrypted" in the sense that an attacker needs a hex editor plus the README. Detect with: `grep -r "PRAGMA key" .` in CI. Key material belongs in a secrets store, never in source.

**2. Key in environment variable without OS-level secret management.**
Environment variables are readable from `/proc/<pid>/environ` by any process with the same UID. On shared-tenant hosts (container runtimes without PID namespace isolation, CI runners), this is a meaningful exposure. Prefer the OS keychain or a secrets manager with lease-based access. If env vars are unavoidable, set them from a credential file that is read once at startup and then overwritten in memory.

**3. Forgetting that `-shm` and `-wal` files are only encrypted by SQLCipher-aware connections.**
If any code path opens the `.db` file using vanilla `sqlite3` while the WAL contains uncommitted transactions, those WAL pages are written in plaintext. This is most likely in migration scripts, diagnostic tools, or test helpers that import `sqlite3` instead of `sqlcipher3`. Enforce a single connection helper that always applies `PRAGMA key` and `PRAGMA journal_mode` on every open.

**4. Backups via plain `cp` of the encrypted file.**
This is correct (the copy is still encrypted) if the backup key is the same key. The hazard is key management: if the key is later rotated, old backups require the old key to restore. Document the key epoch alongside each backup. A restore procedure that calls `PRAGMA key` with the wrong key returns `SQLITE_NOTADB`.

**5. Exception traces leaking the key.**
If `PRAGMA key = '{key}'` is inside an f-string and the connection raises an exception, some exception handlers log the full SQL text. The key appears in log aggregators, sentry, or stdout in plaintext. Sanitize the exception before logging, or use a parameterized key-setting wrapper that does not embed the key in the SQL string. (SQLCipher does not support bound parameters for `PRAGMA key`; the sanitization must happen at the exception handler level.)

## When NOT to use SQLCipher

SQLCipher is the right tool in a narrow threat model. Do not use it when:

- **The host is untrusted at runtime.** If a hypervisor, container orchestrator, or shared-tenant service can read your process memory, encrypting the file is cosmetic. Use a hardware enclave (Intel TDX, AMD SEV, AWS Nitro Enclaves) or an HSM-backed protocol. SQLCipher cannot help.
- **You need column-level or row-level access control.** SQLCipher is whole-file. Once the file is open to a valid key holder, every row is accessible. Column-level encryption requires application-layer field encryption (e.g., encrypt specific fields before writing them, decrypt after reading). These can coexist — file-level SQLCipher plus application-level field encryption — but they serve different threat models.
- **You need searchable encryption or range queries over encrypted values.** SQLCipher encrypts pages; the index structures that support WHERE clauses are also encrypted. A SQLCipher-backed database can answer range queries on plaintext values because the query runs on decrypted pages in memory — but if you need a third party to execute queries without seeing the data (searchable symmetric encryption), SQLCipher is not that tool.
- **The threat model already requires trusting the runtime.** If the attacker must be excluded from the running process and the OS, filesystem encryption (LUKS, FileVault, BitLocker) is a simpler and broader control than SQLCipher. Filesystem encryption protects every file on the volume, has no application dependency, and is managed by the OS. SQLCipher makes sense when you need **application-controlled granularity** — different databases keyed differently, per-tenant keys in a multi-tenant app, or a mobile app where the OS keychain holds the key and revocation must be fast.
- **You are adding SQLCipher to satisfy a compliance checkbox without modelling the actual threat.** Compliance auditors sometimes accept "database encrypted at rest" without specifying the threat model. If you deploy SQLCipher and then leave the key in a `.env` file next to the database, you have checked the box and protected nothing. Do the threat model first.

## Worked example

A complete Python helper for opening an encrypted database and rotating the key, using `sqlcipher3` and the OS keychain via `keyring`:

```python
"""
encrypted_db.py — SQLCipher connection management.
Requires: pip install sqlcipher3 keyring
"""
import contextlib
import shutil
from pathlib import Path
from typing import Generator
import sqlcipher3
import keyring

_SERVICE = "myapp"
_ACCOUNT = "db-key"


def _load_key() -> str:
    key = keyring.get_password(_SERVICE, _ACCOUNT)
    if not key:
        raise RuntimeError("Database key not found in OS keychain.")
    return key


def _quote_sql_literal(value: str) -> str:
    """
    Escape a string for safe inlining into a SQL single-quoted literal.
    SQLCipher's PRAGMA key / PRAGMA rekey do NOT accept bound parameters,
    so the passphrase must be inlined. A passphrase containing an
    apostrophe will otherwise terminate the literal early and either
    raise a syntax error or — worse — appear as a wrong-key error.

    If your key sourcing guarantees high-entropy hex (no apostrophes
    possible), you can use the raw-key form instead:
        PRAGMA key = "x'<64-hex-chars>'"
    """
    return value.replace("'", "''")


def _apply_pragmas(conn: sqlcipher3.Connection, key: str) -> None:
    """
    Apply PRAGMA key and all cipher configuration.
    Must be called immediately after opening the connection,
    before any other statement.
    """
    quoted = _quote_sql_literal(key)
    conn.execute(f"PRAGMA key = '{quoted}'")
    # Explicit cipher configuration — do not rely on defaults surviving a
    # library upgrade or a file created on a different platform.
    conn.execute("PRAGMA cipher_page_size = 4096")
    conn.execute("PRAGMA kdf_iter = 256000")
    conn.execute("PRAGMA cipher_hmac_algorithm = HMAC_SHA512")
    conn.execute("PRAGMA cipher_kdf_algorithm = PBKDF2_HMAC_SHA512")
    # Verify the key is correct. If wrong, this raises OperationalError
    # (SQLITE_NOTADB) on the first real read.
    conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    # Standard production PRAGMAs.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")


@contextlib.contextmanager
def open_db(path: str) -> Generator[sqlcipher3.Connection, None, None]:
    """
    Context manager that opens an encrypted SQLite database.
    Usage:
        with open_db("myapp.db") as conn:
            conn.execute("SELECT ...")
    """
    key = _load_key()
    conn = sqlcipher3.connect(path)
    try:
        _apply_pragmas(conn, key)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def rotate_key(path: str, new_key: str) -> None:
    """
    Rotate the database key. Updates the OS keychain after successful rekey.
    Call this during a scheduled maintenance window for large databases.

    Backup-first discipline: if anything fails, the original file is
    restored and the keychain is not updated.
    """
    db = Path(path)
    backup = db.with_suffix(".db.bak")
    current_key = _load_key()
    new_key_quoted = _quote_sql_literal(new_key)

    shutil.copy2(db, backup)
    conn = sqlcipher3.connect(str(db))
    try:
        _apply_pragmas(conn, current_key)
        conn.execute(f"PRAGMA rekey = '{new_key_quoted}'")
        conn.close()
        # Verify the new key before updating the keychain.
        conn = sqlcipher3.connect(str(db))
        _apply_pragmas(conn, new_key)
        conn.close()
        # Key verified — update the keychain, remove backup.
        keyring.set_password(_SERVICE, _ACCOUNT, new_key)
        backup.unlink()
    except Exception:
        conn.close()
        shutil.copy2(backup, db)
        raise
```

**Logging discipline.** Notice that no helper above ever interpolates the key into a log message or an exception. If `_apply_pragmas` raises (wrong key, mismatched page size), the traceback contains the SQL text — `PRAGMA key = '<quoted>'` — with the key visible. Wrap callers in a `try/except` that catches `sqlcipher3.OperationalError`, strips the SQL text, and re-raises as a sanitised error type (e.g., `class DatabaseKeyError(Exception): pass`) before any log handler sees the original exception.

## Anti-patterns

**Hardcoded key.**
`PRAGMA key = 'password123'` in the application. The database file is encrypted; the key is in the repository. Any developer, CI log, or attacker with source access has the key. The fix: always load the key from an external credential store. Never concatenate a literal into `PRAGMA key`.

**Deriving the key from a username or hostname.**
`key = hashlib.sha256(username.encode()).hexdigest()` produces a low-entropy key if the username space is small or predictable. A key derived from input an attacker can enumerate is not a secret. Keys must come from a cryptographically random source with at least 128 bits of entropy.

**Reusing one key across many databases.**
A single compromised key decrypts every database in the fleet. Per-tenant or per-database keys limit blast radius. The key management overhead is real but necessary for high-sensitivity workloads.

**Logging the key in an exception trace.**
`conn.execute(f"PRAGMA key = '{key}'")` inside a try block whose except clause logs `str(e)` or the SQL text. The key appears in your log aggregator in plaintext. Wrap the PRAGMA call in a helper that catches exceptions, strips the SQL text, and re-raises a generic `DatabaseKeyError`. Never pass raw key material through a logging path.

**Mismatched `cipher_page_size` between creator and opener.**
A database created with `PRAGMA cipher_page_size = 4096` and opened with an implicit `cipher_page_size = 1024` (e.g., an older default or a different library version) returns `SQLITE_NOTADB`. The error looks like a wrong key. Set `cipher_page_size` explicitly on every connection. Enforce it in your single connection helper so no caller can accidentally omit it.

**Using SQLCipher to satisfy a compliance checkbox without modelling the actual threat.**
Encrypting the database file while leaving the key in a `.env` file adjacent to the database, logging it on startup, or storing it in an unprotected config table in the same database provides no meaningful security. Compliance evidence should include the key management design, not just the presence of SQLCipher. Audit the key path, not just the encryption flag.

## Cross-references

- `pragma-discipline.md` — `cipher_page_size`, `kdf_iter`, `cipher_hmac_algorithm`, and `cipher_kdf_algorithm` are cipher PRAGMAs that must be set consistently on every connection; the standard PRAGMA block (`journal_mode`, `foreign_keys`, `busy_timeout`) applies inside an encrypted connection the same way it applies to a plaintext one.
- `concurrent-access-patterns.md` — encrypting a database file does not make it safe to share over a network filesystem; NFS and SMB locking semantics are unchanged by SQLCipher. The WAL-mode caveats about multiple processes apply identically.
- `backup-restore-and-corruption.md` — backup of an encrypted database requires key-epoch bookkeeping; restore requires the key that was active at backup time, not the current key.
- `sqlite-fundamentals.md` — the connection model, WAL, and journal-mode semantics that SQLCipher extends; read this before configuring an encrypted connection for the first time.
- `boundary-and-when-to-leave.md` — if the threat model requires column-level access control, searchable encryption, or exclusion of a compromised runtime, SQLite (encrypted or not) is the wrong tool.
