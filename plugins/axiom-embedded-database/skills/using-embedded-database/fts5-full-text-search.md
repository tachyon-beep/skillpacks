---
name: fts5-full-text-search
description: Use when implementing full-text search in SQLite — virtual table creation, tokenizer selection (unicode61, porter, trigram), contentless and external-content modes, bm25 ranking with column weights, trigger-based sync, and query syntax (phrase, prefix, NEAR, column filter). Covers the sync complexity of each content mode and anti-patterns that silently leave FTS5 out of date.
---

# FTS5 Full-Text Search

**FTS5 earns its complexity only when full-text search is a first-class query mode — not when you have a search box and a LIKE query is "too slow", but when the search box is the product.**

## When this earns its cost

Read this sheet when:

- LIKE-based search is becoming the bottleneck and you are about to add `LIKE '%term%'` to a table with tens of thousands of rows.
- User-visible features need ranked results, phrase queries, prefix matching, or stemming — things a WHERE clause cannot do without a full tokenisation layer behind it.
- You are evaluating Elasticsearch and the data fits on one machine. FTS5 removes an infrastructure dependency and runs inside the same ACID transaction as the rest of your data.
- You need to search identifiers, code snippets, or file paths — candidates for the `trigram` tokenizer rather than word-boundary tokenisers.

Do not reach for FTS5 when:

- A filtered text column with a normal `LIKE 'prefix%'` index would do. A B-tree index covers left-anchored LIKE patterns; that is free and adds no maintenance burden.
- The search term is always an exact value. Use a regular index.
- Ranking, phrase queries, and stemming are not requirements. LIKE has no sync overhead.

`sqlite-fundamentals.md` covers the connection model; `pragma-discipline.md` covers the production connection block. Read those first if you have not.

## FTS5 basics

FTS5 is a built-in SQLite extension available since **SQLite 3.9.0 (2015-10-14)**. It ships compiled into the SQLite amalgamation; no separate install is required. The previous generation (FTS4/FTS3) predates 3.9.0 and should not be used on new projects — FTS5 supersedes both.

An FTS5 virtual table is both the table and its index. There is no separate `CREATE INDEX` step.

```sql
-- Create a virtual table with two searchable columns.
CREATE VIRTUAL TABLE docs_fts USING fts5(
    title,
    body,
    tokenize = 'unicode61'
);

-- Insert rows exactly as you would a normal table.
INSERT INTO docs_fts(rowid, title, body) VALUES
    (1, 'SQLite Guide',    'A complete guide to embedded databases.'),
    (2, 'FTS5 Reference',  'Full-text search for SQLite applications.');

-- Query with MATCH. The MATCH operator is the FTS5 search interface.
SELECT rowid, title
FROM   docs_fts
WHERE  docs_fts MATCH 'guide'
ORDER BY bm25(docs_fts);
```

`MATCH` on the table name searches all columns. `MATCH` on a specific column (e.g. `title MATCH 'guide'`) restricts the search.

FTS5 stores the tokenised index internally. By default it also stores the original text (the "content"). The three content modes below trade storage against sync complexity.

## Content vs contentless tables

### Default (content stored inside FTS5)

Omitting the `content` option stores both the tokenised index and the original text inside the FTS5 virtual table. This is the simplest mode.

**Trade-offs**: doubles storage for the indexed text. Writes go into the FTS5 structure, not a separate normal table. Reading back the original text works transparently.

```sql
CREATE VIRTUAL TABLE docs_fts USING fts5(title, body);
```

### Contentless (`content=''`)

The index is built and queried as normal, but FTS5 does not store the original text. A query for ranked results returns only `rowid` — you must join back to your source table to retrieve columns.

**Trade-offs**: smallest storage. Cannot use `HIGHLIGHT()` or `SNIPPET()` functions (they require stored content). Requires your application to keep the source table and the FTS5 table in sync manually or via triggers.

```sql
CREATE VIRTUAL TABLE docs_fts USING fts5(
    title,
    body,
    content = ''
);

-- After a MATCH query you get rowids only:
SELECT rowid FROM docs_fts WHERE docs_fts MATCH 'embedded';
-- Then fetch full rows from the source:
SELECT * FROM pages WHERE id IN (/* rowids from above */);
```

### External-content (`content='source_table'`)

FTS5 reads through to a regular table for column values on SELECT, but maintains its own tokenised index for MATCH queries. INSERT/UPDATE/DELETE on the FTS5 virtual table update only the index — the source table is not touched.

**Trade-offs**: no storage duplication for text. Supports `HIGHLIGHT()` and `SNIPPET()` because FTS5 can read the source table. Requires a manual `INSERT INTO docs_fts(docs_fts) VALUES('rebuild')` after any schema change to the source table, or after any batch load that bypassed FTS5.

```sql
CREATE TABLE pages (
    id    INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    body  TEXT NOT NULL
);

CREATE VIRTUAL TABLE docs_fts USING fts5(
    title,
    body,
    content      = 'pages',
    content_rowid = 'id'
);

-- Initial population:
INSERT INTO docs_fts(rowid, title, body)
    SELECT id, title, body FROM pages;

-- After schema change or bulk load, rebuild the index:
INSERT INTO docs_fts(docs_fts) VALUES('rebuild');
```

**Choosing a mode**: start with the default (content stored) unless storage is a hard constraint. Use `content=''` only when you are sure you do not need HIGHLIGHT/SNIPPET and you have the discipline to maintain sync. Use external-content when the source table is the single source of truth and you want SELECT to read from it transparently.

## Tokenizers

The tokenizer controls how text is broken into searchable tokens. Set it in the `CREATE VIRTUAL TABLE` statement; it cannot be changed afterward without dropping and recreating the table.

### `unicode61` (default)

Word-boundary tokenisation using Unicode 6.1 character categories. Splits on whitespace and punctuation; lowercases tokens; handles most Latin, Cyrillic, CJK, and other scripts correctly. The right default for natural-language content.

```sql
CREATE VIRTUAL TABLE docs_fts USING fts5(
    title,
    body,
    tokenize = 'unicode61'
);
```

Optional modifier: `remove_diacritics 2` strips combining diacritics from letters, so `résumé` matches `resume`. Omitting this defaults to `remove_diacritics 1` (removes only common ASCII-range diacritics).

```sql
tokenize = 'unicode61 remove_diacritics 2'
```

### `porter`

Applies Porter stemming on top of `unicode61`. Tokens are reduced to their stem, so `searching`, `searched`, and `searches` all match `search`. Correct for English-language content where users expect stemmed results. Reduces precision (stemming can cause false positives); worthwhile only when recall matters more than exact matching.

```sql
tokenize = 'porter unicode61'
```

Note the syntax: `porter` wraps `unicode61`. Porter is a stemmer, not a word splitter; it delegates tokenisation to `unicode61`.

### `ascii`

Word-boundary tokenisation using only ASCII rules. Ignores everything above U+0127. Exists for legacy compatibility. Do not use it for new tables that may contain non-ASCII text.

### `trigram` (SQLite 3.34.0+, 2020-12-01)

Breaks text into overlapping 3-character sequences. `"hello"` → `hel`, `ell`, `llo`. Enables substring and prefix matching without requiring the search term to be at a word boundary.

**When to use**: searching code, identifiers, file paths, log lines, or any content where users expect substring matches (`foo*` is not enough and infix matches are needed). Also useful when the language is not space-delimited.

```sql
CREATE VIRTUAL TABLE code_fts USING fts5(
    content,
    tokenize = 'trigram'
);
```

**Trade-offs**: larger index than `unicode61` for the same content (each word generates many trigrams). Slower inserts. The `porter` stemmer cannot be layered on top of `trigram`. Use `trigram` only when substring matching is the requirement; `unicode61` or `porter` for word-based natural language search.

**Version check**: `trigram` requires SQLite 3.34.0 or later. If your deployment targets older SQLite versions, you cannot use it.

## Ranking with bm25

FTS5 uses BM25 as its ranking algorithm. BM25 is a well-understood probabilistic relevance scoring model; it accounts for term frequency within a document and inverse document frequency across the corpus.

**Critical behaviour**: `bm25()` returns **negative scores**. A document with high relevance has a score closer to negative infinity; a document with low relevance has a score closer to zero. `ORDER BY bm25(table_name)` sorts ascending, which puts the most relevant rows first. Many online examples get this wrong and sort descending, reversing the ranking.

```sql
-- Correct: most relevant first.
SELECT rowid, title
FROM   docs_fts
WHERE  docs_fts MATCH 'embedded database'
ORDER BY bm25(docs_fts);

-- Wrong: least relevant first.
-- ORDER BY bm25(docs_fts) DESC
```

### Column weights

Pass per-column multipliers to weight some columns more than others. The number of weight arguments must match the number of columns in the virtual table.

```sql
-- title weighted 5×, body weighted 1×.
-- A match in the title scores five times as much as a match in the body.
SELECT rowid, title
FROM   docs_fts
WHERE  docs_fts MATCH 'guide'
ORDER BY bm25(docs_fts, 5.0, 1.0);
```

### `rank` vs `bm25()`

FTS5 virtual tables expose a `rank` column that is an alias for `bm25(table_name)` with default weights. `ORDER BY rank` works and is equivalent to `ORDER BY bm25(docs_fts)` with equal column weights. Prefer the explicit `bm25(docs_fts, ...)` form because: (a) it makes the ranking function visible in the query, and (b) it lets you pass column weights. `ORDER BY rank` ties your hands if column weighting is added later.

## Keeping FTS5 in sync with source data

FTS5's index does not automatically update when a separate source table changes. Three patterns handle this:

### Pattern 1: Triggers (recommended)

Define `AFTER INSERT`, `AFTER UPDATE`, and `AFTER DELETE` triggers on the source table. Reliable — the FTS5 index is updated inside the same transaction as the source write.

```sql
CREATE TABLE pages (
    id    INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    body  TEXT NOT NULL
);

CREATE VIRTUAL TABLE pages_fts USING fts5(
    title,
    body,
    content      = 'pages',
    content_rowid = 'id'
);

-- INSERT trigger
CREATE TRIGGER pages_ai AFTER INSERT ON pages BEGIN
    INSERT INTO pages_fts(rowid, title, body)
    VALUES (new.id, new.title, new.body);
END;

-- DELETE trigger
CREATE TRIGGER pages_ad AFTER DELETE ON pages BEGIN
    INSERT INTO pages_fts(pages_fts, rowid, title, body)
    VALUES ('delete', old.id, old.title, old.body);
END;

-- UPDATE trigger
CREATE TRIGGER pages_au AFTER UPDATE ON pages BEGIN
    INSERT INTO pages_fts(pages_fts, rowid, title, body)
    VALUES ('delete', old.id, old.title, old.body);
    INSERT INTO pages_fts(rowid, title, body)
    VALUES (new.id, new.title, new.body);
END;
```

**Cost**: each write on `pages` incurs a second write into `pages_fts`. Acceptable for most workloads; measure if write throughput is a bottleneck.

### Pattern 2: Application-level dual writes

The application explicitly inserts/deletes from both `pages` and `pages_fts` in every code path that modifies `pages`. Wrap both in a transaction.

**Drawback**: fragile. Any write path that bypasses the application layer (a migration script, a CLI import, a direct SQLite shell command) leaves FTS5 out of sync. Prefer triggers for this reason.

### Pattern 3: External-content with periodic rebuild

Use `content='pages'`, and after any batch write that bypasses FTS5, rebuild the index:

```sql
INSERT INTO pages_fts(pages_fts) VALUES('rebuild');
```

Acceptable for infrequent batch loads. Not suitable for real-time sync. A schema change to `pages` also requires a rebuild before FTS5 reads the new shape correctly.

## Query syntax

FTS5 MATCH expressions use a query language distinct from SQL. Key forms:

```sql
-- Simple term: matches documents containing 'sqlite'
WHERE docs_fts MATCH 'sqlite'

-- Phrase: matches 'full text' as a sequence
WHERE docs_fts MATCH '"full text"'

-- Prefix: matches 'embed', 'embedded', 'embedding', ...
WHERE docs_fts MATCH 'embed*'

-- AND (implicit): both terms must appear
WHERE docs_fts MATCH 'sqlite guide'

-- OR: either term
WHERE docs_fts MATCH 'sqlite OR postgres'

-- NOT: first term present, second absent
WHERE docs_fts MATCH 'sqlite NOT fts4'

-- Column filter: term must appear in 'title' column
WHERE docs_fts MATCH 'title:guide'

-- NEAR: terms within N tokens of each other (default N=10)
WHERE docs_fts MATCH 'NEAR(full text, 3)'

-- Combining forms
WHERE docs_fts MATCH 'title:"full text" body:search*'
```

Unquoted terms are AND-ed together by default. Phrase queries require double quotes inside the MATCH string.

## Worked example: wiki-page search

A wiki where pages have a title and a body. Title matches should rank higher than body matches. Search must support phrase queries and prefix completion. Updates to pages must keep FTS5 in sync automatically.

```sql
-- Source table
CREATE TABLE pages (
    id         INTEGER PRIMARY KEY,
    slug       TEXT    NOT NULL UNIQUE,
    title      TEXT    NOT NULL,
    body       TEXT    NOT NULL,
    updated_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- FTS5 virtual table with external-content and porter stemming.
-- title weighted 5× in bm25 calls.
CREATE VIRTUAL TABLE pages_fts USING fts5(
    title,
    body,
    content       = 'pages',
    content_rowid  = 'id',
    tokenize      = 'porter unicode61 remove_diacritics 2'
);

-- Sync triggers
CREATE TRIGGER pages_ai AFTER INSERT ON pages BEGIN
    INSERT INTO pages_fts(rowid, title, body)
    VALUES (new.id, new.title, new.body);
END;

CREATE TRIGGER pages_ad AFTER DELETE ON pages BEGIN
    INSERT INTO pages_fts(pages_fts, rowid, title, body)
    VALUES ('delete', old.id, old.title, old.body);
END;

CREATE TRIGGER pages_au AFTER UPDATE ON pages BEGIN
    INSERT INTO pages_fts(pages_fts, rowid, title, body)
    VALUES ('delete', old.id, old.title, old.body);
    INSERT INTO pages_fts(rowid, title, body)
    VALUES (new.id, new.title, new.body);
END;

-- Search query: phrase + prefix, ranked by title-weighted bm25.
-- Returns up to 20 results with a snippet from the body.
SELECT
    p.id,
    p.slug,
    p.title,
    snippet(pages_fts, 1, '<b>', '</b>', '...', 20) AS excerpt,
    bm25(pages_fts, 5.0, 1.0)                       AS score
FROM   pages_fts
JOIN   pages p ON p.id = pages_fts.rowid
WHERE  pages_fts MATCH '"full text" OR search*'
ORDER BY bm25(pages_fts, 5.0, 1.0)
LIMIT  20;
```

`snippet(table, column_index, open_tag, close_tag, ellipsis, token_count)` works because external-content mode reads through to `pages` for the original text.

## Anti-patterns

### Using FTS5 when LIKE would do

A left-anchored LIKE (`WHERE title LIKE 'guide%'`) is covered by a standard B-tree index and has zero maintenance overhead. FTS5 adds two write paths (triggers or dual writes), a second copy or index structure, and rebuild complexity. Reach for FTS5 when ranking, phrase queries, stemming, or infix matching are actual requirements — not as a first response to a slow LIKE query.

### Not using triggers and forgetting to update FTS5

If you choose application-level dual writes, any write path that bypasses the application — a migration script, a direct shell command, a bulk import — leaves FTS5 silently stale. Users see search results that no longer match the source data. Use triggers so that sync is enforced at the database layer regardless of who writes to the table.

### Using `ORDER BY rank` when you need column weights

`ORDER BY rank` works and sorts by `bm25(table)` with equal column weights. Once you add column weights (`bm25(docs_fts, 5.0, 1.0)`), `rank` no longer reflects your intent. Write `ORDER BY bm25(table_name, ...)` explicitly from the start so adding weights later is a one-line change, not a refactor.

### Using `unicode61` for identifier or code search

`unicode61` tokenises on word boundaries. An identifier like `getUserByEmail` tokenises to two tokens (`getuserbyemail` — actually one token in this case — or splits at underscores in `get_user_by_email`). Users searching for `Email` as a substring will not find `getUserByEmail`. Use the `trigram` tokenizer for code and identifier search so infix matches work. Requires SQLite 3.34.0+.

### Specifying `remove_diacritics` incorrectly for Unicode text

The default `remove_diacritics 1` strips only a subset of common Latin diacritics. Text with Arabic, Greek, or less common Latin diacritics will not be normalised, causing `résumé` ≠ `resume` mismatches on some Unicode ranges. If your content may contain non-ASCII diacritics and you want diacritic-insensitive matching, set `remove_diacritics 2` explicitly.

### External-content table out of sync after schema change

If you rename or drop a column on the source table, the FTS5 external-content index is referencing column positions from when it was created. Queries will silently return wrong results or raise errors. After any schema change to the source table, run `INSERT INTO table_fts(table_fts) VALUES('rebuild')` before any FTS5 queries. Add this to the schema migration procedure alongside the table rebuild steps in `schema-migrations.md`.

### Forgetting that bm25 returns negative scores

`ORDER BY bm25(table) DESC` sorts least relevant first. The correct form is `ORDER BY bm25(table)` (ascending, the default), which puts the most negative (most relevant) score first. This is the most common copy-paste error in FTS5 ranking code.

## Cross-references

- `schema-migrations.md` — required reading before adding FTS5 to a schema with an existing migration history; external-content tables must be rebuilt after any source table rebuild.
- `json1-and-structured-data.md` — if the searchable content lives in a JSON column, extract it into a generated column before feeding it to FTS5.
- `parameterized-sql-only.md` — MATCH values must still be passed as bound parameters; never interpolate search strings into SQL.
- `pragma-discipline.md` — the production connection block that gates FTS5's storage and WAL behaviour.
- `boundary-and-when-to-leave.md` — when search requirements have outgrown SQLite and a dedicated search engine is warranted.
