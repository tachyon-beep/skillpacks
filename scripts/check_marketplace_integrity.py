#!/usr/bin/env python3
"""Marketplace cross-reference integrity checker.

Since `1472fc9` retired the runtime "Cross-Pack Discovery" globs in favour of
static "Related Packs" prose, nothing in the repository verifies that a
cross-pack reference still points at something real. Static text cannot notice
that a pack was renamed, or that the pack it promised as "future" now ships
under a different name. This script is that missing check.

It deliberately does NOT validate relative `.md` links: a sweep of 2494 of them
found zero genuine breaks and ~28 intentional illustrations (a technical-writing
sheet showing what a `docs/` tree looks like, a modding sheet naming example mod
pages). Flagging those would make the checker noise, and a noisy checker gets
ignored. Pack names and slash commands are unambiguous; those are what it checks.

Usage:
    python3 scripts/check_marketplace_integrity.py          # report + exit 1 on error
    python3 scripts/check_marketplace_integrity.py --warn   # never exit nonzero
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGINS = os.path.join(ROOT, "plugins")
FACTIONS = ("axiom", "yzmir", "lyra", "muna", "ordis", "bravos", "meta")

# Faction-prefixed tokens that are ordinary English or third-party names, not
# pack references. Keep this list short and justified -- every entry is a place
# the checker is deliberately blind.
NOT_PACK_NAMES = {
    "meta-learning",      # the ML technique
    "meta-learner",
    "meta-skill",         # "a meta-skill", generic prose
    "meta-skills",
    "meta-llama",         # Meta's Llama model IDs
    "meta-game", "meta-games", "meta-gaming",  # game-design vocabulary
    "meta-controller",    # hierarchical-RL vocabulary
    "meta-refresh",       # the HTML tag
    "meta-router",        # architectural noun in the AI router
    "meta-summary",
    "meta-decision", "meta-decisions",
    "meta-trap",
    "meta-pass",
    "meta-analyzer",
    "meta-optimizer",
    "meta-prompt",
}

# A reference may name a pack that does not exist yet, provided the same line
# says so. This is the house convention and it is load-bearing: it lets a sheet
# route forward to planned work without pretending it shipped.
FORWARD_MARKERS = ("proposed", "planned", "not yet in the marketplace", "not yet implemented", "future")

PACK_TOKEN = re.compile(r"\b(?:" + "|".join(FACTIONS) + r")-[a-z0-9]+(?:-[a-z0-9]+)*\b")

# Slash commands are backtick-wrapped in house style. That alone is not enough:
# the API packs write REST endpoints the same way (`/users`, `/orders`). Every
# real command in this marketplace is hyphenated or namespaced, and no REST
# endpoint in the corpus is -- so requiring a hyphen or colon separates them
# cleanly. Cost: a broken reference to a hypothetical single-word command would
# slip through. That is the deliberate trade for a checker with no false alarms.
SLASH_CMD = re.compile(r"`/([a-z][a-z0-9]*(?:[:-][a-z0-9]+)+)`")

# Hyphenated slash-tokens that are legitimately not commands. Listed explicitly
# rather than loosening the regex, so each blind spot stays visible and auditable.
NOT_COMMANDS = {
    "user-orders",    # a REST anti-pattern example in axiom-web-backend
    "triage-issues",  # an MCP *prompt* primitive, illustrating prompts vs tools
}

FENCE = re.compile(r"^\s*```")
# `[text](#anchor)` -- GitHub slugifies headings, so `## Interaction with
# `axiom-sdlc-engineering/design-and-build`` legitimately yields the anchor
# `#interaction-with-axiom-sdlc-engineeringdesign-and-build`. Anchors are not
# pack references and must not be scanned as such.
ANCHOR = re.compile(r"\]\(#[^)]*\)")


def strip_fences(lines: list[str]) -> list[str]:
    """Blank out fenced code blocks, preserving line numbering."""
    out, inside = [], False
    for line in lines:
        if FENCE.match(line):
            inside = not inside
            out.append("")
            continue
        out.append("" if inside else line)
    return out


def md_files(base: str):
    for dirpath, _, names in os.walk(base):
        for n in names:
            if n.endswith(".md"):
                yield os.path.join(dirpath, n)


def rel(p: str) -> str:
    return os.path.relpath(p, ROOT)


def load_state():
    dirs = sorted(
        d for d in os.listdir(PLUGINS)
        if os.path.isdir(os.path.join(PLUGINS, d)) and not d.startswith(".")
    )
    with open(os.path.join(ROOT, ".claude-plugin", "marketplace.json"), encoding="utf-8") as fh:
        market = json.load(fh)
    market_names = [p["name"] for p in market.get("plugins", [])]

    commands = {
        os.path.basename(f)[:-3]
        for f in md_files(os.path.join(ROOT, ".claude", "commands"))
    }
    for pack in dirs:
        cdir = os.path.join(PLUGINS, pack, "commands")
        if os.path.isdir(cdir):
            for f in os.listdir(cdir):
                if f.endswith(".md"):
                    commands.add(f[:-3])
                    commands.add(f"{pack}:{f[:-3]}")
    return dirs, market_names, commands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warn", action="store_true", help="always exit 0")
    args = ap.parse_args()

    dirs, market_names, commands = load_state()
    packs = set(dirs)
    errors: list[str] = []
    warnings: list[str] = []

    # ---- 1. marketplace.json <-> plugin dirs <-> plugin.json --------------
    for name in sorted(set(market_names) - packs):
        errors.append(f"marketplace.json lists '{name}' but plugins/{name}/ does not exist")
    for name in sorted(packs - set(market_names)):
        errors.append(f"plugins/{name}/ exists but is not registered in marketplace.json")
    dupes = [n for n, c in Counter(market_names).items() if c > 1]
    for n in sorted(dupes):
        errors.append(f"marketplace.json registers '{n}' more than once")

    for pack in dirs:
        meta = os.path.join(PLUGINS, pack, ".claude-plugin", "plugin.json")
        if not os.path.exists(meta):
            errors.append(f"{pack}: missing .claude-plugin/plugin.json")
            continue
        try:
            with open(meta, encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            errors.append(f"{pack}: plugin.json is not valid JSON ({exc})")
            continue
        if data.get("name") != pack:
            errors.append(
                f"{pack}: plugin.json name is '{data.get('name')}', expected '{pack}'"
            )

    # ---- 2. pack-name references resolve ---------------------------------
    referenced: Counter = Counter()
    for path in md_files(PLUGINS):
        with open(path, encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
        owner = rel(path).split(os.sep)[1]
        scan = [ANCHOR.sub("", ln) for ln in lines]
        # A pack may annotate a forward reference once and then use the bare
        # name in the same file -- that reads correctly to a human, so treat the
        # annotation as covering the file rather than only its own line.
        annotated = {
            tok
            for ln in scan
            if any(m in ln.lower() for m in FORWARD_MARKERS)
            for tok in PACK_TOKEN.findall(ln)
        }
        for lineno, line in enumerate(scan, 1):
            for token in set(PACK_TOKEN.findall(line)):
                if token in packs:
                    if token != owner:  # don't count a pack citing itself
                        referenced[token] += 1
                    continue
                if token in NOT_PACK_NAMES or token in annotated:
                    continue
                errors.append(
                    f"{rel(path)}:{lineno}: references '{token}', which is not a pack "
                    f"and is not marked (proposed)/(planned) anywhere in this file"
                )

    # ---- 3. slash commands resolve ---------------------------------------
    for path in md_files(PLUGINS):
        with open(path, encoding="utf-8", errors="ignore") as fh:
            lines = strip_fences(fh.readlines())
        for lineno, line in enumerate(lines, 1):
            for cmd in set(SLASH_CMD.findall(line)):
                if cmd in commands or cmd in NOT_COMMANDS:
                    continue
                if any(m in line.lower() for m in FORWARD_MARKERS):
                    continue
                errors.append(
                    f"{rel(path)}:{lineno}: references command '/{cmd}', which does not exist"
                )

    # ---- 4. report-card coverage -----------------------------------------
    card_dir = os.path.join(ROOT, "reviews", "report-cards")
    if os.path.isdir(card_dir):
        cards = {f[:-3] for f in os.listdir(card_dir) if f.endswith(".md")} - {"INDEX"}
        for pack in sorted(packs - cards):
            warnings.append(f"{pack}: no report card in reviews/report-cards/")

    # ---- 5. discovery orphans --------------------------------------------
    for pack in sorted(packs):
        if referenced[pack] == 0:
            warnings.append(
                f"{pack}: referenced by no other pack -- reachable only if the user "
                f"already knows it exists"
            )

    # ---- report -----------------------------------------------------------
    print(f"packs: {len(dirs)}   marketplace entries: {len(market_names)}   commands: {len(commands)}")
    print(f"errors: {len(errors)}   warnings: {len(warnings)}\n")
    if errors:
        print("ERRORS")
        for e in errors:
            print(f"  {e}")
        print()
    if warnings:
        print("WARNINGS")
        for w in warnings:
            print(f"  {w}")
        print()
    if not errors and not warnings:
        print("clean")
    return 1 if errors and not args.warn else 0


if __name__ == "__main__":
    sys.exit(main())
