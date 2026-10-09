#!/usr/bin/env python3
"""Check published entrypoint metadata, source pointers and non-example local links.

Reference sheets can contain illustrative project paths, so local link validation
is deliberately limited to runtime entrypoints and the current catalog documents.
Intentional maintenance fixtures and operational skills are outside publication.
Requires PyYAML for frontmatter parsing.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'(?<!!)\[[^\]]*\]\(([^)]+)\)')
FENCE = re.compile(r'^\s*(`{3,}|~{3,})')
RETIRED = {'axiom-engineering-foundations', 'axiom-system-architect',
           'yzmir-ai-engineering-expert', 'meta-sme-protocol'}


def prose(text: str) -> str:
    result, fence = [], None
    for line in text.splitlines():
        match = FENCE.match(line)
        if match:
            mark = match.group(1)[0]
            if fence is None:
                fence = mark
            elif fence == mark:
                fence = None
            continue
        if fence is None:
            result.append(line)
    return '\n'.join(result)


def main() -> int:
    errors: list[str] = []
    catalog = json.loads((ROOT / '.claude-plugin/marketplace.json').read_text())
    packs = sorted((ROOT / 'plugins').iterdir())
    packs = [p for p in packs if (p / '.claude-plugin/plugin.json').exists()]
    # A checkout may contain ignored local development files. They are not
    # published; preserve them while validating the actual tracked/new sources.
    ignored = set()
    try:
        result = subprocess.check_output(
            ['git', 'ls-files', '--others', '--ignored', '--exclude-standard', '-z', 'plugins'],
            cwd=ROOT, stderr=subprocess.DEVNULL,
        ).decode().split('\0')
        ignored = {ROOT / p for p in result if p}
    except (OSError, subprocess.CalledProcessError):
        pass  # In a source archive there are no checkout-local ignored files.
    entries = []
    counts = dict(packs=len(packs), skills=0, references=0, commands=0, agents=0)
    for pack in packs:
        for kind, pattern in [('skills', 'skills/**/SKILL.md'),
                              ('commands', 'commands/**/*.md'),
                              ('agents', 'agents/**/*.md')]:
            found = sorted(p for p in pack.glob(pattern) if p not in ignored)
            entries.extend(found)
            counts[kind] += len(found)
        counts['references'] += sum(p.name != 'SKILL.md' and p not in ignored for p in pack.glob('skills/**/*.md'))
    for entry in catalog['plugins']:
        source = ROOT / entry['source']
        if not source.is_dir() or source.name != entry['name']:
            errors.append(f"catalog source mismatch: {entry['name']} -> {entry['source']}")
    docs = [ROOT / p for p in ['README.md', 'CLAUDE.md', 'FACTIONS.md', 'CONTRIBUTING.md',
                               '.claude/SLASH_COMMANDS.md', 'docs/relevance-refresh.md',
                               'docs/sme-agent-protocol.md', 'docs/ai-specialist-catalog.md',
                               'docs/sdlc-prescription-cmmi-levels-2-4.md']]
    wrappers = sorted((ROOT / '.claude/commands').glob('*.md'))
    for path in entries + wrappers:
        text = path.read_text()
        if not text.startswith('---\n') or len(text.split('---', 2)) < 3:
            errors.append(f'{path.relative_to(ROOT)}: missing frontmatter')
            continue
        try:
            front = yaml.safe_load(text.split('---', 2)[1])
        except yaml.YAMLError as exc:
            errors.append(f'{path.relative_to(ROOT)}: invalid YAML: {exc}')
            continue
        required = ['name', 'description'] if path.name == 'SKILL.md' else ['description']
        if not isinstance(front, dict) or any(not isinstance(front.get(k), str) or not front[k].strip() for k in required):
            errors.append(f'{path.relative_to(ROOT)}: required nonempty string fields {required}')
        if path.name == 'SKILL.md' and isinstance(front, dict) and front.get('name') != path.parent.name:
            errors.append(f'{path.relative_to(ROOT)}: name differs from skill directory')
        for name in RETIRED:
            if name in text:
                errors.append(f'{path.relative_to(ROOT)}: runtime dependency on retired name {name}')
    for path in entries + wrappers + docs:
        if not path.exists():
            errors.append(f'missing current document {path.relative_to(ROOT)}')
            continue
        for target in LINK.findall(prose(path.read_text())):
            target = target.strip('<>').split('#', 1)[0]
            if not target or '://' in target or target.startswith(('mailto:', 'data:')):
                continue
            # Actual source links only; HTML title suffixes are not used here.
            if not (path.parent / target).exists():
                errors.append(f'{path.relative_to(ROOT)}: broken local link {target}')
    print('published:', ', '.join(f'{k}={v}' for k, v in counts.items()))
    print(f'entrypoint/catalog errors: {len(errors)}')
    for error in errors:
        print(error)
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
