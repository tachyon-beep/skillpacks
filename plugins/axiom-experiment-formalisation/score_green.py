#!/usr/bin/env python3
"""Score a GREEN-pass deliverable against the pack's own verified inventory.

Written BEFORE reading any GREEN output, so the criteria cannot drift to fit results.
Automated flags are a triage aid ONLY — every hit must be read manually in context,
because a term appearing inside a "this is refuted" sentence is a PASS, not a failure.

Usage: python3 score_green.py <file>
"""
import json, re, sys

D = '/home/john/skillpacks/plugins/axiom-experiment-formalisation/skills/using-experiment-formalisation/'
inv = json.load(open(D + 'expo-owl-inventory.json'))
vt = json.load(open(D + 'verified-terms.json'))

REAL = set(inv['classes']) | set(inv['object_properties'])
CORRECTIONS = {k: v for k, v in vt['expo_corrections'].items() if not k.startswith('$')}
DO_NOT_CITE = {k: v for k, v in vt['do_not_cite'].items() if not k.startswith('$')}
SHIPS_BUT_DENIED = {k for k in vt['expo_present_but_widely_denied'] if not k.startswith('$')}

text = open(sys.argv[1]).read()
# candidate ontology-ish terms: backticked or bare CamelCase / underscore fragments
cands = set(re.findall(r'`([A-Za-z][A-Za-z_.\-/]*)`', text))
cands |= set(re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]*){1,})\b', text))

fabricated  = sorted(t for t in cands if t in DO_NOT_CITE)
paper_only  = sorted(t for t in cands if t in CORRECTIONS)
verified    = sorted(t for t in cands if t in REAL)
denial_risk = sorted(t for t in SHIPS_BUT_DENIED if t in text)

print(f"=== {sys.argv[1]} ===\n")
print(f"VERIFIED terms used ({len(verified)}): {', '.join(verified[:30])}\n")

print(f"[FLAG] fabricated / do-not-cite terms present ({len(fabricated)}):")
for t in fabricated:
    print(f"    {t}  -> {DO_NOT_CITE[t][:100]}")
print(f"\n[FLAG] paper-only names present ({len(paper_only)}):")
for t in paper_only:
    print(f"    {t}  -> should be {CORRECTIONS[t][:70]}")
print(f"\n[CHECK MANUALLY] ships-but-often-denied terms mentioned ({len(denial_risk)}):")
for t in denial_risk:
    for m in re.finditer(re.escape(t), text):
        seg = text[max(0, m.start()-110):m.end()+110].replace('\n', ' ')
        print(f"    ...{seg}...")
        break

# discipline signals
sig = {
    'declines inversion / projection law': r'projection law|never becomes|contract wins|source of truth.*contract|not the runtime',
    'triage or tier decision':             r'Tier 1|Tier 2|triage|do not formalise|don\'t formalise',
    'competency questions':                r'competency question',
    'no-intervention arm':                 r'no-intervention|Treated_Untreated|null .*level',
    'unit of analysis / pairing':          r'unit of analysis|paired|ancestor',
    'absence != zero':                     r'unmeasured|absence|validity (mask|status)',
    'gap register':                        r'gap register',
    'sync check':                          r'sync check',
    'cites the OWL not the paper':         r'expo\.owl|shipped OWL|owl-inventory|URI fragment',
    'disjointness caution':                r'disjoint',
}
print("\n=== discipline signals ===")
for k, pat in sig.items():
    print(f"  {'YES' if re.search(pat, text, re.I) else ' no'}  {k}")

print(f"\nVERDICT INPUT: {len(fabricated)} fabricated, {len(paper_only)} paper-only. "
      f"Any non-zero count requires manual reading before scoring FAIL.")
