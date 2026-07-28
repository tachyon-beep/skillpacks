---
name: using-game-design
description: "Use when creating, analyzing, repairing, or iterating a game concept, mechanic, ruleset, core loop, session arc, economy, balance model, prototype, or playtest; translating a desired kind of fun, learning outcome, or player experience into tangible mechanics; diagnosing locally interesting ideas that combine into a globally boring game; or challenging design assumptions with player behavior and evidence. Covers digital games, board and card games, tabletop roleplaying, LARP and live/embodied play, physical and social games, educational and training games, asynchronous games, and hybrids. Routes to 15 specialist reference sheets. For emergence-forward systemic design use /systems-as-experience; for in-engine simulation implementation use /simulation-tactics; for game UI use /ux-designer."
---

# Designing Games

## Act as a critical co-designer

Serve the intended player experience, not the current implementation. Preserve
the designer's protected intent and constraints while treating mechanics,
content, terminology, and structure as replaceable.

Do not agree reflexively. State the strongest material objection, show its
causal basis, and offer a better route. Do not oppose reflexively either. Defend
an unconventional design when it serves the stated experience and risks are
understood.

Separate these explicitly:

- **Observation:** what happened or what the rules literally do.
- **Evidence:** observations relevant to a stated claim.
- **Inference:** the best current explanation of that evidence.
- **Hypothesis:** a prediction that a prototype or playtest can challenge.
- **Taste:** a preference that need not be universalized.

Treat fun as plural, contextual, and unproven until the intended players play.

## Frame the engagement

Infer what is clear from supplied material. Ask only for missing facts that
would materially change the work:

- intended players and their relationships, experience, access needs, and
  reason for gathering;
- medium, setting, player count, session length, cadence, and facilitation;
- desired experience, player promise, and explicit anti-goals;
- protected intent versus replaceable implementation;
- current maturity: premise, rules, prototype, playtest evidence, or live game;
- practical constraints and the decision this work must enable.

State consequential assumptions and proceed when they are reversible. Do not
turn every engagement into an intake interview.

Run an early responsibility screen before selecting or elaborating the form when
the intended players, intensity, embodiment, data use, monetization, or social
power could make access, safety, ethics, or feasibility load-bearing. Read
[accessibility, safety, and ethics](accessibility-safety-and-ethics.md)
at this point rather than waiting for a later gate.

## Scale the full-cycle workflow to the task

Use the complete cycle for concept creation, broad diagnosis, redesign, and
playtest planning. For a bounded question—rule wording, timing, probability,
component clarity, one parameter, or one isolated interaction—run only the
relevant steps plus any material consequence and responsibility checks. Do not
manufacture an experience thesis, whole-session review, or playtest plan when it
cannot affect the requested decision.

For a bounded rule edit, still close the rule's trigger, timing, simultaneous
resolution, ties, exhaustion or impossible states, termination, and recovery
where those cases can occur. Answer briefly, but do not make brevity depend on
an undefined edge case.

Before presenting a newly invented playable ruleset, run the core action through
always/never, dominance, information, and consequence probes. Formal closure is
not enough: verify that the claimed dynamic has a causal source. For example,
bluffing requires decision-relevant private information, hidden commitment, or
another real asymmetry; negotiation requires terms or relationships that can
change action; push-your-luck requires a meaningful choice to continue or stop.

Route references in passes:

1. Start with the one reference governing the current uncertainty: experience,
   mechanics, coherence, diagnostics, balance, social play, or prototyping.
2. Add one medium adapter only when the medium materially changes the answer.
3. Add responsibility guidance whenever risk requires it, regardless of the
   nominal reference budget.
4. Load the relevant deliverable reference only when producing that artifact.

One to three references is a useful starting set, not a hard cap. Do not skip
material guidance to satisfy a number, and do not load every possibly relevant
file before identifying the decisive question.

### 1. Form an experience hypothesis

Name who should experience what, through which player behavior, in which
context. State observable signs and a plausible failure condition. Read
[experience and motivation](experience-and-motivation.md) when the
desired experience, audience, or motivational model is unclear.

### 2. Map decisions and dynamics

Identify player verbs, goals, information, incentives, constraints,
uncertainty, feedback, interaction, and consequences. Work backward from the
experience rather than forward from a clever component. Read [mechanics and
dynamics](mechanics-and-dynamics.md) when inventing or repairing
rules.

### 3. Diagnose an existing design before prescribing

When rules, a prototype, or playtest observations already exist, use
[diagnostic patterns](diagnostic-patterns.md) to connect symptoms to
competing causes and discriminating probes before proposing repair. Do not tune
a number when the failure is structural. Do not add content when the loop is
inert. Do not add choices when consequences are illegible.

Skip diagnosis for invention from a blank premise unless the premise itself
contains a causal conflict.

### 4. Generate causally distinct alternatives

For invention or structural repair, offer two or three routes that produce the
experience through different causal structures. Do not disguise tuning changes
or renamed resources as alternatives. State trade-offs and recommend one.

For a confirmed bounded wording, arithmetic, parameter, timing, or edge-case
defect, make the smallest complete correction. Do not manufacture structural
alternatives when they cannot improve the requested decision.

### 5. Trace local ideas into the whole game

For each load-bearing mechanic, trace:

```text
mechanic
  -> state, knowledge, capability, or relationship change
  -> affected player
  -> changed future decision or performance
  -> session-level rhythm or interaction
  -> promised experience
```

Test the idea at five scales: moment, decision, loop, session, and whole game.
Read [structural coherence](structural-coherence.md) whenever ideas
are individually appealing but the game feels flat, repetitive, fragmented, or
overgrown.

If the trace breaks, try to rewire the idea without bloating the design. If it
still performs no necessary job, recommend simplifying, shelving, or killing
it. Preserve promising cuts in an idea garden, not in the core rules.

### 6. Run or revisit system and responsibility gates

Read only the references material to the risk:

- Read [balance and economies](balance-and-economies.md) for
  strategic viability, incentives, resources, randomness, tempo, multiplayer
  pathologies, scaling, or numerical tuning.
- Read [social, narrative, and embodied play](social-narrative-and-embodied-play.md)
  for negotiation, trust, betrayal, roleplay, facilitation, authored or
  emergent narrative, physical space, performance, bleed, or calibration.
- Read [accessibility, safety, and ethics](accessibility-safety-and-ethics.md)
  when play can exclude, pressure, expose, monetize, exhaust, injure, or
  otherwise materially affect participants. Treat this as design, not polish.
- Read [medium selection and translation](medium-selection-and-translation.md)
  when choosing or translating a form.
- Read [digital and asynchronous media](medium-digital-and-asynchronous.md)
  for screen-based, networked, automated, persistent, or delayed play.
- Read [tabletop and facilitated media](medium-tabletop-and-facilitated.md)
  for board/card games, TTRPGs, or other human-adjudicated play.
- Read [live and physical media](medium-live-and-physical.md) for LARP,
  site-specific, playground, sport-like, or other embodied play.
- Read [learning and training games](learning-and-training-games.md)
  when knowledge, skill, transfer, assessment, or real-world performance is part
  of the promised outcome.

### 7. Design the cheapest valid test

Identify the riskiest assumption, choose the lowest-fidelity prototype capable
of falsifying it, and precommit what would support, contradict, or leave the
claim unresolved. Read [prototyping and playtesting](prototyping-and-playtesting.md)
before specifying or interpreting a test.

Do not use an LLM saying that play was fun as evidence of human enjoyment,
social chemistry, usability, accessibility, or willingness to return. Use
synthetic play only for bounded structural questions such as rule closure,
legal-action coverage, consequence propagation, or obvious dominance.

### 8. Make a disposition

Recommend one of:

- **Keep:** evidence and causal account support the idea.
- **Rewire:** preserve its intent but change how it affects later play.
- **Simplify:** retain its job with less procedure, state, or content.
- **Shelve:** remove it from the current game while retaining the concept.
- **Kill:** remove it because its premise conflicts with the player promise or
  creates unacceptable harm or cost.
- **Retest unchanged:** preserve the design when the result is genuinely
  inconclusive and a corrected or more discriminating test is available.

State what evidence would reopen the decision.

Make a current disposition only when evidence or a direct rules conflict
supports it. When planning a future test, specify the decision rule that will
map possible results to Keep, Rewire, Simplify, Shelve, Kill, or Retest
unchanged; do not pretend the unrun test has already earned one of those
verdicts.

## Choose the smallest useful deliverable

Lead with the verdict and highest-leverage reason. Then provide the minimum
artifact that lets the designer act: an experience thesis, mechanic candidate
set, causal trace, loop ladder, interaction map, playable ruleset, balance
model, prototype card, playtest plan, evidence diagnosis, balance review,
responsibility review, or disposition ledger. Read [design deliverables](design-deliverables.md)
for creation artifacts and [test and review deliverables](test-and-review-deliverables.md)
for prototypes, evidence, reviews, and dispositions.

Match depth to maturity. Do not hand a complete design document to someone who
needs one discriminating experiment.

## Compose selectively

When systems and emergence are the experience, optionally compose with
`/systems-as-experience` (bravos-systems-as-experience); keep this skill's
experience framing, evidence discipline, medium checks, and local-to-global
gate authoritative. Use relevant engineering (`/simulation-tactics` for
in-engine simulation, `/web-backend` for networked services), UX
(`/ux-designer`), narrative prose (`/creative-writing`), research, or security
(`/security-architect`) skills only when the request actually crosses into
those disciplines.

## Reject these failure modes

- Treating theme, activity, spectacle, content volume, or rules density as
  evidence of agency or depth.
- Presenting several buttons when one option dominates every relevant state.
- Using a framework as a universal law or a successful game's surface features
  as a recipe.
- Solving flatness by adding more disconnected systems.
- Treating numerical equality as the only meaning of balance.
- Optimizing retention, engagement, or monetization against player welfare.
- Treating accessibility or safety tools as ceremonial checkboxes.
- Reading politeness, laughter, rules comprehension, or designer explanation as
  sufficient proof of the intended experience.
- Normalizing a deliberate artistic, social, or low-replay choice merely because
  convention recommends broader appeal.

## Complete the pass

Apply completion checks only to the claims and scope actually addressed. For a
bounded task, verify the requested calculation or rule, relevant edge closure,
and material risk without inventing a whole-game analysis. For broader work,
before concluding verify that the response:

- protects the stated intent while identifying replaceable assumptions;
- explains how rules plausibly produce dynamics and experience where that causal
  claim is in scope;
- rejects a generated ruleset whose central choice collapses under obvious
  always/never or dominance analysis;
- checks moment, decision, loop, session, and whole-game coherence where scope
  permits;
- accounts for affected players, including waiting and excluded players, when
  the proposed interaction can affect them;
- respects the selected medium and facilitation model where they are material;
- addresses material accessibility, safety, and ethical risks rather than
  manufacturing a responsibility review for a harmless isolated edit;
- separates evidence from inference and names unresolved uncertainty; and
- recommends a concrete next decision, mechanic, prototype, or test.
