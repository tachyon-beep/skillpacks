---
name: medium-digital-and-asynchronous
description: "Use for screen-based, networked, software-arbitrated, automated, persistent, asynchronous, or delayed play - the digital/asynchronous medium adapter; load only when the medium materially changes the answer."
---

# Digital and Asynchronous Media

Use this reference for screen-based, networked, software-arbitrated,
asynchronous, or persistent games.

## Contents

- Screen-based affordances and costs
- Controls, feedback, and automation
- Networked and AI-mediated play
- Asynchronous and persistent play

## Use screen-based affordances

Exploit:

- fast and exact resolution;
- private and personalized information;
- large or changing state spaces;
- real-time input, physics, audiovisual feedback, and embodied controls;
- AI-controlled actors and simulation;
- save, undo, replay, procedural generation, networking, and telemetry;
- adaptive presentation and accessibility settings.

Pay for:

- invisible state and causality hidden by automation;
- interface learning, device constraints, latency, failures, and platform rules;
- content production and maintenance;
- remote social cues and moderation;
- patches that can alter the learned game;
- privacy, security, accessibility, and service continuity.

Check:

- Can players predict which state the software will consume?
- Does automation preserve feedback needed to learn?
- Are controls expressive, or merely obstacles between decision and outcome?
- Does real-time pressure serve the experience and supported access needs?
- Can players pause, remap, scale, repeat, or review where appropriate?
- Do network authority and lag create unfair or illegible outcomes?
- Does AI create interaction, or only content and spectacle?
- Does telemetry measure the claim, or only what is easy to log?

Separate game state from presentation, but design their relationship. A hidden
calculation may be correct and still produce an unusable game.

## Make controls and feedback part of the mechanic

Specify the mapping from intention through input to response:

```text
perceived state -> intended action -> physical/input operation
                -> system interpretation -> immediate feedback
                -> game consequence -> later learning
```

Check buffering, cancellation, remapping, focus, error recovery, input latency,
camera behavior, target selection, and simultaneous inputs where material.
Decide which execution difficulty constitutes play and which merely blocks the
strategic, expressive, or narrative layer.

Use animation, sound, haptics, layout, and pacing to reveal ownership, timing,
state change, and consequence. Do not add audiovisual “juice” that obscures
critical information, triggers sensory harm, or makes weak decisions feel
temporarily exciting without repairing them.

## Design networked and AI-mediated play

For networked play, define:

- authoritative state, prediction, rollback, reconnect, timeout, and dispute;
- what latency changes mechanically;
- what remains possible when voice, text, matchmaking, or a service fails;
- moderation, reporting, blocking, identity, privacy, and community boundaries;
- how parties, teams, ranks, and drop-in/drop-out change social power.

For AI-controlled actors or generators, define:

- the player's model of the actor's capabilities and limits;
- which state and history the system may access;
- whether behavior must be deterministic, replayable, explainable, or merely
  plausible;
- how failure, repetition, bias, unsafe content, and service loss resolve;
- whether the AI changes player decisions or only produces spectacle/content.

Do not infer human-facing value from an AI's fluent performance or reported
enjoyment.

## Design asynchronous and persistent play

Exploit:

- reflection, scheduling flexibility, large geography, long arcs, and persistent
  artifacts or worlds;
- communication and decisions that benefit from time;
- participation at different rhythms.

Pay for:

- waiting, stalling, forgotten context, notification pressure, unequal
  availability, abandonment, and re-entry cost;
- time-zone asymmetry and deadlines that favor particular lives;
- persistent social conflict and records;
- service operation and state recovery.

Check:

- What can a player do while awaiting someone else?
- Which windows are hard, soft, negotiated, or automatically resolved?
- How is context restored after absence?
- Can one player hold everyone hostage by not responding?
- Do notifications inform or coerce?
- How do new, returning, or departing players enter and leave?
- Does persistence transform meaning or merely stretch a short loop?
- What data and relationships remain after the game ends?

Design graceful defaults, delegation, bounded windows, summaries, and exit.
Never make constant availability an unstated competitive resource.

