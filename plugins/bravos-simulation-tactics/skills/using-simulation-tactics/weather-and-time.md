# Game time and atmosphere state

Use when time/weather progression changes gameplay, simulation events or persistent state.

## Checks

- Define simulation clock, rendering clock, calendar/time-zone needs and pause/fast-forward/resume semantics. Do not derive persistent gameplay solely from wall-clock frame deltas.
- Specify scheduled-event ordering, catch-up policy and missed/duplicate-event handling across large time jumps and save/load.
- Separate visual interpolation from authoritative weather/gameplay state. Record how wind, precipitation, temperature or season affects dependent systems.
- Use astronomical models only to the required fidelity; latitude/hemisphere/calendar assumptions matter. A fixed sun direction formula is not universal.
- Define transition/stochastic process, bounds and RNG ownership; validate replay and long-run distribution where required.
- Preserve active effects and accrued consequences when changing LOD, skipping time or moving between regions.
- Test zero/large timestep, pause, boundary dates, rapid transitions and dependent subsystem ordering; measure relevant update spikes.

## Deliverable

Clock/weather state and transition contract with visual/gameplay checks and replay/save evidence. See [fidelity](simulation-vs-faking.md), [debugging](debugging-simulation-chaos.md) and simulation foundations for numerical/stochastic requirements.
