---
name: feedback-control-theory
description: PID controllers, tuning methods, anti-windup, transfer functions, and stability margins applied to camera systems, AI pursuit, dynamic difficulty, audio crossfading, and physics damping
---

# Feedback Control Theory for Games

## Overview

Every game system that "tracks" something — a camera following the player, an AI pursuing a target, an audio bus chasing a target volume, an economy chasing a target inflation rate — is a control problem. Most teams reach for `Lerp(current, target, 0.15f)` or a hand-tuned magic number, then bolt on hacks when it overshoots, lags, or oscillates.

Feedback control theory replaces magic numbers with a **principled, tunable, predictable** structure: the **PID controller**. Three terms — Proportional, Integral, Derivative — combined linearly, give you a knob for steady-state error, response speed, and damping that you can tune *systematically* rather than by trial and error.

**Key insight**: Lerp is a degenerate P-only controller with a hidden, frame-rate-dependent gain. Once you see this, you understand why your camera "feels different" at 30 vs 144 Hz, why your AI "stutters" near its target, and why your difficulty system "thrashes." PID makes those failures explicit and fixable.

## When to Use

Load this skill when:

- A value in your game needs to **track a target** smoothly (camera, AI, audio, UI animation, economy meter, weapon recoil recovery)
- You have a `Lerp(current, target, k)` or `current += (target - current) * k` and it doesn't feel right
- A simulated quantity should stay near a setpoint despite **disturbances** (player input, network jitter, varying load)
- Your tuning loop is "tweak the magic number, run the game, watch, repeat"
- You see overshoot, oscillation, sluggish convergence, or steady-state error that won't go away
- You need to integrate a control loop with **stability analysis** (`stability-analysis.md`) before shipping

**Symptoms you need this**:

- Camera judders at high speeds or "swings past" the player on direction changes
- AI accelerates to a target then orbits or oscillates around it
- Dynamic difficulty thrashes between too-easy and too-hard
- Audio levels pump or click on transitions
- Physics damping kills motion at one framerate and explodes at another
- "It works in editor, breaks at 30 FPS"

**Don't use for**:

- Pure event-driven systems with no continuous error signal (use `state-space-modeling.md`)
- Systems where you have a closed-form analytical answer (just compute it)
- Discontinuous response — switching, hysteresis, finite-state behaviour (use a state machine)
- Stochastic targets where the "right" answer is a distribution (use `stochastic-simulation.md`)
- Hard real-time constraints where PID's transient response is unacceptable (use feed-forward + LQR/MPC; see *When to escalate* below)


## RED: Why Magic-Number Control Fails

### Failure 1: The Lerp Camera (Frame-Rate-Dependent Disaster)

**Scenario**: Third-person action game, camera follows the player.

**What they did**:

```csharp
void LateUpdate() {
    Vector3 target = player.position + offset;
    transform.position = Vector3.Lerp(transform.position, target, 0.15f);
}
```

**What went wrong**:

- At 60 FPS the camera felt fine.
- At 30 FPS it felt sluggish; at 144 FPS it felt jittery and snappy.
- QA reported "camera physics changes between machines."
- Adding `* Time.deltaTime` made it framerate-independent but slower; multiplying by 60 made it work at 60 FPS but explode at 144 FPS.

**Why**: `Lerp(a, b, t)` with constant `t` per frame is an exponential decay where the **time constant depends on the frame rate**. Applied each frame:

```
e(n+1) = (1 - t) * e(n)         where e = (current - target)
```

The effective continuous time constant is `τ = -dt / ln(1 - t)`. Halve the frame rate, double `dt`, double `τ`. The camera's *physical behaviour* changes with monitor refresh rate.

**What the team tried**:

- `Lerp(a, b, t * Time.deltaTime)` — closer, but still wrong because `Lerp(a, b, k)` is not exponential decay; the correct framerate-independent form is `a + (b - a) * (1 - exp(-k * dt))`.
- Hard-clamp the maximum offset — masks the symptom, doesn't fix overshoot on direction changes.
- Two layers of lerp (position + look-at) with separate magic numbers — now there are two coupled magic numbers and changing one breaks the other.

**Root cause**: There is no proportional gain, no derivative, no integral. The system has one free parameter that conflates *how aggressive* and *how stable*. PID separates them.


### Failure 2: AI Pursuit That Orbits the Target

**Scenario**: Top-down twin-stick shooter, melee enemies chase the player.

**What they did**:

```python
def update_enemy(enemy, player, dt):
    direction = normalize(player.pos - enemy.pos)
    enemy.vel = direction * enemy.max_speed
    enemy.pos += enemy.vel * dt
```

**What went wrong**:

- Enemies caught the player but **orbited** when in attack range — they couldn't slow down.
- Adding `if distance < attack_range: enemy.vel *= 0` produced visible juddering as the boolean flipped each frame.
- Adding hysteresis (`stop` < `attack_range` < `resume`) helped but made positioning feel mechanical and was visibly tuned per enemy type.
- High-difficulty modes that scaled `max_speed` made the orbit radius larger.

**Why**: This is **bang-bang control** (fixed velocity, on/off). It has no notion of "approaching the target" — it tries to be *exactly* at full speed until *exactly* at the target. PID's derivative term provides the natural braking. Its proportional term provides the natural slow-down on approach.

### Failure 3: Dynamic Difficulty That Thrashes

**Scenario**: Roguelike that scales enemy stats to hold player win rate near 50%.

**What they did**:

```python
if recent_win_rate > 0.55:
    difficulty += 0.1
elif recent_win_rate < 0.45:
    difficulty -= 0.1
```

**What went wrong**:

- After a win streak, difficulty jumped up; the next two runs were brutal; difficulty crashed back; then runs were trivial. The player experienced **whiplash**.
- Smaller step (0.01) made it sluggish — players noticed it took 30 runs to adapt.
- Larger window (last 50 runs) made it stable but also sluggish.
- Players gamed it by intentionally losing two runs to drop difficulty before a real attempt.

**Why**: This is a P-only controller with a discretized output and a long measurement delay. The delay introduces phase lag; the discretized step introduces a limit cycle. Without integral action, steady-state error never closes. Without derivative action, the controller cannot anticipate that a streak is forming.


### The General Pattern

| Symptom | Bad fix | Real cause |
|---|---|---|
| "Feels different at different frame rates" | Multiply by `dt` somewhere | No principled time constant |
| "Overshoots and orbits" | Hysteresis / hard clamp | No derivative term |
| "Steady-state error never closes" | Bigger gain | No integral term |
| "Thrashes" | Bigger window / smaller step | No `Kp/Ki/Kd` decomposition; can't tune separately |
| "Pumps / clicks on transitions" | Smooth the input | Setpoint discontinuity + no rate limit |
| "Integral grows forever in saturation" | Reset on each frame | No anti-windup |

PID gives you a vocabulary and three knobs that map to those three symptoms.


## GREEN: PID Theory

### The Three Terms

A PID controller computes a control signal `u(t)` from an error `e(t) = setpoint - measurement`:

```
u(t) = Kp * e(t)                                # Proportional
     + Ki * ∫₀ᵗ e(τ) dτ                         # Integral
     + Kd * de(t)/dt                            # Derivative
```

Plain English:

- **Proportional**: "How far off am I right now?" Larger `Kp` reacts harder. Too large → oscillation. Too small → sluggish.
- **Integral**: "How much error have I accumulated?" Removes steady-state error (a P-only controller against a constant disturbance has constant residual error). Too large → overshoot, windup.
- **Derivative**: "How fast is the error changing?" Predicts where the error is going and damps it. Too large → noise amplification, chatter.

Each term independently controls one failure mode. That separation is the entire reason PID exists.

### Discrete-Time Form

Games run in discrete steps. The textbook discretization (sometimes called *positional form*) uses the rectangle rule for the integral and a backward difference for the derivative:

```
e_n      = setpoint - measurement_n
I_n      = I_{n-1} + e_n * dt                    # Accumulated error
D_n      = (e_n - e_{n-1}) / dt                  # Rate of error change
u_n      = Kp * e_n + Ki * I_n + Kd * D_n
```

Two properties matter:

1. **Sampling rate sensitivity**: Halving `dt` doubles the integral term contribution per step but halves the derivative — gains stay valid only if `dt` is roughly constant. Variable `dt` requires care (fixed update tick or `dt`-aware tuning).
2. **Initial conditions**: `D_0` is undefined (no `e_{-1}`). Use `D_0 = 0` and let it warm up; never use a stored value from a previous run.

### Reference Implementation (Python)

This is the canonical `PIDController` referenced from the cascade and adaptive examples below and from the router. Translate it once for your engine; the structure is identical in every language.

```python
from dataclasses import dataclass, field

@dataclass
class PIDController:
    """Discrete-time PID controller with anti-windup, derivative-on-measurement,
    and output saturation. Suitable for game-loop use at fixed or near-fixed dt."""

    kp: float
    ki: float
    kd: float

    # Output limits. Always set these in real systems — actuators saturate.
    output_min: float = float("-inf")
    output_max: float = float("+inf")

    # Internal state
    integral: float = 0.0
    prev_measurement: float | None = field(default=None, init=False)
    prev_error: float = field(default=0.0, init=False)

    def reset(self) -> None:
        """Call when you teleport the controlled value (cutscene, respawn, level load)."""
        self.integral = 0.0
        self.prev_measurement = None
        self.prev_error = 0.0

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        if dt <= 0.0:
            return self._clamp(self.kp * (setpoint - measurement))

        error = setpoint - measurement

        # Proportional
        p = self.kp * error

        # Derivative (on measurement, see below). Computed once and reused for
        # both the anti-windup probe and the final output.
        d = self._derivative(measurement, error, dt)

        # Integral with conditional anti-windup (clamping integration when saturated)
        candidate_integral = self.integral + error * dt
        candidate_output = p + self.ki * candidate_integral + d

        # Only commit the integral update if the unclamped output is within limits,
        # OR if integrating would move the output back toward the linear region.
        if self.output_min < candidate_output < self.output_max or (
            (candidate_output >= self.output_max and error < 0) or
            (candidate_output <= self.output_min and error > 0)
        ):
            self.integral = candidate_integral

        u = p + self.ki * self.integral + d

        # Bookkeeping
        self.prev_measurement = measurement
        self.prev_error = error

        return self._clamp(u)

    # --- private helpers --------------------------------------------------

    def _derivative(self, measurement: float, error: float, dt: float) -> float:
        """Derivative on measurement avoids the 'derivative kick' on setpoint changes."""
        if self.prev_measurement is None:
            return 0.0
        d_meas = (measurement - self.prev_measurement) / dt
        # d(error)/dt = -d(measurement)/dt when setpoint is constant.
        return -self.kd * d_meas

    def _clamp(self, u: float) -> float:
        return max(self.output_min, min(self.output_max, u))
```

Three design decisions worth noting:

- **Derivative on measurement, not on error.** A step change in the setpoint produces an infinite `de/dt`, which the textbook form would multiply by `Kd` and emit as a huge spike (the *derivative kick*). Differentiating only the measurement avoids this — the controller still damps real motion, but setpoint changes are clean.
- **Conditional integration anti-windup.** When the actuator saturates, the integral should stop accumulating in the saturating direction but be free to unwind. This is the simplest correct anti-windup; back-calculation is more sophisticated and worth it only for slow plants (rare in games).
- **`reset()` exists and is named.** Every PID in a game eventually needs to be reset on level load, respawn, or cutscene cut. Build that in from day one.

### Reference Implementation (C#, Unity-friendly)

```csharp
using UnityEngine;

[System.Serializable]
public class PIDController {
    public float Kp = 1f;
    public float Ki = 0f;
    public float Kd = 0f;

    public float outputMin = float.NegativeInfinity;
    public float outputMax = float.PositiveInfinity;

    float integral;
    float prevMeasurement;
    bool hasPrev;

    public void Reset() {
        integral = 0f;
        hasPrev = false;
    }

    public float Update(float setpoint, float measurement, float dt) {
        if (dt <= 0f) return Mathf.Clamp(Kp * (setpoint - measurement), outputMin, outputMax);

        float error = setpoint - measurement;
        float p = Kp * error;

        // Derivative on measurement
        float d = 0f;
        if (hasPrev) {
            float dMeas = (measurement - prevMeasurement) / dt;
            d = -Kd * dMeas;
        }

        // Tentative integral + output for anti-windup test
        float candidateIntegral = integral + error * dt;
        float candidateOutput = p + Ki * candidateIntegral + d;
        bool withinLimits = candidateOutput > outputMin && candidateOutput < outputMax;
        bool unwinding =
            (candidateOutput >= outputMax && error < 0f) ||
            (candidateOutput <= outputMin && error > 0f);
        if (withinLimits || unwinding) integral = candidateIntegral;

        float u = p + Ki * integral + d;
        prevMeasurement = measurement;
        hasPrev = true;

        return Mathf.Clamp(u, outputMin, outputMax);
    }
}
```


### Verifying You Wrote It Correctly

Two minimal tests catch nearly every PID implementation bug. Run them before you tune anything.

```python
def test_p_only_holds_steady_state_offset():
    """A P-only controller against a constant disturbance leaves residual error."""
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
    measurement = 0.0
    disturbance = 1.0  # constant external force
    for _ in range(1000):
        u = pid.update(setpoint=10.0, measurement=measurement, dt=0.01)
        # Plant: dx/dt = u - disturbance
        measurement += (u - disturbance) * 0.01
    # P-only cannot remove the steady-state offset — that's the point.
    assert abs(measurement - 10.0) > 0.5

def test_pi_drives_steady_state_error_to_zero():
    """Adding integral term closes the steady-state offset."""
    pid = PIDController(kp=1.0, ki=2.0, kd=0.0)
    measurement = 0.0
    disturbance = 1.0
    for _ in range(5000):
        u = pid.update(setpoint=10.0, measurement=measurement, dt=0.01)
        measurement += (u - disturbance) * 0.01
    assert abs(measurement - 10.0) < 0.05
```

If the first test passes and the second fails, your integral term is wrong. If the first test *fails* (zero steady-state error with `Ki=0`), your "P-only" controller is secretly accumulating something — usually a hidden integral via a leaky low-pass filter or an off-by-one in `prev_error`.


## Tuning Methods

There is no universal "best" tuning. Pick a method that matches what you can measure and how forgiving the application is.

### Method 1: Manual Tuning (Recommended for Games)

For most game applications this is faster than any formal method. The procedure:

1. **Set `Ki = 0`, `Kd = 0`.**
2. **Increase `Kp`** until the system responds quickly to a step input but is on the edge of oscillation. Back off ~20%.
3. **Add `Kd`** to damp out residual oscillation. Start at `Kd ≈ 0.1 * Kp` and increase until oscillations are gone but the system isn't sluggish.
4. **Add `Ki`** only if there's visible steady-state error. Start small (`Ki ≈ 0.01 * Kp`) and increase until steady-state error is gone within an acceptable settling time.

What "edge of oscillation" looks like:

```
Output                                        Underdamped (good Kp + Kd)
  |                                       ___
  |                                  ____/   ‾‾\___
  |                          ___---‾‾                  ← settles cleanly
  |                       __/
  |                    __/
  |____________________
                                    time →
```

Vs:

```
Output                                        Oscillating (Kp too large)
  |
  |        __    __    __    __
  |       /  \  /  \  /  \  /  \             ← never settles
  |      /    \/    \/    \/    \
  |     /
  |____/
                                    time →
```

Vs:

```
Output                                        Overdamped (Kd too large or Kp too small)
  |
  |                                                   ← target
  | ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾ ‾
  |                       ___---‾‾
  |              ___---‾‾                              ← takes forever
  |     ___---‾‾
  |____/
                                    time →
```

### Method 2: Ziegler–Nichols (Closed-Loop, Ultimate-Gain)

Classic, controversial. Often produces aggressive tunings that overshoot ~25% — fine for industrial process control, often *too* aggressive for games. Use it as a starting point and back off.

Procedure:

1. Set `Ki = 0`, `Kd = 0`.
2. Increase `Kp` until the closed loop oscillates with constant amplitude. Call this `Ku` (ultimate gain).
3. Measure the period of oscillation `Pu` (in seconds).
4. Apply the table below.

| Controller | `Kp` | `Ki` | `Kd` |
|---|---|---|---|
| P | `0.50 Ku` | — | — |
| PI | `0.45 Ku` | `1.2 Kp / Pu` | — |
| PID (classic) | `0.60 Ku` | `2 Kp / Pu` | `Kp Pu / 8` |
| PID (less overshoot) | `0.33 Ku` | `2 Kp / Pu` | `Kp Pu / 3` |

**Caveat**: Ziegler-Nichols assumes a self-regulating, roughly first-order-plus-deadtime plant. Game systems rarely match that model. If the table gives you a tune that overshoots wildly, halve `Kp` and re-evaluate. Do not religiously trust the numbers.

### Method 3: Ziegler–Nichols (Open-Loop, Step-Response)

When you can't safely drive the system to oscillation (you'd crash the camera into the level geometry, etc.), measure the open-loop step response:

1. Disable the controller. Apply a step input `ΔU` to the plant.
2. Record the response. Identify:
   - `K` — process gain (final response / `ΔU`)
   - `L` — apparent dead time (delay before response begins)
   - `T` — apparent time constant (time from end of dead time to 63% of final value)
3. Apply the table below.

| Controller | `Kp` | `Ti = Kp / Ki` | `Td = Kd / Kp` |
|---|---|---|---|
| P | `T / (K · L)` | — | — |
| PI | `0.9 T / (K · L)` | `L / 0.3` | — |
| PID | `1.2 T / (K · L)` | `2 L` | `0.5 L` |

For games, `L` is often a single tick of input lag plus rendering latency. If `L → 0`, the table blows up — you do not need PID for that system; pure feed-forward will do.

### Method 4: Lambda Tuning / IMC (Internal Model Control)

When you need predictable settling time and minimal overshoot — for instance, an audio crossfade that must finish in exactly 1.5 seconds — use IMC. You pick the desired closed-loop time constant `λ` directly:

For a first-order-plus-deadtime plant with gain `K`, time constant `T`, dead time `L`:

```
Kp = T / (K · (λ + L))
Ti = T          → Ki = Kp / Ti
Td = 0          (PI controller; PID variant available but rarely needed in games)
```

`λ` is your single design knob. Smaller `λ` = faster response but more sensitive to plant uncertainty; larger `λ` = slower but more robust. A common starting point is `λ ≈ T`.

### Method 5: Tuning by Optimization (When You Have a Plant Model)

If you have a numerical model of the plant (for instance, you've built one with `state-space-modeling.md`), tune by minimizing an integral-of-error cost:

```python
from scipy.optimize import minimize

def cost(gains, plant, setpoint_trajectory, dt):
    kp, ki, kd = gains
    pid = PIDController(kp=kp, ki=ki, kd=kd)
    measurement = plant.initial_state()
    iae = 0.0  # Integral of Absolute Error
    for sp in setpoint_trajectory:
        u = pid.update(sp, measurement, dt)
        measurement = plant.step(measurement, u, dt)
        iae += abs(sp - measurement) * dt
    return iae

result = minimize(
    cost,
    x0=[1.0, 0.1, 0.01],
    args=(plant, trajectory, 1/60),
    method="Nelder-Mead",
)
kp_opt, ki_opt, kd_opt = result.x
```

Common cost functions:

- **IAE** (Integral of Absolute Error): balanced, no preference for sign.
- **ISE** (Integral of Squared Error): penalizes large errors more — produces aggressive tunings.
- **ITAE** (Integral of Time-weighted Absolute Error): penalizes late errors more — produces tunings that settle quickly.

Beware: optimizers will happily find a tune that minimizes the cost on *your specific* setpoint trajectory and falls apart on a different one. Test against multiple trajectories.


## Practical Pitfalls and Fixes

### Pitfall 1: Integral Windup

**Symptom**: The actuator saturates (camera hits a wall, AI hits max speed, audio bus hits 0 dB), but the integral keeps accumulating. When the setpoint finally relaxes, the controller takes seconds to "unwind" and severely overshoots in the opposite direction.

**Demonstration**:

```python
def windup_demo():
    pid = PIDController(kp=1.0, ki=2.0, kd=0.0,
                       output_min=-1.0, output_max=1.0)  # Tight saturation
    measurement = 0.0
    log = []
    for n in range(500):
        # Demand a setpoint we can never reach (output capped at 1.0,
        # plant integrates u → measurement, so steady state at u=1 is ramp)
        sp = 100.0 if n < 250 else 0.0
        u = pid.update(setpoint=sp, measurement=measurement, dt=0.01)
        measurement += u * 0.01
        log.append((sp, measurement, pid.integral))
    return log
```

Without anti-windup, `pid.integral` grows linearly across the first 250 steps, and the controller produces saturated output for hundreds of steps after the setpoint drops to zero — visible as a long, slow undershoot.

**Fixes**:

- **Conditional integration** (the `PIDController` reference above): freeze the integrator while the output is saturated *in the direction the integral would push*.
- **Back-calculation**: subtract the saturated portion from the integral with its own gain `Kt`. More aggressive, harder to tune.
- **Hard clamp on the integral**: `integral = clamp(integral, -I_max, I_max)`. Crude but works.

### Pitfall 2: Derivative Kick

**Symptom**: A small change in setpoint produces a huge spike in output. Camera jumps. Audio clicks. AI bursts forward.

**Cause**: A step change in setpoint produces a step change in error → an infinite `de/dt` → a huge `Kd * de/dt` term.

**Fix**: Differentiate the **measurement** instead of the error (the reference implementation does this). Mathematically equivalent when the setpoint is constant; well-behaved when it isn't.

If you also need smooth response to setpoint changes, low-pass filter the setpoint:

```python
# First-order low-pass on the setpoint
alpha = dt / (tau + dt)
filtered_setpoint += alpha * (raw_setpoint - filtered_setpoint)
```

with `tau` chosen so the filter time constant is ~3–5× the controller's natural settling time.

### Pitfall 3: Noise Amplification by the Derivative Term

**Symptom**: With even small measurement noise (sub-pixel jitter, quantized input), the controller chatters. Visible as twitchy motion or audible as clicking.

**Cause**: `Kd * de/dt` amplifies high-frequency content in the measurement. Noise has lots of high-frequency content.

**Fix**: Replace the raw derivative with a *filtered derivative*. The classic form:

```
D_filtered = (Kd · de/dt) / (1 + s·Kd/N)        # Continuous form, N is filter coefficient
```

Discretized:

```python
# alpha_d = filter time constant; smaller = more aggressive filter
alpha_d = dt / (kd / N + dt)
filtered_derivative += alpha_d * (raw_derivative - filtered_derivative)
```

`N` typically lives in 5–20. Larger `N` → less filtering → more noise. Smaller `N` → more lag in the derivative response. If your derivative term is doing nothing useful and your output is twitchy, the filter is too weak; if your response is sluggish, the filter is too strong.

### Pitfall 4: Variable `dt`

**Symptom**: Tuning that works in the editor falls apart on slower hardware. Or one frame's `dt` is 100× the previous (loading hitch), and the integral or derivative spikes.

**Causes**:

- The integral term scales linearly with `dt`. A 16 ms tick contributes 16× more to the integral than a 1 ms tick *for the same instantaneous error*. Tuning is implicitly a function of `dt`.
- A single huge `dt` (after a hitch) can make `error * dt` huge.

**Fixes**:

- **Run PID in `FixedUpdate`** (Unity) or your fixed simulation tick. PID belongs in the simulation, not the render loop.
- **Clamp `dt`**: `dt_safe = min(dt, 1/30)`. A 30 ms tick is fine; a 500 ms hitch ruins everything.
- **Skip the update on a huge `dt`**: if `dt > threshold`, hold the previous output, increment time, and resume cleanly next frame.

### Pitfall 5: Setpoint Discontinuities

**Symptom**: Audio "pumps" on a sudden volume target change. Camera "snaps" when the target switches.

**Fix**: Rate-limit or smooth the setpoint. Treat the *raw* user-facing target as a goal, and feed the controller a **filtered or rate-limited setpoint** that approaches the goal smoothly:

```python
max_rate = 5.0   # units per second
delta = goal - filtered_setpoint
filtered_setpoint += clamp(delta, -max_rate * dt, +max_rate * dt)
```

The controller then chases the smoothed setpoint and never sees a discontinuity.

### Pitfall 6: Bumpless Transfer

**Symptom**: When you switch from manual to automatic control (e.g., player releases the camera control stick and the auto-cam takes over), the controller emits a large transient.

**Cause**: The controller's internal state (`integral`, `prev_measurement`) doesn't match the current measurement.

**Fix**: When enabling the controller, initialize its integral so that the initial `u` equals the current actuator value:

```python
def enable_with_bumpless_transfer(pid, current_actuator_value, setpoint, measurement):
    pid.reset()
    error = setpoint - measurement
    p = pid.kp * error
    # Solve: current_actuator_value = p + ki * I + 0
    if pid.ki != 0:
        pid.integral = (current_actuator_value - p) / pid.ki
```

### Pitfall 7: Tuning Against the Wrong Plant

**Symptom**: Tunes well in a test scene; behaves poorly in the actual game.

**Cause**: The "plant" the controller is acting on includes everything between control output and measurement: the integration of velocity to position, the network latency, the renderer's interpolation, the input filter, the animation system, the camera collision logic. Tuning against a stripped-down test scene tunes against the wrong plant.

**Fix**: Tune in a representative scene with all the moving parts present. If the controller is *separable* from those moving parts (a pure math object), build an offline model that includes their dynamics and tune against it.


## Transfer Functions and Stability

You can ship games knowing only the discrete PID equations and tuning rules. But when a controller misbehaves and tuning isn't fixing it, the language of transfer functions tells you *why*.

### From Differential Equation to Transfer Function

A second-order plant — say, a mass-spring-damper for a camera arm — is governed by:

```
m·ẍ + c·ẋ + k·x = u(t)
```

In the Laplace domain (replace `d/dt` with `s`):

```
G(s) = X(s)/U(s) = 1 / (m·s² + c·s + k)
```

This is the plant's **transfer function**. PID's transfer function is:

```
C(s) = Kp + Ki/s + Kd·s
```

The **closed-loop transfer function** of `C(s)` controlling `G(s)` with unity feedback is:

```
T(s) = C(s)·G(s) / (1 + C(s)·G(s))
```

The **characteristic equation** `1 + C(s)G(s) = 0` determines stability: stable iff all roots of this equation have negative real parts.

### Reading Stability from Pole Locations

For the closed-loop system to be stable, every closed-loop pole (root of `1 + C(s)G(s) = 0`) must lie in the left-half complex plane. The geometry tells you the response shape:

| Pole location | Response |
|---|---|
| Far left, real | Fast, no oscillation |
| Slightly left, real | Slow, no oscillation |
| Slightly left, complex pair | Oscillates and decays |
| Near imaginary axis, complex pair | Marginally stable, prolonged ringing |
| Right-half plane | **Unstable** |

Classical control gives you the **root locus** — a plot of how closed-loop poles move as `Kp` varies — and **frequency response** plots (Bode plots) that show **gain margin** and **phase margin**. Their definitions for the record:

- **Gain margin (GM)**: how much the open-loop gain can increase before the closed loop becomes unstable. Rule of thumb: > 6 dB.
- **Phase margin (PM)**: how much extra phase lag the open loop can absorb before instability. Rule of thumb: > 30°, comfortable > 45°.

You don't need to compute these by hand. SciPy will do it:

```python
from scipy.signal import lti, bode, margin

# Plant: 1 / (s^2 + 2s + 5)
plant = lti([1.0], [1.0, 2.0, 5.0])
# PID: Kd*s^2 + Kp*s + Ki, all over s
kp, ki, kd = 2.0, 1.5, 0.5
controller = lti([kd, kp, ki], [1.0, 0.0])

# Open-loop transfer function = controller * plant
# (multiply numerators, multiply denominators)
import numpy as np
ol_num = np.polymul(controller.num, plant.num)
ol_den = np.polymul(controller.den, plant.den)
ol = lti(ol_num, ol_den)

gm, pm, wg, wp = margin(ol)
print(f"Gain margin: {gm:.2f} (linear), {20*np.log10(gm):.1f} dB at {wg:.2f} rad/s")
print(f"Phase margin: {pm:.1f}° at {wp:.2f} rad/s")
```

If gain margin is 1.0 (0 dB) or phase margin is 0°, the closed loop is on the edge of instability — back off `Kp` or add `Kd`.

### Why You Care for Games

Most game controllers don't need a Bode analysis. But two situations demand it:

1. **Cascade or coupled controllers** (one PID's output is another's setpoint). The combined system's stability is not the product of each loop's stability — analyse the coupled transfer function explicitly. Cross-reference `stability-analysis.md` for eigenvalue analysis of the coupled system.
2. **Networked / lag-prone systems**. Network round-trip time appears as dead time (a multiplicative `e^{-sL}` factor), which eats phase margin. A controller that's stable locally can be unstable when run over a 100 ms link. Phase margin tells you how much network lag you can tolerate.


## REFACTOR: 6 Game Scenarios

These are the original scenarios from the planning doc, each fully fleshed out with PID tuning, code, and the reasoning behind the gains.

### Scenario 1: Third-Person Camera Following

**Goal**: Camera maintains a target offset behind the player. Smooth on direction changes. No overshoot through walls. Frame-rate independent.

**RED**:

```csharp
void LateUpdate() {
    Vector3 target = player.position - player.forward * distance + Vector3.up * height;
    transform.position = Vector3.Lerp(transform.position, target, 0.15f);
}
```

Frame-rate dependent (see Failure 1 above), no anti-overshoot, no decoupling of "snappiness" from "smoothness."

**GREEN**: One PID per axis. Proportional dominates because the camera should move *toward* the target; derivative damps overshoot when the player suddenly reverses.

```csharp
public class CameraPID : MonoBehaviour {
    public Transform player;
    public Vector3 offset = new Vector3(0, 2, -5);
    public PIDController xPid = new PIDController { Kp = 8f, Ki = 0f, Kd = 4f };
    public PIDController yPid = new PIDController { Kp = 6f, Ki = 0f, Kd = 3f };
    public PIDController zPid = new PIDController { Kp = 8f, Ki = 0f, Kd = 4f };

    Vector3 velocity;

    void LateUpdate() {
        float dt = Time.deltaTime;
        Vector3 target = player.position + offset;
        Vector3 pos = transform.position;

        float ax = xPid.Update(target.x, pos.x, dt);
        float ay = yPid.Update(target.y, pos.y, dt);
        float az = zPid.Update(target.z, pos.z, dt);

        // Treat PID output as acceleration; integrate to velocity, then position.
        velocity += new Vector3(ax, ay, az) * dt;
        velocity *= Mathf.Exp(-3f * dt);  // light velocity damping for safety
        transform.position += velocity * dt;
    }
}
```

**Tuning notes**:

- `Ki = 0` because the camera target moves continuously; integrator would chase a moving goal and add lag.
- `Kd ≈ Kp / 2` is a starting point for second-order tracking. Reduce `Kd` if the camera feels sluggish; increase if it overshoots.
- The exponential velocity damping is a safety net for missing PID's natural damping under extreme inputs (teleport, level transition).

**Validation**: Verify with `transform.position` vs `player.position + offset` over a sweep where the player abruptly reverses direction. The error should decay smoothly to zero with at most a small overshoot (~5%).


### Scenario 2: AI Enemy Pursuit

**Goal**: Melee enemies chase the player. Approach is fast, arrival is gentle, no orbiting at attack range.

**RED**: Bang-bang or fixed-velocity chase (Failure 2 above).

**GREEN**: PID on the *distance error*. Output is the desired forward speed.

```python
class PursuitAI:
    def __init__(self):
        # Setpoint is the desired distance (in attack range, with a small standoff).
        self.attack_range = 1.5
        self.standoff = 1.2
        self.speed_pid = PIDController(
            kp=4.0, ki=0.0, kd=1.5,
            output_min=-2.0,   # may need to back away
            output_max=8.0,    # max forward speed
        )

    def update(self, enemy_pos, player_pos, dt):
        distance = (player_pos - enemy_pos).magnitude
        # Setpoint: the standoff distance. We want distance == standoff.
        # Note the sign: output should be positive (move forward) when distance > standoff.
        desired_speed = self.speed_pid.update(
            setpoint=self.standoff,
            measurement=distance,
            dt=dt,
        )
        # PID returns negative when distance < standoff, so negate for "approach."
        # Equivalently, swap setpoint and measurement when defining the error.
        direction = (player_pos - enemy_pos).normalized
        return direction * (-desired_speed)
```

Equivalent and clearer:

```python
desired_speed = self.speed_pid.update(
    setpoint=0.0,
    measurement=self.standoff - distance,  # positive when too close
    dt=dt,
)
```

**Tuning notes**:

- `Ki = 0` because steady-state error in a pursuit is fine — at steady state the enemy is *at* the standoff.
- `Kd` is critical. Without it, `Kp = 4` would cause orbiting around `standoff`. With `Kd = 1.5`, the enemy decelerates as it approaches.
- `output_min` slightly negative lets the enemy back away if the player charges into it — natural-looking.

**Difficulty scaling**: Scale `Kp` (more aggressive pursuit) and `output_max` (faster top speed) with difficulty. Keep `Kd / Kp` constant to preserve damping.


### Scenario 3: Dynamic Difficulty Scaling

**Goal**: Adjust difficulty over time to hold player win rate near a target (e.g., 50%). Avoid thrashing; avoid gaming.

**RED**: Threshold-based step adjustment (Failure 3 above).

**GREEN**: PI controller on win-rate error. The integral term is *required* here — it's how you eliminate steady-state bias from a player whose true skill differs from the difficulty model.

```python
class DifficultyController:
    def __init__(self, target_win_rate=0.5):
        self.target = target_win_rate
        self.difficulty = 1.0
        self.pid = PIDController(
            kp=2.0,
            ki=0.5,
            kd=0.0,           # No D — win-rate measurement is too noisy
            output_min=-0.05, # Cap per-tick adjustment to avoid whiplash
            output_max=0.05,
        )

    def update(self, recent_win_rate, dt):
        delta = self.pid.update(
            setpoint=self.target,
            measurement=recent_win_rate,
            dt=dt,
        )
        self.difficulty = max(0.1, self.difficulty - delta)
        return self.difficulty
```

**Why no D**: The derivative of a noisy stochastic signal (win rate over a small sample) is dominated by noise. `Kd > 0` would make the controller chatter. If derivative action is genuinely needed, smooth the win-rate measurement first with an exponential moving average over many runs.

**Tuning notes**:

- `dt` here is "per game" or "per session." Treat it consistently — if you call `update()` once per run, `dt = 1.0`; if once per minute of play, `dt = 1/60` of an hour. The Ki value depends on this convention.
- Output saturation (`output_min/max`) caps how fast difficulty can move per tick, which directly addresses the whiplash failure.
- Don't expose the integrator state to save files unless you also expose a way to reset it on dramatic skill change (new player, returning after a year).

**Anti-gaming**: A player who intentionally loses to drop difficulty will eventually push the integral down. Counter by computing win rate on a *long* window (last 50 runs) and using a slow Ki — losing 5 runs to game the system isn't worth it if the controller takes 30 runs to respond.


### Scenario 4: Audio Crossfading

**Goal**: Music bus volume tracks a target. No clicks. No pumping. Predictable settling time.

**RED**: Set the target volume directly. Visible discontinuity → audible click.

**GREEN**: Setpoint smoothing + PI controller. PID is optional here; rate-limited smoothing alone often suffices, but PI is robust against weird DSP-side state.

```python
class AudioBusController:
    def __init__(self, settling_time_seconds=1.5):
        # Lambda tuning: pick lambda = settling_time / 4 for ~4-tau settling
        lam = settling_time_seconds / 4.0
        # Plant is essentially a unity-gain integrator (volume changes are instant);
        # IMC for integrating plant gives Kp = 1/lambda, no Ki, no Kd.
        # We add a tiny Ki to compensate for any DSP-side state.
        self.pid = PIDController(
            kp=1.0 / lam,
            ki=0.05,
            kd=0.0,
            output_min=-1.0,  # dB per second, conservative
            output_max=+1.0,
        )
        self.current_volume_db = -60.0
        # Setpoint smoother
        self.target_db = -60.0
        self.filtered_target_db = -60.0
        self.tau = 0.2  # setpoint smoothing time constant

    def set_target(self, db: float):
        self.target_db = db

    def update(self, dt: float) -> float:
        # First, smooth the setpoint
        alpha = dt / (self.tau + dt)
        self.filtered_target_db += alpha * (self.target_db - self.filtered_target_db)

        # PID drives db/dt
        db_per_sec = self.pid.update(
            setpoint=self.filtered_target_db,
            measurement=self.current_volume_db,
            dt=dt,
        )
        self.current_volume_db += db_per_sec * dt
        return self.current_volume_db
```

**Tuning notes**:

- The setpoint smoother is doing most of the work; the PID makes the result robust to DSP-side latency.
- `output_min/max` in dB/s sets a maximum slew rate — directly tunable as "max volume change per second."
- For genuinely instant transitions (gameplay event, hard cut), call `pid.reset()` and set `current_volume_db = target_db` directly. PID is for the smooth case; trust the cut for the cut.


### Scenario 5: Physics Stabilization (Velocity Damper)

**Goal**: Object's velocity decays smoothly toward zero (or a target velocity) without bouncing or framerate-dependent behaviour.

**RED**:

```csharp
velocity *= friction;  // friction = 0.95, applied per frame
```

This is exponential decay with a frame-rate-dependent time constant. Same root cause as Failure 1.

**GREEN**: PID on the velocity error. Output is acceleration.

```csharp
PIDController velocityPid = new PIDController {
    Kp = 5f, Ki = 0f, Kd = 0.5f,
    outputMin = -50f, outputMax = 50f,  // max acceleration
};

void FixedUpdate() {
    float ax = velocityPid.Update(setpoint: targetVelocity.x, measurement: rb.velocity.x, dt: Time.fixedDeltaTime);
    float ay = velocityPid.Update(setpoint: targetVelocity.y, measurement: rb.velocity.y, dt: Time.fixedDeltaTime);
    rb.AddForce(new Vector2(ax, ay) * rb.mass);  // F = m·a
}
```

**Tuning notes**:

- For pure damping (`targetVelocity = 0`), `Ki = 0`. Adding integral action would make the object actively push against any disturbance, which usually isn't what you want.
- Run in `FixedUpdate` so `dt` is constant — physics PID controllers are especially sensitive to variable `dt`.
- `Kd` here counters PID's tendency to overshoot zero velocity (which would manifest as the object oscillating around its rest state). Small but nonzero.


### Scenario 6: Economy System Balance

**Goal**: Currency inflation rate (or any global economic variable) tracks a target. Adjust spawn/sink rates without overshoot. Recover from shocks (player events, content drops).

**RED**:

```python
if inflation > 0.05:
    spawn_rate -= 0.01
elif inflation < 0.02:
    spawn_rate += 0.01
```

Same threshold-and-step pathology as dynamic difficulty (Failure 3). Reaches steady-state error or thrashes; can't both.

**GREEN**: PI controller on inflation rate. The sink rate (or spawn rate) is the actuator.

```python
class EconomyController:
    def __init__(self, target_inflation_per_day=0.02):
        self.target = target_inflation_per_day
        # dt is "in-game days" between samples. Tune accordingly.
        self.pid = PIDController(
            kp=0.5,
            ki=0.05,
            kd=0.0,           # Inflation measurement is integrative; D would amplify noise
            output_min=-0.02, # Max change in spawn rate per day
            output_max=+0.02,
        )
        self.spawn_rate = 1.0  # Multiplier on baseline spawn

    def daily_update(self, measured_inflation_today: float):
        adjustment = self.pid.update(
            setpoint=self.target,
            measurement=measured_inflation_today,
            dt=1.0,  # one day per call
        )
        # Note sign: high inflation → want to reduce spawn → adjustment negative
        self.spawn_rate = max(0.1, self.spawn_rate - adjustment)
```

**Tuning notes**:

- Cross-reference `stability-analysis.md`: validate that the closed-loop economy is stable by linearising around the target. A PI controller on a first-order plant is always stable for sufficiently small `Kp`; for higher-order economies (player wealth distribution, market feedback), check.
- Use a long measurement window (7-day rolling average inflation, not single-day) to reduce noise. A noisy measurement makes integral action unsafe.
- Anti-windup is essential — economies frequently saturate the actuator (spawn rate hits floor or ceiling) for extended periods.


## Advanced Topics

### Cascade Control (Nested PID Loops)

When the actuator is itself a dynamic system, a single PID can be slow or unstable. **Cascade control** uses an outer PID's output as an inner PID's setpoint. The inner loop runs faster and stabilises the actuator; the outer loop runs slower and drives the goal.

Example: Position control of an AI agent. Outer loop sets *desired velocity*; inner loop achieves it via *applied force*.

```python
class CascadeAIController:
    def __init__(self):
        self.position_pid = PIDController(
            kp=2.0, ki=0.0, kd=0.5,
            output_min=-8.0, output_max=8.0,  # max velocity
        )
        self.velocity_pid = PIDController(
            kp=10.0, ki=0.5, kd=0.0,
            output_min=-100.0, output_max=100.0,  # max force
        )

    def update(self, target_pos, current_pos, current_vel, dt):
        # Outer loop: desired velocity from position error
        desired_vel = self.position_pid.update(
            setpoint=0.0,
            measurement=(current_pos - target_pos),  # positive when overshoot
            dt=dt,
        )
        # Inner loop: applied force from velocity error
        applied_force = self.velocity_pid.update(
            setpoint=desired_vel,
            measurement=current_vel,
            dt=dt,
        )
        return applied_force
```

**Design rule**: The inner loop must be at least 5–10× faster than the outer loop's natural response. If the velocity loop can't track the position loop's setpoint changes faster than the position loop changes them, the cascade is no better than a single PID.

**Validation**: Run *only the inner loop* with a step in `desired_vel`. Confirm it tracks within the time budget. Then close the outer loop.

### Gain Scheduling

When the plant's behaviour changes with state (a flying enemy in air vs on ground; an AI in stealth vs combat; a vehicle at low vs high speed), one set of gains can't be optimal everywhere. **Gain scheduling** changes the gains as a function of state.

```python
def get_gains(speed: float):
    if speed < 5.0:
        return (8.0, 0.0, 4.0)   # Tight tracking at low speed
    elif speed < 15.0:
        return (5.0, 0.0, 3.0)   # Balanced
    else:
        return (3.0, 0.0, 2.5)   # Looser tracking at high speed (stability matters more)

# In update loop:
self.pid.kp, self.pid.ki, self.pid.kd = get_gains(current_speed)
```

**Pitfalls**:

- Discontinuous jumps in gains cause discontinuous output. Interpolate between gain sets.
- Stability is *not* preserved across gain regions automatically — verify each region.
- Adding state variables to the schedule blows up tuning effort combinatorially.

### Feed-Forward + PID

When you can predict part of the required output from the *setpoint* alone (not the error), feed-forward removes that load from the PID:

```python
def update(self, setpoint, measurement, dt):
    # Feed-forward: the part of u we know we need from the setpoint alone
    u_ff = self.feed_forward(setpoint)
    # Feedback: PID corrects for what the model didn't capture
    u_fb = self.pid.update(setpoint, measurement, dt)
    return u_ff + u_fb
```

For a position controller chasing a moving target:

```python
def feed_forward(self, target_pos):
    # If we know the target's velocity, command that velocity directly.
    # PID then only corrects for *errors* in our prediction.
    return target_velocity_estimate
```

Result: PID's error is much smaller, so smaller gains suffice; controller is more robust to disturbances.

### Adaptive PID

When the plant changes over time (player levels up, network conditions vary, hardware ages), gains may need to adapt. **Don't** start with adaptive PID — get a fixed-gain controller working first, then layer adaptation on top if measurements show it's needed.

A simple adaptation loop, monitoring a quality metric:

```python
class AdaptivePID:
    def __init__(self):
        self.pid = PIDController(kp=2.0, ki=0.5, kd=1.0)
        self.quality_window = []   # rolling overshoot/oscillation metric

    def update(self, setpoint, measurement, dt):
        u = self.pid.update(setpoint, measurement, dt)
        # Track the recent error trace
        self.quality_window.append(setpoint - measurement)
        if len(self.quality_window) > 100:
            self.quality_window.pop(0)
            self.adapt()
        return u

    def adapt(self):
        errors = self.quality_window
        # Heuristic: lots of sign changes → too much oscillation → reduce Kp, raise Kd
        sign_changes = sum(1 for i in range(1, len(errors))
                           if errors[i] * errors[i-1] < 0)
        if sign_changes > 30:                # oscillating
            self.pid.kp *= 0.95
            self.pid.kd *= 1.05
        elif sign_changes < 5 and abs(sum(errors)) > 10:  # sluggish with bias
            self.pid.kp *= 1.05
```

**Warning**: Adaptive controllers can become unstable in their own right. Test exhaustively. Always have safe gain bounds (`max(min_gain, min(max_gain, kp))`).

### When to Escalate Beyond PID

PID is a great default. It is not a universal solver. Reach for something else when:

| Symptom | Reach for |
|---|---|
| Multiple coupled outputs and inputs (MIMO system) | **State feedback**, **LQR** (Linear Quadratic Regulator) |
| Hard constraints on output trajectory or state (don't hit walls) | **MPC** (Model Predictive Control) |
| Plant is highly nonlinear; gain scheduling explodes | **Sliding mode control** or **gain-scheduled MPC** |
| Plant model is unknown, but you have data | **Iterative learning control**, **RL-based control** (cross-ref `using-deep-rl`) |
| You only need to *reject disturbances*, not track | **H∞ control**, robust control |
| You need provably bounded behaviour despite measurement noise | **Kalman filter + LQR** (LQG) |

For *games*, the PID + cascade + gain scheduling toolkit covers ~95% of practical needs. The remaining 5% — racing AI on a Formula-1 sim, full-body physics-based locomotion, real-time aerodynamics — justify the extra complexity. Most things don't.


## Testing PID Controllers

PID controllers are deterministic objects with simple mathematical properties. Test them directly.

### 1. Unit tests against a known plant

```python
def test_step_response_settles_within_budget():
    pid = PIDController(kp=2.0, ki=0.5, kd=0.5)
    measurement = 0.0
    dt = 0.01
    settling_time = None
    for n in range(2000):
        u = pid.update(setpoint=1.0, measurement=measurement, dt=dt)
        # First-order plant: dx/dt = -x + u
        measurement += (-measurement + u) * dt
        if abs(measurement - 1.0) < 0.02 and settling_time is None:
            # Stay within 2% for 100 consecutive steps to count as settled
            if n > 100:
                settling_time = n * dt
                break
    assert settling_time is not None and settling_time < 1.0
```

### 2. Disturbance rejection

```python
def test_recovers_from_disturbance():
    pid = PIDController(kp=2.0, ki=0.5, kd=0.5)
    measurement, disturbance = 0.0, 0.0
    dt = 0.01
    for n in range(2000):
        # Inject a sustained disturbance halfway through
        if n == 1000:
            disturbance = 0.5
        u = pid.update(setpoint=1.0, measurement=measurement, dt=dt)
        measurement += (-measurement + u - disturbance) * dt
    # PI controller should drive steady-state error to zero despite disturbance
    assert abs(measurement - 1.0) < 0.05
```

### 3. Anti-windup

```python
def test_no_windup_overshoot():
    pid = PIDController(kp=1.0, ki=2.0, kd=0.0,
                       output_min=-1.0, output_max=1.0)
    measurement = 0.0
    dt = 0.01
    log = []
    for n in range(2000):
        # Demand an unreachable setpoint, then drop it
        sp = 100.0 if n < 500 else 0.0
        u = pid.update(setpoint=sp, measurement=measurement, dt=dt)
        # Plant is integrator clamped at u in [-1, 1]
        measurement += u * dt
        log.append(measurement)
    # After the setpoint drops, undershoot should be bounded
    final_phase = log[1000:]
    assert min(final_phase) > -2.0   # Without anti-windup this would be -50+
```

### 4. Frequency-domain validation (when you care)

For controllers that have to meet specific bandwidth or phase margin specs, validate analytically with `scipy.signal.margin` (example in *Transfer Functions and Stability* above). Don't ship a safety-critical controller without checking margins.

### 5. Property-based tests

Useful invariants:

- **Reset clears state**: `pid.reset()` produces identical output for any subsequent input sequence regardless of prior history.
- **Saturation respected**: output is always within `[output_min, output_max]`.
- **Determinism**: same inputs → same outputs across runs (tie to `using-simulation-foundations` determinism guarantees).

Property tests with Hypothesis or QuickCheck-style libraries are particularly good at finding edge cases — `dt = 0`, very large `dt`, alternating saturation conditions.


## Anti-Patterns

| Anti-pattern | Why it bites you |
|---|---|
| **Tune `Kp` first, declare victory** | Steady-state error and oscillation are hidden by an "OK at first glance" tune. Run the disturbance and step-response tests. |
| **Add `Ki` to fix overshoot** | Integral term *causes* overshoot. If overshoot is the problem, increase `Kd` or decrease `Kp`. |
| **Add `Kd` to fix sluggishness** | Derivative damps motion; it doesn't speed it up. If response is slow, increase `Kp` (or feed-forward). |
| **One PID class shared across instances** | If your controller has internal state (`integral`, `prev_measurement`), each controlled axis/object needs its *own* instance. Sharing causes visible cross-talk. |
| **No `reset()`** | Every PID will eventually need to be reset on cutscene, level load, teleport. If you didn't write it, you'll bolt it on under deadline pressure. |
| **No output limits** | Physical actuators always saturate. If your code doesn't clamp, the rest of your system pretends it didn't, and integral wind-up is inevitable. |
| **PID in `Update` instead of `FixedUpdate`** | Variable `dt` ruins integral and derivative tuning. Run PID in the fixed simulation tick. |
| **Measurement filter inside the controller** | Filter the measurement *before* the controller sees it. Mixing filtering and control inside one class makes both harder to tune. |
| **Hand-coded "smart" overrides** | "If error is small, freeze the output." Adds discontinuities; breaks linear analysis; introduces edge cases that fight the controller. Use anti-windup and gain scheduling instead. |
| **Tuning with `Time.timeScale != 1`** | Fixed update tick scales with timeScale; PID tuning won't transfer back to `timeScale = 1`. Always tune at production timescale. |
| **Differentiating noisy inputs without filtering** | The derivative term explodes on noise. Filter or use derivative-on-measurement. |
| **Treating `Lerp(a, b, t)` as a degenerate PID without acknowledging it** | It is a P-only controller with a frame-rate-dependent gain. Either keep using it knowingly, or upgrade to a real PID. Don't pretend it's something else. |


## Cross-References

- **`differential-equations-for-games.md`** — the underlying ODE perspective: PID is a controller for a plant that itself satisfies an ODE.
- **`stability-analysis.md`** — eigenvalue and Lyapunov tests for *closed-loop* stability of cascade systems and nonlinear plants under PID control.
- **`numerical-methods.md`** — for high-frequency control, the integrator you use to step the plant matters as much as the controller; symplectic methods preserve plant stability margins better than explicit Euler.
- **`state-space-modeling.md`** — when you outgrow PID, state feedback and LQR live here.
- **`stochastic-simulation.md`** — for control loops where the *measurement* is stochastic (player win-rate, polled telemetry), filter and tune accordingly.
- **`bravos-simulation-tactics:physics-simulation-patterns`** — implementation patterns for hooking PID into engine physics.
- **`using-deep-rl`** — when the plant is unknown and the cost function is well-defined, reinforcement learning can outperform PID; usually overkill for games.

---

**PID is the workhorse. Tune it, test it, ship it. Reach for fancier control only when PID demonstrably fails — and when it does, you'll know exactly why.**
