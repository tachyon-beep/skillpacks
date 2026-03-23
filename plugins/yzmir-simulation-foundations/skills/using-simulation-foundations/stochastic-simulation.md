
### Failure 1: Loot Pity Breaking (Gacha Game Collapse)

**Scenario**: Mobile gacha game with 3% 5-star character rate, mercy pity system at 90 pulls.

**What They Did**:
```python
def get_loot():
    if random.random() < 0.03:
        return "5-star"
    return "3-star"

def guaranteed_pity(pulls):
    # Every 90 pulls = guaranteed 5-star
    if pulls % 90 == 0:
        return "5-star"
    return get_loot()
```

**What Went Wrong**:
- Pity counter reset after 5-star acquisition
- But distribution across players was uniform: some got 5-star at pull 1, others at 89
- Streamers documented exploiting the pity system
- Whales spending $10K got same odds as free players
- Community discovered: no difference in spend vs luck
- Player spending dropped 60% when analysis leaked
- Gacha ethics investigation launched

**Why No One Caught It**:
- No statistical testing of distribution fairness
- Didn't track expected value vs actual across player segments
- Assumed uniform randomness solved fairness

**What Stochastic Simulation Shows**:
```python
import numpy as np

# Simulate 10,000 players pulling
pulls_needed = []
for _ in range(10000):
    for pull in range(1, 91):
        if random.random() < 0.03:
            pulls_needed.append(pull)
            break

# Check distribution fairness
print(f"Median pulls: {np.median(pulls_needed)}")  # Expected: ~24
print(f"p99: {np.percentile(pulls_needed, 99)}")   # Expected: ~85
print(f"Min/Max: {min(pulls_needed)}/{max(pulls_needed)}")

# Expected value check: E[pulls] = 1/0.03 = 33.33
print(f"Mean: {np.mean(pulls_needed)}")  # Should be ~33, not skewed
```

Fair system must prove: distribution matches theory across all player segments.


### Failure 2: Crit Streaks Feeling Cheated (RPG Balance)

**Scenario**: Turn-based RPG with 20% crit rate. Player expectations: 1 crit per 5 hits.

**What They Did**:
```python
def apply_crit():
    return random.random() < 0.20

# Player uses sword 5 times
for i in range(5):
    if apply_crit():
        print(f"CRIT on hit {i+1}!")
```

**What Went Wrong**:
- With true 20% independence, probability of 5 hits with 0 crits = 0.8^5 = 0.328 (33%)
- Players experience 3-4 "no crit" streaks per session feeling cheated
- Forums fill with "RNG is broken" complaints
- Actually: RNG is correct, but feels wrong
- Can't change RNG without changing game balance

**Why No One Caught It**:
- No expectation-setting for variance
- Didn't simulate player perception vs actual distribution
- Thought balance numbers = player satisfaction

**What Stochastic Simulation Shows**:
```python
# Simulate 100,000 combat sessions
no_crit_streaks = 0
for session in range(100000):
    crits_in_5 = sum(1 for _ in range(5) if random.random() < 0.20)
    if crits_in_5 == 0:
        no_crit_streaks += 1

print(f"Probability of 0 crits in 5: {no_crit_streaks / 100000}")
# Output: ~0.328 (matches theory: 0.8^5)

# Solution: Use variance reduction (guaranteed crit every X hits) or
# tell players explicitly: "20% per hit means you'll see streaks"
```


### Failure 3: Procedural Generation Repetition (Open World Sameness)

**Scenario**: Roguelike dungeon with seeded randomness for levels.

**What They Did**:
```python
random.seed(level_number)
for x in range(width):
    for y in range(height):
        if random.random() < 0.3:
            place_wall(x, y)
```

**What Went Wrong**:
- Rooms generated from weak LCG seed divergence
- Every run at level 5 generated identical room layout
- Speedrunners memorize every level
- "Procedural generation" feels scripted after 3 playthroughs
- Roguelike replay value becomes memorization

**Why No One Caught It**:
- Didn't verify seed space coverage
- Assumed linear congruential generators had sufficient period
- No ensemble testing of distinctness

**What Stochastic Simulation Shows**:
```python
# Test distinctness using Perlin noise (proper stochastic process)
from opensimplex import OpenSimplex

def better_generation(seed, level_num):
    noise = OpenSimplex(seed=seed)
    for x in range(width):
        for y in range(height):
            # Perlin noise: continuous, smooth variation
            value = noise.noise2(x * 0.1, y * 0.1 + level_num * 100)
            if value > 0.3:
                place_wall(x, y)

# Simulate 100 dungeons
distinctness = set()
for level in range(100):
    layout = frozenset(generate_walls(level))
    distinctness.add(layout)

print(f"Unique layouts from 100 levels: {len(distinctness)}")
# Should be 100, not 2-3
```


### Failure 4: AI Decisions Feeling Stupid (Combat Uncertainty)

**Scenario**: Boss AI makes combat decisions based on random choice.

**What They Did**:
```python
def boss_decide_action():
    choice = random.choice(["attack", "defend", "spell", "dodge"])
    return choice

# Boss picks action every frame independently
```

**What Went Wrong**:
- Boss alternates actions with no pattern or learning
- Randomness per-frame means boss spins around, attacks self, ignores threats
- Feels stupid, not challenging
- Players abuse: dodge random attacks with 25% success, guaranteed to land hits

**Why No One Caught It**:
- Thought randomness = unpredictable = challenging
- Didn't model uncertainty as incomplete information, not noise

**What Stochastic Simulation Shows**:
```python
# Model AI uncertainty as incomplete information about player state
class BossAI:
    def __init__(self):
        self.player_threat_estimate = 0.5  # Markov state
        self.action_count = 0

    def observe_player(self, player_state):
        # Update threat estimate with observation
        # Uncertainty decreases as AI gathers info
        if player_state.health < 0.3:
            self.player_threat_estimate = 0.9
        elif self.action_count % 3 == 0:
            self.player_threat_estimate *= 0.8  # Fade if safe

    def decide(self):
        # Decision depends on threat state + randomness
        if self.player_threat_estimate > 0.7:
            # High threat: favor defense/dodge
            return np.random.choice(
                ["attack", "defend", "spell", "dodge"],
                p=[0.2, 0.3, 0.2, 0.3]  # Biased by state
            )
        else:
            # Low threat: attack more
            return np.random.choice(
                ["attack", "defend", "spell", "dodge"],
                p=[0.5, 0.2, 0.2, 0.1]
            )
```


## GREEN Phase: Stochastic Simulation Foundations

### 1. Introduction to Stochastic Simulation

**What is it?**
A stochastic process is a sequence of random variables indexed by time or space. Unlike deterministic simulation (physics always gives same result), stochastic simulation explicitly models randomness.

**Key Insight**: Randomness is not chaos. With enough samples, random processes converge to predictable distributions—this is the law of large numbers.

**Three Levels**:

1. **Independent randomness**: Each event uncorrelated (coin flips)
   ```python
   # Each coin flip independent
   flips = [random.choice([0, 1]) for _ in range(100)]
   ```

2. **Markov process**: Next state depends only on current state, not history
   ```python
   # Weather: tomorrow depends on today, not yesterday
   state = "sunny"
   transitions = {
       "sunny": {"sunny": 0.8, "rainy": 0.2},
       "rainy": {"sunny": 0.6, "rainy": 0.4}
   }
   next_state = np.random.choice(
       list(transitions[state].keys()),
       p=list(transitions[state].values())
   )
   ```

3. **Continuous stochastic process**: Randomness at every point in time (Brownian motion, SDEs)
   ```python
   # Stock price with drift and volatility
   dt = 0.01
   dW = np.random.normal(0, np.sqrt(dt))
   price_change = 0.05 * price * dt + 0.2 * price * dW
   ```


### 2. Probability Distributions for Games

**Normal Distribution: Continuous abilities, variation around average**

```python
import numpy as np

# Character attack damage: mean 50, std 10
damage = np.random.normal(50, 10)

# Simulate 10,000 attacks to verify distribution
damages = np.random.normal(50, 10, 10000)
print(f"Mean: {np.mean(damages)}")  # ~50
print(f"Std: {np.std(damages)}")    # ~10
print(f"95% range: {np.percentile(damages, 2.5):.1f} - {np.percentile(damages, 97.5):.1f}")
# Output: ~30-70 (within ±2 std)
```

**Exponential Distribution: Time until event (cooldown recovery, enemy arrival)**

```python
# Enemy waves spawn with exponential spacing (mean 30s)
import numpy as np

mean_time_between_spawns = 30
spawn_time = np.random.exponential(mean_time_between_spawns)
print(f"Next wave in {spawn_time:.1f}s")

# Simulate 1000 waves
wave_times = np.random.exponential(30, 1000)
print(f"Average spacing: {np.mean(wave_times):.1f}s")  # ~30s
print(f"p90: {np.percentile(wave_times, 90):.1f}s")     # ~69s (some long waits)
print(f"p10: {np.percentile(wave_times, 10):.1f}s")     # ~3s (sometimes quick)
```

**Poisson Distribution: Discrete event count (enemies per wave, resources per tile)**

```python
# Average 5 enemies per wave, actual varies
import numpy as np

enemy_count = np.random.poisson(5)  # Could be 0, 1, 2, ... 10+

# Simulate 1000 waves
wave_counts = np.random.poisson(5, 1000)
print(f"Average enemies/wave: {np.mean(wave_counts):.1f}")  # ~5
print(f"Most common: {np.argmax(np.bincount(wave_counts))}")  # 5
print(f"p95 wave size: {np.percentile(wave_counts, 95):.0f}")  # ~11 enemies
```

**Beta Distribution: Probabilities and rates (player skill, crit chance)**

```python
# Player skill: most players mediocre, few very good/bad
import numpy as np

skill = np.random.beta(5, 5)  # Symmetric: mean 0.5, concentrated
skill_skewed = np.random.beta(2, 5)  # Right-skewed: more low players

print(f"Fair skill distribution (0-1): {skill:.2f}")
print(f"Skewed (more casual): {skill_skewed:.2f}")

# Can convert to percentile or 0-100 scale
crit_chance = np.random.beta(5, 5) * 0.40  # 0-40% based on skill
```

**Exponential Power Law: Rare events (legendary drops, catastrophic failures)**

```python
# Pareto distribution: 80/20 rule
# 20% of weapons do 80% of damage

def pareto_rarity(min_value=1.0, alpha=1.5, samples=1000):
    return min_value / np.random.uniform(0, 1, samples) ** (1/alpha)

rarities = pareto_rarity(min_value=1.0, alpha=2.0)
print(f"Mean drop rate: {np.mean(rarities):.2f}")
print(f"p99: {np.percentile(rarities, 99):.1f}")  # Legendary: 100x common
```


### 3. Random Walks and Brownian Motion

**Simple Random Walk: Cumulative randomness (player gold over many trades)**

```python
import numpy as np

# Player starts with 100 gold, gains/loses 1 per trade (50/50)
def random_walk(steps, start=100):
    changes = np.random.choice([-1, 1], steps)
    return start + np.cumsum(changes)

positions = random_walk(1000, start=100)
print(f"Starting: 100")
print(f"After 1000 trades: {positions[-1]:.0f}")
print(f"Possible range: 100±√1000 ≈ 100±32")

# Plot to see: looks like Brownian motion
import matplotlib.pyplot as plt
plt.plot(positions)
plt.title("Random Walk: Gold Over Time")
plt.xlabel("Trade #")
plt.ylabel("Gold")
```

**Brownian Motion: Continuous random walk (asset prices, position noise)**

```python
# Price with drift (upward trend) and volatility
def brownian_motion(drift=0.05, volatility=0.2, steps=1000, dt=0.01):
    dW = np.random.normal(0, np.sqrt(dt), steps)
    changes = drift * dt + volatility * dW
    return np.exp(np.cumsum(changes))  # Log-normal price

prices = brownian_motion(drift=0.05, volatility=0.2)
print(f"Starting price: 1.00")
print(f"Expected growth: exp(0.05*10) = {np.exp(0.05*10):.2f}")
print(f"Actual price: {prices[-1]:.2f}")

# With zero drift, price is martingale (fair game)
fair_prices = brownian_motion(drift=0, volatility=0.2)
print(f"Fair game (no drift) final: {fair_prices[-1]:.2f}")
```

**Mean Reversion: Randomness with equilibrium (stamina recovery, health drain)**

```python
# Health drifts back to 100 even with random damage
def mean_reversion(target=100, strength=0.1, volatility=5, steps=1000):
    values = [100]
    for _ in range(steps):
        # Drift toward target + random shock
        change = strength * (target - values[-1]) + np.random.normal(0, volatility)
        values.append(max(0, values[-1] + change))
    return values

health = mean_reversion(target=100, strength=0.2, volatility=5)
print(f"Health over 1000 frames")
print(f"Mean: {np.mean(health):.1f}")  # ~100
print(f"Std: {np.std(health):.1f}")   # ~20 (variance around target)
```


### 4. Monte Carlo Methods

**Estimating Probabilities by Sampling**

```python
import numpy as np

# What's probability of 3+ crits in 10 attacks (20% crit rate)?
def monte_carlo_crit_probability(n_attacks=10, crit_rate=0.20, samples=100000):
    crit_counts = np.random.binomial(n=n_attacks, p=crit_rate, size=samples)
    success = np.sum(crit_counts >= 3)
    return success / samples

prob_3plus = monte_carlo_crit_probability()
print(f"P(3+ crits in 10): {prob_3plus:.4f}")  # ~0.3222

# Theory: P(X >= 3) where X ~ Binomial(10, 0.2)
from scipy.stats import binom
theory_prob = 1 - binom.cdf(2, n=10, p=0.20)
print(f"Theory: {theory_prob:.4f}")
```

**Estimating Expected Value by Averaging**

```python
# What's expected cost to get 5-star with 3% rate and 90-pull pity?
def monte_carlo_expected_pulls(rate=0.03, pity_threshold=90, samples=10000):
    pulls_list = []
    for _ in range(samples):
        for pull in range(1, pity_threshold + 1):
            if np.random.random() < rate:
                pulls_list.append(pull)
                break
        else:
            pulls_list.append(pity_threshold)
    return np.mean(pulls_list), np.std(pulls_list), np.percentile(pulls_list, 99)

mean_pulls, std_pulls, p99 = monte_carlo_expected_pulls()
print(f"Expected pulls: {mean_pulls:.1f} ± {std_pulls:.1f}")
print(f"p99: {p99:.0f}")
# Theory: E[pulls] = 1/0.03 = 33.33 (before pity kicks in)
```

**Path-Dependent Probabilities**

```python
# Gambler's ruin: probability of bankruptcy before reaching goal
def gamblers_ruin_monte_carlo(start=50, goal=100, lose_threshold=0,
                               win_prob=0.5, samples=10000):
    successes = 0
    for _ in range(samples):
        capital = start
        while lose_threshold < capital < goal:
            capital += 1 if np.random.random() < win_prob else -1
        if capital == goal:
            successes += 1
    return successes / samples

# Fair game (50/50): theory says P(success) = start / goal
fair_prob = gamblers_ruin_monte_carlo(start=50, goal=100, win_prob=0.5)
print(f"Fair game P(reach 100 before 0): {fair_prob:.3f}")  # ~0.5

# Unfair game (45/55 against player): much lower success
unfair_prob = gamblers_ruin_monte_carlo(start=50, goal=100, win_prob=0.45)
print(f"Unfair P(reach 100 before 0): {unfair_prob:.3f}")  # ~0.003
```


### 5. Stochastic Differential Equations

**Framework: dX = f(X)dt + g(X)dW**

Where:
- f(X)dt = deterministic drift
- g(X)dW = random shock (dW = Brownian increment)

**Stock Price (Geometric Brownian Motion)**

```python
import numpy as np

# dS = μS dt + σS dW
# Solution: S(t) = S(0) * exp((μ - σ²/2)t + σW(t))

def geometric_brownian_motion(S0=100, mu=0.05, sigma=0.2, T=1.0, steps=252):
    dt = T / steps
    W = np.cumsum(np.random.normal(0, np.sqrt(dt), steps))
    t = np.linspace(0, T, steps)
    S = S0 * np.exp((mu - sigma**2/2) * t + sigma * W)
    return S

prices = geometric_brownian_motion(S0=100, mu=0.05, sigma=0.2)
print(f"Starting: 100")
print(f"Expected final (theory): 100 * exp(0.05) = {100 * np.exp(0.05):.2f}")
print(f"Actual final: {prices[-1]:.2f}")
```

**Mean-Reverting Process (Ornstein-Uhlenbeck)**

```python
# dX = θ(μ - X)dt + σ dW
# Reverts to mean μ at speed θ

def ornstein_uhlenbeck(X0=0, mu=0, theta=0.1, sigma=0.2, T=1.0, steps=252):
    dt = T / steps
    X = np.zeros(steps)
    X[0] = X0
    for i in range(1, steps):
        dW = np.random.normal(0, np.sqrt(dt))
        X[i] = X[i-1] + theta * (mu - X[i-1]) * dt + sigma * dW
    return X

ou_path = ornstein_uhlenbeck(X0=2, mu=0, theta=0.5, sigma=0.2)
print(f"Starting: 2")
print(f"Mean over time: {np.mean(ou_path):.2f}")  # ~0 (target)
print(f"Std: {np.std(ou_path):.2f}")  # ~sqrt(σ²/2θ) = ~0.2
```

**Jump-Diffusion Process (Rare events)**

```python
# dX = μX dt + σX dW + J dN(λ)
# J = jump size, N(λ) = Poisson process (λ jumps per unit time)

def jump_diffusion(X0=100, mu=0.05, sigma=0.2, lambda_=1,
                   jump_mean=-0.1, jump_std=0.05, T=1.0, steps=252):
    dt = T / steps
    X = np.zeros(steps)
    X[0] = X0
    for i in range(1, steps):
        dW = np.random.normal(0, np.sqrt(dt))
        # Diffusion part
        dX = mu * X[i-1] * dt + sigma * X[i-1] * dW
        # Jump part: Poisson rate λ
        jump_count = np.random.poisson(lambda_ * dt)
        if jump_count > 0:
            jump = X[i-1] * np.random.normal(jump_mean, jump_std, jump_count).sum()
            dX += jump
        X[i] = max(0, X[i-1] + dX)
    return X

jd_path = jump_diffusion(X0=100, lambda_=2, jump_mean=-0.05, jump_std=0.02)
print(f"Path includes random crashes (jumps)")
print(f"Min: {np.min(jd_path):.1f}")
print(f"Max: {np.max(jd_path):.1f}")
```


### 6. Game Applications: Loot, Crits, Proc-Gen, AI

#### Loot Drops: Fair Distribution

```python
import numpy as np
from collections import Counter

# System: 3% 5-star, 10% 4-star, 87% 3-star
# With 90-pull pity (guarantees 5-star)
# With 10-pull soft pity (increases rate)

def simulate_loot_system(pulls=1000, samples=10000):
    """Simulate pulls across many players to verify fairness"""
    all_results = []

    for player in range(samples):
        pity_counter = 0
        rarity_counts = {3: 0, 4: 0, 5: 0}

        for pull in range(pulls):
            pity_counter += 1

            # Soft pity: increase 5-star rate after 74 pulls
            rate_5 = 0.03 if pity_counter < 74 else 0.05

            rand = np.random.random()
            if pity_counter == 90:
                # Hard pity guarantee
                rarity = 5
                pity_counter = 0
            elif rand < rate_5:
                rarity = 5
                pity_counter = 0
            elif rand < rate_5 + 0.10:
                rarity = 4
            else:
                rarity = 3

            rarity_counts[rarity] += 1

        all_results.append(rarity_counts)

    # Aggregate statistics
    all_5star_count = [r[5] for r in all_results]
    print(f"5-star drops per {pulls} pulls:")
    print(f"  Mean: {np.mean(all_5star_count):.1f}")  # Should be ~30
    print(f"  Std: {np.std(all_5star_count):.1f}")
    print(f"  Min/Max: {np.min(all_5star_count)}/{np.max(all_5star_count)}")

    # Fairness test: is variance reasonable?
    expected_mean = pulls * 0.03
    print(f"  Expected: {expected_mean:.1f}")
    print(f"  Fair system? {abs(np.mean(all_5star_count) - expected_mean) < 1.0}")

simulate_loot_system(pulls=1000)
```

#### Critical Strikes: Meaningful Variance

```python
import numpy as np

# Problem: 20% crit rate with ±0.8s variance feels unfair
# Solution: Use variance reduction with "guaranteed crit every N hits"

class CritSystem:
    def __init__(self, crit_rate=0.20, guaranteed_every=5):
        self.crit_rate = crit_rate
        self.guaranteed_every = guaranteed_every
        self.attacks_since_crit = 0

    def try_crit(self):
        self.attacks_since_crit += 1

        # Guarantee: every Nth hit
        if self.attacks_since_crit >= self.guaranteed_every:
            self.attacks_since_crit = 0
            return True

        # Otherwise: random with reduced rate
        # Adjust rate so expected hits match original
        effective_rate = self.crit_rate - (1 / self.guaranteed_every)
        if np.random.random() < effective_rate:
            self.attacks_since_crit = 0
            return True

        return False

# Simulate 1000 battles with 20 attacks each
crit_sys = CritSystem(crit_rate=0.20, guaranteed_every=5)
crit_counts = []

for battle in range(1000):
    crits = sum(1 for _ in range(20) if crit_sys.try_crit())
    crit_counts.append(crits)

print(f"Crits per 20-attack battle:")
print(f"  Mean: {np.mean(crit_counts):.1f}")  # Should be ~4 (20% of 20)
print(f"  Std: {np.std(crit_counts):.1f}")   # Reduced variance!
print(f"  Min/Max: {min(crit_counts)}/{max(crit_counts)}")
# With guarantee: 1-7 crits (tighter than pure 0-12)
# Without guarantee: 0-12 crits (includes dry spells)
```

#### Procedural Generation: Stochastic Patterns

```python
import numpy as np
from opensimplex import OpenSimplex

class ProceduralDungeon:
    def __init__(self, seed=None, width=100, height=100):
        self.seed = seed
        self.width = width
        self.height = height
        self.noise = OpenSimplex(seed=seed)

    def generate_room(self, level=0, room_num=0):
        """Generate room using Perlin noise for coherent randomness"""
        grid = np.zeros((self.height, self.width))

        for x in range(self.width):
            for y in range(self.height):
                # Multi-scale noise for natural look
                scale1 = self.noise.noise2(
                    x * 0.05, y * 0.05 + level * 1000 + room_num * 500
                )  # Large features
                scale2 = self.noise.noise2(
                    x * 0.2, y * 0.2 + level * 100 + room_num * 50
                )  # Medium features
                scale3 = self.noise.noise2(
                    x * 0.5, y * 0.5 + level * 10 + room_num * 5
                )  # Detail

                # Combine scales
                value = (0.5 * scale1 + 0.3 * scale2 + 0.2 * scale3) / 1.0

                # Convert to wall placement
                grid[y, x] = 1 if value > 0.2 else 0

        return grid

    def verify_distinct(self, levels=100):
        """Verify each level is unique"""
        layouts = set()
        for level in range(levels):
            room = self.generate_room(level=level)
            # Hash room layout
            layout_hash = hash(room.tobytes())
            layouts.add(layout_hash)

        uniqueness = len(layouts) / levels
        print(f"Uniqueness: {uniqueness:.1%}")  # Should be 100%
        return uniqueness

dungeon = ProceduralDungeon(seed=12345)
dungeon.verify_distinct(levels=50)
```

#### AI Uncertainty: Intelligent Randomness

```python
import numpy as np

class BossAI:
    def __init__(self):
        self.threat_level = 0.5  # 0 = safe, 1 = danger
        self.confidence = 0.1    # How sure is AI about state
        self.action_history = []

    def observe(self, player_health, player_distance, time_since_hit):
        """Update threat estimate based on observations"""
        threats = []

        # Low health = threat
        if player_health < 0.3:
            threats.append(0.9)
        elif player_health < 0.6:
            threats.append(0.6)

        # Close range = threat
        if player_distance < 50:
            threats.append(0.7)
        elif player_distance < 100:
            threats.append(0.4)

        # Just took damage = threat
        if time_since_hit < 2:
            threats.append(0.8)
        elif time_since_hit > 10:
            threats.append(0.2)

        if threats:
            # Exponential moving average: new info weights 20%
            self.threat_level = 0.2 * np.mean(threats) + 0.8 * self.threat_level

        # Confidence increases with data
        self.confidence = min(1.0, self.confidence + 0.05)

    def decide_action(self):
        """Choose action based on threat and uncertainty"""
        # High threat: defensive bias
        if self.threat_level > 0.7:
            actions = ["dodge", "defend", "spell"]
            probs = [0.4, 0.3, 0.3]
        # Medium threat: balanced
        elif self.threat_level > 0.4:
            actions = ["attack", "dodge", "spell", "defend"]
            probs = [0.3, 0.3, 0.2, 0.2]
        # Low threat: aggressive
        else:
            actions = ["attack", "spell", "dodge"]
            probs = [0.5, 0.3, 0.2]

        # Low confidence: add randomness (unsure)
        if self.confidence < 0.5:
            probs = [p * 0.5 + 0.25 for p in probs]
            probs = [p / sum(probs) for p in probs]

        action = np.random.choice(actions, p=probs)
        self.action_history.append(action)
        return action

# Simulate combat
boss = BossAI()
actions_taken = []

for frame in range(200):
    player_health = max(0.1, 1.0 - frame * 0.002)
    player_distance = 75 + 25 * np.sin(frame * 0.1)
    time_since_hit = frame % 30

    boss.observe(player_health, player_distance, time_since_hit)
    action = boss.decide_action()
    actions_taken.append(action)

    if frame in [50, 100, 150, 200]:
        print(f"Frame {frame}: threat={boss.threat_level:.2f}, "
              f"confidence={boss.confidence:.2f}, action={action}")
```


### 7. Implementation Patterns

**Pattern 1: Seeded Randomness for Reproducibility**

```python
import numpy as np

# Create deterministic random generator from seed
class DeterministicRNG:
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def next_float(self, low=0, high=1):
        """Reproducible float"""
        return self.rng.uniform(low, high)

    def next_int(self, low, high):
        """Reproducible integer"""
        return self.rng.randint(low, high)

# Same seed = same results
rng1 = DeterministicRNG(seed=42)
rng2 = DeterministicRNG(seed=42)

results1 = [rng1.next_float() for _ in range(5)]
results2 = [rng2.next_float() for _ in range(5)]

assert results1 == results2  # Reproducible
print(f"Both sequences: {results1}")
```

**Pattern 2: Tracking Distribution Over Time**

```python
import numpy as np
from collections import defaultdict

class DistributionTracker:
    def __init__(self, name, expected_prob=None):
        self.name = name
        self.expected_prob = expected_prob
        self.samples = defaultdict(int)
        self.total = 0

    def record(self, outcome):
        """Record one outcome"""
        self.samples[outcome] += 1
        self.total += 1

    def report(self):
        """Check if distribution matches expectation"""
        print(f"\n{self.name}:")
        for outcome in sorted(self.samples.keys()):
            observed = self.samples[outcome] / self.total
            expected = self.expected_prob.get(outcome, 0) if self.expected_prob else None

            if expected:
                diff = abs(observed - expected)
                status = "OK" if diff < 0.02 else "DEVIATION"
                print(f"  {outcome}: {observed:.4f} (expected {expected:.4f}) {status}")
            else:
                print(f"  {outcome}: {observed:.4f}")

# Track loot rarity
tracker = DistributionTracker(
    "Loot Distribution",
    expected_prob={"common": 0.7, "rare": 0.25, "legendary": 0.05}
)

for _ in range(10000):
    rand = np.random.random()
    if rand < 0.05:
        tracker.record("legendary")
    elif rand < 0.30:
        tracker.record("rare")
    else:
        tracker.record("common")

tracker.report()
```

**Pattern 3: Variance Reduction Techniques**

```python
import numpy as np

# Antithetic variates: pair random values to reduce variance
def estimate_pi_naive(samples=10000):
    """Naive: uniform random points in square"""
    inside = 0
    for _ in range(samples):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        if x**2 + y**2 < 1:
            inside += 1
    return 4 * inside / samples

def estimate_pi_antithetic(samples=10000):
    """Antithetic: use complement points too"""
    inside = 0
    for _ in range(samples // 2):
        # First point
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        if x**2 + y**2 < 1:
            inside += 1

        # Antithetic (reflection): tends to complement first point
        x2, y2 = -x, -y
        if x2**2 + y2**2 < 1:
            inside += 1

    return 4 * inside / samples

# Antithetic has lower variance
estimates_naive = [estimate_pi_naive(1000) for _ in range(100)]
estimates_antithetic = [estimate_pi_antithetic(1000) for _ in range(100)]

print(f"Naive std: {np.std(estimates_naive):.4f}")
print(f"Antithetic std: {np.std(estimates_antithetic):.4f}")
# Antithetic: lower variance
```


### 8. Decision Framework

**When to use each distribution/process:**

| System | Distribution | Reason |
|--------|--------------|--------|
| Ability damage | Normal | Natural variation, doesn't go negative |
| Cooldown timers | Exponential | Time-until-event is memoryless |
| Rare drops | Beta/Pareto | Heavy tail for legendary items |
| Enemy spawns | Poisson | Count of events in time window |
| Stock prices | Geometric BM | Log-normal returns, can't go negative |
| Health | Ornstein-Uhlenbeck | Reverts to max, bounded |
| Procedural terrain | Perlin noise | Spatially coherent randomness |
| AI decisions | Markov chain | State-dependent behavior |

**Questions to ask before implementing randomness:**

1. Is this independent or does history matter?
   - Independent → Bernoulli/uniform trials
   - History matters → Markov/SDE

2. Can the value go negative?
   - No → Log-normal, exponential, Beta
   - Yes → Normal, uniform, mixture

3. Should large jumps be possible?
   - No → Diffusion (Brownian motion)
   - Yes → Jump-diffusion, mixture processes

4. Is there an equilibrium or target?
   - Yes → Mean reversion (Ornstein-Uhlenbeck)
   - No → Random walk (Geometric BM)

5. Should distribution be spatially/temporally coherent?
   - Yes → Perlin/Simplex noise, Gaussian processes
   - No → Independent sampling


### 9. Common Pitfalls

**Pitfall 1: Forgetting Variance Reduction**

```python
# BAD: Every crit is independent, leads to 0-crit and 5-crit runs
def bad_crit(n_attacks=10, rate=0.20):
    return sum(1 for _ in range(n_attacks) if random.random() < rate)

# GOOD: Variance reduction with pity
def good_crit(n_attacks=10, rate=0.20, guaranteed_every=5):
    crit_count = 0
    hits_since_crit = 0
    for _ in range(n_attacks):
        hits_since_crit += 1
        if hits_since_crit >= guaranteed_every:
            crit_count += 1
            hits_since_crit = 0
        elif random.random() < rate * 0.8:  # Reduced rate
            crit_count += 1
            hits_since_crit = 0
    return crit_count
```

**Pitfall 2: Using Bad RNG Generators**

```python
# BAD: Python's default random (Mersenne Twister, low period in some dimensions)
import random
seed_value = random.getrandbits(32)

# GOOD: NumPy's generators with modern algorithms
import numpy as np
rng = np.random.default_rng(seed=42)  # Uses PCG64
value = rng.uniform(0, 1)
```

**Pitfall 3: Ignoring Time-Dependence**

```python
# BAD: Stateless randomness (can lead to repeats)
def bad_spawn_enemies():
    if random.random() < 0.02:  # 2% spawn chance per frame
        spawn_enemy()

# GOOD: Markov process with state
class SpawnerWithState:
    def __init__(self):
        self.time_since_spawn = 0

    def update(self, dt):
        self.time_since_spawn += dt
        # Exponential distribution: spawn when time drawn from Exp(λ)
        if self.time_since_spawn > self.spawn_interval:
            spawn_enemy()
            self.spawn_interval = np.random.exponential(30)  # Mean 30s
            self.time_since_spawn = 0
```

**Pitfall 4: Not Testing Distribution Fairness**

```python
# GOOD: Always verify distribution matches claims
def verify_drop_rates(rate, samples=100000):
    from scipy.stats import binom_test

    successes = sum(1 for _ in range(samples) if random.random() < rate)

    # Binomial test: is observed count statistically consistent with rate?
    p_value = binom_test(successes, samples, rate, alternative='two-sided')

    if p_value > 0.05:
        print(f"Distribution OK: {successes/samples:.4f} ≈ {rate:.4f}")
    else:
        print(f"Distribution SKEWED: {successes/samples:.4f} != {rate:.4f}")
        print(f"p-value: {p_value}")
```


### 10. Testing Stochastic Systems

**Unit Test: Verify Average Behavior**

```python
import numpy as np
from scipy.stats import binom_test

def test_crit_rate():
    """Verify critical strike rate matches expected"""
    crit_sys = CritSystem(crit_rate=0.20)

    crit_count = sum(1 for _ in range(10000) if crit_sys.try_crit())
    expected = 2000  # 20% of 10000

    # Allow 2% deviation (reasonable for randomness)
    assert abs(crit_count - expected) < 200, \
        f"Crit count {crit_count} != expected {expected}"

def test_loot_distribution():
    """Verify loot rates across many players"""
    from collections import Counter

    drops = []
    for player in range(1000):
        for pull in range(100):
            if np.random.random() < 0.03:
                drops.append("5-star")
                break
        else:
            drops.append("none")

    rate_observed = drops.count("5-star") / len(drops)
    rate_expected = 0.03

    # Chi-square test
    from scipy.stats import chi2_contingency
    counts = [drops.count("5-star"), drops.count("none")]
    expected_counts = [1000 * rate_expected, 1000 * (1 - rate_expected)]

    chi2 = sum((o - e)**2 / e for o, e in zip(counts, expected_counts))
    assert chi2 < 10, f"Distribution significantly different: χ² = {chi2}"

def test_monte_carlo_convergence():
    """Verify Monte Carlo estimates improve with samples"""
    estimates = []
    for n_samples in [100, 1000, 10000, 100000]:
        # Estimate P(X >= 3) for Binomial(10, 0.2)
        count = sum(
            1 for _ in range(n_samples)
            if sum(1 for _ in range(10) if np.random.random() < 0.2) >= 3
        )
        estimate = count / n_samples
        estimates.append(estimate)

    # Each estimate should be closer to truth (0.3222)
    errors = [abs(e - 0.3222) for e in estimates]
    assert all(errors[i] >= errors[i+1] * 0.5 for i in range(len(errors)-1)), \
        f"Convergence failed: errors = {errors}"

# Run tests
test_crit_rate()
test_loot_distribution()
test_monte_carlo_convergence()
print("All stochastic tests passed!")
```

**Integration Test: Scenario Simulation**

```python
def test_loot_drop_scenario():
    """Test Scenario 1: Loot drops should be fair across all players"""
    game = GameWorld()

    # 1000 players, each farm 500 mobs
    player_drops = []
    for player_id in range(1000):
        drops = []
        for mob_id in range(500):
            loot = game.defeat_mob(mob_id, player_id)
            if "legendary" in loot:
                drops.append(1)
        player_drops.append(sum(drops))

    # Verify: mean drops should be close to 50 (500 * 0.1%)
    mean_drops = np.mean(player_drops)
    assert 45 < mean_drops < 55, f"Mean drops {mean_drops} out of expected range"

    # Verify: no player should have extreme luck (> 2std)
    std_drops = np.std(player_drops)
    outliers = sum(1 for d in player_drops if abs(d - mean_drops) > 3 * std_drops)
    assert outliers < 5, f"Too many outliers: {outliers} players"

def test_crit_streak_fairness():
    """Test Scenario 2: Crit streaks feel fair within 10 attacks"""
    game = GameWorld()

    # Simulate 10,000 combat sessions
    session_max_streak = []
    for _ in range(10000):
        max_streak = 0
        current_streak = 0
        for attack in range(10):
            if game.apply_crit():
                current_streak += 1
            else:
                max_streak = max(max_streak, current_streak)
                current_streak = 0
        session_max_streak.append(max_streak)

    # Expected: max streak shouldn't exceed 5 more than 10% of time
    p99_streak = np.percentile(session_max_streak, 99)
    assert p99_streak < 6, f"Max streak too high: {p99_streak}"
```


### REFACTOR Scenarios: 6+ Applications

#### Scenario 1: Gacha Loot System
**Goal**: 3% 5-star, pity at 90, soft pity at 75, fairness across all players
**Metrics**: Expected pulls, p95 pulls, fairness χ² test
**Code**: See section 6, loot drops example

#### Scenario 2: Critical Strike System
**Goal**: 20% crit, variance reduction, guaranteed every 5 hits
**Metrics**: Mean crits/10 attacks, std dev, max streak distribution
**Code**: See section 6, crits example

#### Scenario 3: Procedural Dungeon Generation
**Goal**: Unique layouts, coherent rooms, no memorable patterns
**Metrics**: Uniqueness rate, distinctness hash, player recurrence survey
**Code**: See section 6, proc-gen example

#### Scenario 4: AI Decision-Making
**Goal**: Intelligent randomness, state-dependent behavior, fair odds
**Metrics**: Action distribution by threat level, win rate parity
**Code**: See section 6, AI uncertainty example

#### Scenario 5: Market Fluctuations
**Goal**: Price dynamics with drift, volatility, rare crashes
**Metrics**: Mean return, volatility, crash probability
**Implementation**:
```python
def market_simulation():
    # Use Geometric Brownian Motion + jump-diffusion
    # Track price path, verify statistical properties
    prices = jump_diffusion(X0=100, mu=0.05, sigma=0.15, lambda_=0.5)
    returns = np.diff(np.log(prices))

    # Verify properties
    assert abs(np.mean(returns) - 0.05) < 0.01  # Drift matches
    assert abs(np.std(returns) - 0.15) < 0.02   # Volatility matches
    assert np.sum(prices < 50) > 0              # Crashes occur
```

#### Scenario 6: Weather System
**Goal**: Realistic weather patterns with seasonal variation
**Metrics**: State transition probabilities, seasonal drift, memory tests
**Implementation**:
```python
def weather_simulation():
    # Markov chain: sunny/cloudy/rainy with seasonal shifts
    transitions = {
        "sunny": {"sunny": 0.8, "cloudy": 0.15, "rainy": 0.05},
        "cloudy": {"sunny": 0.3, "cloudy": 0.5, "rainy": 0.2},
        "rainy": {"sunny": 0.1, "cloudy": 0.5, "rainy": 0.4}
    }

    # Simulate year
    state = "sunny"
    weather_log = []
    for day in range(365):
        # Seasonal shift (rain more likely in summer)
        season_factor = np.sin(day * 2 * np.pi / 365)
        transitions["rainy"]["rainy"] = 0.4 + 0.1 * season_factor

        # Next state
        state = np.random.choice(
            list(transitions[state].keys()),
            p=list(transitions[state].values())
        )
        weather_log.append(state)

    # Verify: transitional probabilities match theory
    transition_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(weather_log) - 1):
        transition_counts[weather_log[i]][weather_log[i+1]] += 1

    # Check against expected
    for from_state in transitions:
        total = sum(transition_counts[from_state].values())
        for to_state, expected_prob in transitions[from_state].items():
            observed = transition_counts[from_state][to_state] / total
            assert abs(observed - expected_prob) < 0.05, \
                f"Transition {from_state}→{to_state} mismatch"
```


## Advanced Topics

### Statistical Properties of Game Distributions

**Checking Normality: Q-Q Plot Test**

When implementing systems that assume normally distributed randomness, verify the assumption:

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def verify_normal_distribution(data, name="Distribution"):
    """Verify data follows normal distribution"""
    # Q-Q plot: compare to theoretical normal
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Q-Q plot
    stats.probplot(data, dist="norm", plot=axes[0])
    axes[0].set_title(f"{name}: Q-Q Plot (should be linear)")

    # Histogram with normal curve overlay
    axes[1].hist(data, bins=50, density=True, alpha=0.7, label='Data')
    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    axes[1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', label='Normal')
    axes[1].set_title(f"{name}: Histogram vs Normal")
    axes[1].legend()

    # Kolmogorov-Smirnov test
    ks_stat, p_value = stats.kstest(data, 'norm', args=(mu, sigma))
    print(f"{name}: KS test p-value = {p_value:.4f}")
    print(f"  Normal distribution? {'YES' if p_value > 0.05 else 'NO (deviation detected)'}")

    return p_value > 0.05

# Test: ability damage should be normal
damage_samples = np.random.normal(50, 10, 10000)
verify_normal_distribution(damage_samples, "Damage Distribution")
```

**Detecting Bias: Permutation Tests**

Verify randomness isn't biased by player segment:

```python
def permutation_test_fairness(group1, group2, iterations=10000):
    """
    Test if two groups have significantly different outcomes.
    Null hypothesis: no difference in distribution.
    """
    # Observed difference in means
    observed_diff = np.mean(group1) - np.mean(group2)

    # Combine groups
    combined = np.concatenate([group1, group2])

    # Permute and recalculate differences
    permuted_diffs = []
    for _ in range(iterations):
        np.random.shuffle(combined)
        perm_group1 = combined[:len(group1)]
        perm_group2 = combined[len(group1):]
        permuted_diffs.append(np.mean(perm_group1) - np.mean(perm_group2))

    # P-value: how often does permuted difference exceed observed?
    p_value = np.sum(np.abs(permuted_diffs) >= np.abs(observed_diff)) / iterations

    print(f"Observed difference: {observed_diff:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Fair? {'YES' if p_value > 0.05 else 'NO'}")

    return p_value > 0.05

# Example: whale vs free-to-play players
whale_loot = np.random.normal(100, 15, 1000)  # Should be same distribution
f2p_loot = np.random.normal(100, 15, 1000)
permutation_test_fairness(whale_loot, f2p_loot)
```


### Autocorrelation and Memory

**Problem**: Are consecutive outcomes independent or correlated?

```python
def check_autocorrelation(data, max_lag=20):
    """
    Check if sequence has memory (autocorrelation).
    Independent data should have near-zero correlation at all lags.
    """
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / len(data)

    autocorr = []
    for lag in range(1, max_lag + 1):
        c = np.sum((data[:-lag] - mean) * (data[lag:] - mean)) / len(data)
        autocorr.append(c / c0)

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.stem(range(1, max_lag + 1), autocorr, basefmt=' ')
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.axhline(1.96 / np.sqrt(len(data)), color='red', linestyle='--', label='95% CI')
    plt.axhline(-1.96 / np.sqrt(len(data)), color='red', linestyle='--')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation: Check for Memory in Sequence')
    plt.legend()

    # Diagnosis
    max_corr = max(np.abs(autocorr))
    if max_corr < 0.1:
        print(f"Independent: autocorr max = {max_corr:.3f}")
    else:
        print(f"CORRELATED: autocorr max = {max_corr:.3f} - sequence has memory!")

    return autocorr

# Test: pure randomness vs filtered randomness
independent = np.random.normal(0, 1, 1000)
filtered = np.convolve(independent, [0.3, 0.7], mode='same')  # Creates correlation

print("Independent sequence:")
check_autocorrelation(independent[:100])

print("\nFiltered sequence (correlated):")
check_autocorrelation(filtered[:100])
```


### Rare Events and Tail Risk

**Extreme Value Theory: Modeling Black Swan Events**

```python
def model_rare_events(base_rate=0.01, max_samples=100000):
    """
    Model rare catastrophic events using extreme value theory.
    E.g., server crash probability, critical failure rate.
    """
    # Generate events: mostly base_rate, occasionally extreme
    events = []
    for _ in range(max_samples):
        if np.random.random() < base_rate:
            # Normal event
            severity = np.random.exponential(1)
        else:
            # Rare catastrophic event (Pareto tail)
            severity = np.random.pareto(2) * 10

        events.append(severity)

    # Analyze tail
    events_sorted = np.sort(events)
    tail_threshold = np.percentile(events, 99)

    print(f"Base rate events: {base_rate}")
    print(f"P99 severity: {tail_threshold:.2f}")
    print(f"P99.9 severity: {np.percentile(events, 99.9):.2f}")

    # Extrapolate: what's the p99.99 severity?
    tail_data = np.sort(events[events > tail_threshold])
    k = 2  # Shape parameter (Pareto)
    p99_99_estimate = tail_threshold * (0.01 / 0.0001) ** (1/k)

    print(f"P99.99 estimated: {p99_99_estimate:.2f}")
    print(f"  (1 in 10,000 events this severe)")

    return events

catastrophes = model_rare_events(base_rate=0.02)
```


### Multi-Agent Stochastic Systems

**Emergent Behavior from Individual Randomness**

```python
class StochasticAgent:
    """Agent with random decisions that create emergent behavior"""

    def __init__(self, agent_id, world):
        self.id = agent_id
        self.world = world
        self.wealth = 100
        self.position = np.random.uniform(0, 100)
        self.strategy = np.random.choice(['aggressive', 'conservative', 'random'])

    def step(self):
        """One time step"""
        # Random market event
        market_return = np.random.normal(0.01, 0.02)

        if self.strategy == 'aggressive':
            # Leverage wealth
            trade_size = self.wealth * 1.5
            self.wealth *= (1 + market_return * 1.5)
        elif self.strategy == 'conservative':
            # Risk-averse
            trade_size = self.wealth * 0.5
            self.wealth *= (1 + market_return * 0.3)
        else:
            # Random strategy
            trade_size = self.wealth * np.random.uniform(0, 1)
            multiplier = np.random.choice([0.5, 1.0, 1.5])
            self.wealth *= (1 + market_return * multiplier)

        # Bankruptcy check
        if self.wealth < 0:
            self.wealth = 0
            return False  # Agent bankrupt

        # Random move
        self.position += np.random.normal(0, 5)
        self.position = np.clip(self.position, 0, 100)

        return True

class MarketWorld:
    def __init__(self, n_agents=100):
        self.agents = [StochasticAgent(i, self) for i in range(n_agents)]
        self.history = []

    def step(self):
        """One world step: all agents act"""
        alive = 0
        total_wealth = 0

        for agent in self.agents:
            if agent.step():
                alive += 1
                total_wealth += agent.wealth

        stats = {
            'time': len(self.history),
            'alive': alive,
            'total_wealth': total_wealth,
            'avg_wealth': total_wealth / alive if alive > 0 else 0,
            'strategies': {
                'aggressive': sum(1 for a in self.agents if a.strategy == 'aggressive' and a.wealth > 0),
                'conservative': sum(1 for a in self.agents if a.strategy == 'conservative' and a.wealth > 0),
                'random': sum(1 for a in self.agents if a.strategy == 'random' and a.wealth > 0)
            }
        }
        self.history.append(stats)

    def simulate(self, steps=1000):
        """Run simulation"""
        for _ in range(steps):
            self.step()

        # Analyze emergence
        wealth_over_time = [h['total_wealth'] for h in self.history]
        print(f"Starting wealth: {wealth_over_time[0]:.0f}")
        print(f"Final wealth: {wealth_over_time[-1]:.0f}")
        print(f"Agents alive: {self.history[-1]['alive']}")
        print(f"Strategy distribution: {self.history[-1]['strategies']}")

        return self.history

# Run simulation
market = MarketWorld(n_agents=100)
history = market.simulate(steps=500)
```


### Sampling Techniques for Efficiency

**Importance Sampling: Focus on Rare Events**

```python
def estimate_rare_event_probability_naive(target_prob=0.001, samples=100000):
    """Naive: sample until we see rare events"""
    successes = 0
    for _ in range(samples):
        if np.random.random() < target_prob:
            successes += 1

    estimate = successes / samples
    # Problem: might see 0 successes, estimate = 0!
    return estimate, successes

def estimate_rare_event_probability_importance(target_prob=0.001, samples=100000):
    """
    Importance Sampling: sample from easier distribution,
    weight by likelihood ratio.
    """
    # Sample from easier distribution (10x higher probability)
    easy_prob = target_prob * 10

    estimates = []
    for _ in range(samples):
        if np.random.random() < easy_prob:
            # Likelihood ratio: we're 10x more likely to see this
            # Weight down by 10
            weight = target_prob / easy_prob
            estimates.append(weight)
        else:
            estimates.append(0)

    estimate = np.mean(estimates)
    return estimate, sum(1 for e in estimates if e > 0)

# Compare efficiency
naive_est, naive_hits = estimate_rare_event_probability_naive(samples=100000)
importance_est, importance_hits = estimate_rare_event_probability_importance(samples=100000)

print(f"Naive: {naive_est:.6f} ({naive_hits} hits)")
print(f"Importance: {importance_est:.6f} ({importance_hits} hits)")
print(f"True: 0.001000")
print(f"Importance sampling sees rare events 10x more often with better estimate!")
```


## Production Implementation Guide

### Deploying Stochastic Systems Safely

**Phase 1: Offline Testing (Before Beta)**

```python
def comprehensive_randomness_audit(system_name, rng_function, expected_rate=None):
    """
    Complete validation of randomness before deployment.
    Prevents bugs from reaching players.
    """
    samples = 1000000  # 1M samples for precision
    results = [rng_function() for _ in range(samples)]

    # Test 1: Frequency analysis
    if expected_rate:
        observed_rate = sum(1 for r in results if r) / len(results)
        from scipy.stats import binom_test
        p_val = binom_test(sum(results), len(results), expected_rate)
        assert p_val > 0.05, f"Distribution significantly different: p={p_val}"
        print(f"{system_name}: Rate {observed_rate:.6f} == {expected_rate:.6f} ✓")

    # Test 2: No obvious patterns
    from collections import deque
    window = deque(maxlen=100)
    max_consecutive = 0
    current_consecutive = 0
    for r in results[:1000]:  # Check first 1000
        if r == window[-1] if window else False:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
        window.append(r)

    # Test 3: Distribution across player segments
    segments = {
        'low_luck': results[0:len(results)//4],
        'mid_luck': results[len(results)//4:len(results)//2],
        'high_luck': results[len(results)//2:3*len(results)//4],
        'whale': results[3*len(results)//4:]
    }

    segment_rates = {
        seg: (sum(1 for r in data if r) / len(data))
        for seg, data in segments.items()
    }

    # All segments should be similar
    rate_variance = max(segment_rates.values()) - min(segment_rates.values())
    assert rate_variance < 0.002, f"Segment bias detected: variance={rate_variance}"
    print(f"{system_name}: Fair across segments ✓")

    # Test 4: No RNG state leaks
    rng1 = [rng_function() for _ in range(100)]
    rng2 = [rng_function() for _ in range(100)]
    # These should be independent
    correlation = sum(r1 == r2 for r1, r2 in zip(rng1, rng2)) / 100
    assert correlation < 0.6, f"RNG state leak detected: correlation={correlation}"

    print(f"{system_name}: SAFE FOR PRODUCTION ✓")

# Run audit before deploying
# comprehensive_randomness_audit("Loot System", loot_function, expected_rate=0.03)
```

**Phase 2: Gradual Rollout**

```python
def gradual_feature_rollout(feature_name, percentage=1.0):
    """
    Roll out random features gradually to detect issues.
    1% -> 5% -> 25% -> 100%
    """
    # Use player ID hash to determine eligibility
    def is_eligible(player_id):
        # Hash to 0-100
        player_hash = hash(player_id) % 100
        return player_hash < percentage

    return is_eligible

# Example: roll out variance-reduced crit to 1% of players
if gradual_feature_rollout("reduced_crit_variance", percentage=1.0)(player.id):
    crit = use_variance_reduced_crit(player)
else:
    crit = use_standard_crit(player)

# Monitor metrics:
# - Mean crit rate (should match)
# - Variance (should be lower)
# - Player satisfaction surveys
# - Bug reports related to crits
```

**Phase 3: Monitoring Production**

```python
class StochasticSystemMonitor:
    """Track randomness in production to catch drift"""

    def __init__(self, system_name, expected_distribution):
        self.system_name = system_name
        self.expected = expected_distribution
        self.observations = []
        self.last_check = 0
        self.check_interval = 10000  # Check every 10K samples

    def record(self, outcome):
        """Record one outcome"""
        self.observations.append(outcome)

        # Periodic check
        if len(self.observations) % self.check_interval == 0:
            self.check_distribution()

    def check_distribution(self):
        """Verify distribution hasn't drifted"""
        recent = self.observations[-self.check_interval:]

        # Chi-square goodness of fit
        from scipy.stats import chisquare
        observed_counts = np.bincount(recent)
        expected_counts = [
            len(recent) * self.expected.get(i, 0)
            for i in range(len(observed_counts))
        ]

        chi2, p_val = chisquare(observed_counts, expected_counts)

        if p_val < 0.01:
            print(f"ALERT: {self.system_name} distribution drift!")
            print(f"  χ² = {chi2:.2f}, p = {p_val:.4f}")
            print(f"  Observed: {dict(enumerate(observed_counts))}")
            print(f"  Expected: {expected_counts}")
            # TRIGGER INCIDENT: notify ops, disable feature, investigate
            return False

        return True

# In production
crit_monitor = StochasticSystemMonitor("CritSystem", {0: 0.8, 1: 0.2})

for combat_log in incoming_combats:
    crit = apply_crit()
    crit_monitor.record(int(crit))
```


## Summary

Stochastic simulation transforms game randomness from exploitable noise into fair, predictable distributions. Master these concepts and you build systems players trust.

**Key Takeaways**:
1. Every random system has a distribution—measure it
2. Variance reduction (pity systems) feels better than pure randomness
3. State-dependent randomness (Markov) creates believable behavior
4. Always verify your system matches theory with Monte Carlo testing
5. Common distributions solve common problems—use them
6. Deploy gradually, monitor continuously, act on anomalies
7. Test before beta, roll out 1%-5%-25%-100%, watch metrics

**Never Ship Without**:
- 1M sample offline validation
- Distribution checks across player segments
- Gradual rollout monitoring
- Production alerting for statistical drift
- Player satisfaction feedback loop

**Next Steps**:
- Implement a fair loot system with pity mechanics
- Build variance-reduced crit system and A/B test feel
- Create procedural dungeons with Perlin noise
- Test all randomness with statistical rigor before shipping
- Set up monitoring for production systems
- Create incident response plan for distribution drift

