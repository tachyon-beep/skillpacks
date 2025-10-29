# RED Phase: RL Foundations Baseline Failures

## Documented Baseline Failures

### Failure 1: Jumping to Algorithms Without Understanding MDP

**Scenario**: User asks to implement Q-learning without understanding MDPs

**Expected Behavior**:
- Recognize that implementing algorithms without foundational understanding leads to brittle, buggy code
- Users copy-paste Q-learning code but cannot debug when it fails
- Cannot adapt algorithms to new problems

**Baseline Response (Without Skill)**:
```
User: "I want to implement Q-learning for my game AI"

Agent: "Sure! Here's Q-learning code:

```python
import numpy as np

class QLearning:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((num_states, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state, best_next_action]
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
```

Just call `choose_action` and `update` in your game loop!"
```

**Why This Fails**:
1. User has no idea what Q[state, action] represents (expected cumulative reward)
2. Doesn't understand why gamma * Q[next_state, best_next_action] appears (Bellman equation)
3. Cannot debug when Q-values don't converge (don't know convergence conditions)
4. When reward scale changes, doesn't know why algorithm breaks (don't understand discount factor)
5. Copies code without understanding MDP framework (states, actions, rewards, transitions)

**Evidence of Failure**:
- User returns: "Why are all my Q-values zero?"
- User: "What should alpha, gamma, epsilon be?"
- User: "It works on GridWorld but not my game. Why?"
- Cannot answer: "What's the difference between Q(s,a) and V(s)?"

---

### Failure 2: Confusing Value Function with Reward

**Scenario**: User thinks V(s) is the immediate reward at state s

**Expected Behavior**:
- Understand that V(s) = expected CUMULATIVE reward (sum of discounted future rewards)
- Reward r(s,a) is immediate, value V(s) is long-term
- This distinction is critical for policy evaluation

**Baseline Response (Without Skill)**:
```
User: "My value function equals the reward. Is that correct?"

Agent: "Not quite. The value function should represent future rewards, not just immediate reward."

User: "So V(s) = sum of future rewards?"

Agent: "Yes, the discounted sum of expected future rewards."

User: "OK, so V(s) = r(s) + r(s+1) + r(s+2) + ...?"

Agent: "Sort of, but you need to discount: V(s) = r(s) + gamma*r(s+1) + gamma^2*r(s+2) + ..."
```

**Why This Fails**:
1. User still thinks it's a simple sum (doesn't understand expectation over stochastic transitions)
2. No understanding of Bellman equation: V(s) = E[r + gamma*V(s')]
3. Cannot implement value iteration (requires understanding Bellman backup)
4. Confusion between V(s), Q(s,a), and r(s,a) persists
5. When implementing, sums actual trajectory rewards instead of expected value

**Evidence of Failure**:
- User implements: `V[s] = sum(rewards)` instead of `V[s] = reward + gamma * V[next_state]`
- User: "Why does my value function change based on the trajectory?"
- Cannot explain: "Why do we need the Bellman equation if V is just sum of rewards?"

---

### Failure 3: Not Understanding Discount Factor Importance

**Scenario**: User sets gamma=0.9 arbitrarily without understanding consequences

**Expected Behavior**:
- Understand gamma controls how much future matters (0 = only immediate, 1 = all future equally)
- gamma=0.9 means reward 10 steps away worth 0.9^10 = 0.35 of immediate reward
- gamma=0.99 means reward 10 steps away worth 0.99^10 = 0.90 of immediate reward
- Choice of gamma critically affects learned policy

**Baseline Response (Without Skill)**:
```
User: "My agent is too short-sighted. It only cares about immediate rewards."

Agent: "Try increasing gamma. Gamma=0.99 makes it consider future more."

User: "Why 0.99? Can I use 0.95?"

Agent: "0.99 is common. You can try 0.95, it's less far-sighted."

User: "OK, I changed to 0.95 and it still doesn't work."

Agent: "Try 0.99 or 0.999."
```

**Why This Fails**:
1. No mathematical understanding of gamma's effect (horizon = 1/(1-gamma))
2. Cannot calculate effective planning horizon (gamma=0.9 → ~10 steps, gamma=0.99 → ~100 steps)
3. Treats gamma as magic hyperparameter instead of problem-specific choice
4. When task has long-term dependencies (reward after 50 steps), doesn't realize gamma=0.9 makes it impossible
5. No understanding of gamma=1 issues (infinite value in continuing tasks)

**Evidence of Failure**:
- User: "What's a good gamma for games?"
- Cannot answer: "If my goal is 100 steps away, what gamma do I need?"
- Trial-and-error tuning instead of principled choice
- Doesn't understand why gamma=0.5 makes agent myopic

---

### Failure 4: Mixing Up Value Iteration and Policy Iteration

**Scenario**: User confuses value iteration (iteratively update values) with policy iteration (evaluate policy, improve policy)

**Expected Behavior**:
- Value iteration: repeatedly apply Bellman optimality backup until convergence
- Policy iteration: alternate between policy evaluation (compute V^π) and policy improvement
- Both converge to optimal policy, different computational tradeoffs
- Policy iteration converges in fewer iterations, but each iteration more expensive

**Baseline Response (Without Skill)**:
```
User: "Should I use value iteration or policy iteration?"

Agent: "Both work. Value iteration is simpler, policy iteration is faster."

User: "I'll use value iteration. How do I implement it?"

Agent: "Loop until values converge:
```python
for state in states:
    V_new[state] = max over actions of (reward + gamma * sum(P(s'|s,a) * V[s']))
V = V_new
```"

User: "And how does policy iteration differ?"

Agent: "Policy iteration evaluates a policy, then improves it. Repeat until policy stops changing."
```

**Why This Fails**:
1. User doesn't understand algorithmic difference (value iteration extracts policy at end, policy iteration maintains explicit policy)
2. Cannot explain why policy iteration often converges faster (policy often converges before values)
3. No understanding of when to use each (value iteration for small spaces, policy iteration for good initial policy)
4. Confuses policy improvement (making policy greedy w.r.t. current V) with policy evaluation (computing V for fixed policy)
5. Doesn't understand both require full MDP model (transition probabilities)

**Evidence of Failure**:
- User tries policy iteration but only updates policy once (doesn't iterate)
- Confuses value iteration with Q-learning (doesn't understand model-free vs model-based)
- Cannot answer: "When is policy iteration better than value iteration?"
- Implements hybrid that's neither algorithm

---

## Common Thread

All failures stem from **implementing algorithms without understanding the theory**:
- MDP framework (states, actions, rewards, transitions)
- Value functions (cumulative expected return)
- Bellman equations (recursive decomposition)
- Role of discount factor (planning horizon)
- Algorithmic families (DP, MC, TD)

**Users copy-paste code that works on toy examples but fails on real problems because they cannot:**
1. Debug when convergence fails (don't understand Bellman backup)
2. Choose hyperparameters (don't understand gamma, alpha, epsilon)
3. Adapt algorithms (don't understand assumptions: model-free vs model-based, on-policy vs off-policy)
4. Recognize when approach is fundamentally wrong (using DP without model, wrong action space)

## What GREEN Phase Must Provide

The skill must teach:
1. **MDP Formulation**: Complete framework (S, A, P, R, γ) with examples
2. **Value Functions**: V(s) vs Q(s,a) vs r(s,a), mathematical definitions
3. **Bellman Equations**: Derivation, intuition, consistency conditions
4. **Discount Factor**: Mathematical meaning, planning horizon, choice guidelines
5. **Algorithm Families**: DP (value/policy iteration), MC, TD - when each applies
6. **Exploration vs Exploitation**: Why needed, ε-greedy, UCB
7. **Theory-to-Practice**: When understanding theory is sufficient vs when to implement
8. **Pitfalls**: Common misconceptions with examples

**Success Criteria**: After reading skill, user can derive Bellman equation, explain discount factor choice, and understand algorithm assumptions without implementing code.
