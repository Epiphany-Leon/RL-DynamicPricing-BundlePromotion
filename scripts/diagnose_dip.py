"""
Stand-alone training-curve dip diagnostic.

Replays the exact same setup as run_experiments.py (seed 42, env, agent,
hyperparameters) without importing the module, so the random state isn't
disturbed. Reproduces the original training curve and isolates what changes
inside each dip.
"""
import os, json
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(42); random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figs")


# ---------------- copy of the env/agent (verbatim) -------------------
class PricingBundleEnv:
    def __init__(self, horizon=120):
        self.horizon = horizon
        self.base_price = 100.0
        self.min_price = 0.7 * self.base_price
        self.max_price = 1.3 * self.base_price
        self.price_adjustments = np.array([-0.10, -0.05, 0.0, 0.05, 0.10])
        self.bundle_options = [0, 1]
        self.discount_options = [0.0, 0.10]
        self.discipline_options = [0, 1]
        self.action_map = []
        for pa in self.price_adjustments:
            for b in self.bundle_options:
                for d in self.discount_options:
                    for pd in self.discipline_options:
                        self.action_map.append((pa, b, d, pd))
        self.unit_cost = 55.0
        self.holding_cost = 1.2
        self.stockout_penalty = 8.0
        self.bundle_margin_penalty = 2.0
        self.base_demand = 20.0
        self.reference_profit = self.base_demand * (self.base_price - self.unit_cost)
        self.price_change_cost = 0.05
        self._tier_thresholds = [1.5, 1.0, 0.5, 0.0, -9999]
        self._tier_bonuses = [0.15, 0.08, 0.02, -0.03, -0.12]
        self.price_sensitivity = 0.20
        self.bundle_uplift = 0.18
        self.discount_uplift = 0.12
        self.noise_std = 2.5
        self.max_inventory = 120
        self.restock_level = 80

    def n_actions(self): return len(self.action_map)
    def _season_multiplier(self, t): return 1.15 if (t % 7) in [5, 6] else 1.0
    def _bucket_inventory(self, inv): return 0 if inv < 20 else (1 if inv < 60 else 2)
    def _bucket_demand_trend(self, dm): return 0 if dm < 15 else (1 if dm < 25 else 2)
    def _bucket_season(self, t): return 1 if (t % 7) in [5, 6] else 0

    def _get_state(self):
        inv_b = self._bucket_inventory(self.inventory)
        trend_b = self._bucket_demand_trend(np.mean(self.last_demands))
        season_b = self._bucket_season(self.t)
        locked = int(self.lock_remaining > 0)
        return (inv_b, trend_b, season_b, int(self.current_bundle_flag), locked)

    def _tier_bonus(self, r_norm):
        for th, bn in zip(self._tier_thresholds, self._tier_bonuses):
            if r_norm > th: return bn
        return self._tier_bonuses[-1]

    def _apply_price_change(self, pa):
        new_price = self.current_price * (1.0 + pa)
        self.current_price = float(np.clip(new_price, self.min_price, self.max_price))

    def reset(self):
        self.t = 0; self.inventory = 80; self.last_demands = [20, 20, 20]
        self.current_bundle_flag = 0; self.current_price = self.base_price
        self.hysteresis_counter = 0; self.hysteresis_direction = 0
        self.lock_remaining = 0
        return self._get_state()

    def step(self, action_idx):
        pa, bundle, discount, price_discipline = self.action_map[action_idx]
        self.current_bundle_flag = bundle
        was_locked = (self.lock_remaining > 0)
        price_actually_changed = False
        if was_locked:
            self.lock_remaining -= 1
        elif price_discipline == 1:
            if pa != 0.0:
                self._apply_price_change(pa)
                self.lock_remaining = 7
                self.hysteresis_counter = 0; self.hysteresis_direction = 0
                price_actually_changed = True
        else:
            if pa == 0.0:
                self.hysteresis_counter = 0; self.hysteresis_direction = 0
            else:
                direction = 1 if pa > 0 else -1
                if direction == self.hysteresis_direction:
                    self.hysteresis_counter += 1
                else:
                    self.hysteresis_counter = 1
                    self.hysteresis_direction = direction
                if self.hysteresis_counter >= 3:
                    self._apply_price_change(pa)
                    price_actually_changed = True
                    self.hysteresis_counter = 0; self.hysteresis_direction = 0
        listed_price = self.current_price
        effective_price = listed_price * (1.0 - discount)
        season_m = self._season_multiplier(self.t)
        demand_mean = self.base_demand * season_m
        demand_mean *= (1.0 + self.bundle_uplift * bundle
                        + self.discount_uplift * float(discount > 0))
        demand_mean *= np.exp(-self.price_sensitivity
                              * (effective_price - self.base_price) / self.base_price)
        demand = max(0, int(np.random.normal(demand_mean, self.noise_std)))
        sales = min(self.inventory, demand)
        stockout = max(0, demand - self.inventory)
        revenue = sales * effective_price
        procurement = sales * self.unit_cost
        hold_cost = self.holding_cost * max(0, self.inventory - sales)
        stockout_cost = self.stockout_penalty * stockout
        bundle_penalty = self.bundle_margin_penalty * sales * bundle
        raw_profit = revenue - procurement - hold_cost - stockout_cost - bundle_penalty
        r_norm = raw_profit / self.reference_profit
        tier_b = self._tier_bonus(r_norm)
        chg_cost = self.price_change_cost if price_actually_changed else 0.0
        reward = r_norm + tier_b - chg_cost
        self.inventory -= sales
        if self.inventory < self.restock_level:
            self.inventory = min(self.max_inventory, self.inventory + 25)
        self.last_demands.pop(0); self.last_demands.append(demand)
        self.t += 1
        done = self.t >= self.horizon
        info = {'demand':demand,'sales':sales,'stockout':stockout,'revenue':revenue,
                'listed_price':listed_price,'effective_price':effective_price,
                'current_price':self.current_price,'bundle':bundle,'discount':discount,
                'inventory':self.inventory,'raw_profit':raw_profit,
                'price_changed':price_actually_changed,'locked_state':was_locked,
                'lock_remaining':self.lock_remaining}
        return self._get_state(), reward, done, info


class QLearningAgent:
    def __init__(self, n_actions, alpha=0.08, gamma=0.95,
                 eps=1.0, eps_min=0.05, eps_decay=0.995):
        self.n_actions = n_actions
        self.alpha = alpha; self.gamma = gamma
        self.eps = eps; self.eps_min = eps_min; self.eps_decay = eps_decay
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))
    def act(self, state, greedy=False):
        if (not greedy) and (np.random.rand() < self.eps):
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))
    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])
    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# ---------------- training with rich logging --------------------------
EPISODES = 1500
PROBE_STATES = [
    (1, 1, 0, 0, 0), (1, 1, 1, 0, 0), (1, 2, 1, 0, 0),
    (2, 0, 0, 0, 0), (2, 1, 1, 0, 0), (1, 0, 0, 0, 0),
    (0, 1, 0, 0, 0), (2, 2, 0, 0, 0),
]

env = PricingBundleEnv(horizon=120)
agent = QLearningAgent(n_actions=env.n_actions())

ep_reward, ep_profit, ep_lock_choice, ep_locked_state = [], [], [], []
ep_price_chg, ep_bundle, ep_discount, ep_eps = [], [], [], []
ep_pos_chg, ep_neg_chg, ep_random_acts = [], [], []
ep_states_seen = []
probe_actions = {s: [] for s in PROBE_STATES}

for ep in range(EPISODES):
    s = env.reset(); done = False
    tot_r = 0.0; tot_p = 0.0
    lock_choice = lock_state = chg = bnd = dsc = 0
    pos_chg = neg_chg = random_acts = 0
    while not done:
        # Mirror the agent.act logic but track random vs greedy
        if np.random.rand() < agent.eps:
            a = np.random.randint(agent.n_actions)
            random_acts += 1
        else:
            a = int(np.argmax(agent.Q[s]))
        s_next, r, done, info = env.step(a)
        agent.update(s, a, r, s_next, done)
        s = s_next
        tot_r += r; tot_p += info['raw_profit']
        action_tuple = env.action_map[a]
        lock_choice += int(action_tuple[3] == 1)
        lock_state += int(info['locked_state'])
        chg += int(info['price_changed'])
        bnd += int(info['bundle'] == 1)
        dsc += int(info['discount'] > 0)
        if info['price_changed']:
            if action_tuple[0] > 0: pos_chg += 1
            else: neg_chg += 1
    ep_reward.append(tot_r)
    ep_profit.append(tot_p / env.horizon)
    ep_lock_choice.append(lock_choice / env.horizon)
    ep_locked_state.append(lock_state / env.horizon)
    ep_price_chg.append(chg / env.horizon)
    ep_bundle.append(bnd / env.horizon)
    ep_discount.append(dsc / env.horizon)
    ep_eps.append(agent.eps)
    ep_pos_chg.append(pos_chg)
    ep_neg_chg.append(neg_chg)
    ep_random_acts.append(random_acts)
    ep_states_seen.append(len(agent.Q))
    for st in PROBE_STATES:
        if st in agent.Q:
            probe_actions[st].append(int(np.argmax(agent.Q[st])))
        else:
            probe_actions[st].append(-1)
    agent.decay_eps()


# ---------------- find dips on smoothed reward -----------------------
W = 20
def smooth(x, w=W):
    return np.convolve(np.asarray(x, dtype=float), np.ones(w)/w, mode='valid')

sm_r = smooth(ep_reward)
print(f"Smoothed reward: min={sm_r.min():.1f}  max={sm_r.max():.1f}  mean={sm_r.mean():.1f}")
plateau = np.median(sm_r[200:1500-W])
print(f"Median (after warmup): {plateau:.1f}")

threshold = plateau * 0.85   # call anything ≤ 85% of plateau a dip
print(f"Dip threshold (85% of plateau): {threshold:.1f}\n")

dips = []
in_dip = False; start = None
for i, v in enumerate(sm_r):
    if v < threshold and not in_dip:
        in_dip = True; start = i
    elif v >= threshold and in_dip:
        in_dip = False
        if i - start >= 5:   # at least 5 episodes
            dips.append((start, i, sm_r[start:i].min(), int(np.argmin(sm_r[start:i])) + start))
if in_dip and EPISODES - W - start >= 5:
    dips.append((start, len(sm_r), sm_r[start:].min(), int(np.argmin(sm_r[start:])) + start))

print("Detected dips (smoothed reward < 85% plateau):")
for d in dips:
    print(f"  episodes {d[0]:>4} - {d[1]:<4}   min reward {d[2]:>6.1f} at ep {d[3]}")


# ---------------- per-dip table --------------------------------------
def stats_block(metric_arr, sl):
    return float(np.mean(metric_arr[sl]))

for k, d in enumerate(dips, 1):
    s, e = d[0], d[1]
    pre  = slice(max(0, s-50), s)
    cur  = slice(s, e)
    post = slice(e, min(e+50, EPISODES))
    print(f"\n=== Dip #{k}: episodes {s}-{e} (depth {d[2]:.0f}) ===")
    print(f"  {'metric':<25} {'pre':>10} {'during':>10} {'post':>10}")
    metrics = [
        ('avg profit',       ep_profit),
        ('lock-choice',      ep_lock_choice),
        ('locked-state',     ep_locked_state),
        ('price-chg ratio',  ep_price_chg),
        ('bundle ratio',     ep_bundle),
        ('discount ratio',   ep_discount),
        ('pos chg /ep',      ep_pos_chg),
        ('neg chg /ep',      ep_neg_chg),
        ('random acts /ep',  ep_random_acts),
        ('avg ε',            ep_eps),
        ('states_seen end',  ep_states_seen),
    ]
    for name, arr in metrics:
        a = np.asarray(arr, dtype=float)
        fmt = "{:>10.2f}"
        if name in ('avg profit',):
            fmt = "{:>10.0f}"
        if name in ('pos chg /ep', 'neg chg /ep', 'random acts /ep', 'states_seen end'):
            fmt = "{:>10.1f}"
        print(f"  {name:<25} " +
              fmt.format(np.mean(a[pre])) +
              fmt.format(np.mean(a[cur])) +
              fmt.format(np.mean(a[post])))


# ---------------- probe-state action evolution -----------------------
print("\nGreedy action churn at probe states (only transitions inside any dip ±50):")
for st in PROBE_STATES:
    seq = probe_actions[st]
    transitions = []
    last_a = None
    for i, a in enumerate(seq):
        if a != last_a and a != -1:
            transitions.append((i, a))
            last_a = a
    relevant = []
    for d in dips:
        lo, hi = max(0, d[0]-50), min(EPISODES, d[1]+50)
        relevant.extend([t for t in transitions if lo <= t[0] <= hi])
    if relevant:
        print(f"\n  State {st}:")
        for ep_idx, a in relevant[:15]:
            pa, b, d_, pd = env.action_map[a]
            disc = 'lock' if pd == 1 else 'hyst'
            print(f"    ep {ep_idx:>4}  ->  Δp={pa:+.0%}  b={b}  d={d_:.0%}  disc={disc}")


# ---------------- diagnostic figure ----------------------------------
xs = np.arange(len(sm_r))
x = np.arange(EPISODES)
fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

axes[0].plot(xs, sm_r, color='steelblue', linewidth=1.4, label='smoothed reward')
axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8, label='break-even')
for d in dips:
    axes[0].axvspan(d[0], d[1], color='#C0392B', alpha=0.15)
axes[0].set_ylabel('Cumulative\nNormalised Reward')
axes[0].set_title('Diagnostic: anatomy of the training-curve dips')
axes[0].grid(alpha=0.3); axes[0].legend(loc='lower right', fontsize=9)

axes[1].plot(xs, smooth(ep_lock_choice), color='#4F7A4F', linewidth=1.4, label='lock-choice (action level)')
axes[1].plot(xs, smooth(ep_locked_state), color='#142942', linewidth=1.4, label='locked-state (env level)')
axes[1].plot(xs, smooth(ep_price_chg), color='#C0392B', linewidth=1.4, label='actual price-change')
for d in dips: axes[1].axvspan(d[0], d[1], color='#C0392B', alpha=0.15)
axes[1].set_ylabel('Fraction of steps'); axes[1].grid(alpha=0.3)
axes[1].legend(loc='lower right', fontsize=9)

axes[2].plot(xs, smooth(ep_bundle), color='#E76F51', linewidth=1.4, label='bundle ratio')
axes[2].plot(xs, smooth(ep_discount), color='#F4A261', linewidth=1.4, label='discount ratio')
axes[2].plot(xs, smooth([n/120 for n in ep_random_acts]), color='#264653', linewidth=1.0,
             linestyle='--', label='random-action ratio (= ε)')
for d in dips: axes[2].axvspan(d[0], d[1], color='#C0392B', alpha=0.15)
axes[2].set_ylabel('Action ratios'); axes[2].grid(alpha=0.3)
axes[2].legend(loc='upper right', fontsize=9)

axes[3].plot(x, ep_pos_chg, color='steelblue', alpha=0.4, linewidth=0.6, label='upward chg / ep')
axes[3].plot(x, ep_neg_chg, color='#C0392B',   alpha=0.4, linewidth=0.6, label='downward chg / ep')
axes[3].plot(xs, smooth(ep_pos_chg), color='steelblue', linewidth=1.4)
axes[3].plot(xs, smooth(ep_neg_chg), color='#C0392B',   linewidth=1.4)
for d in dips: axes[3].axvspan(d[0], d[1], color='#C0392B', alpha=0.15)
axes[3].set_ylabel('Real price changes\nper episode')
axes[3].set_xlabel('Episode')
axes[3].grid(alpha=0.3); axes[3].legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_dip_diagnostic.png'), dpi=150)
plt.close()
print("\nSaved fig_dip_diagnostic.png")
