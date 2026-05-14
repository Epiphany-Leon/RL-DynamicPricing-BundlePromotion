"""
Re-runs the core experiments from rl_dynamic_pricing_bundle.ipynb
and saves figures + a JSON of summary metrics for use in the report,
PPT slides, and the LaTeX RL-agent document.
"""
import json
import os
import sys
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(42)
random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

# ----------------------------------------------------------------------
#  Environment (verbatim from the notebook, only matplotlib display
#  removed and figures written to disk)
# ----------------------------------------------------------------------
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
        self.t = 0
        self.inventory = None
        self.last_demands = None
        self.current_bundle_flag = 0
        self.current_price = None
        self.hysteresis_counter = 0
        self.hysteresis_direction = 0
        self.lock_remaining = 0

    def n_actions(self):
        return len(self.action_map)

    def _season_multiplier(self, t):
        return 1.15 if (t % 7) in [5, 6] else 1.0

    def _bucket_inventory(self, inv):
        return 0 if inv < 20 else (1 if inv < 60 else 2)

    def _bucket_demand_trend(self, dm):
        return 0 if dm < 15 else (1 if dm < 25 else 2)

    def _bucket_season(self, t):
        return 1 if (t % 7) in [5, 6] else 0

    def _get_state(self):
        inv_b = self._bucket_inventory(self.inventory)
        trend_b = self._bucket_demand_trend(np.mean(self.last_demands))
        season_b = self._bucket_season(self.t)
        locked = int(self.lock_remaining > 0)
        return (inv_b, trend_b, season_b, int(self.current_bundle_flag), locked)

    def _tier_bonus(self, r_norm):
        for th, bn in zip(self._tier_thresholds, self._tier_bonuses):
            if r_norm > th:
                return bn
        return self._tier_bonuses[-1]

    def _apply_price_change(self, pa):
        new_price = self.current_price * (1.0 + pa)
        self.current_price = float(np.clip(new_price, self.min_price, self.max_price))

    def reset(self):
        self.t = 0
        self.inventory = 80
        self.last_demands = [20, 20, 20]
        self.current_bundle_flag = 0
        self.current_price = self.base_price
        self.hysteresis_counter = 0
        self.hysteresis_direction = 0
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
                self.hysteresis_counter = 0
                self.hysteresis_direction = 0
                price_actually_changed = True
        else:
            if pa == 0.0:
                self.hysteresis_counter = 0
                self.hysteresis_direction = 0
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
                    self.hysteresis_counter = 0
                    self.hysteresis_direction = 0
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
        self.inventory = self.inventory - sales
        if self.inventory < self.restock_level:
            self.inventory = min(self.max_inventory, self.inventory + 25)
        self.last_demands.pop(0)
        self.last_demands.append(demand)
        self.t += 1
        done = self.t >= self.horizon
        info = {
            'demand': demand, 'sales': sales, 'stockout': stockout,
            'revenue': revenue, 'listed_price': listed_price,
            'effective_price': effective_price, 'current_price': self.current_price,
            'bundle': bundle, 'discount': discount, 'inventory': self.inventory,
            'raw_profit': raw_profit, 'price_changed': price_actually_changed,
            'locked_state': was_locked, 'lock_remaining': self.lock_remaining,
        }
        return self._get_state(), reward, done, info


class QLearningAgent:
    def __init__(self, n_actions, alpha=0.08, gamma=0.95,
                 eps=1.0, eps_min=0.05, eps_decay=0.995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
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


def find_action_index(env, price_adj, bundle, discount, price_discipline=0):
    for i, (pa, b, d, pd) in enumerate(env.action_map):
        if pa == price_adj and b == bundle and d == discount and pd == price_discipline:
            return i
    raise ValueError("not found")


def fixed_policy(env, state):
    return find_action_index(env, 0.0, 0, 0.0, 0)


def rule_based_policy(env, state):
    inv_b, trend_b, season_b, bundle_flag, price_locked = state
    if inv_b == 2 and trend_b == 0:
        return find_action_index(env, -0.05, 0, 0.10, 0)
    if season_b == 1 and trend_b >= 1:
        return find_action_index(env, 0.0, 1, 0.0, 0)
    return find_action_index(env, 0.0, 0, 0.0, 0)


def run_episode(env, policy_fn=None, agent=None, train=False):
    s = env.reset()
    done = False
    total_reward = 0.0
    total_raw_profit = 0.0
    total_stockout = 0
    inv_sum = 0.0
    bundle_count = discount_count = price_change_count = 0
    lock_choice_count = locked_state_count = 0
    while not done:
        a = agent.act(s, greedy=not train) if agent is not None else policy_fn(env, s)
        s_next, r, done, info = env.step(a)
        if train and agent is not None:
            agent.update(s, a, r, s_next, done)
        s = s_next
        total_reward += r
        total_raw_profit += info['raw_profit']
        total_stockout += info['stockout']
        inv_sum += info['inventory']
        bundle_count += int(info['bundle'] == 1)
        discount_count += int(info['discount'] > 0)
        price_change_count += int(info['price_changed'])
        locked_state_count += int(info['locked_state'])
        action_tuple = env.action_map[a]
        lock_choice_count += int(len(action_tuple) > 3 and action_tuple[3] == 1)
    return {
        'reward': total_reward,
        'raw_profit': total_raw_profit,
        'avg_daily_profit': total_raw_profit / env.horizon,
        'stockout_total': total_stockout,
        'avg_inventory': inv_sum / env.horizon,
        'bundle_ratio': bundle_count / env.horizon,
        'discount_ratio': discount_count / env.horizon,
        'price_change_ratio': price_change_count / env.horizon,
        'lock_choice_ratio': lock_choice_count / env.horizon,
        'locked_state_ratio': locked_state_count / env.horizon,
    }


def train_q_learning(env, episodes=1500):
    agent = QLearningAgent(n_actions=env.n_actions())
    reward_hist = []
    for _ in range(episodes):
        m = run_episode(env, agent=agent, train=True)
        reward_hist.append(m['reward'])
        agent.decay_eps()
    return agent, reward_hist


def evaluate_policy(env, n_eval=200, policy_fn=None, agent=None):
    all_m = []
    for _ in range(n_eval):
        if agent is not None:
            m = run_episode(env, agent=agent, train=False)
        else:
            m = run_episode(env, policy_fn=policy_fn, train=False)
        all_m.append(m)
    keys = all_m[0].keys()
    avg = {k: float(np.mean([x[k] for x in all_m])) for k in keys}
    std = {k: float(np.std([x[k] for x in all_m])) for k in keys}
    return avg, std


# ====================================================================
#                         RUN ALL EXPERIMENTS
# ====================================================================
print("Section 5: training main RL agent for 1500 episodes...")
env = PricingBundleEnv(horizon=120)
agent, reward_hist = train_q_learning(env, episodes=1500)

window = 20
smoothed = np.convolve(reward_hist, np.ones(window) / window, mode='valid')
plt.figure(figsize=(8, 4))
plt.plot(smoothed, color='steelblue')
plt.title('Training Reward (moving avg, w=20)')
plt.xlabel('Episode'); plt.ylabel('Cumulative Normalised Reward')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='break-even')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_training_curve.png'), dpi=160)
plt.close()
print(f"  final eps = {agent.eps:.4f}, states seen = {len(agent.Q)} / 72")

print("Section 6: evaluating policies...")
env_eval = PricingBundleEnv(horizon=120)
avg_fixed, std_fixed = evaluate_policy(env_eval, n_eval=200, policy_fn=fixed_policy)
avg_rule, std_rule = evaluate_policy(env_eval, n_eval=200, policy_fn=rule_based_policy)
avg_rl, std_rl = evaluate_policy(env_eval, n_eval=200, agent=agent)

# Bar chart
labels = ['Fixed', 'Rule-based', 'Q-learning']
profits = [avg_fixed['avg_daily_profit'], avg_rule['avg_daily_profit'], avg_rl['avg_daily_profit']]
errs = [std_fixed['avg_daily_profit'], std_rule['avg_daily_profit'], std_rl['avg_daily_profit']]
colors = ['#aec6cf', '#ffb347', '#77dd77']

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, profits, yerr=errs, capsize=6, color=colors, edgecolor='gray')
ax.set_ylabel('Avg Daily Profit (200 episodes)')
ax.set_title('Policy Comparison: Average Daily Profit')
ax.grid(axis='y', alpha=0.5)
for bar, p in zip(bars, profits):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
            f'{p:.1f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_policy_comparison.png'), dpi=160)
plt.close()

# Discipline behaviour
fig2, axes = plt.subplots(1, 3, figsize=(13, 4))
pcr = [avg_fixed['price_change_ratio'], avg_rule['price_change_ratio'], avg_rl['price_change_ratio']]
lcr = [avg_fixed['lock_choice_ratio'], avg_rule['lock_choice_ratio'], avg_rl['lock_choice_ratio']]
lsr = [avg_fixed['locked_state_ratio'], avg_rule['locked_state_ratio'], avg_rl['locked_state_ratio']]
axes[0].bar(labels, pcr, color=colors, edgecolor='gray'); axes[0].set_title('Actual Price-Change Frequency'); axes[0].set_ylabel('Fraction'); axes[0].grid(axis='y', alpha=0.4)
axes[1].bar(labels, lcr, color=colors, edgecolor='gray'); axes[1].set_title('Lock-Choice Ratio (action-level)'); axes[1].grid(axis='y', alpha=0.4)
axes[2].bar(labels, lsr, color=colors, edgecolor='gray'); axes[2].set_title('Locked-State Ratio (env-level)'); axes[2].grid(axis='y', alpha=0.4)
plt.suptitle('Price Discipline Behaviour'); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_discipline.png'), dpi=160)
plt.close()

print("Section 8: robustness sweep over elasticity...")
elasticities = [0.10, 0.20, 0.35]
profits_fixed, profits_rule, profits_rl = [], [], []
for e in elasticities:
    env_r = PricingBundleEnv(horizon=120)
    env_r.price_sensitivity = e
    ag, _ = train_q_learning(env_r, episodes=400)
    af, _ = evaluate_policy(env_r, n_eval=100, policy_fn=fixed_policy)
    ar, _ = evaluate_policy(env_r, n_eval=100, policy_fn=rule_based_policy)
    aq, _ = evaluate_policy(env_r, n_eval=100, agent=ag)
    profits_fixed.append(af['avg_daily_profit'])
    profits_rule.append(ar['avg_daily_profit'])
    profits_rl.append(aq['avg_daily_profit'])

x = np.arange(len(elasticities)); w = 0.25
plt.figure(figsize=(8, 4))
plt.bar(x - w, profits_fixed, width=w, label='Fixed', color='#aec6cf', edgecolor='gray')
plt.bar(x,     profits_rule,  width=w, label='Rule-based', color='#ffb347', edgecolor='gray')
plt.bar(x + w, profits_rl,    width=w, label='Q-learning', color='#77dd77', edgecolor='gray')
plt.xticks(x, [str(e) for e in elasticities])
plt.xlabel('Price Sensitivity (epsilon)'); plt.ylabel('Avg Daily Profit (100 eps)')
plt.title('Robustness across Demand Elasticity'); plt.legend(); plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_robustness.png'), dpi=160)
plt.close()

print("Section 9: ablation - bundle on/off ...")
class PricingNoBundleEnv(PricingBundleEnv):
    def __init__(self, horizon=120):
        super().__init__(horizon)
        self.action_map = []
        for pa in self.price_adjustments:
            for d in self.discount_options:
                for pd in self.discipline_options:
                    self.action_map.append((pa, 0, d, pd))

env_full = PricingBundleEnv(horizon=120)
env_no_bundle = PricingNoBundleEnv(horizon=120)
agent_full, _ = train_q_learning(env_full, episodes=1500)
agent_no_bundle, _ = train_q_learning(env_no_bundle, episodes=1500)
avg_with, std_with = evaluate_policy(env_full, n_eval=200, agent=agent_full)
avg_without, std_without = evaluate_policy(env_no_bundle, n_eval=200, agent=agent_no_bundle)

labels2 = ['RL w/ Bundle', 'RL w/o Bundle']
p2 = [avg_with['avg_daily_profit'], avg_without['avg_daily_profit']]
e2 = [std_with['avg_daily_profit'], std_without['avg_daily_profit']]
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(labels2, p2, yerr=e2, capsize=6, color=['#77dd77', '#cfcfc4'], edgecolor='gray')
ax.set_ylabel('Avg Daily Profit'); ax.set_title('Ablation: Effect of Bundle Action')
ax.grid(axis='y', alpha=0.4); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_ablation.png'), dpi=160)
plt.close()

print("Section 10: trajectory replay ...")
def replay_episode(env, agent):
    s = env.reset(); done = False
    traj = {'day': [], 'effective_price': [], 'demand': [], 'sales': [],
            'inventory': [], 'raw_profit': [], 'reward': [],
            'bundle': [], 'discount': [], 'price_changed': []}
    day = 0
    while not done:
        a = agent.act(s, greedy=True)
        s, r, done, info = env.step(a)
        traj['day'].append(day)
        traj['effective_price'].append(info['effective_price'])
        traj['demand'].append(info['demand'])
        traj['sales'].append(info['sales'])
        traj['inventory'].append(info['inventory'])
        traj['raw_profit'].append(info['raw_profit'])
        traj['reward'].append(r)
        traj['bundle'].append(info['bundle'])
        traj['discount'].append(info['discount'])
        traj['price_changed'].append(int(info['price_changed']))
        day += 1
    return traj

viz_env = PricingBundleEnv(horizon=120)
traj = replay_episode(viz_env, agent_full)
days = np.array(traj['day'])
profits_d = np.array(traj['raw_profit'])
cum_profit = np.cumsum(profits_d)
fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
axes[0].plot(days, traj['effective_price'], color='#2a9d8f', linewidth=2)
axes[0].set_ylabel('Effective Price'); axes[0].set_title('Pricing Trajectory (Greedy RL Policy)')
axes[0].grid(alpha=0.3)
axes[1].plot(days, traj['demand'], label='Demand', color='#264653', linewidth=1.8)
axes[1].plot(days, traj['sales'], label='Sales', color='#e76f51', linestyle='--', linewidth=1.8)
axes[1].set_ylabel('Units'); axes[1].legend(loc='upper right'); axes[1].grid(alpha=0.3)
axes[2].plot(days, traj['inventory'], color='#457b9d', linewidth=2)
axes[2].set_ylabel('Inventory'); axes[2].grid(alpha=0.3)
axes[3].bar(days, profits_d, color='#f4a261', alpha=0.7, label='Daily Profit')
axes[3].plot(days, cum_profit, color='#1d3557', linewidth=2, label='Cumulative Profit')
axes[3].set_ylabel('Profit'); axes[3].set_xlabel('Day'); axes[3].legend(loc='upper left'); axes[3].grid(alpha=0.3)
change_days = days[np.array(traj['price_changed']) == 1]
for ax in axes:
    for d in change_days:
        ax.axvline(d, color='gray', alpha=0.12, linewidth=1)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig_trajectory.png'), dpi=160)
plt.close()

# Q-table greedy actions table (for the report)
greedy_rows = []
for state in sorted(agent.Q.keys()):
    best_a = int(np.argmax(agent.Q[state]))
    pa, b, d, pd = env.action_map[best_a]
    greedy_rows.append({
        'state': list(state),
        'price_adj': float(pa),
        'bundle': int(b),
        'discount': float(d),
        'discipline': 'hysteresis(3)' if pd == 0 else 'lock(7)',
    })

# Save metrics
ref = avg_fixed['avg_daily_profit']
results = {
    'training': {
        'episodes': 1500,
        'final_eps': float(agent.eps),
        'states_seen': int(len(agent.Q)),
        'states_total': 72,
        'reference_profit': float(env.reference_profit),
    },
    'main_eval': {
        'fixed':  {'avg': avg_fixed,  'std': std_fixed},
        'rule':   {'avg': avg_rule,   'std': std_rule},
        'q_learn':{'avg': avg_rl,     'std': std_rl},
        'rl_vs_fixed_pct': (avg_rl['avg_daily_profit'] - ref)/abs(ref)*100,
        'rule_vs_fixed_pct': (avg_rule['avg_daily_profit'] - ref)/abs(ref)*100,
    },
    'robustness': {
        'elasticities': elasticities,
        'profits_fixed': profits_fixed,
        'profits_rule': profits_rule,
        'profits_rl': profits_rl,
    },
    'ablation': {
        'with_bundle': {'avg': avg_with, 'std': std_with},
        'without_bundle': {'avg': avg_without, 'std': std_without},
    },
    'trajectory_summary': {
        'total_raw_profit': float(cum_profit[-1]),
        'price_change_days': int(np.sum(traj['price_changed'])),
        'bundle_days': int(np.sum(traj['bundle'])),
        'discount_days': int(np.sum(np.array(traj['discount']) > 0)),
        'horizon': int(len(days)),
    },
    'greedy_policy_sample': greedy_rows,
}

with open(os.path.join(HERE, 'metrics.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\nDONE.  metrics.json saved at", os.path.join(HERE, 'metrics.json'))
print("\nSummary:")
print(f"  Fixed     avg daily profit  = {avg_fixed['avg_daily_profit']:.2f} ± {std_fixed['avg_daily_profit']:.2f}")
print(f"  Rule      avg daily profit  = {avg_rule['avg_daily_profit']:.2f} ± {std_rule['avg_daily_profit']:.2f}")
print(f"  Q-learn   avg daily profit  = {avg_rl['avg_daily_profit']:.2f} ± {std_rl['avg_daily_profit']:.2f}")
print(f"  RL  vs Fixed: {results['main_eval']['rl_vs_fixed_pct']:+.1f}%")
print(f"  Rule vs Fixed: {results['main_eval']['rule_vs_fixed_pct']:+.1f}%")
print(f"  Bundle ablation: with={avg_with['avg_daily_profit']:.2f}, without={avg_without['avg_daily_profit']:.2f}")
