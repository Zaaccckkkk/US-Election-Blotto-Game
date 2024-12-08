from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import linprog

# define state votes
state_votes = np.array([19, 16, 16, 15, 11, 10, 6])

# 5% initial spent in each state
base_spending = 5

# generate all possible allocations
extra_pieces = 10
allocations = list(combinations_with_replacement(range(7), extra_pieces))

strategies = []
for alloc in allocations:
    strategy = np.full(7, base_spending, dtype=float)
    for idx in alloc:
        strategy[idx] += 6.5  # one portion of spent: 65%/10=6.5%
    strategies.append(strategy)

strategies = np.array(strategies)

n_strategies = len(strategies)
print("Number of Strategies:", n_strategies)
payoff_matrix = np.zeros((n_strategies, n_strategies))

# Calculate the payoff matrix
for i, d_strategy in enumerate(strategies):
    for j, r_strategy in enumerate(strategies):
        net_score = 0
        for k in range(7):  # 遍历7个州
            if d_strategy[k] > r_strategy[k]:
                net_score += state_votes[k]
            elif d_strategy[k] < r_strategy[k]:
                net_score -= state_votes[k]
        payoff_matrix[i, j] = net_score
print("Payoff Matrix finished.")

# Calculate the Nash equilibrium
c = [-1] + [0] * n_strategies
A_eq = [[0] + [1] * n_strategies]
b_eq = [1]
bounds = [(None, None)] + [(0, None)] * n_strategies

# z <= p^T * A
A_ub = []
b_ub = []
for j in range(n_strategies):
    row = [1] + [-payoff_matrix[i, j] for i in range(n_strategies)]
    A_ub.append(row)
    b_ub.append(0)

result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
democrat_mixed_strategy = result.x[1:]
game_value = -result.fun

# Republican
c = [1] + [0] * n_strategies
A_eq = [[0] + [1] * n_strategies]
b_eq = [1]

# w >= q^T * A
A_ub = []
b_ub = []
for i in range(n_strategies):
    row = [-1] + [payoff_matrix[i, j] for j in range(n_strategies)]
    A_ub.append(row)
    b_ub.append(0)

result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
republican_mixed_strategy = result.x[1:]

# Get the strategies with prob>=1e-2
threshold = 1e-2
d_high_prob_indices = np.where(democrat_mixed_strategy > threshold)[0]
d_high_prob_strategies = strategies[d_high_prob_indices]
d_high_prob_values = democrat_mixed_strategy[d_high_prob_indices]
r_high_prob_indices = np.where(republican_mixed_strategy > threshold)[0]
r_high_prob_strategies = strategies[r_high_prob_indices]
r_high_prob_values = republican_mixed_strategy[r_high_prob_indices]

# Get the result strategy
result = np.zeros(7)
for i in range(len(d_high_prob_indices)):
    result += np.array(d_high_prob_strategies[i]) * d_high_prob_values[i]

# Print the results
print(
    "Game Value:", game_value,
    "d_Strategy:", [list(strategy) for strategy in d_high_prob_strategies],
    "d_Probability:", d_high_prob_values,
    "r_Strategy:", [list(strategy) for strategy in r_high_prob_strategies],
    "r_Probability:", r_high_prob_values)

print("The result strategy is:", result)
