import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


"""
- Problem environment:
    - terminal states at 6 and 42
    - + 1 reward at 6
    - -1 reward at 42
    - every episode starts at 24

- Problem objectives:
    - exact value function of random walk
    - random walk value function through 
        - Gradient Monte Carlo
        - semi-gradient TD(0) 

- Problem constrains:
    - behavior = random walk / equiprobable actions 

- Approach and assumptions:
states => the 49 states are numbered from 0 to 48. The numbering convention is shown below.
                                     [ 0  1  2  3  4  5  6]
                                     [ 7  8  9 10 11 12 13]
                                     [14 15 16 17 18 19 20]
                                     [21 22 23 24 25 26 27]
                                     [28 29 30 31 32 33 34]
                                     [35 36 37 38 39 40 41]
                                     [42 43 44 45 46 47 48]
policy => a vector of length 49 where each entry shows the best action to take in its respective state.
action_value_function => a matrix of size (49, 4) => each row is a state and each column is an action.
actions => encoded as such: 0=up, 1=down, 2=left, 3=right
"""


def visualize_results(vector, title):
    """
    This function plots the value functions.
    """
    plt.figure()
    sns.heatmap(vector.reshape((7, 7)), cmap='coolwarm', annot=True, fmt='.2f', square=True)
    plt.title(f"{title}")
    plt.show()


def find_next_state(state, action):
    """
    returns a pair of (next state, generated reward) given the current state and action.
    """
    next_state = None
    # if state == 6:
    #     return state, 1
    # elif state == 42:
    #     return state, -1
    if state in [6, 42]:
        raise ValueError(f"state {state} should not be fed into this function")
    if state in [0, 7, 14, 21, 28, 35] and action == 2:
        return state, 0
    elif state in [0, 1, 2, 3, 4, 5] and action == 0:
        return state, 0
    elif state in [13, 20, 27, 34, 41, 48] and action == 3:
        return state, 0
    elif state in [43, 44, 45, 46, 47, 48] and action == 1:
        return state, 0

    if action == 0:
        next_state = state-7
    elif action == 1:
        next_state = state+7
    elif action == 2:
        next_state = state-1
    elif action == 3:
        next_state = state+1

    if next_state == 6:
        return next_state, 1
    elif next_state == 42:
        return next_state, -1
    else:
        return next_state, 0


def feature_scheme(count):
    features = []
    if count == 2:
        for i in range(49):
            # features: 1- dist to top right, 2- dist to bottom left
            # features.append([np.sqrt((i//7)**2 + (i % 7 - 6)**2), np.sqrt((i // 7 - 6) ** 2 + (i % 7) ** 2)])
            features.append([((i//7) + np.abs(i % 7 - 6))*1.0, (np.abs(i // 7 - 6) + (i % 7))*1.0])
    elif count == 8:
        # features: 1: x index, 2: y index, 3: x dist to top right, 4: y dist to top right, 5: dist to top right,
        # 6: x dist to bottem left, 7: y dist to bottom left, 8: dist to bottom left
        for i in range(49):
            features.append(np.array([i//7, i % 7, (i//7)**2, (i % 7 - 6)**2, np.sqrt((i//7)**2 + (i % 7 - 6)**2),
                                     (i//7 - 6) ** 2, (i % 7) ** 2, np.sqrt((i // 7 - 6) ** 2 + (i % 7) ** 2)]))
    elif count == 10:
        for i in range(49):
            features.append(np.array([i//7, i % 7, (i//7)**2, (i % 7 - 6)**2, np.sqrt((i//7)**2 + (i % 7 - 6)**2),
                                     (i//7 - 6) ** 2, (i % 7) ** 2, np.sqrt((i // 7 - 6) ** 2 + (i % 7) ** 2),
                                      ((i//7) + np.abs(i % 7 - 6))*1.0, (np.abs(i // 7 - 6) + (i % 7))*1.0]))

    features = np.array(features, dtype=float)
    for i in range(features.shape[1]):
        features[:, i] = features[:, i]/features[:, i].max()
    return features, np.random.random(count) - 0.5


def run_n(algorithm, n, alpha, w_size, discount, episodes):
    value_functions = []
    for i in range(n):
        agent = algorithm(alpha, w_size)
        value_functions.append(agent.learn(discount=discount, episodes=episodes))
    val_func = sum(value_functions)/n
    visualize_results(val_func, title=f"Value function averaged over {n} runs.")

# =============================================================================================================
# ================================================   Part 1   =================================================
# =============================================================================================================
"""
For every s, we have a v(s) and an approximation of it as v(s, w) where w is a vector of w_i's
"""


class GradientMonteCarlo:
    def __init__(self, alpha=0.5, w_size=2):
        self.alpha = alpha
        self.features, self.w = feature_scheme(w_size)

    def approximate_s(self, state=None):
        res = (self.features * self.w).sum(axis=1)
        # res[6] = 0
        # res[42] = 0
        # print(res)
        if state is None:
            return res
        else:
            return res[state]

    def get_w_gradient(self, state):
        return self.features[state, :]

    def learn(self, discount=1, episodes=100000):
        for _ in range(episodes):
            initial_state = 24
            initial_action = np.random.choice(range(4))
            state_sequence = [initial_state]
            action_sequence = [initial_action]
            reward_sequence = []
            steps = 0
            # generating a sequence
            while True:
                steps += 1
                next_state, reward = find_next_state(state_sequence[-1], action_sequence[-1])
                reward_sequence.append(reward)
                state_sequence.append(next_state)
                if next_state in [6, 42]:
                    break
                action_sequence.append(np.random.choice(range(4)))

            # updating w
            g = reward_sequence[-1]
            for i in range(len(reward_sequence) - 2, -1, -1):
                g = g * discount + reward_sequence[i]
                s_t = state_sequence[i]
                self.w += self.alpha*(g - self.approximate_s(state=s_t)) * self.get_w_gradient(state=s_t)
        vec = self.approximate_s()
        vec[6] = 1
        vec[42] = -1
        return vec


# =============================================================================================================
# ================================================   Part 2   =================================================
# =============================================================================================================


class SGTD:
    def __init__(self, alpha=0.5, w_size=2):
        self.alpha = alpha
        self.features, self.w = feature_scheme(w_size)

    def approximate_s(self, state=None):
        res = (self.features * self.w).sum(axis=1)
        res[6] = 0
        res[42] = 0
        # print(res)
        if state is None:
            return res
        else:
            return res[state]

    def get_w_gradient(self, state):
        return self.features[state, :]

    def learn(self, discount=1, episodes=100000):
        for _ in range(episodes):
            state = 24
            steps = 0
            while True:
                steps += 1
                action = np.random.choice(range(4))
                next_state, reward = find_next_state(state, action)
                self.w += self.alpha * (reward + discount * self.approximate_s(state=next_state) - self.approximate_s(state=state)) * self.get_w_gradient(state=state)
                state = next_state
                if next_state in [6, 42]:
                #     print(reward)
                    break
        vec = self.approximate_s()
        vec[6] = 1
        vec[42] = -1
        return vec


# =============================================================================================================
# ================================================    Exact   =================================================
# =============================================================================================================


class PolicyIteration:
    def __init__(self, discount=0.95):
        self.discount = discount
        self.value_function = None
        self.policy_function = None

    def learn(self, threshold=0.01, patience=1e5):
        self.value_function = np.random.normal(size=49)
        self.value_function[6] = 0
        self.value_function[42] = 0
        stop = False
        run_count = 0
        while not stop and run_count < patience:
            run_count += 1
            stop = self.update_value_function(threshold=threshold)
            self.value_function[6] = 0
            self.value_function[42] = 0
        print(f"Iteration count at halt = {run_count}")
        self.value_function[6] = 1
        self.value_function[42] = -1
        return self.value_function

    def update_value_function(self, threshold=0.1):
        old_values = self.value_function.copy()
        for state_index in set(range(49)) - {6, 42}:
            rewards_accumulated = []
            for action_index in range(4):
                next_state, reward = find_next_state(state_index, action_index)
                rewards_accumulated.append(reward + self.discount * old_values[next_state])
            self.value_function[state_index] = sum(rewards_accumulated) * 0.25
        stop = False if np.abs(old_values - self.value_function).any() > threshold else True
        return stop


if __name__ == '__main__':
    # agent = GradientMonteCarlo(alpha=0.1, w_size=10)
    # v_func = agent.learn(discount=1, episodes=100000, max_episode_length=14)
    # visualize_results(v_func, "Gradient Monte Carlo value function estimation")
    # print(v_func)
    # print(agent.w)
    # print(agent.features)
    # run_n(GradientMonteCarlo, n=10, alpha=0.1, w_size=2, discount=1, episodes=100000, max_episode_length=12)
    # a = [np.array([1, 2]), np.array([4, 5])]
    # print(mea(a))
    # agent = SGTD(alpha=0.1, w_size=2)
    # v_func = agent.learn(discount=1, episodes=1000, max_episode_length=20)
    # visualize_results(v_func, "Semi-gradient TD(0) value function estimation")
    # print(v_func)

    # run_n(GradientMonteCarlo, n=10, alpha=0.25, w_size=2, discount=1, episodes=10000)

    # agent = PolicyIteration(discount=1)
    # v_func = agent.learn()
    # visualize_results(v_func, "Policy iteration - exact values")
    pass
