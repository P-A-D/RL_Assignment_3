import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


"""
- Problem environment:
    - terminal states at 6 and 42
    - + 1 reward at 6
    - -1 reward at 42

- Problem objectives:
    - exact value function of random walk
    - random walk value function through 
        - Monte Carlo
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
    if state == 6:
        return state, 1
    elif state == 42:
        return state, -1
    elif state in [0, 7, 14, 21, 28, 35, 42] and action == 2:
        return state, 0
    elif state in [0, 1, 2, 3, 4, 5, 6] and action == 0:
        return state, 0
    elif state in [6, 13, 20, 27, 34, 41, 48] and action == 3:
        return state, 0
    elif state in [42, 43, 44, 45, 46, 47, 48] and action == 1:
        return state, 0
    elif action == 0:
        return state-7, 0
    elif action == 1:
        return state+7, 0
    elif action == 2:
        return state-1, 0
    elif action == 3:
        return state+1, 0


def select_action(policy, state, epsilon=None):
    """
    policy => a vector of length 49 where each entry shows the best action to take in its respective state.
    state  => the state in which the agent is and wants to select an action in.
    epsilon=> the value of epsilon for the e-greedy algorithms. If None, actions will be greedy.
    """
    if epsilon is None:  # greedy
        return policy[state]
    else:                # epsilon-greedy
        return policy[state] if np.random.random() > epsilon else np.random.choice(list({0, 1, 2, 3} - {policy[state]}))


# =============================================================================================================
# ================================================   Part 1   =================================================
# =============================================================================================================

class GradientMonteCarlo:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value_function = np.random.random(49)
        # todo: dafuq is a differentiable function here?
        self.w = np.random.random(49)

    def learn(self, discount=0.95, episodes=100000, max_episode_length=2500):
        for _ in range(episodes):
            initial_state = np.random.choice(range(25))
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
                if next_state in [6, 42] or steps > max_episode_length:
                    break
                state_sequence.append(next_state)
                action_sequence.append(np.random.choice(range(4)))

            # evaluating Q(s,a)
            g = 0
            for i in range(len(reward_sequence) - 1, -1, -1): # todo: inside this loop needs a fix
                g = g * discount + reward_sequence[i]
                reward_sequence.append(g)
                self.state_visits[state_sequence[i]] += 1
                self.value_function[state_sequence[i]] += 1/self.state_visits[state_sequence[i]]*(g - self.state_returns[state_sequence[i]])
        return self.value_function


# =============================================================================================================
# ================================================   Part 2   =================================================
# =============================================================================================================


class SGTD:
    def __init__(self):
        pass

    def learn(self):
        pass


# =============================================================================================================
# ================================================    Exact   =================================================
# =============================================================================================================


class PolicyIteration:
    def __init__(self, discount=0.95):
        self.discount = discount
        self.value_function = None
        self.policy_function = None

    def learn(self, threshold=0.1, patience=1e5):
        self.value_function = np.random.normal(size=49)
        self.value_function[6] = 1
        self.value_function[42] = -1
        stop = False
        run_count = 0
        while not stop and run_count < patience:
            run_count += 1
            stop = self.update_value_function(threshold=threshold)
            self.value_function[6] = 1
            self.value_function[42] = -1
        print(f"Iteration count at halt = {run_count}")
        return self.value_function

    def update_value_function(self, threshold=0.1):
        old_values = self.value_function.copy()
        for state_index in range(49):
            rewards_accumulated = []
            for action_index in range(4):
                next_state, reward = find_next_state(state_index, action_index)
                rewards_accumulated.append(reward + self.discount * old_values[next_state])
            self.value_function[state_index] = np.array(rewards_accumulated).sum() * 0.25
        stop = False
        return stop


if __name__ == '__main__':
    asd = GradientMonteCarlo()
    valf = asd.learn(episodes=100000, max_episode_length=50)
    visualize_results(valf, 'test')
    pass
    asd = PolicyIteration()
    valf = asd.learn(patience=1000)
    visualize_results(valf, 'Exact Value Function Through Policy Iteration')
