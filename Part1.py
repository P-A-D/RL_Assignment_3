import joblib
import numpy as np
import matplotlib.pyplot as plt


"""
- Problem environment:
    - terminal states at 0 and 4
    - -1 reward for any action (-1 for trying to leave the grid. also -1 for movement between two white cells)
    - moving to any of these 10, 11, 13, 14 returns a -20 reward and agent is sent to cell 20 

- Problem objectives:
    - find optimal policy through => 1) sarsa 2) q-learning
    - plot a trajectory of movement for the optimal policies found

- Problem constrains:
    - use e-greedy action selection

- Approach and assumptions:
states => the 25 states are numbered from 0 to 24. The numbering convention is shown below.
                         [ 0  1  2  3  4]
                         [ 5  6  7  8  9]
                         [10 11 12 13 14]
                         [15 16 17 18 19]
                         [20 21 22 23 24]
policy => a vector of length 25 where each entry shows the best action to take in its respective state.
action_value_function => a matrix of size (25, 4) => each row is a state and each column is an action.
actions => encoded as such: 0=up, 1=down, 2=left, 3=right
"""


def find_next_state(state, action):
    """
    returns a pair of (next state, generated reward) given the current state and action.
    """
    if state in [10, 11, 13, 14]:
        raise ValueError
    next_state = None
    if state in [0, 4]:
        return 0, 0
    elif state in [0, 5, 15, 20] and action == 2:
        return state, -1
    elif state in [1, 2, 3] and action == 0:
        return state, -1
    elif state in [20, 21, 22, 23, 24] and action == 1:
        return state, -1
    elif state in [4, 9, 19, 24] and action == 3:
        return state, -1
    elif action == 0:
        next_state = state - 5
    elif action == 1:
        next_state = state + 5
    elif action == 2:
        next_state = state - 1
    elif action == 3:
        next_state = state + 1

    if next_state in [10, 11, 13, 14]:
        return 20, -20
    elif next_state in [0, 4]:
        return next_state, 0
    else:
        return next_state, -1


def select_action(action_value_function, state, epsilon=None):
    """
    policy => a vector of length 25 where each entry shows the best action to take in its respective state.
    state  => the state in which the agent is and wants to select an action in.
    epsilon=> the value of epsilon for the e-greedy algorithms. If None, actions will be greedy.
    """
    if epsilon is None:  # greedy
        return np.argmax(action_value_function[state, :])
    else:                # epsilon-greedy
        return np.argmax(action_value_function[state, :]) if np.random.random() > epsilon else np.random.choice([0, 1, 2, 3])


def plot_reward_pattern(sequence):
    plt.figure()
    plt.title("Reward patterns")
    plt.xlabel("Episode number")
    plt.ylabel("Accumulated reward")
    plt.plot(sequence)
    plt.show()


# =============================================================================================================
# ================================================   Sarsa   ==================================================
# =============================================================================================================


class Sarsa:
    def __init__(self, alpha):
        self.alpha = alpha
        self.action_value_func = np.random.random((5*5, 4))  # random sample from uniform distribution in [0, 1)
        self.action_value_func[0, :] = 0  # terminal state
        self.action_value_func[4, :] = 0  # terminal state
        self.policy = np.random.randint(0, 4, size=25)
        self.reward_sums = []

    def learn(self, epsilon, episode_count=100000, discount=0.95):
        for i in range(episode_count):
            history = []
            state = 20
            action = select_action(self.action_value_func, state, epsilon)
            step_num = 0
            while True:
                next_state, reward = find_next_state(state, action)
                history.append(reward)
                next_action = select_action(self.action_value_func, next_state, epsilon)
                if next_state in [0, 4]:
                    break
                self.action_value_func[state, action] += self.alpha * (reward + discount*self.action_value_func[next_state, next_action] - self.action_value_func[state, action])
                state = next_state
                action = next_action
                step_num += 1
            self.reward_sums.append(sum(history))
            # print(f"Episode {i} ended after {step_num} steps.")
        self.policy = np.argmax(self.action_value_func, axis=1)
        plot_reward_pattern(self.reward_sums)
        print(self.policy.reshape((5, 5)))
        return self.policy.reshape((5, 5))  # returns the policy


# =============================================================================================================
# ==============================================   Q-learning   ===============================================
# =============================================================================================================
class QLearn:
    def __init__(self, alpha):
        self.alpha = alpha
        self.action_value_func = np.random.random((5 * 5, 4))  # random sample from uniform distribution in [0, 1)
        self.action_value_func[0, :] = 0  # terminal state
        self.action_value_func[4, :] = 0  # terminal state
        self.policy = None
        self.reward_sums = []

    def learn(self, epsilon, episode_count=100000, discount=0.95):
        for i in range(episode_count):
            history = []
            state = 20
            step_num = 0
            while True:
                action = select_action(self.action_value_func, state, epsilon)
                next_state, reward = find_next_state(state, action)
                history.append(reward)
                if next_state in [0, 4]:
                    break
                self.action_value_func[state, action] += self.alpha * (reward + discount*np.argmax(self.action_value_func[next_state, :]) - self.action_value_func[state, action])
                state = next_state
                step_num += 1
            self.reward_sums.append(sum(history))
            print(f"Episode {i} ended after {step_num} steps.")
        self.policy = np.argmax(self.action_value_func, axis=1)
        plot_reward_pattern(self.reward_sums)
        print(self.policy.reshape((5, 5)))
        return self.policy.reshape((5, 5))  # returns the policy


if __name__ == '__main__':
    agent = QLearn(alpha=0.1)
    policy = agent.learn(epsilon=0.1, episode_count=10, discount=0.95)
    pass
    # todo: plot the trajectory for the agent in the report

