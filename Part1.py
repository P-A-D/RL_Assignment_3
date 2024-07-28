import joblib
import numpy as np
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


class Sarsa:
    def __init__(self, alpha):
        self.alpha = alpha
        self.action_value_func = np.random.random((7*7, 4))  # random sample from uniform distribution in [0, 1)
        self.policy = np.random.randint(0, 4, size=49)

    def learn(self, epsilon):
        state = np.random.randint(0, 49)
        action = select_action(self.policy, state, epsilon)




if __name__ == '__main__':
    print(np.arange(49).reshape((7, 7)))
    pass

