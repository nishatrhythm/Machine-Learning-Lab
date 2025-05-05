import numpy as np
import random

# Environment setup
states = [0, 1, 2, 3, 4]  # 5 states
actions = [0, 1]          # 0 = left, 1 = right

# Rewards matrix
R = np.array([
    [-1, 0],  # From state 0: left invalid (-1), right to 1 (reward 0)
    [0, 0],   # state 1: left to 0 (reward 0), right to 2
    [0, 0],   # state 2
    [0, 10],  # state 3: right to 4 (goal, reward 10)
    [0, 0]    # state 4: goal state
])

# Q-table initialization
Q = np.zeros_like(R, dtype=float)

# Hyperparameters
alpha = 0.1     # learning rate
gamma = 0.9     # discount factor
epsilon = 0.2   # exploration rate
episodes = 1000

# Q-learning algorithm
for episode in range(episodes):
    state = random.choice(states[:-1])  # Start from a non-terminal state

    while state != 4:
        # Choose action (ε-greedy)
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(Q[state])

        # Take action → next state
        next_state = state + 1 if action == 1 else max(0, state - 1)
        reward = R[state, action]

        # Q-value update
        # Q(s, a) = Q(s, a) + α * (r + γ * max Q(s', a') - Q(s, a))
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state

# Final learned Q-table
print("Learned Q-table:")
print(Q)