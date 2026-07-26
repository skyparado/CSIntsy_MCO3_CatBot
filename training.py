import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env

# --- Reward hyperparameters -------------------------------------------------
STEP_PENALTY = -0.1     # small constant cost per step, discourages dawdling
CATCH_BONUS = 20.0      # large terminal reward, dominates shaping/step terms
SHAPING_WEIGHT = 1.0    # scales the potential-based shaping term

#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

def decode_state(state: int):
    """
    Decode the 4-digit state int into bot/cat (row, col) positions.
    State format: RRCCrrcc where the FIRST two digits are CatBot's
    (row, col) and the LAST two digits are the cat's (row, col), per
    the project spec (e.g. state 2305 -> bot at (2,3), cat at (0,5)).
    """
    bot_row = state // 1000
    bot_col = (state // 100) % 10
    cat_row = (state // 10) % 10
    cat_col = state % 10
    return bot_row, bot_col, cat_row, cat_col

def manhattan_distance(state: int) -> int:
    #Manhattan distance between CatBot and the cat for a given state.
    bot_row, bot_col, cat_row, cat_col = decode_state(state)
    return abs(bot_row - cat_row) + abs(bot_col - cat_col)

def potential(state: int) -> float:
    """
    Potential function for reward shaping: higher (less negative) when
    CatBot is closer to the cat. Using -distance means potential increases
    as the agent closes the gap, which is what we want to reward.
    """
    return -float(manhattan_distance(state))


def compute_reward(state: int, next_state: int, terminated: bool, gamma: float) -> float:
    reward = STEP_PENALTY
    
    if terminated:
        reward += CATCH_BONUS
    else:
        shaping = gamma * potential(next_state) - potential(state)
        reward += SHAPING_WEIGHT * shaping
    
    return reward


#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    alpha = 0.1
    gamma = 0.95

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = (epsilon_min / epsilon) ** (1.0 / episodes)

    max_steps_per_episode = 100
    
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ##############################################################################
        state, _ = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward = compute_reward(state, next_state, terminated, gamma)

            best_next_value = np.max(q_table[next_state])
            td_target = reward + (0.0 if done else gamma * best_next_value)
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            state = next_state
            steps += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table