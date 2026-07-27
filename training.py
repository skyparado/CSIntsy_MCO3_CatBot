import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env

# --- Reward hyperparameters -------------------------------------------------
STEP_PENALTY = -0.2     # small constant cost per step, discourages dawdling
                        # CHANGE: adjusted to -0.2 to encourage more exploration and prevent the bot from standing still
CATCH_BONUS = 100.0     # large terminal reward, dominates shaping/step terms
                        # CHANGE: increased to make catching more rewarding and emphasize goal
SHAPING_WEIGHT = 1.0    # scales the potential-based shaping term
WALL_PENALTY = -2.0     # CHANGE: added penalty for bumping into walls to discourage invalid moves
DISTANCE_PENALTY = -1.8  # CHANGE: added penalty for moving away/staying at the same distance from the cat to encourage pursuit
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
    #CHANGE: shifted from Potential-Based Reward Shaping (PBRS) to Heuristic Reward Shaping (HRS) with 
    #     negative intermidate rewards so naka-lock in lang yung agent with avoiding bigger negative rewards
    if terminated:
        return CATCH_BONUS # caught the cat, give big positive reward
    
    bot_r, bot_c, cat_r, cat_c = decode_state(state)
    next_bot_r, next_bot_c, _, _ = decode_state(next_state)
    
    if bot_r == next_bot_r and bot_c == next_bot_c:
        return WALL_PENALTY # penalize bumping into walls 
    
    # compute Manhattan distances before and after the action
    dist_old = abs(bot_r - cat_r) + abs(bot_c - cat_c)
    dist_new = abs(next_bot_r - cat_r) + abs(next_bot_c - cat_c)
    
    # strictly negative intermediate rewards prevent positive Q-value loop traps (i.e staying in place forever coz that would not give a negative reward)
    if dist_new < dist_old:
        return STEP_PENALTY   # small negative reward for moving closer to the cat
    else:
        return DISTANCE_PENALTY  # large negative reward for moving away or staying at the same distance from the cat

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
    alpha = 0.3     # CHANGE: make learning rate more aggressive
    gamma = 0.98    # CHANGE: increase discount factor to prioritize long-term rewards

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = (1.0 - epsilon_min) / (episodes * 0.85) # CHANGE: slower decay to allow more exploration in early training

    max_steps_per_episode = 60 # CHANGE: limit max steps to 60 so bot is forced to catch within the limit
    total_steps = 0

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

        epsilon = max(epsilon_min, epsilon - epsilon_decay)
        total_steps += steps

        if ep == episodes:
            print(f"Training complete! Total steps taken: {total_steps}")
            print(f"Average steps per episode: {total_steps / episodes:.2f}")

        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table