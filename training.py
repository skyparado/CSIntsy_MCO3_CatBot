import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env

try:
    import custom_kitties
except ImportError:
    custom_kitties = None

STEP_PENALTY = -0.2     
CATCH_BONUS = 100.0     
                        
WALL_PENALTY = -2.0    
DISTANCE_PENALTY = -3.5  
                         
def decode_state(state: int):
    bot_row = state // 1000
    bot_col = (state // 100) % 10
    cat_row = (state // 10) % 10
    cat_col = state % 10
    return bot_row, bot_col, cat_row, cat_col

def compute_reward(state: int, next_state: int, terminated: bool) -> float:
    if terminated:
        return CATCH_BONUS 
    
    bot_r, bot_c, cat_r, cat_c = decode_state(state)
    next_bot_r, next_bot_c, _, _ = decode_state(next_state)
    
    if bot_r == next_bot_r and bot_c == next_bot_c:
        return WALL_PENALTY 
    
    dist_old = abs(bot_r - cat_r) + abs(bot_c - cat_c)
    dist_new = abs(next_bot_r - cat_r) + abs(next_bot_c - cat_c)

    if dist_new < dist_old:
        return STEP_PENALTY 
    else:
        return DISTANCE_PENALTY


MOVE_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
TIE_TOLERANCE = 0.2   


def apply_action(row: int, col: int, action: int, grid_size: int = 8):
    if action >= len(MOVE_DELTAS):
        return row, col  
    d_row, d_col = MOVE_DELTAS[action]
    return (min(max(0, row + d_row), grid_size - 1),
            min(max(0, col + d_col), grid_size - 1))


def repair_greedy_loops(q_table, grid_size: int = 8) -> int:
    repaired = 0
    for bot_row in range(grid_size):
        for bot_col in range(grid_size):
            for cat_row in range(grid_size):
                for cat_col in range(grid_size):
                    if bot_row == cat_row and bot_col == cat_col:
                        continue  
                    state = bot_row * 1000 + bot_col * 100 + cat_row * 10 + cat_col
                    values = q_table[state]
                    greedy = int(np.argmax(values))

                    next_row, next_col = apply_action(bot_row, bot_col, greedy, grid_size)
                    if (next_row, next_col) == (bot_row, bot_col):
                        stuck = True 
                    else:
                        follow_up = next_row * 1000 + next_col * 100 + cat_row * 10 + cat_col
                        follow_action = int(np.argmax(q_table[follow_up]))
                        stuck = apply_action(next_row, next_col, follow_action,
                                             grid_size) == (bot_row, bot_col)
                    if not stuck:
                        continue

                    distance = abs(bot_row - cat_row) + abs(bot_col - cat_col)
                    closer_action, closer_value = None, None
                    for action in range(len(values)):
                        move_row, move_col = apply_action(bot_row, bot_col, action, grid_size)
                        if abs(move_row - cat_row) + abs(move_col - cat_col) < distance:
                            if closer_value is None or values[action] > closer_value:
                                closer_action, closer_value = action, values[action]

                    if (closer_action is not None and closer_action != greedy
                            and values[greedy] - closer_value <= TIE_TOLERANCE):
                        values[closer_action] = values[greedy] + 1e-3
                        repaired += 1
    return repaired


def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    episodes = 5000 
    alpha_start = 0.5
    alpha_end = 0.05
    gamma = 0.98    

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = (1.0 - epsilon_min) / (episodes * 0.85) 

    max_steps_per_episode = 60 
    total_steps = 0
    max_burnin_steps = 25   
    burnin_episode_fraction = 0.9
    noise_env = None
    if custom_kitties is not None:
        try:
            noise_env = make_env(cat_type="trainer")
        except Exception:
            noise_env = None
    noise_episode_fraction = 0.1
    noise_steps_per_round = 20
    
    for ep in range(1, episodes + 1):
        alpha = alpha_start + (alpha_end - alpha_start) * (ep / episodes)
        state, _ = env.reset()
        done = False
        steps = 0

        if random.random() < burnin_episode_fraction:
            burnin_steps = random.randint(0, max_burnin_steps)
            for _ in range(burnin_steps):
                burnin_action = env.action_space.sample()
                state, _, burnin_terminated, burnin_truncated, _ = env.step(burnin_action)
                if burnin_terminated or burnin_truncated:
                    state, _ = env.reset()
                    break

        while not done and steps < max_steps_per_episode:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward = compute_reward(state, next_state, terminated)

            best_next_value = np.max(q_table[next_state])
            td_target = reward + (0.0 if done else gamma * best_next_value)
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error

            state = next_state
            steps += 1

        epsilon = max(epsilon_min, epsilon - epsilon_decay)
        total_steps += steps

        if noise_env is not None and random.random() < noise_episode_fraction:
                    noise_behavior_name, noise_behavior_fn = random.choice(list(custom_kitties.BEHAVIOR_DICT.values()))
                    noise_env.cat.behavior_name = noise_behavior_name
                    noise_env.cat.current_behavior = noise_behavior_fn

                    noise_state, _ = noise_env.reset()
                    noise_done = False
                    noise_steps = 0
                    while not noise_done and noise_steps < noise_steps_per_round:
                        if random.random() < epsilon:
                            noise_action = noise_env.action_space.sample()
                        else:
                            noise_action = int(np.argmax(q_table[noise_state]))

                        next_noise_state, _, noise_terminated, noise_truncated, _ = noise_env.step(noise_action)
                        noise_done = noise_terminated or noise_truncated

                        noise_reward = compute_reward(noise_state, next_noise_state, noise_terminated)

                        best_next_value = np.max(q_table[next_noise_state])
                        td_target = noise_reward + (0.0 if noise_done else gamma * best_next_value)
                        td_error = td_target - q_table[noise_state][noise_action]
                        q_table[noise_state][noise_action] += alpha * td_error

                        noise_state = next_noise_state
                        noise_steps += 1

        if ep == episodes:
            repaired = repair_greedy_loops(q_table)
            print(f"Training complete! Total steps taken: {total_steps}")
            print(f"Average steps per episode: {total_steps / episodes:.2f}")
            print(f"Greedy loops repaired: {repaired}")

        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table