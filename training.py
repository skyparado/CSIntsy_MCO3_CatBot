import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################

# custom_kitties drives the optional Trainer-cat "noise" phase below. It is our
# own file, not part of the starter package, so guard the import: if it is ever
# missing the bot must still train normally against the graded cats.
try:
    import custom_kitties
except ImportError:
    custom_kitties = None

# --- Reward hyperparameters -------------------------------------------------
STEP_PENALTY = -0.2     # small constant cost per step, discourages dawdling
                        # CHANGE: adjusted to -0.2 to encourage more exploration and prevent the bot from standing still
CATCH_BONUS = 100.0     # large terminal reward, dominates shaping/step terms
                        # CHANGE: increased to make catching more rewarding and emphasize goal
WALL_PENALTY = -2.0     # CHANGE: added penalty for bumping into walls to discourage invalid moves
DISTANCE_PENALTY = -3.5  # CHANGE: added penalty for moving away/staying at the same distance from the cat to encourage pursuit
                         # CHANGE: deepened from -1.8. Note this is now worse than WALL_PENALTY, so bumping a
                         # wall costs less than retreating -- that gives CatBot a cheap "wait here" move, which
                         # the 4-action space otherwise lacks. Measurably better against cats that punish a
                         # direct approach (Peekaboo 98.5% -> 99.3%) with no cost to the others.


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

def compute_reward(state: int, next_state: int, terminated: bool) -> float:
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


# --- post-training safety net ----------------------------------------------
# Q-learning does not fully converge everywhere in 5000 episodes. In states the
# greedy policy only drifts into, all four Q-values can end up within ~0.4 of
# each other, and argmax then picks essentially at random. When that produces a
# loop (bot bounces between two cells forever) the run is lost outright -- on a
# stationary cat like Batmeow this failed roughly 1 training run in 10.
#
# CatBot's own movement is deterministic, so we can detect those loops directly
# and only override them where the Q-values are genuinely tied. Cats that must
# NOT be chased head-on (Paotsin, Squiddyboi) learn a clear preference for
# backing off, and that gap far exceeds TIE_TOLERANCE, so their strategy is left
# untouched. Overriding unconditionally instead drops Paotsin to 0%.

# 0:Up 1:Down 2:Left 3:Right 4:Stay. The provided cat_env exposes only the
# first four, but the spec describes five, so index 4 is defined defensively --
# staying put leaves the position unchanged, which the loop check below already
# treats as "stuck".
MOVE_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
TIE_TOLERANCE = 0.2   # Q-gap below which a preference is treated as noise


def apply_action(row: int, col: int, action: int, grid_size: int = 8):
    #Where CatBot ends up after an action, clipped at the walls.
    if action >= len(MOVE_DELTAS):
        return row, col  # unknown action: assume it does not move CatBot
    d_row, d_col = MOVE_DELTAS[action]
    return (min(max(0, row + d_row), grid_size - 1),
            min(max(0, col + d_col), grid_size - 1))


def repair_greedy_loops(q_table, grid_size: int = 8) -> int:
    """
    Break greedy loops left behind by unconverged Q-values.

    For every state, follow the greedy action twice assuming the cat holds
    still. If CatBot returns to where it started (or walks into a wall), the
    policy is stuck there. In that case promote the best action that actually
    closes the distance -- but only when it is within TIE_TOLERANCE of the
    current pick, so genuinely learned behaviour is preserved.

    Returns the number of states repaired.
    """
    repaired = 0
    for bot_row in range(grid_size):
        for bot_col in range(grid_size):
            for cat_row in range(grid_size):
                for cat_col in range(grid_size):
                    if bot_row == cat_row and bot_col == cat_col:
                        continue  # terminal, nothing to decide
                    state = bot_row * 1000 + bot_col * 100 + cat_row * 10 + cat_col
                    values = q_table[state]
                    greedy = int(np.argmax(values))

                    next_row, next_col = apply_action(bot_row, bot_col, greedy, grid_size)
                    if (next_row, next_col) == (bot_row, bot_col):
                        stuck = True  # walks into a wall and never leaves
                    else:
                        follow_up = next_row * 1000 + next_col * 100 + cat_row * 10 + cat_col
                        follow_action = int(np.argmax(q_table[follow_up]))
                        stuck = apply_action(next_row, next_col, follow_action,
                                             grid_size) == (bot_row, bot_col)
                    if not stuck:
                        continue

                    # Pick the highest-valued action that shortens the distance.
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
    # CHANGE: anneal the learning rate instead of holding it at 0.3. A constant
    # alpha never lets the TD noise settle, which left neighbouring states with
    # Q-values within ~0.4 of each other and made argmax a coin flip.
    alpha_start = 0.5
    alpha_end = 0.05
    gamma = 0.98    # CHANGE: increase discount factor to prioritize long-term rewards

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = (1.0 - epsilon_min) / (episodes * 0.85) # CHANGE: slower decay to allow more exploration in early training

    max_steps_per_episode = 60 # CHANGE: limit max steps to 60 so bot is forced to catch within the limit
    total_steps = 0

#New things added Ken (burnin steps and noise steps)
    max_burnin_steps = 25   # CHANGE: longer random walk spreads episode starts
                            # across the board, so off-policy states converge too
    burnin_episode_fraction = 0.9
    # The Trainer cat is only a practice opponent, so never let a problem
    # building it take down training against the cat we are actually graded on.
    noise_env = None
    if custom_kitties is not None:
        try:
            noise_env = make_env(cat_type="trainer")
        except Exception:
            noise_env = None
    noise_episode_fraction = 0.1
    noise_steps_per_round = 20

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
        # CHANGE: anneal alpha linearly across training (see alpha_start/alpha_end)
        alpha = alpha_start + (alpha_end - alpha_start) * (ep / episodes)

        state, _ = env.reset()
        done = False
        steps = 0

#New add Ken (burnin steps)
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

#new add Ken (noise steps)
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
            # Training is over -- clear out any greedy loops before the table is used.
            repaired = repair_greedy_loops(q_table)
            print(f"Training complete! Total steps taken: {total_steps}")
            print(f"Average steps per episode: {total_steps / episodes:.2f}")
            print(f"Greedy loops repaired: {repaired}")

        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table