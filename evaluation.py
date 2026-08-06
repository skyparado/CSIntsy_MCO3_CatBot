import numpy as np
from cat_env import make_env

def evaluate_q_table(q_table, cat_name: str, num_episodes: int = 100, max_steps: int = 60):

    eval_env = make_env(cat_type=cat_name)
    
    cat_caught_count = 0
    total_successful_steps = 0
    step_counts = []

    print(f"\nEvaluating trained bot over {num_episodes} test episodes...")

    for ep in range(num_episodes):
        state, _ = eval_env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = int(np.argmax(q_table[state]))
            state, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            steps += 1

        if terminated: 
            cat_caught_count += 1
            total_successful_steps += steps
            step_counts.append(steps)


    print("\n===========================================")
    print(f"           EVALUATION BENCHMARK            ")
    print("===========================================")
    print(f" Success Rate : {cat_caught_count}/{num_episodes} ({(cat_caught_count / num_episodes) * 100:.1f}%)")
    
    if cat_caught_count > 0:
        avg_steps = total_successful_steps / cat_caught_count
        print(f" Avg Catch    : {avg_steps:.2f} steps")
        print(f" Fastest      : {min(step_counts)} steps")
        print(f" Slowest      : {max(step_counts)} steps")
    print("===========================================\n")