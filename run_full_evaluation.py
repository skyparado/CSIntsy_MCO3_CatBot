"""
Run this from your project folder (same place bot.py lives) with:

    python run_full_evaluation.py

It trains a Q-table for each of the 5 graded cats, evaluates each one over
100 greedy episodes with evaluation.py's evaluate_q_table(), and also runs
the Trainer cat (Loki, your split-personality generalization stress test)
so you have numbers for both the "known cats" table and the "hidden/
generalization" discussion in your report.

No rendering is used so it runs fast and headless-safe.
Copy everything that gets printed and paste it back to Claude.
"""
import time
from training import train_bot
from evaluation import evaluate_q_table

CATS = ["batmeow", "mittens", "paotsin", "peekaboo", "squiddyboi"]

results = {}

for cat in CATS:
    print(f"\n{'#'*60}")
    print(f"# TRAINING AGAINST: {cat}")
    print(f"{'#'*60}")
    start = time.time()
    q_table = train_bot(cat_name=cat, render=-1)
    elapsed = time.time() - start
    print(f"Training time for {cat}: {elapsed:.2f}s")

    evaluate_q_table(q_table, cat_name=cat, num_episodes=100, max_steps=60)

    # Bonus: test this same q_table (trained only on 'cat') against Loki
    # (trainer cat, split-personality) to see how well it generalizes.
    print(f"--- Cross-check: {cat}-trained bot vs Loki (trainer) ---")
    evaluate_q_table(q_table, cat_name="trainer", num_episodes=100, max_steps=60)

print(f"\n{'#'*60}")
print("# DONE. Copy everything above and paste it back to Claude.")
print(f"{'#'*60}")