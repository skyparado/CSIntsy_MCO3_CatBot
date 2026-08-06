
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
    
    print(f"--- Cross-check: {cat}-trained bot vs Loki (trainer) ---")
    evaluate_q_table(q_table, cat_name="trainer", num_episodes=100, max_steps=60)

print(f"\n{'#'*60}")
print("# DONE. Training and evaluation complete for all cats.")
print(f"{'#'*60}")