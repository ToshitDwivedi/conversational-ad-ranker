"""
One-shot driver: regenerate data -> train -> evaluate -> demo the pipeline.

    python run_all.py
"""
import subprocess
import sys

STEPS = [
    ("Step 0  generate synthetic data", [sys.executable, "data/generate_data.py"]),
    ("Steps 3-4  train LightGBM ranker", [sys.executable, "train.py"]),
    ("Steps 5-6  offline metrics + A/B test", [sys.executable, "evaluate.py"]),
    ("Steps 1-4  demo the live pipeline", [sys.executable, "pipeline.py",
        "--user", "u11", "--text", "I'm looking for running shoes under Rs 5000"]),
    ("Benchmark  MovieLens-100K (recognized dataset, ~2-3 min)",
        [sys.executable, "-m", "benchmark.benchmark_movielens"]),
]

for title, cmd in STEPS:
    print("\n" + "#" * 70)
    print("# " + title)
    print("#" * 70)
    subprocess.run(cmd, check=True)
