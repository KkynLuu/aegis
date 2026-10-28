#!/usr/bin/env python3
"""
Parallel Execution Harness for M-CTQW Seeds 47-51 on CTU-13 Scenario 10.
Runs missing seeds concurrently using available CPU cores, then merges into magnetic_checkpoint.json.
"""

import os
import sys
import json
import time
import subprocess
from typing import List

PYTHON_BIN = sys.executable or ".pixi/envs/default/bin/python"
SEEDS_TO_RUN = [47, 48, 49, 50, 51]
BASE_CHECKPOINT = "benchmarks/magnetic_checkpoint.json"
FINAL_RESULTS = "benchmarks/magnetic_results.json"


def run_seed_worker(seed: int) -> subprocess.Popen:
    cmd = [
        PYTHON_BIN,
        "benchmarks/benchmark_magnetic.py",
        "--seeds", str(seed),
        "--epochs", "30",
        "--num_threads", "2",
        "--checkpoint_path", f"benchmarks/chk_ctu10_seed_{seed}.json",
        "--output_json", f"benchmarks/res_ctu10_seed_{seed}.json",
    ]
    log_file = open(f"benchmarks/log_ctu10_seed_{seed}.txt", "w")
    print(f"[*] Spawning worker for seed {seed}...")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    proc._log_file = log_file
    return proc


def main():
    print(f"[*] Starting parallel M-CTQW execution for seeds: {SEEDS_TO_RUN}")
    procs = {seed: run_seed_worker(seed) for seed in SEEDS_TO_RUN}

    # Monitor until all complete
    start_time = time.time()
    while True:
        running = [seed for seed, p in procs.items() if p.poll() is None]
        if not running:
            break
        elapsed = time.time() - start_time
        print(f"    Elapsed: {elapsed:>5.1f}s | Active seeds: {running}")
        time.sleep(15)

    print("[*] All workers completed! Closing log files and checking exit codes...")
    for seed, p in procs.items():
        p._log_file.close()
        code = p.returncode
        print(f"    Seed {seed}: Exit code {code}")
        if code != 0:
            print(f"    [!] Error log for seed {seed}:")
            with open(f"benchmarks/log_ctu10_seed_{seed}.txt") as f:
                print(f.read()[-500:])

    # Merge into base checkpoint
    print(f"[*] Merging results into {BASE_CHECKPOINT}...")
    if os.path.exists(BASE_CHECKPOINT):
        with open(BASE_CHECKPOINT, "r") as f:
            base_data = json.load(f)
    else:
        base_data = {"completed_runs": {}}

    completed = base_data.get("completed_runs", {})

    for seed in SEEDS_TO_RUN:
        chk_file = f"benchmarks/chk_ctu10_seed_{seed}.json"
        if os.path.exists(chk_file):
            with open(chk_file, "r") as f:
                worker_data = json.load(f)
            w_runs = worker_data.get("completed_runs", {})
            print(f"    Adding {len(w_runs)} runs from seed {seed}...")
            completed.update(w_runs)
        else:
            print(f"    [!] Warning: Checkpoint for seed {seed} not found ({chk_file})")

    base_data["completed_runs"] = completed
    with open(BASE_CHECKPOINT, "w") as f:
        json.dump(base_data, f, indent=2)
    print(f"[✓] Checkpoint merged! Total runs in checkpoint: {len(completed)}")

    # Clean up worker temp files
    for seed in SEEDS_TO_RUN:
        for fpath in [
            f"benchmarks/chk_ctu10_seed_{seed}.json",
            f"benchmarks/res_ctu10_seed_{seed}.json",
            f"benchmarks/log_ctu10_seed_{seed}.txt",
        ]:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass

    # Final summary pass over all 10 seeds
    print(f"[*] Running final synthesis pass across all 10 seeds (42-51)...")
    summary_cmd = [
        PYTHON_BIN,
        "benchmarks/benchmark_magnetic.py",
        "--seeds", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51",
        "--epochs", "30",
        "--checkpoint_path", BASE_CHECKPOINT,
        "--output_json", FINAL_RESULTS,
    ]
    res = subprocess.run(summary_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("[!] Stderr:", res.stderr)
    print(f"[✓] Parallel 10-seed expansion complete! Check {FINAL_RESULTS}")


if __name__ == "__main__":
    main()
