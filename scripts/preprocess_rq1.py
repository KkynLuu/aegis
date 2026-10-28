#!/usr/bin/env python3
"""
Preprocess NetFlow Datasets for Experiment 1 (RQ1: Standard Detection Efficacy).

Processes ALL tabular NetFlow records (NF-BoT-IoT-v3 and CTU-13) into
temporal interaction graph splits:
  - 70% Train (strictly precedes validation in time)
  - 10% Validation (for model checkpointing)
  - 20% Test (unseen future traffic for final evaluation)

Outputs .pt files containing dense attributed graph tensors and metadata:
  data/processed/nf_bot_iot/{train.pt, val.pt, test.pt}
  data/processed/ctu13_10/{train.pt, val.pt, test.pt}
"""

import os
import sys
import time
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
import torch

# Ensure repository root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.netflow_graph_builder import (
    CTU13NetFlowDataset,
    NFBoTIoTNetFlowDataset,
    NetFlowGraph,
    split_graphs_chronological,
    split_graphs_per_attack_chronological,
    save_netflow_graphs,
    load_netflow_graphs,
)


def print_split_statistics(name: str, split_name: str, graphs: List[NetFlowGraph]):
    """Prints diagnostic statistics for a dataset split."""
    n_total = len(graphs)
    if n_total == 0:
        print(f"  [{split_name.upper():<5}] 0 graphs.")
        return

    n_mal = sum(g.y for g in graphs)
    n_ben = n_total - n_mal
    mal_pct = (n_mal / n_total) * 100.0

    nodes = [g.num_nodes for g in graphs]
    edges = [g.num_edges for g in graphs]
    avg_nodes = np.mean(nodes)
    avg_edges = np.mean(edges)

    attacks: Dict[str, int] = {}
    for g in graphs:
        attacks[g.attack_name] = attacks.get(g.attack_name, 0) + 1

    top_attacks = ", ".join(f"{k}: {v}" for k, v in sorted(attacks.items(), key=lambda x: -x[1])[:5])

    print(f"  [{split_name.upper():<5}] Graphs: {n_total:<5} | Benign: {n_ben:<5} | Attack: {n_mal:<5} ({mal_pct:>5.1f}%) | "
          f"Avg Nodes: {avg_nodes:>5.1f} | Avg Edges: {avg_edges:>5.1f} | Top Attacks: [{top_attacks}]")


def process_dataset(
    dataset_name: str,
    raw_path: str,
    output_dir: str,
    max_graphs: Optional[int],
    window_sec: float,
    stride_sec: float,
    min_nodes: int,
    max_nodes: int,
    chunk_rows: int,
):
    print("=" * 90)
    print(f"[*] Preprocessing Dataset: {dataset_name.upper()}")
    print(f"    Raw input path: {raw_path}")
    print(f"    Target output:  {output_dir}")
    print(f"    Max graphs:     {max_graphs if max_graphs is not None else 'ALL (Full Dataset)'}")
    print("=" * 90)

    if not os.path.exists(raw_path):
        print(f"[!] Error: Raw file does not exist: {raw_path}")
        return False

    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Extract graphs from raw tabular flows
    if "nf" in dataset_name.lower():
        # NF-BoT-IoT uses millisecond timestamps
        window_ms = window_sec * 1000.0
        ds = NFBoTIoTNetFlowDataset(
            csv_path=raw_path,
            window_ms=window_ms,
            max_graphs=max_graphs,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            chunk_rows=chunk_rows,
        )
    else:
        # CTU-13 uses unix second timestamps
        ds = CTU13NetFlowDataset(
            binetflow_path=raw_path,
            window_sec=window_sec,
            stride_sec=stride_sec,
            max_graphs=max_graphs,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
        )

    all_graphs = list(ds)
    total_graphs = len(all_graphs)
    if total_graphs == 0:
        print(f"[!] No graphs could be extracted from {raw_path}.")
        return False

    # 2. Chronological Train / Val / Test Split (70 / 10 / 20)
    # Uses per-attack/per-class chronological split so all splits have balanced traffic
    train_graphs, val_graphs, test_graphs = split_graphs_per_attack_chronological(
        all_graphs, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2
    )

    # 3. Print Diagnostic Statistics
    print("\n--- Split Diagnostic Statistics ---")
    print_split_statistics(dataset_name, "train", train_graphs)
    print_split_statistics(dataset_name, "val", val_graphs)
    print_split_statistics(dataset_name, "test", test_graphs)

    # 4. Save to Disk Cache (.pt)
    train_path = os.path.join(output_dir, "train.pt")
    val_path = os.path.join(output_dir, "val.pt")
    test_path = os.path.join(output_dir, "test.pt")

    common_meta = {
        "dataset_name": dataset_name,
        "raw_path": raw_path,
        "node_dim": train_graphs[0].x.shape[-1],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "window_sec": window_sec,
    }

    save_netflow_graphs(train_graphs, train_path, metadata={**common_meta, "split": "train"})
    save_netflow_graphs(val_graphs, val_path, metadata={**common_meta, "split": "val"})
    save_netflow_graphs(test_graphs, test_path, metadata={**common_meta, "split": "test"})

    # Check file sizes
    size_tr = os.path.getsize(train_path) / (1024 * 1024)
    size_va = os.path.getsize(val_path) / (1024 * 1024)
    size_te = os.path.getsize(test_path) / (1024 * 1024)

    dt = time.time() - t0
    print(f"\n[✓] Successfully serialized {total_graphs} graphs in {dt:.2f}s:")
    print(f"    - {train_path} ({size_tr:.2f} MB)")
    print(f"    - {val_path} ({size_va:.2f} MB)")
    print(f"    - {test_path} ({size_te:.2f} MB)")
    print("=" * 90 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess NetFlow Datasets for RQ1 Detection Benchmark")
    parser.add_argument("--dataset", type=str, choices=["all", "nf_bot_iot", "ctu13_10", "ctu13_9", "ctu13_11", "ctu13_all"], default="all",
                        help="Dataset to process: nf_bot_iot, ctu13_10, ctu13_9, ctu13_11, ctu13_all, or all (default: all)")
    parser.add_argument("--max_graphs", type=int, default=None,
                        help="Maximum subgraphs to extract per dataset (default: None -> 100% of all data)")
    parser.add_argument("--window_sec", type=float, default=120.0,
                        help="Temporal window size in seconds (default: 120.0s for NF-BoT-IoT and CTU-13)")
    parser.add_argument("--stride_sec", type=float, default=15.0,
                        help="Temporal window sliding stride in seconds for CTU-13 (default: 15.0s)")
    parser.add_argument("--min_nodes", type=int, default=4,
                        help="Minimum nodes required to form an interaction graph (default: 4)")
    parser.add_argument("--max_nodes", type=int, default=128,
                        help="Maximum nodes per interaction graph (default: 128)")
    parser.add_argument("--chunk_rows", type=int, default=2000000,
                        help="Chunk size when streaming large CSV files (default: 2000000)")
    parser.add_argument("--output_base", type=str, default="data/processed",
                        help="Base directory for processed datasets")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Quick smoke test mode (extracts only 30 graphs per dataset)")

    args = parser.parse_args()

    max_graphs = 30 if args.smoke_test else args.max_graphs
    if args.smoke_test:
        print("[*] Running in SMOKE TEST mode (max_graphs = 30)...")

    # Paths
    nf_raw = "data/NF-BoT-IoT/NF-BoT-IoT-v3.csv"
    ctu_10_raw = "data/CTU-13/10/capture20110818.binetflow"
    ctu_9_raw = "data/CTU-13/9/capture20110817.binetflow"
    ctu_11_raw = "data/CTU-13/11/capture20110818-2.binetflow"
    ctu_all_dir = "data/CTU-13"

    success = True
    if args.dataset in ["all", "nf_bot_iot"]:
        res = process_dataset(
            dataset_name="nf_bot_iot",
            raw_path=nf_raw,
            output_dir=os.path.join(args.output_base, "nf_bot_iot"),
            max_graphs=max_graphs,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            chunk_rows=args.chunk_rows,
        )
        success = success and res

    if args.dataset in ["all", "ctu13_10"]:
        res = process_dataset(
            dataset_name="ctu13_10",
            raw_path=ctu_10_raw,
            output_dir=os.path.join(args.output_base, "ctu13_10"),
            max_graphs=max_graphs,
            window_sec=60.0,
            stride_sec=args.stride_sec,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            chunk_rows=args.chunk_rows,
        )
        success = success and res

    if args.dataset in ["all", "ctu13_9"]:
        res = process_dataset(
            dataset_name="ctu13_9",
            raw_path=ctu_9_raw,
            output_dir=os.path.join(args.output_base, "ctu13_9"),
            max_graphs=max_graphs,
            window_sec=60.0,
            stride_sec=args.stride_sec,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            chunk_rows=args.chunk_rows,
        )
        success = success and res

    if args.dataset in ["all", "ctu13_11"]:
        res = process_dataset(
            dataset_name="ctu13_11",
            raw_path=ctu_11_raw,
            output_dir=os.path.join(args.output_base, "ctu13_11"),
            max_graphs=max_graphs,
            window_sec=60.0,
            stride_sec=10.0,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            chunk_rows=args.chunk_rows,
        )
        success = success and res

    if args.dataset == "ctu13_all":
        res = process_dataset(
            dataset_name="ctu13_all",
            raw_path=ctu_all_dir,
            output_dir=os.path.join(args.output_base, "ctu13_all"),
            max_graphs=max_graphs,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
            chunk_rows=args.chunk_rows,
        )
        success = success and res

    if success:
        print("[✓] All requested datasets preprocessed successfully!")
        print("    You can now execute Experiment 1 via:")
        print("    pixi run rq1-nf")
        print("    pixi run rq1-ctu")
    else:
        print("[!] Warning: One or more datasets encountered an error during preprocessing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
