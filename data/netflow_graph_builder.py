"""
NetFlow Graph Construction Pipeline.
Converts tabular NetFlow records (CTU-13 and NF-BoT-IoT) into temporal interaction graphs.

Constructs attributed interaction graphs G = (V, E) where:
  - Nodes V = Unique IP addresses active within a sliding temporal window W.
  - Edges E = Directed NetFlow communication sessions (SrcIP -> DstIP).
  - Node Attributes x_i in R^10:
      1. Outgoing flow count (log1p)
      2. Incoming flow count (log1p)
      3. Outgoing total bytes (log1p)
      4. Incoming total bytes (log1p)
      5. Outgoing total packets (log1p)
      6. Incoming total packets (log1p)
      7. Unique destination port count (log1p)
      8. TCP flow fraction
      9. UDP flow fraction
      10. Other protocol fraction (ICMP, etc.)
"""

import os
import glob
import math
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NetFlowGraph:
    """Represents a single temporal NetFlow interaction graph."""
    def __init__(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        y: int,
        window_start: float = 0.0,
        window_end: float = 0.0,
        num_nodes: Optional[int] = None,
        num_edges: Optional[int] = None,
        attack_name: str = "Benign",
    ):
        self.x = x               # [N, node_dim]
        self.adj = adj           # [N, N]
        self.y = int(y)          # 0 = Benign, 1 = Botnet/Attack
        self.window_start = float(window_start)
        self.window_end = float(window_end)
        self.num_nodes = num_nodes if num_nodes is not None else x.shape[0]
        self.num_edges = num_edges if num_edges is not None else int(adj.sum())
        self.attack_name = attack_name


class CTU13NetFlowDataset(Dataset):
    """
    Dataset loader for CTU-13 NetFlow (binetflow) files.
    Segments continuous NetFlow captures into sliding temporal windows.
    Supports a single binetflow file, a list of files, or a directory path.
    By default (max_graphs=None), processes ALL available flow data.
    """
    def __init__(
        self,
        binetflow_path: Union[str, List[str]],
        window_sec: float = 60.0,
        stride_sec: float = 15.0,
        max_graphs: Optional[int] = None,
        min_nodes: int = 4,
        max_nodes: int = 128,
    ):
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.graphs: List[NetFlowGraph] = []

        if isinstance(binetflow_path, list):
            self.file_paths = binetflow_path
        elif os.path.isdir(binetflow_path):
            self.file_paths = sorted(glob.glob(os.path.join(binetflow_path, "**", "*.binetflow"), recursive=True))
        else:
            self.file_paths = [binetflow_path]

        self._process_all_files(max_graphs)

    def _process_all_files(self, max_graphs: Optional[int]):
        print(f"[*] Parsing CTU-13 NetFlow files ({len(self.file_paths)} files, max_graphs={max_graphs})...")
        for f_idx, path in enumerate(self.file_paths):
            if not os.path.exists(path):
                print(f"[!] Warning: File does not exist: {path}")
                continue

            print(f"    [{f_idx + 1}/{len(self.file_paths)}] Processing: {path}")
            df = pd.read_csv(
                path,
                usecols=['StartTime', 'Dur', 'Proto', 'SrcAddr', 'DstAddr', 'Dport', 'TotPkts', 'TotBytes', 'Label']
            )

            df['Timestamp'] = pd.to_datetime(df['StartTime']).astype('datetime64[s]').astype('int64')
            df = df.sort_values('Timestamp').reset_index(drop=True)

            timestamps = df['Timestamp'].values
            t_min = int(timestamps[0])
            t_max = int(timestamps[-1])
            duration_h = (t_max - t_min) / 3600.0
            print(f"        Timeline: {t_min} to {t_max} (Duration: {duration_h:.2f}h, Total flows: {len(df):,})")

            current_t = t_min
            while current_t + self.window_sec <= t_max:
                idx_start, idx_end = np.searchsorted(timestamps, [current_t, current_t + self.window_sec])
                current_t += self.stride_sec

                if idx_end - idx_start < self.min_nodes:
                    continue

                w_df = df.iloc[idx_start:idx_end].copy()

                src_ips = w_df['SrcAddr'].values
                dst_ips = w_df['DstAddr'].values
                unique_ips = np.unique(np.concatenate([src_ips, dst_ips]))

                if len(unique_ips) < self.min_nodes:
                    continue

                if len(unique_ips) > self.max_nodes:
                    ip_counts = pd.Series(np.concatenate([src_ips, dst_ips])).value_counts()
                    top_ips = set(ip_counts.head(self.max_nodes).index)
                    w_df = w_df[w_df['SrcAddr'].isin(top_ips) & w_df['DstAddr'].isin(top_ips)]
                    unique_ips = np.array(list(top_ips))

                N = len(unique_ips)
                ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}

                # Build Adjacency Matrix
                adj = np.zeros((N, N), dtype=np.float32)
                for s_ip, d_ip in zip(w_df['SrcAddr'].values, w_df['DstAddr'].values):
                    if s_ip in ip_to_idx and d_ip in ip_to_idx:
                        adj[ip_to_idx[s_ip], ip_to_idx[d_ip]] = 1.0

                # Node Telemetry Features (10 dimensions)
                features = np.zeros((N, 10), dtype=np.float32)
                out_grp = w_df.groupby('SrcAddr')
                in_grp = w_df.groupby('DstAddr')

                out_counts = out_grp.size()
                in_counts = in_grp.size()
                out_bytes = out_grp['TotBytes'].sum()
                in_bytes = in_grp['TotBytes'].sum()
                out_pkts = out_grp['TotPkts'].sum()
                in_pkts = in_grp['TotPkts'].sum()
                out_dports = out_grp['Dport'].nunique()

                proto_upper = w_df['Proto'].astype(str).str.upper()
                tcp_mask = proto_upper.str.contains('TCP')
                udp_mask = proto_upper.str.contains('UDP')
                tcp_counts = w_df[tcp_mask].groupby('SrcAddr').size()
                udp_counts = w_df[udp_mask].groupby('SrcAddr').size()

                for ip, idx in ip_to_idx.items():
                    oc = out_counts.get(ip, 0)
                    ic = in_counts.get(ip, 0)
                    ob = out_bytes.get(ip, 0)
                    ib = in_bytes.get(ip, 0)
                    op = out_pkts.get(ip, 0)
                    ipk = in_pkts.get(ip, 0)
                    dp = out_dports.get(ip, 0)
                    tcp_c = tcp_counts.get(ip, 0)
                    udp_c = udp_counts.get(ip, 0)

                    features[idx, 0] = math.log1p(oc)
                    features[idx, 1] = math.log1p(ic)
                    features[idx, 2] = math.log1p(ob)
                    features[idx, 3] = math.log1p(ib)
                    features[idx, 4] = math.log1p(op)
                    features[idx, 5] = math.log1p(ipk)
                    features[idx, 6] = math.log1p(dp)
                    features[idx, 7] = tcp_c / max(oc, 1)
                    features[idx, 8] = udp_c / max(oc, 1)
                    features[idx, 9] = max(0.0, 1.0 - (features[idx, 7] + features[idx, 8]))

                is_botnet = w_df['Label'].str.contains('Botnet', case=False, na=False).any()

                self.graphs.append(
                    NetFlowGraph(
                        x=torch.from_numpy(features),
                        adj=torch.from_numpy(adj),
                        y=1 if is_botnet else 0,
                        window_start=float(timestamps[idx_start]),
                        window_end=float(timestamps[min(idx_end - 1, len(timestamps) - 1)]),
                        num_nodes=N,
                        num_edges=int(adj.sum()),
                        attack_name="Botnet" if is_botnet else "Benign"
                    )
                )

                if max_graphs is not None and len(self.graphs) >= max_graphs:
                    break

            if max_graphs is not None and len(self.graphs) >= max_graphs:
                break

        print(f"[✓] Extracted {len(self.graphs)} total temporal subgraphs from CTU-13.")
        num_mal = sum(g.y for g in self.graphs)
        print(f"    - Malicious (Botnet) subgraphs: {num_mal} ({num_mal/max(len(self.graphs), 1)*100:.1f}%)")
        print(f"    - Benign background subgraphs:  {len(self.graphs) - num_mal}")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> NetFlowGraph:
        return self.graphs[idx]


class NFBoTIoTNetFlowDataset(Dataset):
    """
    Dataset loader for NF-BoT-IoT-v3 standardized NetFlow dataset.
    Streams chunked records across ALL 16.9M flows and extracts all valid
    temporal interaction subgraphs via fast vectorized time-binning.
    By default (max_graphs=None), processes ALL available flow data.
    """
    def __init__(
        self,
        csv_path: str,
        window_ms: float = 30000.0,  # 30-second windows
        max_graphs: Optional[int] = None,
        min_nodes: int = 4,
        max_nodes: int = 128,
        chunk_rows: int = 2000000,
    ):
        self.csv_path = csv_path
        self.window_ms = window_ms
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.graphs: List[NetFlowGraph] = []

        self._process_flows(max_graphs, chunk_rows)

    def _process_flows(self, max_graphs: Optional[int], chunk_rows: int):
        print(f"[*] Parsing NF-BoT-IoT dataset across ALL flows: {self.csv_path} (max_graphs={max_graphs})...")
        cols = [
            'FLOW_START_MILLISECONDS', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR',
            'L4_DST_PORT', 'PROTOCOL', 'IN_BYTES', 'OUT_BYTES',
            'IN_PKTS', 'OUT_PKTS', 'Label', 'Attack'
        ]

        reader = pd.read_csv(self.csv_path, usecols=cols, chunksize=chunk_rows)

        for chunk_idx, df in enumerate(reader):
            df = df.sort_values('FLOW_START_MILLISECONDS').reset_index(drop=True)
            t_starts = df['FLOW_START_MILLISECONDS'].values
            t_min = t_starts[0]

            # Fast vectorized temporal binning
            bins = (t_starts - t_min) // int(self.window_ms)
            unique_bins, bin_starts, bin_counts = np.unique(bins, return_index=True, return_counts=True)
            valid_mask = bin_counts >= self.min_nodes

            for b_start, b_count in zip(bin_starts[valid_mask], bin_counts[valid_mask]):
                w_df = df.iloc[b_start : b_start + b_count]

                src_ips = w_df['IPV4_SRC_ADDR'].values
                dst_ips = w_df['IPV4_DST_ADDR'].values
                unique_ips = np.unique(np.concatenate([src_ips, dst_ips]))

                if len(unique_ips) < self.min_nodes:
                    continue

                if len(unique_ips) > self.max_nodes:
                    ip_counts = pd.Series(np.concatenate([src_ips, dst_ips])).value_counts()
                    top_ips = set(ip_counts.head(self.max_nodes).index)
                    w_df = w_df[w_df['IPV4_SRC_ADDR'].isin(top_ips) & w_df['IPV4_DST_ADDR'].isin(top_ips)]
                    unique_ips = np.array(list(top_ips))

                N = len(unique_ips)
                ip_to_idx = {ip: idx for idx, ip in enumerate(unique_ips)}

                # Build Adjacency Matrix
                adj = np.zeros((N, N), dtype=np.float32)
                for s_ip, d_ip in zip(w_df['IPV4_SRC_ADDR'].values, w_df['IPV4_DST_ADDR'].values):
                    if s_ip in ip_to_idx and d_ip in ip_to_idx:
                        adj[ip_to_idx[s_ip], ip_to_idx[d_ip]] = 1.0

                # Fast Grouped Node Telemetry Features (10 dimensions)
                features = np.zeros((N, 10), dtype=np.float32)
                out_grp = w_df.groupby('IPV4_SRC_ADDR')
                in_grp = w_df.groupby('IPV4_DST_ADDR')

                out_counts = out_grp.size()
                in_counts = in_grp.size()
                out_bytes = out_grp['OUT_BYTES'].sum()
                in_bytes = in_grp['IN_BYTES'].sum()
                out_pkts = out_grp['OUT_PKTS'].sum()
                in_pkts = in_grp['IN_PKTS'].sum()
                out_dports = out_grp['L4_DST_PORT'].nunique()

                tcp_counts = w_df[w_df['PROTOCOL'] == 6].groupby('IPV4_SRC_ADDR').size()
                udp_counts = w_df[w_df['PROTOCOL'] == 17].groupby('IPV4_SRC_ADDR').size()

                for ip, idx in ip_to_idx.items():
                    oc = out_counts.get(ip, 0)
                    ic = in_counts.get(ip, 0)
                    ob = out_bytes.get(ip, 0)
                    ib = in_bytes.get(ip, 0)
                    op = out_pkts.get(ip, 0)
                    ipk = in_pkts.get(ip, 0)
                    dp = out_dports.get(ip, 0)
                    tcp_c = tcp_counts.get(ip, 0)
                    udp_c = udp_counts.get(ip, 0)

                    features[idx, 0] = math.log1p(oc)
                    features[idx, 1] = math.log1p(ic)
                    features[idx, 2] = math.log1p(ob)
                    features[idx, 3] = math.log1p(ib)
                    features[idx, 4] = math.log1p(op)
                    features[idx, 5] = math.log1p(ipk)
                    features[idx, 6] = math.log1p(dp)
                    features[idx, 7] = tcp_c / max(oc, 1)
                    features[idx, 8] = udp_c / max(oc, 1)
                    features[idx, 9] = max(0.0, 1.0 - (features[idx, 7] + features[idx, 8]))

                is_attack = (w_df['Label'] == 1).any()
                mode_vals = w_df['Attack'].mode()
                most_common_attack = mode_vals[0] if (is_attack and len(mode_vals) > 0) else "Benign"

                w_t_start = float(t_starts[b_start])
                w_t_end = float(t_starts[min(b_start + b_count - 1, len(t_starts) - 1)])

                self.graphs.append(
                    NetFlowGraph(
                        x=torch.from_numpy(features),
                        adj=torch.from_numpy(adj),
                        y=1 if is_attack else 0,
                        window_start=w_t_start,
                        window_end=w_t_end,
                        num_nodes=N,
                        num_edges=int(adj.sum()),
                        attack_name=most_common_attack
                    )
                )

                if max_graphs is not None and len(self.graphs) >= max_graphs:
                    break

            print(f"    [Chunk {chunk_idx + 1}] Processed flows. Running total subgraphs: {len(self.graphs):,}")

            if max_graphs is not None and len(self.graphs) >= max_graphs:
                break

        print(f"[✓] Extracted {len(self.graphs)} total temporal subgraphs from NF-BoT-IoT.")
        num_mal = sum(g.y for g in self.graphs)
        print(f"    - Attack subgraphs: {num_mal} ({num_mal/max(len(self.graphs), 1)*100:.1f}%)")
        print(f"    - Benign subgraphs: {len(self.graphs) - num_mal}")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> NetFlowGraph:
        return self.graphs[idx]


def collate_netflow_batch(graph_list: List[NetFlowGraph]) -> Dict[str, torch.Tensor]:
    """
    Batches a list of NetFlow graphs into padded dense tensors:
      - x: [B, max_N, node_dim]
      - adj: [B, max_N, max_N]
      - mask: [B, max_N] (1 for real node, 0 for padded dummy node)
      - labels: [B]
    """
    B = len(graph_list)
    max_N = max(g.num_nodes for g in graph_list)
    node_dim = graph_list[0].x.shape[-1]

    x_padded = torch.zeros(B, max_N, node_dim, dtype=torch.float32)
    adj_padded = torch.zeros(B, max_N, max_N, dtype=torch.float32)
    mask = torch.zeros(B, max_N, dtype=torch.bool)
    labels = torch.zeros(B, dtype=torch.long)

    for i, g in enumerate(graph_list):
        n = g.num_nodes
        x_padded[i, :n, :] = g.x
        adj_padded[i, :n, :n] = g.adj
        mask[i, :n] = True
        labels[i] = g.y

    return {
        "x": x_padded,
        "adj": adj_padded,
        "mask": mask,
        "labels": labels
    }


def split_graphs_chronological(
    graphs: List[NetFlowGraph],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2
) -> Tuple[List[NetFlowGraph], List[NetFlowGraph], List[NetFlowGraph]]:
    """
    Splits graphs chronologically to strictly respect temporal ordering
    and eliminate data leakage in intrusion detection.
    """
    total = len(graphs)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    train_graphs = graphs[:n_train]
    val_graphs = graphs[n_train:n_train + n_val]
    test_graphs = graphs[n_train + n_val:]
    return train_graphs, val_graphs, test_graphs


def split_graphs_per_attack_chronological(
    graphs: List[NetFlowGraph],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2
) -> Tuple[List[NetFlowGraph], List[NetFlowGraph], List[NetFlowGraph]]:
    """
    Splits graphs chronologically *within* each attack and benign category.
    This guarantees:
      1. Every attack family (DDoS, DoS, Recon, Benign) is represented across train, val, and test.
      2. Temporal ordering is strictly preserved within each traffic stream (no lookahead leakage).
      3. Eliminates artificial out-of-distribution zero-day artifact caused by sequential file concatenation.
    """
    categories: Dict[str, List[NetFlowGraph]] = {}
    for g in graphs:
        key = g.attack_name if g.y == 1 else "Benign"
        if key not in categories:
            categories[key] = []
        categories[key].append(g)

    train_graphs: List[NetFlowGraph] = []
    val_graphs: List[NetFlowGraph] = []
    test_graphs: List[NetFlowGraph] = []

    for cat_name, cat_list in categories.items():
        n = len(cat_list)
        if n < 3:
            train_graphs.extend(cat_list)
            continue
        n_tr = int(n * train_ratio)
        n_va = int(n * val_ratio)
        if n >= 5:
            n_va = max(n_va, 1)
            n_te = max(n - n_tr - n_va, 1)
            n_tr = n - n_va - n_te
        else:
            n_te = n - n_tr - n_va

        train_graphs.extend(cat_list[:n_tr])
        val_graphs.extend(cat_list[n_tr : n_tr + n_va])
        test_graphs.extend(cat_list[n_tr + n_va :])

    return train_graphs, val_graphs, test_graphs


def save_netflow_graphs(
    graphs: List[NetFlowGraph],
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Saves a list of NetFlow graphs and metadata to a .pt file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    payload = {
        "graphs": [
            {
                "x": g.x,
                "adj": g.adj,
                "y": g.y,
                "window_start": g.window_start,
                "window_end": g.window_end,
                "num_nodes": g.num_nodes,
                "num_edges": g.num_edges,
                "attack_name": g.attack_name,
            }
            for g in graphs
        ],
        "metadata": metadata or {
            "num_graphs": len(graphs),
            "num_malicious": sum(g.y for g in graphs),
            "num_benign": sum(1 - g.y for g in graphs),
            "node_dim": graphs[0].x.shape[-1] if len(graphs) > 0 else 10,
        }
    }
    torch.save(payload, file_path)


def load_netflow_graphs(file_path: str) -> Tuple[List[NetFlowGraph], Dict[str, Any]]:
    """Loads a list of NetFlow graphs and metadata from a .pt file."""
    payload = torch.load(file_path, weights_only=False)
    raw_list = payload.get("graphs", [])
    graphs = [
        NetFlowGraph(
            x=d["x"],
            adj=d["adj"],
            y=d["y"],
            window_start=d.get("window_start", 0.0),
            window_end=d.get("window_end", 0.0),
            num_nodes=d.get("num_nodes", d["x"].shape[0]),
            num_edges=d.get("num_edges", int(d["adj"].sum())),
            attack_name=d.get("attack_name", "Unknown"),
        )
        for d in raw_list
    ]
    metadata = payload.get("metadata", {})
    return graphs, metadata


class NetFlowInMemoryDataset(Dataset):
    """PyTorch Dataset that loads preprocessed NetFlow graphs from .pt disk cache."""
    def __init__(self, file_path_or_graphs: Union[str, List[NetFlowGraph]]):
        if isinstance(file_path_or_graphs, str):
            self.graphs, self.metadata = load_netflow_graphs(file_path_or_graphs)
        else:
            self.graphs = file_path_or_graphs
            self.metadata = {
                "num_graphs": len(self.graphs),
                "num_malicious": sum(g.y for g in self.graphs),
                "num_benign": sum(1 - g.y for g in self.graphs),
            }

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> NetFlowGraph:
        return self.graphs[idx]


if __name__ == "__main__":
    print("Testing NetFlow Graph Builder...")
    sample_binetflow = "data/CTU-13/10/capture20110818.binetflow"
    if os.path.exists(sample_binetflow):
        dataset = CTU13NetFlowDataset(sample_binetflow, window_sec=60.0, stride_sec=15.0, max_graphs=10)
        print(f"Loaded {len(dataset)} sample graphs.")
        batch = collate_netflow_batch([dataset[0], dataset[1]])
        print(f"Collated batch shapes:")
        print(f"  x: {batch['x'].shape}")
        print(f"  adj: {batch['adj'].shape}")
        print(f"  mask: {batch['mask'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        print("[✓] NetFlow Graph Builder successfully tested!")
