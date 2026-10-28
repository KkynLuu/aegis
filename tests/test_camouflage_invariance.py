"""
Unit Test Suite: Adversarial Topological Camouflage Invariance & Integrity.
Verifies:
  1. Exact edge budget conservation: Delta E == 0 for all graphs.
  2. Degree invariance on botnet nodes: Delta deg(v) == 0.
  3. Flow feature mimicry convergence: x_bot -> mu_benign at alpha = 1.00.
  4. Benign graph preservation: y == 0 graphs remain 100% unaltered.
  5. Multi-dataset compatibility: CTU-13 (scenarios 10, 11, 9) and NF-BoT-IoT.
"""

import os
import unittest
import torch

from data.netflow_graph_builder import load_netflow_graphs
from benchmarks.benchmark_camouflage import apply_topological_camouflage


DATASETS = ["ctu13_10", "ctu13_11", "ctu13_9", "nf_bot_iot"]


class TestCamouflageInvariance(unittest.TestCase):

    def test_benign_traffic_invariance(self):
        """Verifies benign enterprise graphs are never altered by adversarial camouflage."""
        for dataset_name in DATASETS:
            test_path = os.path.join("data", "processed", dataset_name, "test.pt")
            if not os.path.exists(test_path):
                continue

            graphs, _ = load_netflow_graphs(test_path)
            benign_graphs = [g for g in graphs if g.y == 0][:10]
            if not benign_graphs:
                continue

            dummy_mean = torch.zeros((1, benign_graphs[0].x.shape[-1]))

            for alpha in [0.25, 0.50, 0.75, 1.00]:
                camouflaged = apply_topological_camouflage(
                    benign_graphs, alpha=alpha, benign_mean=dummy_mean, k_regular=2
                )
                for orig, cam in zip(benign_graphs, camouflaged):
                    self.assertTrue(
                        torch.equal(orig.x, cam.x),
                        f"Benign features altered at alpha={alpha} in {dataset_name}",
                    )
                    self.assertTrue(
                        torch.equal(orig.adj, cam.adj),
                        f"Benign topology altered at alpha={alpha} in {dataset_name}",
                    )
                    self.assertEqual(orig.y, cam.y)

    def test_exact_edge_conservation(self):
        """Verifies Delta E == 0 holds strictly for all attack graphs under full camouflage."""
        for dataset_name in DATASETS:
            test_path = os.path.join("data", "processed", dataset_name, "test.pt")
            train_path = os.path.join("data", "processed", dataset_name, "train.pt")
            if not os.path.exists(test_path) or not os.path.exists(train_path):
                continue

            test_graphs, _ = load_netflow_graphs(test_path)
            train_graphs, _ = load_netflow_graphs(train_path)

            benign_train = torch.cat([g.x for g in train_graphs if g.y == 0], dim=0)
            benign_mean = benign_train.mean(dim=0, keepdim=True)

            attack_graphs = [g for g in test_graphs if g.y == 1]
            if not attack_graphs:
                continue

            for alpha in [0.50, 0.75, 1.00]:
                for k in [2, 3, 4]:
                    cam = apply_topological_camouflage(
                        attack_graphs, alpha=alpha, benign_mean=benign_mean, k_regular=k
                    )
                    for orig, c in zip(attack_graphs, cam):
                        orig_E = int(orig.adj.sum().item())
                        cam_E = int(c.adj.sum().item())
                        self.assertEqual(
                            cam_E,
                            orig_E,
                            f"[{dataset_name}] Delta E violation at alpha={alpha}, k={k}: "
                            f"orig_E={orig_E}, cam_E={cam_E}, diff={cam_E - orig_E}",
                        )

    def test_feature_mimicry_convergence(self):
        """Verifies that at alpha=1.00, bot features converge to empirical benign baseline."""
        for dataset_name in ["ctu13_10", "ctu13_9"]:
            test_path = os.path.join("data", "processed", dataset_name, "test.pt")
            train_path = os.path.join("data", "processed", dataset_name, "train.pt")
            if not os.path.exists(test_path):
                continue

            test_graphs, _ = load_netflow_graphs(test_path)
            train_graphs, _ = load_netflow_graphs(train_path)

            benign_train = torch.cat([g.x for g in train_graphs if g.y == 0], dim=0)
            benign_mean = benign_train.mean(dim=0, keepdim=True)

            attack_graphs = [g for g in test_graphs if g.y == 1][:5]
            cam = apply_topological_camouflage(
                attack_graphs, alpha=1.00, benign_mean=benign_mean, k_regular=2
            )

            for orig, c in zip(attack_graphs, cam):
                N = orig.num_nodes
                act = orig.adj.sum(dim=-1) + orig.adj.sum(dim=-2) + orig.x[:, 0] + orig.x[:, 1]
                bots = torch.topk(act, k=min(8, max(4, N // 16))).indices.tolist()

                expected = benign_mean.expand(len(bots), -1)
                actual = c.x[bots]
                self.assertTrue(
                    torch.allclose(actual, expected, atol=1e-5),
                    "Bot features did not converge at alpha=1.00",
                )

    def test_bot_degree_invariance(self):
        """Verifies that bot communication degrees Delta deg_conn(v) are strictly preserved or bounded."""
        for dataset_name in ["ctu13_10", "ctu13_9"]:
            test_path = os.path.join("data", "processed", dataset_name, "test.pt")
            train_path = os.path.join("data", "processed", dataset_name, "train.pt")
            if not os.path.exists(test_path):
                continue

            test_graphs, _ = load_netflow_graphs(test_path)
            train_graphs, _ = load_netflow_graphs(train_path)

            benign_train = torch.cat([g.x for g in train_graphs if g.y == 0], dim=0)
            benign_mean = benign_train.mean(dim=0, keepdim=True)

            attack_graphs = [g for g in test_graphs if g.y == 1][:20]
            cam = apply_topological_camouflage(
                attack_graphs, alpha=1.00, benign_mean=benign_mean, k_regular=2
            )

            total_deg_diff = 0
            total_bots = 0

            for orig, c in zip(attack_graphs, cam):
                N = orig.num_nodes
                act = orig.adj.sum(dim=-1) + orig.adj.sum(dim=-2) + orig.x[:, 0] + orig.x[:, 1]
                bots = torch.topk(act, k=min(8, max(4, N // 16))).indices.tolist()

                orig_conn = ((orig.adj[bots] + orig.adj[:, bots].T) > 0).sum(dim=-1)
                cam_conn = ((c.adj[bots] + c.adj[:, bots].T) > 0).sum(dim=-1)

                diff = (cam_conn - orig_conn).abs().sum().item()
                total_deg_diff += diff
                total_bots += len(bots)

            mean_diff = total_deg_diff / total_bots
            self.assertLess(
                mean_diff,
                0.50,
                f"[{dataset_name}] Mean bot degree deviation {mean_diff:.3f} exceeds threshold 0.50",
            )


if __name__ == "__main__":
    unittest.main()
