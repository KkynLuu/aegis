#!/usr/bin/env python3
"""
Inserts Table IV (Multi-Degree Regularity Sweep) into Section V-D of docs/manuscript_draft_tifs.md
and cleanly renumbers subsequent tables IV-VIII to V-IX.
"""

import os
import sys

MANUSCRIPT_PATH = "docs/manuscript_draft_tifs.md"

def update():
    with open(MANUSCRIPT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Renumber existing Tables from IV-VIII to V-IX in reverse order
    replacements = [
        ("**TABLE VIII:", "**TABLE IX:"),
        ("Table VIII", "Table IX"),
        ("**TABLE VII:", "**TABLE VIII:"),
        ("Table VII", "Table VIII"),
        ("**TABLE VI:", "**TABLE VII:"),
        ("Table VI", "Table VII"),
        ("**TABLE V:", "**TABLE VI:"),
        ("Table V", "Table VI"),
        ("**TABLE IV:", "**TABLE V:"),
        ("Table IV", "Table V"),
    ]

    for old_s, new_s in replacements:
        text = text.replace(old_s, new_s)
    print("[✓] Renumbered existing tables IV-VIII to V-IX.")

    # 2. Get Table IV Markdown
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts.update_manuscript_regularity import generate_regularity_table
    table4_md = generate_regularity_table()

    # 3. Build the Section V-D addition
    header = "\n\n#### Empirical Sweep Across Regularity Degrees $k \\in \\{2, 3, 4, 6\\}\n\n"
    intro = "To test whether the adversary can defeat spectral detection by varying overlay density, Table IV evaluates all models across regular chordal topologies of degree $k \\in \\{2, 3, 4, 6\\}$ (simple cycle, cubic Möbius graph, 4-regular chordal ring, and 6-regular dense overlay) under full feature mimicry ($\\alpha = 1.00$).\n\n"
    fig_ref = "*(Degradation curves across regularity degrees $k$ are illustrated in Figure 2, `benchmarks/figure_regularity_sweep_ctu13_10.png` / `benchmarks/figure_regularity_sweep_ctu13_10.pdf`.)*\n\n"

    analysis = """
*Theoretical Insights from the Regularity Sweep:*
1. **Degree-Invariant 1-WL Representation Collapse:** Across all chordal degrees $k \\in \\{2, 3, 4, 6\\}$, GIN ($53.37\\% \\to 59.52\\%$) and RWPE-GIN ($48.20\\% \\to 49.33\\%$) remain statistically paralyzed near chance ($q < 0.01^{**}$). Because the multiset of neighbor states evaluates to identically distributed representations across all $k$ regular nodes, unnormalized spatial summation provides zero structural differentiation regardless of chord length or degree.
2. **Unbroken Dirichlet Defense Floor in QUASAR:** Across degrees $k \\in \\{2, 3, 4, 6\\}$, QUASAR maintains an unwavering defense floor ($80.49\\% \\to 80.70\\% \\to 80.56\\% \\to 80.81\\%$, mean $80.64\\%$). The non-dissipative unitary propagator $\\mathbf{U}(t) = \\exp(-i \\mathcal{H} t)$ preserves node discriminability through coherent phase interference, entirely immune to regular chordal obfuscation.
3. **The GCN Environment Artifact:** On CTU-13 Scenario 10, GCN sustains $\\sim 90.2\\%$ across all degrees $k$ because in this specific sparse network, any dense cluster among the 8 bots creates a contrast against the acyclic tree background. However, as established in Table II and Table III, in realistic topologies containing natural community structures (NF-BoT-IoT and CTU-13 Scenario 11), GCN collapses to $41.92\\%$ and $2.38\\%$, demonstrating that GCN is an environment-fragile heuristic rather than a robust evasion defense."""

    v_d_addition = header + intro + fig_ref + table4_md + "\n" + analysis

    marker = "### E. Multi-Hop Depth Scaling"
    if marker in text:
        parts = text.split(marker)
        v_d_body = parts[0].rstrip()
        if v_d_body.endswith("---"):
            v_d_body = v_d_body[:-3].rstrip()
        new_text = v_d_body + v_d_addition + "\n\n---\n\n### E. Multi-Hop Depth Scaling" + parts[1]
        with open(MANUSCRIPT_PATH, "w", encoding="utf-8") as f:
            f.write(new_text)
        print("[✓] Successfully inserted Table IV and Subsection into Section V-D.")
    else:
        print("[!] Error: Section V-E marker not found.")
        sys.exit(1)

if __name__ == "__main__":
    update()
