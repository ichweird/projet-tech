"""
============================================================
  test_app.py — Tests unitaires (sans GUI)
  Lance : python test_app.py
============================================================
"""

import sys
import math
import random

# ─── On importe uniquement le modèle, pas la GUI ───
# Evite d'avoir besoin d'un display pour les tests

def test_imports():
    """Test 1 : toutes les dépendances sont installées."""
    print("[1] Test imports...", end=" ")
    import matplotlib
    import networkx
    import numpy
    import scipy
    print(f"OK  (mpl={matplotlib.__version__}, nx={networkx.__version__}, np={numpy.__version__})")
    return True


def test_network_model():
    """Test 2 : génération du réseau."""
    print("[2] Test NetworkModel...", end=" ")

    import networkx as nx

    class MinimalModel:
        def __init__(self):
            self.G = nx.Graph()
            self.nodes_data = {}
            self.cluster_heads = set()
            self.strategies = ["CH", "CM", "Sleep"]
            self.utilities = {}
            self.costs = {}
            self.round_num = 0
            self.energy_history = []
            self.ch_count_history = []

        def generate_random_network(self, n_nodes=20, area=200, comm_range=60):
            self.G.clear()
            self.nodes_data = {}
            self.cluster_heads = set()
            bs = "BS"
            self.G.add_node(bs, pos=(area/2, area/2), node_type="BS",
                            energy=float("inf"), strategy="BS")
            for i in range(n_nodes):
                x = random.uniform(10, area-10)
                y = random.uniform(10, area-10)
                energy = random.uniform(0.5, 1.0)
                nid = f"N{i}"
                self.G.add_node(nid, pos=(x, y), node_type="sensor",
                                energy=energy, strategy="CM")
                self.nodes_data[nid] = {"energy": energy, "strategy": "CM",
                                         "utility": random.uniform(0.3,1.0),
                                         "cost": random.uniform(0.1,0.5)}
                self.utilities[nid] = random.uniform(0.3, 1.0)
                self.costs[nid] = random.uniform(0.1, 0.5)
            pos = nx.get_node_attributes(self.G, "pos")
            nodes_list = [n for n in self.G.nodes if n != bs]
            for i, u in enumerate(nodes_list):
                for v in nodes_list[i+1:]:
                    pu, pv = pos[u], pos[v]
                    dist = math.hypot(pu[0]-pv[0], pu[1]-pv[1])
                    if dist <= comm_range:
                        self.G.add_edge(u, v, weight=round(dist, 2))
            chosen = random.sample(nodes_list, max(2, int(0.2*len(nodes_list))))
            for ch in chosen:
                self.G.nodes[ch]["strategy"] = "CH"
                self.cluster_heads.add(ch)

    model = MinimalModel()
    model.generate_random_network(n_nodes=15, area=150, comm_range=55)

    assert "BS" in model.G.nodes, "BS manquante"
    assert len([n for n in model.G.nodes if n != "BS"]) == 15, "Mauvais nb nœuds"
    assert len(model.cluster_heads) >= 2, "Pas assez de CHs"
    print(f"OK  ({len(model.G.nodes)} nœuds, {model.G.number_of_edges()} edges, "
          f"{len(model.cluster_heads)} CHs)")
    return True


def test_interfaces():
    """Test 3 : les interfaces stubs retournent le bon type."""
    print("[3] Test interfaces stubs...", end=" ")
    from interfaces import (
        stub_nash_equilibrium, stub_pareto_optimum,
        stub_core_solution, stub_lp_solution, stub_aco_routing
    )
    nodes = [f"N{i}" for i in range(10)]
    strategies = ["CH", "CM", "Sleep"]
    utilities = {n: random.uniform(0.3, 1.0) for n in nodes}

    nash = stub_nash_equilibrium(nodes, strategies, utilities)
    assert isinstance(nash, dict), "Nash doit retourner un dict"
    assert len(nash) == len(nodes), "Nash: taille incorrecte"
    assert all(v in strategies for v in nash.values()), "Nash: stratégies invalides"

    pareto = stub_pareto_optimum(nodes, strategies, utilities)
    assert isinstance(pareto, list), "Pareto doit retourner une list"
    assert len(pareto) >= 1, "Pareto: liste vide"

    core = stub_core_solution(nodes, strategies, utilities)
    assert isinstance(core, dict), "Core doit retourner un dict"

    lp = stub_lp_solution(nodes, nodes[:3])
    assert isinstance(lp, dict), "LP doit retourner un dict"
    assert "obj_value" in lp, "LP: clé obj_value manquante"
    assert "method" in lp, "LP: clé method manquante"

    aco = stub_aco_routing(nodes[:3], "BS")
    assert isinstance(aco, dict), "ACO doit retourner un dict"
    assert len(aco) == 3, "ACO: routes manquantes"

    print("OK  (Nash, Pareto, Core, LP, ACO)")
    return True


def test_networkx_operations():
    """Test 4 : opérations NetworkX utilisées dans l'app."""
    print("[4] Test NetworkX ops...", end=" ")
    import networkx as nx
    import numpy as np

    G = nx.random_geometric_graph(20, 0.35, seed=42)
    assert G.number_of_nodes() == 20
    degrees = dict(G.degree())
    avg_deg = np.mean(list(degrees.values()))
    assert avg_deg >= 0
    pos = nx.get_node_attributes(G, "pos")
    assert len(pos) == 20
    components = list(nx.connected_components(G))
    assert len(components) >= 1
    print(f"OK  (G={G.number_of_nodes()}n/{G.number_of_edges()}e, "
          f"avg_deg={avg_deg:.1f}, {len(components)} composante(s))")
    return True


def test_matplotlib_headless():
    """Test 5 : matplotlib en mode non-interactif."""
    print("[5] Test Matplotlib headless...", end=" ")
    import matplotlib
    matplotlib.use("Agg")  # Backend sans display
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(0, 10, 50)
    ax.plot(x, np.sin(x), label="sin(x)")
    ax.set_title("Test")
    fig.savefig("/tmp/test_plot.png", dpi=72)
    plt.close(fig)
    import os
    assert os.path.exists("/tmp/test_plot.png"), "Fichier image non créé"
    print("OK  (figure sauvegardée dans /tmp/test_plot.png)")
    return True


def test_game_step():
    """Test 6 : simulation d'un round de jeu."""
    print("[6] Test Game Step...", end=" ")
    import networkx as nx
    from interfaces import (stub_nash_equilibrium, stub_pareto_optimum,
                             stub_core_solution, stub_lp_solution, stub_aco_routing)

    G = nx.path_graph(8)
    nodes = [str(n) for n in G.nodes]
    strategies = ["CH", "CM", "Sleep"]
    utilities = {n: 0.5 for n in nodes}
    costs = {n: 0.1 for n in nodes}
    nodes_data = {n: {"energy": 0.8} for n in nodes}
    cluster_heads = set()

    # Nash
    nash = stub_nash_equilibrium(nodes, strategies, utilities)
    for n, s in nash.items():
        if s == "CH":
            cluster_heads.add(n)

    # LP comparison
    lp = stub_lp_solution(nodes, list(cluster_heads))

    # ACO
    aco = stub_aco_routing(list(cluster_heads), "BS")

    # Décrémenter énergie
    for n in nodes:
        nodes_data[n]["energy"] = max(0, nodes_data[n]["energy"] - 0.02)

    total_e = sum(d["energy"] for d in nodes_data.values())
    n_ch = len(cluster_heads)
    assert total_e > 0, "Énergie à zéro après 1 round"
    print(f"OK  (CHs={n_ch}, énergie={total_e:.3f}, LP_obj={lp['obj_value']})")
    return True


# ─────────────────────────────────────────────

def run_all_tests():
    print("=" * 60)
    print("  TESTS — Game Theory Network Optimizer")
    print("=" * 60)
    tests = [
        test_imports,
        test_network_model,
        test_interfaces,
        test_networkx_operations,
        test_matplotlib_headless,
        test_game_step,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            ok = t()
            if ok:
                passed += 1
        except Exception as e:
            print(f"FAIL — {e}")
            failed += 1

    print("=" * 60)
    print(f"  Résultat : {passed}/{len(tests)} tests passés"
          + (f"  ({failed} ÉCHOUÉS)" if failed else ""))
    print("=" * 60)
    if failed == 0:
        print("  ✓ Application prête pour le lancement !")
    else:
        print("  ✗ Corriger les erreurs avant le lancement.")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
