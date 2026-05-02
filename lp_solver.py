"""
evaluation/lp_solver.py
Personne 3 — Solution centralisée optimale par programmation linéaire
Dépend de : ClusterHeadsGTIACO.py (Node)
"""

import math
import numpy as np
from typing import List, Tuple, Dict

# Import depuis le fichier de la Personne 2
import sys
sys.path.insert(0, '/mnt/user-data/uploads')
from ClusterHeadsGTIACO import Node

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("[lp_solver] PuLP non installé. Utilisation du solveur simplifié.")


# ─── Paramètres du modèle radio (first-order radio model) ───────────────────
E_ELEC = 50e-9    # 50 nJ/bit — énergie circuit émission/réception
E_AMP  = 100e-12  # 100 pJ/bit/m² — amplification
PACKET_SIZE = 4000  # bits par paquet


def _dist(n1: Node, n2_x: float, n2_y: float) -> float:
    return math.sqrt((n1.x - n2_x)**2 + (n1.y - n2_y)**2)


def energy_tx(bits: int, distance: float) -> float:
    """Énergie pour transmettre `bits` sur `distance` mètres."""
    return bits * E_ELEC + bits * E_AMP * distance**2


def energy_rx(bits: int) -> float:
    """Énergie pour recevoir `bits`."""
    return bits * E_ELEC


def find_optimal_lp(nodes: List[Node],
                    BS_x: float, BS_y: float,
                    comm_range: float = 50.0) -> Tuple[List[Node], float]:
    """
    Trouve la sélection de CH qui minimise l'énergie totale consommée.
    Utilise PuLP (programmation linéaire) si disponible,
    sinon utilise un algorithme glouton comme approximation.

    Retourne:
        (liste des nœuds CH optimaux, coût total optimal)
    """
    alive = [n for n in nodes if n.E_res > 0]
    if not alive:
        return [], 0.0

    if PULP_AVAILABLE:
        return _lp_pulp(alive, BS_x, BS_y, comm_range)
    else:
        return _greedy_optimal(alive, BS_x, BS_y, comm_range)


def _lp_pulp(nodes: List[Node], BS_x: float, BS_y: float,
             comm_range: float) -> Tuple[List[Node], float]:
    """Solveur PuLP — solution exacte."""
    import pulp

    prob = pulp.LpProblem("CH_selection_optimal", pulp.LpMinimize)

    # Variables binaires : x[i] = 1 si nœud i est CH
    x = {n.id: pulp.LpVariable(f"x_{n.id}", cat="Binary") for n in nodes}

    # Coût d'un nœud i en tant que CH :
    # il reçoit les données de ses membres + transmet vers BS
    def ch_cost(node):
        d_bs = _dist(node, BS_x, BS_y)
        members = [n for n in nodes if n.id != node.id and _dist(n, node.x, node.y) <= comm_range]
        rx_cost = energy_rx(PACKET_SIZE) * len(members)
        tx_cost = energy_tx(PACKET_SIZE, d_bs)
        return rx_cost + tx_cost

    costs = {n.id: ch_cost(n) for n in nodes}

    # Objectif : minimiser l'énergie totale consommée par les CH
    prob += pulp.lpSum(x[n.id] * costs[n.id] for n in nodes)

    # Contrainte 1 : au moins un CH dans le rayon de chaque nœud
    for node in nodes:
        neighbors_in_range = [n for n in nodes if _dist(node, n.x, n.y) <= comm_range]
        if neighbors_in_range:
            prob += pulp.lpSum(x[n.id] for n in neighbors_in_range) >= 1

    # Contrainte 2 : nombre de CH ≈ 5% du total (entre 1 et 10%)
    n = len(nodes)
    prob += pulp.lpSum(x[nd.id] for nd in nodes) >= max(1, int(0.03 * n))
    prob += pulp.lpSum(x[nd.id] for nd in nodes) <= max(2, int(0.10 * n))

    # Résolution (silencieuse)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    ch_list = [n for n in nodes if pulp.value(x[n.id]) is not None and pulp.value(x[n.id]) > 0.5]
    optimal_cost = pulp.value(prob.objective) or 0.0

    return ch_list, round(optimal_cost, 6)


def _greedy_optimal(nodes: List[Node], BS_x: float, BS_y: float,
                    comm_range: float) -> Tuple[List[Node], float]:
    """
    Approximation gloutonne si PuLP absent :
    sélectionne les CH par ordre décroissant de (énergie / distance_BS).
    """
    scored = sorted(nodes,
                    key=lambda n: (n.E_res / n.E_init) / max(1, _dist(n, BS_x, BS_y)),
                    reverse=True)

    target = max(1, int(0.05 * len(nodes)))
    ch_list = scored[:target]

    total_cost = 0.0
    for ch in ch_list:
        d_bs = _dist(ch, BS_x, BS_y)
        members = [n for n in nodes if n.id != ch.id and _dist(n, ch.x, ch.y) <= comm_range]
        total_cost += energy_rx(PACKET_SIZE) * len(members) + energy_tx(PACKET_SIZE, d_bs)

    return ch_list, round(total_cost, 6)


# ─── Test standalone ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, '/mnt/user-data/uploads')
    from ClusterHeadsGTIACO import create_test_network

    print("Test lp_solver.py")
    nodes = create_test_network(n_nodes=20, area_size=100)
    ch_list, cost = find_optimal_lp(nodes, BS_x=50, BS_y=50)
    print(f"  CH optimaux : {[n.id for n in ch_list]}")
    print(f"  Coût optimal : {cost:.6f} J")
