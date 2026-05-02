"""
evaluation/simulation.py
Personne 3 — Simulation temporelle du réseau round par round
Dépend de : ClusterHeadsGTIACO.py (Node, ClusterHeadSelectionGame)
"""

import math
import sys
from typing import List, Dict, Any
from copy import deepcopy

sys.path.insert(0, '/mnt/user-data/uploads')
from ClusterHeadsGTIACO import Node, ClusterHeadSelectionGame

# ─── Paramètres du modèle radio ──────────────────────────────────────────────
E_ELEC     = 50e-9    # 50 nJ/bit
E_AMP      = 100e-12  # 100 pJ/bit/m²
E_DA       = 5e-9     # 5 nJ/bit — agrégation de données (Data Aggregation)
PACKET_BITS = 4000    # bits par paquet


def _dist(x1, y1, x2, y2) -> float:
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def energy_tx(bits: int, distance: float) -> float:
    """Énergie de transmission."""
    return bits * E_ELEC + bits * E_AMP * distance**2


def energy_rx(bits: int) -> float:
    """Énergie de réception."""
    return bits * E_ELEC


def energy_aggregation(bits: int, n_members: int) -> float:
    """Énergie d'agrégation au niveau du CH."""
    return E_DA * bits * n_members


def run_simulation(nodes_init: List[Node],
                   BS_x: float = 50.0,
                   BS_y: float = 50.0,
                   n_rounds: int = 3000,
                   comm_range: float = 50.0,
                   verbose_every: int = 100) -> List[Dict[str, Any]]:
    """
    Simule le réseau WSN round par round.

    À chaque round :
    1. Sélection des CH (via théorie des jeux, Personne 2)
    2. Chaque nœud membre envoie ses données à son CH → consomme de l'énergie
    3. Chaque CH agrège + envoie vers la Base Station → consomme de l'énergie
    4. Les nœuds à énergie <= 0 sont marqués comme morts
    5. L'état est enregistré dans l'historique

    Retourne:
        List[dict] — un dict par round avec les clés :
        round, alive, n_ch, energy_total, packets_received, ch_ids, fnd_detected
    """
    # Copie profonde pour ne pas modifier le réseau original
    nodes = deepcopy(nodes_init)

    selector = ClusterHeadSelectionGame(
        communication_range=comm_range,
        default_reward_t=1.2,
        default_cost_c=0.5,
        omega=0.15,
        verbose=False   # silencieux pendant la simulation
    )

    history = []
    packets_cumul = 0
    fnd_round = None        # First Node Death
    first_alive = len(nodes)

    print(f"Simulation démarrée — {len(nodes)} nœuds, {n_rounds} rounds max")
    print("-" * 50)

    for r in range(n_rounds):
        alive_nodes = [n for n in nodes if n.E_res > 0]

        if not alive_nodes:
            break

        # ── 1) Sélection des CH (Personne 2) ───────────────────────────────
        cluster_heads = selector.select_cluster_heads_adaptive(
            nodes=alive_nodes,
            BS_x=BS_x,
            BS_y=BS_y,
            round_number=r
        )

        ch_ids = {ch.id for ch in cluster_heads}

        if not cluster_heads:
            # Fallback : le nœud avec le plus d'énergie devient CH
            best = max(alive_nodes, key=lambda n: n.E_res)
            cluster_heads = [best]
            ch_ids = {best.id}

        # ── 2) Association membres → CH le plus proche ────────────────────
        member_map: Dict[int, List[Node]] = {ch.id: [] for ch in cluster_heads}

        for node in alive_nodes:
            if node.id in ch_ids:
                continue
            # trouver le CH le plus proche dans le rayon
            nearest_ch = None
            min_d = float('inf')
            for ch in cluster_heads:
                d = _dist(node.x, node.y, ch.x, ch.y)
                if d <= comm_range and d < min_d:
                    min_d = d
                    nearest_ch = ch
            if nearest_ch:
                member_map[nearest_ch.id].append(node)
            else:
                # pas de CH à portée → transmet directement à BS
                d_bs = _dist(node.x, node.y, BS_x, BS_y)
                cost = energy_tx(PACKET_BITS, d_bs)
                node.E_res = max(0.0, node.E_res - cost)

        # ── 3) Consommation énergétique ────────────────────────────────────
        packets_this_round = 0

        for ch in cluster_heads:
            members = member_map.get(ch.id, [])

            # Membres envoient au CH
            for member in members:
                d = _dist(member.x, member.y, ch.x, ch.y)
                member.E_res = max(0.0, member.E_res - energy_tx(PACKET_BITS, d))

            # CH reçoit de ses membres
            ch.E_res = max(0.0, ch.E_res - energy_rx(PACKET_BITS) * len(members))

            # CH agrège les données
            ch.E_res = max(0.0, ch.E_res - energy_aggregation(PACKET_BITS, len(members)))

            # CH transmet vers BS
            d_bs = _dist(ch.x, ch.y, BS_x, BS_y)
            ch.E_res = max(0.0, ch.E_res - energy_tx(PACKET_BITS, d_bs))

            packets_this_round += len(members) + 1  # membres + CH lui-même

        packets_cumul += packets_this_round

        # ── 4) Détecter les morts ─────────────────────────────────────────
        alive_after = sum(1 for n in nodes if n.E_res > 0)

        if fnd_round is None and alive_after < first_alive:
            fnd_round = r
            print(f"  [Round {r}] Premier nœud mort (FND détecté)")

        # ── 5) Enregistrement de l'état ───────────────────────────────────
        energy_total = sum(n.E_res for n in nodes if n.E_res > 0)

        snapshot = {
            "round":            r,
            "alive":            alive_after,
            "n_ch":             len(cluster_heads),
            "energy_total":     round(energy_total, 6),
            "packets_received": packets_cumul,
            "ch_ids":           list(ch_ids),
            "fnd_detected":     (fnd_round == r),
        }
        history.append(snapshot)

        if r % verbose_every == 0 or alive_after == 0:
            print(f"  Round {r:5d} | vivants: {alive_after:3d} | CH: {len(cluster_heads):2d} "
                  f"| énergie: {energy_total:.4f} J | paquets: {packets_cumul:,}")

        if alive_after == 0:
            break

    print("-" * 50)
    print(f"Simulation terminée — {len(history)} rounds simulés")
    return history


# ─── Test standalone ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from ClusterHeadsGTIACO import create_test_network

    nodes = create_test_network(n_nodes=30, area_size=100)
    history = run_simulation(nodes, BS_x=50, BS_y=50, n_rounds=500, verbose_every=50)
    print(f"\nDernier round enregistré : {history[-1]}")
