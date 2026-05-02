"""
integration.py — Adaptateur Personne 3 → GUI (main_gui.py)
===========================================================
Ce fichier est le SEUL fichier que la Personne 1 doit intégrer dans main_gui.py.
Il traduit les données du format GUI (node IDs strings) vers le format
des algorithmes de la Personne 3 (objets Node de ClusterHeadsGTIACO.py).

Personne 1 doit :
  1. Copier ce fichier dans le même dossier que main_gui.py
  2. Faire 2 imports en haut de main_gui.py
  3. Remplacer 2 appels de fonctions stub dans main_gui.py
  (voir les commentaires TODO ci-dessous)
"""

import sys
import math

sys.path.insert(0, '.')   # ajuste si les fichiers sont dans un sous-dossier

# ── Import des modules Personne 3 ────────────────────────────────────────────
from ClusterHeadsGTIACO import Node
from lp_solver import find_optimal_lp
from simulation import run_simulation, E_ELEC, E_AMP, PACKET_BITS


# ─────────────────────────────────────────────────────────────────────────────
#  ADAPTATEUR 1 : stub_lp_solution  (remplace la version stub dans main_gui.py)
# ─────────────────────────────────────────────────────────────────────────────
def real_lp_solution(nodes_ids: list, clusters: list, graph=None) -> dict:
    """
    Remplace stub_lp_solution() dans main_gui.py.

    Args:
        nodes_ids : liste de strings ["N0","N1",...] — les IDs du graphe GUI
        clusters  : liste de strings — CHs actuels (ignoré, LP recalcule)
        graph     : le NetworkX graph du modèle (self.model.G)

    Returns:
        dict compatible avec ce que le GUI attend :
        {
          "obj_value"  : float  — coût LP optimal
          "method"     : str    — description de la méthode
          "optimal_chs": list   — IDs des CHs optimaux (strings)
          "assignments": dict   — {} (non utilisé par le GUI)
        }
    """
    if graph is None:
        return {"obj_value": 0.0, "method": "LP (erreur: graph manquant)",
                "optimal_chs": [], "assignments": {}}

    # Trouver la position de la base station
    bs_pos = graph.nodes.get("BS", {}).get("pos", (100, 100))
    BS_x, BS_y = bs_pos

    # Convertir les nœuds du GUI en objets Node de la Personne 2
    node_objects = []
    for nid in nodes_ids:
        if nid == "BS":
            continue
        nd = graph.nodes.get(nid, {})
        pos = nd.get("pos", (0, 0))
        energy = nd.get("energy", 0.5)

        node_obj = Node(
            node_id=nid,
            x=pos[0],
            y=pos[1],
            E_init=1.0,       # énergie initiale standard
            E_res=energy,     # énergie résiduelle actuelle du GUI
        )
        # Calculer les voisins (nœuds à portée)
        comm_range = 60.0
        for other_id in nodes_ids:
            if other_id == nid or other_id == "BS":
                continue
            other_pos = graph.nodes.get(other_id, {}).get("pos", (0, 0))
            d = math.hypot(pos[0] - other_pos[0], pos[1] - other_pos[1])
            if d <= comm_range:
                node_obj.neighbors.append(other_id)

        node_objects.append(node_obj)

    if not node_objects:
        return {"obj_value": 0.0, "method": "LP (aucun nœud)",
                "optimal_chs": [], "assignments": {}}

    # Appel au vrai solveur LP de la Personne 3
    comm_range = 60.0
    ch_list, optimal_cost = find_optimal_lp(
        nodes=node_objects,
        BS_x=BS_x,
        BS_y=BS_y,
        comm_range=comm_range
    )

    ch_ids = [ch.id for ch in ch_list]

    return {
        "obj_value":   round(optimal_cost, 6),
        "method":      "LP centralisé (PuLP)",
        "optimal_chs": ch_ids,
        "assignments": {}
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ADAPTATEUR 2 : stub_convergence_sim  (remplace la version stub)
# ─────────────────────────────────────────────────────────────────────────────
def real_convergence_sim(model, n_rounds: int = 20) -> dict:
    """
    Remplace stub_convergence_sim() dans main_gui.py.

    Args:
        model    : l'objet NetworkModel du GUI (self.model)
        n_rounds : nombre de rounds à simuler

    Returns:
        dict compatible avec l'interface :
        {
          "energy_history"  : list[float]
          "ch_count_history": list[int]
          "utility_history" : list[float]
        }
    
    NOTE : Cette fonction lit l'historique déjà accumulé dans le modèle GUI.
    Si l'historique est vide (simulation pas encore lancée), elle retourne 
    des listes vides.
    """
    energy_hist = list(model.energy_history)
    ch_hist     = list(model.ch_count_history)

    # Utilité = énergie normalisée par round (proxy)
    max_e = max(energy_hist) if energy_hist else 1.0
    utility_hist = [round(e / max_e, 4) for e in energy_hist]

    return {
        "energy_history":   energy_hist,
        "ch_count_history": ch_hist,
        "utility_history":  utility_hist,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  METRICS HELPER — pour afficher PoA et FND dans la zone Results
# ─────────────────────────────────────────────────────────────────────────────
def compute_poa(nash_cost: float, lp_cost: float) -> str:
    """Retourne une string lisible du Prix de l'Anarchie."""
    if lp_cost and lp_cost > 0 and nash_cost > 0:
        poa = nash_cost / lp_cost
        return f"{poa:.4f}  (+{(poa-1)*100:.1f}% vs optimal)"
    return "N/A"


def get_fnd(model) -> str:
    """Retourne le round du premier nœud mort (First Node Death)."""
    if not hasattr(model, '_initial_node_count'):
        return "N/A"
    for i, energy in enumerate(model.energy_history):
        # heuristique : si l'énergie chute brutalement
        if i > 0 and (model.energy_history[i-1] - energy) > 0.5:
            return str(i)
    return "N/A (réseau encore actif)"
