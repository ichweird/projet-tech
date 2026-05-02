"""
============================================================
  interfaces.py — Interfaces pour l'intégration équipe
  
  Ce fichier définit les contrats entre les 4 tâches.
  Personne 2 remplace stub_nash_equilibrium, stub_pareto_optimum, stub_core_solution.
  Personne 3 remplace stub_lp_solution et stub_convergence_sim.
  Personne 4 remplace stub_aco_routing.
  
  Chaque fonction a la même signature d'entrée/sortie —
  il suffit de remplacer le corps.
============================================================
"""

import random


# ─────────────────────────────────────────────
#  PERSONNE 2 — Théorie des jeux
# ─────────────────────────────────────────────

def stub_nash_equilibrium(nodes: list, strategies: list, utilities: dict) -> dict:
    """
    Calcule l'équilibre de Nash.
    
    Args:
        nodes     : liste des IDs de nœuds (ex: ['N0', 'N1', ...])
        strategies: stratégies disponibles (ex: ['CH', 'CM', 'Sleep'])
        utilities : dict {node_id: float} — utilité de chaque nœud

    Returns:
        dict {node_id: strategy} — stratégie optimale de chaque nœud à l'équilibre
    
    TODO Personne 2: implémenter Nash pur/mixte (équations 14-19 du papier GTIACO)
    """
    result = {}
    for n in nodes:
        result[str(n)] = random.choice(strategies) if strategies else "CM"
    return result


def stub_pareto_optimum(nodes: list, strategies: list, utilities: dict) -> list:
    """
    Calcule l'ensemble des solutions Pareto-optimales.
    
    Returns:
        list de dict {node_id: strategy} — chaque élément est une solution Pareto
    
    TODO Personne 2: implémenter l'optimum de Pareto
    """
    pareto = []
    for _ in range(min(3, len(nodes))):
        combo = {str(n): random.choice(strategies) for n in nodes} if strategies else {}
        pareto.append(combo)
    return pareto


def stub_core_solution(nodes: list, strategies: list, utilities: dict) -> dict:
    """
    Calcule le cœur (core) du jeu coopératif.
    
    Returns:
        dict {node_id: strategy} — solution du cœur
    
    TODO Personne 2: implémenter le cœur pour version coopérative
    """
    return {str(n): random.choice(strategies) for n in nodes} if strategies else {}


# ─────────────────────────────────────────────
#  PERSONNE 3 — Optimisation centralisée + simulation
# ─────────────────────────────────────────────

def stub_lp_solution(nodes: list, clusters: list) -> dict:
    """
    Résout le problème d'optimisation centralisée par PL.
    
    Args:
        nodes   : liste des IDs de nœuds
        clusters: liste des cluster heads actuels

    Returns:
        dict avec clés:
          - 'obj_value'   : float — valeur objective optimale
          - 'method'      : str   — méthode utilisée
          - 'optimal_chs' : list  — CHs optimaux selon le LP
          - 'assignments' : dict  — {node_id: ch_id}
    
    TODO Personne 3: implémenter PL (scipy.optimize.linprog ou PuLP)
    """
    return {
        "obj_value":   round(random.uniform(50, 100), 2),
        "method":      "LP (stub)",
        "optimal_chs": clusters[:3] if clusters else [],
        "assignments": {}
    }


def stub_convergence_sim(model, n_rounds: int = 20) -> dict:
    """
    Simule l'évolution du réseau sur n_rounds rounds.
    
    Returns:
        dict avec clés:
          - 'energy_history'  : list[float] — énergie totale par round
          - 'ch_count_history': list[int]   — nb CHs par round
          - 'utility_history' : list[float] — utilité totale par round
    
    TODO Personne 3: implémenter la simulation temporelle (section 3.1 du papier)
    """
    e = [random.uniform(10, 20) for _ in range(n_rounds)]
    ch = [random.randint(2, 6) for _ in range(n_rounds)]
    u = [random.uniform(5, 15) for _ in range(n_rounds)]
    return {
        "energy_history":   e,
        "ch_count_history": ch,
        "utility_history":  u,
    }


# ─────────────────────────────────────────────
#  PERSONNE 4 — ACO inter-clusters
# ─────────────────────────────────────────────

def stub_aco_routing(cluster_heads: list, base_station: str) -> dict:
    """
    Calcule les routes multi-sauts entre CHs et la BS via I-ACO.
    
    Args:
        cluster_heads: liste des IDs des cluster heads
        base_station : ID de la base station (ex: 'BS')

    Returns:
        dict {ch_id: list[node_ids]} — route de chaque CH vers la BS
        Ex: {'N3': ['N3', 'N7', 'BS'], 'N12': ['N12', 'BS']}
    
    TODO Personne 4: implémenter I-ACO amélioré (section 3.4 du papier)
    """
    routes = {}
    for ch in cluster_heads:
        routes[ch] = [ch, base_station]
    return routes
