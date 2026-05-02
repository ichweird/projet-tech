"""
evaluation/metrics.py
Personne 3 — Calcul du Prix de l'Anarchie et génération des graphiques
Dépend de : simulation.py (history), lp_solver.py (find_optimal_lp)
"""

import sys
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')  # sans display (pour tests en console)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─── 1) Calcul des métriques ─────────────────────────────────────────────────

def get_metrics(history: List[Dict[str, Any]],
                nash_cost: float,
                lp_cost: float,
                n_nodes_init: int) -> Dict[str, Any]:
    """
    Calcule les métriques de performance du réseau à partir de l'historique.

    Args:
        history    : sortie de simulation.run_simulation()
        nash_cost  : coût de la solution jeu (énergie consommée un round Nash)
        lp_cost    : coût de la solution LP optimale
        n_nodes_init : nombre de nœuds initial

    Retourne un dict avec :
        FND, LND, PoA, throughput, network_lifetime, convergence_round
    """
    if not history:
        return {}

    # FND — First Node Death
    fnd = next((h["round"] for h in history if h["fnd_detected"]), None)
    if fnd is None:
        # chercher manuellement le premier round où alive < n_nodes_init
        fnd = next((h["round"] for h in history if h["alive"] < n_nodes_init), None)

    # LND — Last Node Death
    lnd = history[-1]["round"]

    # Prix de l'Anarchie
    if lp_cost and lp_cost > 0:
        poa = round(nash_cost / lp_cost, 4)
    else:
        poa = None

    # Throughput total
    throughput = history[-1]["packets_received"] if history else 0

    # Convergence : round où le nombre de CH se stabilise (écart-type faible)
    convergence_round = _detect_convergence(history)

    return {
        "FND":               fnd,
        "LND":               lnd,
        "PoA":               poa,
        "throughput":        throughput,
        "network_lifetime":  lnd,
        "convergence_round": convergence_round,
        "nash_cost":         round(nash_cost, 6),
        "lp_cost":           round(lp_cost, 6),
    }


def _detect_convergence(history: List[Dict], window: int = 50) -> Optional[int]:
    """
    Détecte le round à partir duquel le nombre de CH se stabilise.
    Critère : écart-type du nb de CH sur les `window` derniers rounds < 1.0
    """
    n_ch_series = [h["n_ch"] for h in history]
    for i in range(window, len(n_ch_series)):
        window_data = n_ch_series[i - window:i]
        if np.std(window_data) < 1.0:
            return history[i]["round"]
    return None


# ─── 2) Génération des graphiques ────────────────────────────────────────────

def plot_all(history: List[Dict[str, Any]],
             nash_cost: Optional[float] = None,
             lp_cost: Optional[float] = None,
             title_prefix: str = "GTIACO") -> plt.Figure:
    """
    Génère une figure Matplotlib avec 4 sous-graphiques.
    Retourne l'objet Figure — la Personne 1 l'intègre dans la GUI avec FigureCanvasQTAgg.

    Graphiques :
        [0,0] Énergie résiduelle totale vs rounds
        [0,1] Nombre de CH vs rounds
        [1,0] Nœuds vivants vs rounds
        [1,1] Paquets reçus par BS (throughput cumulé)
    """
    rounds   = [h["round"]            for h in history]
    energy   = [h["energy_total"]     for h in history]
    n_ch     = [h["n_ch"]             for h in history]
    alive    = [h["alive"]            for h in history]
    packets  = [h["packets_received"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Performance du réseau — {title_prefix}", fontsize=14, fontweight='bold')

    # Palette couleurs
    C_MAIN  = "#1D9E75"   # teal
    C_CH    = "#7F77DD"   # purple
    C_ALIVE = "#378ADD"   # blue
    C_PKT   = "#D85A30"   # coral

    # ── Graphique 1 : Énergie ──────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(rounds, energy, color=C_MAIN, linewidth=1.5, label="Énergie résiduelle")
    ax.set_title("Énergie résiduelle totale (J)")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Joules")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # ── Graphique 2 : Nombre de CH ─────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(rounds, n_ch, color=C_CH, linewidth=1.5, label="Nombre de CH")
    ax.set_title("Nombre de cluster heads par round")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("CH actifs")
    ax.grid(True, alpha=0.3)
    # Ligne de convergence
    conv = _detect_convergence(history)
    if conv:
        ax.axvline(x=conv, color='orange', linestyle='--', linewidth=1, label=f"Convergence (~{conv})")
    ax.legend(fontsize=9)

    # ── Graphique 3 : Nœuds vivants ───────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(rounds, alive, color=C_ALIVE, linewidth=1.5, label="GTIACO")
    # Marquer FND
    fnd = next((h["round"] for h in history if h["fnd_detected"]), None)
    if fnd:
        ax.axvline(x=fnd, color='red', linestyle=':', linewidth=1.2, label=f"FND (round {fnd})")
    ax.set_title("Nœuds vivants dans le temps")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Nœuds vivants")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # ── Graphique 4 : Throughput ───────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(rounds, packets, color=C_PKT, linewidth=1.5, label="Paquets reçus (cumulés)")
    ax.set_title("Throughput cumulé (paquets reçus par BS)")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Paquets")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_comparison(history_gtiaco: List[Dict],
                    history_leach: Optional[List[Dict]] = None) -> plt.Figure:
    """
    Graphique de comparaison GTIACO vs LEACH (si données LEACH disponibles).
    Sinon affiche GTIACO seul avec annotation.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    rounds_g = [h["round"] for h in history_gtiaco]
    alive_g  = [h["alive"]  for h in history_gtiaco]
    ax.plot(rounds_g, alive_g, color="#1D9E75", linewidth=2, label="GTIACO (théorie des jeux)")

    if history_leach:
        rounds_l = [h["round"] for h in history_leach]
        alive_l  = [h["alive"]  for h in history_leach]
        ax.plot(rounds_l, alive_l, color="#D85A30", linewidth=2, linestyle='--', label="LEACH (référence)")

    ax.set_title("Comparaison protocoles — Nœuds vivants")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Nœuds vivants")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def print_metrics_report(metrics: Dict[str, Any]) -> None:
    """Affiche un rapport texte des métriques (pour la console / GUI)."""
    print("\n" + "=" * 60)
    print("MÉTRIQUES FINALES")
    print("=" * 60)
    if metrics.get("FND") is not None:
        print(f"  First Node Death (FND)   : round {metrics['FND']}")
    print(f"  Last Node Death  (LND)   : round {metrics['LND']}")
    print(f"  Durée de vie réseau      : {metrics['network_lifetime']} rounds")
    print(f"  Throughput total         : {metrics['throughput']:,} paquets")
    if metrics.get("convergence_round"):
        print(f"  Convergence équilibre    : ~round {metrics['convergence_round']}")
    if metrics.get("PoA"):
        print(f"\n  Coût Nash (jeu)          : {metrics['nash_cost']:.6f} J")
        print(f"  Coût LP  (optimal)       : {metrics['lp_cost']:.6f} J")
        print(f"  Prix de l'Anarchie (PoA) : {metrics['PoA']:.4f}")
        gap = (metrics['PoA'] - 1) * 100
        print(f"  → La solution jeu est {gap:.1f}% moins bonne que l'optimale centralisée")
    print("=" * 60)


# ─── Test standalone ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, '/mnt/user-data/uploads')
    from ClusterHeadsGTIACO import create_test_network
    from simulation import run_simulation
    from lp_solver import find_optimal_lp

    print("Test complet Personne 3")
    nodes_init = create_test_network(n_nodes=30, area_size=100)

    # Solution optimale LP
    ch_opt, lp_cost = find_optimal_lp(nodes_init, BS_x=50, BS_y=50)
    print(f"LP optimal: {len(ch_opt)} CH, coût = {lp_cost:.6f} J")

    # Simulation GTIACO
    history = run_simulation(nodes_init, BS_x=50, BS_y=50, n_rounds=300, verbose_every=50)

    # Coût Nash approx = énergie consommée au round 1
    nash_cost = nodes_init[0].E_init * len(nodes_init) - history[0]["energy_total"]

    # Métriques
    metrics = get_metrics(history, nash_cost, lp_cost, n_nodes_init=30)
    print_metrics_report(metrics)

    # Graphiques
    fig = plot_all(history, nash_cost, lp_cost)
    fig.savefig("/mnt/user-data/outputs/performance_charts.png", dpi=120, bbox_inches='tight')
    print("\nGraphiques sauvegardés dans outputs/performance_charts.png")
