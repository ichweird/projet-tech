"""
============================================================
  Game Theory Network Optimizer — Tâche 1: GUI + Visualisation
  Palette: Dégradés de Bleu Nuit
  Technologies: Tkinter + Matplotlib + NetworkX + NumPy
============================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import random
import math
from collections import defaultdict

# ─────────────────────────────────────────────
#  PALETTE BLEU NUIT
# ─────────────────────────────────────────────
C = {
    "bg_dark":      "#050D1A",   # fond principal
    "bg_panel":     "#0A1628",   # panneaux
    "bg_widget":    "#0E1F38",   # widgets
    "bg_hover":     "#142847",   # hover
    "border":       "#1A3A6B",   # bordures
    "accent1":      "#1E5EFF",   # bleu vif principal
    "accent2":      "#0096FF",   # cyan bleu
    "accent3":      "#00D4FF",   # cyan clair
    "cluster_h":    "#FF6B35",   # cluster head (orange pour contraste)
    "node_normal":  "#1E5EFF",
    "node_hover":   "#0096FF",
    "text_primary": "#E8F4FD",
    "text_second":  "#7BA7D0",
    "text_dim":     "#3A6A9A",
    "success":      "#00E676",
    "warning":      "#FFB74D",
    "danger":       "#FF5252",
    "grid":         "#0A1E3A",
    "nash_color":   "#7C4DFF",
    "pareto_color": "#00E5FF",
    "core_color":   "#69F0AE",
}

# ─────────────────────────────────────────────
#  MODÈLE DE DONNÉES & STUB INTERFACES
# ─────────────────────────────────────────────

def stub_nash_equilibrium(nodes, strategies, utilities):
    """Interface stub — sera remplacé par la Personne 2."""
    result = {}
    for n in nodes:
        result[str(n)] = random.choice(strategies) if strategies else "S1"
    return result

def stub_pareto_optimum(nodes, strategies, utilities):
    """Interface stub — sera remplacé par la Personne 2."""
    pareto = []
    for _ in range(min(3, len(nodes))):
        combo = {str(n): random.choice(strategies) for n in nodes} if strategies else {}
        pareto.append(combo)
    return pareto

def stub_core_solution(nodes, strategies, utilities):
    """Interface stub — sera remplacé par la Personne 2."""
    return {str(n): random.choice(strategies) for n in nodes} if strategies else {}

def stub_lp_solution(nodes, clusters):
    """Interface stub — sera remplacé par la Personne 3."""
    return {"obj_value": round(random.uniform(50, 100), 2), "method": "LP (stub)"}

def stub_aco_routing(cluster_heads, base_station):
    """Interface stub — sera remplacé par la Personne 4."""
    routes = {}
    for ch in cluster_heads:
        routes[ch] = [ch, "relay", base_station]
    return routes


class NetworkModel:
    """Modèle du réseau de capteurs."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.G = nx.Graph()
        self.nodes_data = {}
        self.cluster_heads = set()
        self.base_station = "BS"
        self.strategies = ["CH", "CM", "Sleep"]
        self.utilities = {}
        self.costs = {}
        self.nash_result = {}
        self.pareto_result = []
        self.core_result = {}
        self.lp_result = {}
        self.aco_routes = {}
        self.energy_history = []
        self.ch_count_history = []
        self.round_num = 0

    def generate_random_network(self, n_nodes=20, area=200, comm_range=60):
        self.reset()
        positions = {}
        # Base station au centre
        positions[self.base_station] = (area / 2, area / 2)
        self.G.add_node(self.base_station, pos=positions[self.base_station],
                        node_type="BS", energy=float("inf"), strategy="BS")

        # Nœuds capteurs
        for i in range(n_nodes):
            x = random.uniform(10, area - 10)
            y = random.uniform(10, area - 10)
            energy = random.uniform(0.5, 1.0)
            strategy = random.choice(self.strategies)
            node_id = f"N{i}"
            positions[node_id] = (x, y)
            self.G.add_node(node_id, pos=(x, y), node_type="sensor",
                            energy=energy, strategy=strategy)
            self.nodes_data[node_id] = {
                "energy": energy,
                "strategy": strategy,
                "utility": random.uniform(0.3, 1.0),
                "cost": random.uniform(0.1, 0.5),
            }
            self.utilities[node_id] = random.uniform(0.3, 1.0)
            self.costs[node_id] = random.uniform(0.1, 0.5)

        # Connexions selon la portée
        nodes_list = [n for n in self.G.nodes if n != self.base_station]
        for i, u in enumerate(nodes_list):
            for v in nodes_list[i + 1:]:
                pu, pv = positions[u], positions[v]
                dist = math.hypot(pu[0] - pv[0], pu[1] - pv[1])
                if dist <= comm_range:
                    self.G.add_edge(u, v, weight=round(dist, 2))

        # Cluster heads initiaux (aléatoires 20%)
        sensor_nodes = [n for n in self.G.nodes if self.G.nodes[n].get("node_type") == "sensor"]
        n_ch = max(2, int(0.2 * len(sensor_nodes)))
        chosen_ch = random.sample(sensor_nodes, n_ch)
        for ch in chosen_ch:
            self.G.nodes[ch]["strategy"] = "CH"
            self.cluster_heads.add(ch)

        nx.set_node_attributes(self.G, positions, "pos")

    def update_node_strategy(self, node_id, strategy):
        if node_id in self.G.nodes:
            self.G.nodes[node_id]["strategy"] = strategy
            if strategy == "CH":
                self.cluster_heads.add(node_id)
            else:
                self.cluster_heads.discard(node_id)

    def run_game_step(self):
        """Simule un round du jeu."""
        nodes = [n for n in self.G.nodes if n != self.base_station]
        # Nash (stub)
        self.nash_result = stub_nash_equilibrium(
            nodes, self.strategies, self.utilities
        )
        # Appliquer Nash
        for n, s in self.nash_result.items():
            if n in self.G.nodes:
                self.G.nodes[n]["strategy"] = s
                if s == "CH":
                    self.cluster_heads.add(n)
                else:
                    self.cluster_heads.discard(n)
        # Pareto & Core (stub)
        self.pareto_result = stub_pareto_optimum(nodes, self.strategies, self.utilities)
        self.core_result = stub_core_solution(nodes, self.strategies, self.utilities)
        # LP
        self.lp_result = stub_lp_solution(nodes, list(self.cluster_heads))
        # ACO
        self.aco_routes = stub_aco_routing(list(self.cluster_heads), self.base_station)
        # Consommation énergie simulée
        for n in nodes:
            if n in self.nodes_data:
                self.nodes_data[n]["energy"] = max(
                    0, self.nodes_data[n]["energy"] - random.uniform(0.01, 0.05)
                )
                self.G.nodes[n]["energy"] = self.nodes_data[n]["energy"]
        # Historique
        total_e = sum(d["energy"] for d in self.nodes_data.values())
        self.energy_history.append(total_e)
        self.ch_count_history.append(len(self.cluster_heads))
        self.round_num += 1


# ─────────────────────────────────────────────
#  APPLICATION PRINCIPALE
# ─────────────────────────────────────────────

class GameTheoryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Game Theory Network Optimizer — v1.0")
        self.geometry("1600x950")
        self.minsize(1200, 750)
        self.configure(bg=C["bg_dark"])

        # Style ttk
        self._setup_style()

        self.model = NetworkModel()
        self.selected_node = None
        self.anim_running = False
        self._anim_obj = None

        # Construction UI
        self._build_top_bar()
        self._build_main_layout()
        self._build_status_bar()

        # Réseau initial
        self.model.generate_random_network(n_nodes=20)
        self.refresh_all()

    # ──────────────── STYLE ────────────────
    def _setup_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=C["bg_panel"], foreground=C["text_primary"],
                        fieldbackground=C["bg_widget"], font=("Consolas", 10))
        style.configure("TFrame", background=C["bg_panel"])
        style.configure("TLabel", background=C["bg_panel"], foreground=C["text_primary"])
        style.configure("TLabelframe", background=C["bg_panel"], foreground=C["accent3"],
                        bordercolor=C["border"], relief="flat")
        style.configure("TLabelframe.Label", background=C["bg_panel"],
                        foreground=C["accent3"], font=("Consolas", 10, "bold"))
        style.configure("TButton", background=C["bg_widget"], foreground=C["text_primary"],
                        bordercolor=C["border"], relief="flat", padding=6)
        style.map("TButton",
                  background=[("active", C["bg_hover"]), ("pressed", C["accent1"])],
                  foreground=[("active", C["accent3"])])
        style.configure("Accent.TButton", background=C["accent1"],
                        foreground=C["text_primary"], font=("Consolas", 10, "bold"))
        style.map("Accent.TButton",
                  background=[("active", C["accent2"]), ("pressed", C["accent3"])])
        style.configure("TScale", background=C["bg_panel"], troughcolor=C["bg_widget"],
                        sliderthickness=14)
        style.configure("TCombobox", fieldbackground=C["bg_widget"],
                        background=C["bg_widget"], foreground=C["text_primary"])
        style.map("TCombobox", fieldbackground=[("readonly", C["bg_widget"])])
        style.configure("TNotebook", background=C["bg_dark"], tabmargins=[2, 0, 0, 0])
        style.configure("TNotebook.Tab", background=C["bg_widget"],
                        foreground=C["text_second"], padding=[12, 6])
        style.map("TNotebook.Tab",
                  background=[("selected", C["bg_hover"])],
                  foreground=[("selected", C["accent3"])])
        style.configure("TScrollbar", background=C["bg_widget"],
                        troughcolor=C["bg_dark"], arrowcolor=C["text_dim"])
        style.configure("TProgressbar", background=C["accent1"],
                        troughcolor=C["bg_widget"])
        style.configure("Tree.Treeview", background=C["bg_widget"],
                        foreground=C["text_primary"], fieldbackground=C["bg_widget"],
                        rowheight=22)
        style.configure("Tree.Treeview.Heading", background=C["bg_panel"],
                        foreground=C["accent3"], font=("Consolas", 10, "bold"))
        style.map("Tree.Treeview", background=[("selected", C["accent1"])])

    # ──────────────── TOP BAR ────────────────
    def _build_top_bar(self):
        bar = tk.Frame(self, bg=C["bg_panel"], height=56, relief="flat")
        bar.pack(side=tk.TOP, fill=tk.X)
        bar.pack_propagate(False)

        # Logo + titre
        tk.Label(bar, text="◈  GAME THEORY", bg=C["bg_panel"],
                 fg=C["accent3"], font=("Consolas", 16, "bold")).pack(side=tk.LEFT, padx=16, pady=8)
        tk.Label(bar, text="NETWORK OPTIMIZER", bg=C["bg_panel"],
                 fg=C["text_second"], font=("Consolas", 12)).pack(side=tk.LEFT)

        # Séparateur
        tk.Frame(bar, bg=C["border"], width=1).pack(side=tk.LEFT, fill=tk.Y, padx=16, pady=8)

        # Boutons principaux
        btn_cfg = [
            ("▶  GÉNÉRER RÉSEAU", self.cmd_generate, "Accent.TButton"),
            ("⚙  RUN GAME STEP",  self.cmd_run_step, "TButton"),
            ("∿  ANIMATION",      self.cmd_toggle_anim, "TButton"),
            ("↺  RESET",          self.cmd_reset, "TButton"),
        ]
        for text, cmd, st in btn_cfg:
            ttk.Button(bar, text=text, command=cmd, style=st).pack(
                side=tk.LEFT, padx=4, pady=10)

        # Round counter
        self.var_round = tk.StringVar(value="Round : 0")
        tk.Label(bar, textvariable=self.var_round, bg=C["bg_panel"],
                 fg=C["accent3"], font=("Consolas", 11, "bold")).pack(side=tk.RIGHT, padx=20)

    # ──────────────── LAYOUT PRINCIPAL ────────────────
    def _build_main_layout(self):
        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=C["bg_dark"],
                               sashwidth=4, sashrelief="flat", sashpad=0)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── Panneau Gauche : contrôles ──
        left = tk.Frame(paned, bg=C["bg_panel"], width=310)
        left.pack_propagate(False)
        paned.add(left, minsize=260)
        self._build_left_panel(left)

        # ── Centre : visualisation réseau ──
        center = tk.Frame(paned, bg=C["bg_dark"])
        paned.add(center, minsize=600)
        self._build_center_panel(center)

        # ── Droite : tableaux & stats ──
        right = tk.Frame(paned, bg=C["bg_panel"], width=380)
        right.pack_propagate(False)
        paned.add(right, minsize=300)
        self._build_right_panel(right)

    # ──────────────── PANNEAU GAUCHE ────────────────
    def _build_left_panel(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Tab 1 – Paramètres réseau
        tab_net = ttk.Frame(nb)
        nb.add(tab_net, text="  Réseau  ")
        self._build_tab_network(tab_net)

        # Tab 2 – Jeu
        tab_game = ttk.Frame(nb)
        nb.add(tab_game, text="  Jeu  ")
        self._build_tab_game(tab_game)

        # Tab 3 – Nœud sélectionné
        tab_node = ttk.Frame(nb)
        nb.add(tab_node, text="  Nœud  ")
        self._build_tab_node(tab_node)

    def _build_tab_network(self, parent):
        pad = {"padx": 10, "pady": 4}

        frame = ttk.LabelFrame(parent, text=" ◈ TOPOLOGIE DU RÉSEAU ")
        frame.pack(fill=tk.X, **pad)

        rows = [
            ("Nb nœuds capteurs :", "var_n_nodes", 5, 80, 20),
            ("Surface (m²) :",      "var_area",    50, 500, 200),
            ("Portée comm. (m) :",  "var_range",   20, 150, 60),
        ]
        for label, var, mn, mx, default in rows:
            tk.Label(frame, text=label, bg=C["bg_panel"],
                     fg=C["text_second"], font=("Consolas", 9)).pack(anchor="w", padx=8, pady=(6, 0))
            setattr(self, var, tk.IntVar(value=default))
            row_f = tk.Frame(frame, bg=C["bg_panel"])
            row_f.pack(fill=tk.X, padx=8, pady=2)
            sl = ttk.Scale(row_f, variable=getattr(self, var),
                           from_=mn, to=mx, orient=tk.HORIZONTAL)
            sl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl = tk.Label(row_f, textvariable=getattr(self, var),
                           bg=C["bg_panel"], fg=C["accent3"],
                           font=("Consolas", 10, "bold"), width=4)
            lbl.pack(side=tk.RIGHT)

        ttk.Button(frame, text="▶  GÉNÉRER", command=self.cmd_generate,
                   style="Accent.TButton").pack(fill=tk.X, padx=8, pady=8)

        # Affichage
        frame2 = ttk.LabelFrame(parent, text=" ◈ AFFICHAGE ")
        frame2.pack(fill=tk.X, **pad)

        self.var_show_edges = tk.BooleanVar(value=True)
        self.var_show_labels = tk.BooleanVar(value=True)
        self.var_show_energy = tk.BooleanVar(value=True)
        self.var_show_clusters = tk.BooleanVar(value=True)
        checks = [
            ("Afficher connexions", "var_show_edges"),
            ("Afficher labels",     "var_show_labels"),
            ("Afficher énergie",    "var_show_energy"),
            ("Afficher clusters",   "var_show_clusters"),
        ]
        for text, var in checks:
            tk.Checkbutton(frame2, text=text, variable=getattr(self, var),
                           command=self.refresh_network_plot,
                           bg=C["bg_panel"], fg=C["text_primary"],
                           selectcolor=C["accent1"], activebackground=C["bg_hover"],
                           font=("Consolas", 9)).pack(anchor="w", padx=8, pady=2)

        # Légende
        frame3 = ttk.LabelFrame(parent, text=" ◈ LÉGENDE ")
        frame3.pack(fill=tk.X, **pad)
        legends = [
            (C["accent1"],    "Nœud capteur (CM)"),
            (C["cluster_h"],  "Cluster Head (CH)"),
            (C["success"],    "Sleeping"),
            ("#E8F4FD",       "Base Station (BS)"),
            (C["nash_color"], "Équilibre Nash"),
            (C["pareto_color"],"Optimum Pareto"),
        ]
        for color, text in legends:
            f = tk.Frame(frame3, bg=C["bg_panel"])
            f.pack(fill=tk.X, padx=8, pady=1)
            tk.Canvas(f, width=14, height=14, bg=C["bg_panel"],
                      highlightthickness=0).pack(side=tk.LEFT, padx=(0, 6))
            # Dessiner cercle coloré
            c = tk.Canvas(f, width=14, height=14, bg=C["bg_panel"],
                          highlightthickness=0)
            c.create_oval(1, 1, 13, 13, fill=color, outline="")
            c.pack_forget()
            # Simple label coloré
            tk.Label(f, text="●", bg=C["bg_panel"], fg=color,
                     font=("Consolas", 12)).pack(side=tk.LEFT)
            tk.Label(f, text=text, bg=C["bg_panel"], fg=C["text_second"],
                     font=("Consolas", 9)).pack(side=tk.LEFT)

    def _build_tab_game(self, parent):
        pad = {"padx": 10, "pady": 4}

        frame = ttk.LabelFrame(parent, text=" ◈ STRATÉGIES ")
        frame.pack(fill=tk.X, **pad)

        tk.Label(frame, text="Stratégies disponibles :", bg=C["bg_panel"],
                 fg=C["text_second"], font=("Consolas", 9)).pack(anchor="w", padx=8, pady=(6, 2))

        self.var_strat_ch = tk.BooleanVar(value=True)
        self.var_strat_cm = tk.BooleanVar(value=True)
        self.var_strat_sl = tk.BooleanVar(value=True)
        strats = [
            ("CH (Cluster Head)",   "var_strat_ch"),
            ("CM (Cluster Member)", "var_strat_cm"),
            ("Sleep",               "var_strat_sl"),
        ]
        for text, var in strats:
            tk.Checkbutton(frame, text=text, variable=getattr(self, var),
                           command=self._update_strategies,
                           bg=C["bg_panel"], fg=C["text_primary"],
                           selectcolor=C["accent1"], activebackground=C["bg_hover"],
                           font=("Consolas", 9)).pack(anchor="w", padx=8, pady=2)

        # Utilités & Coûts
        frame2 = ttk.LabelFrame(parent, text=" ◈ PARAMÈTRES DU JEU ")
        frame2.pack(fill=tk.X, **pad)

        params = [
            ("Coût transmission :", "var_tx_cost",  0, 100, 30),
            ("Gain énergie CH :",   "var_ch_gain",  0, 100, 70),
            ("Pénalité Sleeping :", "var_sleep_pen",0, 100, 20),
        ]
        for label, var, mn, mx, default in params:
            tk.Label(frame2, text=label, bg=C["bg_panel"],
                     fg=C["text_second"], font=("Consolas", 9)).pack(anchor="w", padx=8, pady=(4, 0))
            setattr(self, var, tk.IntVar(value=default))
            rf = tk.Frame(frame2, bg=C["bg_panel"])
            rf.pack(fill=tk.X, padx=8, pady=2)
            ttk.Scale(rf, variable=getattr(self, var), from_=mn, to=mx,
                      orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(rf, textvariable=getattr(self, var), bg=C["bg_panel"],
                     fg=C["accent3"], font=("Consolas", 10, "bold"),
                     width=4).pack(side=tk.RIGHT)

        # Boutons de jeu
        frame3 = ttk.LabelFrame(parent, text=" ◈ ACTIONS ")
        frame3.pack(fill=tk.X, **pad)

        btns = [
            ("⚡ Nash Equilibrium",  self.cmd_nash),
            ("★  Pareto Optimum",    self.cmd_pareto),
            ("◎  Core Solution",     self.cmd_core),
            ("⊞  LP Benchmark",      self.cmd_lp),
        ]
        for text, cmd in btns:
            ttk.Button(frame3, text=text, command=cmd).pack(
                fill=tk.X, padx=8, pady=3)

        ttk.Button(frame3, text="▶▶ RUN STEP COMPLET",
                   command=self.cmd_run_step,
                   style="Accent.TButton").pack(fill=tk.X, padx=8, pady=6)

    def _build_tab_node(self, parent):
        pad = {"padx": 10, "pady": 4}

        self.node_info_frame = ttk.LabelFrame(parent, text=" ◈ NŒUD SÉLECTIONNÉ ")
        self.node_info_frame.pack(fill=tk.X, **pad)

        self.var_node_id = tk.StringVar(value="—")
        self.var_node_energy = tk.StringVar(value="—")
        self.var_node_strategy = tk.StringVar(value="—")
        self.var_node_utility = tk.StringVar(value="—")
        self.var_node_cost = tk.StringVar(value="—")
        self.var_node_degree = tk.StringVar(value="—")

        info_rows = [
            ("ID nœud",   self.var_node_id),
            ("Énergie",   self.var_node_energy),
            ("Stratégie", self.var_node_strategy),
            ("Utilité",   self.var_node_utility),
            ("Coût",      self.var_node_cost),
            ("Degré",     self.var_node_degree),
        ]
        for label, var in info_rows:
            f = tk.Frame(self.node_info_frame, bg=C["bg_panel"])
            f.pack(fill=tk.X, padx=8, pady=2)
            tk.Label(f, text=label + " :", bg=C["bg_panel"],
                     fg=C["text_dim"], font=("Consolas", 9),
                     width=12, anchor="w").pack(side=tk.LEFT)
            tk.Label(f, textvariable=var, bg=C["bg_panel"],
                     fg=C["accent3"], font=("Consolas", 10, "bold")).pack(side=tk.LEFT)

        # Modifier stratégie
        frame2 = ttk.LabelFrame(parent, text=" ◈ MODIFIER STRATÉGIE ")
        frame2.pack(fill=tk.X, **pad)

        self.var_new_strategy = tk.StringVar(value="CH")
        ttk.Combobox(frame2, textvariable=self.var_new_strategy,
                     values=["CH", "CM", "Sleep"],
                     state="readonly").pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(frame2, text="✓  Appliquer",
                   command=self.cmd_apply_strategy,
                   style="Accent.TButton").pack(fill=tk.X, padx=8, pady=4)

        # Énergie barre
        frame3 = ttk.LabelFrame(parent, text=" ◈ ÉNERGIE NŒUD ")
        frame3.pack(fill=tk.X, **pad)
        self.energy_bar = ttk.Progressbar(frame3, orient=tk.HORIZONTAL,
                                          mode="determinate", length=200)
        self.energy_bar.pack(fill=tk.X, padx=8, pady=6)

        # Voisins
        frame4 = ttk.LabelFrame(parent, text=" ◈ VOISINS ")
        frame4.pack(fill=tk.BOTH, expand=True, **pad)
        self.neighbors_list = tk.Listbox(frame4, bg=C["bg_widget"],
                                         fg=C["text_primary"],
                                         selectbackground=C["accent1"],
                                         font=("Consolas", 9), height=6,
                                         relief="flat", bd=0)
        self.neighbors_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    # ──────────────── PANNEAU CENTRE ────────────────
    def _build_center_panel(self, parent):
        # Titre
        header = tk.Frame(parent, bg=C["bg_dark"], height=30)
        header.pack(fill=tk.X)
        tk.Label(header, text="◈  VISUALISATION RÉSEAU",
                 bg=C["bg_dark"], fg=C["accent3"],
                 font=("Consolas", 11, "bold")).pack(side=tk.LEFT, padx=12, pady=4)
        self.var_net_info = tk.StringVar(value="")
        tk.Label(header, textvariable=self.var_net_info,
                 bg=C["bg_dark"], fg=C["text_second"],
                 font=("Consolas", 9)).pack(side=tk.RIGHT, padx=12)

        # Figure réseau
        self.fig_net = Figure(figsize=(8, 6), dpi=96,
                              facecolor=C["bg_dark"])
        self.ax_net = self.fig_net.add_subplot(111)
        self._style_ax(self.ax_net)

        self.canvas_net = FigureCanvasTkAgg(self.fig_net, master=parent)
        self.canvas_net.get_tk_widget().configure(bg=C["bg_dark"],
                                                   highlightthickness=0)
        self.canvas_net.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas_net.mpl_connect("button_press_event", self._on_network_click)

        # Toolbar matplotlib (optionnel, minimaliste)
        toolbar_frame = tk.Frame(parent, bg=C["bg_panel"], height=28)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas_net, toolbar_frame)
        toolbar.config(bg=C["bg_panel"])
        toolbar.update()

    # ──────────────── PANNEAU DROIT ────────────────
    def _build_right_panel(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Tab Statistiques
        tab_stats = ttk.Frame(nb)
        nb.add(tab_stats, text="  Stats  ")
        self._build_tab_stats(tab_stats)

        # Tab Résultats jeu
        tab_results = ttk.Frame(nb)
        nb.add(tab_results, text="  Résultats  ")
        self._build_tab_results(tab_results)

        # Tab Convergence
        tab_conv = ttk.Frame(nb)
        nb.add(tab_conv, text="  Convergence  ")
        self._build_tab_convergence(tab_conv)

        # Tab Table nœuds
        tab_table = ttk.Frame(nb)
        nb.add(tab_table, text="  Table  ")
        self._build_tab_table(tab_table)

    def _build_tab_stats(self, parent):
        # Métriques clés
        metrics_frame = tk.Frame(parent, bg=C["bg_panel"])
        metrics_frame.pack(fill=tk.X, padx=6, pady=6)

        self.metric_vars = {}
        metrics = [
            ("NŒUDS",    "var_m_nodes",  C["accent3"]),
            ("EDGES",    "var_m_edges",  C["accent2"]),
            ("CH",       "var_m_ch",     C["cluster_h"]),
            ("ÉNERGIE",  "var_m_energy", C["success"]),
        ]
        for i, (label, var, color) in enumerate(metrics):
            f = tk.Frame(metrics_frame, bg=C["bg_widget"],
                         relief="flat", bd=0)
            f.grid(row=0, column=i, padx=3, pady=3, sticky="nsew")
            metrics_frame.columnconfigure(i, weight=1)
            setattr(self, var, tk.StringVar(value="—"))
            tk.Label(f, text=label, bg=C["bg_widget"],
                     fg=C["text_dim"], font=("Consolas", 8)).pack(pady=(6, 0))
            tk.Label(f, textvariable=getattr(self, var),
                     bg=C["bg_widget"], fg=color,
                     font=("Consolas", 14, "bold")).pack(pady=(0, 6))

        # Graphique énergie en temps réel
        frame_chart = ttk.LabelFrame(parent, text=" ◈ ÉNERGIE TOTALE (temps réel) ")
        frame_chart.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.fig_stats = Figure(figsize=(4, 2.5), dpi=88,
                                facecolor=C["bg_panel"])
        self.ax_energy = self.fig_stats.add_subplot(111)
        self._style_ax(self.ax_energy, compact=True)
        self.ax_energy.set_title("Énergie / Round", color=C["text_second"],
                                  fontsize=9, fontfamily="monospace")

        self.canvas_stats = FigureCanvasTkAgg(self.fig_stats, master=frame_chart)
        self.canvas_stats.get_tk_widget().configure(bg=C["bg_panel"],
                                                     highlightthickness=0)
        self.canvas_stats.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_tab_results(self, parent):
        self.results_text = tk.Text(parent, bg=C["bg_widget"],
                                    fg=C["text_primary"],
                                    font=("Consolas", 9),
                                    relief="flat", bd=0,
                                    insertbackground=C["accent3"],
                                    wrap=tk.WORD)
        scroll = ttk.Scrollbar(parent, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Tags de couleur
        self.results_text.tag_configure("header",
            foreground=C["accent3"], font=("Consolas", 10, "bold"))
        self.results_text.tag_configure("nash",
            foreground=C["nash_color"])
        self.results_text.tag_configure("pareto",
            foreground=C["pareto_color"])
        self.results_text.tag_configure("core",
            foreground=C["core_color"])
        self.results_text.tag_configure("lp",
            foreground=C["warning"])
        self.results_text.tag_configure("dim",
            foreground=C["text_dim"])
        self.results_text.tag_configure("value",
            foreground=C["accent2"])

        self._log_result("◈ RÉSULTATS DU JEU\n", "header")
        self._log_result("Appuyez sur 'RUN GAME STEP'\nou lancez Nash / Pareto / Core.\n", "dim")

    def _build_tab_convergence(self, parent):
        self.fig_conv = Figure(figsize=(4, 4), dpi=88,
                               facecolor=C["bg_panel"])
        self.ax_conv = self.fig_conv.add_subplot(211)
        self.ax_ch = self.fig_conv.add_subplot(212)
        self._style_ax(self.ax_conv, compact=True)
        self._style_ax(self.ax_ch, compact=True)
        self.ax_conv.set_title("Convergence énergie", color=C["text_second"],
                                fontsize=9, fontfamily="monospace")
        self.ax_ch.set_title("Nb Cluster Heads", color=C["text_second"],
                              fontsize=9, fontfamily="monospace")
        self.fig_conv.tight_layout(pad=1.5)

        self.canvas_conv = FigureCanvasTkAgg(self.fig_conv, master=parent)
        self.canvas_conv.get_tk_widget().configure(bg=C["bg_panel"],
                                                    highlightthickness=0)
        self.canvas_conv.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                               padx=4, pady=4)

    def _build_tab_table(self, parent):
        cols = ("ID", "Type", "Stratégie", "Énergie", "Utilité", "Degré")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings",
                                  style="Tree.Treeview", height=25)
        col_widths = [50, 60, 80, 70, 70, 55]
        for col, w in zip(cols, col_widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, minwidth=w, anchor="center")

        scroll_y = ttk.Scrollbar(parent, orient=tk.VERTICAL,
                                  command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.tree.bind("<ButtonRelease-1>", self._on_tree_select)

    # ──────────────── STATUS BAR ────────────────
    def _build_status_bar(self):
        bar = tk.Frame(self, bg=C["bg_panel"], height=24, relief="flat")
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.var_status = tk.StringVar(value="Prêt.")
        tk.Label(bar, textvariable=self.var_status, bg=C["bg_panel"],
                 fg=C["text_dim"], font=("Consolas", 9),
                 anchor="w").pack(side=tk.LEFT, padx=10)
        # Indicateurs droite
        self.var_fps = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.var_fps, bg=C["bg_panel"],
                 fg=C["text_dim"], font=("Consolas", 9)).pack(side=tk.RIGHT, padx=10)

    # ──────────────── UTILITAIRES GRAPHIQUES ────────────────
    def _style_ax(self, ax, compact=False):
        ax.set_facecolor(C["bg_dark"])
        ax.tick_params(colors=C["text_dim"], labelsize=7 if compact else 8)
        for spine in ax.spines.values():
            spine.set_edgecolor(C["border"])
        ax.xaxis.label.set_color(C["text_second"])
        ax.yaxis.label.set_color(C["text_second"])
        ax.grid(True, color=C["grid"], linewidth=0.4, alpha=0.6)

    def _log_result(self, text, tag=""):
        self.results_text.configure(state="normal")
        self.results_text.insert(tk.END, text, tag)
        self.results_text.see(tk.END)
        self.results_text.configure(state="disabled")

    def _clear_results(self):
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)
        self.results_text.configure(state="disabled")

    def set_status(self, msg):
        self.var_status.set(msg)

    # ──────────────── DESSIN DU RÉSEAU ────────────────
    def refresh_network_plot(self):
        G = self.model.G
        if len(G.nodes) == 0:
            return

        self.ax_net.clear()
        self._style_ax(self.ax_net)

        pos = nx.get_node_attributes(G, "pos")
        if not pos:
            return

        # Séparer les nœuds par stratégie
        ch_nodes = [n for n in G.nodes if G.nodes[n].get("strategy") == "CH"]
        cm_nodes = [n for n in G.nodes if G.nodes[n].get("strategy") == "CM"]
        sl_nodes = [n for n in G.nodes if G.nodes[n].get("strategy") == "Sleep"]
        bs_nodes = [n for n in G.nodes if G.nodes[n].get("node_type") == "BS"]

        # Edges
        if self.var_show_edges.get():
            nx.draw_networkx_edges(G, pos, ax=self.ax_net,
                                   edge_color=C["border"],
                                   alpha=0.45, width=0.8,
                                   style="solid")
            # Edges vers BS en couleur différente
            bs_edges = [(u, v) for u, v in G.edges
                        if u == "BS" or v == "BS"]
            if bs_edges:
                nx.draw_networkx_edges(G, pos, edgelist=bs_edges,
                                       ax=self.ax_net,
                                       edge_color=C["accent3"],
                                       alpha=0.3, width=1.2,
                                       style="dashed")

        # Nœuds CM (bleu)
        if cm_nodes:
            # Taille selon énergie
            sizes = [200 + 300 * G.nodes[n].get("energy", 0.5) for n in cm_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=cm_nodes,
                                   ax=self.ax_net,
                                   node_color=C["node_normal"],
                                   node_size=sizes,
                                   alpha=0.85)

        # Nœuds CH (orange)
        if ch_nodes:
            sizes = [400 + 400 * G.nodes[n].get("energy", 0.5) for n in ch_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=ch_nodes,
                                   ax=self.ax_net,
                                   node_color=C["cluster_h"],
                                   node_size=sizes,
                                   alpha=0.95)
            # Halo CH
            nx.draw_networkx_nodes(G, pos, nodelist=ch_nodes,
                                   ax=self.ax_net,
                                   node_color="none",
                                   node_size=[s + 200 for s in sizes],
                                   edgecolors=C["cluster_h"],
                                   linewidths=1.5,
                                   alpha=0.3)

        # Nœuds Sleep (vert foncé)
        if sl_nodes:
            sizes = [150 + 200 * G.nodes[n].get("energy", 0.5) for n in sl_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=sl_nodes,
                                   ax=self.ax_net,
                                   node_color=C["success"],
                                   node_size=sizes,
                                   alpha=0.6)

        # Base station
        if bs_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=bs_nodes,
                                   ax=self.ax_net,
                                   node_color=C["text_primary"],
                                   node_size=800,
                                   alpha=1.0)
            nx.draw_networkx_nodes(G, pos, nodelist=bs_nodes,
                                   ax=self.ax_net,
                                   node_color="none",
                                   node_size=1100,
                                   edgecolors=C["accent3"],
                                   linewidths=2)

        # Labels
        if self.var_show_labels.get():
            # Labels CH en orange
            ch_labels = {n: n for n in ch_nodes}
            cm_labels = {n: n for n in cm_nodes[:15]}  # Limite pour lisibilité
            bs_labels = {n: n for n in bs_nodes}
            if ch_labels:
                nx.draw_networkx_labels(G, pos, labels=ch_labels,
                                        ax=self.ax_net,
                                        font_size=7,
                                        font_color=C["cluster_h"],
                                        font_family="monospace")
            if cm_labels:
                nx.draw_networkx_labels(G, pos, labels=cm_labels,
                                        ax=self.ax_net,
                                        font_size=6,
                                        font_color=C["text_second"],
                                        font_family="monospace")
            if bs_labels:
                nx.draw_networkx_labels(G, pos, labels=bs_labels,
                                        ax=self.ax_net,
                                        font_size=9,
                                        font_color=C["text_primary"],
                                        font_family="monospace",
                                        font_weight="bold")

        # Highlight Nash
        if self.model.nash_result:
            nash_ch = [n for n, s in self.model.nash_result.items()
                       if s == "CH" and n in pos]
            if nash_ch:
                nx.draw_networkx_nodes(G, pos, nodelist=nash_ch,
                                       ax=self.ax_net,
                                       node_color="none",
                                       node_size=900,
                                       edgecolors=C["nash_color"],
                                       linewidths=2.5,
                                       alpha=0.7)

        # Nœud sélectionné
        if self.selected_node and self.selected_node in pos:
            nx.draw_networkx_nodes(G, pos,
                                   nodelist=[self.selected_node],
                                   ax=self.ax_net,
                                   node_color="none",
                                   node_size=700,
                                   edgecolors=C["accent3"],
                                   linewidths=3)

        # Routes ACO (si disponibles)
        if self.model.aco_routes:
            for ch, route in self.model.aco_routes.items():
                for i in range(len(route) - 1):
                    u, v = str(route[i]), str(route[i + 1])
                    if u in pos and v in pos:
                        x_vals = [pos[u][0], pos[v][0]]
                        y_vals = [pos[u][1], pos[v][1]]
                        self.ax_net.plot(x_vals, y_vals,
                                         color=C["pareto_color"],
                                         linewidth=1.5, alpha=0.5,
                                         linestyle="--",
                                         zorder=3)

        # Titre
        n_ch = len(ch_nodes)
        n_tot = len([n for n in G.nodes if n != "BS"])
        self.ax_net.set_title(
            f"Réseau : {n_tot} nœuds  |  {n_ch} CHs  |  Round {self.model.round_num}",
            color=C["text_second"], fontsize=9, fontfamily="monospace", pad=6)
        self.ax_net.axis("off")
        self.fig_net.tight_layout(pad=0.5)
        self.canvas_net.draw()

    def refresh_stats_plot(self):
        self.ax_energy.clear()
        self._style_ax(self.ax_energy, compact=True)
        hist = self.model.energy_history
        if hist:
            rounds = list(range(len(hist)))
            self.ax_energy.plot(rounds, hist, color=C["accent1"],
                                linewidth=1.5, alpha=0.9)
            self.ax_energy.fill_between(rounds, hist, alpha=0.15,
                                         color=C["accent1"])
            self.ax_energy.set_ylabel("Énergie", fontsize=7,
                                       color=C["text_dim"],
                                       fontfamily="monospace")
        self.ax_energy.set_title("Énergie Totale", color=C["text_second"],
                                  fontsize=8, fontfamily="monospace")
        self.fig_stats.tight_layout(pad=0.8)
        self.canvas_stats.draw()

    def refresh_convergence_plot(self):
        self.ax_conv.clear()
        self.ax_ch.clear()
        self._style_ax(self.ax_conv, compact=True)
        self._style_ax(self.ax_ch, compact=True)

        h_e = self.model.energy_history
        h_ch = self.model.ch_count_history
        if h_e:
            r = list(range(len(h_e)))
            self.ax_conv.plot(r, h_e, color=C["accent1"],
                              linewidth=1.5, marker="o", markersize=3)
            self.ax_conv.fill_between(r, h_e, alpha=0.12, color=C["accent1"])
            self.ax_conv.set_ylabel("Énergie", fontsize=7,
                                     color=C["text_dim"],
                                     fontfamily="monospace")
        if h_ch:
            r = list(range(len(h_ch)))
            self.ax_ch.plot(r, h_ch, color=C["cluster_h"],
                            linewidth=1.5, marker="s", markersize=3)
            self.ax_ch.fill_between(r, h_ch, alpha=0.12, color=C["cluster_h"])
            self.ax_ch.set_ylabel("# CH", fontsize=7, color=C["text_dim"],
                                   fontfamily="monospace")
            self.ax_ch.set_xlabel("Round", fontsize=7, color=C["text_dim"],
                                   fontfamily="monospace")

        self.ax_conv.set_title("Convergence énergie", color=C["text_second"],
                                fontsize=8, fontfamily="monospace")
        self.ax_ch.set_title("Nombre de Cluster Heads", color=C["text_second"],
                              fontsize=8, fontfamily="monospace")
        self.fig_conv.tight_layout(pad=1.0)
        self.canvas_conv.draw()

    def refresh_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        G = self.model.G
        for n in sorted(G.nodes):
            nd = G.nodes[n]
            energy = round(nd.get("energy", 0), 3)
            util = round(self.model.utilities.get(n, 0), 3)
            deg = G.degree(n)
            node_type = nd.get("node_type", "sensor")
            strategy = nd.get("strategy", "—")
            # Tag couleur
            tag = "ch" if strategy == "CH" else ("bs" if node_type == "BS" else "")
            self.tree.insert("", "end", iid=n,
                              values=(n, node_type, strategy, energy, util, deg),
                              tags=(tag,))
        self.tree.tag_configure("ch",
            foreground=C["cluster_h"], background=C["bg_widget"])
        self.tree.tag_configure("bs",
            foreground=C["accent3"], background=C["bg_widget"])

    def refresh_metrics(self):
        G = self.model.G
        n_nodes = len([n for n in G.nodes if n != "BS"])
        n_edges = G.number_of_edges()
        n_ch = len(self.model.cluster_heads)
        total_e = sum(d.get("energy", 0) for n, d in G.nodes(data=True)
                      if n != "BS")
        self.var_m_nodes.set(str(n_nodes))
        self.var_m_edges.set(str(n_edges))
        self.var_m_ch.set(str(n_ch))
        self.var_m_energy.set(f"{total_e:.1f}")
        self.var_round.set(f"Round : {self.model.round_num}")
        self.var_net_info.set(
            f"Nœuds: {n_nodes}  |  Edges: {n_edges}  |  CHs: {n_ch}"
        )

    def refresh_all(self):
        self.refresh_network_plot()
        self.refresh_metrics()
        self.refresh_table()
        self.refresh_stats_plot()
        self.refresh_convergence_plot()

    def update_node_info(self, node_id):
        G = self.model.G
        if node_id not in G.nodes:
            return
        nd = G.nodes[node_id]
        self.var_node_id.set(node_id)
        energy = nd.get("energy", 0)
        self.var_node_energy.set(f"{energy:.4f}")
        self.var_node_strategy.set(nd.get("strategy", "—"))
        self.var_node_utility.set(
            f"{self.model.utilities.get(node_id, 0):.4f}")
        self.var_node_cost.set(
            f"{self.model.costs.get(node_id, 0):.4f}")
        self.var_node_degree.set(str(G.degree(node_id)))
        # Barre énergie
        self.energy_bar["value"] = min(100, int(energy * 100))
        # Voisins
        self.neighbors_list.delete(0, tk.END)
        for nb_node in G.neighbors(node_id):
            nb_strat = G.nodes[nb_node].get("strategy", "?")
            nb_e = G.nodes[nb_node].get("energy", 0)
            self.neighbors_list.insert(
                tk.END, f"{nb_node} [{nb_strat}] e={nb_e:.2f}")

    # ──────────────── ÉVÉNEMENTS ────────────────
    def _on_network_click(self, event):
        if event.inaxes != self.ax_net or event.button != 1:
            return
        pos = nx.get_node_attributes(self.model.G, "pos")
        if not pos:
            return
        cx, cy = event.xdata, event.ydata
        if cx is None or cy is None:
            return
        # Trouver le nœud le plus proche
        min_dist = float("inf")
        closest = None
        threshold = 15  # pixels dans l'espace réseau
        for node_id, (x, y) in pos.items():
            d = math.hypot(cx - x, cy - y)
            if d < min_dist and d < threshold:
                min_dist = d
                closest = node_id
        if closest:
            self.selected_node = closest
            self.update_node_info(closest)
            self.refresh_network_plot()
            self.set_status(f"Nœud sélectionné : {closest}")

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if sel:
            node_id = sel[0]
            self.selected_node = node_id
            self.update_node_info(node_id)
            self.refresh_network_plot()

    def _update_strategies(self):
        strats = []
        if self.var_strat_ch.get():
            strats.append("CH")
        if self.var_strat_cm.get():
            strats.append("CM")
        if self.var_strat_sl.get():
            strats.append("Sleep")
        self.model.strategies = strats if strats else ["CM"]

    # ──────────────── COMMANDES ────────────────
    def cmd_generate(self):
        n = int(self.var_n_nodes.get())
        a = int(self.var_area.get())
        r = int(self.var_range.get())
        self.model.generate_random_network(n_nodes=n, area=a, comm_range=r)
        self._clear_results()
        self._log_result(f"◈ Réseau généré : {n} nœuds, zone={a}m, portée={r}m\n", "header")
        self.selected_node = None
        self.refresh_all()
        self.set_status(f"Réseau généré : {n} nœuds.")

    def cmd_reset(self):
        self.model.reset()
        self._clear_results()
        self._log_result("◈ Réseau réinitialisé.\n", "header")
        self.selected_node = None
        self.anim_running = False
        if self._anim_obj:
            self._anim_obj.event_source.stop()
        self.refresh_all()
        self.set_status("Reset effectué.")

    def cmd_run_step(self):
        if len(self.model.G.nodes) == 0:
            messagebox.showwarning("Attention", "Générez d'abord un réseau !")
            return
        self.model.run_game_step()
        self._clear_results()
        self._log_results_full()
        self.refresh_all()
        self.set_status(f"Round {self.model.round_num} exécuté.")

    def cmd_nash(self):
        if len(self.model.G.nodes) == 0:
            return
        nodes = [n for n in self.model.G.nodes if n != "BS"]
        self.model.nash_result = stub_nash_equilibrium(
            nodes, self.model.strategies, self.model.utilities)
        self._clear_results()
        self._log_result("◈ ÉQUILIBRE DE NASH\n", "header")
        self._log_result("(Stub — connectez Personne 2)\n\n", "dim")
        ch_nash = [n for n, s in self.model.nash_result.items() if s == "CH"]
        self._log_result(f"CHs Nash : {ch_nash}\n", "nash")
        for n, s in list(self.model.nash_result.items())[:10]:
            self._log_result(f"  {n} → ", "dim")
            self._log_result(f"{s}\n", "nash")
        self.refresh_network_plot()
        self.set_status("Nash Equilibrium calculé (stub).")

    def cmd_pareto(self):
        if len(self.model.G.nodes) == 0:
            return
        nodes = [n for n in self.model.G.nodes if n != "BS"]
        self.model.pareto_result = stub_pareto_optimum(
            nodes, self.model.strategies, self.model.utilities)
        self._clear_results()
        self._log_result("◈ OPTIMUM DE PARETO\n", "header")
        self._log_result("(Stub — connectez Personne 2)\n\n", "dim")
        for i, sol in enumerate(self.model.pareto_result):
            self._log_result(f"Solution Pareto #{i+1}\n", "pareto")
            for n, s in list(sol.items())[:5]:
                self._log_result(f"  {n} → {s}\n", "dim")
        self.set_status("Pareto Optimum calculé (stub).")

    def cmd_core(self):
        if len(self.model.G.nodes) == 0:
            return
        nodes = [n for n in self.model.G.nodes if n != "BS"]
        self.model.core_result = stub_core_solution(
            nodes, self.model.strategies, self.model.utilities)
        self._clear_results()
        self._log_result("◈ SOLUTION DU CŒUR (CORE)\n", "header")
        self._log_result("(Stub — connectez Personne 2)\n\n", "dim")
        for n, s in list(self.model.core_result.items())[:10]:
            self._log_result(f"  {n} → ", "dim")
            self._log_result(f"{s}\n", "core")
        self.set_status("Core Solution calculée (stub).")

    def cmd_lp(self):
        if len(self.model.G.nodes) == 0:
            return
        nodes = [n for n in self.model.G.nodes if n != "BS"]
        self.model.lp_result = stub_lp_solution(nodes, list(self.model.cluster_heads))
        self._clear_results()
        self._log_result("◈ OPTIMISATION CENTRALISÉE (LP)\n", "header")
        self._log_result("(Stub — connectez Personne 3)\n\n", "dim")
        self._log_result(f"Méthode : ", "dim")
        self._log_result(f"{self.model.lp_result.get('method', '—')}\n", "lp")
        self._log_result(f"Obj. valeur : ", "dim")
        self._log_result(f"{self.model.lp_result.get('obj_value', '—')}\n", "lp")

        # Comparaison Nash vs LP
        if self.model.nash_result:
            self._log_result("\n◈ COMPARAISON Nash vs LP\n", "header")
            n_ch_nash = sum(1 for s in self.model.nash_result.values() if s == "CH")
            lp_obj = self.model.lp_result.get("obj_value", 0)
            self._log_result(f"CHs Nash : {n_ch_nash}  |  LP obj : {lp_obj}\n", "value")
        self.set_status("LP Benchmark calculé (stub).")

    def cmd_apply_strategy(self):
        if not self.selected_node:
            messagebox.showinfo("Info", "Sélectionnez d'abord un nœud.")
            return
        if self.selected_node == "BS":
            messagebox.showwarning("Attention", "Impossible de modifier la Base Station.")
            return
        strat = self.var_new_strategy.get()
        self.model.update_node_strategy(self.selected_node, strat)
        self.update_node_info(self.selected_node)
        self.refresh_network_plot()
        self.refresh_table()
        self.refresh_metrics()
        self.set_status(f"Stratégie de {self.selected_node} → {strat}")

    def cmd_toggle_anim(self):
        if self.anim_running:
            self.anim_running = False
            if self._anim_obj:
                self._anim_obj.event_source.stop()
            self.set_status("Animation arrêtée.")
        else:
            if len(self.model.G.nodes) == 0:
                messagebox.showwarning("Attention", "Générez d'abord un réseau !")
                return
            self.anim_running = True
            self._anim_obj = animation.FuncAnimation(
                self.fig_net, self._anim_frame,
                interval=1200, blit=False, cache_frame_data=False)
            self.canvas_net.draw()
            self.set_status("Animation démarrée.")

    def _anim_frame(self, frame):
        if not self.anim_running:
            return
        self.model.run_game_step()
        self.refresh_network_plot()
        self.refresh_metrics()
        self.refresh_stats_plot()
        self.refresh_convergence_plot()
        self.var_fps.set(f"Frame {self.model.round_num}")

    def _log_results_full(self):
        self._log_result(f"◈ ROUND {self.model.round_num}\n", "header")
        # Nash
        if self.model.nash_result:
            self._log_result("─ Nash Equilibrium ─\n", "nash")
            ch_n = [n for n, s in self.model.nash_result.items() if s == "CH"]
            self._log_result(f"  CHs: {ch_n[:5]}{'…' if len(ch_n)>5 else ''}\n", "value")
        # LP
        if self.model.lp_result:
            self._log_result("─ LP Benchmark ─\n", "lp")
            self._log_result(f"  Obj: {self.model.lp_result.get('obj_value','—')}\n", "value")
        # Énergie
        total_e = sum(d.get("energy", 0)
                      for n, d in self.model.G.nodes(data=True) if n != "BS")
        self._log_result(f"─ Énergie résiduelle : {total_e:.3f} ─\n\n", "dim")


# ─────────────────────────────────────────────
#  POINT D'ENTRÉE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = GameTheoryApp()
    app.mainloop()
