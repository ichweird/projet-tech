import itertools

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

#################################
########## PARTIE 1: Structures de données et types ################
##################################

class Strategy(Enum):
    ###stratégies possibles pour un noeud####
    D = "D"      # declare as cluster head
    RD = "RD"    # regular node (no declaration)

@dataclass
class Node:
    ###représentation d'un noeud du réseau###
    id: int
    x: float
    y: float
    E_res: float           # energie résiduelle (Joules)
    E_init: float          # energie initiale (Joules)
    communication_range: float = 50.0  # mètres
    
    #attributs calcules dynamiquement
    d_to_BS: Optional[float] = None
    neighbors: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.d_to_BS is None:
            self.d_to_BS = 0.0
        if self.neighbors is None:
            self.neighbors = []

@dataclass
class PayoffMatrix:
    ###"matrice de paiement pour le jeu à 2 joueurs###
    #matrice: [strategie_joueur1][strategie_joueur2]
    #ordre: D, RD
    player1: List[List[float]]
    player2: List[List[float]]


####
####PARTIE 2: implémentation des équations de l'article (section 3.3)###
####

class GameTheoryCHSelection:
    """
    Implementation complete de la selection des Cluster heads
    basée sur la théorie des jeux.
    
    Équations implementees:
    - (14): Fonction de paiement U_i(S)
    - (17): Probabilite d'equilibre de Nash p
    - (18): Probabilite avec seuil d'energie P_k(i)
    - (19): Critere de departage lambda(i)
    """
    
    def __init__(self):
        self.verbose = False
        
    ############
    # Equation (14): Fonction de paiement##########
    ###########
    def calculate_payoff(self, 
                        my_strategy: Strategy,
                        exists_CH_in_neighborhood: bool,
                        reward_t: float,
                        cost_c: float) -> float:
        """
        calcule le paiement U_i(S) pour un noeud selon l'équation (14)
        
        Args:
            my_strategy: strategie du noeud (D ou RD)
            exists_CH_in_neighborhood: true s'il existe au moins un CH dans le voisinage
            reward_t: recompense t pour completer la transmission
            cost_c: cout c pour l'intégration des donnees (CH uniquement)
        
        Returns:
            Paiement du noeud
        
        Cas selon l'equation (14):
        - si RD et pas de CH voisin: paiement = 0
        - si D: paiement = t - c
        - si RD et au moins un CH voisin: paiement = t
        """
        if my_strategy == Strategy.RD and not exists_CH_in_neighborhood:
            return 0.0
        elif my_strategy == Strategy.D:
            return reward_t - cost_c
        elif my_strategy == Strategy.RD and exists_CH_in_neighborhood:
            return reward_t
        else:
            return 0.0
    
    ###############
    ####### equation (17): probabilité d'équilibre de Nash #########
    ################
    def nash_equilibrium_probability(self, 
                                     cost_c: float, 
                                     reward_t: float, 
                                     NR: int) -> float:
        """
        calcul de la probabilite p d'etre CH à l'equilibre de Nash
        equation (17): p = 1 - (c/t)^(1/(NR-1))
        
        Args:
            cost_c: cout pour etre CH
            reward_t: recompense pour transmission reussie
            NR: nombre de noeuds voisins (Neighbor count + 1?)
        
        Returns:
            probabilite p entre 0 et 1
        """
        if reward_t <= 0 or cost_c >= reward_t:
            return 0.0
        
        if NR <= 1:
            return 1.0 if cost_c < reward_t else 0.0
        
        try:
            p = 1 - (cost_c / reward_t) ** (1 / (NR - 1))
            return max(0.0, min(1.0, p))
        except (ZeroDivisionError, ValueError):
            return 0.0
    
    #######
    # equation 18: probabilité avec seuil d'energie
    ###########
    def probability_cluster_head_with_energy(self,
                                             E_res: float,
                                             E_avg: float,
                                             p: float,
                                             omega: float,
                                             energy_threshold: float = 0.1) -> float:
        """
        calcule P_k(i) - la probabilite ajustée avec l'énergie résiduelle
        Equation (18): 
        P_k(i) = (E_res/E_avg) * p  si omega > E_res
        P_k(i) = 0 sinon
        
        Args:
            E_res: energie résiduelle du noeud i
            E_avg: energie moyenne du réseau
            p: probabilité de base de l'équation (17)
            omega: seuil d'énergie (omega = weight factor)
            energy_threshold: seuil minimal d'énergie (par défaut 10%)
        
        Returns:
            probabilité ajustée P_k(i)
        """
        if omega > E_res:
            if E_avg > 0:
                return (E_res / E_avg) * p
            else:
                return 0.0
        else:
            return 0.0
    
    ##############
    ####### equation (19): critere de départage lambda###########
    ####
    def lambda_criterion(self, 
                        E_res: float, 
                        E_init: float, 
                        d_to_BS: float) -> float:
        """
        Calcule lambda(i) pour departager deux candidats CH
        Équation (19): lambda(i) = (E_res/E_init) * (1/d²(i)_toBS)
        
        Args:
            E_res: energie résiduelle
            E_init: energie initiale
            d_to_BS: distance du noeud à la Base Station
        
        Returns:
            valeur lambda - plus elevee = meilleur candidat CH
        """
        if d_to_BS <= 0:
            return 0.0
        
        return (E_res / max(E_init, 1e-6)) * (1.0 / (d_to_BS ** 2))
    
    ################
    # Calcul de la récompense t (basé sur la force de communication)
    # ###########################
    def calculate_reward_t(self, 
                          R_ij: float, 
                          R_j: float, 
                          alpha: float = 0.5, 
                          theta: float = 0.5) -> float:
        """
        Calcule la récompense t = R_ij * R_j
        R_ij: force de communication entre noeud normal et CH
        R_j: force de communication entre CH et BS
        
        Plus formellement, t = (α * E_res + θ * connectivity) ...
        """
        return R_ij * R_j
    
    def calculate_communication_strength(self, 
                                        E_res: float,
                                        distance: float,
                                        max_distance: float = 100.0) -> float:
        """
        Calcule la force de communication normalisée
        """
        energy_factor = E_res / 0.5  # 0.5J est l'énergie initiale typique
        distance_factor = 1 - (distance / max_distance)
        return 0.6 * min(1.0, energy_factor) + 0.4 * max(0, distance_factor)


################
# PARTIE 3: algorithme de sélection des CH (Algorithm 1 de l'article)
################

class ClusterHeadSelectionGame:
    """
    Implémentation complète de l'algorithme de sélection des CH
    basé sur Algorithm 1 de l'article GTIACO
    """
    
    def __init__(self, 
                 communication_range: float = 50.0,
                 default_reward_t: float = 1.0,
                 default_cost_c: float = 0.5,
                 omega: float = 0.1,
                 verbose: bool = True):
        
        self.communication_range = communication_range
        self.default_reward_t = default_reward_t
        self.default_cost_c = default_cost_c
        self.omega = omega
        self.verbose = verbose
        self.game = GameTheoryCHSelection()
    
    #################
    # fonction auxiliaire: Calcul des distances
    #################
    def _calculate_distance(self, node1: Node, node2: Node) -> float:
        """Calcule la distance euclidienne entre deux noeuds"""
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def _calculate_distance_to_BS(self, node: Node, BS_x: float, BS_y: float) -> float:
        """Calcule la distance d'un noeud à la station de base"""
        return math.sqrt((node.x - BS_x)**2 + (node.y - BS_y)**2)
    
    ################
    # detection des voisins
    #################
    def _find_neighbors(self, nodes: List[Node]) -> None:
        """
        Identifie les voisins de chaque noeud dans le rayon de communication
        """
        n = len(nodes)
        for i in range(n):
            nodes[i].neighbors = []
            for j in range(n):
                if i != j:
                    dist = self._calculate_distance(nodes[i], nodes[j])
                    if dist <= self.communication_range:
                        nodes[i].neighbors.append(j)
    
    ########
    # calcul de l'énergie moyenne du réseau
    ########
    def _calculate_average_energy(self, nodes: List[Node]) -> float:
        """Calcule l'énergie résiduelle moyenne du réseau"""
        if not nodes:
            return 0.0
        total_energy = sum(node.E_res for node in nodes)
        return total_energy / len(nodes)
    
    #################
    # algorithme principal de sélection
    #################
    def select_cluster_heads(self,
                             nodes: List[Node],
                             BS_x: float,
                             BS_y: float,
                             reward_t: Optional[float] = None,
                             cost_c: Optional[float] = None) -> List[Node]:
        """
        algorithme 1 de l'article: Sélection des CH par théorie des jeux
        
        INPUT: c, t, NR, w
        OUTPUT: Status of the node (D or RD)
        
        Args:
            nodes: Liste des noeuds du réseau
            BS_x, BS_y: Position de la base station
            reward_t: Récompense t (optionnel)
            cost_c: Coût c (optionnel)
        
        Returns:
            liste des noeuds sélectionnés comme Cluster Heads
        """
        
        # initialisation
        reward_t = reward_t or self.default_reward_t
        cost_c = cost_c or self.default_cost_c
        
        # maj des distances à la BS
        for node in nodes:
            node.d_to_BS = self._calculate_distance_to_BS(node, BS_x, BS_y)
        
        # detection des voisins
        self._find_neighbors(nodes)
        
        # calcul de l'énergie moyenne
        E_avg = self._calculate_average_energy(nodes)
        
        # resultats
        cluster_heads = []
        
        # dictionnaire pour suivre les annonces
        announcements = {}  # node_id -> Pk(i)
        
        print("\n" + "="*20)
        print("ALGORITHME DE SÉLECTION DES CLUSTER HEADS (THÉORIE DES JEUX)")
        print("="*20)
        
        ################
        # ETAPE 1: calcul de Pk(i) pour chaque noeud
        ###############
        for node in nodes:
            NR = len(node.neighbors) + 1  # nombre de voisins + soi-même
            
            # Equation (17): proba de Nash
            p_nash = self.game.nash_equilibrium_probability(cost_c, reward_t, NR)
            
            # Equation (18): proba avec énergie
            Pk = self.game.probability_cluster_head_with_energy(
                E_res=node.E_res,
                E_avg=E_avg,
                p=p_nash,
                omega=self.omega
            )
            
            announcements[node.id] = Pk
            
            if self.verbose:
                print(f"\nNoeud {node.id}:")
                print(f"  - NR = {NR}, p_nash = {p_nash:.4f}, Pk = {Pk:.4f}")
        
        ############
        ########### ETAPE 2: sélection des CH selon Algorithm 1
        #########
        for node in nodes:
            Pk = announcements[node.id]
            
            # ligne 2: Si Pk(i) > 0
            if Pk > 0:
                # ligne 3: annonce aux autres joueurs dans NR
                # (simulé par la collecte des candidats dans la région)
                
                # identifier les autres candidats dans le voisinage
                candidate_neighbors = []
                for neighbor_idx in node.neighbors:
                    if announcements[neighbor_idx] > 0:
                        candidate_neighbors.append(nodes[neighbor_idx])
                
                # ligne 4: s'il y a d'autres candidats dans le domaine
                if candidate_neighbors:
                    # ajouter le noeud lui-même aux candidats
                    all_candidates = candidate_neighbors + [node]
                    
                    # lignes 5-10: comparer les valeurs de lambda 
                    # equation (19): lambda(i) = (E_res/E_init) * (1/d²_toBS)
                    best_candidate = max(all_candidates, 
                                        key=lambda n: self.game.lambda_criterion(
                                            n.E_res, n.E_init, n.d_to_BS
                                        ))
                    
                    # ligne 7: si i est le meilleur alors D
                    if best_candidate.id == node.id:
                        cluster_heads.append(node)
                        if self.verbose:
                            print(f"\n oui: Noeud {node.id} sélectionné CH (lambda = {self.game.lambda_criterion(node.E_res, node.E_init, node.d_to_BS):.4f})")
                    else:
                        if self.verbose and Pk > 0.1:
                            print(f"\n non: Noeud {node.id} non sélectionné (meilleur CH: {best_candidate.id})")
                else:
                    # ligne 12: s'il n'y a pas d'autre candidat alors D
                    cluster_heads.append(node)
                    if self.verbose:
                        print(f"\n selectionné: Noeud {node.id} sélectionné CH (seul candidat dans sa zone)")
            
            # ligne 14: Si Pk(i) <= 0
            else: 
                # ligne 15: status = RD
                if self.verbose and node.E_res > 0:
                    print(f"\n pas sélectionné: Noeud {node.id}: RD (énérgie insuffisante ou probabilité nulle)")
        
        print("\n")
        print(f"Résultat: {len(cluster_heads)} Cluster Heads sélectionnés")
        print("="*60)
        
        return cluster_heads
    
    #############
    ########### version avec sélection multi-tours (pour simulation temporelle)
    ##############
    def select_cluster_heads_adaptive(self,
                                      nodes: List[Node],
                                      BS_x: float,
                                      BS_y: float,
                                      round_number: int,
                                      adaptation_factor: float = 1.0) -> List[Node]:
        """
        version adaptative qui s'ajuste en fonction du round de simulation
        et permet d'observer l'évolution de la sélection dans le temps
        """
        # ajustement des paramètres en fonction du round
        # plus le réseau vieillit, plus le cout c augmente (énergie rare)
        node_alive_count = sum(1 for n in nodes if n.E_res > 0)
        if node_alive_count > 0:
            network_health = sum(n.E_res for n in nodes) / (node_alive_count * nodes[0].E_init)
        else:
            network_health = 0
        
        # adaptation: le cout augmente quand l'énergie diminue
        cost_c_adapted = self.default_cost_c * (1 + (1 - network_health) * adaptation_factor)
        # la récompense diminue quand le réseau est faible
        reward_t_adapted = self.default_reward_t * network_health
        
        return self.select_cluster_heads(nodes, BS_x, BS_y, reward_t_adapted, cost_c_adapted)


# ============================================================================
# PARTIE 4: equilibre de Nash en stratégies mixtes et Pareto
# ============================================================================

class EquilibriumSolver:
    """
    résoudre différents types d'équilibres pour les jeux:
    - equilibre de Nash (pur et mixte)
    - optimum de Pareto
    - coeur du jeu (pour versions coopératives)
    """
    
    @staticmethod
    def pure_nash_equilibrium(payoff_matrix: PayoffMatrix) -> List[Tuple[int, int]]:
        """
        trouve tous les équilibres de Nash en stratégies pures
        
        args:
            payoff_matrix: Matrice de paiement du jeu
            
        returns:
            liste des paires (strategie_j1, strategie_j2) à l'équilibre
        """
        equilibria = []
        n_strat1 = len(payoff_matrix.player1)
        n_strat2 = len(payoff_matrix.player2[0]) if n_strat1 > 0 else 0
        
        for i in range(n_strat1):
            for j in range(n_strat2):
                # Vérifier si (i,j) est un équilibre de Nash
                # Condition: U1(i,j) >= U1(i',j) pour tout i'
                is_best_response_j1 = True
                for i2 in range(n_strat1):
                    if payoff_matrix.player1[i2][j] > payoff_matrix.player1[i][j]:
                        is_best_response_j1 = False
                        break
                
                # Condition: U2(i,j) >= U2(i,j') pour tout j'
                is_best_response_j2 = True
                for j2 in range(n_strat2):
                    if payoff_matrix.player2[i][j2] > payoff_matrix.player2[i][j]:
                        is_best_response_j2 = False
                        break
                
                if is_best_response_j1 and is_best_response_j2:
                    equilibria.append((i, j))
        
        return equilibria
    
    @staticmethod
    def mixed_nash_equilibrium(payoff_matrix: PayoffMatrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        trouve l'équilibre de Nash en stratégies mixtes
        
        utilise la programmation linéaire pour résoudre:
        max v tel que U_i(s) >= v
        
        Args:
            payoff_matrix: matrice de paiement
            
        Returns:
            (probabilites_joueur1, probabilites_joueur2)
        """
        from scipy.optimize import linprog
        
        n_strat1 = len(payoff_matrix.player1)
        n_strat2 = len(payoff_matrix.player2[0]) if n_strat1 > 0 else 0
        
        if n_strat1 == 0 or n_strat2 == 0:
            return np.array([]), np.array([])
        
        # pour un jeu à somme nulle simplifié on maximise la valeur du jeu
        
        # variable: p[0..n_strat1-1], v
        c = [0] * n_strat1 + [-1]  # Minimiser -v = maximiser v
        
        # contraintes: U_j1 * p >= v pour chaque stratégie de j2
        A_ub = []
        b_ub = []
        
        for j in range(n_strat2):
            constraint = [-payoff_matrix.player1[i][j] for i in range(n_strat1)] + [1]
            A_ub.append(constraint)
            b_ub.append(0)
        
        #contrainte: sum(p) = 1
        A_eq = [[1] * n_strat1 + [0]]
        b_eq = [1]
        
        #bornes: p_i >= 0, v libre
        bounds = [(0, 1)] * n_strat1 + [(None, None)]
        
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            if result.success:
                probs_j1 = result.x[:n_strat1]
                #pour le joueur 2, on résout le problème dual
                probs_j2 = result.ineqlin['dual'] if hasattr(result, 'ineqlin') else np.ones(n_strat2) / n_strat2
                return probs_j1, probs_j2
        except:
            pass
        
        # Fallback: équiprobable
        return np.ones(n_strat1) / n_strat1, np.ones(n_strat2) / n_strat2
    
    @staticmethod
    def pareto_optimum(payoffs_list: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Trouve les optima de Pareto dans un ensemble de paires (U1, U2)
        
        Args:
            payoffs_list: Liste des paires de paiements
            
        Returns:
            Liste des paires "Pareto-optimales"
        """
        pareto_points = []
        
        for i, (u1, u2) in enumerate(payoffs_list):
            is_pareto = True
            for j, (v1, v2) in enumerate(payoffs_list):
                if i != j:
                    # Si un autre point est meilleur ou égal sur les deux dimensions et strictement meilleur sur au moins une
                    if v1 >= u1 and v2 >= u2 and (v1 > u1 or v2 > u2):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_points.append((u1, u2))
        
        return pareto_points
    
    @staticmethod
    def core_of_game(coalition_values: Dict[Tuple[int, ...], float], 
                     players: List[int]) -> List[Dict[int, float]]:
        """
        trouve le coeur du jeu coopératif
        
        Args:
            coalition_values: Valeur de chaque coalition {tuple(players): value}
            players: Liste des joueurs
            
        Returns:
            Liste des allocations stables (dans le coeur)
        """
        import itertools
        
        n = len(players)
        
        #une allocation x = (x1, x2, ..., xn) est dans le coeur si:
        #1 sum(x_i) = v(N): efficacité
        #2 sum_{i in S} x_i >= v(S) pour toute coalition S: rationalité collective
        
        # valeur de la grande coalition
        grand_coalition = tuple(sorted(players))
        v_N = coalition_values.get(grand_coalition, 0)
        
        # verifier si une allocation est dans le coeur
        def is_in_core(allocation):
            total = sum(allocation.values())
            if abs(total - v_N) > 1e-6:
                return False
            
            for coalition, value in coalition_values.items():
                coalition_sum = sum(allocation[p] for p in coalition)
                if coalition_sum < value - 1e-6:
                    return False
            return True
        
        #retourner quelques allocations possibles (simplifie) 
        #dans la pratique on utiliserait la programmation linéaire
        equal_allocation = {p: v_N / n for p in players}
        
        if is_in_core(equal_allocation):
            return [equal_allocation]
        
        #valeur de Shapley comme approximation
        shapley = EquilibriumSolver.shapley_value(coalition_values, players)
        if is_in_core(shapley):
            return [shapley]
        
        return []
    
    @staticmethod
    def shapley_value(coalition_values: Dict[Tuple[int, ...], float], 
                      players: List[int]) -> Dict[int, float]:
        """
        calcule de la valeur de Shapley pour un jeu coopératif
        """
        import math
        from collections import defaultdict
        
        n = len(players)
        shapley = defaultdict(float)
        
        for player in players:
            other_players = [p for p in players if p != player]
            
            #somme sur toutes les coalitions S qui ne contiennent pas player
            for k in range(n):
                #nombre de façons de choisir k joueurs parmi les autres
                n_combinations = math.comb(len(other_players), k)
                
                for subset in itertools.combinations(other_players, k):
                    S = set(subset)
                    S_with_player = S.union({player})
                    
                    v_S = coalition_values.get(tuple(sorted(S)), 0)
                    v_S_with_player = coalition_values.get(tuple(sorted(S_with_player)), 0)
                    
                    #contribution marginale
                    marginal = v_S_with_player - v_S
                    
                    #poids: k!(n-k-1)!/n!
                    weight = (math.factorial(k) * math.factorial(n - k - 1)) / math.factorial(n)
                    
                    shapley[player] += weight * marginal / max(1, n_combinations)
        
        return dict(shapley)


###############
# PARTIE 5: Exemple d'utilisation et test###########""
#############""

def create_test_network(n_nodes: int = 10, area_size: float = 100.0) -> List[Node]:
    """creation d un réseau de test aléatoire"""
    nodes = []
    for i in range(n_nodes):
        node = Node(
            id=i,
            x=np.random.uniform(0, area_size),
            y=np.random.uniform(0, area_size),
            E_res=np.random.uniform(0.1, 0.5),
            E_init=0.5,
            communication_range=50.0
        )
        nodes.append(node)
    return nodes

def run_test():
    """Test complet de toutes les fonctionnalités"""
    
    print("TEST COMPLET (THÉORIE DES JEUX)")
    print("="*80)
    
    # 1 creation du réseau de test
    print("\n1. Création du réseau de test...")
    nodes = create_test_network(n_nodes=15, area_size=100)
    BS_x, BS_y = 50, 50
    
    # 2 jeu de sélection des CH
    print("\n2. Lancement de la sélection des CH...")
    selector = ClusterHeadSelectionGame(
        communication_range=45.0,
        default_reward_t=1.2,
        default_cost_c=0.5,
        omega=0.15,
        verbose=True
    )
    
    cluster_heads = selector.select_cluster_heads(nodes, BS_x, BS_y)
    
    # 3 test des equations individuelles
    
    print("\n")
    print("3. Test des equations individuelles")
    
    game = GameTheoryCHSelection()
    
    # test equation (14)
    print("\nEquation (14) - Fonction de paiement:")
    print(f"  D sans CH voisin: {game.calculate_payoff(Strategy.D, False, 1.0, 0.5)}")
    print(f"  RD avec CH voisin: {game.calculate_payoff(Strategy.RD, True, 1.0, 0.5)}")
    print(f"  RD sans CH voisin: {game.calculate_payoff(Strategy.RD, False, 1.0, 0.5)}")
    
    #test equation (17)
    print("\nEquation (17) - Probabilité de Nash:")
    for NR in [2, 5, 10]:
        p = game.nash_equilibrium_probability(0.5, 1.0, NR)
        print(f"  NR={NR}: p = {p:.4f}")
    
    #test equation (18)
    print("\nEquation (18) - Probabilité avec énergie:")
    Pk = game.probability_cluster_head_with_energy(0.3, 0.4, 0.3, 0.1)
    print(f"  Pk = {Pk:.4f}")
    
    #test equation (19)
    print("\nEquation (19) - Critère lambda:")
    lam = game.lambda_criterion(0.4, 0.5, 30.0)
    print(f"  lambda = {lam:.6f}")
    
    # 4 test des équilibres
    print("="*40)
    print("\n")
    print("4. Test des équilibres")
    
    
    #création d'une matrice de paiement simple
    payoff_mat = PayoffMatrix(
        player1=[[3, 0], [5, 1]],
        player2=[[2, 3], [1, 4]]
    )
    
    solver = EquilibriumSolver()
    pure_eq = solver.pure_nash_equilibrium(payoff_mat)
    print(f"\nEquilibres de Nash purs: {pure_eq}")
    
    # optimum de Pareto
    payoffs = [(3, 2), (0, 3), (5, 1), (1, 4)]
    pareto = solver.pareto_optimum(payoffs)
    print(f"Optima de Pareto: {pareto}")
    
    print("\n")
    print("TEST TERMINÉ AVEC SUCCES")
    
    return cluster_heads, selector

if __name__ == "__main__":
    cluster_heads, selector = run_test()