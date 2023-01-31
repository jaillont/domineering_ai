import numpy as np
import random
import time
import numba
from numba import jit, njit  # jit convertit une fonction python => fonction C
import math

###################################################################

# PLayer 0 => Vertical    Player
# PLayer 1 => Horizontal  Player

# IdMove : code servant à identifier un coup particulier sur le jeu
# P   : id player 0/1
# x,y : coordonnées de la tuile, Player0 joue sur (x,y)+(x,y+1) et Player1 sur (x,y)+(x+1,y)

# convert: player,x,y <=> IDmove

# IDmove=123 <=> player 1 plays at position x = 2 and y = 3
# ce codage tient sur 8 bits !

@jit(nopython=True)
def GetIDmove(player,x,y):
    return player * 100 + x * 10 + y

@jit(nopython=True)
def DecodeIDmove(IDmove):
    y = IDmove % 10
    x = int(IDmove/10) % 10
    player = int(IDmove / 100)
    return player,x,y

###################################################################

# Numba requiert des numpy array pour fonctionner

# toutes les données du jeu sont donc stockées dans 1 seul array numpy

# Data Structure  - numpy array de taille 144 uint8 :
# B[ 0- 63] List of possibles moves
# B[64-127] Gameboard (x,y) => 64 + x + 8*y
# B[-1] : number of possible moves
# B[-2] : reserved
# B[-3] : current player




StartingBoard  = np.zeros(144,dtype=np.uint8)

@jit(nopython=True)   # pour x,y donné => retourne indice dans le tableau B
# I play ? 
def iPxy(x,y):
    return 64 + 8 * y + x

@jit(nopython=True)
def _PossibleMoves(idPlayer,B):   # analyse B => liste des coups possibles par ordre croissant
    nb = 0

    #player V
    if idPlayer == 0 :
        for x in range(8):
            for y in range(7):
                p = iPxy(x,y)
                if B[p] == 0 and B[p+8] == 0 :
                    B[nb] = GetIDmove(0,x,y)
                    nb+=1
    # player H
    if idPlayer == 1 :
        for x in range(7):
            for y in range(8):
                p = iPxy(x,y)
                if B[p] == 0 and B[p+1] == 0 :
                    B[nb] = GetIDmove(1,x,y)
                    nb+=1

    B[-1] = nb

_PossibleMoves(0,StartingBoard)   # prépare le gameboard de démarrage


###################################################################

# Numba ne gère pas les classes...

# fonctions de gestion d'une partie
# les fonctions sans @jit ne sont pas accélérées

# Player 0 win => Score :  1
# Player 1 win => Score : -1


# def CreateNewGame()   => StartingBoard.copy()
# def CopyGame(B)       => return B.copy()

@jit(nopython=True)
#end of the game
def Terminated(B):
    return B[-1] == 0

@jit(nopython=True)
def GetScore(B):
    if B[-2] == 10 : return  1
    if B[-2] == 20 : return -1
    return 0


@jit(nopython=True)
def Play(B,idMove):
    player,x,y = DecodeIDmove(idMove)
    p = iPxy(x,y)

    B[p]   = 1
    if player == 0 : B[p+8] = 1
    else :           B[p+1] = 1

    nextPlayer = 1 - player

    _PossibleMoves(nextPlayer,B)
    B[-3] = nextPlayer

    if B[-1] == 0  :             # gameover
        B[-2] = (player+1)*10    # player 0 win => 10  / player 1 win => 20
    #rajouter dans la liste ou retourner p,x,y


@jit(nopython=True)
def Playout(B):
    while B[-1] != 0:                   # tant qu'il reste des coups possibles
        id = random.randint(0,B[-1]-1)  # select random move
        idMove = B[id]
        Play(B,idMove)


##################################################################
#
#   for demo only - do not use for computation

def Print(B):
    for yy in range(8):
        y = 7 - yy
        s = str(y)
        for x in range(8):
            if     B[iPxy(x,y)] == 1 : s += '::'
            else:                      s += '[]'
        print(s)
    s = ' '
    for x in range(8): s += str(x)+str(x)
    print(s)


    nbMoves = B[-1]
    print("Possible moves :", nbMoves);
    s = ''
    for i in range(nbMoves):
        s += str(B[i]) + ' '
    print(s)



def PlayoutDebug(B,verbose=False):
    Print(B)
    while not Terminated(B):
        id = random.randint(0,B[-1]-1)
        idMove = B[id]
        player,x,y = DecodeIDmove(idMove)
        print("Playing : ",idMove, " -  Player: ",player, "  X:",x," Y:",y)
        Play(B,idMove)
        Print(B)
        print("---------------------------------------")


################################################################
#
#  Version Debug Demo pour affichage et test

#B = StartingBoard.copy()
#PlayoutDebug(B,True)
#print("Score : ",GetScore(B))
#print("")


################################################################
#
#   utilisation de numba => 100 000 parties par seconde

#print("Test perf Numba")

#T0 = time.time()
#nbSimus = 0
#while time.time()-T0 < 2:
#    B = StartingBoard.copy()
#    Playout(B)
#    nbSimus+=1
#print("Nb Sims / second:",nbSimus/2)


################################################################
#
#   utilisation de numba +  multiprocess => 1 000 000 parties par seconde

#print()
#print("Test perf Numba + parallélisme")

@numba.jit(nopython=True, parallel=True)
def ParrallelPlayout(nb):
    Scores = np.empty(nb)
    for i in numba.prange(nb):
        B = StartingBoard.copy()
        Playout(B)
        Scores[i] = GetScore(B)
    return Scores.mean()



#nbSimus = 10 * 1000 * 1000
#T0 = time.time()
#MeanScores = ParrallelPlayout(nbSimus)
#T1 = time.time()
#dt = T1-T0

#print("Nb Sims / second:", int(nbSimus / dt ))


################################################################
#   IA rand 
#   Nous allons mettre en place l’IA de jeu la plus simple possible. En effet, cette IA va récupérer la liste des coups possibles et en choisir 1 au hasard.











@jit(nopython=True)
def IARand(B, player_0=False):
    id = random.randint(0, B[-1]-1)
    idMove = B[id]
    
    return idMove


@jit(nopython=True)
def IA100P(B, player_0):

    if player_0:
        best_mean_score = -2
    else:
        best_mean_score = 2

    # Simuler tous les coups possibles
    for move in range(B[-1]):

        nb_sim = 100

        # Initialiser la grille de score à chaque simulation d'un nouveau coup
        scores_sim = np.zeros(nb_sim)
    
        # Simuler nb_sim fois chaque coup
        for i in range(nb_sim):

            # Créer une copie de la grille en cours
            B_sim = B.copy()
            
            # Jouer le coup que l'on veut simuler
            Play(B_sim, B_sim[move])

            # Simuler la fin de la partie
            score_sim = playout_IA_vs_IA(B_sim, IARand, IARand)
            scores_sim[i] = score_sim

        # Calculer le score moyen obtenu pour le coup simulé
        mean_score = scores_sim.sum()/nb_sim

        if player_0:
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                idBestMove = B_sim[move]
        else:
            if mean_score < best_mean_score:
                best_mean_score = mean_score
                idBestMove = B_sim[move]

    return idBestMove



@njit(nopython=True, parallel=True)
def IA1KP(B, player_0):

    if player_0:
        best_mean_score = -2
    else:
        best_mean_score = 2

    # Simuler tous les coups possibles
    for move in numba.prange(B[-1]):

        nb_sim = 1000

        # Initialiser la grille de score à chaque simulation d'un nouveau coup
        scores_sim = np.zeros(nb_sim)
    
        # Simuler nb_sim fois chaque coup
        for i in numba.prange(nb_sim):

            # Créer une copie de la grille en cours
            B_sim = B.copy()
            
            # Jouer le coup que l'on veut simuler
            Play(B_sim, B_sim[move])

            # Simuler la fin de la partie
            score_sim = playout_IA_vs_IA(B_sim, IARand, IARand)
            scores_sim[i] = score_sim

        # Calculer le score moyen obtenu pour le coup simulé
        mean_score = scores_sim.sum()/nb_sim

        if player_0:
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                idBestMove = B_sim[move]
        else:
            if mean_score < best_mean_score:
                best_mean_score = mean_score
                idBestMove = B_sim[move]

    return idBestMove


@njit(nopython=True, parallel=True)
def IA10KP(B, player_0):

    if player_0:
        best_mean_score = -2
    else:
        best_mean_score = 2

    for move in numba.prange(B[-1]):

        nb_sim = 10000
        scores_sim = np.zeros(nb_sim)

        for i in numba.prange(nb_sim):

            # Créer une copie de la grille en cours
            B_sim = B.copy()
            
            # Jouer le coup que l'on veut simuler
            Play(B_sim, B_sim[move])

            score_sim = playout_IA_vs_IA(B_sim, IARand, IARand)
            scores_sim[i] = score_sim
        
        # Calculer le score moyen obtenu pour le coup simulé
        mean_score = scores_sim.sum()/nb_sim

        if player_0:
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                idBestMove = B[move]
        else:
            if mean_score < best_mean_score:
                best_mean_score = mean_score
                idBestMove = B[move]

    return idBestMove


@jit(nopython=True)
def playout_IA_vs_IA(B, ia_0, ia_1):
    # On affiche la grille initialisée
    #Print(B)
    # Tant qu'il n'y a pas de Game Over
    while not Terminated(B):
        # Si c'est au tour du joueur 0
        if B[-3] == 0:
            # On lance l'ia 0 pour qu'elle choisisse zon coup
            idMove = ia_0(B, player_0=True) 

        # Si c'est au tour du joueur 1
        elif B[-3] == 1:
            # On lance l'ia 1 pour qu'elle choisisse son coup
            idMove = ia_1(B, player_0=False)

        # On jour le coup choisit sur la grille, et on actualise en conséquence
        Play(B,idMove)

        # On affiche la nouvelle grille
        #Print(B)
    
    return GetScore(B)


@jit(nopython=True)
def launch_n_games(n, ia_0, ia_1):
    # On déclare un Numpy Array à 0 pour stocker le score
    scores = np.zeros(n)

    for i in range(n):
        # On initialise la grille à chaque début de nouvelle partie
        B = StartingBoard.copy()

        print('Game :', i)

        # On récupère le score final de la partie
        score = playout_IA_vs_IA(B, ia_0, ia_1)
        # On stcoke le résultat dans un Array Numpy dédié
        scores[i] = score

    return scores










@jit(nopython=True)
def UCT(mean_score,parent_n,n, player_0):
    coef=0.1
    if player_0:
        idx = mean_score+coef*np.sqrt(np.log(parent_n)/n)
    else:
        idx = -mean_score+coef*np.sqrt(np.log(parent_n)/n)
    return idx


#@jit("i8(i8[:],i8,i8,i8[:],i8[:],i8[:])")#float64(int32, int32))
def MCTS(B,n_turn,game_n,game_tree,games_trees,games_scores, player_0):
    #Contrôle si il s'agit d'un début de partie ou d'une nouvelle série de partie
    #create the array that will contain this game nodes
    #print("gametrees",games_trees)
    #game_nodes=np.zeros(0)
    #print(B[3])
    #print(game_n)
    best_move_idx=IARand(B,False)

    if game_n < 100 or n_turn==0: #or games_trees == np.zeros(0):
        best_move_idx=IARand(B,False)
        #print("hello")
        #print(n_turn)
        #print(games_trees)
    else :
        #print("game n:", game_n)
        #print("game_tree:", game_tree)
        best_move_value=0
        for move in range(B[-1]):
            parent_node = game_tree[-1]
            parent_node_count = 0
            node_scores=np.empty(0)
            current_node_count=0
            for i in range(games_trees.shape[0]):
                t=False
                sub_arr = games_trees[i]
                for x in range(sub_arr.shape[0]): 
                    if sub_arr[x] == parent_node:
                        parent_node_count = parent_node_count + 1
                        t=True
                    if sub_arr[x] == B[move]:
                        current_node_count = current_node_count + 1 
                        #print("current node",current_node_count) 
                    if t:
                        node_scores = np.append(node_scores, games_scores[i])
                        #print("node_scores",node_scores)
                        node_mean_score = np.mean(node_scores)
                #if current_node_count != 0 and t:
                    #print("mean",node_scores,"parent_n",parent_node_count,"n",current_node_count)
                #if current_node_count != 0 and not t:
                #    pass
            value=UCT(mean_score=node_mean_score,parent_n=parent_node_count,n=current_node_count, player_0=player_0)
                    #print("value:",value)

            if player_0:
                if value > best_move_value:
                    best_move_idx = B[move]
                    best_move_value = value
            else:
                if value < best_move_value:
                    best_move_idx = B[move]
                    best_move_value = value
    #remove unused indexes from the array
    #Noeuds
    return best_move_idx


#################
### UCT logic ###
#################

class Node:
    
    def __init__(self, board_state, parent_node=None):

        self.board_state = board_state.copy()

        self.score = 0
        self.visits = 0

        self.child_nodes = []
        self.parent_node = parent_node

        self.move = None

    
    def add_child(self, move):
        child = Node(self.board_state.copy(), self)
        Play(child.board_state, move)
        child.move = move
        self.child_nodes.append(child)
        return child

    
    def update(self, result):
        self.visits += 1
        self.score += result


    def fully_expanded(self):
        return all(child.visits > 0 for child in self.child_nodes) and len(self.child_nodes) > 0

    
    def _uct_value(self, coef=1.41421356237):
        return (self.score / self.visits) + coef * (np.log(self.parent_node.visits) / self.visits) ** 0.5


    def best_child(self):
        return max(self.child_nodes, key=lambda x: x._uct_value())


    def simulate(self):
        board_state = self.board_state.copy()
        while not Terminated(board_state):
            # select a random move from the list of possible moves
            idx = random.randint(0, board_state[-1]-1)
            id_move = board_state[idx]
            Play(board_state, id_move)
        # return the score of the game
        return GetScore(board_state)


def monte_carlo_tree_search(board_state):
    root = Node(board_state)

    for i in range(16):

        # Initialisation de l'arbre
        if i == 0:
            if root.child_nodes == []:
                _PossibleMoves(root.board_state[-3], root.board_state)
                possible_moves = root.board_state[:root.board_state[-1]]

                for move in possible_moves:
                    root.child_nodes.append(root.add_child(move))

            if not root.fully_expanded():

                for child in root.child_nodes:
                    result = child.simulate()
                    child.update(result)
                    root.update(result)

        # Profondeur 1
        else:
            if node.child_nodes == []:
                _PossibleMoves(node.board_state[-3], node.board_state)
                possible_moves = node.board_state[:node.board_state[-1]]

                for move in possible_moves:
                    node.child_nodes.append(node.add_child(move))

            if not node.fully_expanded():

                for child in node.child_nodes:
                    result = child.simulate()
                    child.update(result)
                    node.update(result)
                    root.update(result)

        node = root.best_child()

    move = root.best_child().move

    return move

#@overload(np.array)
def MCTS_vs_ia(n_games,ia_1):

    games_scores = np.zeros(n_games)
    data = []

    for i in range(n_games):
        print("game n:",i)
        B = StartingBoard.copy()

        nb_UCT_player = 0

        if nb_UCT_player == 0:
            while B[-1] != 0:
            # Si c'est au tour du joueur 0
                if B[-3] == 0:
                    id_move = monte_carlo_tree_search(B.copy())

            # Si c'est au tour du joueur 1
                elif B[-3] == 1:
                # On lance l'ia 1 pour qu'elle choisisse son coup
                    id_move = ia_1(B.copy(), player_0=False)

                # Get game data
                board = B[64:128].reshape(8,8)
                negative_board = 1 - board
                board = 1 - negative_board
                player, x, y = DecodeIDmove(id_move)
                player_plan = [[player]*8]*8
                data.append([board, negative_board, player_plan, id_move])

                # On jour le coup choisit sur la grille, et on actualise en conséquence
                Play(B,id_move)

            games_scores[i]=GetScore(B)


            # On affiche la nouvelle grille
            #Print(B)

        elif nb_UCT_player == 1:
            while B[-1] != 0:
                # Si c'est au tour du joueur 0
                if B[-3] == 0:
                    # On lance l'ia 1 pour qu'elle choisisse son coup
                    id_move = ia_1(B, player_0=True)

                # Si c'est au tour du joueur 1
                elif B[-3] == 1:
                    board_state = B
                    id_move = monte_carlo_tree_search(board_state)

                # Get game data
                board = B[64:128].reshape(8,8)
                negative_board = 1 - board
                board = 1 - negative_board
                player, x, y = DecodeIDmove(id_move)
                player_plan = [[player]*8]*8
                data.append([board, negative_board, player_plan, id_move])

                # On jour le coup choisit sur la grille, et on actualise en conséquence
                Play(B,id_move)

            #print(games_trees.shape[0])
            #print(games_trees[i])
            games_scores[i]=GetScore(B)
    
    return games_scores, data




nb_games = 10

T0 = time.time()
#scores = launch_n_games(nb_games, IA1KP, IARand)
scores, data = MCTS_vs_ia(nb_games,IARand)
T1 = time.time()
print("time:",T1-T0)

#print("Scores :", scores, '\n')
#scores=scores[100:]
percentage_score_player_0 = scores[scores == 1].sum()*100/nb_games
percentage_score_player_1 = -scores[scores == -1].sum()*100/nb_games

print("Score player 0 :", percentage_score_player_0, '%')
print("Score player 1 :", percentage_score_player_1, '%\n')

if percentage_score_player_0 > percentage_score_player_1:
    print('WINNER : Player 0')
else:
    print('WINNER : Player 1')


import pickle

with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)
