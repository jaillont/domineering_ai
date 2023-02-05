import random
import time
import numba

import numpy as np

from numba import jit  # jit convertit une fonction python => fonction C
from keras.models import load_model

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
#   IA rand 
#   Nous allons mettre en place l’IA de jeu la plus simple possible. En effet, cette IA va récupérer la liste des coups possibles et en choisir 1 au hasard.


def IARand(B):
    id = random.randint(0, B[-1]-1)
    idMove = B[id]

    Play(B,idMove)


def IA100P(B):

    # Si l'IA est joueur 0
    if B[-3] == 0:
        best_mean_score = -2
    # Si l'IA est joueur 1
    elif B[-3] == 1:
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
            Playout(B_sim)

            # Récupérer le score de la partie
            scores_sim[i] = GetScore(B_sim)

        # Calculer le score moyen obtenu pour le coup simulé
        mean_score = scores_sim.sum()/nb_sim

        # Si l'IA est joueur 0
        if B[-3] == 0:
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                idBestMove = B_sim[move]
        # Si l'IA est joueur 1
        elif B[-3] == 1:
            if mean_score < best_mean_score:
                best_mean_score = mean_score
                idBestMove = B_sim[move]

    Play(B, idBestMove)


@numba.jit(nopython=True, parallel=True)
def ParrallelPlayout(nb, B):
    Scores = np.empty(nb)
    for i in numba.prange(nb):
        Playout(B)
        Scores[i] = GetScore(B)
    return Scores.mean()


def IA1KP(B):

    # Si l'IA est joueur 0
    if B[-3] == 0:
        best_mean_score = -2
    # Si l'IA est joueur 1
    elif B[-3] == 1:
        best_mean_score = 2

    # Simuler tous les coups possibles
    for move in range(B[-1]):

        nb_sim = 1000

        # Créer une copie de la grille en cours
        B_sim = B.copy()
        
        # Jouer le coup que l'on veut simuler
        Play(B_sim, B_sim[move])

        # Simuler la fin de la partie
        mean_score = ParrallelPlayout(nb_sim, B_sim)

        # Si l'IA est joueur 0
        if B[-3] == 0:
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                idBestMove = B_sim[move]
        # Si l'IA est joueur 1
        elif B[-3] == 1:
            if mean_score < best_mean_score:
                best_mean_score = mean_score
                idBestMove = B_sim[move]

    Play(B, idBestMove)


def IA10KP(B):

    # Si l'IA est joueur 0
    if B[-3] == 0:
        best_mean_score = -2
    # Si l'IA est joueur 1
    elif B[-3] == 1:
        best_mean_score = 2

    # Simuler tous les coups possibles
    for move in range(B[-1]):

        nb_sim = 10000

        # Créer une copie de la grille en cours
        B_sim = B.copy()
        
        # Jouer le coup que l'on veut simuler
        Play(B_sim, B_sim[move])

        # Simuler la fin de la partie
        mean_score = ParrallelPlayout(nb_sim, B_sim)

        # Si l'IA est joueur 0
        if B[-3] == 0:
            if mean_score > best_mean_score:
                best_mean_score = mean_score
                idBestMove = B_sim[move]
        # Si l'IA est joueur 1
        elif B[-3] == 1:
            if mean_score < best_mean_score:
                best_mean_score = mean_score
                idBestMove = B_sim[move]

    Play(B, idBestMove)


###########################
###### UCT algorithm ######
###########################

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

    
    def _uct_value(self, coef=1.4):
        return (self.score / self.visits) + coef * (np.log(self.parent_node.visits) / self.visits) ** 0.5


    def best_child(self, max_formula):
        if max_formula:
            formula = max(self.child_nodes, key=lambda x: x._uct_value())
        else:
            formula = min(self.child_nodes, key=lambda x: x._uct_value())
        return formula


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

            _PossibleMoves(root.board_state[-3], root.board_state)
            possible_moves = root.board_state[:root.board_state[-1]]

            if len(possible_moves) == 0:
                break

            elif root.child_nodes == []:

                for move in possible_moves:
                    root.child_nodes.append(root.add_child(move))

            if not root.fully_expanded():

                for child in root.child_nodes:
                    result = child.simulate()
                    child.update(result)
                    root.update(result)

        # Profondeur 1
        else:

            for j in range(16):

                if j == 0:

                    _PossibleMoves(node.board_state[-3], node.board_state)
                    possible_moves = node.board_state[:node.board_state[-1]]

                    if len(possible_moves) == 0:
                        break

                    elif node.child_nodes == []:

                        for move in possible_moves:
                            node.child_nodes.append(node.add_child(move))

                    if not node.fully_expanded():

                        for child in node.child_nodes:
                            result = child.simulate()
                            child.update(result)
                            node.update(result)
                            root.update(result)

                # Profondeur 2
                else:
                    _PossibleMoves(node.board_state[-3], node.board_state)
                    possible_moves = node.board_state[:node.board_state[-1]]

                    if len(possible_moves) == 0:
                        break

                    elif node.child_nodes == []:

                        for move in possible_moves:
                            node.child_nodes.append(node.add_child(move))

                    if not node.fully_expanded():

                        for child in node.child_nodes:
                            result = child.simulate()
                            child.update(result)
                            node.parent_node.update(result)
                            node.update(result)
                            root.update(result)
                
                if board_state[-3] == 0:
                    node = node.best_child(max_formula = True)
                else:
                    node = node.best_child(max_formula = False)

        if board_state[-3] == 0:
            node = root.best_child(max_formula = True)
        else:
            node = root.best_child(max_formula = False)

    if board_state[-3] == 0:
        move = root.best_child(max_formula = True).move
    else:
        move = root.best_child(max_formula = False).move

    Play(board_state,move)


#######################
#### DEEP LEARNING ####
#######################

def find_best_action_deep(B):

    model = load_model('alexNet_model_2.h5')

    board = B[64:128].reshape(8,8)
    board = B[64:128].reshape(8,8)
    negative_board = 1 - board
    board = 1 - negative_board

    if B[-3]==0:
        X = np.concatenate((board, negative_board, np.zeros((8,8))), axis=1)
    elif B[-3]==1:
        X = np.concatenate((board, 1 - board, np.ones((8,8))), axis=1)

    X = X.reshape(-1, 8, 8, 3)
    predicted_output = model.predict(X)
    best_move = np.argmax(predicted_output,axis=1)[0]

    Play(B, best_move)


###########################
###### IA interface #######
###########################


def playout_IA_vs_IA(ia_0, ia_1):

    # On initialise la grille à chaque début de nouvelle partie
    B = StartingBoard.copy()

    # Tant qu'il n'y a pas de Game Over
    while not Terminated(B):

        # Si c'est au tour du joueur 0
        if B[-3] == 0:
            # On lance l'ia 0 pour qu'elle joue zon coup
            ia_0(B) 

        # Si c'est au tour du joueur 1
        elif B[-3] == 1:
            # On lance l'ia 1 pour qu'elle joue son coup
            ia_1(B)
    
    return GetScore(B)


def launch_n_games(n, ia_0, ia_1):
    # On déclare un Numpy Array à 0 pour stocker le score
    scores = np.zeros(n)

    for i in range(n):

        print('Game :', i)

        # On récupère le score final de la partie
        score = playout_IA_vs_IA(ia_0, ia_1)
        # On stcoke le résultat dans un Array Numpy dédié
        scores[i] = score

    return scores


nb_games = 100

T0 = time.time()
scores = launch_n_games(nb_games, find_best_action_deep, monte_carlo_tree_search)
#scores, data = MCTS_vs_ia(nb_games,find_best_action_deep)
T1 = time.time()
print("time:",T1-T0)

percentage_score_player_0 = scores[scores == 1].sum()*100/nb_games
percentage_score_player_1 = -scores[scores == -1].sum()*100/nb_games

print("Score player 0 :", percentage_score_player_0, '%')
print("Score player 1 :", percentage_score_player_1, '%\n')

if percentage_score_player_0 > percentage_score_player_1:
    print('WINNER : Player 0')
else:
    print('WINNER : Player 1')
