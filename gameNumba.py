import numpy as np
import random
import time
import numba
from numba import jit  # jit convertit une fonction python => fonction C

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


@jit(nopython=True)
def IA1KP(B, player_0):

    if player_0:
        best_mean_score = -2
    else:
        best_mean_score = 2

    # Simuler tous les coups possibles
    for move in range(B[-1]):

        nb_sim = 1000

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


# @jit(nopython=True, parallel=True)
# def IA10KP(B, player_0):

#     if player_0:
#         best_mean_score = -2
#     else:
#         best_mean_score = 2

#     # Simuler tous les coups possibles
#     for move in numba.prange(B[-1]):

#         nb_sim = 10000

#         # Initialiser la grille de score à chaque simulation d'un nouveau coup
#         scores_sim = np.empty(nb_sim)
    
#         # Simuler nb_sim fois chaque coup
#         for i in range(nb_sim):

#             # Créer une copie de la grille en cours
#             B_sim = B.copy()
            
#             # Jouer le coup que l'on veut simuler
#             Play(B_sim, B_sim[move])

#             # Simuler la fin de la partie
#             while not Terminated(B_sim):
#                 if B[-3] == 0:
#                     sim_move = IARand(B_sim)
#                 else:
#                     sim_move = IARand(B_sim)
#                 Play(B_sim, sim_move)
            
#             score_sim = GetScore(B_sim)
#             scores_sim[i] = score_sim

#         # Calculer le score moyen obtenu pour le coup simulé
#         mean_score = scores_sim.sum()/nb_sim

#         if mean_score > best_mean_score:
#             best_mean_score = mean_score
#             idBestMove = B_sim[move]

#     return idBestMove

from numba import njit

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



nb_games = 1000

T0 = time.time()
scores = launch_n_games(nb_games, IA10KP, IARand)
T1 = time.time()
print("time:",T1-T0)

print("Scores :", scores, '\n')

percentage_score_player_0 = scores[scores == 1].sum()*100/nb_games
percentage_score_player_1 = -scores[scores == -1].sum()*100/nb_games

print("Score player 0 :", percentage_score_player_0, '%')
print("Score player 1 :", percentage_score_player_1, '%\n')

if percentage_score_player_0 > percentage_score_player_1:
    print('WINNER : Player 0')
else:
    print('WINNER : Player 1')




























# @jit(nopython=True)
# def IaRand(B):
#     id = random.randint(0,B[-1]-1)  # select random move
#     idMove = B[id]
#     return idMove


# @jit(nopython=True)
# def IA100P(B):
#     #boucle sur les coups possibles
#     if B[-1] > 1:
#         test_min_score=-1
#         for each in range(B[-1]):
#             #print(each)
#             #nouvelle array des résultats de chaque simulation
#             results_array=np.zeros(100)
#             #boucle sur les 100 simulations
#             for i in range(100):
#                 B_sim = B.copy()
#                 Play(B_sim,B[each])
#                 playout_ia_vs_ia(B_sim, IaRand, IA100P)
#                 results_array[i] = GetScore(B_sim)
#             #moyenne les 100 parties à venir et sauvegarde pour comparaison
#             mean_score=results_array.mean()
#             #meilleur moyenne
#             if mean_score > test_min_score:
#                 test_min_score = mean_score
#                 idBestMove= B[each]
#         #joue le coup correspondant
#         #Play(B,idBestMove)
#     else:
#         id = random.randint(0,B[-1]-1)  # select random move
#         idBestMove = B[id]
#     return idBestMove

# @jit(nopython=True, parallel=True)
# def IA1KP(B):
#     if B[-1] <= 3:
#         #boucle sur les coups possibles
#         test_min_score=-1
#         for each in range(B[-1]):
#             #nouvelle array des résultats de chaque simulation
#             results_array=np.zeros(1000)
#             #boucle sur les 100 simulations
#             for i in numba.prange(1000):
#                 B_sim = B.copy()
#                 Play(B_sim,B[each])
#                 playout_ia_vs_ia(B_sim,IaRand,IaRand)
#                 results_array[i] = GetScore(B_sim)
#             #moyenne les 100 parties à venir et sauvegarde pour comparaison
#             mean_score=results_array.mean()
#             #meilleur moyenne
#             if mean_score > test_min_score:
#                 test_min_score = mean_score
#                 idBestMove= B[each]
#         #joue le coup correspondant
#         #Play(B,idBestMove)
#     else:
#         idBestMove= B[random.randint(0,B[-1]-1)]
#     return idBestMove

# @jit(nopython=True,parallel=True)
# def IA10KP(B):
#     if B[-1] <= 3:
#         #boucle sur les coups possibles
#         test_min_score=-1
#         for each in range(B[-1]):
#             #nouvelle array des résultats de chaque simulation
#             results_array=np.zeros(10000)
#             #boucle sur les 100 simulations
#             for i in range(10000):
#                 B_sim = B.copy()
#                 Play(B_sim,B[each])
#                 playout_ia_vs_ia(B_sim,IA100P,IaRand)
#                 results_array[i] = GetScore(B_sim)
#             #moyenne les 100 parties à venir et sauvegarde pour comparaison
#             mean_score=results_array.mean()
#             #meilleur moyenne
#             if mean_score > test_min_score:
#                 test_min_score = mean_score
#                 idBestMove= B[each]
#         #joue le coup correspondant
#         #Play(B,idBestMove)
#     else:
#         idBestMove= B[random.randint(0,B[-1]-1)]
#     return idBestMove

# @jit(nopython=True)
# def playout_ia_vs_ia(B,ia_0,ia_1):
#     #Print(B)
#     while not Terminated(B):
#         #si c'est le joueur 0
#         if B[-3] == 0:
#             #choix de l'ia
#             idMove=ia_0(B)

#         #si c'est le joueur 1
#         elif B[-3]==1:
#             idMove=ia_1(B)

#         #print("Playing : ",idMove, " -  Player: ",player, "  X:",x," Y:",y)
#         Play(B,idMove)
#         #Print(B)
#         #print("---------------cs-----------------------")


# @jit(nopython=True)
# def launch_n_games(n,player_0,player_1):
#     scores = np.zeros(n)
#     for i in range(n):
#         #if i % 50 ==0:
#         print("game n°",i)
#         B = StartingBoard.copy()
#         playout_ia_vs_ia(B,player_0,player_1)
#         scores[i] = GetScore(B)
#     player_0_purcentage = np.count_nonzero(scores == 1)/n*100
#     player_1_purcentage = np.count_nonzero(scores == -1)/n*100
#     print("player 0 won",player_0_purcentage,"% of the",n,"games")
#     print("player 1 won",player_1_purcentage,"% of the",n,"games")

#     print("Mean score: ",np.mean(scores))

# launch_n_games(100,IaRand,IA100P)