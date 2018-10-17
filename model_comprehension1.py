#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from multiprocessing.pool import ThreadPool
import pickle
import sys

"""
This is the code of the main project of human memory simulation while they are reading a text. 
The idea is to search the closest neighbors of the word that has just been read and put them in the working memory. 
Then, we delete the less convinent words present in the memory
and repeat.
"""

def data_extraction(data):
    """ Extrait une table de correspondance index-mot ainsi que les coordonnées(LSA) de chacun des mots
        contenus dans le vocabulaire"""
    data_iterator = data.iterrows()
    word_coordonates = []
    index_table = dict()
    row = next(data_iterator)
    end = False
    index = 0

    while(not end):
        line = row[1][0].split(" ")
        coordonates = np.array(line[1:]).astype(float)
        word_coordonates.append(coordonates)
        word = line[0]
        index_table[word] = index
        try:
            row = next(data_iterator)
            index += 1
        except StopIteration:
            end = True

    return index_table, word_coordonates


# def get_closest_neighbors(nb_neighbors, word_id, word_coordonates, working_memory, neighbors_cache, word_norms, seuil_add):
#     """Retourne les N mots les plus proche NON PRESENTS EN WM dans l'espace sémantique issu du LSA,
#        Triés par pertinence
#        version avec ajout voisins supplémentaires importants
#        """
#     if word_id not in neighbors_cache.keys(): #Voisins pas en cache
#         closest_neighbors = [(0, 0) for _ in range(nb_neighbors + 2)] # (index, score)
#         for index_id, coord in enumerate(word_coordonates):
#             tuple_score_min = min(closest_neighbors, key = lambda t: t[1])
#             score = get_correspondance_score(word_id, index_id, word_coordonates, word_norms)
#             if score > tuple_score_min[1]:
#                 closest_neighbors.remove(tuple_score_min)
#                 closest_neighbors.append((index_id, score))
#
#         closest_neighbors = sorted(closest_neighbors, key= lambda t: -t[1])
#         #On traite la valeur d'activation totale
#         closest_neighbors_final = closest_neighbors[:nb_neighbors + 1]
#         for voisin in closest_neighbors[nb_neighbors + 1:]:
#             if voisin[1] > seuil_add:
#                 closest_neighbors_final.append(voisin)
#         neighbors_cache[word_id] = closest_neighbors_final
#     else:#On utilise le cache
#         closest_neighbors = neighbors_cache[word_id]
#         closest_neighbors_final = closest_neighbors[:nb_neighbors + 1]
#         for voisin in closest_neighbors[nb_neighbors + 1:]:
#             if voisin[1] > seuil_add:
#                 closest_neighbors_final.append(voisin)
#     print(len(closest_neighbors_final))
#     return closest_neighbors_final

# def get_closest_neighbors(nb_neighbors, word_id, word_coordonates, working_memory, neighbors_cache, word_norms, activation_total):
#     """Retourne les N mots les plus proche NON PRESENTS EN WM dans l'espace sémantique issu du LSA,
#        Triés par pertinence
#        version avec quantité d'activation
#        """
#     if word_id not in neighbors_cache.keys(): #Voisins pas en cache
#         closest_neighbors = [(0, 0) for _ in range(nb_neighbors + 1)] # (index, score)
#         for index_id, coord in enumerate(word_coordonates):
#             tuple_score_min = min(closest_neighbors, key = lambda t: t[1])
#             score = get_correspondance_score(word_id, index_id, word_coordonates, word_norms)
#             if score > tuple_score_min[1]:
#                 closest_neighbors.remove(tuple_score_min)
#                 closest_neighbors.append((index_id, score))
#
#         closest_neighbors = sorted(closest_neighbors, key= lambda t: -t[1])
#         #On traite la valeur d'activation totale
#         closest_neighbors_final = []
#         while activation_total > 0 and closest_neighbors != []:
#             current = closest_neighbors.pop(0)
#             activation_total -= current[1]
#             closest_neighbors_final.append(current)
#         neighbors_cache[word_id] = closest_neighbors_final
#     else:#On utilise le cache
#         closest_neighbors = neighbors_cache[word_id]
#         closest_neighbors_final = []
#         #juste pour trouver bon param.
#         i = 0
#         while activation_total > 0 and i < len(closest_neighbors):
#             current = closest_neighbors[i]
#             activation_total -= current[1]
#             closest_neighbors_final.append(current)
#             i +=1
#     print(len(closest_neighbors_final))
#     return closest_neighbors_final

def get_closest_neighbors(nb_neighbors, word_id, word_coordonates, working_memory, neighbors_cache, word_norms):
    """Retourne les N mots les plus proche dans l'espace sémantique issu du LSA,
       Triés par pertinence"""
    if word_id not in neighbors_cache.keys(): #Voisins pas en cache
        closest_neighbors = [(0, -1) for _ in range(nb_neighbors + 1)] # (index, score)
        for index_id, coord in enumerate(word_coordonates):
            tuple_score_min = min(closest_neighbors, key = lambda t: t[1])
            score = get_correspondance_score(word_id, index_id, word_coordonates, word_norms)
            if score > tuple_score_min[1]:
                closest_neighbors.remove(tuple_score_min)
                closest_neighbors.append((index_id, score))

        closest_neighbors = sorted(closest_neighbors, key= lambda t: -t[1])
        neighbors_cache[word_id] = closest_neighbors
    else:#On utilise le cache
        closest_neighbors = neighbors_cache[word_id]
    return closest_neighbors



def get_correspondance_score(word_id1, word_id2, word_coordonates, word_norms):
    """Calcul le score de similarité entre deux mots données (Cosinus)"""
    return  np.dot(word_coordonates[word_id1], word_coordonates[word_id2])/ (word_norms[word_id1]*word_norms[word_id2])



def get_word_id(index_table, word):
    """Retourne l'ID d'un mot donné, Erreur si le mot n'est pas connu"""
    if word in index_table.keys():
        return index_table[word]
    else:
        return -1


def get_word(index_table, word_id):
    """Retourne le mot associé à un identifiant, Erreur si l'identifiant n'est pas correct"""
    if word_id in index_table.values():
        for word, word_id_dict in index_table.items():
            if word_id == word_id_dict:
                return word
    else:
        raise ValueError('The word_id: ' + word_id + ' is not in the index_table !')



def integration(closest_neighbors, working_memory, episodic_memory, episodic_neighbors, word_coordonates, epsilon, activation_value_min, max_size_wm, word_norms):
    """Réalise le processus de diffusion d'information entre les mots déja en WM, le mot courant et ses voisins
       sémantiques.
       Retourne le nouvelle état de la WM 'épuré'. """
    all_word = closest_neighbors + working_memory + episodic_neighbors
    working_memory_ids = [working_memory[i][0] for i in range(len(working_memory))]
    # print("\nTaille matrice de correspondance : " + str(len(all_word)) + "x" + str(len(all_word)))

    # BUILD CORRESPONDANCE MATRIX
    correspondance_matrix = np.zeros((len(all_word), len(all_word)))
    for index1, word1 in enumerate(all_word):
        for index2, word2 in enumerate(all_word):
            correspondance_matrix[index1][index2] = get_correspondance_score(word1[0], word2[0], word_coordonates, word_norms)


    # INTEGRATION CYCLES
    propagation_vector = np.ones(len(all_word))/len(all_word)
    old_propagation_vector = np.ones(len(all_word))/len(all_word)
    first_cycle = True
    number_of_cycle = 0
    while(first_cycle or np.linalg.norm(propagation_vector - old_propagation_vector) > epsilon):
        first_cycle = False
        number_of_cycle += 1
        old_propagation_vector = propagation_vector
        propagation_vector = np.matmul(correspondance_matrix, old_propagation_vector)
        propagation_vector = propagation_vector / np.max(propagation_vector) #normalisation du vecteur: max à 1

    # print("Number of integration cycle : " + str(number_of_cycle))

    # SELECTION OF BEST SCORE WORDS
    new_working_memory = []
    for index, activation_value in enumerate(propagation_vector):
        if activation_value > activation_value_min: #Mot retenu pour la mémoire de travail !
            if (all_word[index][0], activation_value) not in new_working_memory: #Elimine les doublons qui vient de closest neighbors
                new_working_memory.append((all_word[index][0], activation_value))
        elif all_word[index][0] in working_memory_ids: #Mot anciennement présent en WM qui va donc en EM !
            insert_episodic_memory(episodic_memory, (all_word[index][0], activation_value))

    new_working_memory = sorted(new_working_memory, key= lambda t: -t[1])
    for word in new_working_memory[max_size_wm:]: #on ajoute à l'EM le surplus de mots selectionnés et qui étaient en WM
        if word[0] in working_memory_ids:
            insert_episodic_memory(episodic_memory, word)

    new_working_memory = new_working_memory[:max_size_wm] #On ne garde que les max_size_wm premier
    return new_working_memory

def display(index_table, working_memory):
    """ Fonction d'affichage de la mémoire de travail"""

    print("\n###### Etat de la mémoire de travail ######")
    print("\nNombre de mot : " + str(len(working_memory)))
    for word in working_memory:
        print("Mot : " + get_word(index_table, word[0]) + "["+str(word[0])+"]"+ "  .. activation value : " + str(word[1]))
    print("\n")

def display_episodic(index_table, episodic_memory):
    """ Fonction d'affichage de la mémoire épisodique"""

    print("\n###### Etat de la mémoire épisodique ######")
    print("\nNombre de mot : " + str(len(episodic_memory)))
    for word in episodic_memory:
        print("Mot : " + get_word(index_table, word[0]) + "["+str(word[0])+"]"+ "  .. activation value : " + str(word[1]))
    print("\n")

def display_neighbors(index_table, neighbors):
    """ Fonction d'affichage des voisins"""

    print("\nNombre de mot : " + str(len(neighbors)))
    for word in neighbors:
        print("Mot: " + get_word(index_table, word[0]) + "["+str(word[0])+"]"+ " .. cosinus value : " + str(word[1]))
    print("\n")

def lemmatisation(sentence, lemme_table):
    """ Traitement du texte à traiter. Gestion des majuscules, caractères spéciaux.
        Puis, lématisation des mots obtenus car corpus de base lémmatisé."""
    sentence = sentence.lower().replace(';', ' ').replace("'", ' ').replace(",", ' ') \
    .replace(".", ' ').replace("(", ' ').replace(")", ' ').replace(":", ' ') \
    .replace("(", ' ').replace('"', ' ').split(' ')

    sentence_lemmatised = []
    for indice, word in enumerate(sentence):
        if word in lemme_table.keys():
            # sentence_lemmatised.append(lemme_table[word][0])

            for lemme in lemme_table[word]:
                sentence_lemmatised.append(lemme) #remplacement du mot par son (ses) lemme(s)
        else:
            sentence_lemmatised.append(word)

    return sentence_lemmatised

def lemmatisation2(sentence, lemme_table):
    """ Traitement du texte à traiter. Gestion des majuscules, caractères spéciaux.
        Puis, lématisation des mots obtenus car corpus de base lémmatisé."""
    sentence = sentence.lower().replace(';', ' ').replace("'", ' ').replace(",", ' ') \
    .replace(".", ' ').replace("(", ' ').replace(")", ' ').replace(":", ' ') \
    .replace("(", ' ').replace('"', ' ').replace('?',' ').replace('!', ' ').split(' ')

    sentence_lemmatised = []
    for indice, word in enumerate(sentence):
        if word in lemme_table.keys():
            sentence_lemmatised.append(lemme_table[word][0])

            # for lemme in lemme_table[word]:
            #     sentence_lemmatised.append(lemme) #remplacement du mot par son (ses) lemme(s)
        else:
            sentence_lemmatised.append(word)

    return sentence_lemmatised

def lemme_extraction(data_lemme):
    """Création du dictionnaire "dict[mot] = lemme" utilisé pour la lemmatisation du texte"""
    lemme_table = dict()
    for index, row in data_lemme.iterrows():
        if row['1_ortho'] in lemme_table.keys():
            if row['3_lemme'] not in lemme_table[row['1_ortho']]: #Pas de lemme en doublon
                lemme_table[row['1_ortho']].append(row['3_lemme'])
        else:
            lemme_table[row['1_ortho']] = [row['3_lemme']]
    return lemme_table

def get_episodic_neighbors(episodic_memory, word_coordonates, word_id, activation_value_min_episodic, correlation_value_min_episodic, word_norms):
    """Retourne les mots en EM qui ont une valeur d'activation suffisante et une correlation élevée avec le mot courant"""
    episodic_neighbors = []
    for word in episodic_memory:
        if word[1] >= activation_value_min_episodic and get_correspondance_score(word_id, word[0], word_coordonates, word_norms) >= correlation_value_min_episodic:
            episodic_neighbors.append(word)
    return episodic_neighbors


def insert_episodic_memory(episodic_memory, word):
    """Insertion d'un concept en mémoire épisodique. Sinon, si le concept est déja présent on augmente sa valeur d'activation. Sinon, on insert le concept tel quel"""
    # correspondance_tmp = [get_correspondance_score(word_id[0], word[0], word_coordonates) for word_id in episodic_memory]
    # print(correspondance_tmp)
    if word in episodic_memory:
     # exactement le meme mot avec le meme score: vient d'un doublons des closest neighbors donc on le passe
        return
    episodic_memory_ids = [episodic_memory[i][0] for i in range(len(episodic_memory))]
    if word[0] in episodic_memory_ids:
        #le concept à ajouté est déja présent en EM, on augmente sa valeur d'activation par valueWM + valueEM*(1-valueWM)
        for wordin in episodic_memory:
            if wordin[0] == word[0]:
                new_wordin = (word[0], word[1] + wordin[1]*(1-word[1]))
                episodic_memory.remove(wordin)
                episodic_memory.append(new_wordin)
                break
    else:
        #le concept n'est pas présent, on l'ajoute tel quel
        episodic_memory.append(word)




def decay(episodic_memory, decay_rate):
    """ Applique une coefficient aux valeurs d'activation de la mémoire épisodique pour simuler de l'oubli'"""
    new_episodic_memory = []
    for word in episodic_memory:
        new_episodic_memory.append((word[0], word[1] * decay_rate))
    return new_episodic_memory

def pickle_loader(filename, id):
    """ Utilise pickle pour charger un objet sérialisé """
    return pickle.load(open(filename, 'rb'))

def global_data_loader(corpus):
    """ Charge les données nécessaires au programme en parallèle (from FB ou Le Monde)"""
    pool1 = ThreadPool(processes=1)
    pool2 = ThreadPool(processes=1)
    pool3 = ThreadPool(processes=1)
    pool4 = ThreadPool(processes=1)
    async_result1 = pool1.apply_async(pickle_loader, ('lemme_table.data', '0'))
    if corpus == "lemonde":
        async_result2 = pool2.apply_async(pickle_loader, ('index_table.data', '1'))
        async_result3 = pool3.apply_async(pickle_loader, ('word_coordonates.data', '2'))
        async_result4 = pool4.apply_async(pickle_loader, ('word_norms.data', '3'))
    elif corpus == "fb":
        async_result2 = pool2.apply_async(pickle_loader, ('index_table_fb.data', '1'))
        async_result3 = pool3.apply_async(pickle_loader, ('word_coordonates_fb.data', '2'))
        async_result4 = pool4.apply_async(pickle_loader, ('word_norms_fb.data', '3'))
    elif corpus == 'custom1':
        async_result2 = pool2.apply_async(pickle_loader, ('index_table_custom.data', '1'))
        async_result3 = pool3.apply_async(pickle_loader, ('word_coordonates_custom.data', '2'))
        async_result4 = pool4.apply_async(pickle_loader, ('word_norms_custom.data', '3'))
    elif corpus == 'custom2':
        async_result2 = pool2.apply_async(pickle_loader, ('index_table_custom2.data', '1'))
        async_result3 = pool3.apply_async(pickle_loader, ('word_coordonates_custom2.data', '2'))
        async_result4 = pool4.apply_async(pickle_loader, ('word_norms_custom2.data', '3'))
    else:
        raise ValueError("Corpus not recognized")
    lemme_table = async_result1.get()
    index_table = async_result2.get()
    word_coordonates = async_result3.get()
    word_norms = async_result4.get()
    return lemme_table, index_table, word_coordonates, word_norms


def modelisation(interactive, verbose, fromfile, textfile, neighbors_cache, lemme_table, index_table, word_coordonates, word_norms):
    """
    Fonction principale modélisant la(les) mémoire(s) lors d'une lecture de texte
    """

    ## PARAMETRES DE SIMULATION
    epsilon = 0.001 #pour la phase d'intégration
    nb_semantic_neighbors = 30000 #combien de voisins sémantique on recherche en mémoire
    activation_value_min = 0.18  #0.45 #Valeur d'activation minimum à la fin de l'intégration pour laquelle son gardé les mots
    activation_value_min_episodic = 0.39 #0.35 #Valeur d'activation mini pour faire EM->WM
    correlation_value_min_episodic = 0.39 #0.30 #Valeur de correlation mini entre le mot courant et ceux de l'EM pour faire EM->WM
    decay_rate = 0.95 #activation value de l'EM sont multipliées par ça à chaque cycle pour simulé un 'oubli'
    max_size_wm = 14 #nombre maximum de mot dans la wm
    sentence = "Dans l'océan il y a beaucoup de poissons. La mer est un endroit accueillant pour la vie."

    if fromfile: #Utilisation d'un .txt en ligne de commande
        txt = open(textfile, "r")
        sentence =  txt.read()
        txt.close()

    ## INTERFACE D'UTILISATION DU PROGRAMME
    while(True):
        if interactive:
            print("\nWhat do you want to do?:\n")
            print("D : Default example")
            print("C : Custom text")
            print("P : Modify parameters")
            print("I : Informations")
            print("X : Exit")
            cmd = input()
            while (cmd not in ["D", "X", "C", "I", "P"]):
                print("Option not recognized, please choose between 'D', 'C', 'P', 'I' or 'X'.")
                cmd = input()

            if cmd == "C":
                sentence = input("Enter your own text now :\n")

            elif cmd == "I":
                print("This algorithm is a simulation of the human memory when one is reading a text. It is based on the construction-integration idea of Kintsch.")
                print("The default parameter are:")
                print("\nepislon = 0.001     (represent the stop condition of the integration phase)")
                print("\nnb_semantic_neighbors = 4      (represent the number of neighbors selected in the semantic memory during construction phase)")
                print("\nactivation_value_min = 0.6      (represent the minimum value required to be selected at the end of the integration phase)")
                print("\nmax_size_wm = 12      (represent the maximum number of words in working memory at the same time)")
                continue

            elif cmd == "P":
                epsilon = float(input("Enter epsilon : "))
                nb_semantic_neighbors = int(input("Enter the number of semantic neighbors : "))
                activation_value_min = float(input("Enter the minimal activation value : "))
                continue

            elif cmd == 'X':
                exit(0)


        ## COMPUTATION
        if verbose:
            print("Texte avant lemmatisation: " + sentence)
        sentence = lemmatisation2(sentence, lemme_table)
        if verbose:
            print("Text lemmatisé: ")
            print(sentence)
        working_memory = []
        episodic_memory = []
        for word in sentence:
            word_id = get_word_id(index_table, word)
            if word_id != -1: #word in the vocabulary
                if verbose:
                    print("\n/////////////////      Traitement du mot : " + word +"          ///////////////////////\n")
                closest_neighbors = get_closest_neighbors(nb_semantic_neighbors, word_id, word_coordonates, working_memory, neighbors_cache, word_norms)
                if verbose:
                    print("Neighbors of " + word + " : ")
                    display_neighbors(index_table, closest_neighbors)
                #Récupération potentielle de mots via espisodic memory +  decay après utilisation
                episodic_neighbors = get_episodic_neighbors(episodic_memory, word_coordonates, word_id, activation_value_min_episodic, correlation_value_min_episodic, word_norms)
                episodic_memory = decay(episodic_memory, decay_rate)
                if verbose:
                    display_episodic(index_table, episodic_memory)
                #intégration
                working_memory = integration(closest_neighbors, working_memory, episodic_memory, episodic_neighbors, word_coordonates, epsilon, activation_value_min, max_size_wm, word_norms)
                if verbose:
                    display(index_table, working_memory)
            elif verbose: #skip unknown words
                print("The word : " + word + " is detected but is not in the vocabulary")


        if not interactive:# on sort du programme
            return working_memory, sentence

def evaluate_modelisation(wm, theme_vectors, word_coordonates, textfile, raw_themes):
    """
    Compute cosinus between sum of wordvectors of themes and sum (weighted) of wordvectors of working memory in final state.
    """
    #sum weighted
    for index, word in enumerate(wm):
        if index == 0:
            wm_vector = word_coordonates[word[0]] * word[1]
        else:
            wm_vector += word_coordonates[word[0]] * word[1]


    #cosinus
    scores = np.array([])
    for i in range(len(theme_vectors)):
        score = np.dot(wm_vector, theme_vectors[i])/(np.linalg.norm(wm_vector)*np.linalg.norm(theme_vectors[i]))
        scores = np.append(scores, score)
    # print(scores)
    print(textfile)
    prediction = raw_themes[np.argmax(scores)].replace("é", "e").replace(" ", "_").replace("è", "e").replace("ê", 'e')
    print(prediction)
    print(str(prediction in textfile))
    return prediction in textfile

def evaluate_naive_methode(all_word, theme_vectors, word_coordonates, textfile, raw_themes, index_table):
    """
    Compute cosinus between the sum of wordvectors of themes and the sum of all the wordvectors present in the current text.
    """
    fake_wm = [(get_word_id(index_table, word), 1) for word in all_word if get_word_id(index_table, word) != -1]
    return evaluate_modelisation(fake_wm, theme_vectors, word_coordonates, textfile, raw_themes)

def create_theme_vectors(lemme_table, index_table, word_coordonates):
    """
    Return a list of vectors corresponding to each theme in the semantic space
    """
    txt = open("themes.txt", "r")
    themes =  txt.read()
    themes = themes.split("\n")[:-1]
    raw_themes = themes.copy()
    vectors = []
    for index, theme in enumerate(themes):
        themes[index] = lemmatisation2(theme, lemme_table)
        for index2, word in enumerate(themes[index]):
            if index2 == 0:
                vector = word_coordonates[get_word_id(index_table, word)]
            else:
                vector = np.add(vector, word_coordonates[get_word_id(index_table, word)])
        vectors.append(vector)

    txt.close()
    return vectors, raw_themes


def main():
    """
    Gère les arguments en ligne de commande, charge les donnéees, puis appelle la fonction de modélisation principale
    """
    verbose = False
    interactive = False
    fromfile = False
    textfile = None
    if len(sys.argv) == 1 or ('-t' in sys.argv and '-i' in sys.argv) or '-h' in sys.argv:
        print("utilisation :", sys.argv[0], "[-v] [-i] [-t] file1.txt file2.txt etc.. \n -v  for verbose \n -i  for interactive mode\n -t file.txt to use a specifique .txt. \n\n CARE -i and -t are incompatible")
        sys.exit(1)

    ## DATA EXTRACTION
    lemme_table, index_table, word_coordonates, word_norms = global_data_loader("lemonde")
    theme_vectors, raw_themes = create_theme_vectors(lemme_table, index_table, word_coordonates)
    neighbors_cache = dict()
    if '-v' in sys.argv:
        verbose = True
    if '-i' in sys.argv:
        interactive = True
    if '-t' in sys.argv:
        fromfile = True
        indexfile = sys.argv.index('-t') + 1
        alltext = sys.argv[indexfile:]
        res = []
        # max_acc = 0
        # for seuil_add in np.arange(0.05 , 0.4, 0.01):
        #     counter = 0
        #     # counter2 = 0
        #     for textfile in alltext:
        #         wm, sentence = modelisation(interactive, verbose, fromfile, textfile, neighbors_cache, lemme_table, index_table, word_coordonates, word_norms, seuil_add)
        #         # display(index_table, wm)
        #         if evaluate_modelisation(wm, theme_vectors, word_coordonates, textfile, raw_themes):
        #             counter +=1
        #         #     print("NAIVE")
        #         # if evaluate_naive_methode(sentence, theme_vectors, word_coordonates, textfile, raw_themes, index_table):
        #         #     counter2 +=1
        #     accuracy = counter/len(alltext)
        #     if accuracy >= max_acc:
        #         max_acc = accuracy
        #         print(accuracy)
        #     #print("accuracy = " + str(accuracy) + " on " + str(len(alltext)) + " text(s)")
        #     res.append(accuracy)
        counter = 0
        counter2 = 0
        for textfile in alltext:
            wm, sentence = modelisation(interactive, verbose, fromfile, textfile, neighbors_cache, lemme_table, index_table, word_coordonates, word_norms)
            # display(index_table, wm)
            if evaluate_modelisation(wm, theme_vectors, word_coordonates, textfile, raw_themes):
                counter +=1
                print("NAIVE")
            if evaluate_naive_methode(sentence, theme_vectors, word_coordonates, textfile, raw_themes, index_table):
                counter2 +=1
        accuracy = counter/len(alltext)
        print("accuracy = " + str(accuracy) + " on " + str(len(alltext)) + " text(s)")
        res.append(accuracy)
        print("accuracy naive = " + str(counter2/len(alltext)) + " on " + str(len(alltext)) + " text(s)")
        print(res)

    else:
        modelisation(interactive, verbose, fromfile, textfile, neighbors_cache, lemme_table, index_table, word_coordonates, word_norms)
if __name__ == '__main__':
    main()
