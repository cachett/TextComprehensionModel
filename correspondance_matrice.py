#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pickle
from threading import Thread


"""
This code compute (with threading) the matrix of correspondance of each word in the vocabulary and save it as pickle objet. 
This method has been abandonned since the pickle object is more than 4Gb so I decided to compute correspondance between words on the fly.
"""

data_word_vector = pd.read_csv("export_Le_Monde_LSA.csv")
data_word_vector.drop(0, inplace=True) # première ligne useless
# print(data_word_vector.head())

data_iterator = data_word_vector.iterrows()
word_coordonates = []


row = next(data_iterator)
end = False
while(not end):
    row = row[1][0]
    row = np.array(row.split(" ")[1:]).astype(float)
    word_coordonates.append(row)
    try:
        row = next(data_iterator)
    except StopIteration:
        end = True

# len(word_coordonates) = 35382
correlation_matrix = np.zeros((35382, 35382))
print(len(word_coordonates))

class Calcul(Thread):
    def __init__(self, thread_id, indice_debut, indice_fin):
        Thread.__init__(self)
        self.indice_debut = indice_debut
        self.indice_fin = indice_fin
        self.thread_id = thread_id

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        for index1 in range(self.indice_debut, self.indice_fin):
            # print("Thread : " + self.thread_id + "indice : " + str(index1))
            for index2 in range(index1, self.indice_fin): # matrice symétrique, inutile de calculer la 2e moitié
                correlation_matrix[index1][index2] = np.dot(word_coordonates[index1], word_coordonates[index2])/ \
                (np.linalg.norm(word_coordonates[index1])*np.linalg.norm(word_coordonates[index2]))


# Création des threads
thread_1 = Calcul("0", 0, 8845)
thread_2 = Calcul("1", 8845, 17691)
thread_3 = Calcul("2", 17691, 26536)
thread_4 = Calcul("3", 26536, 35382)

# Lancement des threads
thread_1.start()
thread_2.start()
thread_3.start()
thread_4.start()

# Attend que les threads se terminent
thread_1.join()
thread_2.join()
thread_3.join()
thread_4.join()


# for index1, vecteur1 in enumerate(word_coordonates):
#     for index2, vecteur2 in enumerate(word_coordonates):
#         correlation_matrix[index1][index2] = np.dot(vecteur1, vecteur2)/(np.linalg.norm(vecteur1)*np.linalg.norm(vecteur2))

print(correlation_matrix)
pickle.dump(correlation_matrix[:17691], open("correlation_matrix1.data", "wb"))
pickle.dump(correlation_matrix[17691:], open("correlation_matrix2.data", "wb"))
