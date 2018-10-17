#!/usr/bin/env python3


import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import word2vec

"""
Créer des données d'entrainement pour notre réseau de neurones. L'entrée du réseau est la somme des wordvectors jusque là et la sortie représente le futur mouvement d'oeil: reculer, avancer ou bien rester sur place.
"""

def get_word_coord(word, model):
    """Retourne l'ID d'un mot donné, Erreur si le mot n'est pas connu"""
    try:

        word_coord = np.array(model[word])
        return word_coord
    except KeyError:
        print(word)
        return np.array([])

model = word2vec.load("frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin")
excel_file = 'em-y35_full.xlsx'
texts = pd.read_excel(excel_file)
# print(texts.head())
data_iterator = texts.iterrows()
row = next(data_iterator)
end = False
working_memory = np.array([])
input_data = open("train.in", "a")
output_data = open("train.out", "a")
counter = 0

while(not end):
    fixed_word = row[1][19]
    movement = row[1][17]
    counter += 1
    if type(fixed_word) is not float and movement != '_TOBEREPLACED_' and type(movement) is not float:
        movement = int(movement)
        fixed_word = fixed_word.split('~')[0].lower()
        if "'" in fixed_word:
            fixed_word = fixed_word.split("'")[1]
        fix_num = int(row[1][5])
        if movement > 0:
            #on avance
            movement = [1,0,0]
        elif movement < 0:
            #on recule
            movement = [0,0,1]
        else:
            #immobile
            movement = [0,1,0]

        word_coord = get_word_coord(fixed_word, model)

        if fix_num == 1:
            working_memory = np.array([])

        if len(word_coord) != 0:#mots dans le vocab
            if len(working_memory) == 0:
                working_memory = word_coord
            else:
                working_memory += word_coord

            input_data.write(str(working_memory.tolist()) + '\n')
            output_data.write(str(movement) + '\n')

    try:
        row = next(data_iterator)
    except StopIteration:
        end = True
