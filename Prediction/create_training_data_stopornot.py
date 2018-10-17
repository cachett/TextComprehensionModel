#!/usr/bin/env python3


import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import word2vec


"""
Créer des données d'entrainement pour notre réseau de neurones. L'entrée du réseau est la somme des wordvectors jusque là et la sortie représente la décision du lecteur: poursuite ou non de la lecture.
NB: la lecture des textes était pour un but précis de répondre à une question, on essaye de prédire si le lecteur a accumulé assez d'information ou non.
"""

def get_word_coord(word, model):
    """Retourne l'ID d'un mot donné, Erreur si le mot n'est pas connu"""
    try:

        word_coord = np.array(model[word])
        return word_coord
    except KeyError:
        print(word)
        return np.array([])

model = word2vec.load("frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin")
excel_file = 'em-y35_full.xlsx'
texts = pd.read_excel(excel_file)
# print(texts.head())
data_iterator = texts.iterrows()
row = next(data_iterator)
end = False
working_memory = np.array([])
input_data = open("train_500_stopornot.in", "a")
output_data = open("train_500_stopornot.out", "a")
word_num = 0
fix_num = -1

while(not end):
    fixed_word = row[1][19]
    word_inc = row[1][17]
    if type(fixed_word) is not float :

        fixed_word = fixed_word.split('~')[0].lower()
        if "'" in fixed_word:
            fixed_word = fixed_word.split("'")[1]

    if word_inc == '_TOBEREPLACED_' or type(word_inc) is float:
        word_inc = 1
    word_num += int(word_inc)

    try:
        row = next(data_iterator)
        fix_num = int(row[1][5])
        if fix_num == 1:
            #STOP
            decision = [1,0]
            working_memory = np.array([])
            word_num = 0
        else:
            #CONTINUE
            decision = [0,1]

            word_coord = get_word_coord(fixed_word, model)


        if len(word_coord) != 0:#mots dans le vocab
            if len(working_memory) == 0:
                working_memory = word_coord
            else:
                working_memory += word_coord
            input_data.write(str(working_memory.tolist()) + '\n')
            output_data.write(str(decision) + '\n')
    except StopIteration:
        end = True
