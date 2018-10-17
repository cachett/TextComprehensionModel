#!/usr/bin/env python3


import sys
import word2vec
import numpy as np

def get_word_coord(word, model):
    """Retourne l'ID d'un mot donné, Erreur si le mot n'est pas connu"""
    try:
        word_coord = np.array(model[word])
        return word_coord
    except KeyError:
        print(word)
        return np.zeros(200)

def get_theme_vector(textfile, model):
    textfile = textfile.split("/")[-1]
    textfile = textfile.split("-")[0]
    textfile = textfile.split("_")

    for index, word in enumerate(textfile):
        if index == 0:
            vector = get_word_coord(word, model)
        else:
            vector += get_word_coord(word, model)

    return vector


model = word2vec.load("frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin")
res_global = np.zeros(80)

alltext = sys.argv[1:]
for textfile in alltext:
    print(textfile)
    txt = open(textfile, "r")
    txt = txt.readline()
    txt = txt.lower().replace(';', ' ').replace("'", ' ').replace(",", ' ') \
    .replace(".", ' ').replace("(", ' ').replace(")", ' ').replace(":", ' ') \
    .replace("(", ' ').replace('"', ' ').split(' ')
    res_tmp = np.zeros(80)
    theme_vector = get_theme_vector(textfile, model)
    theme_norm = np.linalg.norm(theme_vector)
    max_index = 0
    for index, word in enumerate(txt):
        if index == 0:
            vector = get_word_coord(word, model)
        else:
            vector += get_word_coord(word, model)


        cos_value = np.dot(vector, theme_vector)/ (np.linalg.norm(vector)* theme_norm)
        res_tmp[index] = cos_value
        max_index = index
    res_tmp[max_index:] = res_tmp[max_index]
    print(res_global)
    res_global = res_global + res_tmp
    print(res_global)


res_global = res_global / len(alltext)
print(res_global.tolist())
