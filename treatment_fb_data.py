#!/usr/bin/env python3
import numpy as np
import pickle
import io
from tqdm import tqdm
import re


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    index_table = dict()
    word_coordonates = []
    word_norms = []
    regex = r"(([a-z]|é|è|à|ï|î|ù|ê|ë)+) (\d+)"
    index = 0
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        if len(tokens[0]) > 3 and all(c in "abcefghijklmnopqrstuvwxyzéàèïîêùë" for c in tokens[0]):
            tmp = np.array(tokens[1:]).astype(float)
            word_coordonates.append(tmp)
            index_table[tokens[0]] = index
            word_norms.append(np.linalg.norm(tmp))
            index += 1

        # print(word_norms[-1])
    pickle.dump(index_table, open("index_table_fb.data", "wb"))
    pickle.dump(word_coordonates, open("word_coordonates_fb.data", "wb"))
    pickle.dump(word_norms, open("word_norms_fb.data", "wb"))
    print(len(word_norms))



def test():
    word_vec = pickle.load(open("word_coordonates_fb.data", 'rb'))
    print(word_vec[0])

load_vectors("wiki.fr.vec")
# test()
