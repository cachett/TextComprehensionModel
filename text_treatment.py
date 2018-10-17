#!/usr/bin/env python3


import os

"""
Met en forme les textes de manière plus claire:

input:

Le 34
chat 343
joue 3
à 43
la 654
balle. 364

output:
Le chat joue à la balle.

"""
for text in os.listdir('./Textes'):
    try:
        txt = open("./Textes/" + text, "r")
        sentence =  txt.read()
        sentence = sentence.split("\n")
        tmp = ''
        for mot in sentence:
            mot = mot.split(" ")
            if mot[0] != '':
                tmp += mot[0] + ' '
            else:
                new_text = open("./NewTextes/" + text, 'w')
                new_text.write(tmp)
                tmp=''
                new_text.close()
        txt.close()
    except UnicodeDecodeError:
        continue
