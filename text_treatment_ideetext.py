#!/usr/bin/env python3

import codecs
import os

for text in os.listdir('./ideetextdata'):
    print("yo")
    with codecs.open('./ideetextdata/' + text, "r",encoding='utf-8', errors='ignore') as txt:
        sentence =  txt.read()
        sentence = sentence.replace("mailto:?subject=", "XXX").split("XXX")
        if len(sentence) >= 2:
            res = sentence[1].split('>')[0].split('&body=')
            try:
                with open('./ideetextdatapropre/' + res[0], "w", encoding='utf-8', errors='ignore') as newtxt:
                    newtxt.write(res[1])
            except FileNotFoundError:
                continue
