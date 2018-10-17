#!/usr/bin/env python3


import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile


excel_file = 'em-y35_treated2.xlsx'
texts = pd.read_excel(excel_file)
# print(texts.head())


data_iterator = texts.iterrows()
row = next(data_iterator)
end = False
file = None
id = 0
word_memory = ''


while(not end):
    filename = row[1][0]
    fix_num = row[1][1]
    fix_dur = row[1][2]
    fix_word = row[1][3].split('~')[0]
    if fix_num == 1:
        #new texts
        id += 1
        file = open("./NewTextesOcculo2/" + filename + str(id), 'a')
        file.write(fix_word + ' ')
    else:
        if fix_word != word_memory:#On enlève les doublons, il faudra additionner les temps
            file.write(fix_word + ' ')
    try:
        word_memory = fix_word
        row = next(data_iterator)
    except StopIteration:
        end = True
