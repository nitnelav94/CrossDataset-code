# -*- coding: utf-8 -*-
"""
Created on Tue May  7 20:04:34 2019

@author: mmval
"""

import numpy as np
import codecs


d='trec07p'
main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/trec07p/data/files_trec07p_data/'
data_file=main_dir+d+'_words.txt'
labels_file=main_dir+'labels_'+d+'.txt'


train_file=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/trec07p/data/files_trec07p_data/trec07p_words_train.txt','w','utf-8')
val_file=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/trec07p/data/files_trec07p_data/trec07p_words_val.txt','w','utf-8')

with open(data_file,'r',encoding='utf-8') as reader:
    i=1
    for line in reader:
        if i<=15057:
            val_file.write(line.rstrip()+'\n')
        else:
            train_file.write(line.rstrip()+'\n')
        i=i+1


labels_train_file=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/trec07p/data/files_trec07p_data/trec07p_labels_words_train.txt','w','utf-8')
labels_val_file=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/trec07p/data/files_trec07p_data/trec07p_labels_words_val.txt','w','utf-8')

with open(labels_file,'r',encoding='utf-8') as reader:
    i=1
    for line in reader:
        if i<=15057:
            labels_val_file.write(line.rstrip()+'\n')
        else:
            labels_train_file.write(line.rstrip()+'\n')
        i=i+1

