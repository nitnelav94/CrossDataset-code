# -*- coding: utf-8 -*-
"""
Created on Sunday Mar  3 18:39:09 2019

@author: mmval
"""

import codecs

main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/trec07p/data/files_trec07p_data/'
labels=main_dir+'labels_trec07p.txt'
emails=main_dir+'data_trec07p.txt'

archiv_text=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/trec07p/ham_data.txt','w','utf-8')
archiv_lab=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/trec07p/ham_labels.txt','w','utf-8')



l=[]
with open(labels,'r',encoding='utf-8') as reader1, open(emails,'r',encoding='utf-8') as reader2:
    i=1
    for line in reader1:
        line=line.strip()
        if line=='1':
            l.append(i)
        i=i+1
    w=1
    for line in reader2:
        if w in l:
            archiv_text.write(line)
            archiv_lab.write('1\n')
        w=w+1
archiv_text.close()
archiv_lab.close()
            
        
