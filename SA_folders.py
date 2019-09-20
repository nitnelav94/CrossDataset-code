# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 21:18:56 2019

@author: mmval
"""

import codecs

def read_file(email_file1,corpus):
    with open(email_file1,'r', encoding='utf-8') as reader1:
        for line in reader1:
            corpus.append(line)


def read_labels(label_file1,labels):
    with open(label_file1,'r',encoding='utf-8') as label_reader1:
        for line in label_reader1:
            labels.append(line)
        
            
folders=['easy_ham','easy_ham_2','hard_ham','spam','spam_2']            
main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/SpamAssassin/'

archiv_text=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/SpamAssassin/full_data.txt','w','utf-8')
archiv_lab=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/SpamAssassin/full_labels.txt','w','utf-8')

text=[]
labels=[]
for f in folders:
    file=main_dir+f+'/files_SA_'+f+'/data_SA_'+f+'.txt'
    read_file(file,text)
    
    lab=main_dir+f+'/files_SA_'+f+'/labels_SA_'+f+'.txt'
    read_labels(lab,labels)
    
for line in text:
    archiv_text.write(line)
    
for lb in labels:
    archiv_lab.write(lb)


    
    