# -*- coding: utf-8 -*-
"""
Created on Sunday Mar  3 18:15:00 2019

@author: mmval
"""


import codecs

def read_file(email_file1,corpus):
    with open(email_file1,'r', encoding='utf-8') as reader1:
        for line in reader1:
            corpus.append(line)
        

def read_labels(label_file1, labels):
    with open(label_file1,'r',encoding='utf-8') as label_reader1:
        for line in label_reader1:
            labels.append(line)



ruta='Enron'
h='_ham'

main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/'+ruta
archiv_text=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/Enron/ham_data.txt','w','utf-8')
archiv_lab=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/Enron/ham_lab.txt','w','utf-8')



text=[]
labels=[]
for i in range(7):
    if i!=0:
        newFolder='files_enron'+str(i)
        folder='/enron'+str(i)+'/'
        enron_ham=main_dir+folder+newFolder+h+'/new_data_enron'+str(i)+h+'.txt'
        read_file(enron_ham,text)
    
   
        label_ham=main_dir+folder+newFolder+h+'/labels_enron'+str(i)+h+'.txt'
        read_labels(label_ham,labels)


for line in text:
    archiv_text.write(line)
    
for line in labels:
    archiv_lab.write(line)
    
    


