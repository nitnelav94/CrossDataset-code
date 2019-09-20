# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:18:22 2019

@author: mmval
"""
import codecs

def read_file(email_file1,email_file2,corpus):
    with open(email_file1,'r', encoding='utf-8') as reader1, open(email_file2,'r', encoding='utf-8') as reader2:
        for line in reader1:
            corpus.append(line)
        for line in reader2:
            corpus.append(line)


def read_labels(label_file1, label_file2,labels):
    with open(label_file1,'r',encoding='utf-8') as label_reader1, open(label_file2,'r',encoding='utf-8') as label_reader2:
        for line in label_reader1:
            labels.append(line)
        for line in label_reader2:
            labels.append(line)


ruta='Enron'
s='_spam'
h='_ham'

main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/'+ruta
archiv_text=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/Enron/full_data.txt','w','utf-8')
archiv_lab=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/Enron/full_lab.txt','w','utf-8')


"""
text=''
with open(enron_spam,'r',encoding='utf-8') as r1, open(enron_ham,'r',encoding='utf-8') as r2:
    for line in r1:
        text+=line
    for line in r2:
        text+=line
        
archiv_text.write(line)

i=1
"""
text=[]
labels=[]
for i in range(7):
    if i!=0:
        newFolder='files_enron'+str(i)
        folder='/enron'+str(i)+'/'
        enron_spam=main_dir+folder+newFolder+s+'/new_data_enron'+str(i)+s+'.txt'
        enron_ham=main_dir+folder+newFolder+h+'/new_data_enron'+str(i)+h+'.txt'
        read_file(enron_spam,enron_ham,text)
    
   
        
        label_spam=main_dir+folder+newFolder+s+'/labels_enron'+str(i)+s+'.txt'
        label_ham=main_dir+folder+newFolder+h+'/labels_enron'+str(i)+h+'.txt'
        read_labels(label_spam,label_ham,labels)


for line in text:
    archiv_text.write(line)
    
for line in labels:
    archiv_lab.write(line)
    
    


