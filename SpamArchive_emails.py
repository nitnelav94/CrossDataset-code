# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:54:24 2019

@author: mmval
"""

import codecs

def read_file(email_file1,corpus):
    with open(email_file1,'r', encoding='utf-8') as reader1:
        for line in reader1:
            corpus.append(line.strip())
        

def read_labels(label_file1, labels):
    with open(label_file1,'r',encoding='utf-8') as label_reader1:
        for line in label_reader1:
            labels.append(line.strip())

text=[]
labels=[]
main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/SpamArchive/'
a1='2015'
a2='2018'

file=main_dir+a2+'/'+'01/'+'files_SA_'+a2+'_01/SpamArchive_'+a2+'_01_words.txt'
file_lab=main_dir+a2+'/'+'01/'+'files_SA_'+a2+'_01/labels_SpamArchive_'+a2+'_01.txt'
read_file(file,text)
read_labels(file_lab,labels)


for i in range(12,0,-1):
    if i>0:
        if i<10:
            file=main_dir+a1+'/'+'0'+str(i)+'/files_SA_'+a1+'_0'+str(i)+'/SpamArchive_'+a1+'_0'+str(i)+'_words.txt'
            file_lab=main_dir+a1+'/'+'0'+str(i)+'/files_SA_'+a1+'_0'+str(i)+'/labels_SpamArchive_'+a1+'_0'+str(i)+'.txt'
        if i>=10:
            file=main_dir+a1+'/'+str(i)+'/files_SA_'+a1+'_'+str(i)+'/SpamArchive_'+a1+'_'+str(i)+'_words.txt'
            file_lab=main_dir+a1+'/'+str(i)+'/files_SA_'+a1+'_'+str(i)+'/labels_SpamArchive_'+a1+'_'+str(i)+'.txt'
        read_file(file,text)
        read_labels(file_lab,labels)
        print('2015',i)
        
     
file=main_dir+'enron_ham_data.txt'
file_lab=main_dir+'enron_ham_labels.txt'  
read_file(file,text)
read_labels(file_lab,labels)


file=main_dir+'trec_ham_data.txt'
file_lab=main_dir+'trec_ham_labels.txt'  
read_file(file,text)
read_labels(file_lab,labels)      
        
archiv_text=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/SpamArchive/full_data.txt','w','utf-8')
archiv_lab=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/SpamArchive/full_labels.txt','w','utf-8')

w=0
for line in text:
    w=w+1
    print(w)
    archiv_text.write(line+'\n')
    
for line in labels:
    archiv_lab.write(line+'\n')
    
#archiv_text.close()
#archiv_lab.close()

