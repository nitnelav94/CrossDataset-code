# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:06:56 2019

@author: mmval
"""

from collections import Counter
from nltk.corpus import stopwords
import pandas as pd
import gensim

#features=['words','links','emoticons']

#PARA TREC
#main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/trec07p/data/files_trec07p_data/'
#main_dir='/media/jc/Disk1/data/Valentin/trec07p'


#PARA ENRON
#main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/Enron/'
#main_dir='/media/jc/Disk1/data/Valentin/Enron'


#PARA SPAMASSASSIN
#main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/SpamAssassin/'


#PARA LS bare
#main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/lingspam_public/bare/LS_Bare2/'

#PAra GenSpam
#main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/GenSpam/GenSpamFull/'



model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/GoogleNews-vectors-negative300.bin', binary=True)

main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/'
names=['SA','LS_Bare','GenSpam','trec07p','Enron']
emails=['SpamAssassin/SA','lingspam_public/bare/LS_Bare2/LS_Bare','GenSpam/GenSpamFull/GenSpam','trec07p/data/files_trec07p_data/trec07p','Enron/enron']
labels=['SpamAssassin/full_labels.txt','lingspam_public/bare/LS_Bare2/LS_Bare_Full_Labels.txt','GenSpam/GenSpamFull/GenSpam_labels_full.txt','trec07p/data/files_trec07p_data/labels_trec07p.txt','Enron/full_labels.txt']


for email,label,name in zip(emails,labels,names):
    feature='_words'


        
    email_file=main_dir+email+feature+'.txt'#'SpamAssassin/SA_words.txt'
    label_file=main_dir+label#'SpamAssassin/full_labels.txt'

    
    with open(email_file,'r', encoding='utf-8') as reader1:
            StopWords = stopwords.words('english')
            item=0
            #le=0
            
            items=[]
            for line in reader1:
                tokens=line.rstrip().split()
                #items=[]
                #item=0
                #le+=len(tokens)
                #print('longitud total: ',len(tokens))
                for token in tokens:
                    if token not in StopWords and len(token)>3 and len(token)<35 and token in model:
                                item+=1
                                items.append(token)
                    #if token not in items:
                        #if feature=='_words':
                            #if token not in StopWords and len(token)>3 and len(token)<35 and token in model:
                             #   item+=1
                              #  items.append(token)
                        #else:
                         #   item+=1
                          #  items.append(token)
                
    
    
    
    #data.insert(j,feature,(ite for ite in items))
    #data.to_csv(file)
    print(name)
    print(feature)
    print('total vocabulary')
    print('vocabulario: ',len(items))
    print('vocabulario2: ',item)
    #inf.append(items)
    #print(len(inf))
    #print(data)
    #j+=1
                
            #print('longitud total: ',le)
            
            
    #print(feature)
    #print(item)
    print('\n***********************')




