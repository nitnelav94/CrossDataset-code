# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 16:08:53 2019

@author: mmval
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:15:04 2019

@author: mmval
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
import numpy as np
import time
import codecs
import gensim 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def read_file(email_file,feature):#,model):
    with open(email_file,'r', encoding='utf-8') as reader:
        StopWords = stopwords.words('english')
        corpus=[]
        for line in reader:
            tokens=line.rstrip().split()
            for token in tokens:
                if feature=='_words':
                    if token not in StopWords and len(token)>3 and len(token)<35 :#and token in model:
                        corpus.append(token)
                else:
                    corpus.append(token)
          
        #print(corpus)
        return corpus
                    


#model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/GoogleNews-vectors-negative300.bin', binary=True)

main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/'

names=['trec07p','GenSpam','SA','Enron','LS_Bare']
emails1=['trec07p/data/files_trec07p_data/trec07p','GenSpam/GenSpamFull/GenSpam','SpamAssassin/SA','Enron/enron','lingspam_public/bare/LS_Bare2/LS_Bare']
emails2=['trec07p/data/files_trec07p_data/trec07p','GenSpam/GenSpamFull/GenSpam','SpamAssassin/SA','Enron/enron','lingspam_public/bare/LS_Bare2/LS_Bare']

Dj=[]
for email1 in emails1:
    r=[]
    for email2 in emails2:
        feature='_hashtags' 
        data1=set()
        data2=set()
        
        print(email1,email2)
        #print(len(email1),len(email2))
        email_file1=main_dir+email1+feature+'.txt'
        email_file2=main_dir+email2+feature+'.txt'
        
        #data1=read_file(email_file1,feature)
        #data1=set(data1)
        
        if email1!=email2:    
            data1=read_file(email_file1,feature)#,model)
            data2=read_file(email_file2,feature)#,model)
            
            data1=set(data1)
            data2=set(data2)
            
            print(len(data1),len(data2))
            i=data1.intersection(data2)
            j=data1.union(data2)
            if len(j)>0:
                dj=len(i)/len(j)
            else:
                dj=0
            r.append(dj)
        else:
            r.append(1)

    print(r)
    Dj.append(r)
print(Dj)
