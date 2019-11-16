# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:03:33 2019

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



def my_tokenizer(s):
    return s.split()

def read_data(files, lfiles,model):
    corpus=[]
    labels=[]
    z=[]
    for file in range(len(files)):
        with open(files[file],'r',encoding='utf-8') as reader, open(lfiles[file],'r',encoding='utf-8') as label_reader:
             StopWords = stopwords.words('english')
             for i,line in enumerate(reader):
                 tokens=line.rstrip().split()
                 le=0
                 l=np.zeros(300)
                 for token in tokens:
                     if token not in StopWords and len(token)>3 and len(token)<35 and token in model:
                         le=le+1
                         l=l+np.array(model[token])
                 #print(le)
                 if le>0:
                     l=l/le
                     corpus.append(l)
                 else:
                     z.append(i)
                     
             for j,line in enumerate(label_reader):
                if j not in z:
                    labels.append(line.strip())
             

             
    return corpus,labels

def minmax(corpus):
    for i in range(len(corpus)):
        corpus[i]=(corpus[i]-np.amin(corpus[i]))/(np.amax(corpus[i])-np.amin(corpus[i]))
    return corpus

start = time.time()

#For PU
"""
c='pu3'
main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/pu_corpora_public/'+c
docs=[]
lab=[]
for i in range(10):
    docs.append(main_dir+'/part'+str(i+1)+'/data_'+c+'_part'+str(i+1)+'_words.txt')
    lab.append(main_dir+'/part'+str(i+1)+'/labels_'+c+'_part'+str(i+1)+'.txt')
"""

#Para LingSpam

main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/lingspam_public/bare'#+c
docs=[]
lab=[]
for i in range(10):
    docs.append(main_dir+'/part'+str(i+1)+'/files_LS_bare_part'+str(i+1)+'/LS_bare_part'+str(i+1)+'_words.txt')
    lab.append(main_dir+'/part'+str(i+1)+'/files_LS_bare_part'+str(i+1)+'/labels_LS_bare_part'+str(i+1)+'.txt')


model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/GoogleNews-vectors-negative300.bin', binary=True)


data_train=[]
data_test=[]
lab_train=[]
lab_test=[]
accuracies = []
precisions = []
recalls = []
f1s = []
kappas = []
scores_roc=[]
i = 0
for w in range(len(docs)):
    print('iteration: ',w)
    for i in range(len(docs)):
        if i==w:
            data_test.append(docs[i])
            lab_test.append(lab[i])
        else:
            data_train.append(docs[i])
            lab_train.append(lab[i])
    
    corpus_train, labels_train=read_data(data_train,lab_train,model)
    corpus_train=np.array(corpus_train)
    #next line just for NB only
    corpus_train=minmax(corpus_train)
    cropus_train=normalize(corpus_train,norm='l2')
    labels_train=np.array(labels_train)
    
    corpus_test,labels_test=read_data(data_test,lab_test,model)
    corpus_test=np.array(corpus_test)
    
    #next line just for NB only
    corpus_test=minmax(corpus_test)
    cropus_test=normalize(corpus_test,norm='l2')
    labels_test=np.array(labels_test)
    
    
    start1 = time.time()
    
    """
    #cs = [0.1, 1.0, 10.0, 100.0] #Logistic regression, SVM
    #cs = [5,10,15,20] #Random forrest
    cs = [1, 2, 3, 5, 10] #KNN
    best_c = 0
    best_score = 0
    
    for c in cs:
        #penalty: especifica la norma utilizada para la penalizacion 
        #solver: algoritmo usado para la optimizacion del problema
        #clf_inner = LogisticRegression(C=c, penalty='l2', solver='liblinear')
        
        #clf_inner = svm.LinearSVC(C=c)
        
        #clf_inner= RandomForestClassifier(n_estimators=c, n_jobs=-1)
        
        clf_inner = KNeighborsClassifier(n_neighbors=c, algorithm = 'brute', metric='cosine')
        
        #clf_inner: objeto utilizado para ajustar los datos
        #los datos a ajustar
        #labels_train: variable objetivo
        #
        #cv:cross-validation splitting strategy
        sub_skf = StratifiedKFold(n_splits=3, random_state=0)
        scores_inner = cross_val_score(clf_inner, corpus_train, labels_train, scoring='f1_macro', cv=sub_skf)
        
        score = np.mean(scores_inner)
        
        if score > best_score:
            best_score = score
            best_c = c
    """
    stop1 = time.time()
        
    start2 = time.time()    
    #clf = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
    #clf = svm.LinearSVC(C=best_c)
    #clf = RandomForestClassifier(n_estimators=best_c, n_jobs=-1)
    #clf = KNeighborsClassifier(n_neighbors=best_c, algorithm = 'brute', metric='cosine')
    clf = MultinomialNB()
    clf.fit(corpus_train, labels_train)
   
    predicted = clf.predict(corpus_test)
    accuracy = metrics.accuracy_score(labels_test, predicted)
    precision = metrics.precision_score(labels_test, predicted, average='macro')
    recall = metrics.recall_score(labels_test, predicted, average='macro')
    f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
    kappa = metrics.cohen_kappa_score(labels_test, predicted)
    roc = metrics.roc_auc_score(labels_test.astype(int), predicted.astype(int))
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1_macro)
    kappas.append(kappa)
    scores_roc.append(roc)
    
    data_train.clear()
    data_test.clear()
    lab_train.clear()
    lab_test.clear()

    
stop2 = time.time()
print('Training time = '+str((stop1 - start1)/60))
print('Testing time = '+str((stop2 - start2)/60))
print('Accuracy: %0.2f (+/- %0.2f)' % (np.mean(accuracies), np.std(accuracies) * 2))
print('Precision: %0.2f (+/- %0.2f)' % (np.mean(precisions), np.std(precisions) * 2))
print('Recall: %0.2f (+/- %0.2f)' % (np.mean(recalls), np.std(recalls) * 2))
print('F1: %0.2f (+/- %0.2f)' % (np.mean(f1s), np.std(f1s) * 2))
print('Kappa: %0.2f (+/- %0.2f)' % (np.mean(kappas), np.std(kappas) * 2))
print('AUC: %0.2f (+/- %0.2f)' % (np.mean(scores_roc), np.std(scores_roc) * 2))

