# *- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:52:23 2019

@author: mmval
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import normalize
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


def my_tokenizer(s):
    return s.split()

def read_data(files):
    corpus=[]
    for file in files:
        with open(file,'r',encoding='utf-8') as reader:
            #StopWords = stopwords.words('english')
            for line in reader:
                tokens=line.rstrip().split()
                #for words
                #text=' '.join([token for token in tokens if token not in StopWords and len(token)>3 and len(token)<35])
                text=' '.join([token for token in tokens])
                corpus.append(text)
    return corpus
            


def read_labels(files):
    labels=[]
    for file in files:
        with open(file,'r',encoding='utf-8') as label_reader:
            for line in label_reader:
                labels.append(line.strip())
        #labels=np.array(labels)
    return labels
          
#Para
    #PU1 (encoded)
    #PU2 (encoded)
    #PU3 (encoded)
    #PUA (encoded)
    #LingSpam(BARE)
      
start = time.time()

#Para PU (they're encoded)
"""
c='pua'
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
    docs.append(main_dir+'/part'+str(i+1)+'/files_LS_bare_part'+str(i+1)+'/LS_bare_part'+str(i+1)+'_links.txt')
    lab.append(main_dir+'/part'+str(i+1)+'/files_LS_bare_part'+str(i+1)+'/labels_LS_bare_part'+str(i+1)+'.txt')


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
    
    corpus_train=read_data(data_train)
    corpus_test=read_data(data_test)
    labels_train=read_labels(lab_train)
    labels_test=read_labels(lab_test)
    labels_test=np.array(labels_test)
    labels_train=np.array(labels_train)
    
    #min_dif=1 ignora palabras que aparezcan menos de una vez
    #norm=l1 norma utilizada para normalizar los vectores
    #analyzer=word indica que las caracteristicas estaran echas de palabras
    #tokenizer= my_tokenizer separa por palabras 
    vec = TfidfVectorizer(min_df=1, norm='l2', analyzer = 'word', tokenizer=my_tokenizer)
    
    train_tfidf = vec.fit_transform(corpus_train)#
    
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
        scores_inner = cross_val_score(clf_inner, train_tfidf, labels_train, scoring='f1_macro', cv=sub_skf)
        
        score = np.mean(scores_inner)
        
        if score > best_score:
            best_score = score
            best_c = c
     
    #clf = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
    #clf = svm.LinearSVC(C=best_c)
    #clf = RandomForestClassifier(n_estimators=best_c, n_jobs=-1)
    clf = KNeighborsClassifier(n_neighbors=best_c, algorithm = 'brute', metric='cosine')
    #clf = MultinomialNB()
    clf.fit(train_tfidf, labels_train)
    test_tfidf = vec.transform(corpus_test)#
    predicted = clf.predict(test_tfidf)
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

    
stop = time.time()
print('Training + test time = '+str((stop - start)/60))
print('Accuracy: %0.2f (+/- %0.2f)' % (np.mean(accuracies), np.std(accuracies) * 2))
print('Precision: %0.2f (+/- %0.2f)' % (np.mean(precisions), np.std(precisions) * 2))
print('Recall: %0.2f (+/- %0.2f)' % (np.mean(recalls), np.std(recalls) * 2))
print('F1: %0.2f (+/- %0.2f)' % (np.mean(f1s), np.std(f1s) * 2))
print('Kappa: %0.2f (+/- %0.2f)' % (np.mean(kappas), np.std(kappas) * 2))
print('AUC: %0.2f (+/- %0.2f)' % (np.mean(scores_roc), np.std(scores_roc) * 2))
#"""  
