# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:19:57 2019

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

def read_file(email_file1,email_file2,corpus):
    with open(email_file1,'r', encoding='utf-8') as reader1, open(email_file2,'r', encoding='utf-8') as reader2:
        StopWords = stopwords.words('english')
        i=0
        for line in reader1:
            tokens=line.rstrip().split()
            #for words
            text=' '.join([token for token in tokens if token not in StopWords and len(token)>3 and len(token)<35])
            #
            #text=' '.join([token for token in tokens])
            corpus.append(text)
            i=i+1
        #print(email_file1,i)
        i=0
        for line in reader2:
            tokens=line.rstrip().split()
            #for words
            text=' '.join([token for token in tokens if token not in StopWords and len(token)>3 and len(token)<35])
            #text=' '.join([token for token in tokens])
            corpus.append(text)
            i=i+1
        #print(email_file1,i)
    
    
def read_labels(label_file1, label_file2, labels):
    with open(label_file1,'r',encoding='utf-8') as label_reader1, open(label_file2,'r',encoding='utf-8') as label_reader2:
        i=0
        for line in label_reader1:
            labels.append(line.strip())
            i=i+1
        #print(label_file1,i)
        i=0
        for line in label_reader2:
            labels.append(line.strip())
            i=i+1
        #print(label_file2,i)
        
start = time.time()
c='words'
main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/GenSpam/'
email_adapt1=main_dir+'adapt_GEN/GenSpam_adapt_GEN_'+c+'.txt'
email_adapt2=main_dir+'adapt_SPAM/GenSpam_adapt_SPAM_'+c+'.txt'
label_adapt1=main_dir+'adapt_GEN/labels_GenSpam_adapt_GEN.txt'
label_adapt2=main_dir+'adapt_SPAM/labels_GenSpam_adapt_SPAM.txt'

email_train1=main_dir+'train_GEN/GenSpam_train_GEN_'+c+'.txt'
email_train2=main_dir+'train_SPAM/GenSpam_train_SPAM_'+c+'.txt'
label_train1=main_dir+'train_GEN/labels_GenSpam_train_GEN.txt'
label_train2=main_dir+'train_SPAM/labels_GenSpam_train_SPAM.txt'

email_test1=main_dir+'test_GEN/GenSpam_test_GEN_'+c+'.txt'
email_test2=main_dir+'test_SPAM/GenSpam_test_SPAM_'+c+'.txt'
label_test1=main_dir+'test_GEN/labels_GenSpam_test_GEN.txt'
label_test2=main_dir+'test_SPAM/labels_GenSpam_test_SPAM.txt'

######ADAPT FILES
adapt_corpus=[]
adapt_labels=[]

read_file(email_adapt1,email_adapt2,adapt_corpus)
read_labels(label_adapt1,label_adapt2,adapt_labels)
adapt_labels=np.array(adapt_labels)


#####TRAIN FILES
train_corpus=[]
train_labels=[]

read_file(email_train1,email_train2,train_corpus)
read_labels(label_train1,label_train2,train_labels)
train_labels=np.array(train_labels)


#####TEST FILES
test_corpus=[]
test_labels=[]



read_file(email_test1,email_test2,test_corpus)
read_labels(label_test1,label_test2,test_labels)
test_labels=np.array(test_labels)



#vectorize train set

vec = TfidfVectorizer(min_df=1, norm='l2', analyzer = 'word', tokenizer=my_tokenizer)
train_tfidf = vec.fit_transform(train_corpus)

adapt_tfidf = vec.transform(adapt_corpus)
models=['svm','lr','nb','knn','rf']
for model in models:
    print(model)
    if model!='nb':
        if model=='lr' or model=='svm':
            cs = [0.1, 1.0, 10.0, 100.0] #Logistic regression, SVM
        if model=='rf':
            cs = [5,10,15,20] #Random forrest
        if model=='knn':
            cs = [1, 2, 3, 5, 10] #KNN
        best_score = 0
        

        for c in cs:
            #clf= LogisticRegression(C=c, penalty='l2', solver='liblinear')
            #clf = svm.LinearSVC(C=c)
            #clf = MultinomialNB()
            #clf = RandomForestClassifier(n_estimators=c, n_jobs=-1)
            #clf = KNeighborsClassifier(n_neighbors=c, algorithm = 'brute', metric='cosine')
            if model=='lr':
                clf= LogisticRegression(C=c, penalty='l2', solver='liblinear')
            if model=='svm':
                clf = svm.LinearSVC(C=c)

            if model=='rf':
                clf= RandomForestClassifier(n_estimators=c, n_jobs=-1)
            if model=='knn':
                clf= KNeighborsClassifier(n_neighbors=c, algorithm = 'brute', metric='cosine')
            
            clf.fit(train_tfidf, train_labels)
            predicted = clf.predict(adapt_tfidf)
            score = metrics.f1_score(adapt_labels, predicted, average='macro')
            if score > best_score:
                best_score = score
                best_c = c

    train_corpus=list(train_corpus)
    train_labels=list(train_labels)
    train_corpus.extend(adapt_corpus)
    train_labels.extend(adapt_labels)
    
    vec = TfidfVectorizer(min_df=1, norm='l2', analyzer = 'word', tokenizer=my_tokenizer)
    train_tfidf = vec.fit_transform(train_corpus)
    test_tfidf = vec.transform(test_corpus)
    
    #clf = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
    #clf = svm.LinearSVC(C=best_c)
    #clf = MultinomialNB()
    #clf = RandomForestClassifier(n_estimators=best_c, n_jobs=-1)
    #clf = KNeighborsClassifier(n_neighbors=best_c, algorithm = 'brute', metric='cosine')
    if model=='lr':
        clf = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
    if model=='svm':
        clf = svm.LinearSVC(C=best_c)
    if model=='nb':
        clf = MultinomialNB()
    if model=='rf':
        clf = RandomForestClassifier(n_estimators=best_c, n_jobs=-1)
    if model=='knn':
        clf = KNeighborsClassifier(n_neighbors=best_c, algorithm = 'brute', metric='cosine')
    clf.fit(train_tfidf, train_labels)
    predicted = clf.predict(test_tfidf)
    accuracy = metrics.accuracy_score(test_labels, predicted)
    precision = metrics.precision_score(test_labels, predicted, average='macro')
    recall = metrics.recall_score(test_labels, predicted, average='macro')
    f1_macro = metrics.f1_score(test_labels, predicted, average='macro')
    kappa = metrics.cohen_kappa_score(test_labels, predicted)
    roc = metrics.roc_auc_score(test_labels.astype(int), predicted.astype(int))
    stop = time.time()
    
    #print('Training + test time = '+str((stop - start)/60))
    #print('Accuracy: %0.2f' % accuracy)
   # print('Precision: %0.2f'% precision)
  #  print('Recall: %0.2f' % recall)
 #   print('F1: %0.2f' % f1_macro)
#    print('Kappa: %0.2f'% kappa)
    print('ROC: %f', roc)
