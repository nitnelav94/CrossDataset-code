# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:05:00 2019

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

def read_file(email_file,label_file,model):
    with open(email_file,'r', encoding='utf-8') as reader, open(label_file,'r',encoding='utf-8') as label_reader:
        StopWords = stopwords.words('english')
        corpus=[]
        labels=[]
        z=[]
        for i,line in enumerate(reader):
            tokens=line.rstrip().split()
            le=0
            l=np.array(300)
            for token in tokens:
                if token not in StopWords and len(token)>3 and len(token)<35 and token in model:
                    le=le+1
                    l=l+np.array(model[token])
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

model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/GoogleNews-vectors-negative300.bin', binary=True)
        

c='words'

main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/'
email_val=main_dir+'lingspam_public/bare/LS_Bare2/LS_Bare_'+c+'_val.txt'
label_val=main_dir+'lingspam_public/bare/LS_Bare2/LS_Bare_labels_'+c+'_val.txt'

email_train=main_dir+'lingspam_public/bare/LS_Bare2/LS_Bare_'+c+'_train.txt'
label_train=main_dir+'lingspam_public/bare/LS_Bare2/LS_Bare_labels_'+c+'_train.txt'

email_test=main_dir+'GenSpam/GenSpamFull/GenSpam_'+c+'.txt'
label_test=main_dir+'GenSpam/GenSpamFull/GenSpam_labels_full.txt'


######ADAPT FILES
adapt_corpus=[]
adapt_labels=[]

adapt_corpus,adapt_labels=read_file(email_val,label_val,model)
adapt_corpus=np.array(adapt_corpus)
adapt_corpus=minmax(adapt_corpus)
adapt_corpus=normalize(adapt_corpus,norm='l2')
adapt_labels=np.array(adapt_labels)


#####TRAIN FILES
train_corpus=[]
train_labels=[]

train_corpus,train_labels=read_file(email_train,label_train,model)
train_corpus=np.array(train_corpus)
train_corpus=minmax(train_corpus)
train_corpus=normalize(train_corpus,norm='l2')
train_labels=np.array(train_labels)


#####TEST FILES
test_corpus=[]
test_labels=[]

test_corpus,test_labels=read_file(email_test,label_test,model)
test_corpus=np.array(test_corpus)
test_corpus=minmax(test_corpus)
test_corpus=normalize(test_corpus,norm='l2')
test_labels=np.array(test_labels)




train_time = time.time()

#cs = [0.1, 1.0, 10.0, 100.0] #Logistic regression, SVM
cs = [5,10,15,20] #Random forrest
#cs = [1, 2, 3, 5, 10] #KNN
best_c = 0
best_score = 0

for c in cs:
    #clf= LogisticRegression(C=c, penalty='l2', solver='liblinear')
    #clf = svm.LinearSVC(C=c)
    #clf = MultinomialNB()
    clf = RandomForestClassifier(n_estimators=c, n_jobs=-1)
    #clf = KNeighborsClassifier(n_neighbors=c, algorithm = 'brute', metric='cosine')
    clf.fit(train_corpus, train_labels)
    predicted = clf.predict(adapt_corpus)
    score = metrics.f1_score(adapt_labels, predicted, average='macro')
    if score > best_score:
        best_score = score
        best_c = c
   
train_time2 = time.time()

train_corpus=list(train_corpus)
train_labels=list(train_labels)
train_corpus.extend(adapt_corpus)
train_labels.extend(adapt_labels)

test_time=time.time()

#clf = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
#clf = svm.LinearSVC(C=best_c)
#clf = MultinomialNB()
clf = RandomForestClassifier(n_estimators=best_c, n_jobs=-1)
#clf = KNeighborsClassifier(n_neighbors=best_c, algorithm = 'brute', metric='cosine')
clf.fit(train_corpus, train_labels)
predicted = clf.predict(test_corpus)
accuracy = metrics.accuracy_score(test_labels, predicted)
precision = metrics.precision_score(test_labels, predicted, average='macro')
recall = metrics.recall_score(test_labels, predicted, average='macro')
f1_macro = metrics.f1_score(test_labels, predicted, average='macro')
kappa = metrics.cohen_kappa_score(test_labels, predicted)
roc = metrics.roc_auc_score(test_labels.astype(int), predicted.astype(int))
test_time2 = time.time()


print('Training time = '+str((train_time2 - train_time)/60))
print('Testing time = '+str((test_time2 - test_time)/60))
print('Accuracy: %0.2f' % accuracy)
print('Precision: %0.2f'% precision)
print('Recall: %0.2f' % recall)
print('F1: %0.2f' % f1_macro)
print('Kappa: %0.2f'% kappa)
print('AUC: %0.2f'% roc)
