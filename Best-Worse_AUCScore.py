# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:05:00 2019

@author: mmval
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 10:19:57 2019

@author: mmval
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:51:22 2019

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

def read_file(email_file1, corpus):
    with open(email_file1,'r', encoding='utf-8') as reader1:
        #StopWords = stopwords.words('english')
        for line in reader1:
            tokens=line.rstrip().split()
            #for words
            #text=' '.join([token for token in tokens if token not in StopWords and len(token)>3 and len(token)<35])
            #
            text=' '.join([token for token in tokens])
            corpus.append(text)

    
    
def read_labels(label_file1, labels):
    with open(label_file1,'r',encoding='utf-8') as label_reader1:
        for line in label_reader1:
            labels.append(line.strip())

        

c='hashtags'

main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/'
email_val=main_dir+'trec07p/FilesTrec07_2/trec07p_'+c+'_val.txt'
label_val=main_dir+'trec07p/FilesTrec07_2/trec07p_labels_'+c+'_val.txt'

email_train=main_dir+'trec07p/FilesTrec07_2/trec07p_'+c+'_train.txt'
label_train=main_dir+'trec07p/FilesTrec07_2/trec07p_labels_'+c+'_train.txt'

email_test=main_dir+'GenSpam/GenSpamFull/GenSpam_'+c+'.txt'
label_test=main_dir+'GenSpam/GenSpamFull/GenSpam_labels_full.txt'


######ADAPT FILES
adapt_corpus=[]
adapt_labels=[]

read_file(email_val,adapt_corpus)
read_labels(label_val,adapt_labels)
adapt_labels=np.array(adapt_labels)


#####TRAIN FILES
train_corpus=[]
train_labels=[]

read_file(email_train,train_corpus)
read_labels(label_train,train_labels)
train_labels=np.array(train_labels)


#####TEST FILES
test_corpus=[]
test_labels=[]

read_file(email_test,test_corpus)
read_labels(label_test,test_labels)
test_labels=np.array(test_labels)



#vectorize train set
train_time = time.time()
vec = TfidfVectorizer(min_df=1, norm='l2', analyzer = 'word', tokenizer=my_tokenizer)
train_tfidf = vec.fit_transform(train_corpus)

adapt_tfidf = vec.transform(adapt_corpus)

"""
#cs = [0.1, 1.0, 10.0, 100.0] #Logistic regression, SVM
#cs = [5,10,15,20] #Random forrest
cs = [1, 2, 3, 5, 10] #KNN
best_c = 0
best_score = 0

for c in cs:
    #clf= LogisticRegression(C=c, penalty='l2', solver='liblinear')
    #clf = svm.LinearSVC(C=c)
    #clf = MultinomialNB()
    #clf = RandomForestClassifier(n_estimators=c, n_jobs=-1)
    clf = KNeighborsClassifier(n_neighbors=c, algorithm = 'brute', metric='cosine')
    clf.fit(train_tfidf, train_labels)
    predicted = clf.predict(adapt_tfidf)
    score = metrics.f1_score(adapt_labels, predicted, average='macro')
    if score > best_score:
        best_score = score
        best_c = c
"""     
train_time2 = time.time()

train_corpus=list(train_corpus)
train_labels=list(train_labels)
train_corpus.extend(adapt_corpus)
train_labels.extend(adapt_labels)

test_time=time.time()
vec = TfidfVectorizer(min_df=1, norm='l2', analyzer = 'word', tokenizer=my_tokenizer)
train_tfidf = vec.fit_transform(train_corpus)
test_tfidf = vec.transform(test_corpus)

#clf = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
#clf = svm.LinearSVC(C=best_c)
clf = MultinomialNB()
#clf = RandomForestClassifier(n_estimators=best_c, n_jobs=-1)
#clf = KNeighborsClassifier(n_neighbors=best_c, algorithm = 'brute', metric='cosine')
clf.fit(train_tfidf, train_labels)
predicted = clf.predict(test_tfidf)
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
print('ROC: %0.2f'% roc)