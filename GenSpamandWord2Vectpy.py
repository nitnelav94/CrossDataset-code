# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:21:21 2019

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
import numpy as np
from nltk.corpus import stopwords

def my_tokenizer(s):
    return s.split()

def read_file(email_file1,email_file2,corpus,label_file1, label_file2, labels,model):
    z=[]
    with open(email_file1,'r', encoding='utf-8') as reader1, open(email_file2,'r', encoding='utf-8') as reader2, open(label_file1,'r',encoding='utf-8') as label_reader1, open(label_file2,'r',encoding='utf-8') as label_reader2:
        StopWords = stopwords.words('english')
        for i,line in enumerate(reader1):
             #print(line)
             tokens=line.rstrip().split()
             le=0
             l=np.zeros(300)
             for token in tokens:
                 if token not in StopWords and len(token)>3 and len(token)<35 and token in model:
                     le=le+1
                     #print('word ',token)
                     #print(model[token][0])
                     l=l+np.array(model[token])
                     #print(l[0]) 
                     
             if le>0:
                 l=l/le
                 corpus.append(l)
             else:
                 z.append(i)
                 
        for w,line in enumerate(label_reader1):
              if w not in z:
                  labels.append(line.strip())
        print(w)


        z=[]
        for j,line in enumerate(reader2):
             tokens=line.rstrip().split()
             le=0
             l=np.zeros(300)
             for token in tokens:
                 if token not in StopWords and len(token)>3 and len(token)<35 and token in model:
                     le=le+1
                     #print('word ',token)
                     #print(model[token][0])
                     l=l+np.array(model[token])
                     #print(l[0]) 
                     
             if le>0:
                 l=l/le
                 corpus.append(l)
             else:
                 z.append(j)
        print(len(z))
        #print(z)

# =============================================================================
#         for w,line in enumerate(label_reader1):
#               if w not in z:
#                   labels.append(line.strip())
#         print(w)
# =============================================================================
                  
        for x,line in enumerate(label_reader2):
              if x not in z:
                  labels.append(line.strip())
        print(x)
                  

def minmax(corpus):
    for i in range(len(corpus)):
        corpus[i]=(corpus[i]-np.amin(corpus[i]))/(np.amax(corpus[i])-np.amin(corpus[i]))
    return corpus

"""       
def read_labels(label_file1, label_file2, labels):
    with open(label_file1,'r',encoding='utf-8') as label_reader1, open(label_file2,'r',encoding='utf-8') as label_reader2:
        i=0
        for line in label_reader1:
            labels.append(line.strip())
            i=i+1
        print(label_file1,i)
        i=0
        for line in label_reader2:
            labels.append(line.strip())
            i=i+1
        print(label_file2,i)
"""
        
        

model = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/GoogleNews-vectors-negative300.bin', binary=True)



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

read_file(email_adapt1,email_adapt2,adapt_corpus,label_adapt1,label_adapt2,adapt_labels,model)
#read_labels(label_adapt1,label_adapt2,adapt_labels)
adapt_labels=np.array(adapt_labels)

adapt_corpus=np.array(adapt_corpus)
adapt_corpus=minmax(adapt_corpus)
adapt_corpus=normalize(adapt_corpus,norm='l2')

#####TRAIN FILES
train_corpus=[]
train_labels=[]

read_file(email_train1,email_train2,train_corpus,label_train1,label_train2,train_labels,model)
#read_labels(label_train1,label_train2,train_labels)
train_labels=np.array(train_labels)

train_corpus=np.array(train_corpus)
train_corpus=minmax(train_corpus)
train_corpus=normalize(train_corpus,norm='l2')


#####TEST FILES
test_corpus=[]
test_labels=[]



read_file(email_test1,email_test2,test_corpus,label_test1,label_test2,test_labels,model)
#read_labels(label_test1,label_test2,test_labels)
test_labels=np.array(test_labels)

test_corpus=np.array(test_corpus)
test_corpus=minmax(test_corpus)
test_corpus=normalize(test_corpus,norm='l2')





#vectorize train set
start1 = time.time()
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
    clf.fit(train_corpus, train_labels)
    predicted = clf.predict(adapt_corpus)
    score = metrics.f1_score(adapt_labels, predicted, average='macro')
    if score > best_score:
        best_score = score
        best_c = c
"""        
end1 = time.time()
start2 = time.time()
train_corpus=list(train_corpus)
train_labels=list(train_labels)
train_corpus.extend(adapt_corpus)
train_labels.extend(adapt_labels)


#clf = LogisticRegression(C=best_c, penalty='l2', solver='liblinear')
#clf = svm.LinearSVC(C=best_c)
clf = MultinomialNB()
#clf = RandomForestClassifier(n_estimators=best_c, n_jobs=-1)
#clf = KNeighborsClassifier(n_neighbors=best_c, algorithm = 'brute', metric='cosine')
clf.fit(train_corpus, train_labels)
predicted = clf.predict(test_corpus)
accuracy = metrics.accuracy_score(test_labels, predicted)
precision = metrics.precision_score(test_labels, predicted, average='macro')
recall = metrics.recall_score(test_labels, predicted, average='macro')
f1_macro = metrics.f1_score(test_labels, predicted, average='macro')
kappa = metrics.cohen_kappa_score(test_labels, predicted)
roc = metrics.roc_auc_score(test_labels.astype(int), predicted.astype(int))
stop = time.time()

print('Training time = '+str((end1 - start1)/60))
print('Testing time = '+str((stop - start2)/60))
print('Accuracy: %0.2f' % accuracy)
print('Precision: %0.2f'% precision)
print('Recall: %0.2f' % recall)
print('F1: %0.2f' % f1_macro)
print('Kappa: %0.2f'% kappa)
print('ROC: %0.2f'% roc)
