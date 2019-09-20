# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:51:47 2019

@author: mmval
"""

import numpy as np

def read_file(file):
	content=[]
	with open(file,'r',encoding='utf-8') as reader:
		for line in reader:
			content.append(line.rstrip())
	return content
	
def sub_sample(data, *idxs):
	sample=[]
	for idx in idxs:
		sample.extend([data[i] for i in idx])		#extend agrega varias cosas y append solo una
	return sample
	
def write_file(file,data):
	with open(file, 'w',encoding='utf-8') as writer:
		for line in data:
			writer.write(line+'\n')
	

#66629726
d='LS_Bare'
c='ats'
main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/lingspam_public/bare/LS_Bare2/'
data_file=main_dir+d+'_'+c+'.txt'
labels_file=main_dir+'LS_Bare_Full_Labels.txt'

train_file=main_dir+d+'_'+c+'_train.txt'
val_file=main_dir+d+'_'+c+'_val.txt'

labels_train_file=main_dir+d+'_labels_'+c+'_train.txt'
labels_val_file=main_dir+d+'_labels_'+c+'_val.txt'


data=read_file(data_file)
labels=read_file(labels_file)

s=2025
c=2893
f=1447



idx_1=np.random.permutation(s) #70
idx_2=np.random.permutation(list(range(s,c))) #70:100

train=sub_sample(data,idx_1[:f],idx_2[:f])	#:50	
val=sub_sample(data,idx_1[f:c],idx_2[f:c]) #50:100


labels_train=sub_sample(labels,idx_1[:f],idx_2[:f])	#:50	
labels_val=sub_sample(labels,idx_1[f:c],idx_2[f:c]) #50:100


write_file(train_file,train)
write_file(val_file,val)


write_file(labels_train_file,labels_train)
write_file(labels_val_file,labels_val)