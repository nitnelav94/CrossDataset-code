# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 20:37:54 2019

@author: mmval
"""

from collections import Counter
from nltk.corpus import stopwords
import pandas as pd

features=['hashtags','words','links','ats','emoticons']

#PARA TREC
main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/trec07p/data/files_trec07p_data/'



#PARA ENRON
#main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/Enron/'



#PARA SPAMASSASSIN
#main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/SpamAssassin/'


#PARA LS bare
#main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/lingspam_public/bare/LS_Bare2/'

#PAra GenSpam
#main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/GenSpam/GenSpamFull/'

file =main_dir+'FeaturesPerEmail_PerDataset.csv'
data=pd.read_csv(file)

#j=0
inf=[]
for feature in features:
    #Para trec07p
    email_file=main_dir+'trec07p_'+feature+'.txt'
    
    #Para enron
    #email_file=main_dir+'Enron_'+feature+'.txt'
    
    #Para SpamAssassin
    #email_file=main_dir+'SA_'+feature+'.txt'
    
    #Para GenSpam
    #email_file=main_dir+'GenSpam_'+feature+'.txt'
    
    #Para LS bare
    #email_file=main_dir+'LS_Bare_'+feature+'.txt'
    
    
    
    with open(email_file,'r', encoding='utf-8') as reader1:
            StopWords = stopwords.words('english')
            #item=0
            #le=0
            #inf=[]
            items=[]
            for line in reader1:
                tokens=line.rstrip().split()
                #items=[]
                item=0
                #le+=len(tokens)
                #print('longitud total: ',len(tokens))
                for token in tokens:
                    if feature=='words':
                        if token not in StopWords and len(token)>3 and len(token)<35:
                            item+=1
                    else:
                        item+=1
                items.append(item)
    
    
    
    #data.insert(j,feature,(ite for ite in items))
    #data.to_csv(file)
    print('longitud total: ',len(items))
    inf.append(items)
    print(len(inf))
    #print(data)
    #j+=1
                
            #print('longitud total: ',le)
            
            
    print(feature)
    print(item)
    print('\n***********************')




df = pd.DataFrame(data={"hashtags": inf[0], "words": inf[1],'links':inf[2],'ats':inf[3],'emoticons':inf[4]})
df.to_csv(file, sep=',',index=False)



"""
import csv
import pandas as pd

file ='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/test.csv'
data=pd.read_csv(file)
data.head()
data.insert(2,'hola',3,4)
data.to_csv(file)
data.head()
"""

