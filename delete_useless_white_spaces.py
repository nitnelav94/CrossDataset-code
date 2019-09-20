# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:27:26 2019

@author: mmval
"""
import codecs
import re 

def del_spaces(phrase):
    phrase=list(phrase)
    spaces=[]
    for i in range(len(phrase)):
        if i+2<=len(phrase) and i-1>=0:
            if re.match('[^a-zA-Z0-9]',phrase[i]):
                if re.match(' ',phrase[i+1]) and re.match(' ',phrase[i-1]):
                    if phrase[i]=='#':
                        spaces.append(i+1)
                    else:
                        spaces.append(i+1)
                        spaces.append(i-1)
    final=[]
    for i in range(len(phrase)):
        if i not in spaces:
            final.append(phrase[i])
    text=''.join(final)
    if text!='':
        return text
    else:
        return 'ERROR'
    
    
dataSet='enron1_spam'
ruta='Enron/'
folder='enron1/'
newFolder='files_enron1_spam'
    
    
main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/'+ruta+folder+newFolder
source='/data_enron1_spam.txt'

archivo=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/'+ruta+folder+newFolder+'/new_data_'+dataSet+'.txt','w','utf-8')

i=0
space=[]
with open(main_dir+source,'r',encoding="utf-8") as content_file:
    for lines in content_file:
       print(i)
       try:
           archivo.write(del_spaces(lines))
       except:
           archivo.write('ERROR')
       i=i+1
                
                    
                    