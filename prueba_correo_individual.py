# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 11:43:50 2019

@author: mmval
"""

import chardet
import codecs 


def readfile(file):
    text=''
    
    #Abre el documento, lo lee por lineas y elimina espacion y saltos de linea
    charset = chardet.detect(open(file, "rb").read()).get('encoding')
    print(file,charset)
    #with open(file,'r',encoding=charset) as content_file:
     #   for lines in content_file:
      #          Lines.append(lines.strip().lower())
    w=False
    with open(file,'r',encoding=charset) as content_file:
        try:
            for lines in content_file:
                if lines.lower().startswith('subject:'):
                    w=True
                    #print(lines)
                if w and not lines.lower().startswith('date:') and not lines.lower().startswith('content-type:'):
                    text+=lines.lower()
                    #print(lines)
        except:
            return 'ERROR'
    
  
    print(text)
  
    #Une todas las lineas en un solo String
    textWords=[]
    textWords=text.split()
    final=[]
    subs=['subject:',':','fw','fwd','re']
    for i in textWords:
        if i not in subs:
            final.append(i)
    print(final)
    text=' '.join(final)
            

    #Elimina ciertas palabras dentro del correo   
    #text=text.replace('subject','')
    #text=text.replace(':','')
    #text=text.replace('fw','')
    #text=text.replace('re','')
    #text=text.replace('fwd','')
    text=text.strip()
    
    return text


main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/Enron/enron1/spam/'
#main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/trec07p_v2/data/'

source='0303.2004-01-24.gp.spam.txt'

archivo=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/SpamAssassin/prueba_.txt','w','utf-8')
#archivo=open('C:/Users/mmval/Documents/prueba_individual.txt','w')
i=0
"""
with open(main_dir+source,'r',encoding="utf-8") as content_file:
    for lines in content_file:
        lines=lines.strip().lower()
        i+=1
        print(i)
        text=readfile(main_dir+lines)
        archivo.write(text+'\n')

"""
text=readfile(main_dir+source)
archivo.write(text+'\n')
archivo.close()
        
