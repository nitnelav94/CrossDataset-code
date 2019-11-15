# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:53:53 2019

@author: mmval
"""
import chardet
import codecs

def readfile(file):
    Lines=[]
    
    #Abre el documento, lo lee por lineas y elimina espacion y saltos de linea
    charset = chardet.detect(open(file, "rb").read()).get('encoding')
    print(file,charset)
    #with open(file,'r',encoding=charset) as content_file:
     #   for lines in content_file:
      #          Lines.append(lines.strip().lower())
    
    with open(file,'r',encoding=charset) as content_file:
        try:
            for lines in content_file:
                Lines.append(lines.strip().lower())
        except:
            return 'ERROR !'
    
  
    
  
    #Une todas las lineas en un solo String
    text=''
    for i in range(len(Lines)):
        text+=Lines[i]+' '

    #Elimina ciertas palabras dentro de los correos   
    text=text.replace('subject','')
    text=text.replace(':','')
    text=text.replace('fw','')
    text=text.replace('re','')
    text=text.replace('fwd','')
    text=text.strip()
    
    return text
 

main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/pu_corpora_public/Enron/enron1/spam/'
source='contenido.txt'

i=0
archivo=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/pu_corpora_public/pua/part10/data_pua_part10.txt','w','utf-8')



archivo_error=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/pu_corpora_public/pua/part10/data_pua_part10_error.txt','w','utf8')

with open(main_dir+source,'r',encoding="utf-8") as content_file:
    for lines in content_file:
        lines=lines.strip().lower()
        i+=1
        print(i)
        #text=readfile(main_dir+lines)
        #archivo.write(text+'\n')
        
        text=readfile(main_dir+lines)
        if text!='ERROR':
            try:
                archivo.write(text+'\n')
            except:
                archivo_error.write(lines+'\n')
        else:
            archivo_error.write(lines+'\n')
        
archivo.close()
archivo_error.close()
