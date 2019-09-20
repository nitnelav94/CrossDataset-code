# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:56:41 2019

@author: mmval
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:53:53 2019

@author: mmval
"""
import chardet
import codecs

def readfile(file):
    text=''
    charset = chardet.detect(open(file, "rb").read()).get('encoding')
    print(file,charset)
    w=False
    with open(file,'r',encoding=charset) as content_file:
        try:
            for lines in content_file:
                #text+=lines.lower()
                if lines.lower().startswith('subject:'):
                    w=True
                    #print(lines)
                if w and not lines.lower().startswith('date:') and not lines.lower().startswith('content-type:'):
                    text+=lines.lower()
        except:
            return 'ERROR'
    
  
    
  
    #Une todas las lineas en un solo String
    textWords=[]
    textWords=text.split()
    final=[]
    subs=['subject:','fw:','fwd:','re:','re','fw:','fw','fwd','date']
    for i in textWords:
        #print(i)
        if i not in subs:
            final.append(i)
    
    text=' '.join(final)

    text=text.strip()
    
    return text
 
dataSet='LS_bare_part10'
ruta='lingspam_public'
folder='/bare/part10/'
newFolder='files_LS_bare_part10'

main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/'+ruta+folder
source='contenido.txt'

i=0
archivo=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/'+ruta+folder+newFolder+'/data_'+dataSet+'.txt','w','utf-8')

archivo_error=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/'+ruta+folder+newFolder+'/data_'+dataSet+'_error.txt','w','utf8')

with open(main_dir+source,'r',encoding="utf-8") as content_file:
    for lines in content_file:
        lines=lines.strip().lower()
        i+=1
        print(i)        
        text=readfile(main_dir+lines)
        if text!='ERROR' and text!='':
            try:
                archivo.write(text+'\n')
            except:
                archivo_error.write(lines+'\n')
        else:
            if text=='':
                archivo_error.write('Correo Vacio '+lines+'\n')
        
archivo.close()
archivo_error.close()
