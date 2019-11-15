# -*- coding: utf-8 -*-
"""
Created on Saturday Feb  9 16:10:45 2019

@author: mmval
"""
import re
import codecs
import chardet

def getTags(file):
    scan=False
    tags=[]
    
    charset = chardet.detect(open(file, "rb").read()).get('encoding')
    print(file,charset)
    
    with open(file, 'r', encoding=charset) as contentFile:
        try:
            for line in contentFile:
                line=line.strip();
                #print(line)
                if line!='\n' and len(line)>0:
                    line=line.lower().split()
                    if line[0]=='subject:' and len(lines[0])>0:
                        scan=True
                    if re.match('[a-z0-9-]+:',line[0]) and not re.match('http[s]:.+',line[0]) and scan:
                        #print('match',line[0])
                        tags.append(line[0])
        except:
            return 'ERROR'
    if  len(tags)>0:
        del tags[0]
    if 'url:' in tags:
        del tags[tags.index('url:')]
    print(tags)
    return tags

def readfile(file):
    tags=getTags(file)
    if tags=='ERROR':
        return 'ERROR'
    wr=False
    tx=''
    charset = chardet.detect(open(file, "rb").read()).get('encoding')
    print(file,charset)
    
    with open(file, 'r', encoding=charset) as file:
        try:
            for line in file:
                #for SpamAssassin
                #line=line.replace('>','')
                line=line.strip()
                if line!='\n' and len(line)>0 and not line.startswith('<'):
                    wordsInLine=line.lower().split()
                    if wordsInLine[0]=='subject:':
                        wr=True
                    #For SpamAssassin
                    #if wr and wordsInLine[0] not in tags and not line.startswith('boundary') and not line.startswith('micalg') and not line.startswith('protocol'):
                    #For trc07
                    if wr and wordsInLine[0] not in tags and not re.match('.+=',wordsInLine[0]): 
                        tx+=line.lower().strip()+' '
        except:
            return 'ERROR'
    subs=['fw:','fwd:','re:','re','fw:','fw','fwd','subject:','url:']
    rx=tx.split()
    for i in range(len(rx)):
        if rx[i] in subs:
            rx[i]=''
    tx=' '.join(rx)
    tx=re.sub('\s+',' ',tx)
    #for trec07
    tx=re.sub(r'<.+>','',tx)
    tx=tx.strip()

    return tx    


#month='12'
dataSet='trec07p'#+month
ruta='trec07p'
folder='/data/'#+month+'/'
newFolder='files_trec07p'#+month


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
            if text=='ERROR':
                archivo_error.write('ERROR! '+lines+'\n')
        
