# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 12:05:36 2019

@author: mmval
"""

from bs4 import BeautifulSoup
import codecs
import chardet
import re 

main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/GenSpam/'
source='train_SPAM.ems'
i=0
w=False
text=''
archivo=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/GenSpam/train_SPAM/data_GenSpam_train_SPAM.txt','w','utf-8')

charset = chardet.detect(open(main_dir+source, "rb").read()).get('encoding')
with open(main_dir+source,'r',encoding=charset) as content_file:
    for lines in content_file:
        lines=lines.lower()
        i+=1
        text+=lines
        
text=text.split()
#print(text)
f=''
l=[]
remov=[':','fw','re','fwd','^','<subject>','<text_normal>','</subject>','</text_normal>']
s=False
for words in text:
    if words=='<subject>' or words=='<message_body>':
        w=True
    if w and not re.match('&\w+',words) and words not in remov and not re.match(r'<(.*?)>',words) and not re.match(r'[.\w+]*<(.*?)>',words) and not re.match(r'</*\w+>',words) and not re.match('\W*<part num=\W[0-9]\W type=\W\w+\W\w+\W>',words):
        f+=words+' '
        #s=False
    if words=='</subject>' or words=='</message_body>':
        w=False
        #s=True
    if words=='</message_body>':
        if not re.match('^\s*$',f):
            l.append(f.strip())
            f=''
        

f='\n'.join(l)
f=re.sub(r'</*\w+>','',f)
f=re.sub('[.\w+]*<(.*?)>','',f)



archivo.write(f)
print(f)


