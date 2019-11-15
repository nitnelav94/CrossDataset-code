# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 12:01:16 2019

@author: mmval
"""
import codecs

main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/trec07p/full/'
source='cleaned_index.txt'

i=0
archivo_labels=codecs.open('C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/trec07p/data/files_trec07p_data/labels_trec07p.txt','w','utf-8')

with open(main_dir+source,'r',encoding="utf-8") as content_file:
    for lines in content_file:
        #archivo_labels.write('0\n')
        if lines.find('spam')==0:
            archivo_labels.write('0\n');
        else:
            archivo_labels.write('1\n');
"""        
        lines=lines.strip().lower()
        i+=1
        #print(i)
        print(type(lines.split()))
        lines=lines.split(',')
        print(lines)
        print('\n',lines[-1])
        if(lines[-1]=='0'):
            archivo_labels.write('1\n')
        else:
            archivo_labels.write('0\n')
        del lines[-1]
        print(lines)
        archivo_new_data.write(' '.join(lines)+'\n')
        #archivo_labels.write('0\n')
"""        
        
        
   
archivo_labels.close()
#archivo_new_data.close()

