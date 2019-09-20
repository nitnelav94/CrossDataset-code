# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 18:15:28 2019

@author: mmval
"""

import re

"""
main_dir='C:/Users/mmval/Documents/Semestre Enero-Junio 2019/Tesis/DataSets/Enron/enron1/ham/'
source='5129.2001-12-17.farmer.ham.txt'


with open(main_dir+source,'r', encoding='utf-8') as file:
    for lines in file:
        link = url_re.findall(lines)
        print(lines)
print(link)
"""      
URLS = r"""			# Capture 1: entire matched URL
  (?:
  https?:				# URL protocol and colon
    (?:
      /{1,3}				# 1-3 slashes
      |					#   or
      [a-z0-9%]				# Single letter or digit or '%'
                                       # (Trying not to match e.g. "URI::Escape")
    )

  )
  (?:					# One or more:
    [^\s()<>{}\[\]]+			# Run of non-space, non-()<>{}[]
    |					#   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
  )+
  (?:					# End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
    |					#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]	# not a space or one of these punct chars
  )
  |					# OR, the following to match naked domains:
  (?:
  	(?<!@)			        # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-/][a-z0-9]{4,})
    [.]
    (?:[a-z]{7})
    \b
    /?
    (?!@)			        # not succeeded by a @,
                            # avoid matching "foo.na" in "foo.na@example.com"
  )
"""

url_re = re.compile(URLS, re.VERBOSE | re.I | re.UNICODE)
phrase='j.subject corp.and/or 2001.aep/hpl i.e.delivery file:egmnom-jan.xls)-egmnom-jan.xls'  
phrase+='a http://209.84.246.173/151.html href = " http://www.teenhardcore.com/cgi-win/yeet with.ellas sol "unsubscribe freebsd-ports" in the body of the message http:book-i.net/mutou https://listman.redhat.com/mailman/listinfo/exmh-users.html in the body of the message @FreeBSD #345'
#nR=re.findall(r'[^https?:][\s*/*]+\w+.+?[/-]\w+[.-_]?\s',phrase)
#print(nR)
#print(re.findall('http[s]?\s*:[\s*/*]*(?:|[\s+]|\w|[$-_@.&+]+|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+?\s',phrase))
#print(re.findall('http[s]?\s*:[\s*/*]*\s*(?:|\s+|[.]|\w+|[$-_@.&+]+|[!*\(\),\s]|(?:%[0-9a-fA-F\s][0-9a-fA-F\s]))+?\s+',phrase))
links = url_re.findall(phrase)
print('regex:',links)
print(len(links))

#print(len(re.findall(r'http',phrase)))
#print(re.findall(r'http',phrase))
#print(len(re.findall(r'#',phrase)))
#print(re.findall(r'#',phrase))
#print(len(re.findall(r'@',phrase)))
#print(re.findall(r'@',phrase))


"""
h='hola " a todos @ hello wolrd from / python-'
h=list(h)
print(len(h))
d=[]
#print(re.findall('[^a-zA-z0-9\s]',h))
for w in range(len(h)):
    if w+1<=len(h):
        if re.match('[^a-zA-z0-9]',h[w]):
            if re.match(' ',h[w-1]) and re.match(' ',h[w+1]):
                d.append(w-1)
                d.append(w+1)
f=[]
for w in range(len(h)):
    if w not in d:
        f.append(h[w])
print(''.join(f))
    


#h.pop(2)
#print(len(h))
#print(''.join(h))
"""