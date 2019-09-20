# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 19:14:54 2019

@author: mmval
"""

import numpy as np 

#w2v

WTR=[1, 0.008071688941511635, 0.008945816470508716, 0.010239402977017854, 0.005188591141792167,
     1, 0.008071688941511635, 0.008945816470508716, 0.010239402977017854, 0.005188591141792167,
     1, 0.008071688941511635, 0.008945816470508716, 0.010239402977017854, 0.005188591141792167,
     1, 0.008071688941511635, 0.008945816470508716, 0.010239402977017854, 0.005188591141792167,
     1, 0.008071688941511635, 0.008945816470508716, 0.010239402977017854, 0.005188591141792167]

WGS=[0.008071688941511635, 1, 0.10131800716196594, 0.12752719530523152, 0.13061124549245404,
     0.008071688941511635, 1, 0.10131800716196594, 0.12752719530523152, 0.13061124549245404,
     0.008071688941511635, 1, 0.10131800716196594, 0.12752719530523152, 0.13061124549245404,
     0.008071688941511635, 1, 0.10131800716196594, 0.12752719530523152, 0.13061124549245404,
     0.008071688941511635, 1, 0.10131800716196594, 0.12752719530523152, 0.13061124549245404]

WSA=[0.008945816470508716, 0.10131800716196594, 1, 0.1066221733841478, 0.09393534510851854,
     0.008945816470508716, 0.10131800716196594, 1, 0.1066221733841478, 0.09393534510851854,
     0.008945816470508716, 0.10131800716196594, 1, 0.1066221733841478, 0.09393534510851854,
     0.008945816470508716, 0.10131800716196594, 1, 0.1066221733841478, 0.09393534510851854,
     0.008945816470508716, 0.10131800716196594, 1, 0.1066221733841478, 0.09393534510851854]


WEN=[0.010239402977017854, 0.12752719530523152, 0.1066221733841478, 1, 0.12140545443301519,
     0.010239402977017854, 0.12752719530523152, 0.1066221733841478, 1, 0.12140545443301519,
     0.010239402977017854, 0.12752719530523152, 0.1066221733841478, 1, 0.12140545443301519,
     0.010239402977017854, 0.12752719530523152, 0.1066221733841478, 1, 0.12140545443301519,
     0.010239402977017854, 0.12752719530523152, 0.1066221733841478, 1, 0.12140545443301519]

WLS=[0.005188591141792167, 0.13061124549245404, 0.09393534510851854, 0.12140545443301519, 1,
     0.005188591141792167, 0.13061124549245404, 0.09393534510851854, 0.12140545443301519, 1,
     0.005188591141792167, 0.13061124549245404, 0.09393534510851854, 0.12140545443301519, 1,
     0.005188591141792167, 0.13061124549245404, 0.09393534510851854, 0.12140545443301519, 1,
     0.005188591141792167, 0.13061124549245404, 0.09393534510851854, 0.12140545443301519, 1]


TR=[0.97, 0.26, 0.37, 0.16, 0.12,
    0.97, 0.26, 0.36, 0.14, 0.12,
    0.95, 0.20, 0.42, 0.16, 0.29,
    0.94, 0.19, 0.29, 0.25, 0.06,
    0.95, 0.42, 0.47, 0.34, 0.29]

GS=[0.41, 0.86, 0.49, 0.44, 0.74,
    0.42, 0.86, 0.47, 0.42, 0.72,
    0.37, 0.82, 0.38, 0.31, 0.86,
    0.26, 0.50, 0.15, 0.17, 0.33,
    0.35, 0.80, 0.48, 0.42, 0.69]

SA=[0.16, 0.25, 0.90, 0.32, 0.80,
    0.24, 0.31, 0.90, 0.39, 0.81,
    0.14, 0.18, 0.72, 0.17, 0.30,
    0.41, 0.40, 0.82, 0.40, 0.60,
    0.27, 0.26, 0.84, 0.33, 0.43]

EN=[0.38, 0.64, 0.19, 0.96, 0.28,
    0.36, 0.64, 0.17, 0.96, 0.29,
    0.51, 0.60, 0.21, 0.96, 0.55,
    0.36, 0.42, 0.27, 0.92, 0.30,
    0.24, 0.63, 0.23, 0.95, 0.42]

LS=[0.23, 0.28, 0.59, 0.47, 0.96,
    0.20, 0.26, 0.58, 0.44, 0.95,
    0.02, 0.07, 0.16, 0.07, 0.52,
    0.16, 0.20, 0.10, 0.36, 0.92,
    0.24, 0.17, 0.48, 0.33, 0.91]

print('TR: ',np.corrcoef(WTR,TR)[0][1])
print('GS: ',np.corrcoef(WGS,GS)[0][1])
print('SA: ',np.corrcoef(WSA,SA)[0][1])
print('EN: ',np.corrcoef(WEN,EN)[0][1])
print('LS: ',np.corrcoef(WLS,LS)[0][1])




