# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 08:38:22 2020

@author: jmallesh
"""

import numpy as np
from sklearn.model_selection import KFold


data = np.array([99, 199, 299, 399, 599, 699, 799])
kflod = KFold(3, shuffle=True, random_state=1)

for train, test in kflod.split(data):
    print("train:{} test:{}".format(train, test))
    print('train:{} , test:{} '.format(data[train], data[test]))
    


    

