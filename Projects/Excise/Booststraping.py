# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:37:30 2020

@author: jmallesh
"""

# scikit-learn bootstrap
from sklearn.utils import resample
# data sample
data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# prepare bootstrap sample
boot = resample(data, replace=True, n_samples=4, random_state=1)
print('Bootstrap Sample: %s' % boot)
# out of bag observations
oob = [x for x in data if x not in boot]
print('OOB Sample: %s' % oob)


"""
Note:
    oob = [x for x in data if x not in boot]
    Here:
        first "for x in data" is excuted first and then if x not in boot.
"""


