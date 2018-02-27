# -*- coding: utf-8 -*-
"""
Created on Tue Sep 05 09:46:02 2017

@author: theDurphy
"""

import pandas as pd
import numpy as np
import time


a = np.random.rand(100000000)
b = np.random.rand(100000000)

tic = time.time()
c = 0
for i in xrange(len(a)):
     c += a[i]*b[i]
toc = time.time()
print "For loop took {0}ms to calculate c = {1}".format((toc-tic)*1000, c)


tic = time.time()
c = (pd.Series(a)*pd.Series(b)).sum()
toc = time.time()
print "Pandas took {0}ms to calculate c = {1}".format((toc-tic)*1000, c)

tic = time.time()
c = np.dot(a, b)
toc = time.time()
print "Numpy Dot took {0}ms to calculate c = {1}".format((toc-tic)*1000, c)
