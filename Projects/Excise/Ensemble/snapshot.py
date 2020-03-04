# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 22:56:04 2020

@author: jmallesh
"""

from matplotlib import pyplot
from math import pi
from math import cos
from math import floor
 
# cosine annealing learning rate schedule
def cosine_annealing(epoch, n_epochs, n_cycles, lrate_max):
	epochs_per_cycle = floor(n_epochs/n_cycles)
	cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
	return lrate_max/2 * (cos(cos_inner) + 1)
 
# create learning rate series
n_epochs = 100
n_cycles = 10
lrate_max = 0.01
series = [cosine_annealing(i, n_epochs, n_cycles, lrate_max) for i in range(n_epochs)]
# plot series
print(len(series))
print(series)
pyplot.plot(series)
pyplot.show()