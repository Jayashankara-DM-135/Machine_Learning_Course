# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:29:59 2020

@author: jmallesh
"""
from math import sqrt

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) /float(len(numbers)-1)
    return sqrt(variance)
    
def sumaring_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    print(summaries)
    del(summaries[-1])
    return summaries
    
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value , rows in separated.items():
        summaries[class_value] = sumaring_dataset(rows)
    return summaries


# Test separating data by class
dataset = [[3.393533211,2.331273381,0],
	[3.110073483,1.781539638,0],
	[1.343808831,3.368360954,0],
	[3.582294042,4.67917911,0],
	[2.280362439,2.866990263,0],
	[7.423436942,4.696522875,1],
	[5.745051997,3.533989803,1],
	[9.172168622,2.511101045,1],
	[7.792783481,3.424088941,1],
	[7.939820817,0.791637231,1]]
summary = summarize_by_class(dataset)
for class_value in summary:
    print("class value : {}".format(class_value))
    for row in summary[class_value]:
        print(row)
