# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 09:47:30 2019

@author: Samson
"""
import numpy as np

def means(rows):
    total = []
    for x in rows[0]:
        total.append(0)
    for row in rows:
        for x in range(0, len(row)):
            total[x] += row[x]
    for x in range(0,len(total)):
        total[x] = total[x]/len(rows)
    return total

# calculates standardization parameters
def standardizer(rows):
    means = list(np.mean(rows,0))
    stds = list(np.std(rows,0))
    return [means, stds]

# standardizes using standardizer stdizer
def standardize(rows, stdizer):
    for y in range(0, len(rows)):
        for x in range(0,len(rows[y])):
            if not stdizer[1][x]==0:
                rows[y][x] = (rows[y][x]-stdizer[0][x])/stdizer[1][x] # subract mean, divide by stdev            
    return rows

# standardize the targetCategories vector using stdizer           
def stdize_cat(targetCategories, targetIndex, stdizer):
    for x in range(0, len(targetCategories)):
        if not stdizer[1][targetIndex]==0:
                targetCategories[x] = (targetCategories[x]-stdizer[0][targetIndex])/stdizer[1][targetIndex] 
                # subract mean, divide by stdev 
    return targetCategories  

#rows = [[0,0,0,3],[10, 20,30,8],[5,10,20,5]]
rows = [[-1,0,5],[0,0,3],[1,0,8]]
targetCat = [3, 4, 5, 6, 7, 8]
"""
print(means(rows))
print(list(np.mean(rows,0)))
print(list(np.std(rows, 0)))
"""
stdizer = standardizer(rows)
stdized = standardize(rows, stdizer)
rounded = []
for row in stdized:
    rrow = []
    for x in row:
        rrow.append(round(x,2))
    rounded.append(rrow)
print(rounded)
stdized_cat = stdize_cat(targetCat,2,stdizer)
rounded_cat = []
for elem in stdized_cat:
    rounded_cat.append(round(elem,2))
print(rounded_cat)