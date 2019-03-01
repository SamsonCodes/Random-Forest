# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 08:58:39 2019

@author: Samson
"""

def getInterval(rows, index):
    minValue = rows[0][index]
    maxValue = rows[0][index]
    for row in rows:
        if row[index] < minValue:
            minValue = row[index]
        elif row[index] > maxValue:
            maxValue = row[index]
    return minValue, maxValue

def minmax_scaler(rows):
    ranges = []
    for i in range(0, len(rows[0])):
        ranges.append(getInterval(rows,i)) # get the range for each row: [min,max]
    return ranges

def scale(rows, scaler):    
    for row in rows:
        for x in range(0,len(row)):
            newvalue = (row[x] - scaler[x][0]) / (scaler[x][1] - scaler[x][0])
            row[x] = newvalue 
    return rows

def scale_cat(targetCategories, targetIndex, scaler):
    breakpoint()
    for x in range(0, len(targetCategories)):
        newvalue = (targetCategories[x] - scaler[targetIndex][0]) / (scaler[targetIndex][1] - scaler[targetIndex][0])
        targetCategories[x] = newvalue 
    return targetCategories 
            
rows = [[0,0,0,3],[10, 20,30,8],[5,10,20,5]]
targetCat = [3, 4, 5, 6, 7, 8]
scaler = minmax_scaler(rows)
scaled = scale(rows, scaler)
scaled_cat = scale_cat(targetCat, 3, scaler)
print(scaled)
print(scaled_cat)
