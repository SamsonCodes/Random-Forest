# -*- coding: utf-8 -*-
"""
Random Forest

Created on Wed Feb 25 2019

@author: Samson
"""
import pandas as pd
import numpy as np
from random import randint
from math import sqrt

operators = ['==','<','>']


#See Word document for encoding
Golf = [[0,2,1,0,0],
        [0,2,1,1,0],
        [1,2,1,0,1],
        [2,1,1,0,1],
        [2,0,0,0,1],
        [2,0,0,1,0],
        [1,0,0,1,1],
        [0,1,1,0,0],
        [0,0,0,0,1],
        [2,1,0,0,0],
        [0,1,0,1,1],
        [1,1,1,1,1],
        [1,2,0,0,1],
        [2,1,1,1,0]]
data1 = pd.DataFrame(data = Golf, columns = ["Outlook","Temp","Humidity","Windy","Play"], copy = False)
targetIndex1 = 4
targetCategories1 = [0,1]

testData1 = []
for x1 in range(0,3):
    for x2 in range(0,3):
        for x3 in range(0,2):
            for x4 in range(0,2):
                    row = [x1,x2,x3,x4]
                    testData1.append(row)
             
# Red wine data
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data2 = pd.read_csv(dataset_url, sep=';')
print('loaded data!')
targetIndex2 = len(data2.values[0]) - 1
print("targetIndex2 = " + str(targetIndex2))
targetSet = set(data2.values[:,targetIndex2])
targetCategories2 = []
for x in targetSet:
    targetCategories2.append(int(x))
print("targetCategories2 = " + str(targetCategories2))

stepAmount = 10 # amount of steps per feature value

# From random-forest/tutorial: Decision tree from scratch
def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

# This is the Leaf class
class Leaf:
    # This is the constructor
    def __init__(self, rows):
        self.predictions = class_counts(rows) # a dictionary of label -> count.
        
# This is the Node class
class Node:
    # This is the constructor
    def __init__(self, question, trueBranch, falseBranch):        
        self.question = question
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch       

# This class saves the splitcondition            
class Question:
    def __init__(self, operator, valueIndex, splitValue):
        self.operator = operator
        self.valueIndex = valueIndex
        self.splitValue = splitValue
    # This returns the splitcondition as a question in string format
    def text(self):
        return (self.valueIndex + ' ' + self.operator 
                + ' ' + str(self.splitValue) + "?")

# This builds the tree recursively
# Adapted from random-forest/tutorial: Decision tree from scratch
def buildTree(rows, targetCategories,depth):
    print("#",end='')
    targetVariable = len(rows[0]) - 1  
    info, question = findBestSplit(targetCategories, targetVariable, rows) #targetVariable is last column
    if info == 0 or question == None or depth == 0: return Leaf(rows)
    trueRows, falseRows = getSubsets(rows, question)
    trueBranch = buildTree(trueRows, targetCategories, depth - 1)
    falseBranch = buildTree(falseRows, targetCategories, depth - 1)
    return Node(question, trueBranch, falseBranch)

# This finds the best split based on the gini index
def findBestSplit(targetCategories, targetVariable, rows): 
    bestGain = None
    bestQuestion = None
    for operator in operators:
        for variableIndex in range(0, len(rows[0])):            
            if not (variableIndex == targetVariable):
                interval = getInterval(rows, variableIndex)
                varRange = interval[1] - interval[0]
                for x in range(0, stepAmount):
                    value = interval[0] + (varRange/stepAmount) * x                    
                    question = Question(operator, variableIndex, value)
                    if not hasEmptySets(getSubsets(rows, question)):
                        gain = infoGain(targetCategories, targetVariable, rows, question)
                        if bestGain == None:                        
                            bestGain = gain
                            bestQuestion = question
                            
                        elif gain > bestGain:                        
                            bestGain = gain
                            bestQuestion = question 
    return bestGain, bestQuestion

# This checks whether a row is a positive or negative on the split question
def answer(question, row):    
    operator = question.operator
    switcher = {
        '<': smallerThan,
        '>': biggerThan,
        '==': equalTo,   
    }
    func = switcher.get(operator, lambda: "Invalid input")
    return func(row[question.valueIndex], question.splitValue) 

def smallerThan(a, b):
    if(a < b):
        return True
    return False

def biggerThan(a,b):
    if(a > b):
        return True
    return False

def equalTo(a, b):
    if(a == b):
        return True
    return False

# This calculates the gini index for a single set
def giniIndex(targetCategories, targetVariable, rows):
    categories = []
    for c in targetCategories:
        categories.append([])
    for row in rows:
        for c in targetCategories:
            if(row[targetVariable] == c):
                categories[targetCategories.index(c)].append(row)           
    blob = 0
    for c in range (0, len(categories)):
        blob+=(len(categories[c])/len(rows))**2
    gini = 1 - blob   
    return gini   

# This calculates the weighted gini index for a set of sets
# That means it calculates the weighted (by size) average
def weightedGini(targetCategories, targetVariable, subsets):
    totalLength = 0
    for subset in subsets:
        totalLength += len(subset)
    weightedSum = 0
    for subset in subsets:
        subGini = giniIndex(targetCategories, targetVariable, subset)   
        weightedSum += subGini * len(subset)/totalLength
    return weightedSum  

#This calculates the information gain that results from the splitting of a set
#Based on a certain splitcondition (question)
def infoGain(targetCategories, targetVariable, rows, question): 
    gini1 = giniIndex(targetCategories, targetVariable, rows)
    gini2 = weightedGini(targetCategories, targetVariable, getSubsets(rows, question))
    return (gini1 - gini2)   

# This splits the rows based on the splitcondition and returns the resulting sets
def getSubsets(rows, question):
    subsets = []
    for s in range(0,2):
        subsets.append([])
    for row in rows:
        if answer(question, row):
            subsets[0].append(row)
        else:
            subsets[1].append(row)  
    return subsets

# Checks whether or not a set contains empty subsets
def hasEmptySets(subsets):
    for s in subsets:
        if(len(s) == 0):
            return True
    return False 

# Gets the interval (range) for a certain row
def getInterval(rows, index):
    minValue = rows[0][index]
    maxValue = rows[0][index]
    for row in rows:
        if row[index] < minValue:
            minValue = row[index]
        elif row[index] > maxValue:
            maxValue = row[index]
    return minValue, maxValue

# Adapted from random-forest/tutorial: Decision tree from scratch
def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + node.question.text())

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.trueBranch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.falseBranch, spacing + "  ")    

# Adapted from random-forest/tutorial: Decision tree from scratch    
def classify(row, node):
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if answer(node.question, row):
        return classify(row, node.trueBranch)
    else:
        return classify(row, node.falseBranch)

def buildforest(rows, targetCategories, targetIndex, n, depth = 10, maxSetSize = 100):
    rows0 = rows
    forest = []
    print("building forest...")
    for x in range (0, n):
        width = len(rows0[0])
        features = width - 1
        k = (features/4)*3
        columns = [targetIndex]
        while len(columns) < k + 1:
            ri = randint(0,features-1)
            if not ri in columns:
                columns.append(ri)
        columns.sort()
        rows = rows0[:,columns]
        trainingSize = min(len(rows0), maxSetSize)
        if(trainingSize == maxSetSize):
            trainingrows = []
            while (len(trainingrows) < trainingSize):
                ri = randint(0,len(rows0)-1)
                if not ri in trainingrows:
                    trainingrows.append(ri)
            trainingrows.sort()
            rows = rows[trainingrows,:]
             
        tree = buildTree(rows, targetCategories,depth)        
        print(str(x+1) + "/" + str(n))
        """
        print("\nTree:")
        print("columns=",columns)
        print_tree(tree)
        """
        forest.append([tree,columns])
    print("forest complete!")
    return forest

def forestclassify(row, forest): 
    row0 = row
    votes = []
    for entry in forest:
        tree = entry[0]
        columns = entry[1]
        nrow = [row0[i] for i in columns]
        row = nrow
        votes.append(classify(row, tree))
    combined = combine_votes(votes)
    prediction = winner(combined)    
    return [combined, prediction, row0[-1] == prediction]
        
def combine_votes(votes):
    """Combines the votes into a single dictionary"""
    votecounts = {}  # a dictionary of label -> count.
    for vote in votes:
        for key in vote:
            if key not in votecounts:
                votecounts[key] = 0
            votecounts[key] += vote[key]   
    return votecounts

def winner(vote):
    maxKey = list(vote.keys())[0]
    maxValue = vote[maxKey]
    for key in vote:
        if vote[key] > maxValue:
            maxKey = key
            maxValue = vote[key]
    return maxKey

def accuracy(forest, testSet):
    correct = 0
    size = len(testSet.values)
    for x in testSet.values:
        predictionData = forestclassify(x, forest) 
        if predictionData[2]:
            correct+=1
        else:
            print("False prediction " + str(x[-1]) + " --> " 
              + str(predictionData[1]) + " (" + str(predictionData[0]) + ")")
    return correct/size

def MSE(forest, testSet): # mean squared error
    size = len(testSet.values)
    TSE = 0 # total squared error
    for x in testSet.values:
        predictionData = forestclassify(x, forest) 
        TSE += error(predictionData[0],x[-1])**2        
    return TSE/size

# Get the weighted prediction
def exact_prediction(vote):
    total = 0
    for key in vote:
        total += vote[key]
    pred = 0
    for key in vote:
        pred += vote[key]/total * key
    return pred

def error(vote, actual): # calculates the error in prediction  
    return actual - exact_prediction(vote)

def correlation(x, y): #For x and y lists of numbers of equal length
    if len(x) == len(y):
        N = len(x)
        sumX = 0
        sumY = 0
        sumXY = 0
        sumX2 = 0
        sumY2 = 0    
        for n in range(0,N):
            sumX += x[n]
            sumY += y[n]
            sumXY += x[n]*y[n]
            sumX2 += x[n]*x[n]
            sumY2 += y[n]*y[n]
        return (N*sumXY - sumX*sumY) / sqrt((N*sumX2 - sumX**2)*(N*sumY2 - sumY**2))
    return 0
        
#MAIN PROGRAM
print("running! \n")
print("Data2 head:")
print(data2.head())

trainingset = data2.loc[0:1499] #Use first 1500 values as the trainingset
testSet = data2.loc[1500:1599] # Use the last 100 values as the testset
forest = buildforest(trainingset.values,targetCategories2,targetIndex2, 10, 5, 50) # Build a random forest of 20 trees

print("Accuracy = " + str(round(accuracy(forest,testSet),2)))
predictionList = []
labelList = []
for x in testSet.values:
    predictionList.append(exact_prediction(forestclassify(x,forest)[0]))
    labelList.append(x[-1])
r = correlation(predictionList,labelList)
print("r = " + str(round(r,2)))
print("r2 = " + str(round(r**2,2)))
print("MSE = " + str(round(MSE(forest,testSet),2)))


print("done!")