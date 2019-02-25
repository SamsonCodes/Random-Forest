# -*- coding: utf-8 -*-
"""
DECISION TREE 2.0

Created on Wed Jan 16 08:01:45 2019

@author: Samson
"""
import pandas as pd
from operator import itemgetter
from tkinter import *  

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
data = pd.DataFrame(data = Golf, columns = ["Outlook","Temp","Humidity","Windy","Play"], copy = False)
targetIndex = 4
targetCategories = [0,1]

testData = []
for x1 in range(0,3):
    for x2 in range(0,3):
        for x3 in range(0,2):
            for x4 in range(0,2):
                testData.append([x1,x2,x3,x4])

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
        return (data.columns[self.valueIndex] + ' ' + self.operator 
                + ' ' + str(self.splitValue) + "?")

# This builds the tree recursively
# Adapted from random-forest/tutorial: Decision tree from scratch
def buildTree(rows):
    info, question = findBestSplit(targetCategories, targetIndex, rows)
    if info == 0 or question == None: return Leaf(rows)
    trueRows, falseRows = getSubsets(rows, question)
    trueBranch = buildTree(trueRows)
    falseBranch = buildTree(falseRows)
    return Node(question, trueBranch, falseBranch)

# This finds the best split based on the gini index
def findBestSplit(targetCategories, targetVariable, rows): 
    bestGain = None
    bestQuestion = None
    stepAmount = 10
    for operator in operators:
        for variableIndex in range(0, len(rows[0])):            
            if not (variableIndex == targetIndex):
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
                categories[c].append(row)           
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

#MAIN PROGRAM
print("running! \n")
print("Data head:")
print(data.head())

print("\nTree:")
tree = buildTree(data.values)
print_tree(tree)

print("\nPredictions:{label->count}")
for row in testData:
    print(str(row) + "--->" + str(classify(row,tree)))    

print("done!")