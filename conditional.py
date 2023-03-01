#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# from tabulate import tabulate
import numpy as np
# import bnlearn as bn
import pgmpy
import itertools
from scipy.stats import chi2_contingency #check for independence between 2 variables
from scipy.stats import power_divergence
from pgmpy.estimators.CITests import log_likelihood
from pgmpy.estimators.CITests import chi_square
# from pgmpy.factors.continuous.discretize import BaseDiscretizer


# In[2]:


def csv_format_discrete(csv_file):
    df = pd.read_csv(csv_file, sep="\s+")
    return df.drop([0])

#returns the csv_file in a pandas dataframe, formatted properly, discrete dataset only


# In[3]:


def csv_format_discrete(csv_file):
    df = pd.read_csv(csv_file, sep="\s+")
    return df.drop([0])


# In[4]:


def gaussian_reader(csv_file):
    df = pd.read_csv(csv_file, sep="\s+")
    return df


# In[5]:


#how do we measure the consistensy
def chi2bool(df):
#returns a tuple(chi2, p_value, dof) if boolean = false
#the null hypothesis is that they are independent of each other
#if true, the p_value is higher than the significance test, we do not reject the null hypothesis
#if false, the p_value is lower than the significance test, we reject the null hypothesis
    v = list(df)
    empty = []
    empty_1 = []
    empty_2 = []
    empty_3 = []
    empty_4 = []
    itertools_combinations = list(itertools.combinations(v, 2)) #finds every possible combination of list(df)
    for i in itertools_combinations:
        empty_1.append(i[0])
        empty_2.append(i[1])
        
    total_number_of_combinations = len(empty_1)
    
    for t in range(len(itertools_combinations)):
        empty.append(itertools_combinations[t])
        
    for value in range(len(empty_1)):
        empty_3.append(value)
        
    for x,y,i,j in zip(empty_1, empty_2, empty, empty_3):
        chisquare = chi_square(X=x, Y=y, Z=[], data=df, significance_level=0.05, boolean=True) #returns chi, p_value, dof
        empty_4.append([j, i , chisquare])
        
    return empty_4


# In[6]:


def chi2val(df):
#returns a tuple(chi2, p_value, dof) if boolean = false
#the null hypothesis is that they are independent of each other
#if true, the p_value is higher than the significance test, we do not reject the null hypothesis
#if false, the p_value is lower than the significance test, we reject the null hypothesis
    v = list(df)
    empty = []
    empty_1 = []
    empty_2 = []
    empty_3 = []
    empty_4 = []
    itertools_combinations = list(itertools.combinations(v, 2)) #finds every possible combination of list(df)
    for i in itertools_combinations:
        empty_1.append(i[0])
        empty_2.append(i[1])
        
    total_number_of_combinations = len(empty_1)
    
    for t in range(len(itertools_combinations)):
        empty.append(itertools_combinations[t])
        
    for value in range(len(empty_1)):
        empty_3.append(value)
        
    for x,y,i,j in zip(empty_1, empty_2, empty, empty_3):
        chisquare = chi_square(X=x, Y=y, Z=[], data=df, significance_level=0.05, boolean=False) #returns chi, p_value, dof
        empty_4.append([j, i , chisquare])
        
    return empty_4


# In[8]:


def g2(df):
#returns a tuple(chi2, p_value, dof) if boolean = false
#the null hypothesis is that they are independent of each other
#if true, the p_value is higher than the significance test, we do not reject the null hypothesis
#if false, the p_value is lower than the significance test, we reject the null hypothesis
    v = list(df)
    empty = []
    empty_1 = []
    empty_2 = []
    empty_3 = []
    empty_4 = []
    itertools_combinations = list(itertools.combinations(v, 2)) #finds every possible combination of list(df)
    for i in itertools_combinations:
        empty_1.append(i[0])
        empty_2.append(i[1])
        
    total_number_of_combinations = len(empty_1)
    
    for t in range(len(itertools_combinations)):
        empty.append(itertools_combinations[t])
        
    for value in range(len(empty_1)):
        empty_3.append(value)
        
    for x,y,i,j in zip(empty_1, empty_2, empty, empty_3):
        g2 = log_likelihood(X=x, Y=y, Z=[], significance_level=0.05, boolean=True, data=df) 
        empty_4.append([j, i , g2])
        
    return empty_4


# In[1]:


def sortReturn(data):
    #just a sorting function
    true_list = []
    false_list = []
    for i in data:
        if True in i:
            true_list.append(i)
        if False in i:
            false_list.append(i)
    return true_list, false_list


# In[188]:


def chi2cond(df):
    test_list_0 = []
    test_list_1 = []
    test_list_2 = []
    chi2 = []
    v = list(df)
    combine = list(itertools.combinations(v, 3))
    for i in combine:
        test_list_0.append(i[0])
        test_list_1.append(i[1])
        test_list_2.append(i[2])
    for a,b,c in zip(test_list_0, test_list_1, test_list_2):
        chisquare = chi_square(X=a, Y=b, Z=[c], data=df, significance_level=0.05, boolean=True)
        chi2.append([a,b,c,chisquare])
    return chi2

