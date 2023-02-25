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
from pgmpy.estimators.CITests import independence_match
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


# In[88]:


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


# In[ ]:


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


# In[2]:


def ind2bool(df):
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
        ind_match = independence_match(X=x, Y=y, Z=[], independencies = xy , data=df) #returns chi, p_value, dof
        empty_4.append([j, i , ind_match])
        
    return empty_4


# In[100]:


def chi2return(df):
    #just a sorting function
    true_list = []
    false_list = []
    chi2 = chi2bool(df)
    for i in chi2:
        if True in i:
            true_list.append(i)
        if False in i:
            false_list.append(i)
    return false_list

