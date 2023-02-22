    #!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
# from tabulate import tabulate
import numpy as np
# import bnlearn as bn
import pgmpy
import itertools
from scipy.stats import chi2_contingency #check for independence between 2 variables
from scipy.stats import power_divergence

from pgmpy.estimators.CITests import chi_square
# from pgmpy.factors.continuous.discretize import BaseDiscretizer


# In[3]:


def csv_format_discrete(csv_file):
    df = pd.read_csv(csv_file, sep="\s+")
    return df.drop([0])

#returns the csv_file in a pandas dataframe, formatted properly, discrete dataset only


# In[5]:


def csv_format_discrete(csv_file):
    df = pd.read_csv(csv_file, sep="\s+")
    return df.drop([0])


# In[6]:


def gaussian_reader(csv_file):
    df = pd.read_csv(csv_file, sep="\s+")
    return df


# In[118]:


def chi2bool(df):
#returns a tuple(chi, p_value, dof) if boolean = false
#the null hypothesis is that they are independent of each other
#if true, the p_value is higher than the significance test, we do not reject the null hypothesis
#if false, the p_value is lower than the significance test, we reject the null hypothesis
    v = list(df)
    empty = []
    empty_1 = []
    empty_2 = []
    empty_3 = []
    empty_4 = []
    empty_5 = []
    itertools_combinations = list(itertools.combinations(v, 2)) #finds every possible combination of list(df)
    for i in itertools_combinations:
        empty_1.append(i[0])
        empty_2.append(i[1])
    for t in range(len(itertools_combinations)):
        empty.append(itertools_combinations[t])
    for x,y,i in zip(empty_1, empty_2, empty):
        chisquare = chi_square(X=x, Y=y, Z=[], data=df, significance_level=0.05, boolean=True) #returns chi, p_value, dof
        empty_3.append([i , chisquare])
    for value in range(len(empty_3)):
        empty_4.append(value)
    for x,y in zip(empty_4, empty_3):
        empty_5.append([x,y])
    return empty_5

def chi2val(df):
    #returns a tuple(chi, p_value, dof) if boolean = false
#the null hypothesis is that they are independent of each other
#if true, the p_value is higher than the significance test, we do not reject the null hypothesis
#if false, the p_value is lower than the significance test, we reject the null hypothesis
    v = list(df)
    empty = []
    empty_1 = []
    empty_2 = []
    empty_3 = []
    empty_4 = []
    empty_5 = []
    itertools_combinations = list(itertools.combinations(v, 2)) #finds every possible combination of list(df)
    for i in itertools_combinations:
        empty_1.append(i[0])
        empty_2.append(i[1])
    for t in range(len(itertools_combinations)):
        empty.append(itertools_combinations[t])
    for x,y,i in zip(empty_1, empty_2, empty):
        chisquare = chi_square(X=x, Y=y, Z=[], data=df, significance_level=0.05, boolean=False) #returns chi, p_value, dof
        empty_3.append([i , chisquare])
    for value in range(len(empty_3)):
        empty_4.append(value)
    for x,y in zip(empty_4, empty_3):
        empty_5.append([x,y])
    return empty_5


