#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
# from tabulate import tabulate
import numpy as np
# import bnlearn as bn
import pgmpy
import itertools
from scipy.stats import chisquare


# In[2]:


from pgmpy.estimators.CITests import chi_square
# from pgmpy.factors.continuous.discretize import BaseDiscretizer


# In[3]:


def csv_format_discrete(csv_file):
    df = pd.read_csv(csv_file, sep="\s+")
    return df.drop([0])

#returns the csv_file in a pandas dataframe, formatted properly, discrete dataset only


# In[4]:


def gaussian_reader(csv_file):
    df = pd.read_csv(csv_file, sep="\s+")
    return df


# In[5]:


df = csv_format_discrete(r'C:\Users\User\Documents\GitHub\ML_FYP\dataset\asia_10000.dat')
dataframe = df
df


# In[37]:


def chi_square_CI(df):
#the null hypothesis is that they are independent of each other
#if true, accept null hypothesis
#if false, reject null hypothesis
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
        chisquare = chi_square(X=x, Y=y, Z=[], data=df, significance_level=0.95) #returns chi, p_value, dof
        empty_3.append([i , chisquare])
    for value in range(len(empty_3)):
        empty_4.append(value)
    for x,y in zip(empty_4, empty_3):
        empty_5.append([x,y])
    return empty_5
#if the chi_squared test gives true, then p_value is > significance level, hence we accept the null hypothesis, false otherwise
<<<<<<< HEAD

# In[ ]:


def chi_square_CI_test(df):
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
        chisquare = chi_square(X=x, Y=y, Z=[], data=df, significance_level=0.95, boolean=False) #returns chi, p_value, dof
        empty_3.append([i , chisquare])
    for value in range(len(empty_3)):
        empty_4.append(value)
    for x,y in zip(empty_4, empty_3):
        empty_5.append([x,y])
    return empty_5


# In[22]:


# df = csv_format_discrete(r'C:\Users\User\Documents\GitHub\ML_FYP\dataset\asia_10000.dat')


# In[ ]:





# In[24]:




# In[27]:


# k = power_divergence(v, lambda_="log-likelihood")
# #returns statistics, p-value


# # In[30]:


# k_ = list(k)
# k_


# In[31]:


30/66


# In[ ]:

=======


# In[38]:


>>>>>>> parent of 9e59f29 (a)
chi_square_CI(df)

