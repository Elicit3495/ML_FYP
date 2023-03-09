#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# from tabulate import tabulate
import numpy as np
from pygobnilp.gobnilp import Gobnilp
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


# In[46]:


df = csv_format_discrete(r"C:\Users\User\Documents\GitHub\ML_FYP\dataset\discrete.dat")
df
df_test = csv_format_discrete(r"C:\Users\User\Documents\GitHub\ML_FYP\dataset\alarm_10000.dat")
df_test


# ### chi-squared test

# In[4]:


#0th order chi2 test
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


# In[5]:


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


# In[6]:


#tests whether x is independent of y given a single variable z 
def chi2condbool(df):
    test_list_0 = []
    test_list_1 = []
    test_list_2 = []
    chi2 = []
    v = list(df)
    combine = list(itertools.combinations(v, 3)) #nC3 
    for i in combine:
        test_list_0.append(i[0])
        test_list_1.append(i[1])
        test_list_2.append(i[2])
    for a,b,c in zip(test_list_0, test_list_1, test_list_2):
        chisquare = chi_square(X=a, Y=b, Z=[c], data=df, significance_level=0.05, boolean=True)
        chi2.append([a,b,c,chisquare])
    return sortReturn(chi2)


# In[7]:


#tests whether x is independent of y given a single variable z 
def chi2condval(df):
    test_list_0 = []
    test_list_1 = []
    test_list_2 = []
    chi2 = []
    v = list(df)
    combine = list(itertools.combinations(v, 3)) #nC3 
    for i in combine:
        test_list_0.append(i[0])
        test_list_1.append(i[1])
        test_list_2.append(i[2])
    for a,b,c in zip(test_list_0, test_list_1, test_list_2):
        chisquare = chi_square(X=a, Y=b, Z=[c], data=df, significance_level=0.05, boolean=False)
        chi2.append([a,b,c,chisquare])
    return chi2


# ### log-likelihood tests

# In[8]:


def g2val(df):
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


# ### Sorting Functions

# In[9]:


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


# In[10]:


#returns a list with a tuple of combinations of false
def false2tuple(data):
    empty = []
    false_list_of_tuples = []
    new_df = pd.DataFrame(data)
    newer_df = new_df[1]
    return newer_df


# ### CONDITIONAL MAIN CODE

# In[11]:


#returns TRUE/FALSE Xs and Ys in a tuple of (X,Y)
def conditional_sort(df):
    output_false = []
    output_true = []
    my_list = [x for x in chi2bool(df) if False in x]
    my_list_2 = [x for x in chi2bool(df) if True in x]
    for i,j in zip(my_list, my_list_2):
        output_true.append(j[1])
        output_false.append(i[1])
    return output_true, output_false


# In[27]:


#returns permutations of x,y,z where x,y does not repeat e.g (a,b,c), (b,a,c)
def conditional_permute(df):
    a = conditional_sort(df)[1] #obtains the list of false outputs, false output means dependent
    k = list(itertools.permutations(df,3))
    permute_list = []
    for i in a:
        permute = list(itertools.permutations(i))
        permute_list.append(permute[1]) #returns a list of permuted items from a
    for items in permute_list:
        a.append(items) #adds all possible permutations to a 
    my_list = [x for x in k if all(x[:2] != y[:2] for y in a)] #checks if all elements in x[:2] != y[:2]
    my_list_2 = [x[:2] for x in my_list]
    a0 = list(tuple(sorted(l)) for l in my_list_2)
    output = [x for x in my_list if x[:2] in a0] #the three lines fixes all the permutations 
    return output


# In[57]:


def conditional_chi2_2(df):
    b0 = conditional_permute(df)
    b_100 = []
    for i,j,k in b0:
        chi2 = chi_square(X=i, Y=j, Z=[k], data=df_test, significance_level=0.05)
        b_100.append((i,j,k, chi2))
    b_100_true = [x for x in b_100 if True in x]
    b_100_false = [x for x in b_100 if False in x]
    return b_100_true, b_100_false


# In[58]:


omega = conditional_chi2_2(df_test)


# In[59]:


omega


# In[ ]:


#returns combinations of Zs
def conditional_combine_1(df):
    first_empty = []
    second_empty = []
    my_list = conditional_sort_2(df)
    for i in my_list:
        first_empty.append(i[0])
        second_empty.append(i[1])
    list_df = list(df)
    v = list(itertools.combinations(list_df, 2))
    for i,j,k in zip(first_empty, second_empty, v):
        chi_square(X=i, Y=j, Z=k, data=df)
    return chi_square


# In[ ]:


m = Gobnilp()


# In[ ]:


m.learn(r"C:\Users\User\Documents\GitHub\ML_FYP\dataset\discrete.dat")

