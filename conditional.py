#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
from tabulate import tabulate
import numpy as np
import bnlearn as bn
import pgmpy
import itertools
from scipy.stats import chisquare


# In[2]:


from pgmpy.estimators.CITests import chi_square
from pgmpy.factors.continuous.discretize import BaseDiscretizer


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


df = csv_format_discrete('alarm_10000.dat')
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
        chisquare = chi_square(X=x, Y=y, Z=[], data=df, significance_level=0.95, boolean=False) #returns chi, p_value, dof
        empty_3.append([i , chisquare])
    for value in range(len(empty_3)):
        empty_4.append(value)
    for x,y in zip(empty_4, empty_3):
        empty_5.append([x,y])
    return empty_5
#if the chi_squared test gives true, then p_value is > significance level, hence we accept the null hypothesis, false otherwise


# In[38]:


chi_square_CI(df)


# In[40]:


np.random.seed(10)
# Sample data randomly at fixed probabilities
type_bottle = np.random.choice(a= ["paper","cans","glass","others","plastic"],
                              p = [0.05, 0.15 ,0.25, 0.05, 0.5],
                              size=1000)
 
# Sample data randomly at fixed probabilities
month = np.random.choice(a= ["January","February","March"],
                              p = [0.4, 0.2, 0.4],
                              size=1000)
 
bottles = pd.DataFrame({"types":type_bottle, 
                       "months":month})
 
bottles_tab = pd.crosstab(bottles.types, bottles.months, margins = True)
 
bottles_tab.columns = ["January","February","March","row_totals"]
 
bottles_tab.index = ["paper","cans","glass","others","plastic","col_totals"]
 
observed = bottles_tab.iloc[0:5,0:3]   # Get table without totals for later use
bottles_tab


# In[43]:


chi_square_CI(bottles_tab.drop(columns='row_totals'))


# In[7]:


df = bn.import_example(data='asia')


# In[8]:


new_df = csv_format_discrete('asia_10000.dat')
new_df


# In[9]:


chi_square_CI(df)


# In[10]:


df


# In[14]:


df_raw = bn.import_example(data='titanic')


# In[16]:


df_raw


# In[17]:


dfhot, dfnum = bn.df2onehot(df_raw)


# In[19]:


dfnum


# In[23]:


chi_square_CI(dfhot)


# In[21]:


DAG = bn.structure_learning.fit(dfnum)
bn.plot(DAG)


# In[ ]:


new_list = pd.DataFrame({'right handed':[239, 157],
                   'left handed': [19, 17]})



# In[ ]:


chi_square_CI(new_list)


# In[ ]:


new_list_df.columns


# In[ ]:


model = bn.structure_learning.fit(new_list_df, methodtype='hc', scoretype='bic')
G = bn.plot(model)

# Compute edge strength using chi-square independence test
model1 = bn.independence_test(model, new_list_df, alpha=0.05, prune=False)
# bn.plot(model1, pos=G['pos'])


# In[ ]:


chi_square_CI(new_list_df)


# In[ ]:


#score based search
def hill_climbing(dataframe):
    model = bn.structure_learning.fit(dataframe, methodtype='hc', scoretype='k2')
#     G = bn.plot(model)
    # Compute edge strength using chi-square independence test
    model1 = bn.independence_test(model, dataframe, alpha=0.05, prune=False)
#     bn.plot(model1, pos=G['pos'])
    tabulated_model = (tabulate(model1['independence_test'], headers="keys"))
    return tabulated_model


# In[ ]:


model = bn.structure_learning.fit(dataframe, methodtype='hc', scoretype='bdeu')
bn.plot(model)
    # Compute edge strength using chi-square independence test
model1 = bn.independence_test(model, dataframe, test='chi_square', prune=True, alpha=0.001)
# chi_square, g_sq, log_likelihood, freeman_turkey, modified_log_likelihood, neyman, cressie_read
bn.plot(model1)

print(tabulate(model1['independence_test'], headers="keys"))


# In[ ]:


k0 = hill_climbing(df)


# In[ ]:


#https://erdogant.github.io/bnlearn/pages/html/Structure%20learning.html#chow-liu
#local discovery algorithm
def chow_liu_search(dataframe):
    dfhot, dfnum = bn.df2onehot(dataframe)
    model = bn.structure_learning.fit(dfnum, methodtype='cl', bw_list_method='nodes')
    G = bn.plot(model)
    model1 = bn.independence_test(model, dfnum, alpha=0.05, prune=True)
    bn.plot(model1, pos=G['pos'])
    print(tabulate(model1['independence_test'], headers="keys"))


# In[ ]:


k1 = chow_liu_search(df)


# In[35]:


#bayesian network classifiers

#naivebayes, tree augmented naive bayes


# In[58]:


model = bn.structure_learning.fit(dataframe, methodtype='hc', scoretype='k2')
G = bn.plot(model)
# Compute edge strength using chi-square independence test
model1 = bn.independence_test(model, dataframe, alpha=0.05, prune=True)
bn.plot(model1, pos=G['pos'])
v1 = (tabulate(model1['independence_test'], headers="keys"))


# In[60]:


dfhot, dfnum = bn.df2onehot(dataframe) #converts dataframe into a one-hot matrix
model_ = bn.structure_learning.fit(dfnum, methodtype='cl') #chowliu algorithm
G = bn.plot(model)
model_1 = bn.independence_test(model, dfnum, test='chi_square', alpha=0.05, prune=False)
bn.plot(model1, pos=G['pos'])
print(tabulate(model1['independence_test'], headers="keys"))


# In[61]:


bn.compare_networks(model, model_, pos=G['pos'])


# In[62]:


bn.compare_networks(model, model_)

