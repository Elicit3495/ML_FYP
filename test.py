#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from pygobnilp.gobnilp import Gobnilp
import pgmpy
import itertools
from pgmpy.estimators.CITests import pearsonr
from pgmpy.estimators.CITests import chi_square
import networkx as nx
import time
import collections


# In[2]:


def csv_format_discrete(csv_file):
    df = pd.read_csv(csv_file, sep="\s+")
    return df.drop([0])

#returns the csv_file in a pandas dataframe, formatted properly, discrete dataset only


# In[3]:


#small network : n <= 20 nodes, medium network: 20 <= n <= 50 , large: 50 <= n <= 100, ... 
df_small = csv_format_discrete(r"C:\Users\User\Documents\GitHub\ML_FYP\dataset\asia_10000.dat")
df_medium = csv_format_discrete(r"C:\Users\User\Documents\GitHub\ML_FYP\dataset\alarm_10000.dat")
# df_large = csv_format_discrete(r"")


# ### chi-squared test

# In[4]:


df_small


# In[5]:


#0th order chi2 test
def chi2bool(df, rho):
    '''
    0th order CI test
    
    Parameters:
    df -> pandas dataframe
    rho -> significance level, only accept 0 <= rho <= 1.0
    '''
    chi2 = []
    v = list(df)
    my_list = list(itertools.combinations(v,2))
    y_0, y_1 = [x[0] for x in my_list], [x[1] for x in my_list]
    for i,j in zip(y_0, y_1):
        chi = chi_square(X=i, Y=j, Z=[], data=df, significance_level=rho)
        chi2.append((i,j,chi))
    true_0 = [x for x in chi2 if True in x]
    false_0 = [x for x in chi2 if False in x]
    return true_0, false_0


# ### Pearson's Product Moment Correlation Coefficient

# In[6]:


df_gaus = pd.read_csv(r"C:\Users\User\Documents\GitHub\ML_FYP\dataset\gaussian.dat", sep="\s+")
df_gaus


# In[7]:


#0th order chi2 test
def PMCC(df, rho):
    '''
    0th order CI test
    
    Parameters:
    df -> pandas dataframe
    rho -> significance level, only accept 0 <= rho <= 1.0
    '''
    R_list = []
    v = list(df)
    my_list = list(itertools.combinations(v,2))
    y_0, y_1 = [x[0] for x in my_list], [x[1] for x in my_list]
    for i,j in zip(y_0, y_1):
        R = pearsonr(X=i, Y=j, Z=[], data=df, significance_level=rho)
        R_list.append((i,j,R))
    true_0 = [x for x in R_list if True in x]
    false_0 = [x for x in R_list if False in x]
    return true_0, false_0


# ### Sorting Functions

# In[8]:


def create_permutations(my_list):
    '''
    takes in a list, remove extra permutations from the list and only creates a pair of permutations
    '''
    p_permutations = []
    new_list = list(set(tuple(sorted(l[:2])) for l in my_list))
    for items in new_list:
        new_list_0 = list(itertools.permutations(items[:2],2))[1]
        p_permutations.append(new_list_0)
    return new_list + p_permutations


# In[9]:


#given 2 lists, append them together and remove duplicates
def remove_permutations(list_):
    '''
    returns a sorted list without permutation
    '''
    return sorted(list(set(tuple(sorted(l[:2])) for l in list_)))


# ### 1st order CI

# In[10]:


#1st order CI
def cond_1_generate(df, rho):
    v = list(df)
    p_permutations = list(itertools.permutations(v,3))
    order_0 = create_permutations(chi2bool(df, rho)[0])
    generated = [x for x in p_permutations if x[:2] not in order_0]
    generated_0 = set(tuple(sorted(items[:2])) for items in generated)
    generate_return = [x for x in generated if x[:2] in generated_0]
    return generate_return


# In[11]:


def cond_1_test(df, rho):
    chi2_data = []
    phi = cond_1_generate(df, rho)
    for i,j,k in phi:
        chi2 = chi_square(X=i, Y=j, Z=[k], data=df, significance_level=rho)
        chi2_data.append((i,j,k,chi2))
    true_list = [x for x in chi2_data if True in x]
    false_list = [x for x in chi2_data if False in x]
    return true_list, false_list


# ### CONDITIONAL MAIN CODE

# In[12]:


def PC_(df, n, rho):
    '''
    Note:
    This function will start from at least 0th order CI
    
    Parameters:
    df (pandas dataframe)
    n an integer, the stopping point of the while loop
    rho (the significance level, only accepts values between 0 and 1 inclusive)
    
    Returns:
    A list which contains every independent X and Y
    '''
    N = 3
    v = list(df)
    # remove_list = [x for x in p_permute]
    remove_list = create_permutations([x[:2] for x in chi2bool(df, rho)[0]])
    #x[:2] for 0th order and its permutations, so we can later remove it 
    empty_list = []
    while N <= n:
        list_permutations = [x for x in itertools.permutations(v, N) if x[:2] not in remove_list]
        p_1, p_2, p_3 = [x[:1] for x in list_permutations], [x[1:2] for x in list_permutations], [x[2:] for x in list_permutations]
        for i,j,k in zip(p_1, p_2, p_3):
            chi2 = chi_square(X=i[0], Y=j[0], Z=k, data=df, significance_level=rho)
            empty_list.append((i[0],j[0],k,chi2))
        true_list = create_permutations([x[:2] for x in empty_list if True in x])
        #create_permutations ensures only 1 set of permutations of (X,Y) and (Y,X) and removes dupes
        remove_list = remove_list + create_permutations(true_list)
        remove_list = create_permutations(remove_list)
        N += 1
        
    return remove_permutations(remove_list) #returns 1 set of permutations

#ON MEDIUM SIZED NETWORKS:
#1st order CI takes 2minutes to run
#2nd order CI 9-15minutes to run
#nP5 takes ??? minutes to run
#nP6 not doable


# In[77]:


G = nx.Graph()
v = list(df_small)
for i in v:
    G.add_node(i)
print(G)


# In[78]:


k = list(itertools.combinations(v,2))
for i,j in k:
    G.add_edge(i,j)


# In[79]:


print(G)
nx.draw(G, with_labels = True)


# In[82]:


ind_0 = [x[:2] for x in chi2bool(df_small, 0.05)[0]]
# ind_0


# In[83]:


for i,j in ind_0:
    G.remove_edge(i,j)


# In[84]:


print(G)
nx.draw(G, with_labels = True)


# In[86]:


false_list = [x[:2] for x in chi2bool(df_small, 0.05)[1]]


# In[92]:


test_list = []
for i,j in false_list:
    test_list.append((i,j,tuple(G.edges(i,j))))


# In[94]:


test_list[0]


# In[ ]:





# In[ ]:





# ### small network

# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', "n = 0\nN = 1\nchi_list = []\nm = Gobnilp()\nwhile n < N:\n    empty_list = []\n    m.learn(r'C:\\Users\\User\\Documents\\GitHub\\ML_FYP\\dataset\\alarm_10000.dat')\n    for i,j in m.adjacency.items():\n        if j.X == 1.0: #j.X == 1.0 implies there is an edge between the nodes\n            empty_list.append(i)\n    #chi2 test\n    empty_list = [list(x) for x in empty_list]\n    phi_0, phi_1 = [x[0] for x in empty_list], [x[1] for x in empty_list]\n    for i,j in zip(phi_0, phi_1):\n        chi2 = chi_square(X=i, Y=j, Z=[], data=df_medium, significance_level=0.05)\n        chi_list.append((i,j,chi2))\n    true_list = [x[:2] for x in chi_list if True in x]\n    for i,j in true_list:\n        m.add_obligatory_independence([i],[j])\n    n += 1\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "n = 0\nN = 1\nchi_list = []\nm = Gobnilp()\nwhile n < N:\n    empty_list = []\n    m.learn(r'C:\\Users\\User\\Documents\\GitHub\\ML_FYP\\dataset\\asia_10000.dat')\n    for i,j in m.adjacency.items():\n        if j.X == 1.0: #j.X == 1.0 implies there is an edge between the nodes\n            empty_list.append(i)\n    #chi2 test\n    empty_list = [list(x) for x in empty_list]\n    phi_0, phi_1 = [x[0] for x in empty_list], [x[1] for x in empty_list]\n    for i,j in zip(phi_0, phi_1):\n        chi2 = chi_square(X=i, Y=j, Z=[], data=df_small, significance_level=0.05)\n        chi_list.append((i,j,chi2))\n    true_list = [x[:2] for x in chi_list if True in x]\n    for i,j in true_list:\n        m.add_obligatory_independence([i],[j])\n    n += 1\n")


# In[ ]:


new_list = [x for x in chi2bool(df_small, 0.4)]
new_list


# In[ ]:


get_ipython().run_cell_magic('time', '', "chi_list = [x[:2] for x in chi2bool(df_small, 0.4)[0]]\nm_small = Gobnilp()\nfor i,j in chi_list:\n    m_small.add_forbidden_adjacency((i,j))\nm_small.learn(r'C:\\Users\\User\\Documents\\GitHub\\ML_FYP\\dataset\\asia_10000.dat')\n")


# In[ ]:


m_small.adjacency


# In[89]:


s_small = Gobnilp()
s_small.learn(r'C:\Users\User\Documents\GitHub\ML_FYP\dataset\asia_10000.dat')


# In[ ]:





# In[ ]:




