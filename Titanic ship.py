#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[3]:


# from google.colab import files
#uploaded = files.upload()

titanic = pd.read_csv("Titanic.csv")
titanic


# In[4]:


titanic.info()


# In[5]:


titanic.describe()


# Observations

# . From these all the columns are categorical in nature             
# . There are no null values               
# . As the columns are categorical, we can adopt one-hot-encoding

# In[8]:


titanic['Class'].value_counts()


# In[7]:


#Plot a bar chart to visualize the category of people on the ship
counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# Observations

# .Maximun no of people travelling in crew

# In[9]:


#Plot a bar chart to visualize the category of people on the ship
counts = titanic['Age'].value_counts()
plt.bar(counts.index, counts.values)


# In[10]:


#Plot a bar chart to visualize the category of people on the ship
counts = titanic['Survived'].value_counts()
plt.bar(counts.index, counts.values)


# In[11]:


#Plot a bar chart to visualize the category of people on the ship
counts = titanic['Gender'].value_counts()
plt.bar(counts.index, counts.values)


# In[13]:


#perform onehot encoding on categorical columns
df=pd.get_dummies(titanic,dtype=int)
df.head()


# In[14]:


df.info()


# # Apriori Algorithm

# In[16]:


# Apply apriori algorithm to get itemset combinations
frequent_itemsets = apriori(df, min_support = 0.05,use_colnames=True, max_len=None)
frequent_itemsets


# In[17]:


frequent_itemsets.info()


# In[19]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules


# In[ ]:




