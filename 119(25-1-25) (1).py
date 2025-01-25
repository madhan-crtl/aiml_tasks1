#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[4]:


data.info()


# In[5]:


print(type(data))
print(data.shape)
print(data.size)


# In[6]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[7]:


data1.info()


# In[8]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[9]:


data1[data1.duplicated(keep = False)]


# In[10]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[11]:


data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# In[13]:


data1.info()


# In[18]:


data1.isnull().sum()


# In[17]:


cols = data1.columns
colors = ['black', 'blue']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[19]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[20]:


data['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[ ]:




