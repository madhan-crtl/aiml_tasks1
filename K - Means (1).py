#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[3]:


Univ.info()


# In[4]:


Univ.describe()


# In[5]:


# Read all numeric columns into univ1
Univ1 = Univ.iloc[:,1:]


# In[6]:


Univ1


# In[13]:


cols


# In[14]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df


# In[16]:


# Build 3 clusters using Kmeans cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[18]:


#print the cluster labels
clusters_new.labels_


# In[19]:


set(clusters_new.labels_)


# In[20]:


Univ['clusterid_new'] = clusters_new.labels_
Univ


# In[21]:


Univ.sort_values(by = "clusterid_new")


# In[22]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# Observations
# 

# . cluster 2 appears to be the top rated universities cluster as the cut off score,Top10,SFRadio parameter mean values are high
# . cluster 1 appears to occupy the middle level rated universities      
# . cluster 0 comes as the lower level rated universities     

# In[25]:


wcss = [] 
for i in range(1,20):
    Kmeans = KMeans(n_clusters=i,random_state=0)
    Kmeans.fit(scaled_Univ_df)
    wcss.append(Kmeans.inertia_)
print(wcss)
plt.plot(range(1,20), wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# Observation :
#     . from the above graph we choose 3 or 4 which indicates the elbow join the rate of change of slope decreases
