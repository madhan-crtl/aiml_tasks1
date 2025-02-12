#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[3]:


#Read all numeric columns into Univ1
Univ1 = Univ.iloc[:,1:]


# In[4]:


Univ1


# In[5]:


cols = Univ1.columns
cols


# In[6]:


#Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df
#scaler.fit_tranform(Univ1)


# In[7]:


#Build 3 clusters using KMeans Cluster alogorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[8]:


##Print the clusters label
clusters_new.labels_


# In[9]:


set(clusters_new.labels_)


# In[10]:


#Assign clusters to the Univ data set
Univ['clusterid_new'] = clusters_new.labels_


# In[11]:


Univ


# In[12]:


Univ.sort_values(by = "clusterid_new")


# In[13]:


#Use groupby() to find aggregated (mean) values in each cluster
Univ.iloc[:,1:].groupby("clusterid_new").mean()


# Observations:
# -Cluster2 appears to be the top related universities cluster as the cut off score, Top 10,SFRatio parameter mean values are highest
# -cluster1 appears to occupy the middle level rated universities
# -cluster0 comes as the lower level rated universities

# Finding optimal k value using elbow plot

# In[14]:


wcss = []
for i in range(1, 20):
    
    kmeans = KMeans(n_clusters=i,random_state=0 )
    kmeans.fit(scaled_Univ_df)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20), wcss)
plt.title('Elbow Methos')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# Obseravtions:
# -From above graph we choose 3 or 4 which indicates the elbow join the rate of change of slope decreases

# In[18]:


# Quality of clusters is expressed in terms of sillhoutte score

from sklearn.metrics import silhouette_score
score = silhouette_score(scaled_Univ_df, clusters_new.labels_ , metric='euclidean')
score


# In[ ]:




