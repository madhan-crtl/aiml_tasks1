#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


data.info()


# In[4]:


print(type(data))
print(data.shape)
print(data.size)


# In[5]:


data1 = data.drop(['Unnamed: 0',"Temp C"], axis =1)
data1


# In[6]:


data1.info()


# In[7]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[8]:


data1[data1.duplicated(keep = False)]


# In[9]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[10]:


data1.rename({'Solar.R': 'Solar'}, axis=1, inplace = True)
data1


# In[11]:


data1.info()


# In[12]:


data1.isnull().sum()


# In[13]:


cols = data1.columns
colors = ['black', 'blue']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar = True)


# In[14]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ", mean_ozone)


# In[15]:


data['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[16]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[17]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[18]:


fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

sns.boxplot(data=data1["Ozone"], ax=axes[0], color='blue', width=0.5, orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone levels")
 
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='green', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone levels")
axes[1].set_ylabel("Frequency")

plt.tight_layout()

plt.show()


# In[19]:


sns.violinplot(data=data1["Ozone"], colors='lightgreen')


# In[20]:


sns.swarmplot(data=data1, x = "Weather", y="Ozone", palette="Set2", size=6)


# In[21]:


sns.stripplot(data=data1, x = "Weather", y = "Ozone",color="orange", palette="Set1", size=6, jitter=True)


# In[22]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="blue")
sns.rugplot(data=data1["Ozone"], color="black")


# In[23]:


sns.boxplot(data = data1, x = "Weather", y="Ozone")


# In[24]:


plt.scatter(data1["Wind"],data1["Temp"])


# In[25]:


data1["Wind"].corr(data["Temp"])


# In[26]:


data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[28]:


data1.info()


# In[29]:


data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[30]:


data1_numeric.corr()


# #### Observations
# - The higher correlation strength is observed between Ozone and Temperature(0.694404)
# - The next higher correlation strength is observed between Ozone and wind(-0.590270)
# - The next higher correlation strength is observed between wind and Temperature(-0.441228)
# - The least correlation is observed between solar and wind(-0.057407)

# In[33]:


sns.pairplot(data1_numeric)


# In[35]:


data2=pd.get_dummies(data1,columns=['Weather','Month'])
data2


# In[36]:


data1_numeric.values


# In[41]:


from numpy import set_printoptions
from sklearn.preprocessing import MinMaxScaler

array = data1_numeric.values

scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(array)

set_printoptions(precision=2)
print(rescaledX[0:10,:])


# In[ ]:




