#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[3]:


data1.info()


# In[4]:


data1.describe()


# In[5]:


sns.boxplot(data=data1['daily'],color='orange',width=0.5,orient = 'h')


# In[6]:


sns.histplot(data=data1['daily'],kde=True,color='blue',bins=30)


# In[7]:


plt.scatter(data1["daily"],data1["sunday"])


# In[8]:


data1["daily"].corr(data1["sunday"])


# Observation   
# . The correlation b/w daily and sunday is observed to be highly positive with 0.9581543140785462

# In[9]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["daily"], vert = False)


# In[10]:


import seaborn as sns
sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# In[11]:


import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()


# In[12]:


model.summary()


# In[13]:


x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 =1.33
# predicted response vector
y_hat = b0 + b1*x
 
# plotting the regression line
plt.plot(x, y_hat, color = "g")
  
# putting labels
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[14]:


# Plot the linear regression line using seaborn regplot() method
sns.regplot(x="daily", y="sunday", data=data1)
plt.xlim([0,1250])
plt.show()


# In[15]:


# import numpy as np
# x = np.arange(10)
# plt.plot(2 + 3 *x)
# plt.show()


# In[16]:


#Coefficients
model.params


# In[17]:


#t and p-Values
print(model.tvalues, '\n', model.pvalues)    
# print(f'model t-values:\n{model.tvalues}\n-----------------\nmodel p-values: \n{model.pvalues}')    


# In[ ]:




