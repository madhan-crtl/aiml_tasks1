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


# EDA

# In[3]:


data1.info()


# In[4]:


data1.isnull().sum()


# In[5]:


data1.describe()


# In[6]:


# Boxplot for daily column

plt.figure(figsize=(6,3))
plt.title("Box plot for Daily sales")
plt.boxplot(data1["daily"],vert = False)
plt.show()


# In[7]:


sns.histplot(data1['daily'], kde = True,stat='density',)
plt.show()


# In[8]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["sunday"], vert = False)


# In[9]:


sns.histplot(data1['sunday'], kde = True,stat='density',)
plt.show()


# Observations
# . There are no missing values
# . The daily column values appears to be right-skewed
# . The sunday column values also appear to be right-skewed
# . There are two outliers in both daily column and also in sunday column as obseerved from the 

# In[10]:


x= data1["daily"]
y = data1["sunday"]
plt.scatter(data1["daily"], data1["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[11]:


data1["daily"].corr(data1["sunday"])


# In[12]:


data1[["daily","sunday"]].corr()


# In[13]:


data1.corr(numeric_only=True)


# Observations on Correlation strength   
# 
# . The relationship b/w x(daily) and y(sunday) is seen to be linear as seen from scatter plot  
# . The correlation is strong and positive with Pearson's correlation coefficient of 0.958154

# In[16]:


# Build regression model

import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()


# In[17]:


model.summary()


# In[20]:


#Plot the scatter plot and overlay the fitted straight line using matplotlib
x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x, y, color = "m", marker = "o", s = 30)
b0 = 13.84
b1 = 1.33
#predicted response vector
y_hat = b0 + b1*x

#plotting the regression line
plt.plot(x,y_hat, color = "g")

#putting labels
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[22]:


model.params


# In[23]:


print(f'model t-values:\n{model.tvalues}\n---------------\nmodel p-values: \n{model.pvalues}')


# In[25]:


#Print the quantity of fitted line (r squared values)
(model.rsquared,model.rsquared_adj)


# In[26]:


#Predict for 200 and daily circilation
newdata=pd.Series([200,300,1500])


# In[27]:


data_pred=pd.DataFrame(newdata,columns=['daily'])
data_pred


# In[28]:


model.predict(data_pred)


# In[30]:


#Predicton all given training data

pred = model.predict(data1["daily"])
pred


# In[31]:


##Add predicted values as  a column in data1
data1["Y_hat"] = pred
data1


# In[32]:


#Compute the error values (residuals) and add as another column
data1["residuals"] = data1["sunday"]-data1["Y_hat"]
data1


# In[33]:


model.predict(data_pred)


# In[35]:


#Predicton all given training data

pred = model.predict(data1["daily"])
pred


# In[36]:


#Compute Mean Squared Error for the model
mse = np.mean((data1["daily"]-data1["Y_hat"])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# In[40]:


# Compute Mean Absolute Error (MAE)
mae = np.mean(np.abs(data1["daily"]-data1["Y_hat"]))
mae


# In[41]:


mape = np.mean((np.abs(data1["daily"]-data1["Y_hat"])/data1["daily"]))*100
mape


# ### Assumptions in Linear simple linear regresion
# 
# 1. **Linearity:** The relationship between the predictors and the response is linear.
# 
# 2. **Independence:** Observations are independent of each other.
# 
# 3. **Homoscedasticity:** The residuals (Y - Y_hat) exhibit constant variance at all levels of th

# In[37]:


plt.scatter(data1["Y_hat"], data1["residuals"])


# In[39]:


# Plot the Q-Q plot (to check the normality of residuals)
import statsmodels.api as sm
sm.qqplot(data1["residuals"], line='45', fit=True)
plt.show()


# In[38]:


sns.histplot(data1["residuals"], kde = True)


# Observations:            
#  . The data points are seen to closely follow the reference line of normality                    
#  . Hence the residuals are approximately normally distributed as also can be seen from the kde distribution
#  . All the assumptions of the model are satisfied and the model is performed well

# In[ ]:




