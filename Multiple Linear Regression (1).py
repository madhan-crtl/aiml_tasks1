#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# In[2]:


# Read the data from csv file
cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


##Rearrange the columns
cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# Description of columns
# .MPG:Milege of the car (Mile per Gallon) (this is Y-column to be predicted)
# .HP:Horse Power of the car(X1 column)
# .Vol:Volume of the car(size)
# .SP:Top speed of the car(Mile per Hour) (X3 cloumn)
# .WT:Weight of the car(Pounds)(X4 column)

# ##Assumptions in Multilinear Regression
# 1.Linearity:The relation between the predictors(X) and the response (Y) is linear
# 2.Independence:Observations are independent of each other
# 3.Homoscedasticity:The residuals(Y-Y_hat) exhibit constant variance at all levels of the predictor.
# 4.Normal Distribution of Errors:The residuals of the model are normally distributed.
# 5.No multicollinearity:The independent variables should not be too highly correlated with each other.
# Violations of these assumptions may lead to inefficiency in the regression 

# EDA

# In[4]:


cars.info()


# In[5]:


#check for missing values
cars.isna().sum()


# Observations:
# .There are no missing values
# .There are 81 Observations (81 different cars data)
# .The data types of the columns are also relevant and valid

# In[6]:


#Create a figure with two subplots(one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (1.5, .85)})

#Create a boxplot
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

#Adjust layout
plt.tight_layout()
plt.show()                                                                   
                                                                   


# In[7]:


#Create a figure with two subplots(one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (1.5, .85)})

#Create a boxplot
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

#Adjust layout
plt.tight_layout()
plt.show()                                                                   
                                        


# In[8]:


#Create a figure with two subplots(one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (1.5, .85)})

#Create a boxplot
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')

#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

#Adjust layout
plt.tight_layout()
plt.show()                                                                   
                                        


# In[9]:


#Create a figure with two subplots(one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (1.5, .85)})

#Create a boxplot
sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')

#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

#Adjust layout
plt.tight_layout()
plt.show()                                                                   
                                        


# In[10]:


#Create a figure with two subplots(one above the other)
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (1.5, .85)})

#Create a boxplot
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')

#Creating a histogram in the same x-axis
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

#Adjust layout
plt.tight_layout()
plt.show()                                                                   
                                        


# Observations:
# -There are some extreme values (outliers) observed in towards the right tail of SP and HP distributions.
# -In VOL and WT columns, a few outliers are observed in both tails of distributions.
# -The extreme values of cars data may have come from specially designed nature of cars
# -As this is multi-dimensional data, theoutliers with respect to spatial dimensions may have to be considered while building the regression model.

# Checking for duplicated rows

# In[11]:


cars[cars.duplicated()]


# #Pair plots and COrrelation Coefficients
# 

# In[12]:


#pair plots
sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[13]:


cars.corr()


# Observation from correlation plots and Coefficients             
# . Between x and y, all the variables are showing moderate to high correlation strengths,highest being b/w HP and MPG   
# . Therefore this dataset qualifies for building a multiple linear regression model to predict MPG  
# . Among x columns (x1,x2,x3 and x4),some very high correlation strengths are observed b/w SP vs HP,VOL vs WT    
# . The high correlation among x columns is not desirable as it might lead to multicollinearity problem   

# Preparing a preliminary model considering all x columns

# In[15]:


# Build model
#import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[16]:


model1.summary()


# Observations from model summary             
# . The R-squared and adjusted R-squared values are good and about 75% of variability in Y is explained by X columns              
# . The probability value with respect to F-statistic is close to zero, indicating that all or someof X columns are significant                    
# . The p-values for VOL and WT are higher tha 5% indicating some interaction issue among themselves, which need to be further explored

# #Performance metrics for model1

# In[19]:


# Find the performance metrics
# create a data frame with actual y and predicted y columns

df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[20]:


# predict for the given x data columns

pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[18]:


cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# In[ ]:




