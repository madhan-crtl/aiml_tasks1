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

# In[14]:


# Build model
#import statsmodels.formula.api as smf
model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[15]:


model1.summary()


# Observations from model summary             
# . The R-squared and adjusted R-squared values are good and about 75% of variability in Y is explained by X columns              
# . The probability value with respect to F-statistic is close to zero, indicating that all or someof X columns are significant                    
# . The p-values for VOL and WT are higher tha 5% indicating some interaction issue among themselves, which need to be further explored

# #Performance metrics for model1

# In[16]:


# Find the performance metrics
# create a data frame with actual y and predicted y columns

df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[17]:


# predict for the given x data columns

pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[18]:


# compute the Mean squared Error (MSE) , RMSE for model1
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"],df1["pred_y1"])
print("MSE :", mse )
print("RMSE :",np.sqrt(mse))


# Checking for multicollinearity among X-columns using VIF method

# In[19]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# Observations for VIF values:
#  . The ideal range of VIF values shall be b/w 0 to 10. However slightly higher values can be tolerated
#   . As seen from the very high VIF values for VOL and WT, it is clear that they are prone to multicollinearity problem.
#   . Hence it is decided to drop one of the columns to overcome the multicollinearity.
#   . It is decided to drop WT and retain VOL column in further models

# In[20]:


cars1 = cars.drop("WT", axis=1)
cars1.head()


# In[21]:


model2 = smf.ols("MPG~HP+VOL+SP",data=cars1).fit()
model2.summary()


# Performance metrics for model2

# In[22]:


# Find the performance metrics
df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[23]:


# predict for the given X data columns
pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[24]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse)
print("RMSE :",np.sqrt(mse))


# Observations from model2 summary()

# . The adjusted R-squared value improved slightly to 0.76       
# . All the p-values for model parameters are less than 5% hence they are  significant         
# . Therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPG response variable            
# . There is no improvement iin MSE value

# Identification of High Influence Points

# #### Leverage (Hat Values):
# Leverage values diagnose if a data point has an extreme value in terms of the independent variables. A point with high leverage has a great ability to influence the regression line. The threshold for considering a point as having high leverage is typically set at 3(k+1)/n, where k is the number of predictors and n is the sample size.

# In[25]:


#Define variables and assign values
k = 3  # no of x=columns in cars1
n = 81 # no of observation (rows)
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff


# In[26]:


from statsmodels.graphics.regressionplots import influence_plot

influence_plot(model1,alpha=.05)

y=[i for i in range(-2,8)]
x=[leverage_cutoff for i in range(10)]
plt.plot(x,y,'r+')

plt.show()


# Observations

# . From the above plot, it is evident that data points 65,70,76,78,79,80 are the influencers.                
# . as their H leverage values are higher and size is higher     

# In[27]:


cars1[cars1.index.isin([65,70,76,78,79,80])]


# In[28]:


#Discard the data points which are influencers and reasign the row number (rest_index)
cars2=cars1.drop(cars1.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)


# In[29]:


cars2


# Build Model3 on cars2 datase

# In[33]:


# Rebuild the model model
model3= smf.ols('MPG~VOL+SP+HP',data = cars2).fit()


# In[34]:


model3.summary()


# Performance Metrics for model3

# In[35]:


df3 = pd.DataFrame()
df3["actual_y3"] = cars2["MPG"]
df3.head()


# In[40]:


pred_y3 = model2.predict(cars2.iloc[:,0:4])
df2["pred_y3"] = pred_y3
df2.head()


# #### Comparison of models
#                      
# 
# | Metric         | Model 1 | Model 2 | Model 3 |
# |----------------|---------|---------|---------|
# | R-squared      | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 
# - **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**

# In[ ]:




