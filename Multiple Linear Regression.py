#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot


# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# Dscription of columns     
# . MPG: Milege of the car(Mile per Gallaon) (This is Y-column to be predicted)    
# . HP: Horse Power of the car(X1 column)    
# . VOL: Volume of the car(size)(X2 column)    
# . SP:Top speed of the car(Miles per Hour)(X3 column)  
# .WT:Weight of the car(Pounds)(X4 column)

# In[ ]:




