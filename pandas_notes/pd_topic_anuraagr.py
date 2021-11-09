# # Anuraag Ramesh : anuraagr@umich.edu

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from IPython.core.display import HTML,display
import random
import scipy.stats as sci
#------------------------------------------------------------------------------

# # Question 0 - Topics in Pandas

# ## Missing Data in pandas
#
# - About
# - Calculations with missing data
# - Filling missing values
# - Interpolation
# - Replacing generic values

# ## About
#
# Missing values are always present in datasets that are developed from the 
# real world , and it is important to understand the functions and 
# methods that are present to deal with them properly.

df = pd.DataFrame({'Name' : ['A' , 'B', 'C', 'D', 'E'],
                   'Score 1' :[90, 85, 86, 67, 45],
                   'Score 2' :[None , 78, 89, 56, 99], 
                   'Score 3' :[80, None , None, 56, 82],
                   'Score 4' : [68, 79, None , 26, 57]})

df

# ### Defining missing values

# In the dataset defined above, we can see that 
# there are few "NaN" of missing values.  
# - The missing or not avialable value is defined using `np.nan`.
# - We can find the missing values in a dataset using `isna()`. 
# The values that show 'True' are missing in the dataset
# - On the other hand, to find if a value is not null we use `notna()`

print(df.isna())
print('\n')
print(df.notna())

# - We can also use `np.nan()` as a 
# parameter to compare various values  
# - Using `isna()` to find the missing values in each column

print(df['Score 1'].isna())
print(df['Score 2'].isna())

# ## Calculations with missing data

# There is missing values in our dataset. But 
# there are several different ways we can 
# handle this to perform calculations.
#
# Suppose, we want to calculate 
# the average of scores for each person. 
# We can use these three methods.
# - Skip the missing values
# - Drop the column with missing values
# - Fill in the missing values with some other value
#
# Note : "NA'" values are automatically excluded while using groupby

# Skipping missing values
print(df.mean(skipna = True, axis = 1))

# +
# Dropping columns or rows with missing values

print(df.dropna(axis = 0)) #Row
print("\n")
print(df.dropna(axis = 1)) #Column
# -

# ## Filling missing values
#
# We can fill the missing values using different methods:
#
# - Filling missing values with 0
# - Filling missing values with a string - eg. NA
# - Filling missing with values with values 
# appearing before or after
# - Filling values with mean of a column

# Filling values with 0
df.fillna(0)
# Filling values with a string
df.fillna("NA")


# Filling values with values appearing after the
# missing values
df.fillna(method = "pad")

# Filling values with mean of individual columns
print(df.fillna(df.mean()))

# ## Interpolation
#
# This is the process of performing linear interpolation 
# to give an expectation assumption of missing values.
#
# There are several different methods of interpolation
#
# - linear : default method
# - quadratic
# - pchip
# - akima
# - spline
# - polynomial

df.interpolate()

df.interpolate(method = "akima")

# Below, we can see that the missing values in 
# `Score 3` is replaced by 55 and 45 respectively

df.interpolate(method = "quadratic")

# ## Replacing generic values
#
# We can simply replace the NaN values from the outside,
# by using `.replace()`  
#
# Here, we can assume and replace the value with a random
# value with 75.

df.replace(np.nan, 75)