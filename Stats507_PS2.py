# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import fastparquet
from collections import defaultdict
from timeit import Timer
import time
from IPython.core.display import HTML,display
#------------------------------------------------------------------------------

# **Anuraag Ramesh**  
# 1st October

# # Question 0 - Code Review Warmup

# ```sample_list = [(1, 3, 5), (0, 1, 2), (1, 9, 8)]  
# op = []  
# for m in range(len(sample_list)):  
#     li = [sample_list[m]]  
#         for n in range(len(sample_list)):  
#             if (sample_list[m][0] == sample_list[n][0] and  
#                     sample_list[m][3] != sample_list[n][3]):  
#                 li.append(sample_list[n])  
#         op.append(sorted(li, key=lambda dd: dd[3], reverse=True)[0])  
# res = list(set(op))```

# #### a.

# The above code snippet prints out the largest tuple with the same value at the first index.  
#

# #### b. Code Review

# - Please take care of the indentation in a python code, after for and while loops
#     - There is an indentation error in the code above 
#     - Line 5: is unnecessarily indented
#     
#     
# - Pay attention to the list ranges while indexing
#     - There is an error here as the sample_list is indexed to 3, which gives an out of range error  
#     
#     
# - The code is not semantically correct, as converting a set to the list does not lead to the same maintained order
#
#
# - Use indexes only when necessary
#     - Here we can use values directly instead of indices after the for loop
#     
#     
# - 'nitpicks' Use more meaningful names as variables

# # Question 1 - List of Tuples

def list_tuples(n, k=3, low=5, high=15):
    """
    Create a list of n- tuples with with size k
    
    Parameters
    ----------
    n : int
    Specifies the number of tuples
    k : int
    Specifies the size of each tuple
    low : int
    Specifies the lower range to pick the numbers 
    high : int
    Specifies the range range to pick the numbers 

    Returns
    -------
    tup_list: list
    Printing out a list of n-tuples
    """
    tup_list = []
    np.random.seed(1234)
    for i in range(n):
        tup = tuple([val 
                     for val in 
                     np.random.randint(low, high, k)])
        tup_list.append(tup)
    return tup_list


# #### Canvas Comment -  Q1: -2 for not checking it is a list
# This is being corrected in the cell below.

# +
for i in list_tuples(4):
    assert isinstance(i, tuple)

# Correction made
assert isinstance(list_tuples(4), list)
# -

# # Question 2 - Refactor the Snippet

sample_list = [(1, 3, 5), (0, 1, 2), (1, 9, 8)]


# ### a.

def func_snip(sample_list, a=0, b=2):
    """
    Defining a function for the above code snippet
    
    Parameters
    ----------
    sample_list : list
    Specifies the list of tuples
    a : int
    Parameterizes the lower index
    b : int
    Parameterizes the upper index 

    Returns
    -------
    res: list
    """
    if(b > len(sample_list[0]) - 1):
        b = len(sample_list[0]) - 1
    op = []
    for m in range(len(sample_list)):
        li = [sample_list[m]]
        for n in range(len(sample_list)):
            if (sample_list[m][a] == sample_list[n][a] 
                and sample_list[m][b] != sample_list[n][b]):
                li.append(sample_list[n])
        op.append(sorted(li, key = lambda dd: dd[b], reverse = True)[0])
    res = list(set(op))
    return res


print(func_snip(sample_list))


# ### b.

def func_imp(sample_list, a=0, b=2):
    """
    Defining a function for sorting based on 'a' value of tuple
    
    Parameters
    ----------
    sample_list : list
    Specifies the list of tuples
    a : int
    Parameterizes the lower index
    b : int
    Parameterizes the upper index 

    Returns
    -------
    res: list
    Outputs the list of highest tuple with same first value
    """
    if(b > len(sample_list[0]) - 1):
        b=len(sample_list[0]) - 1
    # Improvements based on code review
    op = []
    # Using values instead of indexes
    for m in sample_list:
        li = [m]
        for n in sample_list:
            if (m[a] == n[a] and m[b] != n[b]):
                li.append(n)
        op.append(sorted(li, key = lambda x: x[b], reverse = True)[0])
    # Using sorted to convert the set to a list
    res = sorted(set(op))
    return res


print(func_imp(sample_list))


# ### c.

def func_dict(sample_list, a=0, b=2):
    """
    Defining a function for the above code snippet
    
    Parameters
    ----------
    sample_list : list
    Specifies the list of tuples
    a : int
    Parameterizes the lower index
    b : int
    Parameterizes the upper index 

    Returns
    -------
    res: list
    Outputs the list highest tuple for unique first value of tuple
    """
    if(len(sample_list[0])-1):
        b=len(sample_list[0])-1
    op = defaultdict(list)
    # For loop to put values in the defaultdict
    for m in sample_list:
        op[m[a]].append(m)
    
    res = []
    for k, v in op.items():
        # Sorting the list for each key
        res.append(sorted(v, key = lambda x: x[b], reverse = True)[0])
    return sorted(res)


print(func_dict(sample_list, 0, 2))

# ### d. Monte Carlo Study

np.random.seed(1234)
def monte_carlo(min_var, max_var):
    """
    Give a random value using the uniform distribution
    
    Parameters
    ----------
    min_var : int
    Lower range of the variable
    max_var:
    Higher range of the variable
     
    Returns
    -------
    min_var+round(val) : float
    Gives the psuedo random of a variable(from the range)
    """
    range_var = max_var-min_var
    # Using a uniform distribution to produce a pseudo random value
    random_val = np.random.uniform(0, 1)
    val = range_var * random_val
    return(min_var + round(val))


monte_carlo(1,5)


# Doing a monte carlo study with each variable being randomly selected using a *Uniform Distribution* from a range of values.  
# The following variables are being randonly selected:
# - n : number of tuples
# - k : size of each tuple
# - low : lower value of each number in tuple
# - high: lower value of each number in tuple
# - a: The lower index to compare
# - b: The higher index to compare

def monte_study(func, var_list, n_samples=50):
    """
    Performing a monte carlo study for n_samples
    
    Parameters
    ----------
    func : function
    Specifies the function that needs to be tested
    list_var:
    A list of variable taht is part of simulation
    n_samples:
    Number of times the function is tested
     
    Returns
    -------
    final_comp_time : float
    Gives the average value of the comp_time for the function
    """
    sum_time = 0
    #Setting default values for the variables
    n = 6
    k = 3
    low = 0 
    high = 15
    a = 0
    b = k - 1
    for i in range(n_samples):
        # Simulating the number of tuples - range(2,30)
        if('n' in var_list):
            n = monte_carlo(2, 30)
        # Simulating the size of the tuples - range(3,20)
        if('k' in var_list):
            k = monte_carlo(3, 20)
        # Simulating the value of high - range(0,24)
        if('low' in var_list):
            low = monte_carlo(0, 24)
        # Simulating the value of low - range(10,30)
        if('high' in var_list):
            high = monte_carlo(10, 30)
        # Simulating the value of a - range(0,20)
        if('a' in var_list):
            a = monte_carlo(0, 20)
        # Simulating the value of b - range(2,20)
        if('b' in var_list):
            b = monte_carlo(2, 20)
        if(high > low and k > a and k >= b and b > a):
            # Now calculate the time
            li = list_tuples(n, k, low, high)
            start = time.time()
            func(li, a, b)
            end = time.time()
            comp_time = end - start
            sum_time = sum_time + comp_time
        else:
            continue
    final_comp_time = round(sum_time / n_samples, ndigits = 7)
    return(final_comp_time)


# #### Summary of the study

# Now the monte carlo study for the computation times for each of the functions (func_snip, func_imp, func_dict).  
#
# We can define a list and add it to the function to select the number of variables that are being randomized to generate an output

# +
# Initializing the variables
comp_list = []
var_lists = []
var_list = [['n'], ['n', 'k', 'low'], ['n', 'k', 'low', 'high']]
func_name = ['Snippet Func', 'Improve Func', 'Dictionary Func']*3
func = [func_snip, func_imp, func_dict]
functions = [func_snip, func_imp, func_dict]

for i in var_list:
    for j in func:
        comp_time = monte_study(j, i , 1000)
        comp_list.append(comp_time)
        var_lists.append(i)

monte_sim1 = {'sim variables':var_lists, 'function names':func_name,
              'computation time':comp_list}
        

# DataFrame
monte_sim = pd.DataFrame(monte_sim1)
# -

display(HTML(monte_sim.to_html()))

# **We can clearly see that in all the different versions of the monte carlo study that the `#3rd Version: defaultdict` is the one that takes the least amount of time to run.**

# # Question 3 - NHANES Data

# ### a. Appending Nhanes Demographic Data

# The four cohorts are:
# - **Baby Boomers**: Born 1946 - 1964
# - **Generation X**: Born 1965-1976
# - **Millenials or Generation Y**: Born 1977-1995
# - **Gen Z**: Born 1996 and after

# **Meaning of variables:**  
#
# **SEQN**: Respondent sequence number  
# **RIDAGEYR**: Age in years of the participant at the time of screening  
# **RIDRETH3**: Recode of reported race   
# **DMDEDUC2**: Education  
# **DMDMARTL**: Marital Status  
# **RIDSTATR**: Interview and examination status of the participant  
# **SDMVPSU**: Masked variance unit pseudo-PSU variable for variance estimation  
# **SDMVSTRA**: Masked variance unit pseudo-stratum variable for variance estimation  
# **WTMEC2YR**: Full sample 2 year MEC exam weight  
# **WTINT2YR**: Full sample 2 year interview weight

# +
# Defining the SAS files to combine 
nhanes = ['DEMO_G.XPT',
          'DEMO_H.XPT',
          'DEMO_I.XPT',
          'DEMO_J.XPT']

oral_dent = ['OHXDEN_G.XPT',
             'OHXDEN_H.XPT',
             'OHXDEN_I.XPT',
             'OHXDEN_J.XPT']


# -

def year_def(demo_dent):
    """
    Defining the year based on database
    
    Parameters
    ----------
    demo_dent : list
    Gives the locations of the demographic data SAS files

    Returns
    -------
    year: int
    Gives out the year of the dataset
    """
    
    if('G' in demo_dent):
        year = 2011
    elif('H' in demo_dent):
        year = 2013
    elif('I' in demo_dent):
        year = 2015
    elif('J' in demo_dent):
        year = 2017
    return year  


def cohort_def(age,data):
    """
    Defined to cohort based on age 
    
    Parameters
    ----------
    age : The RIDAGEYR variables
    The age variables for unique SEQN
    
    demo_dent: list
    Gives the locations of the demographic data SAS files

    Returns
    -------
    check: string
    Returns the cohort based on year
    """
    
    year=year_def(data)
    if(age <= (year - 1996)):
        return "Gen Z"
    elif(age <= (year - 1977) and age > (year - 1995)):
        return "Gen Y"
    elif(age <= (year - 1965) and age > (year - 1976)):
        return "Gen X"
    else:
        return "Baby Boomer"


# +
#Dictionaries fo the categorical variables for Demographic dataset
ridreth3 = {1 : 'mexican', 2 : 'other hispanic', 
            3 : 'non-hispanic White', 4 : 'non-hispanic black',
            6 : 'non-hispanic Asian', 7 : 'other race'}

ridstatr = {1 : 'interviewed only', 2: 'both'}

dmdecu2 ={1 : 'less than 9th', 2 : '9 to 11', 3:'hs graduate',
          4 : 'college', 5 : 'college graduate', 7 : 'refused',
          9 : 'dont know'}
 
dmdmartl = {1 : 'married', 2 : 'widowed', 3 : 'divorced',
            4 : 'separated', 5 : 'never married',
            6: 'living with partner', 77 : 'refused',
            99 : 'dont know'}
riagendr = {1 : 'male', 2: 'female'}


# -

# Combining NHANES Demographic data from different years
def initial_data(nhanes_link):
    """
    Cleaning and Combining NHANES Demographic DataFrame 
    
    Parameters
    ----------
    nhanes_link : list
    Specifies the location for the demographic dataset(different years)

    Returns
    -------
    nhanes_data: Pandas Dataframe
    Gives the combined Dataframe as output
    
    """
    
    nhanes_data = pd.DataFrame()
    for i in range(len(nhanes_link)):
        data = pd.read_sas(nhanes_link[i])
        data = data.loc[:,['SEQN', 'RIDAGEYR', 'RIAGENDR','RIDRETH3',
          'DMDEDUC2', 'DMDMARTL', 'RIDSTATR',
          'SDMVPSU', 'SDMVSTRA', 'WTMEC2YR', 'WTINT2YR']]
        #Adding the cohort column
        data['cohort'] = data['RIDAGEYR'].apply(cohort_def, data=nhanes_link[i])
        nhanes_data = nhanes_data.append(data)
    
    # Correcting the type of the different columns in the table
    for i in nhanes_data.columns:
        if(i == 'RIDAGEYR' or i == 'SDMVPSU' or i == 'SDMVSTRA'):
            nhanes_data[i] = nhanes_data[i].astype('int64')
        elif(i == 'RIDRETH3'):
            nhanes_data[i] = pd.Categorical(nhanes_data[i].replace(ridreth3))
        elif(i == 'RIAGENDR'):
            nhanes_data[i] = pd.Categorical(nhanes_data[i].replace(riagendr))
        elif(i == 'RIDSTATR'):
            nhanes_data[i] = pd.Categorical(nhanes_data[i].replace(ridstatr))
        elif(i == 'DMDEDUC2'):
            nhanes_data[i] = pd.Categorical(nhanes_data[i].replace(dmdecu2))
        elif(i == 'DMDMARTL'):
            nhanes_data[i] = pd.Categorical(nhanes_data[i].replace(dmdmartl))
        elif(i == 'cohort'):
            nhanes_data[i] = nhanes_data[i].astype('category')
        elif(i == 'WTMEC2YR' or i == 'WTINT2YR'):
            nhanes_data[i] = nhanes_data[i].astype('float64')
            
    nhanes_data = nhanes_data.rename(columns=
                                     {'RIDAGEYR':'age', 
                                      'RIDRETH3':'race',
                                      'RIAGENDR': 'gender',
                                      'DMDEDUC2':'education_level',
                                      'DMDMARTL':'marital', 
                                      'RIDSTATR':'interview_status',
                                      'SDMVPSU':'psuedo_PSU_var', 
                                      'SDMVSTRA':'psuedo_stratum_var', 
                                      'WTMEC2YR':'sample_weight_MEC', 
                                      'WTINT2YR':'sample_weight_int'})
        
    # Tidying and making the dataset clearer to read
    nhanes_data = nhanes_data.set_index('SEQN')
    return nhanes_data


nhanes_data = initial_data(nhanes)

print(nhanes_data.dtypes)

#Converting the nhanes_data to 'parquet' format
nhanes_data.to_parquet('Nhanes Data.parquet')

# ### b. Appending Oral and Dentition Data

# **Meaning of variables:**  
#
# **SEQN**: Respondent sequence number  
# **OHDDESTS**: Dentition Status  
# **Tooth counts (OHXxxTC)**: The Tooth Counts for each tooth (1-32)  
# **Coronal cavities (OHXxxCTC)**:The Cavity Counts for each tooth (1-31)

# Taking out the columns for slicing
oral_rows = []
data = pd.read_sas(oral_dent[0])
for i in data.columns[0:64]:
    if(i == 'OHXIMP' or i == 'OHDEXSTS'):
        pass
    else:
        oral_rows.append(i)

# +
#Dictionaries fo the categorical variables for Demographic dataset
tc = {1 : 'primary tooth',
      2 : 'permanent tooth',
      3 : 'dental implant',
      4 : 'not present',
      5 : 'root',
      9 : 'could not assess'}

# r = restorative
ctc = {'A' : 'restored primary tooth',
       'D' : 'sound primary tooth',
       'E' : 'missing - dental disease',
       'F' : 'restored permanent tooth',
       'J' : 'ppermanent root tip',
       'K' : 'primary tooth - surface condition',
       'M' : 'missing - other causes',
       'P' : 'disease - r',
       'Q' : 'other causes - r',
       'R' : 'other causes - fr',
       'S' : 'sound permanent tooth',
       'T' : 'permanent root - r',
       'U' : 'unerupted',
       'X' : 'other causes - f',
       'Y' : 'not assessed',
       'Z' : 'permanent tooth - surface condition'}

oddests={1 : 'complete',
         2 : 'partial',
         3 : 'not done'}


# -

# Combining Oral and Dentition data from different years
def initial_data_oral(oral_link):
    """
    Cleaning and Combining NHANES Oral and Dentition DataFrame 
    
    Parameters
    ----------
    oral_link : list
    Specifies the location for the nhanes dataset(Different years)

    Returns
    -------
    oral_data: Pandas Dataframe
    Gives the combined Dataframe as output
    
    """
    
    oral_data = pd.DataFrame()
    for i in range(len(oral_link)):
        data = pd.read_sas(oral_link[i])
        data = data.loc[:, oral_rows]
        oral_data = oral_data.append(data)
    # Tidying and making the dataset clearer to read
    # Correcting the type of the different columns in the table
    for i in oral_rows:
        if('CTC' in i):
            oral_data[i] = oral_data[i].str.decode('utf-8') 
            oral_data[i] = pd.Categorical(oral_data[i].replace(ctc))
            oral_data = oral_data.rename(columns={i:'coronial_' + i[3:5]})
        elif('TC' in i):
            oral_data[i] = pd.Categorical(oral_data[i].replace(tc))
            oral_data = oral_data.rename(columns={i:'tooth_count_' + i[3:5]})
        elif(i == 'OHDDESTS'):
            oral_data[i] = pd.Categorical(oral_data[i].replace(oddests))
            oral_data = oral_data.rename(columns={i:'dentition_status'})
    oral_data = oral_data.set_index('SEQN')
    return oral_data


oral_data = initial_data_oral(oral_dent)

print(oral_data.dtypes)

#Converting the data to 'parquet' format
oral_data.to_parquet('Oral Data.parquet')

# Both the datasets are saved in **parquet** format:  
# - Nhanes Data: 'Nhanes Data.gzip'
# - Oral Data: 'Oral Data.gzip'

# ### c. 

# Read NHANES data from parquet
nhanes_data1 = pd.read_parquet('Nhanes Data.parquet')
# Length of Nhanes dataset
print('Number of cases in NHANES data:', len(nhanes_data1))

# Read Oral Dentition data from parquet
oral_data1 = pd.read_parquet('Oral Data.parquet')
# Length of Oral Dentition dataset
print('Number of cases in NHANES data:', len(oral_data1))
