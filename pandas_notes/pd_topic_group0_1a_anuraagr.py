# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: py:light,ipynb
#     notebook_metadata_filter: markdown
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

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#
# **Group 0**
#   

import pandas as pd
import numpy as np

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# + [Pivot tables](#Pivot-tables)
# + [One row to many](#One-row-to-many)
# + [DataFrame.pct_change()](#DataFrame.pct_change()) 
# + [Working with missing data](#Working-with-missing-data)
# + [Cumulative sums](#Title:-pandas.DataFrame.cumsum)
# + [Stack and unstack](#Stack-and-unstack)
# + [Pandas Query](#Pandas-Query) 
# + [Time Series](#Time-Series) 
# + [Window Functions](#Window-Functions) 
# + [Processing Time Data](#Processing-Time-Data)
# + [Pandas Time Series Analysis](#Title:-Pandas-Time-Series-Analysis)
# + [Pivot Table in pandas](#Pivot-Table-in-pandas)
# + [Multi-indexing](#Multi-indexing)
# + [Missing Data in Pandas](#Missing-Data-in-pandas)

# ## Pivot tables
# Zeyuan Li
# zeyuanli@umich.edu
# 10/19/2021
#
#

# ## Pivot tables in pandas
#
# The pivot tables in Excel is very powerful and convienent in handling with numeric data. Pandas also provides ```pivot_table()``` for pivoting with aggregation of numeric data. There are 5 main arguments of ```pivot_table()```:
# * ***data***: a DataFrame object
# * ***values***: a column or a list of columns to aggregate.
# * ***index***: Keys to group by on the pivot table index. 
# * ***columns***:  Keys to group by on the pivot table column. 
# * ***aggfunc***: function to use for aggregation, defaulting to ```numpy.mean```.

# ### Example

df = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 6,
        "B": ["A", "B", "C"] * 8,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 4,
        "D": np.random.randn(24),
        "E": np.random.randn(24),
        "F": [datetime.datetime(2013, i, 1) for i in range(1, 13)]
        + [datetime.datetime(2013, i, 15) for i in range(1, 13)],
    }
)
df


# ### Do aggregation
#
# * Get the pivot table easily. 
# * Produce the table as the same result of doing ```groupby(['A','B','C'])``` and compute the ```mean``` of D, with different values of D shown in seperate columns.
# * Change to another ***aggfunc*** to finish the aggregation as you want.

pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"])


pd.pivot_table(df, values="D", index=["B"], columns=["A", "C"], aggfunc=np.sum)


# ### Display all aggregation values
#
# * If the ***values*** column name is not given, the pivot table will include all of the data that can be aggregated in an additional level of hierarchy in the columns:

pd.pivot_table(df, index=["A", "B"], columns=["C"])


# ### Output
#
# * You can render a nice output of the table omitting the missing values by calling ```to_string```

table = pd.pivot_table(df, index=["A", "B"], columns=["C"])
print(table.to_string(na_rep=""))

# ## One row to many
#
# *Kunheng Li(kunhengl@umich.edu)*

# The reason I choose this function is because last homework. Before the hint from teachers, I found some ways to transfrom one row to many rows. Therefore, I will introduce a function to deal with this type of data.

# First, let's see an example.

data = {
    "first name":["kevin","betty","tony"],
    "last name":["li","jin","zhang"],
    "courses":["EECS484, STATS507","STATS507, STATS500","EECS402,EECS482,EECS491"]   
}
df = pd.DataFrame(data)
df = df.set_index(["first name", "last name"])["courses"].str.split(",", expand=True)\
    .stack().reset_index(drop=True, level=-1).reset_index().rename(columns={0: "courses"})
print(df)

# This is the first method I want to introduce, stack() or unstack(), both are similar. 
# Unstack() and stack() in DataFrame are to make itself to a Series which has secondary index.
# Unstack() is to transform its index to secondary index and its column to primary index, however, 
# stack() is to transform its index to primary index and its column to secondary index.

# However, in Pandas 0.25 version, there is a new method in DataFrame called explode(). They have the result, let's see the example.

df["courses"] = df["courses"].str.split(",")
df = df.explode("courses")
print(df)


# ## DataFrame.pct_change()
# *Dongming Yang*

# +
# This function always be used to calculate the percentage change between the current and a prior element, and always be used to a time series     
# The axis could choose the percentage change from row or columns
# Creating the time-series index 
ind = pd.date_range('01/01/2000', periods = 6, freq ='W') 
  
# Creating the dataframe  
df = pd.DataFrame({"A":[14, 4, 5, 4, 1, 55], 
                   "B":[5, 2, 54, 3, 2, 32],  
                   "C":[20, 20, 7, 21, 8, 5], 
                   "D":[14, 3, 6, 2, 6, 4]}, index = ind) 
  
# find the percentage change with the previous row 
df.pct_change()

# find the percentage change with precvious columns 
df.pct_change(axis=1)

# +
# periods means start to calculate the percentage change between the periods column or row and the beginning

# find the specific percentage change with first row
df.pct_change(periods=3)

# +
# fill_method means the way to handle NAs before computing percentage change by assigning a value to that NAs
# importing pandas as pd 
import pandas as pd 
  
# Creating the time-series index 
ind = pd.date_range('01/01/2000', periods = 6, freq ='W') 
  
# Creating the dataframe  
df = pd.DataFrame({"A":[14, 4, 5, 4, 1, 55], 
                   "B":[5, 2, None, 3, 2, 32],  
                   "C":[20, 20, 7, 21, 8, None], 
                   "D":[14, None, 6, 2, 6, 4]}, index = ind) 
  
# apply the pct_change() method 
# we use the forward fill method to 
# fill the missing values in the dataframe 
df.pct_change(fill_method ='ffill')

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# -

# ## Working with missing data
# *Kailan Xu*

# - Detecting missing data
# - Inserting missing data
# - Calculations with missing data
# - Cleaning / filling missing data
# - Dropping axis labels with missing data

# ### 1. Detecting missing data

# As data comes in many shapes and forms, pandas aims to be flexible with regard to handling missing data. While NaN is the default missing value marker for reasons of computational speed and convenience, we need to be able to easily detect this value with data of different types: floating point, integer, boolean, and general object. In many cases, however, the Python None will arise and we wish to also consider that “missing” or “not available” or “NA”.

# +
import pandas as pd 
import numpy as np 

df = pd.DataFrame(
    np.random.randn(5, 3),
    index=["a", "c", "e", "f", "h"],
    columns=["one", "two", "three"],
)
df2 = df.reindex(["a", "b", "c", "d", "e", "f", "g", "h"])
df2
# -

# To make detecting missing values easier (and across different array dtypes), pandas provides the `isna()` and `notna()` functions, which are also methods on Series and DataFrame objects:

df2.isna()

df2.notna()

# ###  2. Inserting missing data

# You can insert missing values by simply assigning to containers. The actual missing value used will be chosen based on the dtype.
# For example, numeric containers will always use NaN regardless of the missing value type chosen:

s = pd.Series([1, 2, 3])
s.loc[0] = None
s

# Likewise, datetime containers will always use NaT.
# For object containers, pandas will use the value given:

s = pd.Series(["a", "b", "c"])
s.loc[0] = None
s.loc[1] = np.nan
s

# ### 3. Calculations with missing data

# - When summing data, NA (missing) values will be treated as zero.
# - If the data are all NA, the result will be 0.
# - Cumulative methods like `cumsum()` and `cumprod()` ignore NA values by default, but preserve them in the resulting arrays. To override this behaviour and include NA values, use `skipna=False`.

df2

df2["one"].sum()

df2.mean(1)

df2.cumsum()

df2.cumsum(skipna=False)

# ### 4. Cleaning / filling missing data

# pandas objects are equipped with various data manipulation methods for dealing with missing data.
# - `fillna()` can “fill in” NA values with non-NA data in a couple of ways, which we illustrate:

df2.fillna(0)

df2["one"].fillna("missing")

# ### 5.Dropping axis labels with missing data

# You may wish to simply exclude labels from a data set which refer to missing data. To do this, use `dropna()`:

df2.dropna(axis=0)


# # Title: pandas.DataFrame.cumsum
# - Name: Yixuan Feng
# - Email: fengyx@umich.edu

# ## pandas.DataFrame.cumsum
# - Cumsum is the cumulative function of pandas, used to return the cumulative values of columns or rows.

# ## Example 1 - Without Setting Parameters
# - This function will automatically return the cumulative value of all columns.

values_1 = np.random.randint(10, size=10) 
values_2 = np.random.randint(10, size=10) 
group = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'] 
df = pd.DataFrame({'group':group, 'value_1':values_1, 'value_2':values_2}) 
df

df.cumsum()

# ## Example 2 - Setting Parameters
# - By setting the axis to 1, this function will return the cumulative value of all rows.
# - By combining with groupby() function, other columns (or rows) can be used as references for cumulative addition.

df['cumsum_2'] = df[['group', 'value_2']].groupby('group').cumsum() 
df

# [link](https://github.com/fyx1009/Stats507/blob/main/pandas_notes/pd_topic_fengyx.py)

# ## Stack and Unstack
# **Heather Johnston**
#
# **hajohns@umich.edu**
#
# *Stats 507, Pandas Topics, Fall 2021*
#
# ### About stack and unstack
# * Stack and Unstack are similar to "melt" and "pivot" methods for transforming data
# * R users may be familiar with "pivot_wider" and "pivot_longer" (formerly "spread" and "gather")
# * Stack transforms column names to new index and values to column
#
# ### Example: Stack
# * Consider the `example` DataFrame below to be measurements of some value taken on different days at different times.
# * It would be natural to want these to be "gathered" into long format, which we can do using `stack`

example = pd.DataFrame({"day":["Monday", "Wednesday", "Friday"],
                        "morning":[4, 5, 6],
                        "afternoon":[8, 9, 0]})
example.set_index("day", inplace=True)
print(example)
print(example.stack())

# ### Example: Unstack
# * Conversely, for displaying data, it's often handy to have it in a wider format
# * Unstack is especially convenient after using `groupby` on a dataframe

rng = np.random.default_rng(100)
long_data = pd.DataFrame({"group":["a", "a", "a", "a", "b", "b", "b", "b"],
                          "program":["x", "y", "x", "y", "x", "y", "x", "y"],
                         "score":rng.integers(0, 100, 8),
                         "value":rng.integers(0, 20, 8)
                         })
long_data.groupby(["group", "program"]).mean()
long_data.groupby(["group", "program"]).mean().unstack()


# ## Pandas Query ##
#
# ### pd. query ##
#
# ###### Name: Anandkumar Patel
# ###### Email: patelana@umich.edu
# ###### Unique ID: patelana
#
# ### Arguments and Output
#
# **Arguments** 
#
# * expression (expr) 
# * inplace (default = False) 
#     * Do you want to operate directly on the dataframe or create new one
# * kwargs (keyword arguments)
#
# **Returns** 
# * Dataframe from provided query
#
# ## Why
#
# * Similar to an SQL query 
# * Can help you filter data by querying
# * Returns a subset of the DataFrame
# * loc and iloc can be used to query either rows or columns
#
# ## Query Syntax
#
# * yourdataframe.query(expression, inplace = True/False
#
# ## Code Example

import pandas as pd 
import numpy as np


import pandas as pd
import numpy as np
### Q0 code example

# created from arrays or tuples

arrays = [["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
          ["one", "two", "one", "two", "one", "two", "one", "two"]]
tuples = list(zip(*arrays)) # if from arrays, this step is dropped
index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"]) 
# if from arrays, use pd.MultiIndex.from_arrays()

df1 = pd.Series(np.random.randn(8), index=index)

# created from product

iterables = [["bar", "baz", "foo", "qux"], ["one", "two"]]
df2 = pd.MultiIndex.from_product(iterables, names=["first", "second"])

#created directly from dataframe
df3 = pd.DataFrame([["bar", "one"], ["bar", "two"], ["foo", "one"], ["foo", "two"]],
                  columns=["first", "second"])
pd.MultiIndex.from_frame(df)

# Basic Operation and Reindex

df1 + df1[:2]
df1 + df1[::2]

df1.reindex(index[:3])
df1.reindex([("foo", "two"), ("bar", "one"), ("qux", "one"), ("baz", "one")])

#Advanced Indexing 
df1 = df1.T
df1.loc[("bar", "two")]


import pandas as pd
df = pd.DataFrame({'A': range(1, 6),
                   'B': range(10, 0, -2),
                   'C C': range(10, 5, -1)})
print(df)

print('Below is the results of the query')

print(df.query('A > B'))


# ## Time Series
# **Name: Lu Qin**
# UM email: qinlu@umich.edu
#
# ### Overview
#  - Data times
#  - Time Frequency
#  - Time zone
#
# ### Import

import datetime
import pandas as pd
import numpy as np


# ### Datetime
#  - Parsing time series information from various sources and formats

dti = pd.to_datetime(
    ["20/10/2021", 
     np.datetime64("2021-10-20"), 
     datetime.datetime(2021, 10, 20)]
)

dti


# ### Time frequency
# - Generate sequences of fixed-frequency dates and time spans
# - Resampling or converting a time series to a particular frequency

# #### Generate

dti = pd.date_range("2021-10-20", periods=2, freq="H")

dti


# #### convert

idx = pd.date_range("2021-10-20", periods=3, freq="H")
ts = pd.Series(range(len(idx)), index=idx)

ts


# #### resample

ts.resample("2H").mean()


# ### Timezone
#  - Manipulating and converting date times with timezone information
#  - `tz_localize()`
#  - `tz_convert()`

dti = dti.tz_localize("UTC")
dti

dti.tz_convert("US/Pacific")


# ## Window Functions ##
# **Name: Stephen Toner** \
# UM email: srtoner@umich.edu

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web


# Of the many funcitons in Pandas, one which is particularly useful for time
# series analysis is the window function. It lets us apply some aggregation 
# function over a specified lookback period on a rolling basis throughout the
# time series. This is particularly useful for financial analsyis of equity
# returns, so we will compute some financial metrics for Amazon stock using
# this techinique.

# Our first step is to import our data for Amazon ("AMZN") 
# over a healthy time horizon:

amzn_data = web.DataReader("AMZN", 
                           data_source = 'yahoo', 
                           start = "2016-10-01", 
                           end = "2021-10-01")

amzn_data.head()


# While the column labels are largely self-explanatory, two important notes
# should be made:
# * The adjusted close represents the closing price after all is said and done
# after the trading session ends; this may represent changes due to accounts 
# being settled / netted against each other, or from adjustments to financial
# reporting statements.
# * One reason for our choice in AMZN stock rather than others is that AMZN
# has not had a stock split in the last 20 years; for this reason we do not
# need to concern ourselves with adjusting for the issuance of new shares like
# we would for TSLA, AAPL, or other companies with large
# market capitalization.

# Getting back to Pandas, we have three main functions that allow us to
# perform Window operations:
# * `df.shift()`: Not technically a window operation, but helpful for
# computing calculations with offsets in time series
# * `rolling`: For a given fixed lookback period, tells us the 
# aggregation metric (mean, avg, std dev)
# * `expanding`: Similar to `rolling`, but the lookback period is not fixed. 
# Helpful when we want to have a variable lookback period such as "month to 
# date" returns

# Two metrics that are often of interest to investors are the returns of an
# asset and the volume of shares traded. Returns are either calculated on
# a simple basis:
# $$ R_s = P_1/P_0 -1$$
# or a log basis:
# $$ R_l = \log (P_1 / P_2) $$
# Simple returns are more useful when aggregating returns across multiple 
# assets, while Log returns are more flexible when looking at returns across 
# time. As we are just looking at AMZN, we will calculate the log returns
# using the `shift` function:

amzn_data["l_returns"] = np.log(amzn_data["Adj Close"]/
                                amzn_data["Adj Close"].shift(1))


plt.title("Log Returns of AMZN")
plt.plot(amzn_data['l_returns'])


# For the latter, we see that the
# volume of AMZN stock traded is quite noisy:

plt.title("Daily Trading Volume of AMZN")   
plt.plot(amzn_data['Volume'])


# If we want to get a better picture of the trends, we can always take a
# moving average of the last 5 days (last full set of trading days):

amzn_data["vol_5dma"] = amzn_data["Volume"].rolling(window = 5).mean()
plt.title("Daily Trading Volume of AMZN")   
plt.plot(amzn_data['vol_5dma'])


# When we apply this to a price metric, we can identify some technical patterns
# such as when the 15 or 50 day moving average crosses the 100 or 200 day
# moving average (known as the golden cross, by those who believe in it).

amzn_data["ma_15"] = amzn_data["Adj Close"].rolling(window = 15).mean()
amzn_data["ma_100"] = amzn_data["Adj Close"].rolling(window = 100).mean()

fig1 = plt.figure()
plt.plot(amzn_data["ma_15"])
plt.plot(amzn_data["ma_100"])
plt.title("15 Day MA vs. 100 Day MA")

# We can then use the `shift()` method to identify which dates have 
# golden crosses

gc_days = (amzn_data.eval("ma_15 > ma_100") & 
               amzn_data.shift(1).eval("ma_15 <= ma_100"))

gc_prices = amzn_data["ma_15"][gc_days]


fig2 = plt.figure()
plt.plot(amzn_data["Adj Close"], color = "black")
plt.scatter( x= gc_prices.index, 
                y = gc_prices[:],
                marker = "+", 
                color = "gold" 
                )

plt.title("Golden Crosses & Adj Close")


# The last feature that Pandas offers is a the `expanding` window function, 
# which calculates a metric over a time frame that grows with each additional 
# period. This is particularly useful for backtesting financial metrics
# as indicators of changes in equity prices: because one must be careful not
# to apply information from the future when performing backtesting, the 
# `expanding` functionality helps ensure we only use information up until the 
# given point in time. Below, we use the expanding function to plot cumulative
# return of AMZN over the time horizon.

def calc_total_return(x):
    """    
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.log(x[-1] / x[0]) 


amzn_data["Total Returns"] = (amzn_data["Adj Close"]
                              .expanding()
                              .apply(calc_total_return))

fig3 = plt.figure()
ax5 = fig3.add_subplot(111)
ax5 = plt.plot(amzn_data["Total Returns"])
plt.title("Cumulative Log Returns for AMZN")


# * ###  Processing Time Data
#
# **Yurui Chang**
#
# #### Pandas.to_timedelta
#
# - To convert a recognized timedelta format / value into a Timedelta type
# - the unit of the arg
#   * 'W'
#   * 'D'/'days'/'day'
#   * ‘hours’ / ‘hour’ / ‘hr’ / ‘h’
#   * ‘m’ / ‘minute’ / ‘min’ / ‘minutes’ / ‘T’
#   * ‘S’ / ‘seconds’ / ‘sec’ / ‘second’
#   * ‘ms’ / ‘milliseconds’ / ‘millisecond’ / ‘milli’ / ‘millis’ / ‘L’
#   * ‘us’ / ‘microseconds’ / ‘microsecond’ / ‘micro’ / ‘micros’ / ‘U’
#   * ‘ns’ / ‘nanoseconds’ / ‘nano’ / ‘nanos’ / ‘nanosecond’ / ‘N’
#
# * Parsing a single string to a Timedelta
# * Parsing a list or array of strings
# * Converting numbers by specifying the unit keyword argument

time1 = pd.to_timedelta('1 days 06:05:01.00003')
time2 = pd.to_timedelta('15.5s')
print([time1, time2])
pd.to_timedelta(['1 days 06:05:01.00003', '15.5s', 'nan'])

pd.to_timedelta(np.arange(5), unit='d')


# #### pandas.to_datetime
#
# * To convert argument to datetime
# * Returns: datetime, return type dependending on input
#   * list-like: DatetimeIndex
#   * Series: Series of datetime64 dtype
#   * scalar: Timestamp
# * Assembling a datetime from multiple columns of a DataFrame
# * Converting Pandas Series to datetime w/ custom format
# * Converting Unix integer (days) to datetime
# * Convert integer (seconds) to datetime

s = pd.Series(['date is 01199002',
           'date is 02199015',
           'date is 03199020',
           'date is 09199204'])
pd.to_datetime(s, format="date is %m%Y%d")

time1 = pd.to_datetime(14554, unit='D', origin='unix')
print(time1)
time2 = pd.to_datetime(1600355888, unit='s', origin='unix')
print(time2)


# # Title: Pandas Time Series Analysis
# ## Name: Kenan Alkiek (kalkiek)


from matplotlib import pyplot as plt

# Read in the air quality dataset
air_quality = pd.read_csv(
    'https://raw.githubusercontent.com/pandas-dev/pandas/master/doc/data/air_quality_no2_long.csv')
air_quality["datetime"] = pd.to_datetime(air_quality["date.utc"])

# One common method of dealing with time series data is to set the index equal to the data
air_quality = air_quality.set_index('datetime')
air_quality.head()

# Plot the NO2 Over time for Paris france
paris_air_quality = air_quality[(air_quality['city'] == 'Paris') & (air_quality['country'] == 'FR')]

paris_air_quality.plot()
plt.ylabel("$NO_2 (µg/m^3)$")

# Plot average NO2 by hour of the day
fig, axs = plt.subplots(figsize=(12, 4))
air_quality.groupby("date.utc")["value"].mean().plot(kind='bar', rot=0, ax=axs)
plt.xlabel("Hour of the day")
plt.ylabel("$NO_2 (µg/m^3)$")
plt.show()

# Limit the data between 2 dates
beg_of_june = paris_air_quality["2019-06-01":"2019-06-03"]
beg_of_june.plot()
plt.ylabel("$NO_2 (µg/m^3)$")

# Resample the Data With a Different Frequency (and Aggregration)
monthly_max = air_quality.resample("M").max()
print(monthly_max)

# Ignore weekends and certain times
rng = pd.date_range('20190501 09:00', '20190701 16:00', freq='30T')

# Grab only certain times
rng = rng.take(rng.indexer_between_time('09:30', '16:00'))

# Remove weekends
rng = rng[rng.weekday < 5]

rng.to_series()


# ## Pivot Table in pandas
#
#
# *Mingjia Chen* 
# mingjia@umich.edu
#
# - A pivot table is a table format that allows data to be dynamically arranged and summarized in categories.
# - Pivot tables are flexible, allowing you to customize your analytical calculations and making it easy for users to understand the data.
# - Use the following example to illustrate how a pivot table works.


import numpy as np

df = pd.DataFrame({"A": [1, 2, 3, 4, 5],
                   "B": [0, 1, 0, 1, 0],
                   "C": [1, 2, 2, 3, 3],
                   "D": [2, 4, 5, 5, 6],
                   "E": [2, 2, 4, 4, 6]})
print(df)


# ## Index
#
# - The simplest pivot table must have a data frame and an index.
# - In addition, you can also have multiple indexes.
# - Try to swap the order of the two indexes, the data results are the same.

tab1 = pd.pivot_table(df,index=["A"])
tab2 = pd.pivot_table(df,index=["A", "B"])
tab3 = pd.pivot_table(df,index=["B", "A"])
print(tab1)
print(tab2)
print(tab3)


# ## Values 
# - Change the values parameter can filter the data for the desired calculation.


pd.pivot_table(df,index=["B", "A"], values=["C", "D"])


# ## Aggfunc
#
# - The aggfunc parameter sets the function that we perform when aggregating data.
# - When we do not set aggfunc, it defaults aggfunc='mean' to calculate the mean value.
#   - When we also want to get the sum of the data under indexes:


pd.pivot_table(df,index=["B", "A"], values=["C", "D"], aggfunc=[np.sum,np.mean])


# ## Columns
#
# - columns like index can set the column hierarchy field, it is not a required parameter, as an optional way to split the data.
#
# - fill_value fills empty values, margins=True for aggregation


pd.pivot_table(df,index=["B"],columns=["E"], values=["C", "D"],
               aggfunc=[np.sum], fill_value=0, margins=1)


#
# Ziyi Gao
#
# ziyigao@umich.edu
#
# ## Multi-indexing
#
# - Aiming at sophisticated data analysis and manipulation, especially for working with higher dimensional data
# - Enabling one to store and manipulate data with an arbitrary number of dimensions in lower dimensional data structures
#
# ## Creating a multi-indexing dataframe and Reconstructing
#
# - It can be created from:
#     - a list of arrays (using MultiIndex.from_arrays())
#     - an array of tuples (using MultiIndex.from_tuples())
#     - a crossed set of iterables (using MultiIndex.from_product())
#     - a DataFrame (using MultiIndex.from_frame())
# - The method get_level_values() will return a vector of the labels for each location at a particular level
#
# ## Basic Indexing
#
# - Advantages of hierarchical indexing
#     - hierarchical indexing can select data by a “partial” label identifying a subgroup in the data
# - Defined Levels
#     - keeps all the defined levels of an index, even if they are not actually used
#     
# ## Data Alignment and Using Reindex
#
# - Operations between differently-indexed objects having MultiIndex on the axes will work as you expect; data alignment will work the same as an Index of tuples
# - The reindex() method of Series/DataFrames can be called with another MultiIndex, or even a list or array of tuples:
#
# ## Some Advanced Indexing
#
# Syntactically integrating MultiIndex in advanced indexing with .loc is a bit challenging
#
# - In general, MultiIndex keys take the form of tuples

# ## Missing Data in pandas
#
# #### Anuraag Ramesh: anuraagr@umich.edu
#
# - About
# - Calculations with missing data
# - Filling missing values
# - Interpolation
# - Replacing generic values

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from IPython.core.display import HTML,display
import random
import scipy.stats as sci
#------------------------------------------------------------------------------

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
