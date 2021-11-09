## Stats 507
-------------------
The primary purpose of this repo is as a solution for the Problem Set 6 of the course STATS 507. The repo will contain notebooks as well as script solutions to earlier problem sets.

It is also designed to be a public repository to facilitate grading.

### Notebooks

Currently the repo contains the python notebook named `PS2 Question 3.ipynb`.
The notebook is designed to read, clean and combine various data files from the NHANES "National Health and Nutrition Examination Survey" dataset. It appends datasets from the years : 2011-2012, 2013-2014, 2015-2016 and 2017-2018. 

**It specifically combines the Demographics and the Dentition Dataset**

1. It slices the dataset to include only the required columns which are: `id`, `age`, `gender`, `race` and `ethnicity`, `education`, `marital status` along with survey weighting columns from the demographic dataset. 
2. It slices the dentiton dataset to include : `dentition status`, `tooth counts` and `cavity counts`.

"Changes: Now, we added `gender` to the columns of the combined dataset.**  
  
This whole process is done to prepare the dataset for further analysis and applying visulations on the data. Another goal is to apply various models on the dataset as well.

The link for the above script is : [PS2 Q3](/Users/anuraagramesh/Documents/Python-Codes/Stats-507/Github-Stats507/Stats507/PS2-Question-3.ipynb)

The link for this file : [Stats507](/Users/anuraagramesh/Documents/Python-Codes/Stats-507/Github-Stats507/Stats507)
