import pandas as pd 
import numpy as np 
import statsmodels.api as sm 
from sklearn.datasets import load_boston
import matplotlib as plt

boston_data = load_boston() # Host load_boston
df = pd.DataFrame()   # Create an empty dataframe df
df["MedianHomePrice"] = boston_data.target
df2 = pd.DataFrame(boston_data.data)
df["CrimePerCapita"] = df2.iloc[:,0];
df.head()
