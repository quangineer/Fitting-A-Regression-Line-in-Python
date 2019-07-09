import pandas as pd 
import numpy as np 
import statsmodels.api as sm 
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

boston_data = load_boston() # Host load_boston
df = pd.DataFrame()   # Create an empty dataframe df
df["MedianHomePrice"] = boston_data.target
df2 = pd.DataFrame(boston_data.data)
df["CrimePerCapita"] = df2.iloc[:,0];
print(df.head())

df["intercept"]=1
ln = sm.OLS(df["MedianHomePrice"], df[["intercept", "CrimePerCapita"]])
results = ln.fit()
print(results.summary())
print(df.plot.scatter(x="CrimePerCapita",y="MedianHomePrice",c="DarkBlue"))
plt.show()
