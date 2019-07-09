import pandas as pd 
import numpy as np 
import statsmodels.api as sm 
import matplotlib as plt 

df = pd.read_csv("carats.csv", header=None) # To create numerical header for each column
# print (df)
# print (df.columns)  # 0 1
df.columns = ["carats", "sellingprice"]

df["intercept"]=1
ln = sm.OLS(df["sellingprice"], df[["intercept", "carats"]])
results = ln.fit()
print(results.summary())
ax = df.plot.scatter(x="carats",y="price",c="DarkBlue")