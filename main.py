import pandas as pd 
import numpy as np 
import statsmodels.api as sm 

df = pd.read_csv("house_price_area_only.csv")
# print (df.head())

df['intercept'] = 1

ln = sm.OLS(df["price"], df[['intercept', 'area']])
results = ln.fit()
print (results.summary())