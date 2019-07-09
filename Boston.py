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
df.plot.scatter(x="CrimePerCapita",y="MedianHomePrice",c="DarkBlue",)
plt.title("Median Home Price vs CrimePerCapita");
plt.show()


# TO show the line that was fit:
import plotly.plotly as py 
import plotly.graph_objs as go 
from matplotlib import pylab 
from numpy import arange, array, ones 
from scipy import stats 

xi = arange(0,100)
A = array([xi, ones(100)])

y = df["MedianHomePrice"]
x = df["CrimePerCapita"]

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
line = slope*xi+intercept

plt.plot(x,y,'o', xi, line);
plt.xlabel("Crime/Capita");
plt.ylabel("Median Home Price");
pylab.title(" Price vs Crime");
plt.show()