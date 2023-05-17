import streamlit as st
from fredapi import Fred
from wallstreet import Stock

import pandas as pd
import numpy as np

from sklearn.linear_model import HuberRegressor
from scipy.stats import gaussian_kde

import plotly.express as px

s = Stock('GC=F')
fred = Fred(api_key='147c83da741e258846d44af26794a872')

cpi = fred.get_series('CPIAUCSL')
treasury = fred.get_series('DGS10')
gold = s.historical(days_back=2500)
gold['Date']=pd.to_datetime(gold['Date'])
gold = gold.set_index('Date')[['Open']]

df= gold.join(cpi.rename('CPI')).join(treasury.rename('Treasury')).dropna()
len(df)

reg =HuberRegressor()
reg.fit(df.drop(columns=['Open']),df['Open'])
a,b,c=reg.coef_[0],reg.coef_[1], reg.intercept_
df['Predict']=df.eval('@a*CPI+@b*Treasury+@c')

current_cpi = cpi.values[-1]
current_treasury = treasury.values[-1]
current_gold = gold.Open.values[-1]

k = gaussian_kde(df.eval('Open-Predict'))
fig=px.line(x=a*current_cpi+b*current_treasury+c+np.linspace(-350,350,101),y=k(np.linspace(-350,350,101)),
            labels={'x':'Gold Price','y':'Probability Density'}, title=f"Gold Price Prediction as of {df.index[-1].strftime('%D')}")
st.plotly_chart(fig)