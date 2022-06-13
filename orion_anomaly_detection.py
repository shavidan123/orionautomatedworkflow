#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from datetime import datetime
import csv
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from pandas.io.json import json_normalize
import json
import plotly.express as plt
from orion import Orion

ticker = "TSLA"
url = f"https://sandbox.iexapis.com/stable/stock/{ticker}/chart/5y/?token=Tpk_059b97af715d417d9f49f50b51b1c448"
data = requests.get(url).json()


st.header("Data Fetching / Aggregation")

st.markdown("These first few cells are just collecting the data by conducting the get requests using IEX cloud API. I elected to use stock data because there are many metrics and it is updated consistently nearly every day.")

# In[2]:


df = pd.DataFrame(data)
df.head()
st.dataframe(df)


# In[3]:


df = df[["close","date"]] #for this dataset I used closing price since 2017 to keep it simple. usually I would use EWMA or a more effective indicator
df["datetime"] = df["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
df["datetime"] = pd.to_timedelta(df["datetime"])
df["datetime"] = df["datetime"].dt.total_seconds()
df.head()


st.header("Building the Pipeline")
st.markdown("I'm not entirely sure, but I think Orion requires the data to be formatted in a very specific way, with the two columns of the dataframe being labelled 'timestamp' and 'value.' I set up the data as such, then build a pipeline that doesn't vary much from the default ARIMA pipeline that's preprogrammed into the Orion library.")

# In[4]:


df = pd.DataFrame({"timestamp":df["datetime"], "value": df["close"]})


# In[5]:

arima_pipeline = {
    "primitives": [
        "mlprimitives.custom.timeseries_preprocessing.time_segments_aggregate",
        "sklearn.impute.SimpleImputer",
        "sklearn.preprocessing.MinMaxScaler",
        "mlprimitives.custom.timeseries_preprocessing.rolling_window_sequences",
        "numpy.reshape",
        "statsmodels.tsa.arima_model.Arima",
        "orion.primitives.timeseries_errors.regression_errors",
        "orion.primitives.timeseries_anomalies.find_anomalies"
    ],
    "init_params": {
        "mlprimitives.custom.timeseries_preprocessing.time_segments_aggregate#1": {
            "time_column": "timestamp",
            "interval": 21600,
            "method": "mean"
        },
        "sklearn.preprocessing.MinMaxScaler#1": {
            "feature_range": [
                -1,
                1
            ]
        },
        "mlprimitives.custom.timeseries_preprocessing.rolling_window_sequences#1": {
            "target_column": 0,
            "window_size": 250
        },
        "numpy.reshape#1": {
            "newshape": [
                -1,
                250
            ]
        },
        "statsmodels.tsa.arima_model.Arima#1": {
            "steps": 1
        },
        "orion.primitives.timeseries_anomalies.find_anomalies#1": {
            "fixed_threshold": "true"
        }
    },
    "input_names": {
        "orion.primitives.timeseries_anomalies.find_anomalies#1": {
            "index": "target_index"
        },
        "numpy.reshape#1": {
            "y": "X"
        }
    },
    "output_names": {
        "statsmodels.tsa.arima_model.Arima#1": {
            "y": "y_hat"
        },
        "numpy.reshape#1": {
            "y": "X"
        }
    }
}
hp = {
    'mlprimitives.custom.timeseries_preprocessing.time_segments_aggregate#1': {
        'interval': 43200 #changing time interval drastically affects anomaly detection. currently using half-day intervals.
    },
    "statsmodels.tsa.arima_model.Arima#1": {
            "steps": 1
    }
}
arima = Orion(pipeline=arima_pipeline, hyperparameters = hp)


st.header("Detecting Anomalies")
st.markdown("Once the Orion pipeline is constructed, I fit the model to the data and predict anomalous segments. The data is output into a datafrae representing the start and end of the anomalous sequence. Using a for loop, I split the sequence back into its original discrete data points that make it up.")

# In[6]:


anomalies = arima.fit_detect(df)


# In[7]:


anomalies.head()
st.dataframe(anomalies)

# In[8]:


anomalydata = pd.DataFrame(columns = ["timestamp", "value"])
for i in range(len(anomalies.index)): #works for any number of anomalous segments
    start = anomalies.iloc[i, 0]
    end = anomalies.iloc[i, 1]
    anomalydata = anomalydata.append(df[df["timestamp"].between(start, end)])

anomalydata.head()
st.dataframe(anomalydata)

st.header("Graphing the Data")
st.markdown("Finally, I graph the data. I still need to familiarize myself with plotly to create more effective graphs, but this basic graph seems to work for now. Discrete anomalous points are highlighted with red dots, while the timeseries data is graphed over time with the blue line.") 

# In[14]:


fig = go.Figure()

# add the main values to the figure
fig.add_trace(go.Scatter(x = df['timestamp'], y = df['value'],   #blue line is stock price over time (specifically this data set)
                             mode = 'lines',
                             marker =dict(color='blue'),
                             name = 'original_signal'))
fig.add_trace(go.Scatter(x = anomalydata['timestamp'], y = anomalydata['value'], mode = 'lines',
                             marker = dict(color='red'),
                             name = 'detected_anomaly'))
#fig.show() #graph figure with discrete anomalies highlighted in red
st.plotly_chart(fig)

st.header("Conclusions")

st.markdown("It's obviously difficult to draw any sort of conclusions based on just this data set, but it seems like the orion ARIMA pipeline is detecting anomalous segments during certain times of drastic increase in price. Obviously this script will change based on the orion model used and the data set. There's also far too many variables that weren't taken into account to even have an idea about what's going on here.")

# In[ ]:




