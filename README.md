# Time-Series-Analysis
Problem Statement:

Given a simulated time stamps of a support requests for a company, aggregate the time series into 15 min time intervals, visualize and forecast the demand over the following one hour with 15 mins granularity.

Data Analyse:

The data consists of uneven time period to aggregate into 15 min time interval. So, I assumed that the time period starts from 01/01/70 20:12 and compute the 15 minutes granularity keeping this as reference and we simulate the data. The data points are marked as 0 when there is no support request during that time stamp range.
First I started with discovering the patterns of the demand using the rolling mean and rolling variance I could discover the trend and seasonality of the time series which was shown in the various figures attached. In those graphs, it says that there is neither a fixed increasing nor decreasing trend in this time series. Seeing the daily graph (daily.png) and box-plot for every month one can sense that there is weekly seasonality effect in the time series (which is roughly 480 time stamps). This is used for differencing the time series which is used for Method -2 (ARIMA) to make time series stationary. 
Explored some libraries to extract the features of time-series other than the figured feature weekly cycle with the help of tsfresh but didn’t give out good features (mostly are redundant in this time series). So used the daily cycle as the feature for the Method-1 (LSTM model) implemented which consists of 96 features / 50 features as 480 hidden units making the network too big which is time taking for training.
