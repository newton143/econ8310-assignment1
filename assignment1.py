import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pygam import LinearGAM, s

# Load Training Data
df_train = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv", parse_dates=['Timestamp'])
df_train['Timestamp'] = pd.to_datetime(df_train['Timestamp'])
df_train.set_index('Timestamp', inplace=True)

# Set Frequency as Hours in Training Data
df_train = df_train.asfreq('h').dropna()

# Set exogenous features
df_train['year'] = df_train.index.year
df_train['month'] = df_train.index.month
df_train['day'] = df_train.index.day
df_train['hour'] = df_train.index.hour
exog_vars = ['year', 'month', 'day', 'hour']
X_train = df_train[exog_vars].astype(np.float64).values  
y_train = df_train['trips'].astype(np.float64).values

# Train GAM Model
model = LinearGAM(s(0) + s(1) + s(2) + s(3))
modelFit = model.fit(X_train, y_train)

# Generate Predictions for Training Data
y_pred_train = model.predict(X_train)

# Load Test Data
df_test = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv", parse_dates=['Timestamp'])
df_test['Timestamp'] = pd.to_datetime(df_test['Timestamp'])
df_test.set_index('Timestamp', inplace=True)

# Set Frequency as Hours in Test Data
df_test = df_test.asfreq('h').dropna()

# Set exogenous features
df_test['year'] = df_test.index.year
df_test['month'] = df_test.index.month
df_test['day'] = df_test.index.day
df_test['hour'] = df_test.index.hour
X_test = df_test[exog_vars].astype(np.float64).values  

# Generate Predictions for Test Data
y_pred_test = model.predict(X_test)
pred = y_pred_test

# # Save to CSV
df_test['predicted_trips'] = np.round(y_pred_test).astype(int)
df_test[['predicted_trips']].to_csv("predictions_gam.csv")

# Plot Results
fig = go.Figure()

# Scatter plot for actual train data
fig.add_trace(go.Scatter(
    x=df_train.index, y=y_train, mode='markers',
    name='Actual (Train)', marker=dict(color='blue')
))

# Line plot for training predictions
fig.add_trace(go.Scatter(
    x=df_train.index, y=y_pred_train, mode='lines',
    name='GAM Fit (Train)', line=dict(color='red')
))

# Line plot for test predictions
fig.add_trace(go.Scatter(
    x=df_test.index, y=df_test['predicted_trips'], mode='lines',
    name='Predicted (Test)', line=dict(color='green')
))

# Update layout
fig.update_layout(
    title="GAM Model Forecast for Taxi Trips",
    xaxis_title="Time",
    yaxis_title="Number of Trips",
    template="plotly_white"
)

# Show plot
fig.show()