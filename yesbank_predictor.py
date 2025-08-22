import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import datetime


ticker = 'YESBANK.NS' # Yes Bank listed on NSE India
today = datetime.datetime.today().strftime('%Y-%m-%d')
data = yf.download(ticker, start='2015-01-01', end=today)


# Reset index
data.reset_index(inplace=True)
print(data.head())
print(data.tail())
print(data.info())

plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price (INR)')
plt.title(f'{ticker} Historical Close Price')
plt.legend()
plt.show()


data['Close_Lag1'] = data['Close'].shift(1)
data['Close_Lag2'] = data['Close'].shift(2)
data['Close_Lag3'] = data['Close'].shift(3)


# Drop missing rows
data.dropna(inplace=True)


X = data[['Close_Lag1', 'Close_Lag2', 'Close_Lag3']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


#############################################
# 6. Model Training (Random Forest Regressor)
#############################################
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


#############################################
# 7. Model Evaluation
#############################################
y_pred = model.predict(X_test)


print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# Plot actual vs predicted
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Random Forest - Yes Bank Close Price Prediction (Test Set)')
plt.show()


#############################################
# 8. Predict Future Price (25th Dec 2025) & Plot
#############################################
# Take last 3 closing values from dataset
last_data = data.tail(3)['Close'].values


future_features = np.array([last_data[-1], last_data[-2], last_data[-3]]).reshape(1,-1)
future_price = model.predict(future_features)[0]


print(f"Predicted Close Price for {ticker} on 25 Dec 2025: ₹{future_price:.2f}")


# Plot historical + future prediction
plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Close'], label='Historical Close Price')
future_date = datetime.datetime(2025,12,25)
plt.scatter(future_date, future_price, color='red', label='Predicted 25 Dec 2025', s=100)
plt.xlabel('Date')
plt.ylabel('Close Price (INR)')
plt.title(f'{ticker} Historical & Predicted Close Price')
plt.legend()
plt.show()
