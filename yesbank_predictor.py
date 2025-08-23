# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# import datetime


# ticker = 'YESBANK.NS' # Yes Bank listed on NSE India
# today = datetime.datetime.today().strftime('%Y-%m-%d')
# data = yf.download(ticker, start='2015-01-01', end=today)


# # Reset index
# data.reset_index(inplace=True)
# print(data.head())
# print(data.tail())
# print(data.info())

# plt.figure(figsize=(12,6))
# plt.plot(data['Date'], data['Close'], label='Close Price')
# plt.xlabel('Date')
# plt.ylabel('Close Price (INR)')
# plt.title(f'{ticker} Historical Close Price')
# plt.legend()
# plt.show()


# data['Close_Lag1'] = data['Close'].shift(1)
# data['Close_Lag2'] = data['Close'].shift(2)
# data['Close_Lag3'] = data['Close'].shift(3)


# # Drop missing rows
# data.dropna(inplace=True)


# X = data[['Close_Lag1', 'Close_Lag2', 'Close_Lag3']]
# y = data['Close']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# #############################################
# # 6. Model Training (Random Forest Regressor)
# #############################################
# model = RandomForestRegressor(n_estimators=200, random_state=42)
# model.fit(X_train, y_train)


# #############################################
# # 7. Model Evaluation
# #############################################
# y_pred = model.predict(X_test)


# print("MSE:", mean_squared_error(y_test, y_pred))
# print("R2 Score:", r2_score(y_test, y_pred))


# # Plot actual vs predicted
# plt.figure(figsize=(10,6))
# plt.plot(y_test.values, label='Actual')
# plt.plot(y_pred, label='Predicted')
# plt.legend()
# plt.title('Random Forest - Yes Bank Close Price Prediction (Test Set)')
# plt.show()


# #############################################
# # 8. Predict Future Price (25th Dec 2025) & Plot
# #############################################
# # Take last 3 closing values from dataset
# last_data = data.tail(3)['Close'].values


# future_features = np.array([last_data[-1], last_data[-2], last_data[-3]]).reshape(1,-1)
# future_price = model.predict(future_features)[0]


# print(f"Predicted Close Price for {ticker} on 25 Dec 2025: ₹{future_price:.2f}")


# # Plot historical + future prediction
# plt.figure(figsize=(12,6))
# plt.plot(data['Date'], data['Close'], label='Historical Close Price')
# future_date = datetime.datetime(2025,12,25)
# plt.scatter(future_date, future_price, color='red', label='Predicted 25 Dec 2025', s=100)
# plt.xlabel('Date')
# plt.ylabel('Close Price (INR)')
# plt.title(f'{ticker} Historical & Predicted Close Price')
# plt.legend()
# plt.show()
# Simple authentication


# Hardcoded credentials (for demo)


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime

# ------------------ LOGIN ------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if not st.session_state.logged_in:
    st.subheader("Sign In")
    username = st.text_input("Enter your username")
    login_btn = st.button("Login")

    if login_btn:
        if username.strip() != "":
            st.session_state.logged_in = True
            st.session_state.username = username.strip()
            st.success(f"Welcome {st.session_state.username}!")
        else:
            st.error("Please enter a valid username")
    st.stop()

st.write(f"Hello, {st.session_state.username}! You can now access the stock predictor app.")

# ------------------ STOCK SELECTION ------------------
stocks = [
    'YESBANK.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
    'AXISBANK.NS', 'BANKBARODA.NS', 'PNB.NS', 'INDUSINDBK.NS', 'IDFCFIRSTB.NS',
    'FEDERALBNK.NS'
]

stock1 = st.sidebar.selectbox("Select first stock", stocks, index=0)
stock2 = st.sidebar.selectbox("Select second stock", stocks, index=1)
start_date = st.sidebar.date_input('Start Date', datetime.date(2015,1,1))
future_date_input = st.sidebar.date_input('Prediction Date', datetime.date(2025,12,25))
today = datetime.date.today()

# ------------------ FETCH DATA ------------------
@st.cache_data
def fetch_data(ticker):
    df = yf.download(ticker, start=start_date, end=today)
    if df.empty:
        return None
    if 'Close' in df.columns:
        close = df['Close']
    else:
        close = df.iloc[:, 0]
    close = pd.Series(pd.to_numeric(close, errors='coerce')).dropna()
    close = close.reset_index()
    close.columns = ['Date', 'Close']
    return close

data1 = fetch_data(stock1)
data2 = fetch_data(stock2)

if data1 is None or data2 is None:
    st.error("Could not fetch data for one of the stocks.")
    st.stop()

# ------------------ HISTORICAL PLOTS ------------------
st.subheader(f"Historical Close Price: {stock1}")
plt.figure(figsize=(12,5))
plt.plot(data1['Date'], data1['Close'], label=stock1, color='blue')
plt.xlabel("Date")
plt.ylabel("Close Price (INR)")
plt.legend()
st.pyplot(plt)

st.subheader(f"Historical Close Price: {stock2}")
plt.figure(figsize=(12,5))
plt.plot(data2['Date'], data2['Close'], label=stock2, color='green')
plt.xlabel("Date")
plt.ylabel("Close Price (INR)")
plt.legend()
st.pyplot(plt)

# ------------------ RANDOM FOREST FORECAST FUNCTION ------------------
def forecast_stock(data):
    if len(data) < 4:
        return None, None
    df = data.copy()
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['Lag3'] = df['Close'].shift(3)
    df.dropna(inplace=True)

    X = df[['Lag1','Lag2','Lag3']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    last_data = df.tail(3)['Close'].values
    future_features = np.array([last_data[-1], last_data[-2], last_data[-3]]).reshape(1,-1)
    future_price = model.predict(future_features)[0]
    return future_price, model

# ------------------ FORECAST ------------------
forecast1, _ = forecast_stock(data1)
forecast2, _ = forecast_stock(data2)

# ------------------ SEPARATE FORECAST PLOTS ------------------
st.subheader(f"Forecasted Close Price: {stock1}")
plt.figure(figsize=(12,5))
plt.plot(data1['Date'].iloc[-100:], data1['Close'].iloc[-100:], label=f'{stock1} Historical', color='blue')
plt.scatter(future_date_input, forecast1, color='red', label=f'{stock1} Forecast', s=100)
plt.xlabel("Date")
plt.ylabel("Close Price (INR)")
plt.legend()
st.pyplot(plt)
st.write(f"Forecasted {stock1} Close Price on {future_date_input}: ₹{forecast1:.2f}")

st.subheader(f"Forecasted Close Price: {stock2}")
plt.figure(figsize=(12,5))
plt.plot(data2['Date'].iloc[-100:], data2['Close'].iloc[-100:], label=f'{stock2} Historical', color='green')
plt.scatter(future_date_input, forecast2, color='orange', label=f'{stock2} Forecast', s=100)
plt.xlabel("Date")
plt.ylabel("Close Price (INR)")
plt.legend()
st.pyplot(plt)
st.write(f"Forecasted {stock2} Close Price on {future_date_input}: ₹{forecast2:.2f}")
