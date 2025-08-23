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
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# ------------------ RANDOM USER LOGIN ------------------
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

# ------------------ STOCK PREDICTOR APP ------------------
stocks = [
    'YESBANK.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
    'AXISBANK.NS', 'BANKBARODA.NS', 'PNB.NS', 'INDUSINDBK.NS', 'IDFCFIRSTB.NS',
    'FEDERALBNK.NS'
]

ticker = st.sidebar.selectbox('Select a stock', stocks)
start_date = st.sidebar.date_input('Start Date', datetime.date(2015,1,1))
future_date_input = st.sidebar.date_input('Prediction Date', datetime.date(2025,12,25))
today = datetime.date.today()

# ------------------ FETCH DATA ------------------
st.write(f"Fetching data for {ticker}...")
data = yf.download(ticker, start=start_date, end=today)

if data.empty:
    st.error("No data fetched. Check ticker or date range.")
    st.stop()

# ------------------ SAFE CLOSE SERIES ------------------
if 'Close' in data.columns:
    data_close = data['Close']  # already a Series
else:
    data_close = data.iloc[:, 0]  # fallback to first column
    if isinstance(data_close, pd.DataFrame):
        data_close = data_close.iloc[:, 0]  # ensure 1-D Series

# Ensure 1-D Series
if not isinstance(data_close, pd.Series):
    data_close = pd.Series(data_close.values.flatten())

# Convert to numeric and drop NaNs
data_close = pd.to_numeric(data_close, errors='coerce')
data_close.dropna(inplace=True)

# Reset index to get Date column
data_close = data_close.reset_index()
data_close.columns = ['Date', 'Close']
data = data_close.copy()

# ------------------ PLOT HISTORICAL CLOSE PRICE ------------------
st.subheader('Historical Close Price')
plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price (INR)')
plt.title(f'{ticker} Historical Close Price')
plt.legend()
st.pyplot(plt)

# ------------------ FEATURE ENGINEERING ------------------
if len(data) < 4:
    st.error("Not enough data to create lag features.")
else:
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Close_Lag2'] = data['Close'].shift(2)
    data['Close_Lag3'] = data['Close'].shift(3)
    data.dropna(inplace=True)

    X = data[['Close_Lag1', 'Close_Lag2', 'Close_Lag3']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader('Model Performance on Test Set')
    st.write('MSE:', mean_squared_error(y_test, y_pred))
    st.write('R2 Score:', r2_score(y_test, y_pred))

    # Predict future price
    last_data = data.tail(3)['Close'].values
    future_features = np.array([last_data[-1], last_data[-2], last_data[-3]]).reshape(1,-1)
    future_price = model.predict(future_features)[0]

    st.subheader(f'Predicted Close Price for {future_date_input}')
    st.write(f"₹{future_price:.2f}")

    # ------------------ PLOT HISTORICAL + PREDICTED ------------------
    st.subheader('Historical & Predicted Price')
    plt.figure(figsize=(12,6))
    plt.plot(data['Date'], data['Close'], label='Historical Close Price')
    plt.scatter(future_date_input, future_price, color='red', label=f'Predicted {future_date_input}', s=100)
    plt.xlabel('Date')
    plt.ylabel('Close Price (INR)')
    plt.legend()
    st.pyplot(plt)

    # ------------------ LAST CLOSE ------------------
    last_close = data['Close'].iloc[-1]
    st.write(f"Last historical close price: ₹{last_close:.2f}")

