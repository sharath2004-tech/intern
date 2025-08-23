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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

# ------------------ USER LOGIN ------------------
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

st.write(f"Hello, {st.session_state.username}! Access the stock predictor below.")

# ------------------ STOCK SELECTION ------------------
stocks = [
    'YESBANK.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
    'AXISBANK.NS', 'BANKBARODA.NS', 'PNB.NS', 'INDUSINDBK.NS', 'IDFCFIRSTB.NS',
    'FEDERALBNK.NS'
]

selected_stock = st.sidebar.selectbox("Select a stock", stocks)
start_date = st.sidebar.date_input('Start Date', datetime.date(2015,1,1))
forecast_date = st.sidebar.date_input('Forecast End Date', datetime.date(2025,12,25))
today = datetime.date.today()

# ------------------ FETCH DATA ------------------
data = yf.download(selected_stock, start=start_date, end=today)
if data.empty:
    st.error("No data fetched. Check ticker or date range.")
    st.stop()

# ------------------ SAFE CLOSE SERIES ------------------
if 'Close' in data.columns:
    data_close = data['Close']
else:
    data_close = data.iloc[:, 0]

# Ensure 1-D Series
if isinstance(data_close, pd.DataFrame):
    data_close = data_close.iloc[:, 0]
if not isinstance(data_close, pd.Series):
    data_close = pd.Series(data_close.values)

# Convert to numeric and drop NaNs
data_close = pd.to_numeric(data_close, errors='coerce')
data_close.dropna(inplace=True)

# Reset index and rename columns explicitly
data_close = data_close.reset_index()
data_close.columns = ['Date', 'Close']  # <-- Key fix

# ------------------ SCALE DATA ------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_close['Close'].values.reshape(-1,1))

# ------------------ CREATE SEQUENCES ------------------
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, 0])
        y.append(data[i,0])
    return np.array(X), np.array(y)

SEQ_LEN = 60
X, y = create_sequences(scaled_data, SEQ_LEN)
X = X.reshape((X.shape[0], X.shape[1], 1))

# ------------------ TRAIN LSTM ------------------
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

# ------------------ FORECAST ------------------
last_seq = scaled_data[-SEQ_LEN:].reshape(1,SEQ_LEN,1)
forecast_days = (forecast_date - data_close['Date'].iloc[-1].date()).days

# Monte Carlo simulation for confidence interval
num_sim = 50
simulations = []

for _ in range(num_sim):
    temp_seq = last_seq.copy()
    sim = []
    for _ in range(forecast_days):
        pred = model.predict(temp_seq, verbose=0)[0,0]
        pred += np.random.normal(0, 0.002)  # small noise
        sim.append(pred)
        temp_seq = np.append(temp_seq[:,1:,:], [[[pred]]], axis=1)
    simulations.append(sim)

simulations = np.array(simulations)
forecast_mean = scaler.inverse_transform(simulations.mean(axis=0).reshape(-1,1)).flatten()
forecast_lower = scaler.inverse_transform(np.percentile(simulations, 2.5, axis=0).reshape(-1,1)).flatten()
forecast_upper = scaler.inverse_transform(np.percentile(simulations, 97.5, axis=0).reshape(-1,1)).flatten()

forecast_dates = pd.bdate_range(start=data_close['Date'].iloc[-1].date() + datetime.timedelta(days=1), periods=forecast_days)

forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Forecasted_Close': forecast_mean,
    'Lower_CI': forecast_lower,
    'Upper_CI': forecast_upper
})

# ------------------ PLOTS ------------------
st.subheader("Historical Close Price")
plt.figure(figsize=(12,6))
plt.plot(data_close['Date'], data_close['Close'], label='Historical Close')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
st.pyplot(plt)

st.subheader(f"Forecasted Close Price till {forecast_date}")
plt.figure(figsize=(12,6))
plt.plot(data_close['Date'], data_close['Close'], label='Historical Close')
plt.plot(forecast_df['Date'], forecast_df['Forecasted_Close'], linestyle='--', color='orange', label='Forecast')
plt.fill_between(forecast_df['Date'], forecast_df['Lower_CI'], forecast_df['Upper_CI'], color='orange', alpha=0.2, label='95% CI')
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
st.pyplot(plt)

st.subheader("Forecast Table")
st.dataframe(forecast_df)

# ------------------ LAST CLOSE AND EXPECTED RETURN ------------------
last_close = data_close['Close'].iloc[-1]
st.write(f"Last Close: ₹{last_close:.2f}")
expected_return = (forecast_mean[-1] - last_close)/last_close*100
st.write(f"Forecasted Price on {forecast_date}: ₹{forecast_mean[-1]:.2f}")
st.write(f"Expected Return: {expected_return:.2f}%")

