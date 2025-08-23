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

selected_stocks = st.sidebar.multiselect("Select one or more stocks", stocks, default=['YESBANK.NS'])
start_date = st.sidebar.date_input('Start Date', datetime.date(2015,1,1))
future_date_input = st.sidebar.date_input('Prediction End Date', datetime.date(2025,12,25))
today = datetime.date.today()

if len(selected_stocks) == 0:
    st.warning("Please select at least one stock.")
    st.stop()

# ------------------ FORECAST FUNCTION WITH CI ------------------
def forecast_future_prices_with_ci(model, last_data, n_days):
    forecasts = []
    lower_ci = []
    upper_ci = []
    data_window = list(last_data[-3:])

    for _ in range(n_days):
        X_new = np.array(data_window[-3:]).reshape(1, -1)
        all_tree_preds = np.array([tree.predict(X_new)[0] for tree in model.estimators_])
        y_pred = np.mean(all_tree_preds)
        std_pred = np.std(all_tree_preds)
        forecasts.append(y_pred)
        lower_ci.append(y_pred - 1.96 * std_pred)
        upper_ci.append(y_pred + 1.96 * std_pred)
        data_window.append(y_pred)
    return forecasts, lower_ci, upper_ci

# ------------------ PROCESS EACH SELECTED STOCK ------------------
st.subheader("Stock Forecasts")
combined_forecast_plot = plt.figure(figsize=(12,6))
plt.title("Historical & Forecasted Close Prices")
plt.xlabel("Date")
plt.ylabel("Close Price (INR)")

all_forecasts = {}  # for CSV download
summary_table = []  # for expected return comparison

for ticker in selected_stocks:
    st.write(f"### {ticker}")
    # Fetch data
    data = yf.download(ticker, start=start_date, end=today)
    if data.empty:
        st.error(f"No data fetched for {ticker}")
        continue

    # Handle Close column
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            data_close = data['Close']
        else:
            st.error(f"Close column not found for {ticker}")
            continue
    elif 'Close' in data.columns:
        data_close = data['Close']
    else:
        data_close = data.iloc[:, 0]
        data_close = data_close.to_frame(name='Close')

    # Numeric type
    if isinstance(data_close, pd.DataFrame):
        data_close['Close'] = pd.to_numeric(data_close.iloc[:, 0], errors='coerce')
    elif isinstance(data_close, pd.Series):
        data_close = pd.to_numeric(data_close, errors='coerce').to_frame(name='Close')
    else:
        st.error(f"Cannot process Close for {ticker}")
        continue

    data_close.dropna(subset=['Close'], inplace=True)
    data_close.reset_index(inplace=True)
    data = data_close.copy()

    # Last close
    last_close = data['Close'].iloc[-1]
    last_close_date = data['Date'].iloc[-1]
    st.write(f"Last Close: Date {last_close_date.date()}, Price ₹{last_close:.2f}")

    # Plot historical
    plt.plot(data['Date'], data['Close'], label=f'{ticker} Historical')

    # Feature engineering
    if len(data) < 4:
        st.warning(f"Not enough data for {ticker} to forecast.")
        continue
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Close_Lag2'] = data['Close'].shift(2)
    data['Close_Lag3'] = data['Close'].shift(3)
    data.dropna(inplace=True)

    X = data[['Close_Lag1','Close_Lag2','Close_Lag3']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write('Model Performance: MSE:', mean_squared_error(y_test, y_pred), ', R2:', r2_score(y_test, y_pred))

    # Forecast
    max_forecast_days = (future_date_input - data['Date'].iloc[-1].date()).days
    if max_forecast_days < 1:
        st.warning(f"Prediction date for {ticker} is before or equal to last available date.")
        continue
    forecast_days = st.sidebar.slider(f"Forecast Days for {ticker}", 1, max_forecast_days, min(5, max_forecast_days))

    last_data = data['Close'].tail(3).values
    forecasts, lower_ci, upper_ci = forecast_future_prices_with_ci(model, last_data, forecast_days)
    forecast_dates = pd.bdate_range(start=data['Date'].iloc[-1].date() + datetime.timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted_Close': forecasts,
        'Lower_CI': lower_ci,
        'Upper_CI': upper_ci
    })

    st.dataframe(forecast_df)
    plt.plot(forecast_dates, forecasts, linestyle='--', label=f'{ticker} Forecast')
    plt.fill_between(forecast_dates, lower_ci, upper_ci, alpha=0.2)

    # Save for CSV download
    all_forecasts[ticker] = forecast_df

    # Calculate expected return
    expected_return = (forecasts[-1] - last_close) / last_close * 100
    summary_table.append({
        'Stock': ticker,
        'Last_Close': last_close,
        'Forecasted_Close': forecasts[-1],
        'Expected_Return_%': expected_return
    })

# ------------------ SHOW COMBINED PLOT ------------------
plt.legend()
st.pyplot(combined_forecast_plot)

# ------------------ SUMMARY TABLE ------------------
if summary_table:
    st.subheader("Forecast Summary Table")
    summary_df = pd.DataFrame(summary_table)
    st.dataframe(summary_df.sort_values('Expected_Return_%', ascending=False))

# ------------------ DOWNLOAD FORECASTS AS CSV ------------------
if all_forecasts:
    combined_csv = pd.concat(all_forecasts, names=['Stock']).reset_index(level=0)
    csv_bytes = combined_csv.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download All Forecasts CSV", csv_bytes, "multi_stock_forecast.csv", "text/csv")
