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

# ------------------ FORECAST FUNCTION ------------------
def forecast_future_prices_with_ci(model, last_features_df, n_days):
    forecasts = []
    lower_ci = []
    upper_ci = []
    future_df = last_features_df.copy()
    
    for _ in range(n_days):
        X_new = future_df.iloc[-1].values.reshape(1, -1)  # full 11 features
        all_tree_preds = np.array([tree.predict(X_new)[0] for tree in model.estimators_])
        y_pred = np.mean(all_tree_preds)
        std_pred = np.std(all_tree_preds)
        forecasts.append(y_pred)
        lower_ci.append(y_pred - 1.96 * std_pred)
        upper_ci.append(y_pred + 1.96 * std_pred)
        
        # Build new row for next day
        new_row = {}
        # Shift lag features
        for i in range(7,0,-1):
            if i == 1:
                new_row[f'Close_Lag{i}'] = y_pred
            else:
                new_row[f'Close_Lag{i}'] = future_df.iloc[-1][f'Close_Lag{i-1}']
        # Update moving averages
        last_closes_3 = [new_row[f'Close_Lag{i}'] for i in range(1,4)]
        new_row['MA3'] = np.mean(last_closes_3)
        last_closes_5 = [new_row[f'Close_Lag{i}'] for i in range(1,6)]
        new_row['MA5'] = np.mean(last_closes_5)
        # Returns
        new_row['Return1'] = (new_row['Close_Lag1'] - future_df.iloc[-1]['Close_Lag1']) / future_df.iloc[-1]['Close_Lag1']
        new_row['Return3'] = (new_row['Close_Lag1'] - future_df.iloc[-1]['Close_Lag3']) / future_df.iloc[-1]['Close_Lag3']
        
        future_df = pd.concat([future_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return forecasts, lower_ci, upper_ci

# ------------------ PROCESS EACH STOCK ------------------
all_forecasts = {}
summary_table = []

for ticker in selected_stocks:
    st.write(f"### {ticker}")
    data = yf.download(ticker, start=start_date, end=today)
    if data.empty: 
        st.error(f"No data fetched for {ticker}")
        continue

    # ------------------ SAFE CLOSE COLUMN ------------------
    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            data_close = data['Close']
        else:
            data_close = data.iloc[:,0]
    else:
        data_close = data['Close'] if 'Close' in data.columns else data.iloc[:,0]

    # Ensure 1-D Series
    if isinstance(data_close, pd.DataFrame):
        data_close = data_close.iloc[:,0]

    # Convert to numeric and drop NaN
    close_series = pd.to_numeric(data_close, errors='coerce')
    close_series.dropna(inplace=True)

    # Final DataFrame
    data = pd.DataFrame({'Date': data.index, 'Close': close_series.values}).reset_index(drop=True)

    if data.empty:
        st.warning(f"No valid Close prices for {ticker}")
        continue

    # ------------------ LAST CLOSE ------------------
    last_close = data['Close'].iloc[-1]
    st.write(f"Last Close: Date {data['Date'].iloc[-1].date()}, Price ₹{last_close:.2f}")

    # ------------------ HISTORICAL CLOSE PLOT ------------------
    st.subheader(f'{ticker} Historical Close Price')
    plt.figure(figsize=(10,4))
    plt.plot(data['Date'], data['Close'], label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price (INR)')
    plt.title(f'{ticker} Historical Close Price')
    plt.legend()
    st.pyplot(plt)

    # ------------------ FEATURE ENGINEERING ------------------
    for i in range(1, 8):
        data[f'Close_Lag{i}'] = data['Close'].shift(i)
    data['MA3'] = data['Close'].rolling(3).mean()
    data['MA5'] = data['Close'].rolling(5).mean()
    data['Return1'] = data['Close'].pct_change(1)
    data['Return3'] = data['Close'].pct_change(3)
    data.dropna(inplace=True)

    feature_cols = [f'Close_Lag{i}' for i in range(1,8)] + ['MA3','MA5','Return1','Return3']
    X = data[feature_cols]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write('Model Performance: MSE:', mean_squared_error(y_test, y_pred), ', R2:', r2_score(y_test, y_pred))

    # ------------------ FORECAST ------------------
    max_forecast_days = (future_date_input - data['Date'].iloc[-1].date()).days
    if max_forecast_days < 1:
        st.warning(f"Prediction date for {ticker} is before last available date.")
        continue
    forecast_days = st.sidebar.slider(f"Forecast Days for {ticker}", 1, max_forecast_days, min(5, max_forecast_days))

    # Prepare last row for forecasting
    last_features_df = data[feature_cols].tail(1).reset_index(drop=True)
    forecasts, lower_ci, upper_ci = forecast_future_prices_with_ci(model, last_features_df, forecast_days)
    forecast_dates = pd.bdate_range(start=data['Date'].iloc[-1].date() + datetime.timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted_Close': forecasts,
        'Lower_CI': lower_ci,
        'Upper_CI': upper_ci
    })
    st.dataframe(forecast_df)
    all_forecasts[ticker] = forecast_df

    # Expected return
    expected_return = (forecasts[-1] - last_close) / last_close * 100
    summary_table.append({
        'Stock': ticker,
        'Last_Close': last_close,
        'Forecasted_Close': forecasts[-1],
        'Expected_Return_%': expected_return
    })

# ------------------ COMBINED FORECAST PLOT ------------------
st.subheader("Combined Forecasted Prices with 95% CI")
combined_forecast_fig, ax = plt.subplots(figsize=(12,6))
ax.set_title("Forecasted Close Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price (INR)")

colors = plt.cm.tab10.colors
ticker_colors = {ticker: colors[i % len(colors)] for i, ticker in enumerate(all_forecasts.keys())}
max_return_stock = max(summary_table, key=lambda x: x['Expected_Return_%'])['Stock'] if summary_table else None

for ticker, df_forecast in all_forecasts.items():
    color = ticker_colors[ticker]
    linewidth = 3 if ticker == max_return_stock else 1.5
    ax.plot(df_forecast['Date'], df_forecast['Forecasted_Close'], linestyle='--', color=color, linewidth=linewidth, label=f'{ticker} Forecast')
    ax.fill_between(df_forecast['Date'], df_forecast['Lower_CI'], df_forecast['Upper_CI'], color=color, alpha=0.2)

ax.legend()
st.pyplot(combined_forecast_fig)

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

