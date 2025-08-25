# # import yfinance as yf
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.ensemble import RandomForestRegressor
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import mean_squared_error, r2_score
# # import datetime


# # ticker = 'YESBANK.NS' # Yes Bank listed on NSE India
# # today = datetime.datetime.today().strftime('%Y-%m-%d')
# # data = yf.download(ticker, start='2015-01-01', end=today)


# # # Reset index
# # data.reset_index(inplace=True)
# # print(data.head())
# # print(data.tail())
# # print(data.info())

# # plt.figure(figsize=(12,6))
# # plt.plot(data['Date'], data['Close'], label='Close Price')
# # plt.xlabel('Date')
# # plt.ylabel('Close Price (INR)')
# # plt.title(f'{ticker} Historical Close Price')
# # plt.legend()
# # plt.show()


# # data['Close_Lag1'] = data['Close'].shift(1)
# # data['Close_Lag2'] = data['Close'].shift(2)
# # data['Close_Lag3'] = data['Close'].shift(3)


# # # Drop missing rows
# # data.dropna(inplace=True)


# # X = data[['Close_Lag1', 'Close_Lag2', 'Close_Lag3']]
# # y = data['Close']
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


# # #############################################
# # # 6. Model Training (Random Forest Regressor)
# # #############################################
# # model = RandomForestRegressor(n_estimators=200, random_state=42)
# # model.fit(X_train, y_train)


# # #############################################
# # # 7. Model Evaluation
# # #############################################
# # y_pred = model.predict(X_test)


# # print("MSE:", mean_squared_error(y_test, y_pred))
# # print("R2 Score:", r2_score(y_test, y_pred))


# # # Plot actual vs predicted
# # plt.figure(figsize=(10,6))
# # plt.plot(y_test.values, label='Actual')
# # plt.plot(y_pred, label='Predicted')
# # plt.legend()
# # plt.title('Random Forest - Yes Bank Close Price Prediction (Test Set)')
# # plt.show()


# # #############################################
# # # 8. Predict Future Price (25th Dec 2025) & Plot
# # #############################################
# # # Take last 3 closing values from dataset
# # last_data = data.tail(3)['Close'].values


# # future_features = np.array([last_data[-1], last_data[-2], last_data[-3]]).reshape(1,-1)
# # future_price = model.predict(future_features)[0]


# # print(f"Predicted Close Price for {ticker} on 25 Dec 2025: ₹{future_price:.2f}")


# # # Plot historical + future prediction
# # plt.figure(figsize=(12,6))
# # plt.plot(data['Date'], data['Close'], label='Historical Close Price')
# # future_date = datetime.datetime(2025,12,25)
# # plt.scatter(future_date, future_price, color='red', label='Predicted 25 Dec 2025', s=100)
# # plt.xlabel('Date')
# # plt.ylabel('Close Price (INR)')
# # plt.title(f'{ticker} Historical & Predicted Close Price')
# # plt.legend()
# # plt.show()
# # Simple authentication


# # Hardcoded credentials (for demo)


# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score
# import datetime





# # ------------------ LOGIN ------------------
# if 'logged_in' not in st.session_state:
#     st.session_state.logged_in = False
#     st.session_state.username = None

# if not st.session_state.logged_in:
#     st.subheader("Sign In")
#     username = st.text_input("Enter your username")
#     login_btn = st.button("Login")
#     if login_btn:
#         if username.strip() != "":
#             st.session_state.logged_in = True
#             st.session_state.username = username.strip()
#             st.success(f"Welcome {st.session_state.username}!")
#         else:
#             st.error("Please enter a valid username")
#     st.stop()

# st.write(f"Hello, {st.session_state.username}! Access the stock predictor app.")




# # ------------------ SETTINGS ------------------
# stocks = [
#     'YESBANK.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
#     'AXISBANK.NS', 'BANKBARODA.NS', 'PNB.NS', 'INDUSINDBK.NS', 'IDFCFIRSTB.NS',
#     'FEDERALBNK.NS'
# ]

# stock1 = st.sidebar.selectbox('Select first stock', stocks, index=0)
# stock2 = st.sidebar.selectbox('Select second stock', stocks, index=1)
# start_date = st.sidebar.date_input('Start Date', datetime.date(2015,1,1))
# future_date_input = st.sidebar.date_input('Prediction Date', datetime.date(2025,12,31))
# today = datetime.date.today()






# # ------------------ FETCH DATA FUNCTION ------------------
# @st.cache_data
# def fetch_data(ticker):
#     df = yf.download(ticker, start=start_date, end=today)
#     if df.empty:
#         return None

#     # Handle MultiIndex
#     if isinstance(df.columns, pd.MultiIndex):
#         if 'Close' in df.columns.get_level_values(0):
#             close = df['Close']
#         else:
#             close = df.iloc[:, 0]
#     else:
#         close = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]

#     if isinstance(close, pd.DataFrame):
#         close = close.iloc[:, 0]  # Ensure 1-D Series

#     close = pd.to_numeric(close, errors='coerce')
#     close = close.dropna().reset_index()
#     close.columns = ['Date', 'Close']
#     return close

# data1 = fetch_data(stock1)
# data2 = fetch_data(stock2)

# if data1 is None or data2 is None:
#     st.error("Failed to fetch data for one or both stocks. Check ticker or date range.")
#     st.stop()





# # ------------------ PLOT HISTORICAL ------------------
# st.subheader(f"{stock1} Historical Close")
# plt.figure(figsize=(10,5))
# plt.plot(data1['Date'], data1['Close'], label=f'{stock1} Close')
# plt.xlabel('Date')
# plt.ylabel('Close Price (INR)')
# plt.legend()
# st.pyplot(plt)

# st.subheader(f"{stock2} Historical Close")
# plt.figure(figsize=(10,5))
# plt.plot(data2['Date'], data2['Close'], label=f'{stock2} Close', color='orange')
# plt.xlabel('Date')
# plt.ylabel('Close Price (INR)')
# plt.legend()
# st.pyplot(plt)





# # ------------------ FEATURE ENGINEERING & MODEL ------------------
# def train_and_forecast(data, future_date):
#     # Lag features
#     data['Lag1'] = data['Close'].shift(1)
#     data['Lag2'] = data['Close'].shift(2)
#     data['Lag3'] = data['Close'].shift(3)
#     data.dropna(inplace=True)

#     X = data[['Lag1','Lag2','Lag3']]
#     y = data['Close']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#     model = RandomForestRegressor(n_estimators=200, random_state=42)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     # Forecast next day (or n days)
#     last_data = data.tail(3)['Close'].values
#     future_features = np.array([last_data[-1], last_data[-2], last_data[-3]]).reshape(1,-1)
#     future_price = model.predict(future_features)[0]

#     return model, future_price, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)

# model1, forecast1, mse1, r21 = train_and_forecast(data1, future_date_input)
# model2, forecast2, mse2, r22 = train_and_forecast(data2, future_date_input)





# # ------------------ DISPLAY FORECAST ------------------
# st.subheader(f"Predicted Close Price for {future_date_input}")
# st.write(f"{stock1}: ₹{forecast1:.2f} (MSE: {mse1:.2f}, R2: {r21:.2f})")
# st.write(f"{stock2}: ₹{forecast2:.2f} (MSE: {mse2:.2f}, R2: {r22:.2f})")





# # ------------------ PLOT BOTH HISTORICAL + FORECAST ------------------
# st.subheader("Historical & Predicted Close Price")
# plt.figure(figsize=(10,5))
# plt.plot(data1['Date'], data1['Close'], label=f'{stock1} Historical')
# plt.plot(data2['Date'], data2['Close'], label=f'{stock2} Historical', color='orange')
# plt.scatter(future_date_input, forecast1, color='blue', label=f'{stock1} Forecast', s=100)
# plt.scatter(future_date_input, forecast2, color='red', label=f'{stock2} Forecast', s=100)
# plt.xlabel('Date')
# plt.ylabel('Close Price (INR)')
# plt.legend()
# st.pyplot(plt)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# ------------------ DATABASE ------------------
conn = sqlite3.connect("stock_app.db")
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY
            )""")
c.execute("""CREATE TABLE IF NOT EXISTS predictions (
                username TEXT,
                stock TEXT,
                prediction_date TEXT,
                predicted_price REAL,
                mse REAL,
                r2 REAL
            )""")
conn.commit()

def add_user(username):
    c.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username,))
    conn.commit()

def save_prediction(username, stock, prediction_date, price, mse, r2):
    c.execute("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
              (username, stock, prediction_date, price, mse, r2))
    conn.commit()
    get_user_history.clear()  # 🔥 Clears ONLY the cached history


@st.cache_data
def get_user_history(username):
    query = f"SELECT * FROM predictions WHERE username='{username}'"
    return pd.read_sql(query, conn)


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
            add_user(username.strip())
            st.success(f"Welcome {st.session_state.username}!")
        else:
            st.error("Please enter a valid username")
    st.stop()

st.write(f"Hello, {st.session_state.username}! Access the stock predictor app.")


# ------------------ SETTINGS ------------------
stocks = [
    'YESBANK.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS',
    'AXISBANK.NS', 'BANKBARODA.NS', 'PNB.NS', 'INDUSINDBK.NS', 'IDFCFIRSTB.NS',
    'FEDERALBNK.NS'
]

selected_stocks = st.sidebar.multiselect('Select stocks to compare', stocks, default=[stocks[0], stocks[1]])
start_date = st.sidebar.date_input('Start Date', datetime.date(2015,1,1))
future_date_input = st.sidebar.date_input('Prediction Date', datetime.date(2025,12,31))
test_size = st.sidebar.slider("Test Size (for model)", 0.1, 0.5, 0.2, 0.05)
n_estimators = st.sidebar.slider("Random Forest Trees", 50, 500, 200, 50)
today = datetime.date.today()


# ------------------ FETCH DATA ------------------
@st.cache_data
def fetch_data(ticker, start_date, today):
    df = yf.download(ticker, start=start_date, end=today)
    if df.empty:
        return None
    close = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce').dropna().reset_index()
    close.columns = ['Date', 'Close']
    return close

@st.cache_data
def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    todays_data = stock.history(period='1d')
    return todays_data['Close'].iloc[-1] if not todays_data.empty else None


# ------------------ MODEL (CACHED) ------------------
@st.cache_data
def train_and_forecast(data, future_date, test_size, n_estimators):
    data = data.copy()  # Prevent modifying cached object
    data['Lag1'] = data['Close'].shift(1)
    data['Lag2'] = data['Close'].shift(2)
    data['Lag3'] = data['Close'].shift(3)
    data.dropna(inplace=True)

    X = data[['Lag1', 'Lag2', 'Lag3']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    last_data = data.tail(3)['Close'].values
    future_features = np.array([last_data[-1], last_data[-2], last_data[-3]]).reshape(1, -1)
    future_price = model.predict(future_features)[0]

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return future_price, mse, r2


# ------------------ APP TABS ------------------
tab1, tab2 = st.tabs(["📈 Present Stock Prices", "🔮 Future Predictions"])

# ---- TAB 1: Present Prices ----
with tab1:
    st.subheader("Current Stock Prices & Historical Data")
    for stock in selected_stocks:
        data = fetch_data(stock, start_date, today)
        current_price = get_current_price(stock)

        if data is None:
            st.error(f"Failed to fetch data for {stock}")
            continue

        st.markdown(f"**{stock}**: Current Price ₹{current_price:.2f}" if current_price else f"**{stock}**: Price unavailable")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name=f'{stock} Historical'))
        st.plotly_chart(fig, use_container_width=True)


# ---- TAB 2: Future Predictions ----
with tab2:
    st.subheader(f"Predicted Stock Prices for {future_date_input}")
    for stock in selected_stocks:
        data = fetch_data(stock, start_date, today)
        if data is None:
            st.error(f"Failed to fetch data for {stock}")
            continue

        price, mse, r2 = train_and_forecast(data, future_date_input, test_size, n_estimators)
        save_prediction(st.session_state.username, stock, str(future_date_input), price, mse, r2)

        st.markdown(f"**{stock}**: Predicted ₹{price:.2f} on {future_date_input} (MSE: {mse:.2f}, R²: {r2:.2f})")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name=f'{stock} Historical'))
        fig.add_trace(go.Scatter(x=[future_date_input], y=[price], mode='markers', marker=dict(size=10, color='red'), name='Prediction'))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Your Prediction History")
    history = get_user_history(st.session_state.username)
    st.dataframe(history)
