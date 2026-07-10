import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta

# Load the trained models
try:
    lstm_model = tf.keras.models.load_model("lstm_model.keras")
    mae_model = tf.keras.models.load_model("mae_model.keras")
    scaler = pickle.load(open("scaler.pkl", "rb"))  # Load the scaler used in training
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Function to fetch stock data
def fetch_stock_data(stock, start, end):
    df = yf.download(stock, start=start, end=end)
    return df

# Function to preprocess data
def preprocess_data(data, time_step=100):
    data_scaled = scaler.transform(data)
    X, Y = [], []
    for i in range(len(data_scaled) - time_step - 1):
        X.append(data_scaled[i : i + time_step, 0])
        Y.append(data_scaled[i + time_step, 0])
    return np.array(X), np.array(Y)

# Get today's date dynamically
today = datetime.today().date()
max_date = today - timedelta(days=1)
min_date = max_date - timedelta(days=365 * 10)  # 10 years back

# Streamlit UI
st.title("Stock Price Prediction App 📈")

# User input for stock symbol
stock_symbol = st.text_input("Enter Stock Ticker (e.g., AAPL or NEPSE symbol):", "AAPL")

# User input for date range selection
start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

if st.button("Predict"):
    st.write(f"Fetching data for {stock_symbol} from {start_date} to {end_date}...")
    df = fetch_stock_data(stock_symbol, start=start_date, end=end_date)
    
    if df.empty:
        st.error("Invalid stock symbol or no data available!")
    else:
        st.write("Data fetched successfully!")
        close_prices = df["Close"].values.reshape(-1, 1)
        X_test, Y_test = preprocess_data(close_prices)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)

        # Make predictions
        lstm_predictions = lstm_model.predict(X_test)
        mae_predictions = mae_model.predict(X_test)
        
        lstm_predictions = scaler.inverse_transform(lstm_predictions)
        mae_predictions = scaler.inverse_transform(mae_predictions)
        Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))
        
        # Last 60 days for comparison
        dates_last_60 = pd.date_range(end=today - timedelta(days=1), periods=60).strftime('%B %d, %Y')

        # Define currency format
        currency = "$" if "AAPL" in stock_symbol else "Rs"
        
        # Plot Actual vs LSTM (Last 60 Days)
        st.subheader("Actual vs LSTM Prediction (Last 60 Days) 📊")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates_last_60, Y_test_actual[-60:], label="Actual Price", color="green", linestyle="--", linewidth=2)
        ax.plot(dates_last_60, lstm_predictions[-60:], label="LSTM Predicted Price", color="red", linewidth=2)

        # Formatting x-axis ticks (Show fewer dates)
        ax.set_xticks(dates_last_60[::5])  # Show every 5th date
        ax.set_xticklabels(dates_last_60[::5], rotation=45, ha='right')

        ax.set_xlabel("Date")
        ax.set_ylabel(f"Stock Price ({currency})")
        ax.set_title(f"{stock_symbol} - Actual vs LSTM Prediction")
        ax.legend()
        st.pyplot(fig)


        # Plot Actual vs MAE (Last 60 Days)
        st.subheader("Actual vs MAE Prediction (Last 60 Days) 📊")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates_last_60, Y_test_actual[-60:], label="Actual Price", color="green", linestyle="--", linewidth=2)
        ax.plot(dates_last_60, mae_predictions[-60:], label="MAE Predicted Price", color="blue", linewidth=2)

        # Formatting x-axis ticks (Show fewer dates)
        ax.set_xticks(dates_last_60[::5])  # Show every 5th date
        ax.set_xticklabels(dates_last_60[::5], rotation=45, ha='right')

        ax.set_xlabel("Date")
        ax.set_ylabel(f"Stock Price ({currency})")
        ax.set_title(f"{stock_symbol} - Actual vs MAE Prediction")
        ax.legend()
        st.pyplot(fig)

        
        # Next 30 days forecast
        future_dates = pd.date_range(start=today, periods=30).strftime('%B %d, %Y')

        # Plot LSTM vs MAE for Next 60 Days
        st.subheader("LSTM vs MAE Prediction (Next 30 Days) 📊")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(future_dates, lstm_predictions[-30:], label="LSTM Predicted Price", color="red", linewidth=2)
        ax.plot(future_dates, mae_predictions[-30:], label="MAE Predicted Price", color="blue", linewidth=2)
        ax.set_xlabel("Date")
        ax.set_ylabel(f"Stock Price ({currency})")
        ax.set_title(f"{stock_symbol} - LSTM vs MAE Prediction (Next 30 Days)")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Show latest predicted values
        st.write("### Latest Predictions")
        st.write(f"📌 **LSTM Model Prediction:** {currency} {lstm_predictions[-1][0]:,.2f}")
        st.write(f"📌 **MAE Model Prediction:** {currency} {mae_predictions[-1][0]:,.2f}")

        # Model Performance Evaluation
        lstm_mae = mean_absolute_error(Y_test_actual, lstm_predictions)
        mae_mae = mean_absolute_error(Y_test_actual, mae_predictions)

        # Display the best model suggestion
        if lstm_mae < mae_mae:
            st.success(f"🚀 **LSTM Model** performed better with a lower MAE of {lstm_mae:.4f}.")
        else:
            st.success(f"🚀 **MAE Model** performed better with a lower MAE of {mae_mae:.4f}.")
        
        st.warning("**Note:** For better confirmation, analyze market statistics, economic data, and global trends before making decisions.")
