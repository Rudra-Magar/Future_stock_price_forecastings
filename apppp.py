import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
from pathlib import Path

# -----------------------------------------------------
# Load Models and Scaler
# -----------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

try:
    lstm_model = tf.keras.models.load_model(
        BASE_DIR / "lstm_model.keras"
    )

    mae_model = tf.keras.models.load_model(
        BASE_DIR / "mae_model.keras"
    )

    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# -----------------------------------------------------
# Fetch Stock Data
# -----------------------------------------------------

def fetch_stock_data(stock, start, end):
    df = yf.download(
        stock,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )
    return df


# -----------------------------------------------------
# Preprocess Data
# -----------------------------------------------------

def preprocess_data(data, time_step=100):

    data_scaled = scaler.transform(data)

    X = []
    Y = []

    # Prevent empty sequence creation
    if len(data_scaled) <= time_step:
        return np.array([]), np.array([])

    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i - time_step:i, 0])
        Y.append(data_scaled[i, 0])

    return np.array(X), np.array(Y)


# -----------------------------------------------------
# Dates
# -----------------------------------------------------

today = datetime.today().date()
max_date = today - timedelta(days=1)
min_date = max_date - timedelta(days=365 * 10)


# -----------------------------------------------------
# Streamlit UI
# -----------------------------------------------------

st.title("Stock Price Prediction App 📈")

stock_symbol = st.text_input(
    "Enter Stock Ticker (e.g., AAPL):",
    "AAPL"
)

start_date = st.date_input(
    "Start Date",
    value=min_date,
    min_value=min_date,
    max_value=max_date
)

end_date = st.date_input(
    "End Date",
    value=max_date,
    min_value=min_date,
    max_value=max_date
)

if st.button("Predict"):

    st.write(
        f"Fetching data for {stock_symbol} "
        f"from {start_date} to {end_date}..."
    )

    df = fetch_stock_data(
        stock_symbol,
        start=start_date,
        end=end_date
    )

    # Check if data exists
    if df.empty or "Close" not in df.columns:
        st.error(
            "Invalid stock ticker or no data "
            "available for the selected date range."
        )
        st.stop()

    st.success("Data fetched successfully!")

    close_prices = df["Close"].values.reshape(-1, 1)

    X_test, Y_test = preprocess_data(close_prices)

    # Debug Information
    st.subheader("Debug Information")
    st.write("Close Prices Shape:", close_prices.shape)
    st.write("X_test Shape Before Reshape:", X_test.shape)
    st.write("Y_test Shape:", Y_test.shape)
    st.write("LSTM Model Input Shape:", lstm_model.input_shape)
    st.write("MAE Model Input Shape:", mae_model.input_shape)

    # Prevent IndexError
    if len(X_test) == 0:
        st.error(
            "Not enough historical data available.\n"
            "Please select a longer date range "
            "(minimum 100 trading days)."
        )
        st.stop()

    # -------------------------------------------------
    # Reshape for LSTM
    # -------------------------------------------------

    X_test = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1],
        1
    )

    st.write("X_test Shape After Reshape:", X_test.shape)

    # -------------------------------------------------
    # Predictions
    # -------------------------------------------------

    try:
        lstm_predictions = lstm_model.predict(X_test)
        mae_predictions = mae_model.predict(X_test)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # Convert back to original prices
    lstm_predictions = scaler.inverse_transform(
        lstm_predictions
    )

    mae_predictions = scaler.inverse_transform(
        mae_predictions
    )

    Y_test_actual = scaler.inverse_transform(
        Y_test.reshape(-1, 1)
    )

    # -------------------------------------------------
    # Currency
    # -------------------------------------------------

    currency = "$" if stock_symbol.upper() == "AAPL" else "Rs"

    # -------------------------------------------------
    # Last 60 Days Dates
    # -------------------------------------------------

    num_points = min(
        60,
        len(Y_test_actual),
        len(lstm_predictions),
        len(mae_predictions)
    )

    dates_last_60 = pd.date_range(
        end=today - timedelta(days=1),
        periods=num_points
    ).strftime('%B %d, %Y')

    # -------------------------------------------------
    # Actual vs LSTM
    # -------------------------------------------------

    st.subheader("Actual vs LSTM Prediction 📊")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        dates_last_60,
        Y_test_actual[-num_points:],
        label="Actual Price",
        color="green",
        linestyle="--",
        linewidth=2
    )

    ax.plot(
        dates_last_60,
        lstm_predictions[-num_points:],
        label="LSTM Prediction",
        color="red",
        linewidth=2
    )

    ax.set_xticks(dates_last_60[::5])
    ax.set_xticklabels(
        dates_last_60[::5],
        rotation=45,
        ha='right'
    )

    ax.set_xlabel("Date")
    ax.set_ylabel(f"Stock Price ({currency})")
    ax.set_title(
        f"{stock_symbol} - Actual vs LSTM Prediction"
    )

    ax.legend()

    st.pyplot(fig)

    # -------------------------------------------------
    # Actual vs MAE
    # -------------------------------------------------

    st.subheader("Actual vs MAE Prediction 📊")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        dates_last_60,
        Y_test_actual[-num_points:],
        label="Actual Price",
        color="green",
        linestyle="--",
        linewidth=2
    )

    ax.plot(
        dates_last_60,
        mae_predictions[-num_points:],
        label="MAE Prediction",
        color="blue",
        linewidth=2
    )

    ax.set_xticks(dates_last_60[::5])

    ax.set_xticklabels(
        dates_last_60[::5],
        rotation=45,
        ha='right'
    )

    ax.set_xlabel("Date")
    ax.set_ylabel(f"Stock Price ({currency})")
    ax.set_title(
        f"{stock_symbol} - Actual vs MAE Prediction"
    )

    ax.legend()

    st.pyplot(fig)

    # -------------------------------------------------
    # Next 30 Days Comparison
    # -------------------------------------------------

    future_points = min(
        30,
        len(lstm_predictions),
        len(mae_predictions)
    )

    future_dates = pd.date_range(
        start=today,
        periods=future_points
    ).strftime('%B %d, %Y')

    st.subheader(
        "LSTM vs MAE Prediction (Next 30 Days) 📊"
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        future_dates,
        lstm_predictions[-future_points:],
        label="LSTM Prediction",
        color="red",
        linewidth=2
    )

    ax.plot(
        future_dates,
        mae_predictions[-future_points:],
        label="MAE Prediction",
        color="blue",
        linewidth=2
    )

    ax.set_xlabel("Date")
    ax.set_ylabel(f"Stock Price ({currency})")

    ax.set_title(
        f"{stock_symbol} - Future Predictions"
    )

    ax.legend()

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # -------------------------------------------------
    # Latest Predictions
    # -------------------------------------------------

    st.subheader("Latest Predictions")

    st.write(
        f"📌 LSTM Prediction: "
        f"{currency} {lstm_predictions[-1][0]:,.2f}"
    )

    st.write(
        f"📌 MAE Prediction: "
        f"{currency} {mae_predictions[-1][0]:,.2f}"
    )

    # -------------------------------------------------
    # Model Evaluation
    # -------------------------------------------------

    lstm_mae = mean_absolute_error(
        Y_test_actual,
        lstm_predictions
    )

    mae_mae = mean_absolute_error(
        Y_test_actual,
        mae_predictions
    )

    if lstm_mae < mae_mae:
        st.success(
            f"LSTM model performed better "
            f"with MAE = {lstm_mae:.4f}"
        )
    else:
        st.success(
            f"MAE model performed better "
            f"with MAE = {mae_mae:.4f}"
        )

    st.warning(
        "Predictions should not be used as the "
        "sole basis for investment decisions."
    )