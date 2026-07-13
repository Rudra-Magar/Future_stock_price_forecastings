import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pickle

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------
# Streamlit Page Config
# -----------------------------------

st.set_page_config(
    page_title="Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 LSTM Stock Forecast Dashboard")
st.write(
    "Recursive Multi-Step Stock Forecasting using LSTM"
)

# -----------------------------------
# Load Saved Model and Scaler
# -----------------------------------

@st.cache_resource
def load_saved_model():
    return load_model("mse_model.keras")

@st.cache_resource
def load_saved_scaler():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

model = load_saved_model()
scaler = load_saved_scaler()

# -----------------------------------
# Sidebar Inputs
# -----------------------------------

ticker = st.sidebar.text_input(
    "Stock Ticker",
    value="AAPL"
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=7,
    max_value=60,
    value=30
)

time_step = 60

# -----------------------------------
# Download Latest Data
# -----------------------------------

df = yf.download(
    ticker,
    start="2015-01-01"
)

if df.empty:
    st.error("Invalid ticker symbol.")
    st.stop()

close_prices = df["Close"].values.reshape(-1,1)

scaled_data = scaler.transform(close_prices)

# -----------------------------------
# Recursive Forecasting
# -----------------------------------

future_predictions = []

last_sequence = scaled_data[-time_step:]
current_sequence = last_sequence.reshape(
    1,
    time_step,
    1
)

for _ in range(forecast_days):

    next_prediction = model.predict(
        current_sequence,
        verbose=0
    )

    future_predictions.append(
        next_prediction[0,0]
    )

    current_sequence = np.append(
        current_sequence[:,1:,:],
        [[[next_prediction[0,0]]]],
        axis=1
    )

future_predictions = scaler.inverse_transform(
    np.array(
        future_predictions
    ).reshape(-1,1)
)

# -----------------------------------
# Future Dates
# -----------------------------------

future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=forecast_days,
    freq="B"
)

forecast_df = pd.DataFrame(
    {
        "Date": future_dates,
        "Forecast Price": future_predictions.flatten()
    }
)

# -----------------------------------
# Metrics Section
# -----------------------------------

st.subheader("Forecast Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Current Price",
        f"${close_prices[-1][0]:.2f}"
    )

with col2:
    st.metric(
        "30-Day Forecast End",
        f"${future_predictions[-1][0]:.2f}"
    )

with col3:
    change = (
        (
            future_predictions[-1][0]
            - close_prices[-1][0]
        )
        / close_prices[-1][0]
    ) * 100

    st.metric(
        "Expected Change",
        f"{change:.2f}%"
    )

# -----------------------------------
# Visualization
# -----------------------------------

st.subheader("Historical Price + Forecast")

fig, ax = plt.subplots(
    figsize=(14,6)
)

ax.plot(
    df.index[-120:],
    close_prices[-120:],
    label="Historical Price"
)

ax.plot(
    future_dates,
    future_predictions,
    label="Forecast"
)

ax.set_title(
    f"{ticker} Recursive Forecast"
)

ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")
ax.legend()

st.pyplot(fig)

# -----------------------------------
# Forecast Table
# -----------------------------------

st.subheader("Forecast Table")

st.dataframe(
    forecast_df,
    use_container_width=True
)

# -----------------------------------
# Download CSV
# -----------------------------------

csv = forecast_df.to_csv(
    index=False
)

st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name=f"{ticker}_forecast.csv",
    mime="text/csv"
)