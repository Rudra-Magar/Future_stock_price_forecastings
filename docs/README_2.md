# Architecture A - Version 2

## Productionizing LSTM Financial Forecasting

---

# Objective

The goal of Architecture A Version 2 is not to improve prediction accuracy alone.

The objective is to transform a tutorial-level LSTM stock prediction model into a lightweight, production-ready financial forecasting system suitable for deployment on Streamlit and GitHub.

---

# Problems Identified in Version 1

## Problem 1: False Future Forecasting

Version 1 displayed historical test predictions as future forecasts.

Example:

Historical Test Data
↓
Model Predictions
↓
Displayed as "Next 30 Days Forecast"

This is incorrect because the model already had access to the historical context for those predictions.

The model was performing:

One-Step Ahead Prediction

rather than

True Multi-Step Forecasting

---

## Problem 2: Invalid Accuracy Metric

Version 1 used:

Accuracy = 100 - MAPE

This is mathematically incorrect.

MAPE is an error metric rather than a classification accuracy metric.

Version 2 removes this metric entirely.

---

## Problem 3: No Overfitting Monitoring

Version 1 trained for:

50 epochs

without monitoring whether the model had already converged.

This can lead to:

* Overfitting
* Wasted computation
* Higher CPU usage
* Longer training time

---

## Problem 4: No Deployment Optimization

Every Streamlit interaction caused:

* model reload
* scaler reload
* repeated data downloads

This unnecessarily increases memory usage and CPU consumption.

---

# Improvements Introduced

## 1. Early Stopping

Early stopping monitors validation loss.

Training automatically stops once the model stops improving.

Benefits:

* Lower training time
* Reduced overfitting
* Lower CPU consumption
* Better deployment experience

---

## 2. Cached Data Downloads

Yahoo Finance downloads are cached.

Benefits:

* Faster interaction
* Lower API calls
* Better user experience

---

## 3. Cached Model Loading

Models are loaded only once during application startup.

Benefits:

* Faster Streamlit reruns
* Lower memory usage
* Better scalability

---

## 4. Training History Visualization

Version 2 records:

* Training Loss
* Validation Loss

This allows visualization of:

* convergence behavior
* underfitting
* overfitting

---

## 5. Recursive Forecasting

Version 2 introduces true future forecasting.

Pipeline:

Last 100 Known Days
↓
Predict Day 101
↓
Append Prediction
↓
Predict Day 102
↓
Append Prediction
↓
Repeat

This method is known as:

Recursive Multi-Step Forecasting

---

## 6. Confidence Intervals

Financial forecasting should not provide a single deterministic value.

Instead of:

Tomorrow Price = $220

Version 2 provides:

Tomorrow Price = $220 ± $7

This better reflects uncertainty in financial markets.

---

# Updated Pipeline

Historical Market Data
↓
Close Price Extraction
↓
Normalization
↓
Train Test Split
↓
Sliding Window Generation
↓
LSTM Model (MSE Loss)
↓
LSTM Model (MAE Loss)
↓
Evaluation Metrics
↓
Historical Predictions
↓
Recursive Forecasting
↓
Confidence Interval Estimation
↓
Deployment Optimization
↓
Streamlit Dashboard

---

# Engineering Lessons Learned

* Time series data must never be shuffled.
* Prediction is not the same as forecasting.
* Error metrics are not accuracy metrics.
* Deployment considerations are part of machine learning engineering.
* Model quality and software quality are equally important.

---

# Next Stage

Architecture B introduces:

* RSI
* MACD
* Moving Averages
* Volume Analysis
* Multi-Feature Inputs

Architecture C introduces:

* News Analysis
* Sentiment Analysis
* LLM Reasoning
* AI Generated Investment Reports
* Decision Support System
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++okay i mistakely write the mse as lstm now its fine now tell me one thing can i use the AMZN in the ticker name and get. the forecast of it ? 
as i cann not take the data on my own whats with the date like :

df = yf.download(
    ticker,
    start="2015-01-01"
)
does this means the model is being trained form 2015 data til today which is july13 2026 i don not understand.

Yes, you can absolutely use:

ticker = "AMZN"

or:

ticker = "TSLA"
ticker = "MSFT"
ticker = "GOOGL"
ticker = "NVDA"
ticker = "META"
ticker = "NFLX"

because yfinance downloads the data directly from Yahoo Finance.

Now for the important part:

What exactly is happening here?
df = yf.download(
    ticker,
    start="2015-01-01"
)

Suppose today is:

July 13, 2026

and you choose:

ticker = "AMZN"

Then yfinance downloads:

AMZN historical prices

2015-01-01
2015-01-02
2015-01-05
...
...
2026-07-13

So yes:

The model is trained on all historical data from January 1, 2015 until today.

Then what is the train/test split doing?

Suppose you downloaded:

2015 → 2026

which gives approximately:

2900 trading days

Your code does:

train_size = int(len(data_scaled) * 0.8)

Therefore:

Training Data:
2015 → approximately early 2024

Testing Data:
early 2024 → July 2026

Example:

2015 -------------------- 2024 -------------------- 2026
|----------------80%------|---------20%------------|
         TRAIN                        TEST
During training what does the model see?

The model only sees:

2015 → 2024

during training.

It never sees:

2024 → 2026

during training.

During testing what happens?

Suppose the test set contains:

Day 1 -> 180
Day 2 -> 181
Day 3 -> 182
...
Day 60 -> 195
Day 61 -> ?

The model receives:

180,181,182,...,195

and predicts:

Day 61

Then for the next prediction in the test set it receives:

181,182,...,195,196

where:

196

is the real historical value, not the predicted one.

This is the Version 1 historical evaluation we discussed earlier.

What happens during recursive forecasting?

After training finishes:

The model takes:

Last 60 real days available:
May 2026 → July 2026

and predicts:

July 14, 2026

Then:

July 14 prediction

is fed back into the model to predict:

July 15, 2026

Then:

July 15 prediction

is used to predict:

July 16, 2026

and so on.

So the recursive forecast looks like:

Real data:
2015 -------------------- July 13, 2026

Prediction:
July 14, 2026
July 15, 2026
July 16, 2026
...
August 12, 2026

This is a genuine future forecast because those future prices do not exist yet.

Does changing ticker retrain the model?

With your current app.py:

No.

The model loaded is:

model = load_model("mse_model.keras")

That model was trained only on:

AAPL

Therefore:

AMZN
TSLA
NVDA

would produce predictions, but they are not meaningful because the model learned Apple's behavior.

This is actually one of the biggest limitations of Architecture A Version 2.

Proper ways to support multiple stocks
Option 1 (simple)

Train one model per stock:

aapl_model.keras
amzn_model.keras
nvda_model.keras

and load based on ticker.

Good for portfolios.

Option 2 (better)

When user selects:

AMZN

download AMZN data and retrain the model.

This works but takes:

20-60 seconds

which is not ideal for Streamlit.

Option 3 (best long-term)

Train a universal model using:

AAPL
AMZN
TSLA
MSFT
NVDA
META
GOOGL
...

and include ticker embeddings/features.

This is closer to how professional forecasting systems work.

For Architecture A Version 2, the recommendation is:

AAPL model -> predicts AAPL
AMZN model -> predicts AMZN

Simple, fast, deployable, and portfolio-friendly.

The next architectural jump (Architecture B) is where we move toward multi-stock models and additional features like volume, RSI, MACD, and moving averages.




-------------------------------------------------------------------------------------------------------
ONE OF THE MAJOR PROBLEM IS THAT THE WHOLE MODEL IS TRAINED ON THE APPL DATA ONLY SO THE MORE LIKELY WE CAN SEE SIMILAR PREDICTION LIKE OF THE APPL EVEN OTHER TICKERS ARE USED.