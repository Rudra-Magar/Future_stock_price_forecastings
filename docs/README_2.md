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
