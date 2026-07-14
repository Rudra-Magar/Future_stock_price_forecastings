# Architecture A: LSTM Financial Forecasting System (Version 1 → Version 2)

## Project Goal

Build a lightweight financial forecasting system that evolves from an
experimental notebook into a deployable Streamlit application.

## Problem Being Solved

Most tutorial stock projects evaluate historical predictions and present
them as future forecasts. This project separates: - historical model
evaluation, - true future forecasting, - deployment engineering.

## Version 1: Experimental Study

### Research Question

How do MSE and MAE loss functions affect LSTM financial forecasting
performance?

Pipeline: Historical Data → Close Price → Scaling → Train/Test Split →
Sliding Window → LSTM(MSE) → LSTM(MAE) → Evaluation

### Findings

-   MSE punished large mistakes more aggressively.
-   MAE produced smoother predictions.
-   MSE achieved better MAE, RMSE, MAPE, R² and EVS scores.

### Limitations

-   One-step historical prediction only.
-   No true future forecasting.
-   Historical predictions could be mistaken for future forecasts.

## Historical Prediction vs Forecasting

Version 1: Real Historical Values → Predict Next Historical Value

Version 2: Real Values → Predict Future Value → Feed Prediction Back →
Predict Again

## Version 2 Improvements

-   Reduced time_step from 100 to 60.
-   Reduced LSTM units from 50 to 32.
-   Added recursive multi-step forecasting.
-   Optimized for Streamlit deployment.

## Recursive Forecasting Workflow

Last 60 Days → Predict Day 61 → Append Prediction → Predict Day 62 →
Repeat

## Version Comparison

  Feature                               V1    V2
  ------------------------------------- ----- ---------
  Uses real future historical values    Yes   No
  Uses previous predictions as inputs   No    Yes
  Suitable for evaluation               Yes   Limited
  Suitable for real forecasting         No    Yes

## Deployment Workflow

Train Model → Save .keras model → Save scaler.pkl → Load in Streamlit →
Generate forecast

## Lessons Learned

-   Prediction and forecasting are different problems.
-   Time-series data should not be shuffled.
-   Error metrics are not accuracy metrics.
-   Deployment constraints matter.

## Future Roadmap

Architecture B: - RSI - MACD - Volume - Moving averages

Architecture C: - News sentiment - LLM reasoning - AI investment reports