"""
(Comparative Study of MSE and MAE Loss Functions in LSTM-Based Financial Forecasting)

THIS IS FOR THE ARHITECTURE A -VERSION 1 

"What problem is this project actually solving?"
Current answer:
Predict Apple's future stock prices using historical closing prices.

Professional answer: Good story for GitHub.
1. Simply an experiment Comparing the impact of different loss functions (MSE vs MAE) on LSTM-based time series forecasting models for financial market data.
2. This project is actually more of an: Experimental study of loss functions in financial forecasting.
3. Here the only REAL.ipny file is explained because the main model is being trained here.

The Entire Pipeline:
Importing modules and packages
        ↓
Historical Stock Data
        ↓
Close Price Extraction
        ↓
Normalization
        ↓
Train-Test Split
        ↓
Sliding Window Creation
        ↓
LSTM Model A (MSE Loss)
        ↓
LSTM Model B (MAE Loss)
        ↓
Evaluation
        ↓
Deployment


1. Importing tools and packages.
import numpy as np
import pandas as pd

Library	Why?
numpy	          numerical computation
pandas	          dataframe operations
matplotlib        visualization
sklearn	          scaling and metrics
yfinance	  stock data
keras	          deep learning


2. Historical Stock Data (downloading)
stock='AAPL'
start_date='2015-01-01'
end_date='2024-01-01'

Why 2015?
Because LSTM needs lots of sequential data.

2015-2024 gives: ~2300 trading days which is reasonable.

df = yf.download(...)
Produces:

Date	Open	High	Low	Close	Volume


3. Close Price Extraction
data = df['Close']

This is your first modeling assumption.

You are saying:

The closing price contains enough information to predict future prices.

This assumption is weak.

But acceptable for Architecture A.

4. Normalization and Transformation
scaler = MinMaxScaler()

Transforms:
170
175
180
190

into:
0.00
0.25
0.50
1.00

Why?
Gradient descent performs better. Such to fit the data with the dimensions between 0 -1 and easy for model training


5. Train-Test Split
train_size = int(len(data_scaled) * 0.8)

This creates:80% training and 20% testing data 
This is correct for time series. Did NOT shuffled data.
Important:
A common beginner mistake is:

train_test_split(shuffle=True)    -> which leaks future information.

Here that is avoided.


6. Sliding window Creation

This is the heart of the project.

create_dataset()

Suppose prices are:

1
2
3
4
5
6
7
8
9
10

and

time_step = 3

Model sees:

1 2 3 -> predicts 4
2 3 4 -> predicts 5
3 4 5 -> predicts 6

This process is called: Sliding Window Sequence Generation. This is exactly how most forecasting systems work.

Why 100 days?
time_step=100

This means: Use the previous 100 trading days to predict the next day.

100 trading days ≈ 4.5 months

This is a hyperparameter.

Possible values:

30
60
90
120
180
252

252 is one trading year.


7. Shaping the datas for the LSTM coz it takes 3D input
reshape(samples,timestep,features)

LSTM requires 3D input.

Your shape becomes:

(1730,100,1)

meaning:

1730 samples
100 days memory
1 feature


8. Model Architecture
LSTM(50, return_sequences=True)

Output:100 hidden states

This allows the second LSTM to process the entire sequence.

LSTM(50, return_sequences=False)

Output: single compressed representation

Think:
100 days history
↓
compressed into market memory

Dense(25,relu)
Acts as a feature extraction layer.

Dense(1)
Produces: tomorrow price

Model structure:

100 prices
↓
LSTM
↓
LSTM
↓
Dense
↓
Dense
↓
Tomorrow Price


Why compare MSE and MAE?
This is the actual research question.

MODEL 1
loss='mean_squared_error'    ->  Punishes large mistakes aggressively.

Error:
1 -> 1
5 -> 25
10 -> 100

Large mistakes become very expensive.

Useful when: large forecasting errors are unacceptable.


MODEL 2
loss='mean_absolute_error'    -> Gives the absolute even when negative

Error:
1 -> 1
5 -> 5
10 -> 10

Every error treated equally.

Better with noisy markets.


9. Running the ecpochs
epochs=50

Means: entire training dataset shown 50 times

batch_size=32

Means: 32 examples processed before updating weights.


10. Saving the models
Saving models.

.save()

avoids retraining during deployment.


11. Evaluation metrics.

MAE = Average absolute error.
Example:
prediction = 100
actual = 110
error = 10


RMSE = Punishes large errors more heavily.

MAPE = Percentage error. (This is often preferred in finance.)
Example:
actual = 100
predicted = 105
5%


R² Score = Measures variance explained.
Range:
1.0 = perfect
0.0 = predicts mean
negative = terrible model
One Major Issue

THINGS TO CONSIDER WHEN EVALUATION

Example:
MAPE = 30%   -> does NOT imply  -> accuracy = 70%

NEEDED : 
MAE
RMSE
MAPE
R²


Issue: 
1. max data to download would be ranging betwen 100 days since that was our inital plan.
2. This project does NOT forecast future prices.
It predicts: historical unseen prices.
This is called: One-step ahead prediction on test data.
This distinction matters enormously.Our Streamlit app currently labels them as future predictions, which is misleading.

We will fix this in Architecture A version 2.
"""
