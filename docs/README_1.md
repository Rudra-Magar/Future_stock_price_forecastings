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


Issue: Are overcamed by the version -2
1. max data to download would be ranging betwen 100 days since that was our inital plan.
2. This project does NOT forecast future prices.
It predicts: historical unseen prices.
This is called: One-step ahead prediction on test data.
This distinction matters enormously.Our Streamlit app currently labels them as future predictions, which is misleading.

We will fix this in Architecture A version 2.
"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ADDED:

So the short answer is:

Feature	Version 1	Recursive Forecasting
Uses real historical future values	✅	❌
Uses previous predictions as inputs	❌	✅
Suitable for model evaluation	✅	❌
Suitable for real future forecasting	❌	✅
Error accumulation exists	Minimal	Significant

This distinction is exactly why Architecture A Version 2 exists.

i am still confues on the version 1 where you said it usses real historical future values mkae me more clarify as i am seeing no difference in the version 1-what it is called  and the recursive forcasting 

This confusion is extremely common. Even many people who build stock prediction projects don't realize the difference.

The easiest way is to completely ignore stocks for a moment and imagine predicting exam scores.

Suppose these are the real scores
Day 1 = 10
Day 2 = 12
Day 3 = 15
Day 4 = 16
Day 5 = 18
Day 6 = 20
Day 7 = 22
Day 8 = 25
Day 9 = 28
Day 10 = 30

Let's use:

time_step = 3
Version 1 (Your current notebook)

Training data contains all 10 days already.

The model sees:

Input:
10,12,15

Target:
16

Then:

Input:
12,15,16

Target:
18

Then:

Input:
15,16,18

Target:
20

Then:

Input:
16,18,20

Target:
22

and so on.

Notice something?

When predicting:

22

the model already knows:

16
18
20

because those values already existed in the dataset.

Visualizing Version 1

Suppose we are predicting Day 8.

The model receives:

Day 5 = 18
Day 6 = 20
Day 7 = 22

and predicts:

Day 8 = 25

Perfectly fine.

Now let's predict Day 9.

The model receives:

Day 6 = 20
Day 7 = 22
Day 8 = 25

Wait.

Where did:

Day 8 = 25

come from?

It came from the actual historical dataset.

Not from the model.

The model was given the real answer for Day 8.

Now predict Day 10.

Input:

Day 7 = 22
Day 8 = 25
Day 9 = 28

Again:

Day 8
Day 9

are real values from history.

The model never had to generate them itself.

Recursive Forecasting

Now imagine today is Day 7.

The future does not exist yet.

You know only:

10
12
15
16
18
20
22

You ask:

What is Day 8?

Model says:

24

(not 25 because models aren't perfect)

Now we want Day 9.

What data do we have?

18
20
22
24

Do we know the real Day 8?

No.

Day 8 hasn't happened yet.

So the model must use:

24

its own prediction.

Thus:

Input:
20
22
24

Predict:
26

Now predict Day 10.

Input:

22
24
26

Notice:

24
26

are not real data.

They are predictions.

The Difference in One Picture
Version 1
Real Data
↓
Predict Day 8

Real Data
↓
Predict Day 9

Real Data
↓
Predict Day 10

The model always receives real historical values.

Recursive Forecasting
Real Data
↓
Predict Day 8 = 24

Real Data + Prediction
↓
Predict Day 9 = 26

Prediction + Prediction
↓
Predict Day 10 = 27

The model must live with its own mistakes.

Why Version 1 looks so good

Because the model is always fed correct information.

Example:

Real Day 8 = 25
Real Day 9 = 28
Real Day 10 = 30

Those values are already available.

The model benefits from that information indirectly.

Why Recursive Forecasting usually looks worse

Suppose:

True Day 8 = 25
Model predicts Day 8 = 24

Small error:

-1

Now Day 9 prediction uses:

20
22
24

instead of:

20
22
25

Suppose it predicts:

26

True value was:

28

Error becomes:

-2

Now Day 10 uses:

22
24
26

instead of:

22
25
28

Prediction:

27

True value:

30

Error becomes:

-3

This is called:

Error accumulation.

Your current project is doing this:
Real 1-100
↓
Predict 101

Real 2-101
↓
Predict 102

Real 3-102
↓
Predict 103

Notice:

101
102

are real historical prices from Yahoo Finance.

Recursive forecasting would do:

Real 1-100
↓
Predict 101 = 210

Real 2-99 + 210
↓
Predict 102 = 212

Real 3-98 + 210 + 212
↓
Predict 103 = 214

Those last values are generated by the model itself.

This is why we say:

Version 1 is evaluating how good the model is on unseen historical data.

while

Recursive forecasting is trying to predict a future that does not exist yet.

Those are related tasks, but they are not the same problem.

No.

Your understanding is very close, but the important correction is:

Version 1 is still predicting, not pasting.

The difference is that Version 1 predicts using real historical values as inputs, whereas recursive forecasting predicts using its own previous predictions as inputs, causing errors to accumulate over time.

okay lets we go agin so :
for example the start date is from august 1 2015 to august 1 2026 now the version 1 using the real historical values not the predcted ones does means we are predicting the 30+days from the august 1 2026 to august 30 2026 but the model does have the historical data of till august 30 2026 if this is the case then what it is doing and also in the recurssive version the prediction are acumulated and whcih are used to predict the next day - i understood the recurssive one but lost in the verion -1 

You are extremely close now. The confusion comes from one thing:

The app is lying about the dates.

Let's use your exact example.

Dataset available

Suppose Yahoo Finance gives us:

Aug 1 2015
...
Jul 30 2026
Jul 31 2026
Aug 1 2026
Aug 2 2026
...
Aug 30 2026

Assume today is actually:

Aug 30 2026

and therefore Yahoo Finance already contains all these values.

What Version 1 does

Suppose:

time_step = 100

To predict:

Aug 1 2026

the model receives:

Apr 23 2026
Apr 24 2026
...
Jul 31 2026

and predicts:

Aug 1 2026

Good.

Then to predict:

Aug 2 2026

the model receives:

Apr 24 2026
Apr 25 2026
...
Aug 1 2026

Notice:

Aug 1 2026

is the real value from Yahoo Finance.

The model did not predict it.

Yahoo Finance already knew it.

Then to predict:

Aug 3 2026

the input becomes:

Apr 25 2026
...
Aug 1 2026
Aug 2 2026

Again:

Aug 1 2026
Aug 2 2026

are real historical values.

So Version 1 is actually doing:

Known History
↓
Predict Next Known Historical Day
↓
Known History
↓
Predict Next Known Historical Day
↓
Known History

This is called:

Historical backtesting

or

One-step ahead prediction on test data.

Where the confusion happens

Your Streamlit app does this:

future_dates = pd.date_range(start=today, periods=30)
ax.plot(future_dates, lstm_predictions[-30:])

The code says:

Aug 1 2026
Aug 2 2026
...
Aug 30 2026

but the predictions being plotted are actually:

historical predictions from the test set

not future predictions.

This is why I called it:

False future forecasting.

The dates are fake labels.

What Recursive Forecasting would do

Suppose today is:

Aug 1 2026

and Yahoo Finance only knows prices until:

Aug 1 2026

The model predicts:

Aug 2 2026 = 250

Now we want Aug 3.

Problem:

Aug 2 does not exist yet.

So we use:

Predicted Aug 2 = 250

as input.

Then:

Predict Aug 3 = 251

Now predict Aug 4:

Input includes:

Predicted Aug 2
Predicted Aug 3

not real values.

The simplest sentence
Version 1

"How well can my model predict prices that already happened but were hidden from the model during training?"

Recursive Forecasting

"What do I think prices will be for days that have not happened yet?"

That single distinction separates model evaluation from actual forecasting.

Your Version 1 model is excellent for evaluation.

Your Version 2 recursive model will be for forecasting.

  okay one line feeback :
version 1- did predict but wht the real historical values not the actual prediction
version 2- did predict without the real historical values aka tha ctual prediction
does this menns the recurssive forecasting is sutiable and also what in the both model are being tradined with as you said at the end

Your understanding is almost correct, with one small correction:

Version 1: ✅ Did predict, but each prediction used real historical values from Yahoo Finance as the next inputs, not previous predictions.
Version 2 (Recursive Forecasting): ✅ Did predict, and each prediction uses previous predictions as future inputs, because the real future values do not exist yet.
Is recursive forecasting more suitable?

For actual future forecasting: ✅ Yes

Because in reality, when predicting tomorrow's stock price, you obviously do not have tomorrow's actual price available.

Then why train with Version 1 style?

Because:

Version 1 is used for model evaluation and training validation.
Version 2 is used after training for actual forecasting.

Think of it like this:

Training and Evaluation:
Real Historical Data
↓
One-Step Prediction (Version 1)
↓
Measure MAE, RMSE, R²

After you're satisfied with the model:

Deployment and Future Forecasting:
Last Known 100 Days
↓
Recursive Forecasting (Version 2)
↓
Next 30 Days Forecast

So in practice:

Stage	Method Used
Model Training	Historical one-step prediction
Model Evaluation	Historical one-step prediction
Real-world Future Forecast	Recursive forecasting