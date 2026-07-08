# Stock Price Prediction App 📈

## 📝 Overview
This project is an interactive, deployable Streamlit dashboard for forecasting stock prices using deep learning models. It includes trained models, data preprocessing tools, a notebook with the full ML pipeline, and instructions for running and deploying the app.

---

## 📁 Project Structure & File Explanations

```
THE PROOOO/
├── app.py                # Main Streamlit web app
├── requirements.txt      # List of required Python packages
├── REAL.ipynb            # (Optional) Notebook for all data science/modeling
├── lstm_model.keras      # Saved LSTM model for prediction
├── mae_model.keras       # Saved MAE model for prediction
├── scaler.pkl            # Pickle file for scaling inputs/outputs
```

### `app.py`
- **What:** Streamlit dashboard for real-time stock price forecasting.
- **Why:** Provides a user-friendly, visual web interface for the ML models.
- **How:**
  1. Loads pretrained LSTM and MAE Keras models, along with the scaler.
  2. Lets you select the stock ticker, start/end dates.
  3. Fetches price data (`yfinance`), scales data, prepares model input.
  4. Runs predictions, compares both methods.
  5. Plots results (actual vs. predicted) and shows model performance.

### `REAL.ipynb`
- **What:** Jupyter notebook detailing collection, cleaning, scaling, model training, and evaluation steps.
- **Why:** Reproducible pipeline for model development and experimentation.
- **How:** Uses Pandas, Scikit-learn, Matplotlib, Keras and others to fetch, prepare, and model stock data.

### `lstm_model.keras` / `mae_model.keras`
- **What:** Saved trained Keras models for inference (LSTM and MAE network architectures).
- **Why:** Fast predictions, no retraining needed.
- **How:** Loaded at runtime in `app.py` via `keras.models.load_model()`.

### `scaler.pkl`
- **What:** Pickled MinMaxScaler (or similar) object from training.
- **Why:** Ensures data is scaled the same way for inference as for model training—crucial for accurate results.
- **How:** Loaded with `pickle` in `app.py`.

### `requirements.txt`
- **What:** List of libraries and exact package versions needed.
- **Why:** Allows anyone to install dependencies and run the app the same way as the author.
- **How:** Install via `pip install -r requirements.txt`.

---

## 🖥️ Setup & Running Locally (Mac/Linux/Windows)

1. **Install Python 3.10 or later**
   - Recommended: use [python.org](https://www.python.org/downloads/) or `brew install python` (for Mac)
   - Or install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for full scientific stack

2. **Create & Activate a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
   *Or, with Conda:*
   ```bash
   conda create --name stock_predictor python=3.10
   conda activate stock_predictor
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *Minimal requirements.txt (recommended for sharing):*
   ```
   streamlit
yfinance
numpy
pandas
matplotlib
scikit-learn
tensorflow
keras
   ```
   *(Pin versions only for full reproducibility)*

4. **Run the Web App:**
   ```bash
   streamlit run app.py
   ```
   - Follow the link to `http://localhost:8501` in your browser.

---

## 🚀 Live Demo (Deploy for Free!)
This type of app must run on a Python server—**not** just static hosting like GitHub Pages.

### 1. [Streamlit Community Cloud](https://share.streamlit.io/)
The simplest way to show your app online!
- Push your cleaned repo (without prebuilt venv, but *with* models and code) to GitHub.
- Go to [Streamlit Cloud](https://share.streamlit.io/), log in with GitHub, and deploy—select main as `app.py`.
- Set any secrets/environment variables if needed (not required here).
- Share the generated public URL!

### 2. Other (For Large Model Files):
- If model files are too large for Streamlit Cloud limits, host on HuggingFace Spaces, Replit, or manually deploy on your own VM/server.
- For private model storage, use S3/Drive and update `app.py` to download on first run.

---

## 🧹 Cleaning/Optimizing the Repo
- **Delete** the `app/` directory (the included Python virtual environment).
- Only commit code, data, models, and documentation—not any `.venv`, `env/`, `.DS_Store`, etc.
- Add these to a `.gitignore`:
   ```
   venv/
   app/
   __pycache__/
   *.pyc
   .DS_Store
   *.ipynb_checkpoints
   ```

---

## 📣 Contributions & Issues
PRs and issues welcome! Please document changes clearly.


---

## 🙏 Acknowledgments
- Yahoo Finance for data
- Streamlit, Keras
- Myself
