from flask import Flask, request, render_template
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].upper()
    start_date = request.form['start']
    end_date = request.form['end']

    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return render_template('index.html', error="No data found. Please check the inputs.")

        df = df[['Close']].reset_index()
        df['Day'] = np.arange(len(df))

        X = df[['Day']].values
        y = df['Close'].values
        model = LinearRegression().fit(X, y)

        # Predict next 15 days
        future_days = np.arange(len(df), len(df)+15).reshape(-1, 1)
        predictions = model.predict(future_days)

        future_start = pd.to_datetime(end_date) + timedelta(days=1)
        future_end = future_start + timedelta(days=14)
        future_df = yf.download(
            ticker,
            start=future_start.strftime('%Y-%m-%d'),
            end=(future_end + timedelta(days=1)).strftime('%Y-%m-%d')
        )

        actual_prices = future_df['Close'].tolist()
        dates = future_df.index.strftime('%Y-%m-%d').tolist()

        # Prepare results (show 'N/A' if actual prices are not available)
        results = list(zip(
            dates[:len(predictions)],
            predictions[:len(dates)],
            actual_prices[:len(dates)] if len(actual_prices) else ['N/A'] * len(dates)
        ))

        # Calculate errors only if actuals are present
        if actual_prices:
            mae = mean_absolute_error(actual_prices[:len(dates)], predictions[:len(dates)])
            rmse = mean_squared_error(actual_prices[:len(dates)], predictions[:len(dates)], squared=False)
        else:
            mae = None
            rmse = None

        return render_template('index.html', results=results, mae=mae, rmse=rmse, ticker=ticker)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
