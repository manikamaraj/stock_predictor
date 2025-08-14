import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data


def main():
    # User input or defaults
    ticker = input("Enter stock ticker (default: AAPL): ") or "AAPL"
    start_date = input("Enter start date (YYYY-MM-DD, default: 2020-01-01): ") or "2020-01-01"
    end_date = input("Enter end date (YYYY-MM-DD, default: 2023-01-01): ") or "2023-01-01"

    # Fetch data
    df = fetch_stock_data(ticker, start_date, end_date)
    if df.empty:
        print("No data found for the given ticker and date range.")
        return

    # Visualize closing price
    plt.figure(figsize=(10, 4))
    plt.plot(df['Close'])
    plt.title(f"{ticker} Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.show()

    # Prepare features: use previous day's close to predict next day's close
    df['PrevClose'] = df['Close'].shift(1)
    df = df.dropna()
    X = df[['PrevClose']].values
    y = df['Close'].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Plot results
    plt.figure(figsize=(10, 4))
    plt.plot(df.index[-len(y_test):], y_test, label='Actual')
    plt.plot(df.index[-len(y_test):], y_pred, label='Predicted')
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.legend()
    plt.show()

    # Print model performance
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"Mean Squared Error: {mse:.2f}")


if __name__ == "__main__":
    main() 