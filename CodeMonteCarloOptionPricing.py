import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tabulate as tb
import math
import datetime as dt

# ------------------ Utility Functions ------------------
def validate_ticker(default="AAPL"):
    """
    Validate the stock ticker by checking if it contains data.
    """
    while True:
        ticker = input(f"Enter the stock ticker (e.g., AAPL, MSFT, TSLA) (default: {default}) : ").strip().upper() or default
        try:
            stock = yf.Ticker(ticker)
            prices = stock.history(period="1d")
            if prices.empty:
                print(f"Error: The ticker '{ticker}' does not contain valid data.")
                continue
            return ticker
        except Exception as e:
            print(f"Error: Unable to validate the ticker '{ticker}'. Details: {e}")

def download_prices(ticker, start_date, end_date):
    """
    Download closing price data for a given time period. Handles missing values.
    """
    prices = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))[['Close']]
    if prices.empty:
        raise ValueError(f"No data downloaded for {ticker}. Check the dates or the ticker.")
    return prices.ffill().bfill()

def calculate_stat_returns(prices, start_sub_period, end_sub_period):
    """
    Calculate logarithmic returns, their mean, and standard deviation.
    """
    sub_prices = prices.loc[start_sub_period:end_sub_period]
    if sub_prices.empty:
        raise ValueError(f"No data available for the period {start_sub_period} to {end_sub_period}.")
    returns = np.log(sub_prices['Close']).diff().dropna()
    return returns, returns.mean(), returns.std()

def monte_carlo_simulation(S, K, T, r, sigma, nb_simulation=1000, option_type="call"):
    """
    Perform Monte Carlo simulation for option pricing.
    """
    dt = T / 252
    payoffs = []
    for _ in range(nb_simulation):
        price = S
        for _ in range(int(T * 252)):
            price *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())
        if option_type.lower() == "call":
            payoff = max(price - K, 0)
        elif option_type.lower() == "put":
            payoff = max(K - price, 0)
        payoffs.append(payoff)
    return np.mean(payoffs) * np.exp(-r * T)

def black_scholes(S, K, T, r, sigma, option_type):
    """
    Compute the price of a European option using Black-Scholes model.
    """
    if T <= 0:
        raise ValueError("Time to maturity (T) must be positive.")
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type.lower() == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

# ------------------ Main Code ------------------
if __name__ == "__main__":
    print("### Monte Carlo Simulation for European Option Pricing ###")

    # Step 1: Validate the ticker
    ticker = validate_ticker()
    stock = yf.Ticker(ticker)

    # Step 2: Download historical data
    today = dt.datetime.today()
    default_start_date = today - dt.timedelta(days=365)  # Default: past year
    prices = download_prices(ticker, default_start_date, today)

    # Step 3: Select an expiration date
    expiration_dates = stock.options
    while True:
        try:
            data_list = [[i + 1, date] for i, date in enumerate(expiration_dates)]
            headers = ["No.", "Expiration Date"]
            print(f"\nAvailable expiration dates for {ticker}:")
            print(tb.tabulate(data_list, headers=headers, tablefmt="fancy_grid"))
            selected_date = expiration_dates[int(input("Choose an expiration date (by No.) (default: 1): ") or 1) - 1]
            if pd.to_datetime(selected_date) <= today:
                raise ValueError("Selected expiration date must be in the future.")
            break
        except (IndexError, ValueError) as e:
            print(f"Error: {e}. Please enter a valid number.")

    # Step 4: Choose an option
    options = stock.option_chain(selected_date)
    while True:
        try:
            option_type = input("Option type (Call/Put) (default: Call): ").strip().lower() or "call"
            if option_type == "call":
                option_data = options.calls
            elif option_type == "put":
                option_data = options.puts
            else:
                print("Error: Please enter 'Call' or 'Put'.")
                continue

            option_data.reset_index(drop=True, inplace=True)
            option_data.insert(0, "No.", range(1, len(option_data) + 1))
            data_list = option_data.values.tolist()
            headers = option_data.columns.tolist()
            print("\nAvailable options for the selected type and date:")
            print(tb.tabulate(data_list, headers=headers, tablefmt="fancy_grid"))
            selected_option = option_data.iloc[int(input("Select an option (by No.) (default: 1): ") or 1) - 1]
            break
        except (IndexError, ValueError):
            print("Error: Please enter a valid number.")

    # Step 5: Calculate historical volatility and returns
    start_sub_period = default_start_date.strftime('%Y-%m-%d')
    end_sub_period = today.strftime('%Y-%m-%d')
    returns, mean, std = calculate_stat_returns(prices, start_sub_period, end_sub_period)

    # Step 6: Compute time to maturity (T)
    T = max((pd.to_datetime(selected_date) - today).days / 252, 0.001)

    # Step 7: Monte Carlo simulation
    print("\nRunning Monte Carlo simulation...")
    monte_carlo_price = monte_carlo_simulation(
        S=prices['Close'].iloc[-1],
        K=selected_option['strike'],
        T=T,
        r=0.03,  # Risk-free rate
        sigma=std,
        nb_simulation=1000,
        option_type=option_type
    )

    # Step 8: Black-Scholes calculation
    print("Calculating price using Black-Scholes model...")
    black_scholes_price = black_scholes(
        S=prices['Close'].iloc[-1],
        K=selected_option['strike'],
        T=T,
        r=0.03,
        sigma=std,
        option_type=option_type
    )

    # Step 9: Display results
    print("\n### Results ###")
    print(f"Market price of the selected option: {selected_option['lastPrice']:.2f} USD")
    print(f"Monte Carlo price: {monte_carlo_price:.2f} USD")
    print(f"Black-Scholes price: {black_scholes_price:.2f} USD")

    # Compare prices
    print("\nPrice comparison:")
    if monte_carlo_price > selected_option['lastPrice']:
        print("The option appears undervalued according to Monte Carlo.")
    else:
        print("The option appears overvalued according to Monte Carlo.")

    if black_scholes_price > selected_option['lastPrice']:
        print("The option appears undervalued according to Black-Scholes.")
    else:
        print("The option appears overvalued according to Black-Scholes.")