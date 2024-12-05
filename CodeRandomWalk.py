import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import datetime as dt

# ------------------ Utility Functions ------------------

def validate_date(prompt, default=None, prices=None):
    """
    Validates a date entered by the user.
    If `prices` is provided, ensures the date is a valid trading day in the data.
    """
    while True:
        try:
            date_input = input(f"{prompt} (default: {default}) : ").strip()
            if not date_input and default:
                date_input = default
            date = dt.datetime.strptime(date_input, "%Y-%m-%d").date()

            if prices is not None and date not in pd.to_datetime(prices.index).date:
                print(f"Error: The date {date} must be a valid trading day in the data.")
                continue

            return date
        except ValueError:
            print("Error: The entered date is invalid. Please follow the YYYY-MM-DD format.")

def validate_date_range(prompt_start, prompt_end, default_start=None, default_end=None, prices=None):
    """
    Validates a date range entered by the user with default values.
    """
    while True:
        start_date = validate_date(prompt_start, default_start, prices)
        end_date = validate_date(prompt_end, default_end, prices)

        if start_date > end_date:
            print("Error: The end date must be greater than or equal to the start date.")
            continue

        return start_date, end_date

def validate_ticker(default="AAPL"):
    """
    Validates the stock ticker by checking if it contains data.
    """
    while True:
        ticker = input(f"Enter the stock ticker (e.g., AAPL, MSFT, TSLA) (default: {default}) : ").strip().upper()
        ticker = ticker if ticker else default
        try:
            stock = yf.Ticker(ticker)
            prices = stock.history(period="1d")  # Minimal verification
            if prices.empty:
                print(f"Error: The ticker '{ticker}' does not contain valid data.")
                continue
            return ticker
        except Exception as e:
            print(f"Error: Unable to validate the ticker '{ticker}'. Details: {e}")

# ------------------ Main Functions ------------------

def download_prices(ticker, start_date, end_date):
    """
    Downloads closing price data for a given time period.
    Handles missing values.
    """
    print(f"\nDownloading data for {ticker} from {start_date} to {end_date}...")
    prices = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))[['Close']]
    if prices.empty:
        raise ValueError(f"No data downloaded for {ticker}. Check the dates or the ticker.")
    if prices.isnull().values.any():
        print("Warning: Missing values detected. They will be replaced using ffill() and bfill().")
        prices = prices.ffill().bfill()
    print(f"Data successfully downloaded for {ticker}.\n")
    return prices

def calculate_stat_returns(prices, start_sub_period=None, end_sub_period=None):
    """
    Calculates logarithmic returns, their mean, and their standard deviation for a given period.
    """
    if start_sub_period and end_sub_period:
        prices = prices.loc[start_sub_period:end_sub_period]
    if prices.empty:
        raise ValueError(f"No data available for the period {start_sub_period} to {end_sub_period}.")
    returns = np.log(prices['Close']).diff().dropna()
    return returns, returns.mean(), returns.std()

def monte_carlo_simulation(ticker, prices, nb_simulation, nb_step, initial_date, initial_price, mean, std):
    """
    Simulates future price trajectories of an asset using Monte Carlo and plots the results.
    """
    # Validate alignment of dates
    if pd.Timestamp(initial_date) not in prices.index:
        raise ValueError(f"initial_date ({initial_date}) is not a valid date in the price history.")

    # Generate simulation dates
    dates = pd.bdate_range(start=initial_date, periods=nb_step + 1, freq='B')
    simulations = np.zeros((nb_simulation, nb_step + 1))

    # Generate random walks
    for i in range(nb_simulation):
        random_walk = np.random.normal(mean, std, nb_step)
        simulations[i, :] = initial_price * np.exp(np.cumsum(np.insert(random_walk, 0, 0)))

    # Visualization
    plt.figure(figsize=(12, 8))
    plt.plot(prices.index, prices['Close'], color='red', linewidth=2, label=f'{ticker} Price History')
    plt.scatter(initial_date, initial_price, color='black', zorder=3, label="Simulation Start")
    for i in range(min(nb_simulation, 100)):
        plt.plot(dates, simulations[i, :], linewidth=1, alpha=0.5)
    plt.title(f"Monte Carlo Simulation for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.show()

    return simulations[:, -1], dates

def calculate_fair_price(mean_final_price, dates, risk_free_rate):
    """
    Calculates the fair price based on the final prices from the Monte Carlo simulations.
    
    final_prices: List of final prices from simulations
    dates: List of dates generated during the simulation
    risk_free_rate: Risk-free rate
    return: Fair price
    """
    t = (dates[-1] - dates[0]).days / 365  # Time in years
    fair_price = mean_final_price * np.exp(-risk_free_rate * t)
    return fair_price

# ------------------ Main Flow ------------------

# Define consistent default values
default_start_date = "2023-01-03"
default_end_date = date.today().strftime("%Y-%m-%d")  # Today's date as the default for data download
default_start_sub_period = "2023-01-03"
default_end_sub_period = "2024-03-01"
default_start_sim_period = "2024-03-04"
default_end_sim_period = date.today().strftime("%Y-%m-%d")
default_nb_simulation = 10
default_risk_free_rate = 0.03

print("Welcome to the random walk simulator. Define the simulation parameters:")

# Validate the ticker
ticker = validate_ticker()

# Validate the global period
start_date, end_date = validate_date_range(
    f"\nEnter the start date for downloading {ticker} price history:",
    f"Enter the end date for downloading {ticker} price history:",
    default_start_date,
    default_end_date
)

# Download price data
prices = download_prices(ticker, start_date, end_date)

# Validate the sub-period for statistical parameters
start_sub_period, end_sub_period = validate_date_range(
    f"Enter the start date for calculating {ticker} statistical parameters:",
    f"Enter the end date for calculating {ticker} statistical parameters:",
    default_start_sub_period,
    default_end_sub_period,
    prices
)

# Calculate returns
returns, mean, std = calculate_stat_returns(prices, start_sub_period, end_sub_period)
variance = std ** 2 
print(f"\nReturns calculated for the period {start_sub_period} to {end_sub_period}:")
print(f"Expected value (mean returns): {mean:.6f}")
print(f"Standard deviation of returns: {std:.6f}")
print(f"Variance of returns: {variance:.6f}")

# Validate the sub-period for random walks
start_sim_period = validate_date(
    f"\nEnter the start date for random walks:",
    default_start_sim_period,
    prices
)

nb_simulation = int(input(f"Enter the number of simulations to perform (default: {default_nb_simulation}): ") or default_nb_simulation)
risk_free_rate = float(input(f"Enter the risk-free rate (default: {default_risk_free_rate}): ") or default_risk_free_rate)

# Calculate the number of steps for the simulation
nb_step = len(pd.bdate_range(start=start_sim_period, end=end_date)) - 2  # -2 to exclude start and end dates

# Monte Carlo Simulation
final_prices, simulation_dates = monte_carlo_simulation(
    ticker=ticker,
    prices=prices,
    nb_simulation=nb_simulation,
    nb_step=nb_step,
    initial_date=start_sim_period,
    initial_price=prices.loc[pd.Timestamp(start_sim_period), 'Close'],
    mean=mean,
    std=std
)

mean_final_price = np.mean(final_prices)

# Calculate fair price
fair_price = calculate_fair_price(
    mean_final_price=mean_final_price,
    dates=simulation_dates,
    risk_free_rate=risk_free_rate
)

print(f"\nThe estimated fair price for {ticker} is: {fair_price:.2f}")