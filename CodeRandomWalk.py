import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import datetime as dt

def validate_ticker(default="AAPL"):
    while True:
        ticker = input(f"Entrez le ticker (par ex. AAPL, MSFT, TSLA) (par défaut : {default}) : ").strip().upper()
        ticker = ticker if ticker else default
        stock = yf.Ticker(ticker)
        stock.history(period="1d")
        prices = stock.history(period="1d")
        if not prices.empty:
            return ticker  
        print(f"Le ticker {ticker} n'est pas valide. Veuillez réessayer.")

def validate_date(prompt, default=None, prices=None):
    while True:
        try:
            date_input = input(f"{prompt} (par défaut : {default}) : ").strip()
            date_input = date_input if date_input else default
            date = dt.datetime.strptime(date_input, "%Y-%m-%d").date()
            if prices is not None and date not in pd.to_datetime(prices.index).date:
                print(f"Erreur : La date {date} doit être un jour de trading valide dans les données.")
                continue
            return date
        except ValueError:
            print("Erreur : La date saisie est invalide. Veuillez respecter le format AAAA-MM-JJ.")

def validate_date_range(prompt_start, prompt_end, default_start=None, default_end=None, prices=None):
    while True:
        start_date = validate_date(prompt_start, default_start, prices)
        end_date = validate_date(prompt_end, default_end, prices)
        if start_date > end_date:
            print("Erreur : La date de fin doit être postérieure ou égale à la date de début.")
            continue
        return start_date, end_date

def download_prices(ticker, start_date, end_date):
    print(f"Téléchargement des données pour {ticker} de {start_date} à {end_date}...")
    prices = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))[['Close']]
    if prices.empty:
        raise ValueError(f"Aucune donnée téléchargée pour {ticker}. Vérifiez les dates ou le ticker.")
    if prices.isnull().values.any():
        print("Attention : Des valeurs manquantes ont été détectées. Elles seront remplacées par ffill() et bfill().")
        prices = prices.ffill().bfill()
    print(f"Données téléchargées avec succès pour {ticker}.")
    return prices

def calculate_stat_returns(prices, start_sub_period=None, end_sub_period=None):
    if start_sub_period and end_sub_period:
        prices = prices.loc[start_sub_period:end_sub_period]
    if prices.empty:
        raise ValueError(f"Aucune donnée disponible pour la période {start_sub_period} à {end_sub_period}.")
    returns = np.log(prices['Close']).diff().dropna()
    return returns, returns.mean(), returns.std()

def monte_carlo_simulation(ticker, prices, nb_simulation, nb_step, initial_date, initial_price, mean, std):
    if pd.Timestamp(initial_date) not in prices.index:
        raise ValueError(f"initial_date ({initial_date}) n'est pas une date valide dans l'historique des prix.")

    dates = pd.bdate_range(start=initial_date, periods=nb_step + 1, freq='B')
    simulations = np.zeros((nb_simulation, nb_step + 1))

    for i in range(nb_simulation):
        random_walk = np.random.normal(mean, std, nb_step)
        simulations[i, :] = initial_price * np.exp(np.cumsum(np.insert(random_walk, 0, 0)))

    plt.figure(figsize=(12, 8))
    plt.plot(prices.index, prices['Close'], color='red', linewidth=2, label=f'Historique des prix de {ticker}')
    plt.scatter(initial_date, initial_price, color='black', zorder=3, label="Début des simulations")

    for i in range(min(nb_simulation, 100)):
        plt.plot(dates, simulations[i, :], linewidth=1, alpha=0.5)
        
    plt.title(f"Simulation Monte Carlo pour {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid()
    plt.show()

    return simulations[:, -1], dates

def calculate_fair_price(mean_final_price, dates, risk_free_rate):
    t = (dates[-1] - dates[0]).days / 365
    fair_price = mean_final_price * np.exp(-risk_free_rate * t)
    return fair_price

# Date du jour
today = date.today().strftime("%Y-%m-%d")

# Plage téléchargement de l'historique
default_start_date = "2023-01-03"
default_end_date = today

# Plage pour les calculs des paramètres statistiques
default_start_sub_period = "2023-01-03"
default_end_sub_period = "2024-03-01"

# Plage pour les marches aléatoires
default_start_sim_period = "2024-10-04"
default_end_sim_period = today

# Nombre de marches aléatoires
default_nb_simulation = 10

# Taux sans risque 
default_risk_free_rate = 0.03

if __name__ == "__main__":
    print("Welcome to the random walk simulator. Define the simulation parameters:")
    ticker = validate_ticker()

    start_date, end_date = validate_date_range(
        f"Entrez la date de début du téléchargement de l'historique des prix de {ticker} :",
        f"Entrez la date de fin du téléchargement de l'historique des prix de {ticker} :",
        default_start_date,
        default_end_date
    )

    prices = download_prices(ticker, start_date, end_date)

    start_sub_period, end_sub_period = validate_date_range(
        f"Enter the start date for calculating {ticker} statistical parameters:",
        f"Enter the end date for calculating {ticker} statistical parameters:",
        default_start_sub_period,
        default_end_sub_period,
        prices
    )

    returns, mean, std = calculate_stat_returns(prices, start_sub_period, end_sub_period)
    variance = std ** 2

    print(f"Rendements calculés pour la période {start_sub_period} à {end_sub_period} :")
    print(f"Valeur attendue (moyenne des rendements) : {mean:.6f}")
    print(f"Écart-type des rendements : {std:.6f}")
    print(f"Variance des rendements : {variance:.6f}")

    start_sim_period = validate_date(
        f"\nEnter the start date for random walks:",
        default_start_sim_period,
        prices
    )

    nb_simulation = int(input(f"Entrez le nombre de simulations à effectuer (par défaut : {default_nb_simulation}) : ") or default_nb_simulation)
    risk_free_rate = float(input(f"Enter the risk-free rate (default: {default_risk_free_rate}): ") or default_risk_free_rate)
    nb_step = len(pd.bdate_range(start=start_sim_period, end=end_date)) - 1

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
    
    print(f"La moyenne des prix finaux des simulations est de {mean_final_price:.2f}")

    fair_price = calculate_fair_price(
        mean_final_price=mean_final_price,
        dates=simulation_dates,
        risk_free_rate=risk_free_rate
    )

    print(f"\nThe estimated fair price for {ticker} is: {fair_price:.2f}")