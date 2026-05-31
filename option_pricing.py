"""
inputs

df--> this is the data frame of the closing prices of an option
Type--> this is the type of the option the types are
    1)Call option
    2)Put option
S--> this is the price of the underlying stock 
k--> this is the strike price
r--> this is the risk free rate or the interest rate 
--------
output

the only output of this code is the price of the europian options (Call, Put) determined theoretically using black-scholes equation (Black-Scholes-Merton model)
"""
def option_pricing(df, Type, S, T, k, r):
    #imports

    import numpy as np
    import pandas as pd
    from arch import arch_model
    from scipy.stats import norm
    from datetime import datetime

    #calculate the number of days till expiry and the number of years to expiry
    
    expiry_date = datetime.strptime(T, "%Y-%m-%d")
    today = datetime.now()
    difference = expiry_date - today
    days_to_exp = max(difference.days,1)
    T_years = days_to_exp / 252

    #calculating the returns 

    returns = np.log(df/df.shift(1))
    returns = returns.dropna()

    #initialise the model and fit the model onto the returns

    model = arch_model(returns, p=1, q=1, vol="GARCH", dist='t')
    result = model.fit()

    #forecast the volatility for the future till the maturity 

    forecast = result.forecast(horizon = days_to_exp)
    daily_var = forecast.variance.iloc[-1]
    avg_daily_var = np.mean(daily_var)
    sigma = np.sqrt(avg_daily_var*252)

    #initialise the d1 and d2 for the cdf for the black scholes eqn and calculate the price of the options using the black schols eqn

    d1 = (np.log(S/k)+(r+(sigma**2/2))*T_years)/(sigma*np.sqrt(T_years))
    d2 = d1 - (sigma*np.sqrt(T_years))
    if Type == "Call":
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        call_option_price = (S*N_d1) - (k*np.exp(-r * T_years)*N_d2)
        return call_option_price
    elif Type == "Put":
        N_d1 = norm.cdf(-d1)
        N_d2 = norm.cdf(-d2)
        put_option_price = (k * np.exp(-r * T_years) * N_d2) - (S * N_d1)
        return put_option_price