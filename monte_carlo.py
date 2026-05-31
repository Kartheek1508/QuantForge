"""
the input takes 

df--> the data frame of the closing price of a stock 
T--> this is the number of days we want to simulate the stock price
initial_price--> as the name suggests this is the initial price which we have invested in this stock 
-------
the function returns

expectation--> mean of the final prices from all the 10,000 universes simulated at the given number of days
voltality--> standard deviation of the final prices for all the universes 
losses--> this is the simple difference between the initial and the final price
VaR--> this is the value at risk or the bottom 5 percent cases where the market crashes and we lose more than the Value at risk
"""
def monte_carlo(df, T, initial_price):
    #imports

    import numpy as np
    import pandas as pd
    from arch import arch_model

    #calculate the returns

    returns = np.log(df/df.shift(1))
    returns = returns.dropna()

    #initialise the model and fit it 

    model = arch_model(returns, p=1, q=1, vol="GARCH", dist='t')
    result = model.fit()

    #extractimng the parameters

    nu = result.params['nu']
    omega = result.params['omega']
    alpha = result.params['alpha']
    beta = result.params['beta']
    last_vol = result.conditional_volatility.iloc[-1]
    
    #generate the random number generator and initialise the arrays for price paths, current price,current volatility 

    rng = np.random.default_rng()
    price_paths = np.zeros((10000, T))
    current_vol = np.full(10000, last_vol)
    current_prices = np.full(10000, initial_price)

    #loop for the number of days and fill the arrays with the daily volatility and price fluctuations and generate the next days volitality

    for t in range(T):
        z = rng.standard_t(df=nu, size = 10000)
        returns_daily = current_vol * z
        current_prices = current_prices * np.exp(returns_daily)
        price_paths[:,t] = current_prices 
        current_vol = np.sqrt(omega + (alpha * (returns_daily**2)) + (beta * (current_vol**2)))
    
    #the final price would be the last coloumn of the array of the final price 

    final_price = price_paths[:,-1]
    expectation = np.mean(final_price)
    voltality = np.std(final_price)
    losses = final_price - initial_price
    VaR = np.percentile(losses, 5)
    return [expectation, voltality, losses, VaR]