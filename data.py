import yfinance as yf
def monte_carlo_data(ticker:str,start_date:str,end_date:str):
    ticker=ticker.strip().upper()
    if ticker  == "":
        raise ValueError("Ticker cannot be empty")
    try:
        data=yf.download(ticker,start=start_date,end=end_date)
    except Exception:
        raise ValueError(f"Download failed for {ticker}")
    if data.empty:
        raise ValueError(f"No data found for {ticker}")  
    close = data["Close"].squeeze()
    close=close.dropna()
    if close.empty:
        raise ValueError(f"No valid closing prices fo {ticker}")
    return close


def reinforcement_data(ticker:str,start_date:str,end_date:str):
    ticker=ticker.strip().upper()
    if ticker  == "":
        raise ValueError("Ticker cannot be empty")
    try:
        data=yf.download(ticker,start=start_date,end=end_date)
    except Exception:
        raise ValueError(f"Download failed for {ticker}")
    if data.empty:
        raise ValueError(f"No data found for {ticker}")  
    close=data["Close"]
    close=close.ffill()
   
    close=close.dropna()
    if close.empty:
        raise ValueError(f"No valid closing prices fo {ticker}")
    return close
