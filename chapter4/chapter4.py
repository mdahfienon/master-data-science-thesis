# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 21:53:15 2023

@author: MATHIAS
"""

#%% LIBRAIRIES
import numpy as np
import datetime
import pandas as pd
from yahoo_fin.stock_info import get_data

#%% data collection from yahoo

# startDate , as per our convenience we can modify
startDate = datetime.datetime(2015, 10, 31)
 
# endDate , as per our convenience we can modify
endDate = datetime.datetime(2023, 10, 31)

ticker_list = ["META", "AAPL", "GOOG", "MSFT", "AMZN"]
historical_datas = {}
for ticker in ticker_list:
    historical_datas[ticker] = get_data(ticker, 
                                        start_date=startDate,
                                        end_date=endDate)
closing = pd.DataFrame(data = 0, columns=ticker_list, 
                       index = historical_datas["META"].index)

for d in ticker_list:
    closing[d] = historical_datas[d]["adjclose"]
    
    
del ticker, startDate, endDate, ticker_list, d, historical_datas

#%%
closing.to_csv("closing.csv")

#%% 


closing_returns = 100*closing.pct_change().dropna()

closing_log_returns = np.log((closing/closing.shift(1)).dropna())


#%%

closing_log_returns.to_csv("closing_log_returns.csv")

#%%
closing.plot(subplots =True)


closing_log_returns.plot(subplots =True)