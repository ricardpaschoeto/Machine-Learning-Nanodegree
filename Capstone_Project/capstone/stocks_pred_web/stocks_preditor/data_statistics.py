import pandas as pd
import numpy as np
import datetime
from arch import arch_model
import matplotlib.pyplot as plt

class DataStatistics:
    
    ''' Function to calculate the daily return to use inside of the garch function '''
    def daily_return(self, df, stock):
        df_temp = pd.DataFrame()
        df_temp = pd.concat([df['Adj Close'].pct_change(1).dropna()], axis=1)
        return df_temp

    ''' Function to calculate historical volatility by garch model'''
    def garch_volatility(self, df, stock):
        returns = 100*self.daily_return(df, stock)
        am = arch_model(returns)
        res = am.fit(update_freq=5)
        df_vol = pd.DataFrame(columns=['GarchAnnualized'])
        df_vol['GarchAnnualized'] = res.conditional_volatility*np.sqrt(252)
        return df_vol
            