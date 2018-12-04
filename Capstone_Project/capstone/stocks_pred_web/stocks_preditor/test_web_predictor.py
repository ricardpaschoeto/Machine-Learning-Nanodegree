import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_process import DataProcessing
from data_statistics import DataStatistics
from sklearn import metrics
from supervised_models import SupervisedModels
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tickers = ['PETR4.SA','VALE3.SA','ITUB4.SA', 'ABEV3.SA','BBAS3.SA']
dp = DataProcessing()
ds = DataStatistics()
sm = SupervisedModels()

#dp.collect_data(tickers, 'yahoo',  '2017-05-10', '2018-08-29')
#df_stocks = dp.read_data(tickers[0],'2017-07-09', '2018-08-24')
#print(df_stocks.tail())
#df = dp.mount_data(tickers, '2017-07-14', '2018-08-20')
#df = dp.analisys(tickers, '2016-02-09', '2018-08-24', 28)
#print(df)
########################
path = BASE_DIR+'\\stocks_preditor\\stocks_preditor_app\\data\\'
dp.delete_files(path)
"""df_final = dp.read_data(tickers[0])
y,pred = sm.benchmarch(df_final, 28)
df_pred = pd.DataFrame(pred, index=pd.to_datetime(y.index), columns=['Predict'])
#print(df_pred.loc['2018-08-29', 'Predict'])
df_final_dt = pd.DataFrame(y, index=pd.to_datetime(y.index)) 
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)
ax.plot(pd.to_datetime('2018-08-20'), df_pred.loc['2018-08-20', 'Predict'], color='red', marker='X', linewidth=1, markersize=8)
ax.plot(df_final_dt)
ax.legend(['{} Predict along to {} days'.format(tickers[0],28), tickers[0]])
plt.show()"""

"""df_adj_close = pd.DataFrame()
for ticker in arr:
    #legends.append(ticker)
    df_temp = dp.read_data(ticker)
    df_adj_close = df_temp.iloc[:, 5]
    df_adj_close = df_adj_close.astype(float)
    temp.append(df_adj_close)
    df = pd.concat(temp, axis=1)
print(df.index)"""
"""plt.scatter(y.astype(float), pred.astype(float))
plt.xlabel('Adj Close')
plt.ylabel('Predicted Adj Close')
plt.show()

print('RMSE:', np.sqrt(metrics.mean_squared_error(y, pred)))
print('R^2:', metrics.r2_score(y, pred))"""
#########################
"""df_final = dp.read_data(tickers[0], '2017-07-14', '2018-08-20')
y_test, pred = sm.tree_model(df_final,28)

plt.scatter(y_test.astype(float),pred.astype(float))
plt.xlabel('Adj Close')
plt.ylabel('Predicted Adj Close')
plt.show()

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
print('\n')
print('R^2:', metrics.r2_score(y_test, pred))"""
#########################
"""df_final = dp.read_data(tickers[0], '2017-07-14', '2018-08-20')
y_test, pred = sm.tree_model_adaboost(df_final,28)

plt.scatter(y_test.astype(float),pred.astype(float))
plt.xlabel('Adj Close')
plt.ylabel('Predicted Adj Close')
plt.show()

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
print('\n')
print('R^2:', metrics.r2_score(y_test, pred))"""
##########################
"""df_final = dp.read_data(tickers[0], '2017-07-14', '2018-08-20')
y_test, pred = sm.sgd_model(df_final,28)

plt.scatter(y_test.astype(float),pred.astype(float))
plt.xlabel('Adj Close')
plt.ylabel('Predicted Adj Close')
plt.show()

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
print('\n')
print('R^2:', metrics.r2_score(y_test, pred))"""
##########################
"""df_final = dp.read_data(tickers[0], '2017-07-14', '2018-08-20')
y_test, pred = sm.sgd_model_adaboost(df_final,28)

plt.scatter(y_test.astype(float),pred.astype(float))
plt.xlabel('Adj Close')
plt.ylabel('Predicted Adj Close')
plt.show()

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
print('\n')
print('R^2:', metrics.r2_score(y_test, pred))"""
#########################
