import datetime
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError, SymbolWarning
from sklearn import metrics
from PIL import Image
import io
import base64

from data_statistics import DataStatistics
from supervised_models import SupervisedModels

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DataProcessing:

    ''' Function to collect and preprocessing data '''
    def collect_data(self, tickers, source, start, end):
        self.delete_files( BASE_DIR+'\\stocks_preditor\\stocks_preditor_app\\data\\')
        dates = pd.date_range(start, end)
        df = pd.DataFrame(index=dates)
        ds = DataStatistics()
        df_foe = data.DataReader(['FOE'], source, start, end)['Adj Close']
        df_wti =  data.DataReader(['WTI'], source, start, end)['Adj Close']
        df_dexbzus = data.DataReader(['DEXBZUS'], 'fred', start, end)
        for ticker in tickers:
            try:
                df_stock = data.DataReader(ticker, source, start, end)
                df_garch = ds.garch_volatility(df_stock, ticker)
                df = pd.concat([df_stock, df_foe, df_wti, df_dexbzus, df_garch], axis=1, sort=False)
                df.replace(r'\s+', np.nan, inplace=True)
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)            
                self.write_data(df, BASE_DIR+"\\stocks_preditor\\stocks_preditor_app\\data\\{}.csv".format(ticker))
            except SymbolWarning:
                print('Problems with Symbol Collect ' + ticker)
                continue
            except RemoteDataError:
                print('Problems with Remote Data Access ' +ticker)
                continue

    ''' Function to write data collected from web in data directory (.cvs format) '''
    def write_data(self, data_collected, fileName):
        data_collected.to_csv(fileName)

    ''' Function to delete files from data directory '''
    def delete_files(self, path):
        for f in os.listdir(path):
            f_path = os.path.join(path, f)
            try:
                if os.path.isfile(f_path) and 'csv' in f:
                    os.unlink(f_path)
            except Exception as e:
                print(e)
    
    ''' Function to read data from data directory ''' 
    def read_data(self, ticker):
        df_final = pd.read_csv(BASE_DIR+"\\stocks_preditor\\stocks_preditor_app\\data\\{}.csv".format(ticker),
        index_col=0)

        return df_final

    ''' Function create a pandas Dataframe adjusted '''
    def mount_data(self, tickers):
        vdf = []
        df_read = pd.DataFrame(dtype=np.float)
        for ticker in tickers:
            df_read = self.read_data(ticker).iloc[-1,:].astype(float).round(3)
            df_read.reset_index()
            df_read['Stock'] = ticker
            df_read['AdjClose'] = df_read['Adj Close']
            df_read.drop('Adj Close', inplace=True)
            vdf.append(df_read)
            df = pd.concat(vdf, axis=1)
        return df.T
    
    ''' Function to generate a graphic image from pandas Dataframe '''
    def getimage(self, tickers):
        temp = []
        legends = []
        df_adj_close = pd.DataFrame()
        for ticker in tickers:
            legends.append(ticker)
            df_temp = self.read_data(ticker)
            df_adj_close = df_temp.iloc[:, 5]
            df_adj_close = df_adj_close.astype(float)
            temp.append(df_adj_close)
            df = pd.concat(temp, axis=1)
 
        return self.create_fig_historic(df, legends)

    ''' Function to create a graphic buffer to training page '''
    def create_fig_historic(self, df, legends):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(pd.to_datetime(df.index), df)
        plt.xticks(rotation=30)
        plt.ylabel('Prices')
        ax.set_title('Data History')
        ax.grid(True)
        ax.legend(legends,loc='best')

        canvas = fig.canvas
        buf, size = canvas.print_to_buffer()
        image = Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)
        buffer= io.BytesIO()
        image.save(buffer,'PNG')
        graphic = buffer.getvalue()
        graphic = base64.b64encode(graphic)
        buffer.close()

        return graphic

    ''' Function to calculate the models metrics'''
    def analisys(self, tickers, test_size):
        sm = SupervisedModels()
        darray = np.ndarray(shape=(len(tickers),10), dtype=float)
        arrays = [np.array(['Benchmarch', 'Benchmarch','Tree', 'Tree', 'TreeAdaboost', 'TreeAdaboost',
                             'SGD', 'SGD', 'SGDAdaboost', 'SGDAdaboost']),
                            np.array(['RMSE', 'R2_score', 'RMSE', 'R2_score','RMSE', 
                            'R2_score','RMSE', 'R2_score','RMSE', 'R2_score'])]
        i = 0
        for ticker in tickers:
            df_benc = self.read_data(ticker)

            y,pred  = sm.benchmarch(df_benc, test_size)
            rmse = np.sqrt(metrics.mean_squared_error(y, pred))
            r2_score = metrics.r2_score(y, pred)

            darray[i][0] = rmse.astype(float).round(4)
            darray[i][1] = r2_score.astype(float).round(4)

            y,pred = sm.tree_model(df_benc, test_size)
            rmse = np.sqrt(metrics.mean_squared_error(y, pred))
            r2_score = metrics.r2_score(y, pred)

            darray[i][2] = rmse.astype(float).round(4)
            darray[i][3] = r2_score.astype(float).round(4)

            y,pred= sm.tree_model_adaboost(df_benc, test_size)
            rmse = np.sqrt(metrics.mean_squared_error(y, pred))
            r2_score = metrics.r2_score(y, pred)

            darray[i][4] = rmse.astype(float).round(4)
            darray[i][5] = r2_score.astype(float).round(4)

            y,pred= sm.sgd_model(df_benc, test_size)
            rmse = np.sqrt(metrics.mean_squared_error(y, pred))
            r2_score = metrics.r2_score(y, pred)

            darray[i][6] = rmse.astype(float).round(4)
            darray[i][7] = r2_score.astype(float).round(4)

            y,pred = sm.sgd_model_adaboost(df_benc, test_size)
            rmse = np.sqrt(metrics.mean_squared_error(y, pred))
            r2_score = metrics.r2_score(y, pred)

            darray[i][8] = rmse.astype(float).round(4)
            darray[i][9] = r2_score.astype(float).round(4)
            i += 1 

        df = pd.DataFrame(darray, index=tickers, columns=arrays)
        df.columns.names = ['Models','Results']
        return df
    
    ''' Function to execute each models '''
    def image_models(self, date, tickers, model, test_size):
        temp_pred = []
        temp_y =[]
        legends = []
        sm = SupervisedModels()
        if model == 'linear':
            title = 'Linear Regression'
            for ticker in tickers:
                legends.append(ticker)
                y, pred = sm.benchmarch(self.read_data(ticker), test_size)
                df_pred, df_y = self.mount_dfs(pred, y)
                temp_y.append(df_y)
                temp_pred.append(df_pred)
        if model == 'tree':
            title = 'Decision Tree Regressor'
            for ticker in tickers:
                legends.append(ticker)
                y, pred = sm.tree_model(self.read_data(ticker), test_size)
                df_pred, df_y = self.mount_dfs(pred, y)
                temp_y.append(df_y)
                temp_pred.append(df_pred)
        if model == 'treeAdaBoost':
            title = 'Decision Tree Regressor AdaBoost'
            for ticker in tickers:
                legends.append(ticker)
                y, pred = sm.tree_model_adaboost(self.read_data(ticker), test_size)
                df_pred, df_y = self.mount_dfs(pred, y)
                temp_y.append(df_y)
                temp_pred.append(df_pred)
        if model == 'sgd':
            title = 'SGD'
            for ticker in tickers:
                legends.append(ticker)
                y, pred = sm.sgd_model(self.read_data(ticker), test_size)
                df_pred, df_y = self.mount_dfs(pred, y)
                temp_y.append(df_y)
                temp_pred.append(df_pred)
        if model == 'sgdAdaBoost':
            title = 'SGD AdaBoost'
            for ticker in tickers:
                legends.append(ticker)
                y, pred = sm.sgd_model_adaboost(self.read_data(ticker), test_size)
                df_pred, df_y = self.mount_dfs(pred, y)
                temp_y.append(df_y)
                temp_pred.append(df_pred)
        if date == '':
            return self.create_fig_result(temp_pred, temp_y, title, test_size, legends)
        else:
            return self.create_fig_result_one_date(date, temp_pred, temp_y, title, test_size, legends)

    ''' Function to create a graphic buffer to predict page (Results)'''
    def create_fig_result(self, temp_pred, temp_y, title, test_size, legends):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        df = pd.concat(temp_pred, axis = 1)
        df_test = pd.concat(temp_y, axis = 1)
        ax.plot(pd.to_datetime(df_test.index), df_test, linestyle='-', linewidth=2, markersize=2)
        ax.plot(pd.to_datetime(df.index), df, marker='X', linestyle='dashed',linewidth=1, markersize=1)
        plt.xticks(rotation=25)
        plt.ylabel('Real Prices/Predict')
        ax.set_title('{} to {} days'.format(title, test_size))
        ax.grid(True)
        ax.legend(legends)

        canvas = fig.canvas
        buf, size = canvas.print_to_buffer()
        image = Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)
        buffer= io.BytesIO()
        image.save(buffer,'PNG')
        graphic = buffer.getvalue()
        graphic = base64.b64encode(graphic)
        buffer.close()

        return graphic

    ''' Function to create a graphic buffer to predict page with single date to predict'''
    def create_fig_result_one_date(self, date, temp_pred, temp_y, title, test_size, legends):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        df = pd.concat(temp_pred, axis = 1)
        df_test = pd.concat(temp_y, axis = 1)
        dates=[pd.to_datetime(date) for i in range(len(df.loc[date]))]
        ax.plot(pd.to_datetime(df_test.index), df_test, linestyle='-', linewidth=2, markersize=2)
        ax.plot(dates, df.loc[date], marker='X', linestyle='dashed',linewidth=1, markersize=5)
        plt.xticks(rotation=25)
        plt.ylabel('Real Prices/Predict')
        ax.set_title('{} to {} days'.format(title, test_size))
        ax.grid(True)
        ax.legend(legends)

        canvas = fig.canvas
        buf, size = canvas.print_to_buffer()
        image = Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)
        buffer= io.BytesIO()
        image.save(buffer,'PNG')
        graphic = buffer.getvalue()
        graphic = base64.b64encode(graphic)
        buffer.close()

        return graphic

    ''' Function to mount pandas Dataframes to compute results'''
    def mount_dfs(self, pred, y):
        df_pred = pd.DataFrame(pred, index=pd.to_datetime(y.index), columns=['Predict'])
        df_y = pd.DataFrame(y, index=pd.to_datetime(y.index))
        df_pred.sort_index(axis=1, inplace=True)
        df_y.sort_index(axis=1, inplace=True)
        return df_pred, df_y 
    
    ''' Function to list .csv files from data directory'''
    def file_list(self):
        return ['.'.join(f.split('.')[:-1]) for f in os.listdir(BASE_DIR+'\\stocks_preditor\\stocks_preditor_app\\data\\')]
