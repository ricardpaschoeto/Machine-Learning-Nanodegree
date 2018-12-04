from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
import pandas as pd

class SupervisedModels:

    ''' Function to apply Linear Regression'''
    def benchmarch(self, df, len_test):
        # Split the data into training/testing sets
        len_test = int(len_test)
        X_train = df.iloc[1:len(df)-len_test,[0,1,2,4,6,7,8,9]]
        y_train = df.iloc[1:len(df)-len_test,5]
    
        # Split the data into training/testing sets
        X_test = df.iloc[-len_test:,[0,1,2,4,6,7,8,9]]
        y_test = df.iloc[-len_test:,5]
    
        lm = LinearRegression()
        parameters = {'fit_intercept':[True, False], 'normalize':[True, False], 'copy_X':[True, False]}
        grid = GridSearchCV(lm, param_grid=parameters, cv=None)
        grid.fit(X_train, y_train)
        pred = grid.predict(X_test)

        return y_test, pred

    ''' Function to apply tree model'''
    def tree_model(self, df, len_test):
        # Split the data into training/testing sets
        len_test = int(len_test)
        X_train = df.iloc[1:len(df)-len_test,[0,1,2,4,6,7,8,9]]
        y_train = df.iloc[1:len(df)-len_test,5]
    
        # Split the data into training/testing sets
        X_test = df.iloc[-len_test:,[0,1,2,4,6,7,8,9]]
        y_test = df.iloc[-len_test:,5]
    
        tree = DecisionTreeRegressor()
        parameters = {'max_depth': np.arange(3, 15, 3), 'min_samples_split': range(2, 10)}
        grid = GridSearchCV(tree, param_grid=parameters, cv=None)
        grid.fit(X_train, y_train)
        pred = grid.predict(X_test)
    
        return y_test, pred

    ''' Function to apply tree model with Adaboost algorithm'''
    def tree_model_adaboost(self, df, len_test):
        # Split the data into training/testing sets
        len_test = int(len_test)
        X_train = df.iloc[1:len(df)-len_test,[0,1,2,4,6,7,8,9]]
        y_train = df.iloc[1:len(df)-len_test,5]
    
        # Split the data into training/testing sets
        X_test = df.iloc[-len_test:,[0,1,2,4,6,7,8,9]]
        y_test = df.iloc[-len_test:,5]
    
        rng = np.random.seed()
        tree = DecisionTreeRegressor(max_depth=5)
        parameters = {'max_depth': np.arange(3, 15, 3), 'min_samples_split': range(2, 10)}
        grid = GridSearchCV(tree, param_grid=parameters, cv=None)
        ada_tree = AdaBoostRegressor(grid,n_estimators=300, random_state=rng)
        ada_tree.fit(X_train, y_train)
        pred = ada_tree.predict(X_test)

        return y_test, pred

    ''' Function to apply SGD model'''
    def sgd_model(self, df, len_test):
        len_test = int(len_test)
        X_pred = df.iloc[1:,[0,1,2,4,6,7,8,9]]
        y_pred = df.iloc[1:,5]
    
        scaler = StandardScaler()
        scaler.fit(X_pred)
        df_norm = scaler.transform(X_pred)
        df_param = pd.DataFrame(df_norm, columns=df.columns.drop(['Adj Close', 'Close']))
    
        # Split the data into training/testing sets
        X_train = df_param.iloc[1:len(df)-len_test]
        X_test = df_param.iloc[-len_test:]
       
        # Split the data into training/testing sets
        y_train = df.iloc[1:len(df)-len_test,5]
        y_test = df.iloc[-len_test:,5]
        parameters = {'max_iter': np.arange(500, 1500, 10)}
        sgd = SGDRegressor()
        grid = GridSearchCV(sgd, param_grid=parameters, cv=None)
        grid.fit(X_train, y_train)
        pred = grid.predict(X_test)
    
        return y_test, pred

    ''' Function to apply SGD model with Adaboost algorithm'''
    def sgd_model_adaboost(self, df, len_test):
        len_test = int(len_test)
        X_pred = df.iloc[1:,[0,1,2,4,6,7,8,9]]
        y_pred = df.iloc[1:,5]
    
        scaler = StandardScaler()
        scaler.fit(X_pred)
        df_norm = scaler.transform(X_pred)
        df_param = pd.DataFrame(df_norm, columns=df.columns.drop(['Adj Close', 'Close']))
    
        # Split the data into training/testing sets
        X_train = df_param.iloc[1:len(df)-len_test]
        X_test = df_param.iloc[-len_test:]
       
        # Split the data into training/testing sets
        y_train = df.iloc[1:len(df)-len_test,5]
        y_test = df.iloc[-len_test:,5]
        parameters = {'max_iter': np.arange(500, 1500, 10)}
        rng = np.random.seed()
        sgd = SGDRegressor()
        grid = GridSearchCV(sgd, param_grid=parameters, cv=None)
        ada_sgd = AdaBoostRegressor(grid,n_estimators=300, random_state=rng)
        ada_sgd.fit(X_train, y_train)
        pred = ada_sgd.predict(X_test)
    
        return y_test, pred