from django.template.loader import get_template
from django.template.response import TemplateResponse
from django.http import HttpResponse
from data_process import DataProcessing
from supervised_models import SupervisedModels
from django.shortcuts import render
import numpy as np
import pandas as pd
import datetime

''' Function to call the trainning page'''
def index(request):
    return render(request, 'trainning_page.html')

''' Function to call the Predict page'''
def predict(request):
    return render(request, 'predict_page.html')

''' Function to call the Predict page to predcit a single date price'''
def predict_date(request):
    date = request.POST['pdate']
    test_size =  request.POST['test']
    if 'models' in request.POST:
        dp = DataProcessing()
        model = request.POST['models']
        predict = request.POST['pdate']
        test_size =  request.POST['test']
        tickers = dp.file_list()
        graphic = dp.image_models(date, tickers, model, test_size)
        
    return render(request, 'predict_page.html',{'graphic':graphic})

''' Function to call collect data, create data table and show in the trainning page'''
def search(request):
    if 'stocks' in request.POST:
        stock = request.POST['stocks']
        start = request.POST['sdate']
        end = request.POST['edate']
        tickers = stock.split(',')

        request.session['sdate'] = start
        request.session['edate'] = end
        request.session['stocks'] = stock

        dp = DataProcessing()
        dp.collect_data(tickers, 'yahoo', start, end)

        df = dp.mount_data(tickers)
        graphic = dp.getimage(tickers)

        return render(request, 'trainning_page.html',{'data': df.iterrows(),'graphic':graphic})

''' Function to read data from data (.csv) directory'''
def read_database(request):
    dp = DataProcessing()
    tickers = dp.file_list()
    if len(tickers) != 0:
        df = dp.mount_data(tickers)
        graphic = dp.getimage(tickers)
        return render(request, 'trainning_page.html',{'data': df.iterrows(),'graphic':graphic})
    else:
        return render(request, 'trainning_page.html')

''' Funtion to create and show the metric table'''    
def models_analisys(request):
    if 'test' in request.POST:
        dp = DataProcessing()
        test_size =  request.POST['test']
        tickers = dp.file_list()

        df_results = dp.analisys(tickers, test_size)
        df = dp.mount_data(tickers)

        graphic = dp.getimage(tickers)

        return render(request, 'trainning_page.html',{'results': df_results.iterrows(), 'data': df.iterrows(),
                        'graphic':graphic})






