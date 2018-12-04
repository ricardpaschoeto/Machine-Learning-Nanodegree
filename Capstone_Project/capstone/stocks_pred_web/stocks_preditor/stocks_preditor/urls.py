"""stocks_preditor URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from stocks_preditor_app.views import index, search, models_analisys, predict, predict_date, read_database

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^index/$', index),
    url(r'^search/$', search),
    url(r'^models_analisys/$', models_analisys),
    url(r'^predict/$', predict),
    url(r'^predict_date/$', predict_date),
    url(r'^read_database/$', read_database),
]
