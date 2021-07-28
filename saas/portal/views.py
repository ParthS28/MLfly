import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import plotly
import plotly.express as px
import plotly.graph_objects as go

import io
import csv
from io import BytesIO
import base64
import functools
import operator
import random
import unicodedata
import datetime
import pandas as pd
import seaborn as sns
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB

from django.template.loader import get_template, render_to_string
from django.shortcuts import render
from django.views.generic import TemplateView
from django.views.generic.detail import DetailView
from django.views import View
from django.db.models import Count
from django.shortcuts import redirect

from portal.models import Subscribe
from django.http import Http404, JsonResponse, HttpResponse, HttpResponseRedirect, HttpResponseServerError
from django.db.models import Q
 
df = pd.DataFrame() 

class HomeView(View):
    def get(self, request, *args, **kwargs):
        context={}
        context['is_home'] = True
        context['lazyjs'] = True
        context['valoracionesjs'] = False
        context['valoracionesTiendajs'] = False
        context['normal_footer_cat'] = True

        return render(request, 'home.html', context)


class ResultView(View):
    def get(self, request, *args, **kwargs):
            if request.method == 'POST':
                return render(request, 'result.html')

def result(request):
    df = pd.DataFrame()
    if request.method == 'POST':
        csv = request.FILES['csv_file']
        df = pd.read_csv(csv)
        with open('data.csv', 'wb+') as destination:
            for chunk in csv.chunks():
                destination.write(chunk)

        
        df_target = df
        grafica = request.POST['graphic']
        if grafica == "scatter":
            cols = df.columns
            x = cols[0]
            y = cols[0]
            fig = px.scatter(df, x=x, y=y)
            fig = plotly.offline.plot(fig, auto_open=False, output_type='div')
            context = {'columns': cols, 'fig': fig, 'csv': df}
            return render(request, 'scatter.html', context)


def scatter(request):
    if request.method == 'POST':
        df = pd.read_csv('data.csv')
        x = request.POST['x-axis']
        y = request.POST['y-axis']
        cols = df.columns
        # print(x, y)
        # print(df)
        fig = px.scatter(df, x=x, y=y)
        fig = plotly.offline.plot(fig, auto_open=False, output_type='div')
        # fig.update_layout(template='plotly_white')
        # fig.write_html(fig)
        context = {'columns': cols, 'fig': fig}
        return render(request, 'scatter.html', context)
    
		
        # return render(request, 'result.html')

def algo_result(request):
    print('--------------------------------------------------')
    df = pd.DataFrame()
    if request.method == 'POST':
        csv = request.FILES['csv_file']
        df = pd.read_csv(csv)
        with open('data.csv', 'wb+') as destination:
            for chunk in csv.chunks():
                destination.write(chunk)

        df_target = df
        lon = len(list(df.head(0)))
        header = list(df[0:lon])
        target = header[lon-1]
        y = np.array(df[target])
        df.drop(target,axis=1,inplace=True)
        X = df.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101) 
        matrix = ''
        report = ''
        pred = ''
        algo = request.POST['algorithm']
        if algo == 'Logistic Regression':
            lm = LogisticRegression()
            model = lm.fit(X_train,y_train)
            pred = lm.predict(X_test)
            MAE = metrics.mean_absolute_error(y_test,pred)
            MSE = metrics.mean_squared_error(y_test,pred)
            RMSE = np.sqrt(metrics.mean_squared_error(y_test,pred))
            # print(RMSE, MAE, MSE)
            # print(y_test)
            # print(pred)
            matrix = confusion_matrix(y_test,pred)
            report = classification_report(y_test,pred)
            context = {'matrix': matrix, 'report': report}
            return render(request, 'algo_result.html', context)
        if algo == 'Support Vector Machine':
            param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
            grid = GridSearchCV(SVC(),param_grid,verbose=3)
            model = grid.fit(X_train,y_train)
            pred = grid.predict(X_test)
            MAE = metrics.mean_absolute_error(y_test,pred)
            MSE = metrics.mean_squared_error(y_test,pred)
            RMSE = np.sqrt(metrics.mean_squared_error(y_test,pred))
            matrix = confusion_matrix(y_test,pred)
            report = classification_report(y_test,pred)
            context = {'matrix': matrix, 'report': report}
            return render(request, "algo_result.html", context)
        if algo == 'K-Means':
            kmeans = KMeans(n_clusters=4)
            model = kmeans.fit(X_train)
            clusters = kmeans.cluster_centers_
            labels = kmeans.labels_
        if algo == 'K-Nearest Neighbor':
            knn = KNeighborsClassifier(n_neighbors=1)
            model = knn.fit(X_train,y_train)
            pred = knn.predict(X_test)
            MAE = metrics.mean_absolute_error(y_test,pred)
            MSE = metrics.mean_squared_error(y_test,pred)
            RMSE = np.sqrt(metrics.mean_squared_error(y_test,pred))
            matrix = confusion_matrix(y_test,pred)
            report = classification_report(y_test,pred)
            context = {'matrix': matrix, 'report': report}
            return render(request, "algo_result.html", context)
        if algo == 'Naive Bayes':
            gnb = GaussianNB()
            pred = gnb.fit(X_train, y_train).predict(X_test)
            MAE = metrics.mean_absolute_error(y_test,pred)
            MSE = metrics.mean_squared_error(y_test,pred)
            RMSE = np.sqrt(metrics.mean_squared_error(y_test,pred))
            matrix = confusion_matrix(y_test,pred)
            report = classification_report(y_test,pred)
            context = {'matrix': matrix, 'report': report}
            return render(request, "algo_result.html", context)
        if algo == 'Decision Trees':
            dtree = DecisionTreeClassifier()
            model = dtree.fit(X_train,y_train)
            pred = dtree.predict(X_test)
            MAE = metrics.mean_absolute_error(y_test,pred)
            MSE = metrics.mean_squared_error(y_test,pred)
            RMSE = np.sqrt(metrics.mean_squared_error(y_test,pred))
            matrix = confusion_matrix(y_test,pred)
            report = classification_report(y_test,pred)
            context = {'matrix': matrix, 'report': report}
            return render(request, "algo_result.html", context)
        if algo == 'Random Forest':
            forest = RandomForestClassifier(n_estimators=200)
            model = forest.fit(X_train,y_train)
            pred = forest.predict(X_test)
            MAE = metrics.mean_absolute_error(y_test,pred)
            MSE = metrics.mean_squared_error(y_test,pred)
            RMSE = np.sqrt(metrics.mean_squared_error(y_test,pred))
            matrix = confusion_matrix(y_test,pred)
            report = classification_report(y_test,pred)
            context = {'matrix': matrix, 'report': report}
            return render(request, "algo_result.html", context)