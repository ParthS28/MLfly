from django.conf.urls import include, url
from django.urls import path
from django.views.generic import TemplateView

from django.views.generic.base import RedirectView
from portal import views as portal_views

app_name = 'portal'
from .views import HomeView, ResultView, result, scatter, algo_result


urlpatterns = [
        url(r'^$', HomeView.as_view(), name='home'), 
        url(r'^result/', result, name='result'), 
        url(r'^scatter/', scatter, name='scatter'),
        url(r'^algo/', algo_result, name='algo_result'),
]