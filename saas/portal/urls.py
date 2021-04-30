from django.conf.urls import include, url
from django.urls import path
from django.views.generic import TemplateView

from django.views.generic.base import RedirectView
from portal import views as portal_views

app_name = 'portal'
from .views import HomeView, ResultView, result, scatter, algo_result
# , upload_csv, FuncionaView, contacto, ContactoView, CondicionesView, PrivacidadView, subscribe, successView, contacto


urlpatterns = [
        url(r'^$', HomeView.as_view(), name='home'), 
        url(r'^result/', result, name='result'), 
        url(r'^scatter/', scatter, name='scatter'),
        url(r'^algo/', algo_result, name='algo_result'),
        # url(r'^how-it-works/$', FuncionaView.as_view(), name='funciona'),
        # url(r'^subscribe/', subscribe, name="subscribe"),
        # url(r'^contact/', contacto, name='contacto'),
        # url(r'^contact/$', ContactoView.as_view(), name='contacto'),
        # url(r'^results/$', upload_csv, name='upload_csv'),
        # url(r'^deep-learning/$', NeuralView.as_view(), name='neuralview'),
        # url(r'^neural-results/$', neural, name= 'neural'),
]