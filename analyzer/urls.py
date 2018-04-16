from django.urls import path
from . import views

urlpatterns = [
path('', views.index, name='index')
]
urlpatterns += [
    path('visualization/',views.vis1,name='vis1'),
]
urlpatterns += [
    path('visualization1/',views.vis2,name='vis2'),
]
urlpatterns += [
    path('visualization2/',views.vis3,name='vis3'),
]
urlpatterns += [
    path('visualization3/',views.vis4,name='vis4'),
]
urlpatterns += [
    path('visualization/redirect',views.hist,name='redirect'),
]
urlpatterns += [
    path('visualization/redirect/histogram',views.redirect,name='histogram'),
]
urlpatterns += [
    path('visualization1/redirect2',views.bands,name='redirect2'),
]
urlpatterns += [
    path('visualization1/redirect2/bollinger',views.redirect2,name='bollinger'),
]
urlpatterns += [
    path('visualization2/redirect3',views.candle,name='redirect3'),
]
urlpatterns += [
    path('visualization2/redirect3/candlestick',views.redirect3,name='candle'),
]
urlpatterns += [
    path('visualization3/redirect4',views.analyze,name='redirect4'),
]
urlpatterns += [
    path('visualization3/redirect4/analyze',views.redirect4,name='analyze'),
]
urlpatterns += [
    path('profile',views.redirect5,name='profile'),
]
urlpatterns += [
    path('portfolio/',views.redirect6,name='redirect6'),
]