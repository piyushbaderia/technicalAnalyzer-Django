from django import forms
import bs4 as bs
import pickle as pickle
from .models import trade
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import requests



class StockForm(forms.Form):
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
    Choice=[]

    for ticker in tickers:
            str1=ticker
            str2=ticker
            Choice.append((str1,str2))
    stock=forms.CharField(label="Here are your choices!!",widget=forms.Select(choices=Choice))

class TradeForm(forms.Form):
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
    Choice = []

    for ticker in tickers:
        str1 = ticker
        str2 = ticker
        Choice.append((str1, str2))

    Deal_Choices = (('BUY', 'BUY'), ('HOLD', 'HOLD'), ('SELL', 'SELL'))
    stock = forms.CharField(label="Here are your choices!!", widget=forms.Select(choices=Choice))
    deal=forms.CharField(label="Select the type of trade!!", widget=forms.Select(choices=Deal_Choices))
    quantity = forms.IntegerField(label="Please enter the quantity")
    price=forms.DecimalField(label="Can I please have the price at which you made the trade")
    trade.stock=stock
    trade.quantity=quantity
    trade.deal=deal


