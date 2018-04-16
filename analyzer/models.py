from django.db import models
import pickle as pickle

# Create your models here.
class trade(models.Model):
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)
    Choice = []
    for ticker in tickers:
            str1=ticker
            str2=ticker
            Choice.append((str1,str2))
    Deal_Choices=(('BUY','BUY'),('HOLD','HOLD'),('SELL','SELL'))
    deal=models.CharField(max_length=4, choices=Deal_Choices,help_text="Enter your trade type")
    stock=models.CharField(max_length=5, choices=Choice, help_text="The Scrip!!!")
    quantity=models.IntegerField(help_text="Enter your quantity here!!",default=0)
    totalQuantity=models.IntegerField(help_text="Enter your quantity here",default=0)
    value=models.DecimalField(help_text="Current value of your stock",max_digits=15,decimal_places=4,default=0)
    price=models.DecimalField(help_text="The price at which you traded",max_digits=10,decimal_places=4,default=0)