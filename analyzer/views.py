from .forms import StockForm,TradeForm
from django.shortcuts import render
from django.http import HttpResponseRedirect
import os.path
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import pandas as pd
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from .models import trade
from django.shortcuts import get_object_or_404

st=""
pred=""
def redirect6(request):
    if request.method == 'POST':
        form = TradeForm(request.POST)
        tr=get_object_or_404(trade)
        if form.is_valid():
            st=form.cleaned_data['stock']
            trade_set=trade.objects.filter(stock=st)
            if trade_set.exists():
                if(form.cleaned_data['deal']=="BUY" or form.cleaned_data['deal']=="HOLD"):
                    trade_set.value=trade_set.value+(form.cleaned_data['price']*form.cleaned_data['quantity'])
                    trade_set.totalQuantity=trade_set.totalQuantity+form.cleaned_data['quantity']
                else:
                    trade_set.value = trade_set.value -(form.cleaned_data['price'] * form.cleaned_data['quantity'])
                    trade_set.totalQuantity=trade_set.totalQuantity-form.cleaned_data['quantity']
            else:
                if(form.cleaned_data['deal']=="BUY" or form.cleaned_data['deal']=="HOLD"):
                    tr.value=form.cleaned_data['price']*form.cleaned_data['quantity']
                    tr.totalQuantity=trade.quantity
                    tr.stock=form.cleaned_data['stock']
                    tr.save()
            trade_set.update()
            return HttpResponseRedirect('')
    else:
        form = TradeForm()
    return render(request, 'registration/addnew.html', {'form': form})

def redirect5(request):
    trades=trade.objects.all()
    dict=[]
    for tr in trades:
        dict.append({'Stock':tr.stock,'Quantity':tr.totalQuantity,'Value':tr.value  })
    context={'trades':dict}
    return render(
        request,'registration/profile.html',context
    )

def index(request):
    return render(
        request,'index.html'
    )
def vis2(request):
    if request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect('')
    else:
        form = StockForm()

    return render(request, 'visualization2.html', {'form': form})

def vis3(request):
    if request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect('')
    else:
        form = StockForm()

    return render(request, 'visualization3.html', {'form': form})

def vis4(request):
    if request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect('')
    else:
        form = StockForm()

    return render(request, 'visualization4.html', {'form': form})

def vis1(request):
    if request.method == 'POST':
        form = StockForm(request.POST)
        if form.is_valid():
            return HttpResponseRedirect('')
    else:
        form = StockForm()

    return render(request, 'visualization1.html', {'form': form})

def analyze(request):
    st=request.POST['stock']
    histogram(st)
    bollinger_band(st)
    candlestick(st)
    pred=do_ml(st)
    request.session['predict']=pred
    request.session['stock']=st
    return render(
        request, 'visualization/redirect4.html',
    )


def hist(request):
    st=request.POST['stock']
    histogram(st)
    request.session['stock']=st
    return render(
        request, 'visualization/redirect.html',
    )

def redirect(request):
    stock=request.session['stock']
    return render(
        request, 'visualization/histogram.html',{'stock':stock}
    )

def bands(request):
    st=request.POST['stock']
    histogram(st)
    request.session['stock']=st
    return render(
        request, 'visualization/redirect2.html',
    )

def redirect2(request):
    stock=request.session['stock']
    return render(
        request, 'visualization/bollinger.html',{'stock':stock}
    )

def candle(request):
    st=request.POST['stock']
    candlestick(st)
    request.session['stock']=st
    return render(
        request, 'visualization/redirect3.html',
    )

def redirect3(request):
    stock=request.session['stock']
    return render(
        request, 'visualization/candlestick.html',{'stock':stock}
    )

def redirect4(request):
    stock=request.session['stock']
    predict=request.session['predict']
    return render(
        request, 'visualization/analyze.html',{'stock':stock,'predict':predict}
    )

def compute_daily_return(df):
    daily_ret = df.copy()
    daily_ret[1:] = (df[1:] / df[:-1].values) - 1
    daily_ret.ix[0] = 0
    return daily_ret


def plot_data(df, title="Histogram"):
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()

def histogram(ticker):
    df = pd.read_csv("CompiledData.csv", parse_dates=True, na_values=['nan'])
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    dailyreturn = compute_daily_return(df[ticker])
    dailyreturn = dailyreturn.values
    plt.hist(dailyreturn, 20)
    if os.path.isfile(("static/histogram/{}.png".format(ticker))):
        os.remove("static/histogram/{}.png".format(ticker))
    plt.savefig("static/histogram/{}.png".format(ticker))

def bands(request):
    st = request.POST['stock']
    bollinger_band(st)
    request.session['stock'] = st
    return render(
        request, 'visualization/redirect2.html',
    )


def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return pd.rolling_mean(values, window=window)


def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return pd.rolling_std(values, window=window)


def get_bollinger_bands(rm, rstd):
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2
    return upper_band, lower_band


def bollinger_band(ticker):
    df = pd.read_csv("CompiledData.csv", parse_dates=True, na_values=['nan'])
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    RollingMean = get_rolling_mean(df[ticker], window=20)
    RollingStd = get_rolling_std(df[ticker], window=20)
    upper_band, lower_band = get_bollinger_bands(RollingMean, RollingStd)
    ax = df[ticker].plot(title="Bollinger Bands", label=ticker)
    RollingMean.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    if os.path.isfile(("static/bollinger/{}.png".format(ticker))):
        os.remove("static/bollinger/{}.png".format(ticker))
    plt.savefig("static/bollinger/{}.png".format(ticker))

def candlestick(ticker):
    style.use('ggplot')
    df = pd.read_csv('stock_dfs/{}.csv'.format(ticker), parse_dates=True, index_col=0)
    df_ohlc = df['AdjClose'].resample('50D').ohlc()
    df_ohlc.fillna(method="ffill",inplace=True)
    df_ohlc.fillna(method="bfill",inplace=True)
    df_volume = df['Volume'].resample('50D').sum()
    df_volume.fillna(method="ffill",inplace=True)
    df_volume.fillna(method="bfill",inplace=True)
    df_ohlc.reset_index(inplace=True)
    df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
    ax1.xaxis_date()
    candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
    ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
    if os.path.isfile(("static/candlestick/{}.png".format(ticker))):
        os.remove("static/candlestick/{}.png".format(ticker))
    plt.savefig("static/candlestick/{}.png".format(ticker))


style.use('ggplot')

counter = 0
count_0 = 0
count_1 = 0
count__1 = 0

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('CompiledData.csv', index_col=0)
    tickers = df.columns.values.tolist()
    # df.fillna(0, inplace=True)
    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    # df.fillna(0, inplace=True)
    return tickers, df


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    # print('Data spread:',Counter(str_vals))
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                                         y,
                                                                         test_size=0.15)

    # clf = neighbors.KNeighborsClassifier()

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    # print('accuracy:',confidence)
    predictions = clf.predict(X_test)
    c = Counter(predictions)
    # print('predicted class counts:',c)
    li = list(c.items())
    li = sorted(li, key=lambda l: l[1], reverse=True)
    # print (li)
    # print (li[0][0])
    if (li[0][0] == 0):
        pr="HOLD"
        return pr
    if (li[0][0] == 1):
        pr="BUY"
        return pr
    if (li[0][0] == -1):
        pr="SELL"
        return pr