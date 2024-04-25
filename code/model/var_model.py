
'''
Macro Category Division:

Inflation: Key material information -- Crude oil
Interest Rates: gc001, r007
Economic Momentum: Work commencement rates, real estate transactions
Overseas: Exchange rates, NASDAQ, S&P 500 -- Price-to-Earnings ratios, index information of the U.S. stock market
Market Sentiment: Margin trading, stock market turnover
'''
name = ['valuation','oversea','ecomoment','interest','emotion','priceindex']

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

'''
step1: get regression relationship
To prevent the amplification of error term interference, 
conducted an Augmented Dickey-Fuller (ADF) test on the factors
screened to ensure the stability of each factor's time series within the VAR model.
'''
def Get_Reg_Result(y, factor_data, lag, year):
    reg_result = pd.DataFrame()
    for factor in factor_data.columns:
        begin = max(pd.Timestamp('{}-1-1'.format(year)), factor_data[factor].dropna().index[0])
        end = y.index[-1]

        reg_result.loc[factor, 'begin_date'] = begin
        reg_result.loc[factor, 'end_date'] = end

        reg_data = pd.concat([y, factor_data[factor]], axis=1).loc[begin: end].fillna(method='ffill')
        reg_data[factor] = reg_data[factor].shift()
        reg_data = reg_data.dropna()
        
        reg_result.loc[factor, 'IC'] = reg_data.corr().iloc[0, 1]
        reg_result.loc[factor, 'RankIC'] = reg_data.corr(method='spearman').iloc[0, 1]
        adfmodel = adfuller(factor_data[factor],autolag='AIC',maxlag=lag)
        reg_result.loc[factor, 'ADF_pvalue'] = adfmodel[1]
        reg_result.loc[factor, 'ADF_lag'] = adfmodel[2]

        print(year, begin, end, factor, factor)
    return reg_result

'''
step2:model training
During model training, 
constructed VAR models for each category by adjusting the lag order k, 
and observe the win rate of each category's judgment on style changes. 
For data selection, a four-year time window is used as the training set, 
with a one-year window as the test set. 
The model selection criteria use the Akaike Information Criterion (AIC) 
and the Bayesian Information Criterion (BIC) 
to assess model complexity and goodness-of-fit.
'''
def VarModelLoop(lags,index_train,trainset,name,result=None,impulse=False):
    '''
    testing params
    '''
    if impulse == False:
        for lag in range(1,lags):
            temp_factor = pd.concat([index_train,trainset],axis=1,join='inner')
            temp_var = sm.tsa.VARMAX(temp_factor.astype(float),order=(lag,0),trend='c',exog=None)
            fitMod = temp_var.fit(maxiter=50,disp=False)
            result.loc[name+'lag'+str(lag),'aic'] = fitMod.aic
        modelimpulse, fitvalue, fitvalue = None, None, None
    if impulse == True:
        temp_factor = pd.concat([index_train,trainset],axis=1,join='inner')
        temp_var = sm.tsa.VARMAX(temp_factor.astype(float),order=(lags,0),trend='c',exog=None)
        fitMod = temp_var.fit(maxiter=50,disp=False)
        modelimpulse = fitMod.impulse_responses(10, orthogonalized=True)
        fitvalue = fitMod.fittedvalues
    return result, fitMod, modelimpulse, fitvalue


def ImpulseImport(fitMod,classindex,name):
    plt.figure(dpi=600,figsize=(8,5.5))
    impulse = fitMod.impulse_responses(10, orthogonalized=True)
    ax = plt.subplot(1,1,1)
    df = impulse
    ax.plot(df,markersize=5)
    plt.grid(None)
    columns = ['index']+classindex.columns.tolist()
    ax.legend(columns,loc='upper right')
    ax.set_title(name+'Impulse_Plot',fontproperties=font)
    
    plt.tight_layout()
    sheetname = name+'Impulse_Plot'
    figpath = sheetname +'.png'
    plt.savefig(figpath,bbox_inches='tight')
    plt.show()    
    return impulse


'''
step3:
Starting from the VAR model for a single asset class, 
used the direction of change in continuous series as a signal and 
the constructed index's adjusted net value as the investment target 
for backtesting to observe the model's performance.
'''
'''
The method for selecting long and short positions is filtered through the win rate. 
Two methods are used to calculate the win rate. 
The first method calculates the cumulative win rate by combining the upper and lower thresholds. 
The second method separates the thresholds of the upper and lower boundaries.
'''
def WinrateCount(signal, vartrainset):
    ratio = pd.DataFrame()
    x1 = signal.iloc[:,0]  # testing signal
    signal = vartrainset.apply(lambda x : Compare(x))
    signal = signal.loc[signal.index.isin(signal.index),:]
    x2 = signal  # index_returns signal
    sumlen = signal.shape[0]
    # odd_ratio
    if (sum((x1 == 1) & (x2 == -1)) + sum((x1 == -1) & (x2 == 1))) == 0:
        pass
    else:
        count1 = (sum((x1 == 1) & (x2 == 1)) + sum((x1 == -1) & (x2 == -1))) / (sum((x1 == 1) & (x2 == -1)) + sum((x1 == -1) & (x2 == 1)))
        count2 = (sum((x1 == 1) & (x2 == 1)) + sum((x1 == -1) & (x2 == -1))) / sumlen        
        ratio.loc[j,'OddRatio'] = count1
        ratio.loc[j,'WinRatio'] = count2
    return ratio , signal


def ModelForecast(factor,factor_select,index_return,index_train,index_test,fitMod,names):
    var_test = pd.DataFrame()
    for i,j in names.items():
        num = index_test.shape[0]
        print(i,j)
        temp_select = factor.loc[:,factor_select]
        temp_test = pd.DataFrame(fitMod.forecast(steps=1)['index'])
        temp_test.index = [index_test.index[0]]
        for n in range(1,num):
            temp_index = index_return.iloc[index_train.shape[0]+n:index_train.shape[0]+n+1,:]
            temp_class = temp_select.loc[temp_select.index.isin(temp_index.index),:]
            temp_factor = pd.concat([temp_index,temp_class],axis=1,join='inner')
            temp_value = pd.DataFrame(fitMod.extend(temp_factor,refit=False).forecast(steps=1)['指数'])
            temp_value.index = [temp_index.index[0]]
            print(temp_value)
            temp_test = pd.concat([temp_test,temp_value],axis=0)
        var_test = pd.concat([var_test,temp_test],axis=1)
    return var_test