'''
1 -- Buy Growth
-1 -- Buy Value
Divide funds into several portions for investment -- Leverage amount (different style factors -- Growth Value Valuation Leverage)
'''
import numpy as np

def BackTest(pred,p0,k=0,lev=1):
    '''
    Backtest function:
    pred: Signal
    p0: Open price
    c0: Close price # Calculate daily floating profit and loss
    k: Commission rate
    alpha: Stop loss
    '''
    m=len(pred) 
    capital=p0[0]/lev
    pricesB=np.full(m,np.nan)
    net=capital  
    pos=0
    for i in range(1,m):
        p=p0
        if pos==1: 
            if pred[i]==1:                                                       
                net=np.append(net,(capital-pricesB[np.max(np.where(~np.isnan(pricesB)))]+p[i]))#持续做多：原有浮动收益+次日价格（open）
                pos=1
            elif pred[i]==0:               
                capital=capital-pricesB[np.max(np.where(~np.isnan(pricesB)))]+(1-k)*p[i]
                pos=0
                net=np.append(net,capital)
                    
        elif pos==0:          
            if pred[i]==1:      
                pricesB[i]=p[i]
                capital=capital-k*pricesB[i]
                pos=1
                net=np.append(net,capital-pricesB[i]+p[i])
            else:
                net=np.append(net,capital)
    return net/net[0]
