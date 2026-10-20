import numpy as np
from scipy.stats import norm

def d1(E,S0,r,sig,T):
    logSE=np.log(S0/E)
    return((logSE+(r+sig**2/2)*T)/(sig*np.sqrt(T)))

def d2(E,S0,r,sig,T):
    return(d1(E,S0,r,sig,T)-sig*np.sqrt(T))

def eurocallbenchmark(E,S0,r,sig,T):
    endee1=norm.cdf(d1(E,S0,r,sig,T))
    endee2=norm.cdf(d2(E,S0,r,sig,T))
    return(S0*endee1-E*np.exp(-r*T)*endee2)

def europutbenchmark(E,S0,r,sig,T):
    return(eurocallbenchmark(E,S0,r,sig,T)+np.exp(-r*T)*E-S0)

def deltacall(E,S0,r,sig,T):
    return(norm.cdf(d1(E,S0,r,sig,T))) 

def deltaput(E,S0,r,sig,T):
    return(deltacall(E,S0,r,sig,T)-1)

