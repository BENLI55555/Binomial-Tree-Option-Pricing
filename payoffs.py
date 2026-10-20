import numpy as np
from simulation import *
from functools import lru_cache


# for convenience of writing greeks our output of our payoff function will be the option price at all nodes in our binomial tree model 
def european_payoff(S, E, r, sig, T, N, option_type = 'call'):
    S = S[-1]
    dt = T/N
    u = np.exp(sig * np.sqrt(dt))
    d = np.exp(-sig * np.sqrt(dt))
    p = (np.exp(r * dt) - d)/(u - d)
    V = [0] * (N + 1)

    if option_type == 'call':
        V[N] = np.maximum(S - E, 0)
    
    elif option_type == 'put':
        V[N] = np.maximum(E - S, 0)
    
    for i in range(N, 0, -1):
        down = V[i][:i]
        up = V[i][1:(i + 1)]
        V[i - 1] = np.exp(-r * dt) * (p * up + (1 - p) * down)

    return V

#the time complexity is super high in asian_payoff
def asian_payoff(S, E, r, sig, T, N, option_type = 'call', asian_average = 'arithmetic'):
    dt = T/N
    u = np.exp(sig * np.sqrt(dt))  
    d = np.exp(-sig * np.sqrt(dt))
    p = (np.exp(r * dt) - d)/(u - d)
    S0 = S[0]
    V = [[] for _ in range(N + 1)]
    if asian_average == 'geometric':
        def dfs_geo(i, stock, total):
            if i == N:
                average = total ** (1/N)
                if option_type == 'call':
                    payoff = np.maximum(average - E, 0)
                    V[N].append(payoff)
                    return payoff
                elif option_type == 'put':
                    payoff = np.maximum(E - average, 0)
                    V[N].append(payoff)
                    return payoff
            else: 
                down = dfs_geo(i + 1, stock * d, total * stock * d) 
                up = dfs_geo(i + 1, stock * u, total * stock * u)

                value = np.exp(-r * dt) * (p * up + (1 - p) * down)
                V[i].append(value)
                return value
                           
        dfs_geo(0, S0, 1)
     
    elif asian_average == 'arithmetic':
        def dfs_arith(i, stock, total):
            if i == N:
                average = total / N
                if option_type == 'call':
                    payoff = np.maximum(average - E, 0)
                    V[N].append(payoff)
                    return payoff
                elif option_type == 'put':
                    payoff = np.maximum(E - average, 0)
                    V[N].append(payoff)
                    return payoff
            else:
                down = dfs_arith(i + 1, stock * d, total + stock * d)
                up = dfs_arith(i + 1, stock * u, total + stock * u)

                value = np.exp(-r * dt) * (p * up + (1 - p) * down)
                V[i].append(value)
                return value

        dfs_arith(0, S0, 0)
        
    return V

                
def barrier_payoff(S, E, r, sig, T, N, B, barrier_type, option_type = 'call'): 
    dt = T/N
    u = np.exp(sig * np.sqrt(dt))
    d = np.exp(-sig * np.sqrt(dt))
    p = (np.exp(r * dt) - d)/(u - d)
    ST = S[-1]
    V = [0] * (N + 1)  
    C = [0] * (N + 1)
    A = [0] * (N + 1)

    if option_type == 'call':
        V[N] = np.maximum(ST - E, 0)
        C[N] = V[N].copy()
        A[N] = V[N].copy()
    elif option_type == 'put':
        V[N] = np.maximum(E - ST, 0)
        C[N] = V[N].copy()
        A[N] = V[N].copy()

    if barrier_type == 'down_and_out':
        V[N][S[-1] <= B] = 0
        for i in range(N, 0, -1):
            down = V[i][:i]
            up = V[i][1:(i + 1)]
            V[i - 1] = np.exp(-r * dt) * (p * up + (1 - p) * down)      
            V[i - 1][S[i - 1] <= B] = 0        
        
    elif barrier_type == 'up_and_out':
        V[N][S[-1] >= B] = 0
        for i in range(N, 0, -1):
            down = V[i][:i]
            up = V[i][1:(i + 1)]
            V[i - 1] = np.exp(-r * dt) * (p * up + (1 - p) * down)      
            V[i - 1][S[i - 1] >= B] = 0

    elif barrier_type == 'down_and_in':  #here A denotes out option and C denotes euro option
        A[N][S[-1] <= B] = 0
        V[N] = C[N] - A[N]
        for i in range(N, 0, -1):
            down = A[i][:i]
            up = A[i][1:(i + 1)]
            A[i - 1] = np.exp(-r * dt) * (p * up + (1 - p) * down)      
            A[i - 1][S[i - 1] <= B] = 0  

            down = C[i][:i]
            up = C[i][1:(i + 1)]
            C[i - 1] = np.exp(-r * dt) * (p * up + (1 - p) * down)   
            V[i - 1] = C[i - 1] - A[i - 1]

    elif barrier_type == 'up_and_in':
        A[N][S[-1] >= B] = 0
        V[N] = C[N] - A[N]
        for i in range(N, 0, -1):
            down = A[i][:i]
            up = A[i][1:(i + 1)]
            A[i - 1] = np.exp(-r * dt) * (p * up + (1 - p) * down)      
            A[i - 1][S[i - 1] >= B] = 0  

            down = C[i][:i]
            up = C[i][1:(i + 1)]
            C[i - 1] = np.exp(-r * dt) * (p * up + (1 - p) * down)   
            V[i - 1] = C[i - 1] - A[i - 1]    
    
    return V   
            

def american_payoff(S, E, r, sig, T, N, option_type = 'put'):
    dt = T/N
    u = np.exp(sig * np.sqrt(dt))
    d = np.exp(-sig * np.sqrt(dt))
    p = (np.exp(r * dt) - d)/(u - d)
    ST = S[-1]
    V = [0] * (N + 1)


    if option_type == 'call':
        return european_payoff(S, E, r, sig, T, N, option_type = 'call')

    elif option_type == 'put':
        V[N] = np.maximum(E - ST, 0)
        for i in range(N, 0, -1):
            down = V[i][:i]
            up = V[i][1:(i + 1)]
            V[i - 1] = np.exp(-r * dt) * (p * up + (1 - p) * down)
            mask = (E - S[i - 1] >= V[i - 1]) & (E - S[i - 1] > 0)
            V[i - 1][mask] = E - S[i - 1][mask]

    return V

