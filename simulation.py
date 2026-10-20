import numpy as np
from scipy.stats import binom

def simulate_tree(S0, sig, T, N): #i takes value from 0 to N
    dt = T / N
    u = np.exp(sig * np.sqrt(dt))
    d = np.exp(-sig * np.sqrt(dt))
    upwr = u ** np.arange(0, N + 1)
    dowr = d ** np.arange(N, -1, -1)
    S = [[] for _ in range(N+1)]
    for i in range(N+1):
        up = upwr[:(i + 1)]
        do = dowr[(N - i):]
        S[i] = S0 * up * do

    return S

def simulate_probtree(r, sig, T, N):
    dt = T / N
    u = np.exp(sig * np.sqrt(dt))
    d = np.exp(-sig * np.sqrt(dt))   
    p = (np.exp(r * dt) - d)/(u - d) 
    P = [[] for _ in range(N + 1)]
    for i in range(N+1):
        P[i] = binom.pmf(np.arange(i + 1), i, p)
        
    return P

def barrier_tree(S0, E, r, sig, T, N, B,
                 option_type = 'call',
                 barrier_type = 'up_and_out',
                 initial_state = True): 
    dt = T / N                                
    u = np.exp(sig * np.sqrt(dt))
    d = np.exp(-sig * np.sqrt(dt))
    p = (np.exp(r * dt) - d)/(u - d) 
    mask_tree = [[] for _ in range(N + 1)]
    option_price = [[] for _ in range(N + 1)]
    asset = [[] for _ in range(N + 1)]
    P = [[] for _ in range(N + 1)]
    Op = [[] for _ in range(N + 1)]  # this is for up_and_in and down_and_in
    res = [[] for _ in range(N + 1)] 

    def dfs(i, prob, stock, mark):  
        nonlocal mask_tree, option_price, asset, P, Op, res

        if i <= N:
            asset[i].append(stock)
            P[i].append(prob)

            if barrier_type == 'up_and_out' or barrier_type == 'up_and_in':
                if stock >= B:
                    mark = False
            elif barrier_type == 'down_and_out' or barrier_type == 'down_and_in':
                if stock <= B:
                    mark = False
            mask_tree[i].append(mark)

            if i == N:
                if option_type == 'call':
                    VT = np.maximum(stock - E, 0)
                    OT = VT.copy()
                    VT = VT if mark else 0
                    option_price[N].append(VT)
                    Op[N].append(OT)
                    res[N].append(OT - VT)
                    return OT, VT
                elif option_type == 'put':
                    VT = np.maximum(E - stock, 0)
                    OT = VT.copy()
                    VT = VT if mark else 0
                    option_price[N].append(VT)
                    Op[N].append(OT)
                    res[N].append(OT - VT)
                    return OT, VT

            down = dfs(i + 1, prob * (1 - p), stock * d, mark)
            up = dfs(i + 1, prob * p, stock * u, mark)
            V = np.exp(-r * dt) * (p * up[1] + (1 - p) * down[1])
            O = np.exp(-r * dt) * (p * up[0] + (1 - p) * down[0])
            V = V if mark else 0
            option_price[i].append(V)
            Op[i].append(O)
            res[i].append(O - V)
            return O, V

    dfs(0, 1, S0, initial_state)

    if barrier_type == 'up_and_out' or barrier_type == 'down_and_out':
        return {'option_price': option_price, 'asset_price': asset, 
                'probability': P, 'mask_tree': mask_tree}
    elif barrier_type == 'up_and_in' or barrier_type == 'down_and_in':
        return {'option_price': res, 'asset_price': asset, 
                'probability': P, 'mask_tree': mask_tree}


def asian_tree(S0, E, r, sig, T, N,   
               option_type = 'call', 
               asian_average = 'arithmetic',
               total = None,
               total_N = None):  #total_N is only used to ensure the consistency while averaging
    
    if total == None:
        total = 1 if asian_average == 'geometric' else 0

    if total_N == None:
        total_N = N
    
    dt = T/N
    u = np.exp(sig * np.sqrt(dt))  
    d = np.exp(-sig * np.sqrt(dt))
    p = (np.exp(r * dt) - d)/(u - d)
    V = [[] for _ in range(N + 1)]
    asset = [[] for _ in range(N + 1)]
    P = [[] for _ in range(N + 1)]
    tot = [[] for _ in range(N + 1)]

    if asian_average == 'geometric':
        def dfs_geo(i, prob, stock, t):
            if i <= N:
                P[i].append(prob)
                asset[i].append(stock)
                tot[i].append(t)

                if i == N:
                    average = t ** (1/total_N)
                    if option_type == 'call':
                        payoff = np.maximum(average - E, 0)
                        V[N].append(payoff)
                        return payoff
                    elif option_type == 'put':
                        payoff = np.maximum(E - average, 0)
                        V[N].append(payoff)
                        return payoff
                
                elif i < N:
                    down = dfs_geo(i + 1, prob * (1 - p), stock * d, t * stock * d)
                    up = dfs_geo(i + 1, prob * p, stock * u, t * stock * u)
                    value = np.exp(-r * dt) * (p * up + (1 - p) * down)
                    V[i].append(value)
                    return value
                           
        dfs_geo(0, 1, S0, total)
     
    elif asian_average == 'arithmetic':
        def dfs_arith(i, prob, stock, t):
            if i <= N:
                P[i].append(prob)
                asset[i].append(stock)
                tot[i].append(t)

                if i == N:
                    average = t / total_N
                    if option_type == 'call':
                        payoff = np.maximum(average - E, 0)
                        V[N].append(payoff)
                        return payoff
                    elif option_type == 'put':
                        payoff = np.maximum(E - average, 0)
                        V[N].append(payoff)
                        return payoff
                elif i < N:
                    down = dfs_arith(i + 1, prob * (1 - p),stock * d, t + stock * d)
                    up = dfs_arith(i + 1, prob * p, stock * u, t + stock * u)
                    value = np.exp(-r * dt) * (p * up + (1 - p) * down)
                    V[i].append(value)
                    return value
        
        dfs_arith(0, 1, S0, total)
        
    return {'option_price': V, 'asset_price': asset, 
                'probability': P, 'total': tot}              

        





    

