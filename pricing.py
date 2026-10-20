from payoffs import *
from simulation import *

def price_option(S0, E, r, sig, T, 
                 i = 0,  
                 N = None,                   # i means the ith time point we are considering 
                 option_type = 'call',
                 option_style = 'european',
                 barrier = None,
                 barrier_type = None,
                 asian_average = 'arithmetic'): 
    
    if not N:
        N = int(T / 0.5) if T < 30 else int(T)     

    S = simulate_tree(S0, sig, T, N)

    if option_style == 'european':
        V0 = european_payoff(S, E, r, sig, T, N, option_type)[i]

    elif option_style == 'american':
        V0 = american_payoff(S, E, r, sig, T, N, option_type)[i]

    elif option_style == 'barrier':
        V0 = barrier_payoff(S, E, r, sig, T, N, barrier, barrier_type, option_type)[i]

    elif option_style == 'asian': #目前control功能只能和我们的asian arithemetic同时出现
        V0 = asian_payoff(S, E, r, sig, T, N, option_type, asian_average)[i]

    return V0

