from pricing import *
from simulation import *
import numpy as np
from math import comb

def delta(S0, E, r, sig, T, i, h,     #i denotes the ith time point  0 <= i < N  #h is only for asian option
                 N = None,
                 option_type = 'call',
                 option_style = 'european',
                 barrier = None,
                 barrier_type = None,
                 asian_average = 'arithmetic'): 

    if not N:
        dt = 0.5 if T < 30 else 1
        N = int(T/dt)
    else:
        dt = T/N

    if option_style == 'european' or option_style == 'american':
        V = price_option(S0, E, r, sig, T, i + 1,
                        N,
                        option_type, 
                        option_style,
                        barrier,
                        barrier_type,
                        asian_average)
        X = simulate_tree(S0, sig, T, N)
        S = X[i + 1]
        Sm = X[i]
        Y = simulate_probtree(r, sig, T, N)
        P = Y[i]
        if i > 0:
            Pm = Y[i - 1]
        else:
            Pm = None

        Vtminu = np.array(V[:-1])
        Vtplus = np.array(V[1:])

        Sminu = np.array(S[:-1])
        Splus = np.array(S[1:])

        de = (Vtplus - Vtminu)/(Splus - Sminu)
        es = np.sum(P * de)

    elif option_style == 'barrier':
        X = barrier_tree(S0, E, r, sig, T, N, barrier,
                        option_type,
                        barrier_type)
        V = X['option_price'][i + 1]
        S = X['asset_price'][i + 1]
        Sm = X['asset_price'][i]
        P = X['probability'][i]
        if i > 0:
            Pm = X['probability'][i - 1]
        else:
            Pm = None

        Vminu = np.array(V[::2])
        Vplus = np.array(V[1::2])

        Sminu = np.array(S[::2])
        Splus = np.array(S[1::2])

        prob = np.array(P)

        de = (Vplus - Vminu)/(Splus - Sminu)
        es = np.sum(prob * de)

    elif option_style == 'asian':
        X = asian_tree(S0, E, r, sig, T, N,
                       option_type,
                       asian_average)
        S = X['asset_price'][i]
        tot = X['total'][i]
        prob = X['probability'][i]
        de = np.zeros((2 ** i,))

        for j, s in enumerate(S):
            Xplus = asian_tree(s + h, E, r, sig, T - i * dt, N - i,
                               option_type, 
                               asian_average,
                               tot[j],
                               total_N=N)

            Xminu = asian_tree(s - h, E, r, sig, T - i * dt, N - i,
                               option_type,
                               asian_average,
                               tot[j],
                               total_N=N)

            Vplus = Xplus['option_price'][0][0]
            Vminu = Xminu['option_price'][0][0]

            de[j] = (Vplus - Vminu)/(2 * h)

        de = np.array(de)
        prob = np.array(prob)
        es = np.sum(prob * de)
        Pm = prob  
        Sm = None

    return {'delta': de, 'expected_delta':es, 'asset_price':Sm, 'probability':Pm}
    #Note that the asset_price the function returns is the current layer and the probability return is the layer before current layer
    
def gamma(S0, E, r, sig, T, i, h,     #i  0<= i < N - 1   #the gamma function we wrote is two-nodes estimation
                 N = None,
                 option_type = 'call',
                 option_style = 'european',
                 barrier = None,
                 barrier_type = None,
                 asian_average = 'arithmetic'): 
    if not N:
        dt = 0.5 if T < 30 else 1
        N = int(T/dt)
    else:
        dt = T/N

    if option_style == 'european' or option_style == 'american' or option_style == 'barrier':
        X = delta(S0, E, r, sig, T, i + 1, h,
                    N,      
                    option_type, 
                    option_style,
                    barrier,
                    barrier_type,
                    asian_average)

        de = X['delta']
        S = X['asset_price']
        P = X['probability']

        if option_style == 'european' or option_style == 'american':
            deminu = np.array(de[:-1])
            deplus = np.array(de[1:])

            Sminu = np.array(S[:-1])
            Splus = np.array(S[1:])

        elif option_style == 'barrier':
            deminu = np.array(de[::2])
            deplus = np.array(de[1::2])

            Sminu = np.array(S[::2])
            Splus = np.array(S[1::2])

        ga = (deplus - deminu)/(Splus - Sminu)
        P = np.array(P)
        es = np.sum(P * ga)

    elif option_style == 'asian':
        X = asian_tree(S0, E, r, sig, T, N,
                       option_type,
                       asian_average)
        V = X['option_price'][i]
        S = X['asset_price'][i]
        tot = X['total'][i]
        prob = X['probability'][i]
        ga = np.zeros((2 ** i,))

        for j, s in enumerate(S):
            Xplus = asian_tree(s + h, E, r, sig, T - i * dt, N - i,
                               option_type, 
                               asian_average,
                               tot[j],
                               total_N=N)

            Xminu = asian_tree(s - h, E, r, sig, T - i * dt, N - i,
                               option_type,
                               asian_average,
                               tot[j],
                               total_N=N)

            Vplus = Xplus['option_price'][0][0]
            Vminu = Xminu['option_price'][0][0]
            v = V[j]

            ga[j] = (Vplus + Vminu - 2 * v)/(h ** 2)

        ga = np.array(ga)
        prob = np.array(prob)
        es = np.sum(prob * ga)
        
    return {'gamma': ga, 'expected_gamma':es}


def vega(S0, E, r, sig, T, i, h,       # 0 <= i <= N # h denotes the drift in sigma
                 N = None,
                 option_type = 'call',
                 option_style = 'european',
                 barrier = None,
                 barrier_type = None,
                 asian_average = 'arithmetic'): 

    if not N:
        dt = 0.5 if T < 30 else 1
        N = int(T/dt)
        t = i * dt
    else:
        dt = T/N
        t = i * dt

    if option_style == 'european' or option_style == 'american':
        Si = simulate_tree(S0, sig, T, N)[i]
        ve = np.zeros((i + 1,))

        if i < N :
            P = simulate_probtree(r, sig, T, N)
            prob = P[i]

            for j, S in enumerate(Si):
                Vplus = price_option(S, E, r, sig + h, T - t, 0,
                                    N - i,
                                    option_type, 
                                    option_style,
                                    barrier,
                                    barrier_type,
                                    asian_average)
            
                Vminu = price_option(S, E, r, sig - h, T - t, 0,
                                    N - i,
                                    option_type, 
                                    option_style,
                                    barrier,
                                    barrier_type,
                                    asian_average)

                v = (Vplus - Vminu)/(2 * h)
                ve[j] = v

        elif i == N:
            prob = np.zeros(ve.shape)

    elif option_style == 'barrier':
        nN = N - i
        ve = np.zeros((2 ** i,))

        if i < N:
            X = barrier_tree(S0, E, r, sig, T, N, barrier,
                            option_type,
                            barrier_type)
            Si = X['asset_price'][i]
            prob = X['probability'][i]
            mask = X['mask_tree'][i]

            for j, S in enumerate(Si):
                Xplus = barrier_tree(S, E, r, sig + h, T - t, nN,
                                    barrier,
                                    option_type,
                                    barrier_type,
                                    mask[j])   
                Xminu = barrier_tree(S, E, r, sig - h, T - t, nN,
                                    barrier,
                                    option_type,
                                    barrier_type, 
                                    mask[j])  

                Vplus = Xplus['option_price'][0][0] 
                Vminu = Xminu['option_price'][0][0]

                v = (Vplus - Vminu)/(2*h)
                ve[j] = v

        elif i == N:
            prob = np.zeros(ve.shape)
        
         #for the last layer, there is no influence by the change in sigma

    elif option_style == 'asian':
        ve = np.zeros((2 ** i,))

        if i < N:
            X = asian_tree(S0, E, r, sig, T, N,
                        option_type,
                        asian_average)
            S = X['asset_price'][i]
            tot = X['total'][i]
            prob = X['probability'][i]

            for j, s in enumerate(S):
                Xplus = asian_tree(s, E, r, sig + h, T - i * dt, N - i,
                                option_type, 
                                asian_average,
                                tot[j],
                                total_N=N)

                Xminu = asian_tree(s, E, r, sig - h, T - i * dt, N - i,
                                option_type,
                                asian_average,
                                tot[j],
                                total_N=N)

                Vplus = Xplus['option_price'][0][0]
                Vminu = Xminu['option_price'][0][0]

                ve[j] = (Vplus - Vminu)/(2 * h)

        elif i == N:
            prob = np.zeros(ve.shape)           
    
    ve = np.array(ve)
    prob = np.array(prob)
    es = np.sum(prob * ve)
        
    return {'vega': ve, 'expected_vega':es}

  
def rho(S0, E, r, sig, T, i, h,       # 0 <= i <= N # h denotes the drift in sigma
                 N = None,
                 option_type = 'call',
                 option_style = 'european',
                 barrier = None,
                 barrier_type = None,
                 asian_average = 'arithmetic'): 

    if not N:
        dt = 0.5 if T < 30 else 1
        N = int(T/dt)
        t = i * dt
    else:
        dt = T/N
        t = i * dt

    if option_style == 'european' or option_style == 'american':
        Si = simulate_tree(S0, sig, T, N)[i]
        rh = np.zeros((i + 1,))  
             
        if i < N :
            P = simulate_probtree(r, sig, T, N)
            prob = P[i]      
       
            for j, S in enumerate(Si):
                Vplus = price_option(S, E, r + h, sig, T - t, 0,
                                    N - i,
                                    option_type, 
                                    option_style,
                                    barrier,
                                    barrier_type,
                                    asian_average)
            
                Vminu = price_option(S, E, r - h, sig, T - t, 0,
                                    N - i,
                                    option_type, 
                                    option_style,
                                    barrier,
                                    barrier_type,
                                    asian_average)

                v = (Vplus - Vminu)/(2 * h)
                rh[j] = v

        elif i == N:
            prob = np.zeros(rh.shape)

    elif option_style == 'barrier':
        nN = N - i
        rh = np.zeros((2 ** i,))

        if i < N:
            X = barrier_tree(S0, E, r, sig, T, N, barrier,
                            option_type,
                            barrier_type)
            Si = X['asset_price'][i]
            prob = X['probability'][i]
            mask = X['mask_tree'][i]

            for j, S in enumerate(Si):
                Xplus = barrier_tree(S, E, r + h, sig, T - t, nN,
                                    barrier,
                                    option_type,
                                    barrier_type,
                                    mask[j])   
                Xminu = barrier_tree(S, E, r - h, sig, T - t, nN,
                                    barrier,
                                    option_type,
                                    barrier_type, 
                                    mask[j])  

                Vplus = Xplus['option_price'][0][0] 
                Vminu = Xminu['option_price'][0][0]

                v = (Vplus - Vminu)/(2*h)
                rh[j] = v

        elif i == N:
            prob = np.zeros(rh.shape)
        
    elif option_style == 'asian':
        rh = np.zeros((2 ** i,))

        if i < N:
            X = asian_tree(S0, E, r, sig, T, N,
                        option_type,
                        asian_average)
            S = X['asset_price'][i]
            tot = X['total'][i]
            prob = X['probability'][i]

            for j, s in enumerate(S):
                Xplus = asian_tree(s, E, r + h, sig, T - i * dt, N - i,
                                option_type, 
                                asian_average,
                                tot[j],
                                total_N=N)

                Xminu = asian_tree(s, E, r - h, sig, T - i * dt, N - i,
                                option_type,
                                asian_average,
                                tot[j],
                                total_N=N)

                Vplus = Xplus['option_price'][0][0]
                Vminu = Xminu['option_price'][0][0]

                rh[j] = (Vplus - Vminu)/(2 * h)

        elif i == N:
            prob = np.zeros(rh.shape) 
    
    rh = np.array(rh)
    prob = np.array(prob)
    es = np.sum(prob * rh)
        
    return {'rho': rh, 'expected_rho':es}


def theta(S0, E, r, sig, T, i,  # 0 <= i <= N - 2  for the second to last layer we go back to the fourth to last layer  
                 N = None,
                 option_type = 'call',
                 option_style = 'european',
                 barrier = None,
                 barrier_type = None,
                 asian_average = 'arithmetic'): 

    if not N:
        dt = 0.5 if T < 30 else 1
        N = int(T/dt)
        t = i * dt
    else:
        dt = T/N
        t = i * dt

    if option_style == 'european' or option_style == 'american':  
        Vplus = price_option(S0, E, r, sig, T, i + 2, N, 
                            option_type, 
                            option_style,
                            barrier,
                            barrier_type,
                            asian_average)[1:-1]
        
        Vminu = price_option(S0, E, r, sig, T, i, N,
                            option_type, 
                            option_style,
                            barrier,
                            barrier_type,
                            asian_average)

        th = (Vplus - Vminu)/(2 * dt)

        P = simulate_probtree(r, sig, T, N)
        es = np.sum(P[i] * th)

    elif option_style == 'barrier':
        Nplus = N - i - 1
        Nminu = N - i + 1
        tplus = t + dt
        tminu = t - dt
        th = np.zeros((2 ** i,))

        X = barrier_tree(S0, E, r, sig, T, N, barrier,
                        option_type,
                        barrier_type)
        S = X['asset_price'][i]
        prob = X['probability'][i]
        mask = X['mask_tree'][i]

        for j, s in enumerate(S):
            Xplus = barrier_tree(s, E, r, sig, T - tplus, Nplus,
                                barrier,
                                option_type,
                                barrier_type,
                                mask[j])   
            Xminu = barrier_tree(s, E, r, sig, T - tminu, Nminu,
                                barrier,
                                option_type,
                                barrier_type, 
                                mask[j])  

            Vplus = Xplus['option_price'][0][0] 
            Vminu = Xminu['option_price'][0][0] 

            th[j] = (Vplus - Vminu)/(2 * dt)

        th = np.array(th)
        prob = np.array(prob)
        es = np.sum(prob * th)    

#Asian Theta is approximated by a central finite difference across adjacent valuation dates, 
#holding the stock price and accumulated fixing state constant, 
# while preserving the original contract's total fixing count.

    elif option_style == 'asian':
        Nplus = N - i - 1
        Nminu = N - i + 1
        tplus = t + dt
        tminu = t - dt
        th = np.zeros((2 ** i,))

        X = asian_tree(S0, E, r, sig, T, N,
                    option_type,
                    asian_average)
        S = X['asset_price'][i]
        tot = X['total'][i]
        prob = X['probability'][i]

        for j, s in enumerate(S):
            Xplus = asian_tree(s, E, r, sig, T - tplus, Nplus,
                            option_type, 
                            asian_average,
                            tot[j],
                            total_N=N)

            Xminu = asian_tree(s, E, r, sig, T - tminu, Nminu,
                            option_type,
                            asian_average,
                            tot[j],
                            total_N=N)

            Vplus = Xplus['option_price'][0][0]
            Vminu = Xminu['option_price'][0][0]

            th[j] = (Vplus - Vminu)/(2 * dt)

        th = np.array(th)
        prob = np.array(prob)
        es = np.sum(prob * th)  

    return {'theta': th, 'expected_theta':es}
