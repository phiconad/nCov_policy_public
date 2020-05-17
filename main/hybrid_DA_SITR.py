import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import optimize
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from numpy import log

#os.chdir("github\\nCov-policyforecast")


##SITR

def initial_R(observation):
    obs = observation #infected people
    #plt.plot(obs) # behave like exponential process

    #induce stationarity
    I1_obs = np.diff(obs)
    I0_obs = np.diff(I1_obs)
    #plt.plot(I0_obs)

    #univariate arma
    def arma_test(observation):
        
        I0_obs = observation

        #check for stationarity
        result = adfuller(I0_obs)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])

        #check for optimal lag order and estimate arma
        arma_mod10 = sm.tsa.ARMA(I0_obs, (1,0)).fit(disp=False)
        print(arma_mod10.params)

        #get residual covariace matrix
        errors = arma_mod10.resid
        R_estimate = np.var(errors)

        return(R_estimate)
    
    #multivariate var
    def var_test():
        pass
    
    R_estimate = arma_test(observation=I0_obs)

    return(R_estimate)



def initial_R_ratio(observation):
    obs = observation

    #induce stationarity
    I1_obs = np.diff(obs)
    I0_obs = np.diff(I1_obs)
    #plt.plot(I0_obs)
    def relative_growth(obs):
        I1_obs = obs
        I0_rates = np.diff(I1_obs)/I1_obs[0:-1]
        return(I0_rates)

    I0_rates = relative_growth(obs=I1_obs)


    #univariate arma
    def arma_test(observation):
        
        I0_obs = observation

        #check for stationarity
        result = adfuller(I0_obs)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])

        #check for optimal lag order and estimate arma
        arma_mod10 = sm.tsa.ARMA(I0_obs, (1,0)).fit(disp=False)
        print(arma_mod10.params)

        #get residual covariace matrix
        errors = arma_mod10.resid
        R_estimate = np.var(errors)

        return(R_estimate)
    
    #multivariate var
    def var_test():
        pass
    
    R_estimate = arma_test(observation=I0_rates)

    return(R_estimate)




def build_ensemble(X_b,N):
    N = N # ensemble number
    m,n = np.shape(X_b)
    ensemble = np.zeros(shape=(N,n))
    V_ens = np.zeros(shape=(N,n))

    #create ensemble
    for i, subvector in enumerate(np.split(X_b,N)):
        mu = np.mean(subvector,axis = 0) 
        sigma = np.std(subvector,axis = 0) 
        ensemble[i] = np.random.normal(mu,sigma) #draw from multivariate normal
        
    ensemble_mean = np.mean(ensemble,axis=0)
    
    #create standard dev matrix nxN
    for i in range(N):

        V_ens[i] = ensemble[i]-ensemble_mean

    B_star = V_ens.T@V_ens

    return(V_ens, B_star)

def fit_S0_I0_beta(T_obs, R_obs, ndiv=20):

    def fitting_error(SIbeta):
        S, I, beta = SIbeta
        x0_f = [S, I, T0, R0]
        p0_f = [beta, alpha0, gamma0]

        x_m = forwrad_SITR_dt(x0_f, p0_f, len(R_obs))
        T_m = [x[2] for x in x_m]
        R_m = [x[3] for x in x_m]
        mse_loss = np.sum(np.square(np.array(T_m) - np.array(T_obs))/np.array(T_obs))
        return mse_loss

    rranges = ((200e4, 500e4), (1000, 1e4), (0, 2))
    SIbeta_est = optimize.brute(fitting_error, ranges=rranges, Ns=ndiv, finish=optimize.fmin)

    S0_est, I0_est, beta0_est = SIbeta_est
    x0_est = [S0_est, I0_est, T0, R0]
    p0_est = [beta0_est, alpha0, gamma0]

    return x0_est, p0_est

def forwrad_SITR(x0, p0):
    S0, I0, T0, R0 = x0
    N0 = S0 + I0 + T0 + R0
    beta0, alpha0, gamma0 = p0
    S1 = S0 - beta0*S0/N0*I0
    I1 = I0 + beta0*S0/N0*I0 - alpha0*I0
    T1 = T0 + alpha0*I0 - gamma0*T0
    R1 = R0 + gamma0*T0    
    x1 = [S1, I1, T1, R1]
    return x1

def forwrad_SITR_ratio(x0, p0):
    S0, I0, T0, R0 = x0
    N0 = S0 + I0 + T0 + R0
    beta0, alpha0, gamma0 = p0
    S1 = S0 - beta0*S0/N0*I0
    I1 = I0 + beta0*S0/N0*I0 - alpha0*I0
    T1 = T0 + alpha0*I0 - gamma0*T0
    R1 = R0 + gamma0*T0    
    x1 = [S1, I1, T1, R1]
    return x1, N0



def forwrad_SITR_dt(x0, p0, dt):
    x_m = []
    n = []
    for k in range(int(dt)):
        if k == 0:
            x_m.append(x0)
        else:
            x_m.append(forwrad_SITR(x_m[k - 1], p0))
    return x_m

def forwrad_SITR_dt_ratio(x0, p0, dt):
    x_m = []
    n = []
    for k in range(int(dt)):
        if k == 0:
            x_m.append(x0)
            n = 0
        else:
            x_m.append(forwrad_SITR_ratio(x_m[k - 1], p0)[0])
            n = forwrad_SITR_ratio(x_m[k - 1], p0)[1]
    return x_m, n






#x0_prior, p0_prior = fit_S0_I0_beta(T_obs, R_obs, ndiv=50)


def plot_forward(x0, p0, days, fig):
    X = forwrad_SITR_dt(x0, p0, len(days))
    S = [x[0] for x in X]
    I = [x[1] for x in X]
    T = [x[2] for x in X]
    R = [x[3] for x in X]

    ax = fig.gca()
    ax.plot(days, np.array(I), 'r-', label="I(t)")
    ax.plot(days, T, 'g-', label="T(t)")
    ax.plot(days, R, 'k-', label="R(t)")
    ax.set_xlabel('Days')
    ax.set_ylabel('Number')
    ax.grid()

    plt.show()

    return S,I,T,R

def plot_forward_ratio(x0, p0, days, fig):
    X, N0 = forwrad_SITR_dt_ratio(x0, p0, len(days))

    S = [x[0] for x in X]
    I = [x[1] for x in X]
    T = [x[2] for x in X]
    R = [x[3] for x in X]

    S_r = S/N0
    I_r = I/N0
    T_r = T/N0
    R_r = R/N0

    ax = fig.gca()
    ax.plot(days, np.array(I_r), 'r-', label="I(t)")
    ax.plot(days, T_r, 'g-', label="T(t)")
    ax.plot(days, R_r, 'k-', label="R(t)")
    ax.set_xlabel('Days')
    ax.set_ylabel('Number')
    ax.grid()

    plt.show()


    return S_r,I_r,T_r,R_r


if __name__ == "__main__":

    # Load wuhan data
    df = pd.read_csv('data/DXY/wuhan_history.csv')
    df = df.loc[(df['time'] >= '2020-01-21') & (df['time'] <= '2020-02-30')]
    days = np.arange(0, len(df))
    R_obs = df['cum_heal'].to_numpy() + df['cum_dead'].to_numpy()
    C_obs = df['cum_confirm'].to_numpy()
    T_obs = C_obs - R_obs
    C0 = C_obs[0]
    T0 = T_obs[0]
    R0 = R_obs[0]

    alpha0 = 1/10
    gamma0 = 1/38


    #recheck code for new data
    if 0:
        df_global = load_globaldata(country_selector='United Kingdom')
        df = df_global

        df = df.iloc[20:]
        days = np.arange(0, len(df))
        R_obs = df['cum_heal'].to_numpy() + df['cum_dead'].to_numpy()
        C_obs = df['cum_confirm'].to_numpy()
        T_obs = C_obs - R_obs
        C0 = C_obs[0]
        T0 = T_obs[0]
        R0 = R_obs[0]


    test = plt.figure()
    t = days
    t = np.array(list(range(0,200)))

    S_p,I_p,T_p,R_p = plot_forward(x0=x0_prior, p0=p0_prior, days=t, fig=test)
    X_p = plot_forward(x0=x0_prior, p0=p0_prior, days=t, fig=test)

    beta_p,alpha_p,gamma_p = p0_prior




    beta_list = np.linspace(beta_p*0.8, beta_p*1.2,10)
    alpha_list = np.linspace(alpha_p*0.8, alpha_p*1.2,10)
    gamma_list = np.linspace(gamma_p*0.8, gamma_p*1.2,10)

    X_b_list = []
    for beta_p in beta_list:
        for alpha_p in alpha_list:
            for gamma_p in gamma_list:

                X_b = plot_forward(x0=x0_prior, p0=[beta_p, alpha_p,gamma_p], days=t, fig=test)
                X_b_list.append(X_b)

    X_b_list = np.array(X_b_list)



    N=20
    # run all generated X_b series 
    B_list ,V_list = [],[]
    for x in X_b_list:
        X_b = x.T
        v, b = build_ensemble(X_b,N=N)
        B_list.append(b)
        V_list.append(v)

        np.shape(B_list)
        #take the mean of 16x3x3 structure to 1x3x3
        B_avg = np.mean(B_list,axis=0)




    x0_prior, p0_prior = fit_S0_I0_beta(T_obs, R_obs, ndiv=50) 

    beta_list = np.linspace(beta_p*0.8, beta_p*1.2,10)
    alpha_list = np.linspace(alpha_p*0.8, alpha_p*1.2,10)
    gamma_list = np.linspace(gamma_p*0.8, gamma_p*1.2,10)

    X_b_list = []
    for beta_p in beta_list:
        for alpha_p in alpha_list:
            for gamma_p in gamma_list:


                X_b = plot_forward_ratio(x0=x0_prior, p0=[beta_p, alpha_p,gamma_p], days=t, fig=test)
                X_b_list.append(X_b)

    X_b_list = np.array(X_b_list)

    N=20
    # run all generated X_b series 
    B_list ,V_list = [],[]
    for x in X_b_list:
        X_b = x.T
        v, b = build_ensemble(X_b,N=N)
        B_list.append(b)
        V_list.append(v)

        np.shape(B_list)
        #take the mean of 16x3x3 structure to 1x3x3
        B_avg_ratio = np.mean(B_list,axis=0)




    Q_prior = B_avg
    Q_prior_ratio = B_avg_ratio

    R_prior = initial_R(T_obs)
    R_prior_ratio = initial_R_ratio(T_obs)


