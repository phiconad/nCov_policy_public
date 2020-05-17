import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
import os
import datetime
from numpy import linalg as LA

#%%

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


def forwrad_SITR_dt(x0, p0, dt):
    x_m = []
    for k in range(int(dt)):
        if k == 0:
            x_m.append(x0)
        else:
            x_m.append(forwrad_SITR(x_m[k - 1], p0))
    return x_m


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

    return fig


def load_globaldata(country_selector = 'United Kingdom'):
    df_conf = pd.read_csv('data\\Kaggle\\time_series_covid_19_confirmed.csv')
    df_rec = pd.read_csv('data\\Kaggle\\time_series_covid_19_recovered.csv')
    df_dec = pd.read_csv('data\\Kaggle\\time_series_covid_19_deaths.csv')

    def extract_country(df_raw):
        df_conf = df_raw
        
        #select countries
        df_countries = df_conf[df_conf['Country/Region'].str.match('Italy|Germany|United Kingdom|France|US|Korea|Japan|Spain')]
        
        #select only national level
        df_countries['Province/State'] = df_countries['Province/State'].fillna(df_conf['Country/Region'])
        df_countries = df_countries[df_countries['Province/State'].str.match('Italy|Germany|United Kingdom|France|US|Korea|Japan|Spain')]

        #transpose
        df_countries = df_countries.drop(['Province/State','Lat', 'Long'],axis=1)
        df_countries = df_countries.transpose()

        #reset indeces
        df_countries.columns = df_countries.loc['Country/Region',:]
        df_countries = df_countries.drop(['Country/Region'])

        df_countries.plot()

        return(df_countries)
    
    df_conf = extract_country(df_conf)
    df_rec = extract_country(df_rec)
    df_dec = extract_country(df_dec)

    #transform to wuhan data
    #country_selector = 'United Kingdom'
    df_wuhan = df # select wuhan dataset
    df_global = df_wuhan.reindex(range(0,len(df_conf)))

    df_global.index = df_conf.index
    df_global['time'] = df_conf.index
    df_global['cum_confirm'] = df_conf[country_selector]
    df_global['cum_heal'] = df_rec[country_selector]
    df_global['cum_dead'] = df_dec[country_selector]
    df_global.plot()

    return(df_global)


#load global data
if 0:
    df_global = load_globaldata(country_selector='United Kingdom')
    df = df_global
    df = df.iloc[30:] #set timeframe

    days = np.arange(0, len(df))
    R_obs = df['cum_heal'].to_numpy() + df['cum_dead'].to_numpy()
    C_obs = df['cum_confirm'].to_numpy()
    T_obs = C_obs - R_obs
    C0 = C_obs[0]
    T0 = T_obs[0]
    R0 = R_obs[0]



# Load wuhan data
if 1:
    df = pd.read_csv('data/DXY/wuhan_history.csv')
    df = df.loc[(df['time'] >= '2020-01-21') & (df['time'] <= '2020-02-30')]
    days = np.arange(0, len(df))
    R_obs = df['cum_heal'].to_numpy() + df['cum_dead'].to_numpy()
    C_obs = df['cum_confirm'].to_numpy()
    T_obs = C_obs - R_obs
    C0 = C_obs[0]
    T0 = T_obs[0]
    R0 = R_obs[0]



# fixed parameters
alpha0 = 1/10 #I to T ,  taken from literature
gamma0 = 1/38 # T R ,model calculation fit

#weight loss function terms
j1=1
j2=0

#set covariance matrices
cov_obs = [[R_prior]]
cov_model = Q_prior



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



def assimilate_window(x0_prior, p_prior, T_obs, R_obs,obs_error,model_error,j1weight,j2weight, ndiv=20):

    S0_prior, I0_prior, T0, R0 = x0_prior
    N0 = np.sum(x0_prior)

    beta0, alpha0, gamma0 = p_prior

    E_0 = obs_error
    E_R = model_error


    loss2_weightj1 = j1weight
    loss2_weightj2 = j2weight

    ## step 1. assimilate x0_a
    def J1_loss(SI):
        S, I = SI
        x0 = [S, I, T0, R0]
        x_m = forwrad_SITR_dt(x0, p_prior, len(R_obs)) 
        T_m, R_m = [x[2] for x in x_m], [x[3] for x in x_m]  colum of the 4 states


        loss1 = np.sum(np.square(T_m - np.array(T_obs))) / len(T_obs)  *np.linalg.inv(E_0)[0][0] 

        loss2 = np.square(S-S0_prior)*np.linalg.inv(E_R)[0][0] + np.square(I-I0_prior)*np.linalg.inv(E_R)[1][1]   

        loss2_star = np.square(S-S0_prior)*np.linalg.inv(E_R)[0][0] + np.square(I-I0_prior)*np.linalg.inv(E_R)[1][1]

        loss2_star = loss2_star + np.square(S-S0_prior)*np.linalg.inv(E_R)[0][1] +np.square(I-I0_prior)* np.linalg.inv(E_R)[0][1]

        loss = loss1 + loss2_weightj1 *loss2_star 
        return loss

    rranges = ((S0_prior*0.8, S0_prior*1.2), (I0_prior*0.8, I0_prior*1.2)) 

    SI_est = optimize.brute(J1_loss, ranges=rranges, Ns=ndiv, finish=optimize.fmin)
    S0_a, I0_a = SI_est
    x0_a = [S0_a, I0_a, T0, R0]  

    # step 2. calibrate system parameter
    def J2_loss(beta):
        p_f = [beta, alpha0, gamma0]
        x_m = forwrad_SITR_dt(x0_a, p_f, len(R_obs))
        T_m, R_m = [x[2] for x in x_m], [x[3] for x in x_m]
        loss1 = np.sum(np.square(T_m - np.array(T_obs))) / len(T_obs)*np.linalg.inv(E_0)[0][0]
        loss2 = np.abs(beta-beta0)   # smooth change #penalizes derivation of beta
        loss = loss1 + loss2_weightj2*loss2
        return loss
    rranges = ((0, 1),)
    beta_est = optimize.brute(J2_loss, ranges=rranges, Ns=20, finish=None) #evaluate grid, then do gradient above
    p_a = [beta_est, alpha0, gamma0]

    x_a_win = forwrad_SITR_dt(x0_a, p_a, len(R_obs))
    return x_a_win, p_a



#sliding window to save all observations

tao = 5
x0_prior, p0_prior = fit_S0_I0_beta(T_obs, R_obs, ndiv=50)
# save assimilating history
x_a_series, p_a_series = [], []
x_prior_series, x_pred_series = [], []

for k in range(tao-1, len(days)):
    T_win = T_obs[k + 1 - tao: k + 1] #assimilate windows
    R_win = R_obs[k+1-tao: k+1] 
    if k == tao-1:

        # assimilate x with window length tao
        x_a_win, p_a = assimilate_window(x0_prior, p0_prior, T_win, R_win, ndiv=50,obs_error=cov_obs,model_error=cov_model,j1weight=j1,j2weight=j2)
        # warm up for first window
        x_a_series = x_a_win 
        p_a_series = [p_a] * tao
        x_pred_series = [[np.nan] * 4] * tao 
        # predict window next state
        x_pred_series.append(forwrad_SITR(x_a_win[-1], p_a)) 
        x0_prior = x_a_win[1]
        p0_prior = p_a
        x_prior_series.append(x0_prior)
    else:
        # assimilate x with window length tao
        x_a_win, p_a = assimilate_window(x0_prior, p0_prior, T_win, R_win, ndiv=50,obs_error=cov_obs,model_error=cov_model,j1weight=j1,j2weight=j2)
        x_a_series.append(x_a_win[-1])
        x_pred_series.append(forwrad_SITR(x_a_win[-1], p_a))
        p_a_series.append(p_a)
        x0_prior = x_a_win[1]
        p0_prior = p_a
        x_prior_series.append(x0_prior)



if 1:
    print("weight of loss j1 : ",j1)
    print("weight of loss j2 : ",j2)
    print("model error covariance: ",cov_model)
    print("observation error covariance: ",cov_obs)


    # plot prediction
    fig = plt.figure()
    I_pred = np.array(x_pred_series)[:-1, 1]
    T_pred = np.array(x_pred_series)[:-1, 2]
    R_pred = np.array(x_pred_series)[:-1, 3]
    plt.plot(days, I_pred, 'b.-', label='predict_I')
    plt.plot(days, T_pred, 'g-', label='predict_being treated')
    plt.plot(days, T_obs, 'go', label='Being treated')
    plt.plot(days, R_pred, 'k-', label='predict_Recovered')
    plt.plot(days, R_obs, 'k*', label='Recovered')
    plt.legend()
    plt.xticks(days, df['time'])
    plt.gca().xaxis.set_tick_params(rotation=90, labelsize=8)
    plt.grid()
    plt.show()



    #MRSFE
    T_MRSFE = np.sum(np.sqrt(np.square(T_pred[tao:] - T_obs[tao:])))/len(T_obs[tao:])
    R_MRSFE = np.sum(np.sqrt(np.square(R_pred[tao:] - R_obs[tao:])))/len(R_obs[tao:])
    print("T_MRSFE :",T_MRSFE,T_MRSFE_norm)
    print("R_MRSFE :",R_MRSFE,R_MRSFE_norm)  


    # plot beta
    plt.figure()
    beta_a = np.array(p_a_series)[:, 0]
    plt.stem(days, beta_a)
    plt.grid()
    plt.title('beta')
    plt.xticks(days, df['time'])
    plt.gca().xaxis.set_tick_params(rotation=90, labelsize=8)
    plt.ylabel('beta')
    plt.show()


#Long Range Forecast Plots
if 1:

    def plot_forward_future(x0, p0, days,x_a_series ,fig):

        X = forwrad_SITR_dt(x0, p0, len(days))
        S = [x[0] for x in X]
        I = [x[1] for x in X]
        T = [x[2] for x in X]
        R = [x[3] for x in X]


        #mergedata
        S_b = np.array(x_pred_series)[:,0]
        I_b = np.array(x_pred_series)[:,1]
        T_b = np.array(x_pred_series)[:,2]
        R_b = np.array(x_pred_series)[:,3]

        S_full = np.append(S_b,S)
        I_full = np.append(I_b,I)
        T_full = np.append(T_b,T)
        R_full = np.append(R_b,R)

        #read long date vector create in excel
        longdate = pd.read_excel('datevector_excel.xls',squeeze=True)
        longdate = longdate[6:]

        plt.figure()
        plt.plot(longdate[:len(T_full)], np.array(I_full), 'r-', label="I(t)")
        plt.plot(longdate[:len(T_full)], T_full, 'g-', label="T(t)")
        plt.plot(longdate[:len(T_full)], R_full, 'k-', label="R(t)")

        ax = plt.gca()
        #ax.axvline(x=.5, ymin=0.25, ymax=0.75)
        ax.set_xlabel('Days')
        #ax.set_ylabel('Number')
        ax.xaxis_date()
        ax.xaxis.set_tick_params(rotation=40, labelsize=8)
        plt.grid()
        plt.legend()
        plt.savefig('long_fc.pdf')
        plt.show()


        return fig

    days = np.arange(0, 223)
    p_a_series[-1:]
    x_a_series[-1:]
    t = np.array(list(range(0,150)))

    plot_forward_future(x0=x_a_series[-1:][0], p0=p_a_series[-1:][0], days=t,x_a_series = x_pred_series, fig="test")

