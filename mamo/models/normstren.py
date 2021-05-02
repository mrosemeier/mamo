import numpy as np
from scipy import optimize


def ttg(T, Tg):
    '''
    Transform T and Tg to T/Tg in K * K**-1
    :param: T: float test ambient temperature in degree C
    :param: Tg: float glass transition temperature (Tg_mid) in degree C
    :return: T/Tg: float in K/K
    '''
    T0 = 273.15
    return (T0 + T) / (T0 + Tg)


def Rt_fiedler(ttg, m, Rt0):
    ''' Returns static strengths

    :param: m: float slope
    :param: Rt0: float strength at 0K    
    '''
    return m * ttg + Rt0


def fit_fiedler(ttg, Rt):
    ''' 
    Fits linearly static strength to T/Tg
    :param: T/Tg: float in K/K
    :return: m: float slope
    :return: Rt0: float strength at 0K
    '''

    xdata = ttg
    ydata = Rt
    (m, Rt0), _ = optimize.curve_fit(
        Rt_fiedler, xdata, ydata, p0=[200.E6, np.average(Rt)])

    return m, Rt0


def Rt_norm_fiedler(ttg_in, ttg_out, Rt_in, m):
    ''' 
    Normalizes linearly static strength to a T/Tg
    Source: Rosemeier and Antoniou 2021
    :param: ttg_in: float in K/K
    :param: ttg_out: float in K/K
    :param: Rt_in: float in K/K
    :param: m: float slope
    :return: Rt_out: float normalized strength
    '''

    return m * (ttg_out - ttg_in) + Rt_in
