'''
Created on May 1, 2020

@author: Malo Rosemeier
'''

import numpy as np
import pandas as pd
from scipy import interpolate
import scipy.optimize as optimize
from scipy.special import gamma
import cmath
from collections import OrderedDict


def explogx(x, a, b, c):
    return a * x**-b + c


def explogx1(x, a, b):
    ''' Exponential function that goes through point (1,1) with c=1-a'''
    return a * (x**-b - 1.) + 1.


def poly1dlogx(x, a, b):
    return a * np.log10(x) + b


def akima1d_extrap_right(alp_c, n):
    ''' linearly extrapolates and Akima spline based on the
    gradient in the last point available
    :param: alp_c: Akima1DInterpolator object
    :param: n: x coordinate out of interpolation interval
    :return: alp: extrpolated value
    '''
    nexp_end = alp_c.x[-1]  # last value in interval
    dalp = alp_c.derivative(1)(nexp_end)  # gradient at last point
    alp_end = alp_c(nexp_end)  # last value interpolated value in interval
    b = alp_end - dalp * nexp_end  # intersection with y-axis
    return poly1dlogx(n, dalp, b)


def akima1d_extrap_left(alp_c, n):
    ''' linearly extrapolates and Akima spline based on the
    gradient in the last point available
    :param: alp_c: Akima1DInterpolator object
    :param: n: x coordinate out of interpolation interval
    :return: alp: extrpolated value
    '''
    nexp_end = alp_c.x[0]  # first value in interval
    dalp = alp_c.derivative(1)(nexp_end)  # gradient at first point
    alp_end = alp_c(nexp_end)  # first value interpolated value in interval
    b = alp_end - dalp * nexp_end  # intersection with y-axis
    return poly1dlogx(n, dalp, b)


def p_weibull(x, lmbda, delta, beta):
    ''' Obains probability of a given Weibull distribution for a given variable x
    Source: Castillo 2009 A unified statistical methoddology for modelling
    fatigue damage
    Note: A negative delta flips the Weibull curve about x-axis
    :param: x: float value probability
    :param: lmbda: float Weibull location parameter
    :param: delta: float Weibull scale parameter
    :param: beta: float Weibull shape parameter
    :return: p: float failure probability'''
    return 1. - np.exp(-((x - lmbda) / delta)**beta)


def x_weibull(p, lmbda, delta, beta):
    ''' Obains x value of a Weibull dsitribution for a given probability p
    solve p = 1. - np.exp(-((x - lmbda) / delta)**beta) for x if delta>0
    Note: A negative delta flips the Weibull curve about x-axis
    :param: p: float failure probability
    :param: lmbda: float Weibull location parameter
    :param: delta: float Weibull scale parameter
    :param: beta: float  Weibull shape parameter
    :return: x: float percentile '''
    return lmbda + delta * (-np.log(1. - p))**(1. / beta)


def s_weibull(lmdba, delta, beta):
    ''' Shifley and Lentz 1985, Quick estimation of the three-parameter Weibull
    to describe tree size distributions
    :param: lmdba: float Weibull location parameter
    :param: delta: float Weibull scale parameter
    :param: beta: float Weibull shape parameter
    :return: mu: float mean
    :return: sig: float standard deviation
    '''
    # mean value
    mu_alp = delta * gamma(1. + (1. / beta))  # Eq. 2
    mu = mu_alp + lmdba  # Eq. 2
    # variance
    sig2 = delta**2 * (gamma(1. + 2. / beta) -
                       gamma(1. + (1. / beta))**2)  # Eq. 3
    # standard deviation
    sig = np.sqrt(sig2)

    return mu, sig


def pwm_weibull(xs):
    ''' Fits a three parameter Weibull distribution using PWM
    Sources: Toasa Caiza and Ummenhofer 2011
    Rinne 2008 The Weibull Distribution, p. 473
    :param: xs: array of unsorted random variables
    :return: lmbda: float Weibull location parameter
    :return: delta: float Weibull scale parameter
    :return: beta: float Weibull parameter '''
    n = len(xs)
    # sort xs ascending
    xs_sort = np.sort(xs)

    # eq. 12.21 Rinne
    m0 = 1. / n * np.sum(xs_sort)  # A0

    sum1 = 0.
    for i in range(1, n + 1):
        sum1 += (n - i) / (n - 1.) * xs_sort[i - 1]
    m1 = 1. / n * sum1  # A1

    sum2 = 0.
    for i in range(1, n + 1):
        sum2 += (n - i) * (n - i - 1.) / \
            ((n - 1.) * (n - 2.)) * xs_sort[i - 1]
    m2 = 1. / n * sum2  # A2

    def fun(beta_):
        # eq. 12.23a/b
        # eq. 16 Toasa Caiza and Ummenhofer 2011
        lhs = (3.**(-1. / beta_) - 1.) / (2.**(-1. / beta_) - 1.)
        rhs = (3. * m2 - m0) / (2. * m1 - m0)
        return lhs - rhs

    def grad(beta_):
        # eq. 12.23c
        return 1. / (beta_**2 * (2.**(-1. / beta_) - 1.)**2) *\
            (3.**(-1. / beta_) * (2.**(-1. / beta_) - 1.) *
             np.log(3.) + 2.**(-1. / beta_) *
             (3.**(-1. / beta_) - 1.) * np.log(2.))

    beta = optimize.brentq(fun, a=1E-12, b=1E2)

    # eq. 10 Toasa Caiza and Ummenhofer 2011
    gamma_ = gamma(1. + 1. / beta)
    # eq. 17 Toasa Caiza and Ummenhofer 2011
    delta = (2. * m1 - m0) / ((2.**(-1. / beta) - 1) * gamma_)
    # eq. 18 Toasa Caiza and Ummenhofer 2011
    lmbda = m0 - delta * gamma_

    return lmbda, delta, beta


def R_ratio(sm, sa):
    ''' Stress ratio as function of amplitude and mean
    :param: sm: float mean stress
    :param: sa: float stress amplitude
    :return: R: float stress ratio'''
    smin = sm - abs(sa)
    smax = sm + abs(sa)
    return smin / smax


def sm(sa, R):
    ''' Means stress as function of amplitude and stress ratio
    :param: sa: float stress amplitude
    :param: R: float stress ratio
    :return: sm: float mean stress '''
    if R == 1:
        sm = 0. * sa
    else:
        sm = sa * (1. + R) / (1. - R)
    return sm


def sa(sm, R):
    ''' Stress amplitude as function of mean and stress ratio
    :param: sm: float mean stress 
    :param: R: float stress ratio
    :return: sa: float stress amplitude
    '''
    if R == 1:
        sa = 0. * sm
    else:
        sa = sm / (1. + R) / (1. - R)
    return sa


def smax_sa(sa, R):
    ''' Max stress as function of amplitude and stress ratio
    :param: sa: float stress amplitude
    :param: R: float stress ratio
    :return: smax: float allowable maximum stress'''
    return 2. * sa / (1. - R)


def sa_smax(smax, R):
    ''' Stress amplitude as function of max stress and stress ratio
    Source: Rosemeier and Antoniou 2021, Eq. 10
    :param: smax: float maximum stress
    :param: R: float stress ratio
    :return: sa: float stress amplitude'''
    return 0.5 * smax * (1. - R)


def sm_smax(smax, R):
    ''' Stress amplitude as function of max stress and stress ratio
    Source: Rosemeier and Antoniou 2021, Eq. 11
    :param: smax: float maximum stress
    :param: R: float stress ratio
    :return: sm: float mean stress '''
    return 0.5 * smax * (1. + R)


def sa_goodman(sa0, sm, Rt):
    ''' Stress amplitude according to modified Goodman 1899 
    :param: sa0: float stress amplitude at zero means stress (R=-1)
    :param: sm: float mean stress
    :param: Rt: float tensile strength
    :return: sa: float stress amplitude'''
    return sa0 * (1. - sm / Rt)


def sa_gerber(sa0, sm, Rt):
    ''' Stress amplitude according to Gerber 1874
    :param: sa0: float stress amplitude at zero means stress (R=-1)
    :param: sm: float mean stress
    :param: Rt: float tensile strength
    :return: sa: float stress amplitude'''
    return sa0 * (1. - (sm / Rt)**2)


def sa_loewenthal(sa0, sm, Rt):
    ''' Stress amplitude according to Loewenthal 1975
    :param: sa0: float stress amplitude at zero means stress (R=-1)
    :param: sm: float mean stress
    :param: Rt: float tensile strength
    :return: sa: float stress amplitude'''
    return sa0 * np.sqrt(1. - (sm / Rt)**2)


def sa_swt(sa0, sm):
    ''' Stress amplitude according to Smith Watson Topper 1970
    :param: sa0: float stress amplitude at zero means stress (R=-1)
    :param: sm: float mean stress
    :return: sa: float stress amplitude'''
    return 0.5 * (np.sqrt(sm**2 + 4. * sa0**2) - sm)


def sa_tosa(sa0, sm, alp):
    ''' Stress amplitude according to Topper Sandor 1970
    :param: sa0: float stress amplitude at zero means stress (R=-1)
    :param: sm: float mean stress
    :return: sa: float stress amplitude'''
    return sa0 - sm**alp


def sa_boerstra(sa0, sm, Rt, alp):
    ''' Stress amplitude according to Boerstra 2007
    :param: sa0: float stress amplitude at zero means stress (R=-1)
    :param: sm: float mean stress
    :param: Rt: float tensile strength
    :return: sa: float stress amplitude'''
    return sa0 * (1. - (sm / Rt)**alp)


def sm_boerstra(sa0, sa, Rt, alp):
    ''' Mean stress according to Boerstra 2007
    :param: sa0: float stress amplitude at zero means stress (R=-1)
    :param: sa: float mean stress
    :param: Rt: float tensile strength
    :return: sm: float stress amplitude'''
    return Rt * (1. - sa / sa0)**(1. / alp)


def sa_basquin(n, m, Rt):
    ''' Stress amplitude according to Basquin 1910
    :param: n: float cycle number
    :param: m: float negative inverse SN curve coefficient
    :param: Rt: float static strength
    :return: sa: float stress amplitude'''
    return Rt * n**(-1. / m)


def dsa_dn_basquin(n, m, Rt):
    '''Gradient of stress amplitude of an SN curve according to Basquin 1910
    Source: Rosemeier and Antoniou 2021, Eq. 25
    :param: n: float cycle number
    :param: m: float negative inverse SN curve coefficient
    :param: Rt: float static strength
    :return: sa: float stress amplitude'''
    return - Rt * n**(-1. / m - 1.) / m


def dn_dsa_basquin(n1, sa1, sa2, m):
    '''Gradient of cycle number of an SN curve through 2 points according to
    Basquin
    :param: sa1: float load level at point 1 (stress amplitude)
    :param: n1: float cycles at point 1
    :param: sa1: float load level at point 2 (stress amplitude)
    :param: m: float negative inverse SN curve coefficient
    :param: Rt: float static strength
    :return: sa: float stress amplitude'''
    return - m * n1 / sa2 * (sa2 / sa1)**-m


def n_basquin(sax, m, N1, sa1):
    '''Cycle number Nx for a given load level sx in according to Basquin
    :param: sax: float target load level
    :param: m: float negative inverse SN curve coefficient at point 1 (stress amplitude)
    :param: N1: float cycles at point 1
    :param: sa1: float load level at point 1 (stress amplitude)
    :return: Nx: float corresponding cycles to sax'''
    return N1 * (sax / sa1)**(-m)


def m_basquin_2p(sa1, n1, sa2, n2):
    ''' Negative inverse SN curve exponent derived from 2 points in SN grid
     according to Basquin
    :param: sa1: float load level at point 1 (stress amplitude)
    :param: n1: float cycles at point 1
    :param: sa1: float load level at point 2 (stress amplitude)
    :param: n2: float cycles at point 2
    :return: m: float negative inverse SN curve exponent'''
    return -(np.log(n1) - np.log(n2)) / (np.log(sa1) - np.log(sa2))


def m_basquin(Rt, sa1, n1):
    ''' Negative inverse SN curve exponent for a given Rt and point in SN grid
     according to Basquin
    :param: Rt: float static strength
    :param: n1: float cycles at point 1
    :param: sa1: float load level at point 1 (stress amplitude)
    :return: m: float negative inverse SN curve exponent'''
    return 1. / (np.log(Rt / sa1) / np.log(n1))


def smax_basquin_goodman(n, R, m, Rt, M=1):
    '''Max stress for given cycle number of an SN curve according to
    Basquin-Goodman
    Source: Rosemeier Antoniou, 2021, Eq. 14
    :param: n: float cycle number
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient
    :param: Rt: float static strength
    :param: M: mean stress sensitivity
    :return: smax: float allowable maximum stress'''
    return 2. * Rt / ((1. - R) * n**(1. / m) + M * abs(R + 1.))


def n_basquin_goodman(smax, R, m, Rt, M=1):
    '''Allowable cycles for given max stress of an SN curve according to
    Basquin-Goodman
    Source: Rosemeier and Antoniou 2021, Eq. 15
    :param: smax: float max stress
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient
    :param: Rt: float static strength
    :param: M: mean stress sensitivity
    :return: N: float allowable cycles'''
    return ((2. * Rt) / ((1. - R) * smax) - M * abs(1. + R) / (1. - R))**m


def Rt_basquin_goodman(smax, N, R, m, M=1):
    '''Static strength for a given point (N,smax) with neg. inverse S-N curve
    exponent according to Basquin-Goodman
    :param: smax: float max stress
    :param: N: float allowable cycles
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient
    :return: Rt: float static strength'''
    return 0.5 * smax * ((1. - R) * N**(1. / m) + M * (R + 1.))


def xrand_smax_basquin_goodman(smax_i, N_i, R_i, m_fit, Rt_fit):
    ''' Random variable x using smax
    Source: Rosemeier and Antoniou 2021, Eq. 29
    :return: xrand: float value'''
    return smax_i - smax_basquin_goodman(N_i, R_i, m_fit, Rt_fit)


def xrand_Rt_basquin_goodman(smax_i, N_i, R_i, m_fit, Rt_fit):
    ''' Random variable x using smax
    Source: Rosemeier and Antoniou 2021, Eq. 29
    :return: xrand: float value'''
    Rt_i = Rt_basquin_goodman(smax_i, N_i, R_i, m_fit)
    return Rt_i - Rt_fit


def smax_basquin_goodman_weibull(n, R, m, Rt_fit, p, lmbda, delta, beta):
    ''' Source: Rosemeier and Antoniou 2021, Eq. 31'''
    return x_weibull(p, lmbda, delta, beta) +\
        smax_basquin_goodman(n, R, m, Rt_fit)


def n_basquin_goodman_weibull(smax, R, m, Rt_fit, p, lmbda, delta, beta):
    '''Cycle number for given max stress and probability of an SN curve according to Stuessi-Goodman-Weibull
    Source: Rosemeier and Antoniou 2021, Eq. 41
    :param: smax: float max stress
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt_fit: float static strength (fit)
    :param: p: float probability
    :param: lmbda: float Weibull location parameter
    :param: delta: float Weibull scale parameter
    :param: beta: float  Weibull shape parameter
    :param: M: mean stress sensitivity (optional)
    :return: N: float allowable cycle number'''
    return n_basquin_goodman(smax - x_weibull(p, lmbda, delta, beta), R, m, Rt_fit)


def Rt_basquin_goodman_weibull(smax, N, R, m, Rt_fit, p, lmbda, delta, beta, M=1):
    ''' Max stress for given cycle number and probability of an SN curve
    according to Stuessi-Goodman-Weibull
    :param: Rt_fit: float static strength (fit)
    :param: p: float probability
    :param: lmbda: float Weibull location parameter
    :param: delta: float Weibull scale parameter
    :param: beta: float  Weibull shape parameter
    :return: smax: float allowable maximum stress'''
    x_w = x_weibull(p, lmbda, delta, beta) + Rt_fit
    return 0.5 * (x_w + smax) * ((1. - R) * N**(1. / m) + M * (R + 1.))


def smax_limit_basquin_goodman_weibull(p, Rt_fit, lmbda, delta, beta):
    ''' Endurance limit as function of p and R
    :param: p: float probability
    :param: Rt_fit: float static strength (fit)
    :param: lmbda: float Weibull location parameter
    :param: delta: float Weibull scale parameter
    :param: beta: float  Weibull shape parameter
    :return: Rt: float static strength for arbitrary p
    '''
    xw_p = x_weibull(p, lmbda, delta, beta)
    Rt = xw_p + Rt_fit

    return Rt


def xrand_smax_basquin_goodman_weibull(smax_i, N_i, R_i, m_fit, Rt_fit, M_fit, p, lmbda, delta, beta):
    ''' Random variable x using smax
    Source: Rosemeier and Antoniou 2021, Eq. 29
    :return: xrand: float value '''
    return smax_i - smax_basquin_goodman_weibull(
        N_i, R_i, m_fit, Rt_fit, M_fit, p, lmbda, delta, beta)


def _b(m, Re, Rt):
    '''Transforms Basquin's m into Stuessi's b
    (1) Derive Basquin for s: dn/ds(s) = -(Na*m*(s/R_a)**(-m))/s
    (2) Derive Stuessi for s: dn/ds(s) = -(Na*b*(Re-Rt)*
                                    ((s-Rt)/(Re-s))**(b-1))/((Re-s)**2)
    (3) Set R_a = 0.5*(Re+Rt)
    (4) Set (1)=(2) with (3) and solve for b
    Source: Rosemeier and Antoniou 2021, Eq.4 and Appendix B'''
    return (m * (-Re + Rt)) / (2. * (Re + Rt))


def sa_stuessi(n, m, Rt, Re, Na, n0=1.0):
    '''Stress amplitude of an SN curve according to Stuessi
    Source: Rosemeier and Antoniou 2021, Eq.3 and Appendix A
    :param: n: float cycle number
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: Na: float inflection point
    :param: n0: intersection with smax axis
    :return: sa: float allowable stress amplitude
    '''
    b = _b(m, Re, Rt)
    nNa = _nNa(n, Na, b, n0)
    return (Re * nNa + Rt) / (1. + nNa)


def dsa_dn_stuessi(n, m, Rt, Re, Na, n0=1.0):
    ''' Gradient of stress amplitude of an SN curve according to Stuessi
    Source: Rosemeier and Antoniou 2021, Eq. 24
    :param: n: float cycle number
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: Na: float inflection point
    :param: n0: intersection with smax axis
    :return: sa: float allowable stress amplitude '''
    b = _b(m, Re, Rt)
    nNa = _nNa(n, Na, b, n0)
    return ((Re - Rt) * nNa) / (b * (n - n0) * (nNa + 1.)**2)


def dn_dsa_stuessi(sa, m, Rt, Re, Na):
    ''' Gradient of cycle number wrt stress amplitude of Stuessi's curve
    :param: sa: float allowable stress amplitude
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: Na: float inflection point
    :return: dn/dsa: float cycle number '''
    b = _b(m, Re, Rt)
    return Na * b * (Re - Rt) * ((sa - Rt) / (Re - sa))**(b - 1) / (Re - sa)**2


def n_stuessi(sa, m, Rt, Re, Na, n0=1.0):
    ''' Cycle number of an SN curve according to Stuessi 1955
    :param: n: float cycle number
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: Na: float inflection point
    :param: n0: intersection with smax axis
    :return: sa: float allowable stress amplitude
    '''
    b = _b(m, Re, Rt)
    return n0 + Na * ((Rt - sa) / (sa - Re))**b


def sa_stuessi_orig(n, Rt, Re, c, p):
    ''' Stuessi's original formulation as of 1955, Eq.13
    Source: Rosemeier and Antoniou 2021, Eq.46
    :param: n: float cycle number
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: c: float geometric parameter
    :param: p: float geometric parameter
    :return: sa: float allowable stress amplitude
    '''
    return (Rt + c * n**p * Re) / (1 + c * n**p)


def dsa_dn_stuessi_orig(n, Rt, Re, c, p):
    ''' First derivative of sa_stuessi_orig
    :param: n: float cycle number
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: c: float geometric parameter
    :param: p: float geometric parameter
    :return: dsa/dn: float first derivative of allowable stress amplitude
    '''
    return (c * p * (Re - Rt) * n**(p - 1.)) / (c * n**p + 1.)**2


def d2sa_dn2_stuessi_orig(n, Rt, Re, c, p):
    ''' Second derivative of sa_stuessi_orig
    Source: Rosemeier and Antoniou 2021, Eq.49
    :param: n: float cycle number
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: c: float geometric parameter
    :param: p: float geometric parameter
    :return: d**2sa/dn**2: float second derivative of allowable stress amplitude
    '''
    return -(c * n**(p - 2.) * p * (1. + c * n**p - p + c * n**p * p) *
             (Re - Rt)) / (1. + c * n**p)**3


def sa_stuessi_orig_semilogx(i, Rt, Re, c, p):
    ''' Combine sa_stuessi_orig with n=10**i
    Source: Rosemeier and Antoniou 2021, Eq.48
    :param: i: float exponente of cycle number
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: c: float geometric parameter
    :param: p: float geometric parameter
    :return: sa: float allowable stress amplitude
    '''
    return (Rt + c * 10**(i * p) * Re) / (1 + c * 10**(i * p))


def dsa_di_stuessi_orig_semilogx(i, Rt, Re, c, p):
    ''' First derivative of sa_stuessi_orig_semilogx
    :param: i: float exponente of cycle number
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: c: float geometric parameter
    :param: p: float geometric parameter
    :return: dsa/di: float first derivative of allowable stress amplitude
    '''
    return (c * p * 10**(i * p) * (Re - Rt) * np.log(10)) /\
        (1. + c * 10**(i * p))**2


def d2sa_di2_stuessi_orig_semilogx(i, Rt, Re, c, p):
    ''' Second derivative of sa_stuessi_orig_semilogx
    Source: Rosemeier and Antoniou 2021, Eq.49
    :param: i: float exponente of cycle number
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: c: float geometric parameter
    :param: p: float geometric parameter
    :return: d**2sa/di**2: float second derivative of allowable stress amplitude
    '''
    return ((1. - c * 10**(i * p)) * c * p**2 * 10**(i * p) *
            (Re - Rt) * np.log(10)**2) / (1. + c * 10**(i * p))**3


def Na_stuessi_orig(c, p):
    ''' Inflection point x coordinate of stuessi_orig
    (1) set d2sa_dn2_stuessi_orig_semilogx=0 to find inflection point ia
    (2) solve for ia => ia = log(1/c)/(p*log(10))
    (3) combine with Na=10**ia
    Source: Rosemeier and Antoniou 2021, Eq.50
    :param: c: float geometric parameter
    :param: p: float geometric parameter
    :return: Na: float allowable cycle number at inflection point
    '''
    return (1. / c)**(1. / p)


def c_stuessi_orig(Na, p):
    '''Convert Na into original Stuessi's geometric parameter c
    Source: Rosemeier and Antoniou 2021, Eq.52
    :param: p: float geometric parameter
    :param: Na: float allowable cycle number at inflection point
    :return: c: float geometric parameter
    '''
    return (1. / Na)**p


def Ra_stuessi_orig(Rt, Re):
    ''' Inflection point y coordinate of stuessi_orig
    (1) insert ia = log(1/c)/(p*log(10)) into sa_stuessi_orig_semilogx
    (2) solve for sa
    Source: Rosemeier and Antoniou 2021, Eq.51
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :return: Ra: float allowable stress amplitude at inflection point
    '''
    return 0.5 * (Re + Rt)


def _nNa(n, Na, b, n0=1):
    return ((n - n0) / Na)**(1. / b)


def sa_stuessi_boerstra(n, m, Rt, Re, Na, alp):
    b = _b(m, Re, Rt)
    nNa = _nNa(n, Na, b)
    sa = (1. - (sm / Rt)**alp) * (Re * nNa + Rt) / (1. + nNa)
    return sa


def sm_stuessi_boerstra(n, m, Rt,  Re, Na, d):
    b = _b(m, Re, Rt)
    nNa = _nNa(n, Na, b)
    sm = Rt * (1. - sa * (nNa - 1.) / (Re * nNa + Rt))**(1. / d)
    return sm


def n_sa_sm_stuessi_boerstra(m, Rt, Re, Na, alp, n0=1.0):
    b = _b(m, Re, Rt)
    n = n0 + Na * (-1. * (sa + Rt * ((sm / Rt)**alp - 1.)) /
                   (sa + Re * ((sm / Rt)**alp - 1.)))**b
    return n


def _Rnum(R):
    EPS = 1E-12
    if R == -1.0:
        Rnum = R + EPS
    elif R == +1.0:
        Rnum = R - EPS
    else:
        Rnum = R
    return Rnum


def _salp(smax, Rnum, Rt, alp):
    return (abs(smax * (1. + Rnum)) / (2. * Rt))**alp - 1.


def _rhs_nom(smax, Rnum, Rt, salp):
    return 0.5 * smax * (1. - Rnum) + Rt * salp


def _rhs_den(smax, Rnum, Re, salp):
    return _rhs_nom(smax, Rnum, Re, salp)


def getalp(n, alp_c, alp_fit):
    if alp_fit == 'lin':
        alp = poly1dlogx(n, alp_c[0], alp_c[1])
    elif alp_fit == 'exp':
        alp = explogx1(n, alp_c[0], alp_c[1])
    elif alp_fit == 'aki':
        nexp_end = alp_c.x[-1]
        if n == 0.0:
            alp = 100
        elif n > 10**nexp_end:
            alp = akima1d_extrap_right(alp_c, n)
            if alp <= 0.0:  # alp can get negative when extrapolated
                alp = 0.0
        elif n < 1.0:
            alp = akima1d_extrap_left(alp_c, n)
        else:
            alp = alp_c(np.log10(n))
    return alp


def _rootwarn_n(n, R):
    print('Root not found for n=%0.2e, R=%0.2f' % (n, R))


def _rootwarn_smax(smax, R):
    print('Root not found for smax=%0.2e, R=%0.2f' % (smax, R))


def _getsmax_sca_arr(n, R, _getsmax):

    if isinstance(n, np.ndarray) or isinstance(R, np.ndarray):
        if isinstance(n, np.ndarray) and isinstance(R, np.ndarray):
            na = n
            Ra = R
        elif not isinstance(n, np.ndarray) and isinstance(R, np.ndarray):
            na = np.ones_like(R) * n
            Ra = R
        elif isinstance(n, np.ndarray) and not isinstance(R, np.ndarray):
            na = n
            Ra = np.ones_like(n) * R
        smax = np.zeros_like(na)
        for ni, (n_, R_) in enumerate(zip(na, Ra)):
            try:
                smax[ni] = _getsmax(n_, R_)
            except:
                smax[ni] = np.nan
                _rootwarn_n(n_, R_)
    else:
        try:
            smax = _getsmax(n, R)
        except:
            smax = np.nan
            _rootwarn_n(n, R)
    return smax


def smax_stuessi_boerstra(n, R, m, Rt, Re, Na, alp_c, alp_fit='exp'):
    ''' Maximum stress for Stuessi-Boerstra model
    Source: Rosemeier and Antoniou 2021, Eq. 21,22
    :param: n: permissible cycle number
    :param: R: stress ratio
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: Na: float inflection point
    :param: alp_fit: string alp fit type
    :param: alp_c: array holding fit coefficients or Akima object
    :return: smax: float allowable maximum stress '''
    def _getsmax(n_, R_):
        alp = getalp(n_, alp_c, alp_fit)

        def _fun(smax_):
            Rnum = _Rnum(R_)
            lhs = _nNa(n_, Na, b)
            salp = _salp(smax_, Rnum, Rt, alp)
            rhs_nom = _rhs_nom(smax_, Rnum, Rt, salp)
            rhs_den = _rhs_den(smax_, Rnum, Re, salp)
            rhs = -1. * rhs_nom / rhs_den
            return lhs - rhs

        def _fun_rhs_den(smax_):
            Rnum = _Rnum(R_)
            salp = _salp(smax_, Rnum, Rt, alp)
            rhs_den = _rhs_den(smax_, Rnum, Re, salp)
            return rhs_den

        if n_ == 1 and R_ >= -1.:
            smax = Rt
        else:
            # find asymptote and use as lower bound
            # asymptote is found when denominater = zero
            sasymp = optimize.brentq(f=_fun_rhs_den, a=0., b=smax_upper)
            smax = optimize.brentq(f=_fun, a=sasymp + smax_delta, b=smax_upper)
        return smax

    b = _b(m, Re, Rt)

    smax_upper = Rt
    smax_delta = 1E-6

    smax = _getsmax_sca_arr(n, R, _getsmax)

    return smax


def n_stuessi_boerstra(smax, R, m, Rt, Re, Na, alp_c, alp_fit='exp', n_end=1E12):
    ''' Permissible cycle number for Stuessi-Boerstra model
    :param: smax: float allowable maximum stress
    :param: R: stress ratio
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: Na: float inflection point
    :param: alp_fit: string alp fit type
    :param: alp_c: array holding fit coefficients or Akima object
    :param: n_end: cycle number at which endurance limit smax_end is defined, a
        values below smax_end result in infinite cycle number
    :return: n: permissible cycle number '''
    def _getn(smax_, R_):

        def _fun(n_):
            alp = getalp(n_, alp_c, alp_fit)
            Rnum = _Rnum(R_)
            lhs = _nNa(n_, Na, b)
            salp = _salp(smax_, Rnum, Rt, alp)
            rhs_nom = _rhs_nom(smax_, Rnum, Rt, salp)
            rhs_den = _rhs_den(smax_, Rnum, Re, salp)
            rhs = -1. * rhs_nom / rhs_den
            return lhs - rhs

        def _fun_rhs_den(n_):
            alp = getalp(n_, alp_c, alp_fit)
            Rnum = _Rnum(R_)
            salp = _salp(smax_, Rnum, Rt, alp)
            rhs_den = _rhs_den(smax_, Rnum, Re, salp)
            return rhs_den

        def _fun_rhs_den_2(n_):
            alp = getalp(n_, alp_c, alp_fit)
            Rnum = _Rnum(R_)
            a_ = smax_ * (Rnum - 1)
            b_ = 2 * Re
            c_ = (abs(smax_ * (1. + Rnum)) / (2. * Rt))

            alp_0 = np.log((a_ + b_) / b_) / np.log(c_)
            return alp - alp_0

        def _fun_rhs_den_3(smax_):
            Rnum = _Rnum(R_)
            f_ = smax_ * (Rnum - 1.)
            g_ = 2. * Re
            i_ = 2. * Rt
            h_ = (abs(smax_ * (1. + Rnum)) / (2. * Rt))
            a_ = alp_c[0]
            d_ = alp_c[1]

            nasymp = np.real((a_ + cmath.log((f_ + g_) / g_) /
                              cmath.log(h_) - 1.) / a_)**(-1. / d_)

            D_ = f_ + g_ * (1. - h_**explogx1(nasymp, a_, d_))
            Z_ = f_ + i_ * (1. - h_**explogx1(nasymp, a_, d_))

            # denominater must be zero and nominator's value not
            nom = (nasymp - 1.)**(1 / b) * D_ - Z_ * Na**(1 / b)
            if nom == 0.:
                nasymp = 0.0
            return nasymp

        if alp_fit == 'lin':
            nasymp = None  # not implemented
        elif alp_fit == 'exp':
            nasymp = _fun_rhs_den_3(smax_)
        elif alp_fit == 'aki':
            try:
                nasymp = optimize.brentq(f=_fun_rhs_den, a=0.0, b=1E21)
            except:
                nasymp = 0.0

        n_delta = 0.5
        n_upper = 10**(np.log10(nasymp) + 10)
        if nasymp < 1.0:
            nasymp = 1.0
            n_delta = 0.
            n_upper = 1E7

        n = optimize.brentq(f=_fun, a=nasymp + n_delta, b=n_upper)

        return n

    b = _b(m, Re, Rt)

    if isinstance(smax, np.ndarray):
        n = np.zeros_like(smax)
        if not isinstance(R, np.ndarray):
            Ra = np.ones_like(n) * R
        else:
            Ra = R
        for smaxi, (smax_, R_) in enumerate(zip(smax, Ra)):
            smax_end = smax_stuessi_boerstra(  # endurance limit
                n_end, R_, m, Rt, Re, Na, alp_c, alp_fit)
            if smax_ <= smax_end:
                n[smaxi] = np.inf
            else:
                try:
                    n[smaxi] = _getn(smax_, R_)
                except:
                    n[smaxi] = np.nan
                    _rootwarn_smax(smax_, R_)
    else:
        smax_end = smax_stuessi_boerstra(  # endurance limit
            n_end, R, m, Rt, Re, Na, alp_c, alp_fit)
        if smax <= smax_end:
            n = np.inf
        else:
            try:
                n = _getn(smax, R)
            except:
                n = np.nan
                _rootwarn_smax(smax, R)

    return n


def smax_basquin_boerstra(n, R, m, Rt, alp_c, alp_fit='exp'):
    ''' Maximum stress for Basquin-Boerstra model
    Source: Rosemeier and Antoniou 2021, Eq. 18
    :param: n: permissible cycle number
    :param: R: stress ratio
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt: float static strength
    :param: alp_fit: string alp fit type
    :param: alp_c: array holding fit coefficients or Akima object
    :return: smax: float allowable maximum stress '''
    def _getsmax(n_, R_):
        alp = getalp(n_, alp_c, alp_fit)

        def _fun(smax_):
            Rnum = _Rnum(R_)
            lhs = n_**(1. / m)
            salp = _salp(smax_, Rnum, Rt, alp)
            rhs = -1. * 2. * Rt * salp / (smax_ * (1. - Rnum))
            return lhs - rhs

        if n_ == 1 and R_ >= -1.:
            smax = Rt
        else:
            sasymp = 0.
            smax = optimize.brentq(f=_fun, a=sasymp + smax_delta, b=smax_upper)
        return smax

    smax_upper = Rt
    smax_delta = 1E-6

    smax = _getsmax_sca_arr(n, R, _getsmax)

    return smax


def n_basquin_boerstra(smax, R, m, Rt, alp_c, alp_fit='exp', n_upper=1E29):
    ''' Permissible cycle number for Basquin-Boerstra model
    :param: smax: float allowable maximum stress
    :param: R: stress ratio
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt: float static strength
    :param: alp_fit: string alp fit type
    :param: alp_c: array holding fit coefficients or Akima object
    :param: n_upper: float upper limit for root finding
    :return: n: permissible cycle number '''
    def _getn(smax_, R_):

        def _fun(n_):
            alp = getalp(n_, alp_c, alp_fit)
            Rnum = _Rnum(R_)
            lhs = n_**(1. / m)
            salp = _salp(smax_, Rnum, Rt, alp)
            rhs = -1. * 2. * Rt * salp / (smax_ * (1. - Rnum))
            return lhs - rhs
        n_lower = 1.
        n = optimize.brentq(f=_fun, a=n_lower, b=n_upper)
        return n

    if isinstance(smax, np.ndarray):
        n = np.zeros_like(smax)
        if not isinstance(R, np.ndarray):
            Ra = np.ones_like(n) * R
        else:
            Ra = R
        for smaxi, (smax_, R_) in enumerate(zip(smax, Ra)):
            try:
                n[smaxi] = _getn(smax_, R_)
            except:
                n[smaxi] = np.nan
                _rootwarn_smax(smax_, R_)
    else:
        try:
            n = _getn(smax, R)
        except:
            n = np.nan
            _rootwarn_smax(smax, R)

    return n


def xrand_sa_stuessi(sa_i, N_i, m_fit, Rt_fit, Re_fit, Na_fit):
    ''' Random variable x using sa
    Source: Rosemeier and Antoniou 2021, Eq. 29
    :return: xrand: float value
    '''
    return sa_i - sa_stuessi(N_i, m_fit, Rt_fit, Re_fit, Na_fit)


def sa_stuessi_weibull(n, m, Rt_fit, Re_fit, Na, p, lmbda, delta, beta):
    '''Stress amplitude for given cycle number and probability of an SN curve according to Stuessi-Goodman-Weibull
    :param: n: float cycle number
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt_fit: float static strength (fit)
    :param: Re_fit: float endurance limit for R=-1 (fit)
    :param: Na: float inflection point
    :param: p: float probability
    :param: lmbda: float Weibull location parameter
    :param: delta: float Weibull scale parameter
    :param: beta: float  Weibull shape parameter
    :return: sa: float allowable maximum stress
    '''
    return x_weibull(p, lmbda, delta, beta) + sa_stuessi(n, m, Rt_fit, Re_fit, Na)


def smax_stuessi_boerstra_weibull(n, R, m, Rt_fit, Re_fit, alp_c, alp_fit, Na, p, lmbda, delta, beta):
    '''Source: Rosemeier and Antoniou 2021, Eq. 31'''
    return x_weibull(p, lmbda, delta, beta) +\
        smax_stuessi_boerstra(n, R, m, Rt_fit, Re_fit, Na, alp_c, alp_fit)


def n_stuessi_boerstra_weibull(smax, R, m, Rt_fit, Re_fit, alp_c, alp_fit, Na, p, lmbda, delta, beta, n_end=1E12):
    '''Source: Rosemeier and Antoniou 2021, Eq. 43'''
    return n_stuessi_boerstra(smax - x_weibull(p, lmbda, delta, beta), R, m, Rt_fit, Re_fit, Na, alp_c, alp_fit, n_end)


def smax_basquin_boerstra_weibull(n, R, m, Rt_fit, alp_c, alp_fit, p, lmbda, delta, beta):
    '''Source: Rosemeier and Antoniou 2021, Eq. 31'''
    return x_weibull(p, lmbda, delta, beta) +\
        smax_basquin_boerstra(n, R, m, Rt_fit, alp_c, alp_fit)


def n_basquin_boerstra_weibull(smax, R, m, Rt_fit, alp_c, alp_fit, p, lmbda, delta, beta):
    '''Source: Rosemeier and Antoniou 2021, Eq. 42'''
    return n_basquin_boerstra(smax - x_weibull(p, lmbda, delta, beta), R, m, Rt_fit, alp_c, alp_fit)


def m_RtRd(m_fit, Rt_fit, Re_fit, Rt_tar, Re_tar):
    ''' Transform m_fit when Stuessi curve is shifted in smax-direction
    Stuessi's b_fit = b_tar
    where b = (m * (-Re + Rt)) / (2 * (Re + Rt))
    and (-Re_fit + Rt_fit) = (-Re_tar + Rt_tar)
    '''
    return m_fit * (Re_tar + Rt_tar) / (Re_fit + Rt_fit)


def m_tangent(n, m, Rt, Re, Na):
    ''' sa_basquin == sa_stuessi, solve for mt (Basquin)
    Source: Rosemeier and Antoniou 2021, Eq. 23
    :param: n: float cycle number at which Basqiun should intersect Stuessi 
    :return: mt: float negative inverse SN curve exponent for n
    '''
    return np.log(n) / np.log(Rt / sa_stuessi(n, m, Rt, Re, Na))


def xrand_smax_stuessi_boerstra(smax_i, N_i, R_i, m_fit, Rt_fit, Re_fit, Na_fit, alp_c, alp_fit):
    ''' Random variable x using smax
    Source: Rosemeier and Antoniou 2021, Eq. 29
    :return: xrand: float value
    '''
    return smax_i - smax_stuessi_boerstra(N_i, R_i, m_fit, Rt_fit, Re_fit, Na_fit, alp_c, alp_fit)


def xrand_smax_basquin_boerstra(smax_i, N_i, R_i, Rt_fit, m_fit, alp_c, alp_fit):
    ''' Random variable x using smax
    Source: Rosemeier and Antoniou 2021, Eq. 29
    :return: xrand: float value
    '''
    return smax_i - smax_basquin_boerstra(N_i, R_i, m_fit, Rt_fit, alp_c, alp_fit)


def fit_basquin_goodman_weibull(dff, m_start, Rt_start):
    '''Fits m and Rt to a set of experimental data
    :param: cyc_data: dict with experimental data
    :param: m_start: float start value
    :param: Rt_start: float start value
    :return: m_fit
    :return: Rt_fit'''

    initial_guess = [m_start, Rt_start]

    def objective(params):
        m, Rt = params  # multiple design variables

        smax_fit = smax_basquin_goodman(
            n=dff['N'],
            R=dff['R'],
            m=m,
            Rt=Rt)
        critsum = np.sum((np.log10(smax_fit) - np.log10(dff['smax']))**2)
        return critsum

    result = optimize.minimize(objective, initial_guess, method='Nelder-Mead')
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)
    m_fit, Rt_fit = fitted_params

    xs = xrand_smax_basquin_goodman(smax_i=dff['smax'],
                                    N_i=dff['N'],
                                    R_i=dff['R'],
                                    m_fit=m_fit,
                                    Rt_fit=Rt_fit)
    lmbda, delta, beta = pwm_weibull(xs)

    return m_fit, Rt_fit, lmbda, delta, beta, xs


def fit_basquin_boerstra_weibull(dff, m_start, Rt_start, alp_c, alp_fit):
    ''' Fits Basquin-Boerstra model to experimental data
    :param: dff: dataframe with experimental data
    :param: m_start: float start value
    :param: Rt_start: float start value
    :param: alp_c: array
    :param: alp_fit: string
    :return: m_fit
    :return: Rt_fit
    :return: lmbda: float Weibull location parameter
    :return: delta: float Weibull scale parameter
    :return: beta: float  Weibull shape parameter'''

    initial_guess = [m_start, Rt_start]

    def objective(params):
        m, Rt = params  # multiple design variables
        smax_fit = smax_basquin_boerstra(
            n=dff['N'].to_numpy(),
            R=dff['R'].to_numpy(),
            m=m,
            Rt=Rt,
            alp_c=alp_c,
            alp_fit=alp_fit)
        critsum = np.sum((np.log10(smax_fit) - np.log10(dff['smax']))**2)
        return critsum

    result = optimize.minimize(objective, initial_guess, method='Nelder-Mead')
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)
    m_fit, Rt_fit = fitted_params

    xs = xrand_smax_basquin_boerstra(smax_i=dff['smax'].to_numpy(),
                                     N_i=dff['N'].to_numpy(),
                                     R_i=dff['R'].to_numpy(),
                                     m_fit=m_fit,
                                     Rt_fit=Rt_fit,
                                     alp_c=alp_c,
                                     alp_fit=alp_fit)
    lmbda, delta, beta = pwm_weibull(xs)

    return m_fit, Rt_fit, lmbda, delta, beta, xs


def fit_stuessi_boerstra_weibull(dff, m_start, Rt_start, Re_start, Na_start, alp_c,
                                 alp_fit):
    '''Fits m, Rt, Re, Na and Weibull parameters to a set of experimental data
    :param: cyc_data: dict with experimental data
    :param: m_start: float start value
    :param: Rt_start: float start value
    :param: Re_start: float start value
    :param: Na_start: float start value
    :return: m_fit
    :return: Rt_fit
    :return: Re_fit
    :return: Na_fit
    :return: lmbda: float Weibull location parameter
    :return: delta: float Weibull scale parameter
    :return: beta
    '''
    initial_guess = [m_start, Rt_start, Re_start, Na_start]

    def objective(params):
        m, Rt, Re, Na = params  # multiple design variables
        try:
            smax_fit = smax_stuessi_boerstra(
                n=dff['N'].to_numpy(),
                R=dff['R'].to_numpy(),
                m=m,
                Rt=Rt,
                Re=Re,
                Na=Na,
                alp_c=alp_c,
                alp_fit=alp_fit)
            critsum = np.sum((np.log10(smax_fit) - np.log10(dff['smax']))**2)
        except:
            critsum = np.inf
        return critsum

    result = optimize.minimize(objective, initial_guess, method='Nelder-Mead')
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)
    m_fit, Rt_fit, Re_fit, Na_fit = fitted_params

    xs = xrand_smax_stuessi_boerstra(smax_i=dff['smax'].to_numpy(),
                                     N_i=dff['N'].to_numpy(),
                                     R_i=dff['R'].to_numpy(),
                                     m_fit=m_fit,
                                     Rt_fit=Rt_fit,
                                     Re_fit=Re_fit,
                                     Na_fit=Na_fit,
                                     alp_c=alp_c,
                                     alp_fit=alp_fit)
    lmbda, delta, beta = pwm_weibull(xs)

    return m_fit, Rt_fit, Re_fit, Na_fit, lmbda, delta, beta, xs


class SNFit(object):
    '''Fits Basquin-Goodman/Boerstra and Stuessi-Boerstra type curves to a set
    of experimental data points'''

    def __init__(self, fit_type='basquin_goodman'):
        '''
        :param: fit_type: str basquin_goodman, basquin_boerstra or stuessi-boerstra'''
        self.fit_type = fit_type

        if self.fit_type == 'basquin-goodman':
            self._init_basquin_goodman()
        elif self.fit_type == 'stuessi-boerstra':
            self._init_stuessi_boerstra()

    def _init_basquin_goodman(self):
        self.m_start = 10.0
        self.Rt_start = 50.0E+6

    def _init_stuessi_boerstra(self):
        self.Re_start = 9.E+6
        self.Na_start = 68.
        self.alp_c = [1., 0., 0.]

    def load_data(self, df, grp_entries):
        '''
        :param: df: pandas dataframe holding at least columns smax, R, N, (runout)
        :param: grp_entries: list of group entries in data
        '''

        self.df = df
        if 'runout' in df:
            self.dff = df[df['runout'] == 0].reset_index()  # exclude runouts
        else:
            self.dff = df
            self.dff['runout'] = 0

        self.N_exp_lower = np.min(
            self.dff['N'][(self.dff['R'] != 1.0)].to_numpy())
        self.N_exp_upper = np.max(
            self.dff['N'][(self.dff['R'] != 1.0)].to_numpy())

        self.grps = grp_entries
        self.grps_ratio = np.zeros(len(self.grps))
        for i, grp in enumerate(self.grps):
            self.grps_ratio[i] = np.mean(self.dff['R'][grp])

    def fit_sn(self):
        '''
        :return: sn_fit: dict with fitting parameters
        '''

        self.sn_fit = OrderedDict()

        if self.fit_type == 'basquin-goodman':
            m_fit, Rt_fit, lmbda, delta, beta, xs =\
                fit_basquin_goodman_weibull(self.dff,
                                            self.m_start,
                                            self.Rt_start)

        elif self.fit_type == 'basquin-boerstra':
            m_fit, Rt_fit, lmbda, delta, beta, xs =\
                fit_basquin_boerstra_weibull(self.dff,
                                             self.m_start,
                                             self.Rt_start,
                                             self.alp_c,
                                             self.alp_fit)

        elif self.fit_type == 'stuessi-boerstra':
            m_fit, Rt_fit, Re_fit, Na, lmbda, delta, beta, xs =\
                fit_stuessi_boerstra_weibull(self.dff,
                                             self.m_start,
                                             self.Rt_start,
                                             self.Re_start,
                                             self.Na_start,
                                             self.alp_c,
                                             self.alp_fit
                                             )
            self.sn_fit['Re_fit'] = Re_fit
            self.sn_fit['Na'] = Na

        self.sn_fit['xs'] = xs.tolist()
        self.sn_fit['m_fit'] = m_fit
        self.sn_fit['Rt_fit'] = Rt_fit
        self.sn_fit['lmbda'] = lmbda
        self.sn_fit['delta'] = delta
        self.sn_fit['beta'] = beta

    def project_data(self, N_fit_upper):
        ''' Projects experimental data points along the fit curve on a range of
        cycle numbers and determines respective amplitude and sa and mean sm
        :param: N_fit_upper: upper cycle bound for fitting
        :return: sa_proj: projected stress amplitudes 
        :return: sm_proj: projected mean stresses'''
        self.N_fit_lower = 1
        self.N_fit_upper = N_fit_upper
        self.npoint = npoint = 17
        self.ns = ns = np.logspace(np.log10(1), np.log10(N_fit_upper), npoint)
        self.dfns = pd.DataFrame(data={'ns': ns})

        m_fit = self.sn_fit['m_fit']
        Rt_fit = self.sn_fit['Rt_fit']
        if self.fit_type == 'stuessi-boerstra':
            Re_fit = self.sn_fit['Re_fit']
            Na_fit = self.sn_fit['Na']

        # project exp data points to (s_m/s_a/N)
        ndat = len(self.dff['smax'])

        self.sa_proj = np.zeros((ndat, npoint))
        self.sm_proj = np.zeros((ndat, npoint))

        for ni, n_ in enumerate(ns):

            R_ = self.dff['R'].to_numpy()
            xs = self.sn_fit['xs']

            if self.fit_type == 'stuessi-boerstra':
                smax_fit = smax_stuessi_boerstra(
                    n_, R=R_, m=m_fit, Rt=Rt_fit,
                    Re=Re_fit, Na=Na_fit, alp_c=self.alp_c,
                    alp_fit=self.alp_fit)
            elif self.fit_type == 'basquin-goodman':
                smax_fit = smax_basquin_goodman(
                    n_, R_, m_fit, Rt=Rt_fit)
            elif self.fit_type == 'basquin-boerstra':
                smax_fit = smax_basquin_boerstra(
                    n_, R_, m_fit, Rt=Rt_fit,
                    alp_c=self.alp_c,
                    alp_fit=self.alp_fit)

            smax_proj = smax_fit + xs

            self.sa_proj[:, ni] = sa_smax(smax_proj, R_)
            self.sm_proj[:, ni] = sm_smax(smax_proj, R_)

    def fit_cld(self):
        '''Least-squares fit of CL curves to projected data
        :return: alp: Boerstra coefficient for each cycle number
        :return: sa0: Intersection point with y-axis for CL curve of each cycle
        number'''
        alp_lower = 0.0  # lower bound for alp
        npoint = self.npoint
        Rt_fit = self.sn_fit['Rt_fit']

        self.alp = np.zeros(npoint)
        self.sa0 = np.zeros(npoint)
        for ni in range(npoint):

            xdata = self.sm_proj[:, ni]
            ydata = self.sa_proj[:, ni]

            if self.fit_type == 'basquin-boerstra' or self.fit_type == 'stuessi-boerstra':
                def func(x, alp, sa0_):
                    return sa_boerstra(sa0_, x, Rt_fit, alp)

                popt, _ = optimize.curve_fit(
                    func, xdata, ydata, p0=[1.0, Rt_fit],
                    bounds=((alp_lower, 0.0), (2.0, Rt_fit)))

                self.alp[ni] = popt[0]
                self.sa0[ni] = popt[1]

            elif self.fit_type == 'basquin-goodman':
                def func(x, sa0_):
                    return sa_goodman(sa0_, x, Rt_fit)

                popt, _ = optimize.curve_fit(
                    func, xdata, ydata, p0=[Rt_fit],
                    bounds=((0.0), (Rt_fit)))

                self.alp[ni] = np.nan
                self.sa0[ni] = popt[0]

        self.dfns['sa0'] = self.sa0
        self.dfns['alp'] = self.alp

    def fit_alp(self):
        ''' Fits a curve through given Beoerstra exponents over cycles
        Possible fit curves: linear, exponential, Akima spline
        :return: alp_c: array holding fit coefficients or Akima object
        '''
        # idxs of cycles numbers to be used for alp fitting
        alp_idxs = self.dfns[(self.dfns['ns'] >= self.N_exp_lower) & (
            self.dfns['ns'] <= self.N_fit_upper)].index.to_list()
        alp_idxs.insert(0, 0)  # N=1

        if self.fit_type == 'basquin-boerstra' or self.fit_type == 'stuessi-boerstra':
            if not alp_idxs:
                alp_idxs = range(self.npoint)

            self.alp[0] = 1.0  # force 1.0

            xdata = self.ns[alp_idxs]
            ydata = self.alp[alp_idxs]
            if self.alp_fit == 'lin':
                popt, _ = optimize.curve_fit(poly1dlogx, xdata, ydata)
            elif self.alp_fit == 'exp':
                popt, _ = optimize.curve_fit(explogx1, xdata, ydata)
            elif self.alp_fit == 'aki':
                popt = interpolate.Akima1DInterpolator(np.log10(xdata), ydata)
            self.alp_c = popt

    def fit_data(self, N_fit_upper, dsig_tol=0.01, maxiter=100):
        ''' Fits SN and CLD until convergence between consecutive loops is
        reached
        :param: N_fit_upper: upper cycle bound for fitting
        :param: dsig_tol: tolerance between consecutive loops
        :param: maxiter: maximum number of iterations  '''

        # start values
        self.alp_fit = 'exp'
        self.alp_c = [1., 0., 0.]
        sig = 50E+6

        i = 0
        while i < maxiter:
            i += 1
            self.fit_sn()
            self.project_data(N_fit_upper)
            self.alp_fit = 'aki'
            self.fit_cld()
            self.fit_alp()

            # standard deviation used as indicator for fit quality
            _, sign = s_weibull(
                self.sn_fit['lmbda'],
                self.sn_fit['delta'],
                self.sn_fit['beta']
            )
            dsig = 1. - sign / sig
            sig = sign
            if dsig < dsig_tol:
                print('SN/CLD fit converged')
                break

    def tangent_basquin_boerstra(self, p):
        ''' Find the neg. inv. S/N curve exponent for a Basquin fit that tangentes
         a p% quantile Stuessi curve
        Source: Rosemeier and Antoniou 2021, Eq. 26
        :param: p: target quantile curve
        '''
        m_fit = self.sn_fit['m_fit']
        Rt_fit = self.sn_fit['Rt_fit']
        Re_fit = self.sn_fit['Re_fit']
        Na = self.sn_fit['Na']
        lmbda = self.sn_fit['lmbda']
        delta = self.sn_fit['delta']
        beta = self.sn_fit['beta']

        # upper cycle number for exponential alp fit
        # use next lower cycle number within experemental cycle range
        Ne = np.max(
            self.dfns['ns'][(self.dfns['ns'] <= self.N_exp_upper)].to_numpy())

        # determine Re and Rt for given p
        xw_p = x_weibull(p, lmbda, delta, beta)
        Re_p = xw_p + Re_fit
        Rt_p = xw_p + Rt_fit
        # neg inv SN curve exponent at inflection point
        m_p = m_RtRd(m_fit, Rt_fit, Re_fit, Rt_p, Re_p)

        def fun(n_):
            # neg inv SN curve exponent at n_ on Stuessi curve
            mt_ = m_tangent(n_, m_p, Rt_p, Re_p, Na)
            # gradient of basquin == gadient of stuessi
            lhs = dsa_dn_basquin(n_, mt_, Rt_p)
            rhs = dsa_dn_stuessi(n_, m_p, Rt_p, Re_p, Na)
            return lhs - rhs

        N_start = Na
        N_end = self.ns[-1]
        # tangent point cycle number
        Nt = optimize.brentq(f=fun, a=N_start, b=N_end)
        # neg inv sn curve exponent through tangent point
        mt = m_tangent(Nt, m_p, Rt_p, Re_p, Na)

        # refit alp for tangent curve through 3 points
        # point 1: (N=1, alp=1)
        # point 2: (Nt, alpt)
        # point 3: (Ne, alpe)

        # alp at Nt
        alpt = self.alp_c(np.log10(Nt))
        # alp at Ne
        alpe = self.alp_c(np.log10(Ne))

        xdata = np.array([Nt, Ne])
        ydata = np.array([alpt, alpe])
        popt, _ = optimize.curve_fit(explogx1, xdata, ydata)

        self.alpt_c = popt
        self.alpt_fit = 'exp'

        self.sn_fit['Nt'] = Nt
        self.sn_fit['alpt_c'] = self.alpt_c.tolist()
        self.sn_fit['Re_p'] = Re_p
        self.sn_fit['Rt_p'] = Rt_p
        self.sn_fit['mt'] = mt
