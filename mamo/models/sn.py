import numpy as np
import scipy.optimize as optimize
from scipy.special import gamma
from collections import OrderedDict
from __builtin__ import False


def x_weibull(p, alpha, beta, gamma):
    ''' 
    Obains x value of a Weibull dsitribution for a given probability p
    (1) solve p = 1. - np.exp(-((x - alpha) / beta)**gamma) for x if beta>0
    and 
    (2) solve p = np.exp(-((x - alpha) / beta)**gamma) for x if beta<0
    Note: A negative beta flips the Weibull curve about x-axis
    :param: p: float probability
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: x: float value
    '''
    if bet < 0:
        x = alpha + beta * (-np.log(p))**(1. / gamma)
    else:
        x = alpha + beta * (-np.log(1. - p))**(1. / gamma)
    return x


def pwm_weibull(xs):
    '''
    Fits a three parameter Weibull distribution using PWM
    Source: Taosa Caiza and Ummenhofer 2011
    :param: xs: array of unsorted random variables
    :return: alp: float Weibull parameter
    :return: bet: float Weibull parameter
    :return: gam: float Weibull parameter
    '''
    n = len(xs)
    # sort xs ascending
    xs_sort = np.sort(xs)
    # eq. 19
    m0 = 1. / n * np.sum(xs_sort)
    # eq. 20
    sum1 = 0.
    for i in range(n):
        sum1 += (n - i + 1) * xs_sort[i]
    m1 = 1. / (n * (n - 1.)) * sum1
    # eq. 21
    sum2 = 0.
    for i in range(n - 1):
        sum2 += (n - i + 1) * (n - i) * xs_sort[i]
    m2 = 1. / (n * (n - 1.) * (n - 2.)) * sum2

    def objective(params):
        gam_ = params[0]
        # eq. 16
        crit = (3. * m2 - m0) / (2. * m1 - m0) - \
            (3.**(-1. / gam_) - 1.) / (2.**(-1. / gam_) - 1.)
        return crit
    gam_start = 0.5
    initial_guess = [gam_start]
    result = optimize.minimize(objective, initial_guess, method='SLSQP')
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)
    gam = fitted_params[0]

    gamma_gam = gamma(1. + 1. / gam)
    # eq. 17
    bet = (2. * m1 - m0) / ((2.**(-1. / gam) - 1) * gamma_gam)
    # eq. 18
    alp = m0 - bet * gamma_gam

    return alp, bet, gam


def R_ratio(sm, sa):
    ''' Stress ratio as function of amplitude and mean
    :param: sm: float mean stress
    :param: sa: float stress amplitude

    '''
    smin = sm - abs(sa)
    smax = sm + abs(sa)
    return smin / smax


def sm(sa, R):
    ''' Means stress as function of amplitude and stress ratio
    :param: sa: float stress amplitude
    :param: R: float stress ratio
    :return: sm: float mean stress 
    '''
    if R == 1:
        sm = 0.
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
        sa = 0.
    else:
        sa = sm / (1. + R) / (1. - R)
    return sa


def smax_sa(sa, R):
    ''' Max stress as function of amplitude and stress ratio
    :param: sa: float stress amplitude
    :param: R: float stress ratio
    :return: smax: float allowable maximum stress
    '''
    if R == 1:
        smax = sa
    else:
        smax = sa * ((1. + R) / (1. - R) + 1)
    return smax


def sa_smax(smax, R):
    ''' Stress amplitude as function of max stress and stress ratio
    :param: smax: float maximum stress
    :param: R: float stress ratio
    :return: sa: float stress amplitude
    '''
    if R == 1:
        sa = 0.
    else:
        sa = smax / ((1. + R) / (1. - R) + 1)
    return sa


def smax_basquin_goodman(n, R=-1, m=10, Rt=1.0, M=1):
    '''
    Max stress for given cycle number of an SN curve according to
    Basquin-Goodman
    :param: n: float cycle number
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient
    :param: Rt: float static strength
    :param: M: mean stress sensitivity
    :return: smax: float allowable maximum stress
    '''
    e_max = 2. / ((1. - R) * n**(1. / m) + M * (R + 1))
    return Rt * e_max


def N_basquin_goodman(smax, R, m, Rt, M):
    '''
    Allowable cycles for given ma xstress of an SN curve according to
    Basquin-Goodman
    :param: smax: float max stress
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient
    :param: Rt: float static strength
    :param: M: mean stress sensitivity
    :return: Nallow: float allowable cycles
    '''
    e_max = smax / Rt
    return ((e_max * M * (R + 1.) - 2.) / (e_max * (R - 1.)))**m


def Rt_basquin_goodman(smax, N, R, m, M):
    '''
    Static strength for a given point (N,smax) with neg. inverse S-N curve
    exponent according to Basquin-Goodman
    :param: smax: float max stress
    :param: N: float allowable cycles
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient
    :return: Rt: float static strength
    '''
    return 0.5 * smax * ((1. - R) * N**(1. / m) + M * (R + 1.))


def xrand_smax_basquin_goodman(smax_i, N_i, R_i, m_fit, Rt_fit, M_fit):
    ''' Random variable x using smax
    :return: xrand: float value
    '''
    return smax_i - smax_basquin_goodman(N_i, R_i, m_fit, Rt_fit, M_fit)


def xrand_Rt_basquin_goodman(smax_i, N_i, R_i, m_fit, Rt_fit, M_fit):
    ''' Random variable x using smax
    :return: xrand: float value
    '''
    Rt_i = Rt_basquin_goodman(smax_i, N_i, R_i, m_fit, M_fit)
    return Rt_i - Rt_fit


def smax_basquin_goodman_weibull(n, R, m, Rt_fit, M, p, alpha, beta, gamma):
    '''
    Max stress for given cycle number and probability of an SN curve according to Stuessi-Goodman-Weibull
    :param: n: float cycle number
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt_fit: float static strength (fit)
    :param: M: mean stress sensitivity
    :param: p: float probability
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: smax: float allowable maximum stress
    '''
    return x_weibull(p, alpha, beta, gamma) +\
        smax_basquin_goodman(n, R, m, Rt_fit, M)


def Rt_basquin_goodman_weibull(smax, N, R, m, Rt_fit, M, p, alpha, beta, gamma):
    '''
    Max stress for given cycle number and probability of an SN curve according to Stuessi-Goodman-Weibull
    :param: Rt_fit: float static strength (fit)
    :param: p: float probability
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: smax: float allowable maximum stress
    '''
    x_w = x_weibull(p, alpha, beta, gamma) + Rt_fit
    return 0.5 * (x_w + smax) * ((1. - R) * N**(1. / m) + M * (R + 1.))


def smax_limit_basquin_goodman_weibull(p, Rt_fit, alpha, beta, gamma):
    '''
    Endurance limit as function of p and R
    :param: p: float probability
    :param: Rt_fit: float static strength (fit)
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: Rt: float static strength for arbitrary p
    '''
    xw_p = x_weibull(p, alpha, beta, gamma)
    Rt = xw_p + Rt_fit

    return Rt


def _b(m, Re, Rt):
    '''
    Transforms Basquin's m into Stuessi's b
    (1) Derive Basquin for s: dn/ds(s) = -(Na*m*(s/R_a)**(-m))/s
    (2) Derive Stuessi for s: dn/ds(s) = -(Na*b*(Re-Rt)*
                                    ((-Re+s)/(Rt-s))**(-b))/((Re-s)*(Rt-s))
    (3) Set R_a = 0.5*(Re+Rt)
    (4) Set (1)=(2) with (3) and solve for b
    '''
    return (m * (-Re + Rt)) / (2 * (Re + Rt))


def sa_stuessi(n, R, m, Rt, M, Re, Na, n0=1.0):
    '''
    Stress amplitude of an SN curve according to Stuessi
    :param: n: float cycle number
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: Na: float turning point
    :param: n0: intersection with smax axis
    :return: sa: float allowable stress amplitude
    '''
    b = _b(m, Re, Rt)
    return (Re * ((n - n0) / Na)**(1. / b) + Rt) / \
        (1. + ((n - n0) / Na)**(1. / b))


def smax_stuessi_goodman(n, R=-1, m=10, Rt=1.0, M=1, Re=30, Na=1E3, n0=1.0):
    '''
    Max stress of an SN curve according to Stuessi-Goodman
    :param: n: float cycle number
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt: float static strength
    :param: M: mean stress sensitivity
    :param: Re: float endurance limit for R=-1
    :param: Na: float turning point
    :param: n0: intersection with smax axis
    :return: smax: float allowable maximum stress
    '''
    b = _b(m, Re, Rt)
    return (2. * Rt * (Re * ((n - n0) / Na)**(1. / b) + Rt)) / \
        (-Rt * (-1 + ((n - n0) / Na)**(1. / b) * (R - 1) - R *
                (M - 1.) - M) + Re * ((n - n0) / Na)**(1. / b) * (1. + R) * M)


def N_smax_stuessi_goodman(smax, R, m, Rt, M, Re, Na, n0):
    '''
    N(smax) of an SN curve according to Stuessi-Goodman
    :param: smax: float maximum stress
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt: float static strength
    :param: M: mean stress sensitivity
    :param: Re: float endurance limit for R=-1
    :param: Na: float turning point
    :param: n0: intersection with smax axis
    :return: n: float allowable cycle number
    '''
    b = _b(m, Re, Rt)
    return n0 + Na * ((smax * Rt * (1 + R * (M - 1) + M) - 2. * Rt**2) /
                      (2. * Re * Rt - smax * ((1. - R) * Rt + Re * (1. + R) * M)))**b


def smax_limit_stuessi_goodman(R, Rt, Re, M=1):
    '''
    Endurance limit of function smax_stuessi_goodman for n->inf
    :param: R: float stress ratio
    :param: Rt: float static strength
    :param: Re: float endurance limit for R=-1
    :param: M: mean stress sensitivity
    :return: smax: float endurance limit for arbitrary R
    '''
    return (2. * Re * Rt) / (Re * M * (R + 1.) + Rt * (1. - R))


def smax_limit_stuessi_goodman_RdRt(R, Rt, Rd_Rt, M=1):
    '''
    Endurance limit of function smax_stuessi_goodman for n->inf
    :param: R: float stress ratio
    :param: Rt: float static strength
    :param: Rt_Rd: float ratio between static strength and endurance limit (R=-1)
    :param: M: mean stress sensitivity
    :return: smax: float endurance limit for arbitrary stress ratios R and Rd/Rt ratios
    '''
    return (2. * Rd_Rt * Rt) / (Rd_Rt * M * (R + 1.) + (1. - R))


def xrand_sa_stuessi(sa_i, N_i, R_i, m_fit, Rt_fit, M_fit, Re_fit, Na_fit, n0=0.):
    ''' Random variable x using sa
    :return: xrand: float value
    '''
    return sa_i - sa_stuessi(N_i, R_i, m_fit, Rt_fit, M_fit, Re_fit, Na_fit, n0)


def xrand_smax_stuessi_goodman(smax_i, N_i, R_i, m_fit, Rt_fit, M_fit, Re_fit, Na_fit, n0=0.):
    ''' Random variable x using smax
    :return: xrand: float value
    '''
    return smax_i - smax_stuessi_goodman(N_i, R_i, m_fit, Rt_fit, M_fit, Re_fit, Na_fit, n0)


def sa_stuessi_weibull(n, R, m, Rt_fit, M, Re_fit, Na, p, alpha, beta, gamma, n0):
    '''
    Max stress for given cycle number and probability of an SN curve according to Stuessi-Goodman-Weibull
    :param: n: float cycle number
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt_fit: float static strength (fit)
    :param: M: mean stress sensitivity
    :param: Re_fit: float endurance limit for R=-1 (fit)
    :param: Na: float turning point
    :param: p: float probability
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: sa: float allowable maximum stress
    '''
    return x_weibull(p, alpha, beta, gamma) + sa_stuessi(n, R, m, Rt_fit, M, Re_fit, Na, n0)


def smax_stuessi_goodman_weibull(n, R, m, Rt_fit, M, Re_fit, Na, p, alpha, beta, gamma, n0):
    '''
    Max stress for given cycle number and probability of an SN curve according to Stuessi-Goodman-Weibull
    :param: n: float cycle number
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt_fit: float static strength (fit)
    :param: M: mean stress sensitivity
    :param: Re_fit: float endurance limit for R=-1 (fit)
    :param: Na: float turning point
    :param: p: float probability
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: smax: float allowable maximum stress
    '''
    return x_weibull(p, alpha, beta, gamma) +\
        smax_stuessi_goodman(n, R, m, Rt_fit, M, Re_fit, Na, n0)


def N_smax_stuessi_goodman_weibull(smax, R, m, Rt_fit, M, Re_fit, Na, p, alpha, beta, gamma, n0):
    '''
    N(smax) for given max stress and probability of an SN curve according to Stuessi-Goodman-Weibull
    :param: smax: float maximum stress
    :param: R: float stress ratio
    :param: m: float negative inverse SN curve coefficient at Na for R=-1
    :param: Rt_fit: float static strength (fit)
    :param: M: mean stress sensitivity
    :param: Re_fit: float endurance limit for R=-1 (fit)
    :param: Na: float turning point
    :param: n0: intersection with smax axis
    :param: p: float probability
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: n: float allowable cycle number
    '''
    xw = x_weibull(p, alpha, beta, gamma)
    return N_smax_stuessi_goodman(xw + smax, R, m, Rt_fit, M, Re_fit, Na, n0)


def smax_limit_stuessi_goodman_weibull(p, R, Rt_fit, M, Re_fit, alpha, beta, gamma):
    '''
    Endurance limit as function of p and R
    :param: p: float probability
    :param: R: float stress ratio
    :param: Rt_fit: float static strength (fit)
    :param: M: mean stress sensitivity
    :param: Re_fit: float endurance limit for R=-1 (fit)
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: Rt: float static strength for arbitrary p
    :return: Re: float endurance limit for arbitrary R and p
    '''
    xw_p = x_weibull(p, alpha, beta, gamma)
    Re = xw_p + smax_limit_stuessi_goodman(R, Rt_fit, Re_fit, M)
    Rt = xw_p + Rt_fit

    return Rt, Re


def fit_basquin_goodman(cyc_data,
                        m_start=11.,
                        Rt_start=40.):
    '''
    Fits m and Rt to a set of experimental data
    :param: cyc_data: dict with experimental data
    :param: m_start: float start value
    :param: Rt_start: float start value
    :return: m_fit
    :return: Rt_fit
    '''

    initial_guess = [m_start, Rt_start]

    def objective(params):
        m, Rt = params  # multiple design variables

        crits = []
        for smax_actual, R, Nactual in zip(
                cyc_data['cyc_stress_max'][cyc_data['grps']],
                cyc_data['cyc_ratios'][cyc_data['grps']],
                cyc_data['cyc_cycles'][cyc_data['grps']]):
            if not R == 1:
                N = N_basquin_goodman(smax_actual, R, m, Rt, M=1)
            else:
                N = 1
            crit1 = (np.log10(N) - np.log10(Nactual))**2
            crits.append(crit1)

            include_sigma_max_fit = True
            if include_sigma_max_fit:
                smax = smax_basquin_goodman(Nactual, R, m, Rt, M=1)
                crit2 = (np.log10(smax) - np.log10(smax_actual))**2
                crits.append(crit2)

        critsum = np.sum(np.array(crits))  # objective function
        return critsum
    result = optimize.minimize(objective, initial_guess, method='Nelder-Mead')
    if result.success:
        fitted_params = result.x
        # print(fitted_params)
    else:
        raise ValueError(result.message)

    m_fit, Rt_fit = fitted_params

    return m_fit, Rt_fit


def Rt_weibull(cyc_data, Rt_fit, m_fit, xw):
    M = 1  # constants
    xs = np.zeros(len(cyc_data['grps']))
    for i, (smax_actual, R, Nactual) in enumerate(
        zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
            cyc_data['cyc_ratios'][cyc_data['grps']],
            cyc_data['cyc_cycles'][cyc_data['grps']])):

        # TODO: replace with xrand_Rt_basquin_goodman_weibull to calculate
        # distance to R_50 value
        xs[i] = xrand_Rt_basquin_goodman(
            xw + smax_actual, Nactual, R, m_fit, Rt_fit, M)

    alp, bet, gam = pwm_weibull(xs)
    return alp, bet, gam


def fit_basquin_goodman_weibull(cyc_data,
                                m_start=11.,
                                Rt_start=40.):
    '''
    Fits m and Rt to a set of experimental data
    :param: cyc_data: dict with experimental data
    :param: m_start: float start value
    :param: Rt_start: float start value
    :return: m_fit
    :return: Rt_fit
    '''

    initial_guess = [m_start, Rt_start]

    def objective(params):
        m, Rt = params  # multiple design variables
        M = 1  # constants
        xs = np.zeros(len(cyc_data['grps']))
        for i, (smax_actual, R, Nactual) in enumerate(
            zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                cyc_data['cyc_ratios'][cyc_data['grps']],
                cyc_data['cyc_cycles'][cyc_data['grps']])):

            use_smax = True
            if use_smax:
                xs[i] = xrand_smax_basquin_goodman(
                    smax_actual, Nactual, R, m, Rt, M)
            else:
                xs[i] = xrand_Rt_basquin_goodman(
                    smax_actual, Nactual, R, m, Rt, M)

        global alp, bet, gam
        alp, bet, gam = pwm_weibull(xs)

        crits = []

        for i, (smax_actual, R, Nactual) in enumerate(zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                                                          cyc_data['cyc_ratios'][cyc_data['grps']],
                                                          cyc_data['cyc_cycles'][cyc_data['grps']])):

            p = 0.50
            if use_smax:
                smax_50 = smax_basquin_goodman_weibull(
                    Nactual, R=R, m=m, Rt_fit=Rt, M=M,
                    p=p, alpha=alp, beta=bet, gamma=gam)

                crit1 = (np.log10(smax_50) - np.log10(smax_actual))**2
            else:
                Rt_50 = Rt_basquin_goodman_weibull(
                    Rt_fit=Rt, p=p, alpha=alp, beta=bet, gamma=gam)

                Rt_actual = Rt_basquin_goodman(smax_actual, Nactual, R, m)

                crit1 = (Rt_50 - Rt_actual)**2

            crits.append(crit1)

        critsum = np.sum(np.array(crits))  # objective function
        return critsum

    result = optimize.minimize(objective, initial_guess, method='Nelder-Mead')
    if result.success:
        fitted_params = result.x
        # print(fitted_params)
    else:
        raise ValueError(result.message)

    m_fit, Rt_fit = fitted_params

    return m_fit, Rt_fit, alp, bet, gam


def fit_stuessi_weibull(cyc_data,
                        m_start=11.,
                        Rt_start=70.,
                        Re_start=40.,
                        Na_start=1E3,
                        n0=0.):
    '''
    Fits m, Rt, Re, Na and Weibull parameters to a set of experimental data
    :param: cyc_data: dict with experimental data
    :param: m_start: float start value
    :param: Rt_start: float start value
    :param: Re_start: float start value
    :param: Na_start: float start value
    :return: m_fit
    :return: Rt_fit
    :return: Re_fit
    :return: Na_fit
    :return: alp
    :return: bet
    :return: gam
    Note: a fitted curve can be obtained with:
    smax = smax_stuessi_goodman_weibull(n, R=R, m=m_fit, Rt=Rt_fit, M=M,
                Re=Re_fit, Na=Na_fit, p=p, alpha=alp, beta=bet, gamma=gam)
    '''
    initial_guess = [m_start, Rt_start, Re_start, Na_start]

    def objective(params):
        m, Rt, Re, Na = params  # multiple design variables
        M = 1  # constants
        xs_smax = np.zeros(len(cyc_data['grps']))
        for i, (smax_actual, R, Nactual) in enumerate(
            zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                cyc_data['cyc_ratios'][cyc_data['grps']],
                cyc_data['cyc_cycles'][cyc_data['grps']])):
            xs_smax[i] = xrand_sa_stuessi(
                smax_actual, Nactual, R, m, Rt, M, Re, Na, n0)

        global alp, bet, gam
        alp, bet, gam = pwm_weibull(xs_smax)

        crits = []
        for i, (smax_actual, R, Nactual) in enumerate(
            zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                cyc_data['cyc_ratios'][cyc_data['grps']],
                cyc_data['cyc_cycles'][cyc_data['grps']])):

            p = 0.50
            smax_50 = sa_stuessi_weibull(
                Nactual, R=R, m=m, Rt_fit=Rt, M=M,
                Re_fit=Re, Na=Na,
                p=p, alpha=alp, beta=bet, gamma=gam, n0=n0)

            use_log10 = True
            if use_log10:
                crit1 = (np.log10(smax_50) - np.log10(smax_actual))**2
            else:
                crit1 = (smax_50 - smax_actual)**2

            crits.append(crit1)

        critsum = np.sum(np.array(crits))  # objective function
        return critsum

    options = {}
    #options['xtol'] = 1E-1
    options['disp'] = False
    options['maxiter'] = 1E5
    options['maxfev'] = 1E5
    result = optimize.minimize(
        objective, initial_guess, method='Nelder-Mead', options=options)
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)

    m_fit, Rt_fit, Re_fit, Na_fit = fitted_params

    return m_fit, Rt_fit, Re_fit, Na_fit, alp, bet, gam


def fit_stuessi_goodman_weibull(cyc_data,
                                m_start=11.,
                                Rt_start=70.,
                                Re_start=40.,
                                Na_start=1E3,
                                n0=0.):
    '''
    Fits m, Rt, Re, Na and Weibull parameters to a set of experimental data
    :param: cyc_data: dict with experimental data
    :param: m_start: float start value
    :param: Rt_start: float start value
    :param: Re_start: float start value
    :param: Na_start: float start value
    :return: m_fit
    :return: Rt_fit
    :return: Re_fit
    :return: Na_fit
    :return: alp
    :return: bet
    :return: gam
    Note: a fitted curve can be obtained with:
    smax = smax_stuessi_goodman_weibull(n, R=R, m=m_fit, Rt=Rt_fit, M=M,
                Re=Re_fit, Na=Na_fit, p=p, alpha=alp, beta=bet, gamma=gam)
    '''
    initial_guess = [m_start, Rt_start, Re_start, Na_start]

    def objective(params):
        m, Rt, Re, Na = params  # multiple design variables
        M = 1  # constants
        xs_smax = np.zeros(len(cyc_data['grps']))
        for i, (smax_actual, R, Nactual) in enumerate(
            zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                cyc_data['cyc_ratios'][cyc_data['grps']],
                cyc_data['cyc_cycles'][cyc_data['grps']])):
            xs_smax[i] = xrand_smax_stuessi_goodman(
                smax_actual, Nactual, R, m, Rt, M, Re, Na, n0)

        global alp, bet, gam
        alp, bet, gam = pwm_weibull(xs_smax)

        crits = []
        for i, (smax_actual, R, Nactual) in enumerate(
            zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                cyc_data['cyc_ratios'][cyc_data['grps']],
                cyc_data['cyc_cycles'][cyc_data['grps']])):

            p = 0.50
            smax_50 = smax_stuessi_goodman_weibull(
                Nactual, R=R, m=m, Rt_fit=Rt, M=M,
                Re_fit=Re, Na=Na,
                p=p, alpha=alp, beta=bet, gamma=gam, n0=n0)

            use_log10 = True
            if use_log10:
                crit1 = (np.log10(smax_50) - np.log10(smax_actual))**2
            else:
                crit1 = (smax_50 - smax_actual)**2

            crits.append(crit1)

        critsum = np.sum(np.array(crits))  # objective function
        return critsum

    options = {}
    #options['xtol'] = 1E-1
    options['disp'] = False
    options['maxiter'] = 1E5
    options['maxfev'] = 1E5
    result = optimize.minimize(
        objective, initial_guess, method='Nelder-Mead', options=options)
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)

    m_fit, Rt_fit, Re_fit, Na_fit = fitted_params

    return m_fit, Rt_fit, Re_fit, Na_fit, alp, bet, gam


class SNFit(object):
    '''
    Fits Basquin- and Stuessi-Goodman type curves to a set of experimental data
    '''

    def __init__(self, fit_type='basquin_goodman'):
        '''
        :param: fit_type: str basquin_goodman or stuessi-goodman
        '''
        self.fit_type = fit_type

        if self.fit_type == 'basquin-goodman':
            self._init_basquin_goodman()
        elif self.fit_type == 'stuessi-goodman' or self.fit_type == 'stuessi':
            self._init_stuessi_goodman()

    def _init_basquin_goodman(self):
        self.m_start = 10.0
        self.Rt_start = 50.0E+6

    def _init_stuessi_goodman(self):
        self.m_start = 8.5
        self.Rt_start = 67.E+6
        self.Re_start = 9.E+6
        self.Na_start = 68.
        self.n0 = 1.

    def load_data(self, data, grp_entries):
        '''
        :param: data: array with test data format: # ID, sigma_a, sigma_m,  ratio , cycles to failure
        :param: grp_entries: list of group entries in data
        '''
        self.cyc_data = {}
        self.cyc_data['ids'] = data[:, 0]
        self.cyc_data['cyc_stress_a'] = cyc_stress_a = abs(data[:, 1])
        self.cyc_data['cyc_stress_m'] = cyc_stress_m = abs(data[:, 2])
        cyc_stress_a_sign = np.ones_like(cyc_stress_a)  # np.sign(data[:, 0])
        cyc_stress_m_sign = np.ones_like(cyc_stress_a)  # np.sign(data[:, 1])

        self.cyc_data['cyc_stress_max'] = cyc_stress_max = cyc_stress_m_sign * cyc_stress_m + \
            cyc_stress_a  # MPa
        self.cyc_data['cyc_stress_min'] = cyc_stress_min = cyc_stress_m_sign * cyc_stress_m - \
            cyc_stress_a  # MPa
        self.cyc_data['cyc_ratios'] = cyc_ratios = cyc_stress_min / \
            cyc_stress_max
        self.cyc_data['cyc_cycles'] = cyc_cycles = data[:, 4]  # N

        self.cyc_data['grplist'] = grps = grp_entries
        #self.cyc_data['grplist'] = grps = []
        # for i, ni in enumerate(grp_entries):
        #    if i == 0:
        #        grp = list(np.arange(ni))
        #        grps.append(grp)
        #    else:
        #        grp0 = grps[i - 1]
        #        grp = list(np.arange(grp0[-1] + 1, grp0[-1] + ni + 1))
        #        grps.append(grp)

        self.cyc_data['grps'] = [item for sublist in grps
                                 for item in sublist]

        self.cyc_data['cyc_ratio_grp'] = np.zeros(len(grps))

        for i, grp in enumerate(grps):

            self.cyc_data['cyc_ratio_grp'][i] = np.mean(cyc_ratios[grp])

    def fit_data(self):
        '''
        :return: sn_fit: dict with fitting parameters
        '''

        self.sn_fit = OrderedDict()

        if self.fit_type == 'basquin-goodman':
            m, Rt_fit, alpha_, beta_, gamma_ =\
                fit_basquin_goodman_weibull(self.cyc_data,
                                            self.m_start,
                                            self.Rt_start)

            Rt_50 = smax_limit_basquin_goodman_weibull(p=0.5,
                                                       Rt_fit=Rt_fit,
                                                       alpha=alpha_,
                                                       beta=beta_,
                                                       gamma=gamma_)
            alpha, beta, gamma = alpha_, beta_, gamma_

            # xw_50 = x_weibull(p=0.5, alpha=alpha_,
            #                  beta=beta_, gamma=gamma_)

            #alpha, beta, gamma = Rt_weibull(self.cyc_data, Rt_fit, m, xw_50)

            #Rt_fit = Rt_50

        elif self.fit_type == 'stuessi-goodman':

            m, Rt_fit, Re_fit, Na, alpha, beta, gamma =\
                fit_stuessi_goodman_weibull(self.cyc_data,
                                            self.m_start,
                                            self.Rt_start,
                                            self.Re_start,
                                            self.Na_start,
                                            self.n0)

            Rt_50, Re_50 = smax_limit_stuessi_goodman_weibull(p=0.5,
                                                              R=-1,
                                                              Rt_fit=Rt_fit,
                                                              M=1,
                                                              Re_fit=Re_fit,
                                                              alpha=alpha,
                                                              beta=beta,
                                                              gamma=gamma)

            _, Re_50_R = smax_limit_stuessi_goodman_weibull(p=0.5,
                                                            R=self.cyc_data['cyc_ratio_grp'],
                                                            Rt_fit=Rt_fit,
                                                            M=1,
                                                            Re_fit=Re_fit,
                                                            alpha=alpha,
                                                            beta=beta,
                                                            gamma=gamma)

            self.sn_fit['Re_fit'] = Re_fit
            self.sn_fit['Na'] = Na
            self.sn_fit['Re_50'] = Re_50
            self.sn_fit['Re_50_R'] = Re_50_R.tolist()

        elif self.fit_type == 'stuessi':
            m, Rt_fit, Re_fit, Na, alpha, beta, gamma =\
                fit_stuessi_weibull(self.cyc_data,
                                    self.m_start,
                                    self.Rt_start,
                                    self.Re_start,
                                    self.Na_start,
                                    self.n0)

            Rt_50 = 0.
            Re_50 = 0.
            Re_50_R = np.array([0., 0.])
            self.sn_fit['Re_fit'] = Re_fit
            self.sn_fit['Na'] = Na
            self.sn_fit['Re_50'] = Re_50
            self.sn_fit['Re_50_R'] = Re_50_R.tolist()

        self.sn_fit['m'] = m
        self.sn_fit['Rt_fit'] = Rt_fit
        self.sn_fit['alpha'] = alpha
        self.sn_fit['beta'] = beta
        self.sn_fit['gamma'] = gamma
        self.sn_fit['Rt_50'] = Rt_50

    def cld(self, ns):
        '''
        Obrain CLD from SNfit
        :return: sas: array (ngrp+1,npoint) stress amplitudes
        :return: sms: array (ngrp+1,npoint) mean stress
        :return: sn_grp: dict with sub dicts of fitting parameters per group
        '''

        self.ns = ns
        ngrp = len(self.cyc_data['grplist'])
        npoint = len(ns)

        nR = 1000

        self.sas = np.zeros((nR, npoint))
        self.sms = np.zeros((nR, npoint))

        R = np.linspace(-1.0, +1.0, nR)

        if self.fit_type == 'stuessi-goodman':
            grps = self.cyc_data['grplist']
            cyc_stress_max = self.cyc_data['cyc_stress_max']
            cyc_cycles = self.cyc_data['cyc_cycles']
            cyc_ratio_grp = self.cyc_data['cyc_ratio_grp']
            m_fit = self.sn_fit['m']
            Rt_fit = self.sn_fit['Rt_fit']
            Re_fit = self.sn_fit['Re_fit']
            Na_fit = self.sn_fit['Na']
            alp_smax_fit = self.sn_fit['alpha']
            bet_smax_fit = self.sn_fit['beta']
            gam_smax_fit = self.sn_fit['gamma']
            Rt_50 = self.sn_fit['Rt_50']
            Re_50 = self.sn_fit['Re_50']
            Re_50_R = self.sn_fit['Re_50_R']
            n0 = self.n0

            p = 0.50
            M_fit = 1
            for ni, n_ in enumerate(ns):
                for Ri, R_ in enumerate(R):
                    smax_50 = smax_stuessi_goodman_weibull(
                        n_, R=R_, m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                        Re_fit=Re_fit, Na=Na_fit,
                        p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)

                    self.sas[Ri, ni] = sa_50 = sa_smax(smax_50, R_)
                    if R_ == 1:
                        sm_50 = smax_50
                    else:
                        sm_50 = sm(sa_50, R_)
                    self.sms[Ri, ni] = sm_50


class CLDFit(SNFit):
    '''
    Object that fits Stuessi to groups of measurement data independently
    As result constant life digrams are generated from these fits
    '''

    def __init__(self):
        super(CLDFit, self).__init__(fit_type='stuessi')  # needs to be stuessi

    def fit_data_groups(self, ns, data, grp_entries_list):
        '''
        :return: sas: array (ngrp+1,npoint) stress amplitudes
        :return: sms: array (ngrp+1,npoint) mean stress
        :return: sn_grp: dict with sub dicts of fitting parameters per group
        '''

        self.ns = ns

        ngrp = len(grp_entries_list)
        npoint = len(ns)

        self.sas = np.zeros((ngrp + 1, npoint))
        self.sms = np.zeros((ngrp + 1, npoint))
        Rt = np.zeros(ngrp)

        self.sn_grp = OrderedDict()

        for i, grp_entries in enumerate(grp_entries_list):
            self.load_data(data, grp_entries)
            self.fit_data()

            self.sn_grp[i] = OrderedDict()
            self.sn_grp[i]['sn_fit'] = self.sn_fit
            self.sn_grp[i]['cyc_data'] = self.cyc_data

            grps = self.cyc_data['grplist']
            cyc_stress_max = self.cyc_data['cyc_stress_max']
            cyc_cycles = self.cyc_data['cyc_cycles']
            cyc_ratio_grp = self.cyc_data['cyc_ratio_grp']
            m_fit = self.sn_fit['m']
            Rt_fit = self.sn_fit['Rt_fit']
            Re_fit = self.sn_fit['Re_fit']
            Na_fit = self.sn_fit['Na']
            alp_smax_fit = self.sn_fit['alpha']
            bet_smax_fit = self.sn_fit['beta']
            gam_smax_fit = self.sn_fit['gamma']
            Rt_50 = self.sn_fit['Rt_50']
            Re_50 = self.sn_fit['Re_50']
            Re_50_R = self.sn_fit['Re_50_R']
            n0 = self.n0

            M_fit = 1

            if len(grp_entries) == 1:  # only cyclic data group present
                gidx = 0
            elif len(grp_entries) == 2:  # one static and one cyclic group present
                gidx = 1
            R = cyc_ratio_grp[gidx]
            p = 0.50
            smax_50 = sa_stuessi_weibull(
                ns, R=R, m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                Re_fit=Re_fit, Na=Na_fit,
                p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)

            self.sas[i + 1, :] = sa_50 = sa_smax(smax_50, R)
            self.sms[i + 1, :] = sm_50 = sm(sa_50, R)
            Rt[i] = Rt_50 = smax_50[0]

        self.sms[0, :] = np.mean(Rt)

        del self.sn_fit
        del self.cyc_data
