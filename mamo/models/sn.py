import numpy as np
import scipy.optimize as optimize
from scipy.special import gamma
from collections import OrderedDict


def bisection(fun, lower=0, upper=1, tol=1e-6, maxiter=100, callback=None):
    '''Bisection method to solve fun(x)=0,
       assuming root is between lower and upper.
       Callback function arguments (iter,x,lower,upper)
    '''
    if fun(lower) * fun(upper) > 0:
        raise(ValueError('Bad initial lower and upper limits'))
    for iter in range(maxiter):
        x = (lower + upper) / 2
        if callback:
            callback(iter, x, lower, upper)
        if fun(lower) * fun(x) > 0:
            lower = x
        else:
            upper = x
        if np.abs(upper - lower) < tol:
            return (lower + upper) / 2
    else:
        raise(RuntimeError('Failed to converge, increase maxiter'))


def bnewton(fun, grad, lower=0, upper=1, tol=1e-6, maxiter=100, callback=None):
    '''Polyalgorithm that combines bisections and Newton-Raphson
       to solve fun(x)=0 within given lower and upper bounds.
       Callback function arguments (iter,itertype,x,x1,lower,upper)
    '''
    sign_lower = np.sign(fun(lower))
    sign_upper = np.sign(fun(upper))
    if sign_lower * sign_upper > 0:
        raise(ValueError('Bad initial lower and upper limits'))
    x = (lower + upper) / 2
    for iter in range(maxiter):
        newt = x - fun(x) / grad(x)  # Newton step
        if newt < lower or newt > upper:
            # bisection step
            if np.sign(fun(x)) * sign_lower > 0:
                if callback:
                    callback(iter, 'bisect', x, (x + upper) / 2, lower, upper)
                lower = x  # and the lower sign remains
            else:
                if callback:
                    callback(iter, 'bisect', x, (x + lower) / 2, lower, upper)
                upper = x  # and the upper sign remains
            x1 = (lower + upper) / 2
            stopping = np.abs(upper - lower)
        else:
            x1 = newt
            stopping = np.abs(x1 - x)
            if callback:
                callback(iter, 'newton', x, x1, lower, upper)
        x = x1
        if stopping < tol:
            return x1
    else:
        raise(RuntimeError('Failed to converge, increase maxiter'))


def explogx(x, a, b, c):
    return a * x**-b + c


def poly1dlogx(x, a, b, c):
    return a * np.log10(x) + b


def p_weibull(x, alpha, beta, gamma):
    ''' 
    Obains probability of a given Weibull distribution for a given variable x
    p = 1. - np.exp(-((x - alpha) / beta)**gamma) for x if beta>0
    p = np.exp(-((x - alpha) / beta)**gamma) for x if beta<0
    Note: A negative beta flips the Weibull curve about x-axis
    :param: x: float value probability
    :param: alpha: float Weibull parameter
    :param: beta: float Weibull parameter
    :param: gamma: float Weibull parameter
    :return: p: float probability
    '''
    if beta < 0:
        x = np.exp(-((x - alpha) / beta)**gamma)
    else:
        x = 1. - np.exp(-((x - alpha) / beta)**gamma)
    return x


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
    if beta < 0:
        x = alpha + beta * (-np.log(p))**(1. / gamma)
    else:
        x = alpha + beta * (-np.log(1. - p))**(1. / gamma)
    return x


def s_weibull(alp, bet, gam):
    ''' Shifley and Lentz 1985, Quick estimation of the three-parameter Weibull
    to describe tree size distributions

    '''
    # mean value
    mu_alp = bet * gamma(1. + (1. / gam))  # Eq. 2
    mu = mu_alp + alp  # Eq. 2

    # variance
    sig2 = bet**2 * (gamma(1. + 2. / gam) - gamma(1. + (1. / gam))**2)  # Eq. 3

    # standard deviation
    sig = np.sqrt(sig2)

    return mu, sig


def pwm_weibull(xs, method='Ms'):
    '''
    Fits a three parameter Weibull distribution using PWM
    Source: Toasa Caiza and Ummenhofer 2011
    :param: xs: array of unsorted random variables
    :return: alp: float Weibull parameter
    :return: bet: float Weibull parameter
    :return: gam: float Weibull parameter
    '''
    n = len(xs)
    # sort xs ascending
    xs_sort = np.sort(xs)

    if method == 'Ms':
        '''
        Implementation of eq. 20 and 21 as of Toasa not correct when compared to
        Rinne 2008 The Weibull Distribution, p. 473
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
        '''

        # eq. 12.21 Rinne
        m0 = A0 = 1. / n * np.sum(xs_sort)

        sum1 = 0.
        for i in range(1, n + 1):
            sum1 += (n - i) / (n - 1.) * xs_sort[i - 1]
        m1 = A1 = 1. / n * sum1

        sum2 = 0.
        for i in range(1, n + 1):
            sum2 += (n - i) * (n - i - 1.) / \
                ((n - 1.) * (n - 2.)) * xs_sort[i - 1]
        m2 = A2 = 1. / n * sum2

        def objective(gam_):
            # eq. 16
            crit = (3.**(-1. / gam_) - 1.) / (2.**(-1. / gam_) - 1.) - \
                (3. * m2 - m0) / (2. * m1 - m0)

            return crit
        gam_start = 0.5
        # try:
        gam = optimize.newton(objective, gam_start)
        # except:
        #    gam = np.nan
        '''
        import matplotlib.pyplot as plt
        gam_ = np.linspace(0., 10, 1E+5)
        plt.plot(gam_, objective(gam_))
        '''
        # eq. 10
        gamma_gam = gamma(1. + 1. / gam)
        # eq. 17
        bet = (2. * m1 - m0) / ((2.**(-1. / gam) - 1) * gamma_gam)
        # eq. 18
        alp = m0 - bet * gamma_gam

    elif method == 'Mr':

        # eq. 38
        m0 = 1. / n * np.sum(xs_sort)

        # eq. 39
        sum1 = 0.
        for i in range(n):
            sum1 += i * xs_sort[i]
        m1 = 1. / (n * (n - 1.)) * sum1

        # eq. 40
        sum2 = 0.
        for i in range(n):
            sum2 += i * (i - 1.) * xs_sort[i]
        m2 = 1. / (n * (n - 1.) * (n - 2.)) * sum2

        def objective(gam_):
            # eq. 35
            crit = (3. * m2 - m0) / (2. * m1 - m0) - \
                (2. - 3. * 2.**(-1. / gam_) + 3. ** (-1. / gam_)) /\
                (1. - 2.**(-1. / gam_))
            return crit
        gam_start = 0.5
        gam = optimize.newton(objective, gam_start)

        '''
        import matplotlib.pyplot as plt
        gam_ = np.linspace(0., 10, 1E+5)
        plt.plot(gam_, objective(gam_))
        '''

        # eq. 29
        gamma_gam = gamma(1. + 1. / gam)

        # eq. 36
        bet = (2. * m1 - m0) / ((1. - 2.**(-1. / gam)) * gamma_gam)
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


def sa_goodman(sa0, sm, Rt):
    ''' Stress amplitude according to modified Goodman 1899 
    :param: sa0: float stress amplitude at zero means stress (R=-1)
    :param: sm: float mean stress
    :param: Rt: float tensile strength
    :return: sa: float stress amplitude
    '''
    return sa0 * (1. - sm / Rt)


def sa_gerber(sa0, sm, Rt):
    ''' Stress amplitude according to Gerber 1874
    :param: sa0: float stress amplitude at zero means stress (R=-1)
    :param: sm: float mean stress
    :param: Rt: float tensile strength
    :return: sa: float stress amplitude
    '''
    return sa0 * (1. - (sm / Rt)**2)


def sa_loewenthal(sa0, sm, Rt):
    ''' Stress amplitude according to Loewenthal 1975
    :param: sa0: float stress amplitude at zero means stress (R=-1)
    :param: sm: float mean stress
    :param: Rt: float tensile strength
    :return: sa: float stress amplitude
    '''
    return sa0 * np.sqrt(1. - (sm / Rt)**2)


def sa_swt(sa0, sm, Rt):
    ''' Stress amplitude according to Smith Watson Topper 1970
    :param: sa0: float stress amplitude at zero means stress (R=-1)
    :param: sm: float mean stress
    :param: Rt: float tensile strength
    :return: sa: float stress amplitude
    '''
    return 0.5 * (np.sqrt(sm**2 + 4. * sa0**2) - sm)


def sa_tosa(sa0, sm, Rt, alp):
    ''' Stress amplitude according to Topper Sandor 1970
    :param: sa0: float stress amplitude at zero means stress (R=-1)
    :param: sm: float mean stress
    :param: Rt: float tensile strength
    :return: sa: float stress amplitude
    '''
    return sa0 - sm**alp


def sa_boerstra(sa0, sm, Rt, alp):
    ''' Stress amplitude according to Topper Sandor 1970
    :param: sa0: float stress amplitude at zero means stress (R=-1)
    :param: sm: float mean stress
    :param: Rt: float tensile strength
    :return: sa: float stress amplitude
    '''
    return sa0 * (1. - (sm / Rt)**alp)


def sa_basquin(n, m, Rt):
    ''' Stress amplitude according to Basquin
    :param: n: float cycle number
    :param: m: float negative inverse SN curve coefficient
    :param: Rt: float static strength
    :return: sa: float stress amplitude
    '''
    return Rt * n**(-1. / m)


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


def xrand_smax_basquin_goodman_weibull(smax_i, N_i, R_i, m_fit, Rt_fit, M_fit, p, alpha, beta, gamma):
    ''' Random variable x using smax
    :return: xrand: float value
    '''
    return smax_i - smax_basquin_goodman_weibull(
        N_i, R_i, m_fit, Rt_fit, M_fit, p, alpha, beta, gamma)


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


def N_stuessi(sa, m, Rt, Re, Na, n0=1.0):
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
    return n0 - Na * ((Rt - sa) / (sa - Re))**b


def sa_stuessi_boerstra(n, R, m, Rt, M, Re, Na, alp, n0=1.0):
    b = _b(m, Re, Rt)
    n_ = ((n - n0) / Na)**(1. / b)
    sa = (1. - (sm / Rt)**alp) * (Re * n_ + Rt) / (1. + n_)
    return sa


def sm_stuessi_boerstra(n, R, m, Rt, M, Re, Na, d, n0=1.0):
    b = _b(m, Re, Rt)
    n_ = ((n - n0) / Na)**(1. / b)
    #sm = Rt * ((sa * (n_ - 1.) + Re * n_ + Rt) / (Re * n_ + Rt))**(1. / d)
    sm = Rt * (1. - sa * (n_ - 1.) / (Re * n_ + Rt))**(1. / d)
    return sm


def N_sa_sm_stuessi_boerstra(n, R, m, Rt, M, Re, Na, alp, n0=1.0):
    b = _b(m, Re, Rt)
    n = n0 + Na * (-1. * (sa + Rt * ((sm / Rt)**alp - 1.)) /
                   (sa + Re * ((sm / Rt)**alp - 1.)))**b
    return n


def N_smax_stuessi_boerstra(smax, R, m, Rt, Re, Na, alp, n0=1.0):
    b = _b(m, Re, Rt)

    if R == -1.0:
        Rnum = R + 1E-12
    elif R == +1.0:
        Rnum = R - 1E-12
    else:
        Rnum = R

    salp = (abs(smax * (1. + Rnum)) / (2. * Rt))**alp - 1.

    rhs_nom = 0.5 * smax * (1. - Rnum) + Rt * salp
    rhs_den = 0.5 * smax * (1. - Rnum) + Re * salp

    rhs = 1. * rhs_nom / rhs_den

    n = n0 + Na * rhs**b
    return n


def smax_stuessi_boerstra(n, R, m, Rt, Re, Na, alp_c, n0=1.0):

    def getsmax(n_):
        a_ = alp_c[0]
        b_ = alp_c[1]
        c_ = alp_c[2]

        alp = explogx(n_, a_, b_, c_)

        def fun(smax_):
            if R == -1.0:
                Rnum = R + 1E-12
            elif R == +1.0:
                Rnum = R - 1E-12
            else:
                Rnum = R

            lhs_b = ((n_ - n0) / Na)**(1. / b)

            #lhs = (n_ - n0) / Na

            salp = (abs(smax_ * (1. + Rnum)) / (2. * Rt))**alp - 1.

            rhs_nom = 0.5 * smax_ * (1. - Rnum) + Rt * salp
            rhs_den = 0.5 * smax_ * (1. - Rnum) + Re * salp

            rhs = -1. * rhs_nom / rhs_den

            # rhs_b = (-1. * (0.5 * smax_ * (1. - Rnum) + Rt * salp) /
            #       (0.5 * smax_ * (1. - Rnum) + Re * salp))**b
            return lhs_b - rhs  # lhs - rhs_b

        def fun_rhs_den(smax_):
            if R == -1.0:
                Rnum = R + 1E-12
            elif R == +1.0:
                Rnum = R - 1E-12
            else:
                Rnum = R

            salp = (abs(smax_ * (1. + Rnum)) / (2. * Rt))**alp - 1.

            rhs_den = 0.5 * smax_ * (1. - Rnum) + Re * salp

            return rhs_den

        def grad(smax_):
            '''
            -b ((-0.5^d d (R + 1)^2 s (1/t)^(d - 1) abs((R + 1) s)^(d - 2) - 0.5 (1 - R))/
            (0.5^d l (1/t)^d abs((R + 1) s)^d - l + 0.5 (1 - R) s) - ((-0.5^d (1/t)^(d - 1) abs((R + 1) s)^d - 0.5 (1 - R) s + t)
            (0.5^d d l (R + 1)^2 s (1/t)^d abs((R + 1) s)^(d - 2) + 0.5 (1 - R)))/(0.5^d l (1/t)^d abs((R + 1) s)^d - l + 0.5 (1 - R) s)^2)
             ((-0.5^d (1/t)^(d - 1) abs((R + 1) s)^d - 0.5 (1 - R) s + t)/(0.5^d l (1/t)^d abs((R + 1) s)^d - l + 0.5 (1 - R) s))^(b - 1)
            '''
            if R == -1.0:
                Rnum = R + 1E-12
            elif R == +1.0:
                Rnum = R - 1E-12
            else:
                Rnum = R

            Rtalp1 = (1. / Rt)**(alp - 1.)
            Rtalp = (1. / Rt)**alp
            Rsalp = abs((Rnum + 1.) * smax_)**alp
            Rsalp2 = abs((Rnum + 1.) * smax_)**(alp - 2.)

            term1 = (-0.5**alp * alp * (Rnum + 1.)**2 * smax_ * Rtalp1 * Rsalp2 - 0.5 * (1. - Rnum)) /\
                (0.5**alp * Re * Rtalp * Rsalp - Re + 0.5 * (1. - Rnum) * smax_) -\
                ((-0.5**alp * Rtalp1 * Rsalp - 0.5 * (1. - Rnum) * smax_ + Rt) *
                 (0.5**alp * alp * Re * (Rnum + 1.)**2 * smax_ * Rtalp * Rsalp2 + 0.5 * (1. - Rnum))) /\
                (0.5**alp * Re * Rtalp * Rsalp - Re + 0.5 * (1. - Rnum) * smax_)**2
            term2 = (-0.5**alp * Rtalp1 * Rsalp - 0.5 * (1. - Rnum) * smax_ + Rt) /\
                (0.5**alp * Re * Rtalp * Rsalp - Re + 0.5 * (1. - Rnum) * smax_)
            drhs_dsmax = -b * term1 * term2**(b - 1.)
            return drhs_dsmax
        '''
        import matplotlib.pyplot as plt
        #smax_ = np.linspace(smin_start, smax_start, 100)
        smax_ = np.linspace(0, smax_start, 100)
        fig, ax = plt.subplots()
        ax.plot(smax_, fun(smax_))
        ax.plot(smax_, fun_rhs_den(smax_))
        #plt.plot(smax_, grad(smax_))
        plt.ylim(-1., None)
        plt.xlim(15E6, 15.1E6)
        '''

        if n_ == 1 and R >= -1.:
            smax = Rt
        #smax_start = smax[ni] - smax_delta
        else:
            # try find asymptote (if it exists) and use as lower bound
            # asymptote is found when denominater = zero
            sasymp = bisection(fun_rhs_den, lower=0., upper=smax_start,
                               tol=1e-6, maxiter=100, callback=None)
            smax = bisection(fun, lower=sasymp + smax_delta, upper=smax_start,
                             tol=1e-6, maxiter=100, callback=None)
        # except:
        #    smax = sasymp
        # bnewton(fun, grad, lower=smin_start, upper=smax_start,
        #        tol=1e-6, maxiter=100, callback=None)
        #smax[ni] = optimize.newton(fun, smax_start, grad)
        #smax_start = smax[ni]
        return smax

    b = _b(m, Re, Rt)

    smax_start = Rt
    smin_start = Re
    smax_delta = 1E-6

    if isinstance(n, np.ndarray):
        smax = np.zeros_like(n)
        for ni, n_ in enumerate(n):
            smax[ni] = getsmax(n_)
    else:
        smax = getsmax(n)

    return smax


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


def smax_stuessi_boerstra_weibull(n, R, m, Rt_fit, Re_fit, alp_c, Na, p, alpha, beta, gamma, n0):
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
        smax_stuessi_boerstra(n, R, m, Rt_fit, Re_fit, Na, alp_c, n0)


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


def xrand_smax_stuessi_goodman_weibull(smax_i, N_i, R_i, m_fit, Rt_fit, M_fit,
                                       Re_fit, Na_fit, p, alpha, beta, gamma, n0=0.):
    ''' Random variable x using smax
    :return: xrand: float value
    '''
    return smax_i - smax_stuessi_goodman_weibull(N_i, R_i, m_fit, Rt_fit, M_fit,
                                                 Re_fit, Na_fit, p, alpha, beta, gamma, n0)


def m_RtRd(m_fit, Rt_fit, Re_fit, Rt_tar, Re_tar):
    ''' Transform m_fit when Stuessi curve is shifted in smax-direction
    Stuessi's b_fit = b_tar
    where b = (m * (-Re + Rt)) / (2 * (Re + Rt))
    and (-Re_fit + Rt_fit) = (-Re_tar + Rt_tar)
    '''
    return m_fit * (Re_tar + Rt_tar) / (Re_fit + Rt_fit)


def xrand_smax_stuessi_boerstra(smax_i, N_i, R_i, m_fit, Rt_fit, Re_fit, Na_fit, alp_c, n0=0.):
    ''' Random variable x using smax
    :return: xrand: float value
    '''
    return smax_i - smax_stuessi_boerstra(N_i, R_i, m_fit, Rt_fit, Re_fit, Na_fit, alp_c, n0)


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
                                Rt_start=40.,
                                include_weibull=False):
    '''
    Fits m and Rt to a set of experimental data
    :param: cyc_data: dict with experimental data
    :param: m_start: float start value
    :param: Rt_start: float start value
    :return: m_fit
    :return: Rt_fit
    '''

    initial_guess = [m_start, Rt_start]

    if include_weibull:
        global alp, bet, gam, xs

    def objective(params):
        m, Rt = params  # multiple design variables
        M = 1  # constants
        if include_weibull:
            global alp, bet, gam, xs
            xs = np.zeros(len(cyc_data['grps']))
            for i, (smax_actual, R, Nactual) in enumerate(
                zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                    cyc_data['cyc_ratios'][cyc_data['grps']],
                    cyc_data['cyc_cycles'][cyc_data['grps']])):

                xs[i] = xrand_smax_basquin_goodman(
                    smax_actual, Nactual, R, m, Rt, M)

            alp, bet, gam = pwm_weibull(xs)

        crits = []

        for i, (smax_actual, R, Nactual) in enumerate(zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                                                          cyc_data['cyc_ratios'][cyc_data['grps']],
                                                          cyc_data['cyc_cycles'][cyc_data['grps']])):

            p = 0.50
            if include_weibull:
                smax_50 = smax_basquin_goodman_weibull(
                    Nactual, R=R, m=m, Rt_fit=Rt, M=M,
                    p=p, alpha=alp, beta=bet, gamma=gam)
            else:
                smax_50 = smax_basquin_goodman(
                    Nactual, R=R, m=m, Rt=Rt, M=M)
            crit1 = (np.log10(smax_50) - np.log10(smax_actual))**2
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

    if not include_weibull:
        xs = np.zeros(len(cyc_data['grps']))
        for i, (smax_actual, R, Nactual) in enumerate(
                zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                    cyc_data['cyc_ratios'][cyc_data['grps']],
                    cyc_data['cyc_cycles'][cyc_data['grps']])):

            xs[i] = xrand_smax_basquin_goodman(
                smax_actual, Nactual, R, m_fit, Rt_fit, M_fit=1)

        alp, bet, gam = pwm_weibull(xs)

    return m_fit, Rt_fit, alp, bet, gam, xs


def fit_stuessi_weibull(cyc_data,
                        m_start=11.,
                        Rt_start=70.,
                        Re_start=40.,
                        Na_start=1E3,
                        n0=0.,
                        include_weibull=False):
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

    if include_weibull:
        global alp, bet, gam, xs_smax

    def objective(params):
        m, Rt, Re, Na = params  # multiple design variables
        M = 1  # constants
        if include_weibull:
            global alp, bet, gam, xs_smax
            xs_smax = np.zeros(len(cyc_data['grps']))
            for i, (smax_actual, R, Nactual) in enumerate(
                zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                    cyc_data['cyc_ratios'][cyc_data['grps']],
                    cyc_data['cyc_cycles'][cyc_data['grps']])):
                xs_smax[i] = xrand_sa_stuessi(
                    smax_actual, Nactual, R, m, Rt, M, Re, Na, n0)

            alp, bet, gam = pwm_weibull(xs_smax)

        crits = []
        for i, (smax_actual, R, Nactual) in enumerate(
            zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                cyc_data['cyc_ratios'][cyc_data['grps']],
                cyc_data['cyc_cycles'][cyc_data['grps']])):

            p = 0.50
            if include_weibull:
                smax_50 = sa_stuessi_weibull(
                    Nactual, R=R, m=m, Rt_fit=Rt, M=M,
                    Re_fit=Re, Na=Na,
                    p=p, alpha=alp, beta=bet, gamma=gam, n0=n0)
            else:
                smax_50 = sa_stuessi(
                    Nactual, R=R, m=m, Rt=Rt, M=M,
                    Re=Re, Na=Na, n0=n0)

            crit1 = (np.log10(smax_50) - np.log10(smax_actual))**2

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

    if not include_weibull:
        M = 1
        xs_smax = np.zeros(len(cyc_data['grps']))
        for i, (smax_actual, R, Nactual) in enumerate(
                zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                    cyc_data['cyc_ratios'][cyc_data['grps']],
                    cyc_data['cyc_cycles'][cyc_data['grps']])):
            xs_smax[i] = xrand_sa_stuessi(
                smax_actual, Nactual, R, m_fit, Rt_fit, M, Re_fit, Na_fit, n0)

        alp, bet, gam = pwm_weibull(xs_smax)

    return m_fit, Rt_fit, Re_fit, Na_fit, alp, bet, gam, xs_smax


def fit_stuessi_goodman_weibull(cyc_data,
                                m_start=11.,
                                Rt_start=70.,
                                Re_start=40.,
                                Na_start=1E3,
                                n0=0.,
                                include_weibull=False):
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

    if include_weibull:
        global alp, bet, gam, xs_smax

    def objective(params):
        m, Rt, Re, Na = params  # multiple design variables
        M = 1  # constants
        try:
            crits = []
            if include_weibull:
                global alp, bet, gam, xs_smax
                xs_smax = np.zeros(len(cyc_data['grps']))
                for i, (smax_actual, R, Nactual) in enumerate(
                    zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                        cyc_data['cyc_ratios'][cyc_data['grps']],
                        cyc_data['cyc_cycles'][cyc_data['grps']])):
                    xs_smax[i] = xrand_smax_stuessi_goodman(
                        smax_actual, Nactual, R, m, Rt, M, Re, Na, n0)

                alp, bet, gam = pwm_weibull(xs_smax)
                #mu, sig = s_weibull(alp, bet, gam)
                #crit0 = (sig)**2
                # crits.append(crit0)
            for i, (smax_actual, R, Nactual) in enumerate(
                zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                    cyc_data['cyc_ratios'][cyc_data['grps']],
                    cyc_data['cyc_cycles'][cyc_data['grps']])):

                p = 0.50
                if include_weibull:
                    smax_50 = smax_stuessi_goodman_weibull(
                        Nactual, R=R, m=m, Rt_fit=Rt, M=M,
                        Re_fit=Re, Na=Na,
                        p=p, alpha=alp, beta=bet, gamma=gam, n0=n0)
                else:
                    smax_50 = smax_stuessi_goodman(
                        Nactual, R=R, m=m, Rt=Rt, M=M,
                        Re=Re, Na=Na)

                crit1 = (np.log10(smax_50) - np.log10(smax_actual))**2

                crits.append(crit1)

            critsum = np.sum(np.array(crits))  # objective function
        except:
            critsum = np.inf
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

    if not include_weibull:
        M = 1
        xs_smax = np.zeros(len(cyc_data['grps']))
        for i, (smax_actual, R, Nactual) in enumerate(
                zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                    cyc_data['cyc_ratios'][cyc_data['grps']],
                    cyc_data['cyc_cycles'][cyc_data['grps']])):
            xs_smax[i] = xrand_smax_stuessi_goodman(
                smax_actual, Nactual, R, m_fit, Rt_fit, M, Re_fit, Na_fit, n0)

        alp, bet, gam = pwm_weibull(xs_smax)

    return m_fit, Rt_fit, Re_fit, Na_fit, alp, bet, gam, xs_smax


def fit_stuessi_boerstra_weibull(cyc_data,
                                 m_start=11.,
                                 Rt_start=70.,
                                 Re_start=40.,
                                 Na_start=1E3,
                                 alp_c=[1., 0., 0.],
                                 n0=0.,
                                 include_weibull=False):
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

    if include_weibull:
        global alp, bet, gam, xs_smax

    def objective(params):
        m, Rt, Re, Na = params  # multiple design variables
        try:
            crits = []
            if include_weibull:
                global alp, bet, gam, xs_smax
                xs_smax = np.zeros(len(cyc_data['grps']))
                for i, (smax_actual, R, Nactual) in enumerate(
                    zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                        cyc_data['cyc_ratios'][cyc_data['grps']],
                        cyc_data['cyc_cycles'][cyc_data['grps']])):
                    xs_smax[i] = xrand_smax_stuessi_boerstra(
                        smax_actual, Nactual, R, m_fit, Rt_fit, Re_fit, Na_fit, alp_c, n0=0.)

                alp, bet, gam = pwm_weibull(xs_smax)
                #mu, sig = s_weibull(alp, bet, gam)
                #crit0 = (sig)**2
                # crits.append(crit0)
            for i, (smax_actual, R, Nactual) in enumerate(
                zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                    cyc_data['cyc_ratios'][cyc_data['grps']],
                    cyc_data['cyc_cycles'][cyc_data['grps']])):

                p = 0.50
                if include_weibull:
                    smax_50 = smax_stuessi_boerstra_weibull(
                        Nactual, R=R, m=m, Rt_fit=Rt, Re_fit=Re, alp_c=alp_c, Na=Na,
                        p=p, alpha=alp, beta=bet, gamma=gam, n0=n0)
                else:
                    smax_50 = smax_stuessi_boerstra(
                        Nactual, R=R, m=m, Rt=Rt, Re=Re, Na=Na, alp_c=alp_c, n0=n0)

                crit1 = (np.log10(smax_50) - np.log10(smax_actual))**2

                crits.append(crit1)

            critsum = np.sum(np.array(crits))  # objective function
        except:
            critsum = np.inf
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

    if not include_weibull:
        xs_smax = np.zeros(len(cyc_data['grps']))
        for i, (smax_actual, R, Nactual) in enumerate(
                zip(cyc_data['cyc_stress_max'][cyc_data['grps']],
                    cyc_data['cyc_ratios'][cyc_data['grps']],
                    cyc_data['cyc_cycles'][cyc_data['grps']])):
            xs_smax[i] = xrand_smax_stuessi_boerstra(
                smax_actual, Nactual, R, m_fit, Rt_fit, Re_fit, Na_fit, alp_c, n0=0.)

        alp, bet, gam = pwm_weibull(xs_smax)

    return m_fit, Rt_fit, Re_fit, Na_fit, alp, bet, gam, xs_smax


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
        elif self.fit_type == 'stuessi-boerstra':
            self._init_stuessi_boerstra()

    def _init_basquin_goodman(self):
        self.m_start = 10.0
        self.Rt_start = 50.0E+6

    def _init_stuessi_goodman(self):
        self._init_basquin_goodman()
        self.Re_start = 9.E+6
        self.Na_start = 68.
        self.n0 = 1.

    def _init_stuessi_boerstra(self):
        self._init_stuessi_goodman()
        self.alp_c = [1., 0., 0.]

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

    def fit_data(self, include_weibull):
        '''
        :return: sn_fit: dict with fitting parameters
        '''

        self.sn_fit = OrderedDict()

        if self.fit_type == 'basquin-goodman':
            m_fit, Rt_fit, alpha_, beta_, gamma_, xs =\
                fit_basquin_goodman_weibull(self.cyc_data,
                                            self.m_start,
                                            self.Rt_start,
                                            include_weibull)

            Rt_50 = smax_limit_basquin_goodman_weibull(p=0.5,
                                                       Rt_fit=Rt_fit,
                                                       alpha=alpha_,
                                                       beta=beta_,
                                                       gamma=gamma_)
            alpha, beta, gamma = alpha_, beta_, gamma_

        elif self.fit_type == 'stuessi-goodman':

            m_fit, Rt_fit, Re_fit, Na, alpha_, beta_, gamma_, xs =\
                fit_stuessi_goodman_weibull(self.cyc_data,
                                            self.m_start,
                                            self.Rt_start,
                                            self.Re_start,
                                            self.Na_start,
                                            self.n0,
                                            include_weibull)

            Rt_50, Re_50 = smax_limit_stuessi_goodman_weibull(p=0.5,
                                                              R=-1,
                                                              Rt_fit=Rt_fit,
                                                              M=1,
                                                              Re_fit=Re_fit,
                                                              alpha=alpha_,
                                                              beta=beta_,
                                                              gamma=gamma_)

            _, Re_50_R = smax_limit_stuessi_goodman_weibull(p=0.5,
                                                            R=self.cyc_data['cyc_ratio_grp'],
                                                            Rt_fit=Rt_fit,
                                                            M=1,
                                                            Re_fit=Re_fit,
                                                            alpha=alpha_,
                                                            beta=beta_,
                                                            gamma=gamma_)

            m_50 = m_RtRd(m_fit, Rt_fit, Re_fit, Rt_50, Re_50)

            alpha, beta, gamma = alpha_, beta_, gamma_

            self.sn_fit['m_50'] = m_50
            self.sn_fit['Re_fit'] = Re_fit
            self.sn_fit['Na'] = Na
            self.sn_fit['Re_50'] = Re_50
            self.sn_fit['Re_50_R'] = Re_50_R.tolist()

        elif self.fit_type == 'stuessi-boerstra':

            m_fit, Rt_fit, Re_fit, Na, alpha_, beta_, gamma_, xs =\
                fit_stuessi_boerstra_weibull(self.cyc_data,
                                             self.m_start,
                                             self.Rt_start,
                                             self.Re_start,
                                             self.Na_start,
                                             self.alp_c,
                                             self.n0,
                                             include_weibull)

            Rt_50, Re_50 = smax_limit_stuessi_goodman_weibull(p=0.5,
                                                              R=-1,
                                                              Rt_fit=Rt_fit,
                                                              M=1,
                                                              Re_fit=Re_fit,
                                                              alpha=alpha_,
                                                              beta=beta_,
                                                              gamma=gamma_)

            _, Re_50_R = smax_limit_stuessi_goodman_weibull(p=0.5,
                                                            R=self.cyc_data['cyc_ratio_grp'],
                                                            Rt_fit=Rt_fit,
                                                            M=1,
                                                            Re_fit=Re_fit,
                                                            alpha=alpha_,
                                                            beta=beta_,
                                                            gamma=gamma_)

            m_50 = m_RtRd(m_fit, Rt_fit, Re_fit, Rt_50, Re_50)

            alpha, beta, gamma = alpha_, beta_, gamma_

            self.sn_fit['m_50'] = m_50
            self.sn_fit['Re_fit'] = Re_fit
            self.sn_fit['Na'] = Na
            self.sn_fit['Re_50'] = Re_50
            self.sn_fit['Re_50_R'] = Re_50_R.tolist()

        elif self.fit_type == 'stuessi':
            m_fit, Rt_fit, Re_fit, Na, alpha, beta, gamma, xs =\
                fit_stuessi_weibull(self.cyc_data,
                                    self.m_start,
                                    self.Rt_start,
                                    self.Re_start,
                                    self.Na_start,
                                    self.n0,
                                    include_weibull)

            Rt_50 = 0.
            Re_50 = 0.
            Re_50_R = np.array([0., 0.])
            self.sn_fit['Re_fit'] = Re_fit
            self.sn_fit['Na'] = Na
            self.sn_fit['Re_50'] = Re_50
            self.sn_fit['Re_50_R'] = Re_50_R.tolist()

        self.sn_fit['xs'] = xs.tolist()
        self.sn_fit['m_fit'] = m_fit
        self.sn_fit['Rt_fit'] = Rt_fit
        self.sn_fit['alpha'] = alpha
        self.sn_fit['beta'] = beta
        self.sn_fit['gamma'] = gamma
        self.sn_fit['Rt_50'] = Rt_50

    def cld(self, ns):
        '''
        Obtain CLD from SNfit
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

        grps = self.cyc_data['grplist']
        cyc_stress_max = self.cyc_data['cyc_stress_max']
        cyc_cycles = self.cyc_data['cyc_cycles']
        cyc_ratio_grp = self.cyc_data['cyc_ratio_grp']
        m_fit = self.sn_fit['m_fit']
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
                if self.fit_type == 'stuessi-goodman':
                    smax_50 = smax_stuessi_goodman_weibull(
                        n_, R=R_, m=m_fit, Rt_fit=Rt_fit, M=M_fit,
                        Re_fit=Re_fit, Na=Na_fit,
                        p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)
                elif self.fit_type == 'stuessi-boerstra':
                    smax_50 = smax_stuessi_boerstra_weibull(
                        n_, R=R_, m=m_fit, Rt_fit=Rt_fit,
                        Re_fit=Re_fit, Na=Na_fit, alp_c=self.alp_c,
                        p=p, alpha=alp_smax_fit, beta=bet_smax_fit, gamma=gam_smax_fit, n0=n0)
                self.sas[Ri, ni] = sa_50 = sa_smax(smax_50, R_)
                if R_ == 1:
                    sm_50 = smax_50
                else:
                    sm_50 = sm(sa_50, R_)
                self.sms[Ri, ni] = sm_50

    def shift_sn(self, dRt, p):
        '''

        '''


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
            self.fit_data(include_weibull=False)  # works only w/o

            self.sn_grp[i] = OrderedDict()
            self.sn_grp[i]['sn_fit'] = self.sn_fit
            self.sn_grp[i]['cyc_data'] = self.cyc_data

            grps = self.cyc_data['grplist']
            cyc_stress_max = self.cyc_data['cyc_stress_max']
            cyc_cycles = self.cyc_data['cyc_cycles']
            cyc_ratio_grp = self.cyc_data['cyc_ratio_grp']
            m_fit = self.sn_fit['m_fit']
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

    def fit_alp(self, fit='linear', idxs=[]):

        sm0 = self.sms[0, 0]
        sa0 = Rt = self.sas[-1, 0]

        ns = self.ns
        npoint = len(ns)

        nsm = 1000
        sm = np.linspace(0., sm0, nsm)

        # fit boerstra CLs and find alps
        sas_b = np.zeros((nsm, npoint))

        self.alp = np.zeros(npoint)
        for ni in range(npoint):
            xdata = self.sms[:, ni]
            ydata = self.sas[:, ni]
            sa0_ = self.sas[-1, ni]

            def func(x, alp):
                return sa_boerstra(sa0_, x, Rt, alp)

            popt, pcov = optimize.curve_fit(func, xdata, ydata)
            self.alp[ni] = popt[0]

            sas_b[:, ni] = sa_boerstra(sa0_, sm, Rt, self.alp[ni])
        '''
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        for i, n in enumerate(ns):
            col = next(ax._get_lines.prop_cycler)['color']
            ax.plot(sm, sas_b[:, i], '-', color=col)
            ax.plot(self.sms[:, i], self.sas[:, i],
                'o--', color=col, label=r'$N=\num{%0.0E}$' % (n))

        '''
        # fit alp over n
        if not idxs:
            idxs = range(len(ns))

        xdata = ns[idxs]
        ydata = self.alp[idxs]
        popt = np.polyfit(xdata, ydata, 1)
        if fit == 'linear':
            popt, pcov = optimize.curve_fit(poly1dlogx, xdata, ydata)
        elif fit == 'exp':
            popt, pcov = optimize.curve_fit(explogx, xdata, ydata)
        self.alp_c = alp_opt = popt
        '''
        import matplotlib.pyplot as plt
        exp_start = 0  # 10^0
        exp_end = 7  # 10^7
        nsf = np.logspace(exp_start, exp_end, 1000)
        fig, ax = plt.subplots()
        ax.semilogx(ns, self.alp, 'o', label=r'Boerstra exponent $\alpha$')
        ax.semilogx(nsf, explogx(nsf, *alp_opt), '-',
                    label=r'Fit $%0.2fx^{-%0.2f}+%0.2f$' % (alp_opt[0], alp_opt[1], alp_opt[2]))
        ax.semilogx(nsf, poly1dlogx(nsf,*alp_opt), '-.')

        '''
