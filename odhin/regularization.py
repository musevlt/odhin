# -*- coding: utf-8 -*-

"""
@author: raphael.bacher@gipsa-lab.fr
"""

import itertools
import numpy as np

from .lines_estimation import getLinesSupportList


def glasso_bic(X, Y, ng=2, multivar=True, listMask=None, returnCriterion=False,
               greedy=True, averaged=True):
    """
    If given listMask, on each mask do a BIC selection (cf lasso_bic)

    Parameters
    ----------
    X : regressors (intensityMaps, n x k)
    Y : data (n x lmbda)
    ng : size of spectral blocks
    multivar : if True, in BIC selection consider different variance for each
    wavelength (useless if averaged is True)
    listMask : list of lines mask. The regularization is done independently
    on each of these masks
    returnCriterion : return values of BIC criterion
    greedy : if True use greedy approximation of BIC
    averaged : if True do the BIC selection on averaged data (before doing
    the regression on original data)

    Output:
    ------
    coeff : estimated coefficients (spectra k x lmbda)
    intercepts : background (1 x lmbda)
    (criterion : list of BIC values)

    """
    coeff = np.zeros((X.shape[1], Y.shape[1]))
    criterion = []
    intercepts = np.zeros((1, Y.shape[1]))

    if listMask is None:
        for k in range(Y.shape[1]):
            res = lasso_bic(X, Y[:, np.maximum(0, k - ng):k + ng + 1],
                            multivar=multivar, averaged=averaged,
                            greedy=greedy)
            coeff[:, k] = res[0][:, np.minimum(k, ng)]
            intercepts[:, k] = res[1][np.minimum(k, ng)]
            criterion.append(res[2])
    else:
        for mask in listMask:
            res = lasso_bic(X, Y[:, mask], multivar=multivar,
                            averaged=averaged, greedy=greedy)
            coeff[:, mask] = res[0]
            intercepts[:, mask] = res[1]
            criterion.append(res[2])

    if returnCriterion:
        return coeff, intercepts, criterion
    else:
        return coeff, intercepts


def lasso_bic(X, Y, multivar=True, greedy=False, averaged=True,
              returnAll=False):
    r"""
    Estimate spectra from X, Y using BIC.

    BIC  is defined as  K\log(n) -2\log(\hat{L}) with K the number of free
    parameters, n the number of samples and L the likelihood.

    Here BIC = (k+1)\log(n) + \log(\widehat{\sigma}^2) where sigma^2 is the
    variance of the residuals.

    So for each possible model (=combination of a selection of
    spectra/objects/regressors) we compute the regression (least square
    inversion), the number of free paramaters and the residuals. From that we
    get the BIC value associated with this

    Parameters
    ----------
    X : regressors (intensityMaps, n x k)
    Y : data (n x lmbda)
    multivar : if True, in BIC selection consider different variance for
    each wavelength (useless if averaged is True)
    greedy : if True use greedy approximation of BIC
    averaged : if True do the BIC selection on averaged data (before doing
    the regression on original data)

    Output:
    ------
    coeff : estimated coefficients (=spectra k x lmbda)
    intercepts : background (1 x lmbda)

    """
    if averaged:  # work on averaged data for model selection
        Y_all = Y.copy()
        Y = np.mean(Y, axis=1)[:, None]

    n_models = X.shape[1]
    if returnAll:
        coef_path_all = []

    if not greedy:
        # compute all possible combinations of non-nul objects
        listComb = []
        for k in range(1, n_models + 1):
            listComb += [i for i in itertools.combinations(
                np.arange(n_models), k)]
    else:
        # add iteratively the regressor the most strongly correlated to the
        # data in the remaining regressors
        listComb = [[]]
        listModels = list(range(n_models))
        lprod = np.dot(X.T, Y).mean(axis=1)
        a = np.argmax(np.abs(lprod))
        residuals = Y - (lprod[a] / np.linalg.norm(X[:, a]) * X[:, a])[:, None]
        listModels.pop(a)
        listComb[0] = [a]
        for k in range(1, n_models):
            lprod = [np.mean(np.dot(X[:, i], residuals)) for i in listModels]
            a = np.argmax(np.abs(lprod))
            a_m = listModels[a]
            residuals = Y - \
                (lprod[a] / np.linalg.norm(X[:, a_m]) * X[:, a_m])[:, None]
            listModels.pop(a)
            listComb.append(listComb[k - 1] + [a_m])

    # center data
    X_offset = np.mean(X, axis=0)
    Y_offset = np.mean(Y, axis=0)
    X = X - X_offset
    Y = Y - Y_offset

    # compute the coeffs (estimated spectra) for each possible model.
    coef_path_ = []
    for ind in listComb:
        coef_path_.append(np.linalg.lstsq(X[:, ind], Y, rcond=None)[0])
        if returnAll:
            coef_path_all.append(np.linalg.lstsq(X[:, ind], Y_all,
                                                 rcond=None)[0])

    n_samples = X.shape[0]
    n_targets = Y.shape[1]
    K = np.log(n_samples * n_targets)  # BIC factor

    # compute mean squared errors
    mean_squared_error = []
    for k in range(len(coef_path_)):
        R = Y - np.dot(X[:, listComb[k]], coef_path_[k])  # residuals
        mean_squared_error.append(np.mean(R ** 2, axis=0))
    mean_squared_error = np.array(mean_squared_error)

    # get the number of degrees of freedom
    df = np.zeros(len(coef_path_), dtype=np.int)  # Degrees of freedom
    for k, coef in enumerate(coef_path_):
        df[k] = coef.size
        if multivar:  # add one for each wavelength variance
            df[k] = df[k] + n_targets
        else:  # add one for the global variance
            df[k] = df[k] + 1

        # add one value per wavelength for intercept
        df[k] = df[k] + n_targets

    if multivar:
        criterion_ = n_samples * \
            np.sum(np.log(mean_squared_error), axis=1) + K * df
    else:
        criterion_ = n_samples * n_targets * \
            np.log(np.mean(mean_squared_error, axis=1)) + K * df

    n_best = np.argmin(criterion_)

    # compute
    if multivar:
        r0 = n_samples * np.sum(np.log(np.mean(Y**2, axis=0))) + K * n_targets
    else:
        r0 = n_samples * n_targets * np.log(np.mean(Y**2)) + K
    # add df for intercepts
    r0 = r0 + n_targets * K

    if returnAll:
        likelihood = np.concatenate([
            np.array([n_samples * np.sum(np.log(np.mean(Y**2, axis=0)))]),
            n_samples * np.sum(np.log(mean_squared_error), axis=1)])
        penalty = np.concatenate([
            np.array([r0 - n_samples * np.sum(np.log(np.mean(Y**2, axis=0)))]),
            K * df])

    if averaged:  # we now get back to the whole dataset
        Y = Y_all
        n_targets = Y.shape[1]
        # centering
        Y_offset = np.average(Y, axis=0)
        Y = Y - Y_offset

    coeff = np.zeros((n_models, n_targets))

    if criterion_[n_best] < r0:  # if not, all regressors stay at 0
        if averaged:
            coeff[listComb[n_best], :] = np.linalg.lstsq(
                X[:, listComb[n_best]], Y, rcond=None)[0]
        else:
            coeff[listComb[n_best], :] = coef_path_[n_best]

    if returnAll:
        return coef_path_all, likelihood, penalty
    else:
        intercepts = Y_offset - np.dot(X_offset, coeff)
        return coeff, intercepts, np.concatenate([np.array([r0]), criterion_])


def gridge_cv(X, Y, ng=1, alphas=np.logspace(-5, 2, 50), sig2=None,
              support=None):
    """
    Estimate coefficients using ridge regression and various methods for
    regularization parameter estimation.

    Use the 'one sigma' rule to increase regularization efficiency
    Use gridge_gcv_spectral method for the estimation of regularization
    parameter

    Parameters
    ----------
    X : regressors (intensityMaps, n x k)
    Y : data (n x lmbda)
    ng : size of spectral blocks
    alphas : list of regul parameters to test
    sig2 : variance of each wavelength slice
    support : mask of samples (pixels) with enough signal, where the cross
    validation will be applied

    Output:
    ------
    coeff : estimated coefficients (spectra k x lmbda)
    intercepts : background (1 x lmbda)

    """
    import sklearn.linear_model as sklm
    coeff = np.zeros((X.shape[1], Y.shape[1]))
    intercepts = np.zeros((1, Y.shape[1]))
    # FIXME: not used, remove ?
    # RCV_slid = sklm.RidgeCV(alphas=alphas, fit_intercept=True,
    #                         normalize=True, store_cv_values=True)
    listAlpha = np.zeros((Y.shape[1]))
    listRSS = []

    X_centr = X - np.mean(X, axis=0)
    Y_centr = Y - np.mean(Y, axis=0)

    X_centr /= np.linalg.norm(X_centr, axis=0)

    for k in range(int(np.ceil(Y.shape[1] / ng))):
        # prefered method : gcv_spe
        alpha, rss = gridge_gcv_spectral(
            X_centr, Y_centr[:, k * ng:(k + 1) * ng], alphas=alphas,
            Sig2=sig2[k * ng:(k + 1) * ng], support=support)
        listAlpha[k * ng:(k + 1) * ng] = alpha
        listRSS.append(rss.mean(axis=0).mean(axis=0))
        Ridge = sklm.Ridge(alpha=alpha, fit_intercept=True, normalize=True)
        Ridge.fit(X, Y[:, k * ng:(k + 1) * ng])
        coeff[:, k * ng:(k + 1) * ng] = Ridge.coef_.T
        intercepts[:, k * ng:(k + 1) * ng] = Ridge.intercept_.T

    return coeff, intercepts, listAlpha, listRSS


def gridge_gcv_spectral(X, Y, support, alphas=np.logspace(-5, 2, 50),
                        Sig2=None, maxAlphaFrac=2.):
    """
    Estimate coefficients using ridge regression and various methods for
    regularization parameter estimation

    Parameters
    ----------
    X : 2d array (n pixels  x  k objects)
        regressors (intensityMaps)
    Y : 2d array (n pixels  x lmbda wavelengths)
        data
    support : 1d array of bool
        mask of samples (pixels) with enough signal, where the cross
        validation will be applied
    alphas : list of float
        list of regul parameters to test
    Sig2 :
        variance of each wavelength slice (1d array n_targets)
    maxAlphaFrac : float
        fraction of the max singular value of X that will be the
        upper limit for regularization parameter

    Output:
    ------
    alpha : estimated regularization parameter
    rss : errors of prediction (ndarray n_targets,n_alphas)

    """
    Ys = Y[support]
    Xs = X[support]
    if Sig2 is None:
        Sig2 = np.ones(Y.shape[1])

    # FIXME: not used, remove ?
    sumSig2 = Sig2[:-2] + Sig2[2:]
    U, sval, V = np.linalg.svd(Xs, full_matrices=False)
    UtY = np.dot(U.T, Ys)
    UtY2 = UtY**2

    rss = np.zeros((Y.shape[1], len(alphas)))
    for a, alpha in enumerate(alphas):
        # GCV approx S_ii =Tr(S)/n
        TrS = np.sum(sval**2 / (sval**2 + alpha))
        residuals = np.sum(((alpha / (sval**2 + alpha))**2)
                           [:, None] * UtY2, axis=0)

        rss[:, a] = residuals / (1 - TrS / Xs.shape[0])**2

    alpha = alphas[np.argmin(np.average(rss, axis=0, weights=1 / Sig2))]
    TrS = np.sum(sval**2 / (sval**2 + alpha))
    Xbeta = np.dot(U, _diag_dot(sval**2 / (sval**2 + alpha), UtY))
    rss_alpha = ((Ys - Xbeta) / (1 - TrS / Xs.shape[0]))**2
    cv = rss_alpha.shape[0] * rss_alpha.shape[1]
    min_mse_std = np.std(np.std(rss_alpha, axis=0), axis=0) / np.sqrt(cv)
    min_mse = np.mean(np.mean(rss_alpha, axis=0), axis=0)
    alpha = np.max([alphas[i] for i in range(len(alphas))
                    if np.mean(rss[:, i]) < min_mse + min_mse_std])
    if alpha > np.max(sval) * maxAlphaFrac:
        alpha = np.max(sval) * maxAlphaFrac

    return alpha, rss


def _diag_dot(D, B):
    """compute dot(diag(D), B)"""
    if len(B.shape) > 1:
        # handle case where B is > 1-d
        D = D[(slice(None), ) + (np.newaxis, ) * (len(B.shape) - 1)]
    return D * B


def regulDeblendFunc(X, Y, Y_c=None, ng=200, n_alphas=100, eps=1e-3,
                     alpha_c=0.0001, support=None, trueLines=None,
                     alphas=np.logspace(-5, 2, 50), filt_w=101, Y_sig2=None):
    """
    Estimate coefficients using ridge regression and various methods for
    regularization parameter estimation

    Parameters
    ----------
    X : 2d array (n pixels  x  k objects)
        regressors (intensityMaps)
    Y : 2d array (n pixels  x lmbda wavelengths)
        data
    Y_c : 2d array (n pixels  x lmbda wavelengths)
        data continuum (if pre-estimated)
    l_method : str
        method to use for emission lines estimation
    ng : int
        size of spectral bin for regularization

    Output:
    ------
    res: 2d array (k objects lambda wavelengths)
        estimation of objects spectra
    intercepts:
        estimation backroung spectrum
    listMask :
        list of masks of detected lines
    c_coeff: 2d array (k objects x lambda wavelengths)
        estimation of continuum of objects spectra
    l_coeff: 2d array (k objects x lambda wavelengths)
        estimation of lines of objects spectra
    Y: 2d array (n pixels  x lmbda wavelengths)
        data
    Y_l: 2d array (n pixels  x lmbda wavelengths)
        data lines
    Y_c: 2d array (n pixels  x lmbda wavelengths)
        data remaining continuum after lines estimation
    c_alphas:
        array of regularization parameters
    listRSS:
        array of prediction errors
    listA : 2d array (k objects x lambda wavelengths)
         array of flux correction factors
    """
    # get emission lines only (Y_l)
    Y_l = Y - Y_c

    if trueLines is not None:  # use prior knowlegde of lines support if given
        listMask = trueLines
    else:  # find lines support
        # First we compute one spectrum per object (rough estimation by summing
        # over each intensity map)
        listSpe = np.dot(X.T, Y_l)

        # Then we seek all spectral lines on this bunch of spectra
        listMask = getLinesSupportList(listSpe, w=2, wmax=20, wmin=2,
                                       alpha=2.5, n_sig=1.4, filt=None)

    # preferred approach : group bic approach
    l_coeff, l_intercepts = glasso_bic(X, Y_l, ng=ng, listMask=listMask)

    # remove estimated contribution from emission lines
    Y_c = Y - np.dot(X, l_coeff) - l_intercepts

    # we now work on remaining data Y_c
    # FIXME: not used, remove ?
    # X1 = X[support, :]
    # Y_c1 = Y_c[support, :]

    # preferred method : sliding Ridge GCV
    c_coeff, c_intercepts, c_alphas, listRSS = gridge_cv(
        X, Y_c, ng=ng, support=support, sig2=Y_sig2, alphas=alphas)

    # correct flux
    # add one row of ones for background/intercept to keep corresponding
    # positions in result arrays
    listA = np.ones((X.shape[1] + 1, Y.shape[1]))

    c_coeff, la = corrFlux(X, Y_c, c_coeff)
    listA[1:, :] = la[:, None]

    # combine coeffs from lines and continuum
    res = c_coeff + l_coeff
    intercepts = c_intercepts + l_intercepts

    return (res, intercepts, listMask, c_coeff, l_coeff, Y, Y_l, Y_c,
            c_alphas, listRSS, listA)


def corrFlux(X, Y, beta):
    """Correct coefficients to limit flux loss.

    We seek a diagonal matrix A to minimize ||Y-X*A*beta||^2 with A_ii>=1
    (as we know there has been some flux loss).

    Parameters
    ----------
    X : 2d array (n pixels  x  k objects)
        regressors (intensityMaps)
    Y : 2d array (n pixels  x lambda wavelengths)
        data
    beta : 2d array (k objects x lambda wavelengths)
        spectra

    Output
    ------
    beta_c : 2d array (k objects x lambda wavelengths)
        corrected spectra
    listA : 1d array (k objects)
        correction factors

    """
    niter = 5
    beta_c = beta.copy()
    listA = np.ones(X.shape[1])
    listI = np.zeros(X.shape[1], dtype=bool)
    k = 0
    while (False in listI) and k < niter:  # iter until no coeff is under one.
        if k == 0:
            listI[:] = True
        Y_t = np.dot(Y, np.linalg.pinv(beta[listI, :]))
        for j, i in enumerate(np.arange(X.shape[1])[listI]):
            a = np.dot(Y_t[:, j], X[:, i]) / np.linalg.norm(X[:, i])**2
            if a < 1:
                listI[i] = False
                listA[i] = 1
            else:
                listA[i] = a
        k = k + 1

    beta_c *= listA[:, None]
    return beta_c, listA
