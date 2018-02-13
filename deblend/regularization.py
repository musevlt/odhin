# -*- coding: utf-8 -*-

"""
@author: raphael.bacher@gipsa-lab.fr
"""

import numpy as np

import scipy.signal as ssl
import scipy.stats as sst
import itertools
import sklearn.linear_model as sklm
from scipy.ndimage.morphology import binary_dilation, grey_dilation


def glasso_bic(X, Y, ng=2, multivar=True, listMask=None,
               returnCriterion=False, greedy=True,
               averaged=True):
    """
    If given listMask, on each mask do a BIC selection (cf lasso_bic)

    Parameters
    ----------
        X : regressors (intensityMaps, n x k)
        Y : data (n x lmbda)
        ng : size of spectral blocks
        multivar : if True, in BIC selection consider different variance for each wavelength (useless if averaged is True)
        listMask : list of lines mask. The regularization is done independently on each of these masks
        returnCriterion : return values of BIC criterion
        greedy : if True use greedy approximation of BIC
        averaged : if True do the BIC selection on averaged data (before doing the regression on original data)


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
            res = lasso_bic(X,
                            Y[:,
                              np.maximum(0,
                                         k - ng):k + ng + 1],
                            multivar=multivar,
                            averaged=averaged,
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
    return coeff, intercepts


def lasso_bic(
        X,
        Y,
        multivar=True,
        greedy=False,
        averaged=True,
        returnAll=False):
    """
    Estimate spectra from X, Y using BIC.

    BIC  is defined as  K\log(n) -2\log(\hat{L}) with K the number of free parameters,
    n the number of samples and L the likelihood.

    Here BIC = (k+1)\log(n) + \log(\widehat{\sigma}^2) where sigma^2 is the variance of
    the residuals.

    So for each possible model (=combination of a selection of spectra/objects/regressors)
    we compute the regression (least square inversion), the number of free paramaters
    and the residuals. From that we get the BIC value associated with this

    Parameters
    ----------
        X : regressors (intensityMaps, n x k)
        Y : data (n x lmbda)
        multivar : if True, in BIC selection consider different variance for each wavelength (useless if averaged is True)
        greedy : if True use greedy approximation of BIC
        averaged : if True do the BIC selection on averaged data (before doing the regression on original data)



    Output:
    ------
        coeff : estimated coefficients (=spectra k x lmbda)
        intercepts : background (1 x lmbda)

    """

    if averaged:  # work on averaged data for model selection
        Y_all = Y.copy()
        Y = np.mean(Y, axis=1)[:, None]

    n_samples = X.shape[0]
    n_models = X.shape[1]
    n_targets = Y.shape[1]
    coef_path_ = []
    if returnAll:
        coef_path_all = []
    listComb = []
    if greedy == False:  # compute all possible combinations of non-nul objects
        for k in range(1, n_models + 1):
            listComb += [
                i for i in itertools.combinations(
                    np.arange(n_models), k)]
    else:  # add iteratively the regressor the most strongly correlated to the data in the remaining regressors
        listComb = [[]]
        listModels = list(range(n_models))
        lprod = [np.mean(np.dot(X[:, i], Y)) for i in listModels]
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
    X_offset = np.average(X, axis=0)
    Y_offset = np.average(Y, axis=0)
    X = X - X_offset
    Y = Y - Y_offset

    # compute the coeffs (estimated spectra) for each possible model.
    for ind in listComb:
        coef_path_.append(np.linalg.lstsq(X[:, ind], Y, rcond=None)[0])
        if returnAll:
            coef_path_all.append(np.linalg.lstsq(
                X[:, ind], Y_all, rcond=None)[0])

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
        likelihood = np.concatenate([np.array([n_samples *
                                               np.sum(np.log(np.mean(Y**2, axis=0)))]), n_samples *
                                     np.sum(np.log(mean_squared_error), axis=1)])
        penalty = np.concatenate(
            [np.array([r0 - n_samples * np.sum(np.log(np.mean(Y**2, axis=0)))]), K * df])

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

    intercepts = Y_offset - np.dot(X_offset, coeff)

    if returnAll:
        return coef_path_all, likelihood, penalty
    return coeff, intercepts, np.concatenate([np.array([r0]), criterion_])


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed(
    )  # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def getLinesSupportList(
        listSpe,
        w=2,
        wmin=1,
        wmax=20,
        alpha=1.4,
        beta=1.2,
        n_sig=1.2,
        f=0.6,
        returnAll=False,
        filt=None,
        localConstraint=True):
    """
    Get emission/absorption lines spectral support

    Parameters
    ----------
    listSpe : list of 1d arrays (spectra of size lambda)
        list of spectra where to seek lines
    w: int
        minimal width for local extrema research
    wmin : int
        minimal half-width
    wmax : int
        maximal half-width of line
    alpha : float
        threshold for matched filter data: keep only extrema that are above
        alpha*sig_filt (standard deviation of matched filtered spectrum)
    beta : float
        Stop line spectral support where signal is lower than beta*sig.
    n_sig: float
        A half-width corresponds to n_sig standard deviations of the kernel.

    f: float
        to avoid too many overlapping masks : if already covered at more than a fraction f by an existing
        mask don't add current mask
    returnAll : bool
        return all intermediate results
    filt: 1d-array (None by default)
        pattern for matching filter (gaussian shape will be made by default)
    localConstraint : bool
        if True, reject peaks where immediate neighbors of extrema are not > 1 std
        (maxima) or < -1 std (minima).


    Output:
    ------
        listMask : list of masks (each mask is a boolean array of size lambda)



    """
    if filt is None:
        filt = sst.norm.pdf(np.linspace(-2 * w, 2 * w, 4 * w + 1), scale=w)
        filt = filt / np.linalg.norm(filt)
    listMask = []
    for l in range(len(listSpe)):
        spe = listSpe[l]
        sig = 1.489 * mad(spe)  # compute standard deviation estimator from MAD

        spe_filt = ssl.fftconvolve(
            spe, filt, mode='same')  # matched filter using filt
        # compute standard deviation estimator of filtered data from MAD
        sig_filt = 1.489 * mad(spe_filt)
        lRejected = 0

        # find local extrema
        mask0 = np.ones(2 * w + 1)
        mask0[w] = 0
        B = grey_dilation(np.abs(spe_filt), footprint=mask0)
        listArgExtrema = np.nonzero(np.abs(spe_filt) > B)[0]
        listExtrema = spe_filt[listArgExtrema]

        # generate kernels for estimation of line width
        listKernel = genKernels(listWidth=np.concatenate(
            [np.array([0.1]), np.arange(1, 2 * wmax + 2, 2)]), n=2 * wmax + 1, n_sig=n_sig)

        # compute number of kept extrema after thresholding
        nThresh = np.sum(np.abs(spe_filt[listArgExtrema]) > alpha * sig_filt)

        for k, m in zip(listArgExtrema, listExtrema):
            if (np.abs(spe_filt[k]) > alpha * sig_filt) and ((localConstraint == False)
                                                             or (spe[np.maximum(k - 1, 0):k + 2] > np.sign(spe[k]) * sig).all()):
                mask = np.zeros_like(spe).astype(bool)
                kmin = np.maximum(k - wmax, 0)

                # create sub spectrum of good size for width estimation
                if k - wmax < 0:
                    line = np.concatenate(
                        [np.zeros(wmax - k), spe[kmin:k + wmax + 1]])
                elif wmax + k + 1 > len(spe):
                    line = np.concatenate(
                        [spe[kmin:k + wmax + 1], np.zeros(k + wmax + 1 - len(spe))])
                else:
                    line = spe[kmin:k + wmax + 1]

                # width line estimation
                line = line / np.linalg.norm(line)
                width = calcWidth(line, listKernel=listKernel, n_sig=n_sig, listWidth=np.concatenate(
                    [np.array([0.1]), np.arange(1, 2 * wmax + 2, 2)]))
                width = int(width)

                # keep only peaks larger than minimal width
                if width >= 2 * wmin + 1:
                    if len(np.nonzero(
                            spe[np.maximum(k - width, 0):k] < beta * sig)[0]) > 0:
                        a = np.maximum(k - width,
                                       1) + np.nonzero(spe[np.maximum(k - width,
                                                                      0):k] < beta * sig)[0][-1] - 1
                    else:
                        a = np.maximum(k - width, 0)
                    if len(np.nonzero(
                            spe[k:k + width + 1] < beta * sig)[0]) > 0:
                        b = k + \
                            np.nonzero(spe[k:k + width + 1] < beta * sig)[0][0] + 2
                    else:
                        b = k + width + 1
                else:
                    lRejected += 1
                    continue

                # if already covered at more than a fraction f by an existing
                # mask don't add current mask
                if np.any([np.sum(x[a:b]) > (b - a) * f for x in listMask]):
                    continue

                mask[a:b] = True
                listMask.append(mask)
    if returnAll:
        return listMask, lRejected, len(
            listExtrema), nThresh, listExtrema, listArgExtrema, spe_filt, sig_filt, sig
    return listMask


def genKernels(listWidth=np.arange(5, 42, 2), n=41, n_sig=2):
    """
    Generate list of gaussian kernels with varying widths

    Parameters
    ----------
    listWidth : list of int
        list of widths to be tested
    n: int
        length of a kernel 1d-array
    n_sig: float
         A half-width corresponds to n_sig standard deviations.


    Output
    ----------
    listKernel: list of 1d-array
        list of gaussian kernels with varying widths

    """
    listKernel = []
    x = np.linspace(-20, 20, n)
    for k in listWidth:
        g = sst.norm.pdf(x, scale=k / (n_sig * 2.))
        listKernel.append(g / np.linalg.norm(g))
    return listKernel


def calcWidth(spe, listKernel=None, n_sig=1, listWidth=np.arange(5, 42, 2)):
    """
    Estimate width of peak *spe* by correlation with a list of gaussian kernels
    with varying widths.

    Parameters
    ----------
    spe: 1d-array
        part of spectrum centered on the peak whose width has to estimated
    listKernel: list of 1d-array
        list of gaussian kernels with varying widths
    n_sig: int
         A half-width corresponds to n_sig standard deviations.
    listWidth : list of int
        list of widths to be tested

    Output
    ----------
    res : int
        estimated width

    """
    if listKernel is None:
        listKernel = []
        x = np.linspace(-20, 20, listWidth[-1])
        for k in listWidth:
            g = sst.norm.pdf(x, scale=k / (n_sig * 2.))
            listKernel.append(g / np.linalg.norm(g))
    listCorr = []
    for g in listKernel:
        listCorr.append(np.dot(spe, g))
    res = listWidth[np.argmax(np.abs(listCorr))]
    return res


def oneSigRuleRidge(LRCV):
    """
    Get regularization parameter using 1-sig rule

    Parameters
    ----------
    LRCV: sklearn.LinearModel.RidgeCV

    Returns
    -------
    alpha: regularisation parameter

    """
    rss = LRCV.cv_values_
    alphas = LRCV.alphas
    alpha = oneSigRuleMain(alphas, rss)
#    ind=np.argmin(np.mean(np.mean(LRCV.cv_values_,axis=0),axis=0))
#    min_mse=np.mean(np.mean(LRCV.cv_values_,axis=0),axis=0)[ind]
#    cv=LRCV.cv_values_.shape[0]*LRCV.cv_values_.shape[1]
#    min_mse_std=np.std(np.std(LRCV.cv_values_,axis=0),axis=0)[ind]/np.sqrt(cv)
#    alpha=np.max([LRCV.alphas[i] for i in range(len(LRCV.alphas)) if np.mean(LRCV.cv_values_[:,:,i])<min_mse+min_mse_std])
    return alpha


def oneSigRuleMain(alphas, rss):
    """
    Get regularization parameter using 1-sig rule

    Parameters
    ----------
    alphas: list of regularization parameters
    rss: array of prediction errors (shape : n_samples, n_targets, n_alphas or n_targets, n_alphas)

    Returns
    -------
    alpha: regularisation parameter

    """
    ind = np.argmin(np.mean(np.mean(rss, axis=0), axis=0))
    min_mse = np.mean(np.mean(rss, axis=0), axis=0)[ind]
    cv = rss.shape[0] * rss.shape[1]
    min_mse_std = np.std(np.std(rss, axis=0), axis=0)[ind] / np.sqrt(cv)
    alpha = np.max([alphas[i] for i in range(len(alphas))
                    if np.mean(rss[:, :, i]) < min_mse + min_mse_std])
    return alpha


def gridge_bic(X, Y, alphas=np.logspace(-5, 2, 50), multivar=False,
               averaged=True):
    """
    Estimate best regularization parameter alpha for ridge regression using BIC criterion.
    For ridge regression, the number of free parameters is estimated as
    Trace(S) where S = X(X^TX+alpha*I)^(-1)X^T

    Parameters
    ----------
        X : regressors (intensityMaps, n x k)
        Y : data (n x lmbda)
        alphas : regularization parameters to test
        multivar : if True, in BIC selection consider different variance for each wavelength (useless if averaged is True)
        averaged : if True do the BIC selection on averaged data (before doing the regression on original data)
    Output:
    ------
        alpha : estimated regularization parameter
    """
    if averaged:  # work on averaged data for model selection
        Y = np.mean(Y, axis=1)[:, None]

    n_samples = X.shape[0]
    n_targets = Y.shape[1]

    X_offset = np.average(X, axis=0)
    Y_offset = np.average(Y, axis=0)
    X = X - X_offset
    Y = Y - Y_offset

    K = np.log(n_samples * n_targets)  # BIC factor
    # compute singular value decomposition (SVD)
    U, sval, V = np.linalg.svd(X, full_matrices=False)

    df = np.zeros(len(alphas))  # Degrees of freedom
    mean_squared_error = []
    UtY = np.dot(U.T, Y)  # precomputation

    for k, alpha in enumerate(alphas):
        # compute coeffs using SVD
        # coef = (X^TX+alpha*I)^(-1)X^TY
        coef_ = np.dot(np.dot(V.T, np.diag(sval / (sval**2 + alpha))), UtY)
        R = Y - np.dot(X, coef_)  # residuals
        mean_squared_error.append(np.mean(R ** 2, axis=0))
        # estimation of degrees of freedom
        df[k] = np.sum(sval**2 / (alpha + sval**2))

    mean_squared_error = np.array(mean_squared_error)

    if multivar:
        criterion_ = n_samples * \
            np.sum(np.log(mean_squared_error), axis=1) + K * df
    else:
        criterion_ = n_samples * n_targets * \
            np.log(np.mean(mean_squared_error, axis=1)) + K * df

    n_best = np.argmin(criterion_)

    return alphas[n_best]


def gridge_cv(X, Y, ng=1, alphas=np.logspace(-5, 2, 50),
              oneSig=False, method='gcv_spe', sig2=None, support=None):
    """

    Estimate coefficients using ridge regression and various methods for
    regularization parameter estimation

    Parameters
    ----------
        X : regressors (intensityMaps, n x k)
        Y : data (n x lmbda)
        ng : size of spectral blocks
        alphas : list of regul parameters to test
        oneSig : if True, use the 'one sigma' rule to increase regularization efficiency
        method : choice of method for the estimation of regularization parameter
        sig2 : variance of each wavelength slice
        support : mask of samples (pixels) with enough signal, where the cross validation will be applied

    Output:
    ------
        coeff : estimated coefficients (spectra k x lmbda)
        intercepts : background (1 x lmbda)
    """

    coeff = np.zeros((X.shape[1], Y.shape[1]))
    intercepts = np.zeros((1, Y.shape[1]))
    RCV_slid = sklm.RidgeCV(alphas=alphas, fit_intercept=True, normalize=True,
                            store_cv_values=True)
    listAlpha = np.zeros((Y.shape[1]))
    listRSS = []

    X_centr = X - np.mean(X, axis=0)
    Y_centr = Y - np.mean(Y, axis=0)

    for x in range(X_centr.shape[1]):
        X_centr[:, x] = X_centr[:, x] / np.linalg.norm(X_centr[:, x])
    for k in range(int(np.ceil(Y.shape[1] / float(ng)))):

        if method == "bic":
            alpha = gridge_bic(
                X_centr, Y_centr[:, k * ng:(k + 1) * ng], alphas)
            Ridge = sklm.Ridge(alpha=alpha, fit_intercept=True, normalize=True)
            Ridge.fit(X, Y[:, k * ng:(k + 1) * ng])
            coeff[:, k * ng:(k + 1) * ng] = Ridge.coef_.T
            intercepts[:, k * ng:(k + 1) * ng] = Ridge.intercept_.T
        elif method == 'gcv_spe':

            alpha, rss = gridge_gcv_spectral(X_centr, Y_centr[:, k * ng:(
                k + 1) * ng], alphas=alphas, Sig2=sig2[k * ng:(k + 1) * ng], support=support, oneSig=oneSig)
            listAlpha[k * ng:(k + 1) * ng] = alpha
            listRSS.append(rss.mean(axis=0).mean(axis=0))
            Ridge = sklm.Ridge(alpha=alpha, fit_intercept=True, normalize=True)
            Ridge.fit(X, Y[:, k * ng:(k + 1) * ng])
            coeff[:, k * ng:(k + 1) * ng] = Ridge.coef_.T
            intercepts[:, k * ng:(k + 1) * ng] = Ridge.intercept_.T

        else:  # ridge with GCV
            RCV_slid.fit(X, Y[:, k * ng:(k + 1) * ng])
            if oneSig:
                alpha = oneSigRuleRidge(RCV_slid)
            else:
                alpha = RCV_slid.alpha_

            listAlpha[k * ng:(k + 1) * ng] = alpha
            listRSS.append(RCV_slid.cv_values_)
            Ridge = sklm.Ridge(alpha=alpha, fit_intercept=True, normalize=True)
            Ridge.fit(X, Y[:, k * ng:(k + 1) * ng])
            coeff[:, k * ng:(k + 1) * ng] = Ridge.coef_.T
            intercepts[:, k * ng:(k + 1) * ng] = Ridge.intercept_.T

    return coeff, intercepts, listAlpha, listRSS


def gridge_gcv_spectral(X,
                        Y,
                        support,
                        alphas=np.logspace(-5,
                                           2,
                                           50),
                        Sig2=None,
                        cross_spectral=False,
                        oneSig=True,
                        maxAlphaFrac=2.):
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
            mask of samples (pixels) with enough signal, where the cross validation will be applied
        alphas : list of float
            list of regul parameters to test
        Sig2 :
            variance of each wavelength slice (1d array n_targets)
        cross_spectral : bool
            do the cross-validation on previous and following spectral band.
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
    sumSig2 = Sig2[:-2] + Sig2[2:]
    U, sval, V = np.linalg.svd(Xs, full_matrices=False)
    UtY = np.dot(U.T, Ys)
    UtY2 = UtY**2
    # listInd=np.nonzero(support)[0]

    # rss=np.zeros((len(listInd),Y.shape[1],len(alphas)))
    rss = np.zeros((Y.shape[1], len(alphas)))
    for a, alpha in enumerate(alphas):
        if not cross_spectral:
            # S=(U * _diag_dot(sval**2/(sval**2+alpha),U.T).T).sum(-1) # get diag values of np.dot(Xs,np.dot(XtX,Xs.T))
            # GCV approx S_ii =Tr(S)/n
            TrS = np.sum(sval**2 / (sval**2 + alpha))
            # Xbeta=np.dot(U,_diag_dot(sval**2/(sval**2+alpha),UtY))
            residuals = np.sum(((alpha / (sval**2 + alpha))**2)
                               [:, None] * UtY2, axis=0)
            # rss[:,:,a]=((Ys-Xbeta)/(1-S)[:,None])**2
            # rss[:,:,a]=((Ys-Xbeta)/(1-TrS/Xs.shape[0]))**2
            rss[:, a] = residuals / (1 - TrS / Xs.shape[0])**2
        else:
            TrS = np.sum(sval**2 / (sval**2 + alpha))
            Xbeta = np.dot(U, _diag_dot(sval**2 / (sval**2 + alpha), UtY))
            res_left = ((Ys[:, :-1] - Xbeta[:, 1:]) /
                        (1 - TrS / Xs.shape[0]))**2
            res_right = ((Ys[:, 1:] - Xbeta[:, :-1]) /
                         (1 - TrS / Xs.shape[0]))**2
            rss[0, :, a] = res_right[:, 0]
            rss[1:-1, :, a] = (res_left[:, :-1] * 1 * Sig2[2:] +
                               res_right[:, 1:] * Sig2[:-2]) / sumSig2
            rss[-1, :, a] = res_left[:, -1]

    if oneSig:
        alpha = alphas[np.argmin(np.average(rss, axis=0, weights=1 / Sig2))]
        TrS = np.sum(sval**2 / (sval**2 + alpha))
        Xbeta = np.dot(U, _diag_dot(sval**2 / (sval**2 + alpha), UtY))
        rss_alpha = ((Ys - Xbeta) / (1 - TrS / Xs.shape[0]))**2
        cv = rss_alpha.shape[0] * rss_alpha.shape[1]
        min_mse_std = np.std(np.std(rss_alpha, axis=0), axis=0) / np.sqrt(cv)
        min_mse = np.mean(np.mean(rss_alpha, axis=0), axis=0)
        alpha = np.max([alphas[i] for i in range(len(alphas))
                        if np.mean(rss[:, i]) < min_mse + min_mse_std])
        #alpha = oneSigRuleMain(alphas,rss,std)
    else:
        #alpha = alphas[np.argmin(np.mean(np.average(rss,axis=1,weights=1/Sig2),axis=0))]
        alpha = alphas[np.argmin(np.average(rss, axis=0, weights=1 / Sig2))]
    if alpha > np.max(sval) * maxAlphaFrac:
        alpha = np.max(sval) * maxAlphaFrac

    return alpha, rss


def _diag_dot(D, B):
    # compute dot(diag(D), B)
    if len(B.shape) > 1:
        # handle case where B is > 1-d
        D = D[(slice(None), ) + (np.newaxis, ) * (len(B.shape) - 1)]
    return D * B


def _decomp_diag(v_prime, Q):
    # compute diagonal of the matrix: dot(Q, dot(diag(v_prime), Q^T))
    return (v_prime * Q ** 2).sum(axis=-1)


def medfilt(x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by symmetry (instead of zeros like scipy medfilt).
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[:j][::-1]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[len(x) - j:][::-1]
    return np.median(y, axis=1)


def regulDeblendFunc(X,
                     Y,
                     Y_c=None,
                     l_method='glasso_bic',
                     ng=10,
                     c_method='RCV',
                     cv_l=5,
                     cv_c=None,
                     n_alphas=100,
                     eps=1e-3,
                     alpha_c=0.0001,
                     oneSig=True,
                     support=None,
                     trueLines=None,
                     multivar=True,
                     alphas=np.logspace(-5,
                                        2,
                                        50),
                     recompute=True,
                     filt_w=101,
                     corrflux=True,
                     Y_sig2=None):
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
        c_method : str
            method to use for continuum estimation (of objects spectra)

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
    if Y_c is None:
        Y_c = np.vstack([medfilt(y, filt_w) for y in Y])
    Y_l = Y - Y_c

    if trueLines is not None:  # use prior knowlegde of lines support if given
        listMask = trueLines
    else:  # find lines support

        # First we compute one spectrum per object (rough estimation by summing
        # over each intensity map)
        listSpe = []
        for i in range(X.shape[1]):
            listSpe.append(np.dot(X[:, i:i + 1].T, Y_l)[0])

        # Then we seek all spectral lines on this bunch of spectra
        listMask = getLinesSupportList(
            listSpe,
            w=2,
            wmax=20,
            wmin=2,
            alpha=2.5,
            n_sig=1.4,
            filt=None)

    if l_method == 'glasso_bic':  # preferred approach : group bic approach
        l_coeff, l_intercepts = glasso_bic(X, Y_l, ng=ng, listMask=listMask,
                                           multivar=multivar)

    elif l_method == 'glasso_cv':  # use group lasso with cross validation
        # to avoid CV instability we work with pixels with strong enough signal
        # (*support*)
        if support is not None:
            X1 = X[support, :]
            Y_l1 = Y_l[support, :]
        else:
            X1 = X
            Y_l1 = Y_l
        l_coeff, l_intercepts = glasso_cv(X1, Y_l1, ng=ng, cv=cv_l, recompute=recompute,
                                          n_alphas=n_alphas, eps=eps, listMask=listMask,
                                          oneSig=oneSig)

    # remove estimated contribution from emission lines
    Y_c = Y - np.dot(X, l_coeff) - l_intercepts

    # we now work on remaining data Y_c
    if support is not None:
        X1 = X[support, :]
        Y_c1 = Y_c[support, :]
    else:
        X1 = X
        Y_c1 = Y_c
    if c_method == 'RCV':  # find one global regul parameter by GCV
        RCV = sklm.RidgeCV(alphas=alphas, normalize=True, cv=cv_c,
                           store_cv_values=True, fit_intercept=True)
        # to avoid GCV instability we work with pixels with strong enough
        # signal (*support*)
        RCV.fit(X1, Y_c1)
        c_coeff = RCV.coef_.T
        c_intercepts = RCV.intercept_.T

    elif c_method == 'Ridge':  # use given global regul parameter alpha
        Ridge = sklm.Ridge(alpha=alpha_c, normalize=True, fit_intercept=True)
        Ridge.fit(X, Y_c)
        c_coeff = Ridge.coef_.T
        c_intercepts = Ridge.intercept_.T

    elif c_method == 'LR':  # no regularization
        LR = sklm.LinearRegression(normalize=True, fit_intercept=True)
        LR.fit(X, Y_c)
        c_coeff = LR.coef_.T
        c_intercepts = LR.intercept_.T

    elif c_method == 'gridge_cv':  # preferred method : sliding Ridge GCV
        c_coeff, c_intercepts, c_alphas, listRSS = gridge_cv(
            X, Y_c, ng=ng, support=support, sig2=Y_sig2,
            oneSig=oneSig, alphas=alphas)

    # correct flux
    # add one row of ones for background/intercept to keep corresponding
    # positions in result arrays
    listA = np.ones((X.shape[1] + 1, Y.shape[1]))

    if corrflux:
        #        for k in range(int(np.ceil(Y_c.shape[1]/float(ng)))):
        #            r=corrFlux(X,Y_c[:,k*ng:(k+1)*ng],c_coeff[:,k*ng:(k+1)*ng])
        #            c_coeff[:,k*ng:(k+1)*ng] = r[0]
        #            listA[1:,k*ng:(k+1)*ng] = r[1][:,None]
        #   compute on whole block
        r = corrFlux(X, Y_c, c_coeff)
        c_coeff = r[0]
        listA[1:, :] = r[1][:, None]

    # combine coeffs from lines and continuum
    res = c_coeff + l_coeff
    intercepts = c_intercepts + l_intercepts

    if c_method == 'gridge_cv':
        return res, intercepts, listMask, c_coeff, l_coeff, Y, Y_l, Y_c, c_alphas, listRSS, listA
    return res, intercepts


def corrFlux(X, Y, beta):
    """
    Correct coefficients to limit flux loss
    We seek a diagonal matrix A to minimize ||Y-X*A*beta||^2 with A_ii>=1
    (as we know there has been some flux loss)

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
    listI = np.zeros(X.shape[1]).astype(bool)
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
    for i, a in enumerate(listA):
        beta_c[i, :] = beta_c[i, :] * a
    return beta_c, listA
