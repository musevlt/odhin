import numpy as np
import scipy.stats as sst
from astropy.stats import mad_std
from scipy.ndimage.morphology import grey_dilation
from scipy.signal import fftconvolve


def getLinesSupportList(listSpe, w=2, wmin=1, wmax=20, alpha=1.4, beta=1.2,
                        n_sig=1.2, f=0.6, returnAll=False, filt=None,
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
        to avoid too many overlapping masks : if already covered at more than
        a fraction f by an existing mask don't add current mask
    returnAll : bool
        return all intermediate results
    filt: 1d-array (None by default)
        pattern for matching filter (gaussian shape will be made by default)
    localConstraint : bool
        if True, reject peaks where immediate neighbors of extrema are not
        > 1 std (maxima) or < -1 std (minima).

    Output
    ------
    listMask
        list of masks (each mask is a boolean array of size lambda)

    """
    if filt is None:
        filt = sst.norm.pdf(np.linspace(-2 * w, 2 * w, 4 * w + 1), scale=w)
        filt /= np.linalg.norm(filt)

    # generate kernels for estimation of line width
    listWidth = np.concatenate([np.array([0.1]),
                                np.arange(1, 2 * wmax + 2, 2)])
    listKernel = genKernels(listWidth=listWidth, n=2 * wmax + 1, n_sig=n_sig)

    listMask = []
    for spe in listSpe:
        # compute standard deviation estimator from MAD
        sig = mad_std(spe)
        # matched filter using filt
        spe_filt = fftconvolve(spe, filt, mode='same')
        # compute standard deviation estimator of filtered data from MAD
        sig_filt = mad_std(spe_filt)
        lRejected = 0

        # find local extrema
        mask0 = np.ones(2 * w + 1)
        mask0[w] = 0
        B = grey_dilation(np.abs(spe_filt), footprint=mask0)
        listArgExtrema = np.nonzero(np.abs(spe_filt) > B)[0]
        listExtrema = spe_filt[listArgExtrema]

        # compute number of kept extrema after thresholding
        nThresh = np.sum(np.abs(spe_filt[listArgExtrema]) > alpha * sig_filt)

        for k, extrema in zip(listArgExtrema, listExtrema):
            if (np.abs(extrema) > alpha * sig_filt) and (
                    (localConstraint is False) or np.all(
                        spe[np.maximum(k - 1, 0):k + 2] > np.sign(spe[k]) * sig
                    )):
                mask = np.zeros_like(spe, dtype=bool)
                kmin = np.maximum(k - wmax, 0)

                # create sub spectrum of good size for width estimation
                if k - wmax < 0:
                    line = np.concatenate([np.zeros(wmax - k),
                                           spe[kmin:k + wmax + 1]])
                elif wmax + k + 1 > len(spe):
                    line = np.concatenate([spe[kmin:k + wmax + 1],
                                           np.zeros(k + wmax + 1 - len(spe))])
                else:
                    line = spe[kmin:k + wmax + 1]

                # width line estimation
                line = line / np.linalg.norm(line)
                width = calcWidth(line, listKernel=listKernel, n_sig=n_sig,
                                  listWidth=listWidth)
                width = int(width)

                # keep only peaks larger than minimal width
                if width >= 2 * wmin + 1:
                    if len(np.nonzero(
                            spe[np.maximum(k - width, 0):k] < beta * sig)[0]) > 0:
                        a = (np.maximum(k - width, 1) +
                             np.nonzero(
                                 spe[np.maximum(k - width, 0):k] < beta * sig
                             )[0][-1] - 1)
                    else:
                        a = np.maximum(k - width, 0)
                    if len(np.nonzero(
                            spe[k:k + width + 1] < beta * sig)[0]) > 0:
                        b = k + \
                            np.nonzero(spe[k:k + width + 1] <
                                       beta * sig)[0][0] + 2
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
        return (listMask, lRejected, len(listExtrema), nThresh, listExtrema,
                listArgExtrema, spe_filt, sig_filt, sig)
    else:
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
        g /= np.linalg.norm(g)
        listKernel.append(g)
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
    return listWidth[np.argmax(np.abs(listCorr))]
