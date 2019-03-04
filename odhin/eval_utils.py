#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:49:56 2017

@author: raphael.bacher@gipsa-lab.fr
"""


import numpy as np
import scipy.signal as ssl


def calcCondNumber(debl, listobj=None):
    """
    Compute condition number on a odhin object
    """
    if listobj is None:
        mat = np.array(debl.listIntensityMapLRConvol[0][0][1:])
    else:
        mat = np.array(debl.listIntensityMapLRConvol[0][0][listobj][1:])
    for col in mat:
        col[:] = col / np.sum(col)
    cond = np.linalg.cond(mat)
    return cond


def calcXi2_tot(debl):
    xi2 = 1 / (np.size(debl.residuals) - 3) * np.sum(debl.residuals**2 / debl.cubeLRVar)
    return xi2


def calcXi2_source(debl, k):
    mask = debl.listIntensityMapLRConvol[0][0][k].reshape(debl.shapeLR) > 0
    xi2 = 1 / (np.size(debl.residuals[:, mask]) - 3) * np.sum(debl.residuals[:, mask]**2 / debl.cubeLRVar[:, mask])
    return xi2


def calcCondNumber2(mat):
    for col in mat:
        col[:] = col / np.sum(col)
    cond = np.linalg.cond(mat)
    return cond


def calcInterCorr(debl, w=101):
    spe0 = debl.sources[0] - ssl.medfilt(debl.sources[0], kernel_size=w)
    spe1 = debl.sources[1] - ssl.medfilt(debl.sources[1], kernel_size=w)
    res = np.dot(spe0, spe1) / np.linalg.norm(spe0) / np.linalg.norm(spe1)
    return res


def calcInterCorr2(spe0, spe1, w=101):
    spe0_ = spe0 - ssl.medfilt(spe0, kernel_size=w)
    spe1_ = spe1 - ssl.medfilt(spe1, kernel_size=w)
    res = np.dot(spe0_, spe1_) / np.linalg.norm(spe0_) / np.linalg.norm(spe1_)
    return res


def calcRMSE(listSpe, listSpeRef):
    res = np.mean([np.sqrt(np.mean((spe0 - spe1)**2))
                   for spe0, spe1 in zip(listSpe, listSpeRef)])
    return res


def calcVar(listSpe, listMask=None):
    listVar = []
    if listMask is None:
        listMask = [np.ones(len(listSpe[0])).astype(bool)] * len(listSpe)
    for spe0, mask in zip(listSpe, listMask):
        listVar.append(np.var(spe0[mask]))
    return listVar


def calcRefCorr(debl, simu):
    res0 = np.dot(debl.sources[0], simu.spectraTot[0]) / \
        np.linalg.norm(debl.sources[0]) / np.linalg.norm(simu.spectraTot[0])
    res1 = np.dot(debl.sources[1], simu.spectraTot[1]) / \
        np.linalg.norm(debl.sources[1]) / np.linalg.norm(simu.spectraTot[1])
    return res0, res1


def calcRefCorr2(listSpe, listSpeRef):
    listRes = []
    for spe0, spe1 in zip(listSpe, listSpeRef):
        listRes.append(
            np.dot(
                spe0,
                spe1) /
            np.linalg.norm(spe0) /
            np.linalg.norm(spe1))
    return listRes


def calcFlux(listSpe, listSpeRef, listMask=None):
    listFluxRatio = []
    if listMask is None:
        listMask = [np.ones(len(listSpe[0])).astype(bool)] * len(listSpe)
    for spe0, spe1, mask in zip(listSpe, listSpeRef, listMask):
        listFluxRatio.append(np.sum(spe0[mask]) / np.sum(spe1[mask]))
    return listFluxRatio
