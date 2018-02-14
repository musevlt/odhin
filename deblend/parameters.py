# -*- coding: utf-8 -*-

"""
@author: raphael.bacher@gipsa-lab.fr
"""

import os
import numpy as np

DEFAULT_HSTFILTER606 = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    '../data/HST_ACS_WFC.F606W_81.dat')
DEFAULT_HSTFILTER775 = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    '../data/HST_ACS_WFC.F775W_81.dat')
DEFAULT_HSTFILTER814 = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    '../data/HST_ACS_WFC.F814W_81.dat')
DEFAULT_HSTFILTER850 = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    '../data/HST_ACS_WFC.F850LP_81.dat')

class Params():
    
    def __init__(self,
                 overlap=True,
                 betaHST=1.6,
                 fwhmHST=0.085,
                 nBands=10,
                 listFiltName=[DEFAULT_HSTFILTER606,
                                          DEFAULT_HSTFILTER775,
                                          DEFAULT_HSTFILTER814,
                                          DEFAULT_HSTFILTER850],
                ):
        
        self.overlap = overlap
        self.betaHST = betaHST
        self.fwhmHST = fwhmHST
        # expressed in MUSE pixels
        self.alphaHST = np.sqrt((self.fwhmHST / 0.2 * 15)**2 / (4 * (2**(1 / self.betaHST) - 1)))
        self.nBands = nBands
        self.listFiltName = listFiltName
    