# -*- coding: utf-8 -*-

class Params():

    def __init__(self,
                 LW=20,

                 ):
        """
        Param: int *LW*, Lambda Window where the correlation test will occur (that must cover the half-width of the line emission)
        Param: int *SW*, Spatial Window for the exploration, if None the cube is fully explored spatially
        Param: bool *sim*, indicates if given sources are simulated ones (without wcs and wave objects)
        Param: int *lmbdaShift*, maximum shift in one direction to construct a family of target spectra
        from the estimated source spectrum. A dictionary with 2*lmbdaShift+1 spectra will be built.
        Param: string *centering*, choose to center all spectra ('all') or 'none' or only the target spectra ('ref')
        Param: bool *norm*, choose to norm (correlation approach) or not (matched filter approach)
        Param: np array *fsf*, fsf over wavelength (3D array)
        Param: np array *kernerl_mf*, kernel for matched filter
        """
        betaHST = 1.6
        alphaHST = np.sqrt((0.085/0.2*15)**2/(4*(2**(1/betaHST)-1)))  # expressed in MUSE pixels
        self.LW=LW
        self.SW=SW
        self.sim=sim
        self.lmbdaShift=lmbdaShift
        self.version=version
        self.origin=['SHADE',version]
        self.fsf= fsf
        self.kernel_mf=kernel_mf


        ### Regularization parameters

        #### Lines detection
        self.alpha = 2.5
        self