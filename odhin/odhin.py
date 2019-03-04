"""
@author: raphael.bacher@gipsa-lab.fr
"""

import matplotlib.patches as mpatches
import multiprocessing
import tqdm
from astropy import table
from mpdaf import CPU

from . import multi_deblend
from .deblend_utils import calcMainKernelTransfert
from .grouping import doGrouping
from .parameters import Params


class ODHIN():

    def __init__(self, cube, hstimages, segmap, cat, params=None, imMUSE=None, imHST=None, main_kernel_transfert=None, write_dir=None):
        """
        Main class for deblending process


        Parameters
        ----------
        cube : mpdaf Cube
            The mpdaf Cube object to be deblended
        hstimages: mpdaf Image
            HST mpdaf images
        segmap : mpdaf Image
            segmentation map at HST resolution
        cat: mpdaf Catalog
            catalog of sources
        params : class with all parameters
            defined in module parameters
        imMUSE (opt):
            white image MUSE (if none created by summation on the cube)
        imHST (opt):
            reference HST image (if none the first of hstimages is taken)
        main_kernel_transfert (opt):
            kernel HST->MUSE to be used for preprocessing (grouping).
            If none provided build transfer kernel from default parameters
        write_dir (opt):
            if not None, results of each group is saved in this directory

        Attributes
        ----------
        results
        table_groups
        table_sources
        dict_estimated_cube
        dict_observed_cube
        dict_spec
        """

        self.cube = cube
        self.hstimages = hstimages
        self.segmap = segmap
        self.cat = cat
        self.params = params
        self.write_dir = write_dir
        if params is None:
            self.params = Params()
        self.groups = None
        if imMUSE is None:  # if nothing provided take white image of the cube
            self.imMUSE = self.cube.sum(axis=0)
        else:
            self.imMUSE = imMUSE
        if imHST is None:  # if nothing provided take the first of the HST images
            self.imHST = self.hstimages[0]
        else:
            self.imHST = imHST
        if main_kernel_transfert is None:  # if nothing provided build transfer kernel from default parameters
            self.main_kernel_transfert = calcMainKernelTransfert(self.params, self.imHST)
        else:
            self.main_kernel_transfert = main_kernel_transfert
        self.results = dict({})

    def grouping(self, verbose=True, cut=None):
        """
        Segment all sources in a number of connected (at the MUSE resolution) groups
        and build a table of the groups
        """
        if cut is not None:
            self.params.cut = cut

        self.groups, self.imLabel = doGrouping(self.cube, self.imHST, self.segmap,
                                               self.imMUSE, self.cat, self.main_kernel_transfert,
                                               params=self.params, verbose=verbose)
        self.buildGroupTable()

    def deblend(self, listGroupToDeblend=None, cpu=None, verbose=True):
        """
        Parralelized deblending on a list of groups
        """
        if self.groups is None:
            print(
                "No groups were defined. Please call a grouping method before doing a deblend")
            return 0

        # if no special groups are listed, do on all groups
        if listGroupToDeblend is None:
            listGroupToDeblend = range(len(self.groups))

        # Determine the number of processes:
        # - default: all CPUs except one.
        # - mdaf.CPU
        # - cpu_count parameter
        cpu_count = multiprocessing.cpu_count() - 1
        if CPU > 0 and CPU < cpu_count:
            cpu_count = CPU
        if cpu is not None and cpu < cpu_count:
            cpu_count = cpu

        pool = multiprocessing.Pool(processes=cpu_count)
        if verbose:
            ntasks = len(listGroupToDeblend)
            # add progress bar
            pbar = tqdm.tqdm(total=ntasks)

            def update(*a):
                pbar.update()
        else:
            def update(*a):
                pass

        results_async = []
        for i in listGroupToDeblend:
            reg = self.groups[i].region
            blob = (self.imLabel == i + 1)
            subcube, subhstimages, subsegmap, listObjInBlob, listHSTObjInBlob = multi_deblend.getInputs(
                self.cube, self.hstimages, self.segmap, blob, reg.bbox, self.imLabel, self.cat)
            results_async.append(pool.apply_async(multi_deblend.deblendGroup, args=(
                subcube, subhstimages, subsegmap, listObjInBlob, listHSTObjInBlob, i, self.write_dir), callback=update))

        pool.close()
        pool.join()

        self.buildResults(results_async)

    # Results functions

    def buildResults(self, results):

        self.table_sources = None
        self.dict_spec = {}
        self.dict_estimated_cube = {}
        self.dict_observed_cube = {}
        for res in results:
            if self.write_dir is None:
                table_tmp, dict_spec_tmp, cube_observed_tmp, cube_estimated_tmp, group_id, cond_number, xi2 = res.get()
            else:
                table_tmp, dict_spec_tmp, group_id, cond_number, xi2 = res.get()
            if self.table_sources is None:
                self.table_sources = table_tmp
            else:
                self.table_sources = table.vstack([self.table_sources, table_tmp])
            self.dict_spec.update(dict_spec_tmp)
            if self.write_dir is None:
                self.dict_estimated_cube[group_id] = cube_estimated_tmp
                self.dict_observed_cube[group_id] = cube_observed_tmp
            self.table_groups.loc[group_id]['Xi2'] = xi2
            self.table_groups.loc[group_id]['Condition Number'] = cond_number

        self.results['Table Groups'] = self.table_groups
        self.results['Table Sources'] = self.table_sources
        self.results['Observed Cubes'] = self.dict_observed_cube
        self.results['Estimated Cubes'] = self.dict_estimated_cube
        self.results['Spectra'] = self.dict_spec

    def buildGroupTable(self):
        self.table_groups = table.Table(names=('G_ID', 'nbSources', 'listIDs', 'Area', 'Xi2', 'Condition Number'),
                                        dtype=(int, int, tuple, float, float, float))

        for i, group in enumerate(self.groups):
            self.table_groups.add_row([i, group.nbSources, tuple(group.listSources), group.region.area, 0, 0])

        self.table_groups.add_index('G_ID')

    # Plotting functions

    def plotGroups(self, ax, groups=None):
        """
        ax : matplotlib axe
        groups: list of groups
        """
        ax.imshow(self.imLabel)
        if groups is None:
            groups = self.groups
        for group in groups:
            minr, minc, maxr, maxc = group.region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    def plotAGroup(self, ax, group_id):
        """
        ax : matplotlib axe
        group_id: group id
        """
        group = self.groups[group_id]
        minr, minc, maxr, maxc = group.region.bbox
        self.imMUSE[minr:maxr, minc:maxc].plot(ax=ax)
        ax.contour(self.imLabel[minr:maxr, minc:maxc] == group_id + 1, levels=1, colors='r')
        listX = []
        listY = []
        for src in group.listSources:
            if 'bg' not in str(src):
                row = self.cat.loc['ID', src]
                y, x = self.imMUSE[minr:maxr, minc:maxc].wcs.sky2pix((row['DEC'], row['RA']))[0]
                listX.append(x)
                listY.append(y)
        ax.scatter(listX, listY, c="r")

    def plotHistArea(self, ax, nbins=20):
        """
        Plot histogram of group areas


        ax : matplotlib axe
        nbins: number of bins for histogram
        """
        listArea = []
        for group in self.groups:
            listArea.append(group.region.area)
        ax.hist(listArea, bins=nbins)

    def plotHistNbS(self, ax, nbins=20):
        """
        Plot histogram of group number of sources


        ax : matplotlib axe
        nbins: number of bins for histogram
        """
        listNbS = []
        for group in self.groups:
            listNbS.append(group.nbSources)
        ax.hist(listNbS, bins=nbins)
