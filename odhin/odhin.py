"""
@author: raphael.bacher@gipsa-lab.fr
"""

import multiprocessing
import numpy as np
import tqdm
from astropy.table import Table, vstack
from mpdaf import CPU

from . import multi_deblend
from .deblend_utils import calcMainKernelTransfert, get_fig_ax, cmap
from .grouping import doGrouping
from .parameters import Params


class ODHIN():

    def __init__(self, cube, hstimages, segmap, cat, params=None, imMUSE=None,
                 imHST=None, main_kernel_transfert=None, write_dir=None):
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

        # if nothing provided take white image of the cube
        self.imMUSE = imMUSE or self.cube.sum(axis=0)

        # if nothing provided take the first of the HST images
        self.imHST = imHST or self.hstimages[0]

        # if nothing provided build transfer kernel from default parameters
        self.main_kernel_transfert = main_kernel_transfert or \
            calcMainKernelTransfert(self.params, self.imHST)

        self.results = dict({})

    def grouping(self, verbose=True, cut=None):
        """
        Segment all sources in a number of connected (at the MUSE resolution)
        groups and build a table of the groups.
        """
        if cut is not None:
            self.params.cut = cut

        self.groups, self.imLabel = doGrouping(
            self.cube, self.imHST, self.segmap, self.imMUSE, self.cat,
            self.main_kernel_transfert, params=self.params, verbose=verbose)
        self.buildGroupTable()

    def deblend(self, listGroupToDeblend=None, cpu=None, verbose=True):
        """
        Parallelized deblending on a list of groups
        """
        if self.groups is None:
            raise ValueError("No groups were defined. Please call a grouping "
                             "method before doing a deblend")

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

        cpu_count = min(cpu_count, len(listGroupToDeblend))
        if cpu_count > 1:
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
                # args: subcube, subhstimages, subsegmap, listObjInBlob,
                # listHSTObjInBlob
                args = multi_deblend.getInputs(
                    self.cube, self.hstimages, self.segmap, blob, reg.bbox,
                    self.imLabel, self.cat)
                job = pool.apply_async(multi_deblend.deblendGroup,
                                       args=args+(i, self.write_dir),
                                       callback=update)
                results_async.append(job)

            pool.close()
            pool.join()

            self.buildResults([res.get() for res in results_async])
        else:
            results = []
            for i in listGroupToDeblend:
                reg = self.groups[i].region
                blob = (self.imLabel == i + 1)
                # args: subcube, subhstimages, subsegmap, listObjInBlob,
                # listHSTObjInBlob
                args = multi_deblend.getInputs(
                    self.cube, self.hstimages, self.segmap, blob, reg.bbox,
                    self.imLabel, self.cat)
                res = multi_deblend.deblendGroup(*args, i, self.write_dir)
                results.append(res)

            self.buildResults(results)

    # Results functions

    def buildResults(self, results):

        self.table_sources = None
        self.dict_spec = {}
        self.dict_estimated_cube = {}
        self.dict_observed_cube = {}
        for res in results:
            if self.write_dir is None:
                (table_tmp, dict_spec_tmp, cube_observed_tmp,
                 cube_estimated_tmp, group_id, cond_number, xi2) = res
            else:
                table_tmp, dict_spec_tmp, group_id, cond_number, xi2 = res
            if self.table_sources is None:
                self.table_sources = table_tmp
            else:
                self.table_sources = vstack([self.table_sources, table_tmp])
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
        names = ('G_ID', 'nbSources', 'listIDs', 'Area', 'Xi2',
                 'Condition Number')
        groups = [[i, group.nbSources, tuple(group.listSources),
                   group.region.area, 0, 0]
                  for i, group in enumerate(self.groups)]
        self.table_groups = Table(names=names, rows=groups,
                                  dtype=(int, int, tuple, float, float, float))
        self.table_groups.add_index('G_ID')

    # Plotting functions

    def plotGroups(self, ax=None, groups=None, linewidth=1):
        """
        ax : matplotlib axe
        groups: list of groups
        """
        import matplotlib.patches as mpatches
        ax = get_fig_ax(ax)
        cm = cmap(self.imLabel.max(), random_state=12345)
        ax.imshow(self.imLabel, cmap=cm)
        if groups is None:
            groups = self.groups
        for group in groups:
            minr, minc, maxr, maxc = group.region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red',
                                      linewidth=linewidth)
            ax.add_patch(rect)

    def plotAGroup(self, ax=None, group_id=None):
        """
        ax : matplotlib axe
        group_id: group id
        """
        assert group_id is not None
        ax = get_fig_ax(ax)
        group = self.groups[group_id]
        minr, minc, maxr, maxc = group.region.bbox
        subim = self.imMUSE[minr:maxr, minc:maxc]
        subim.plot(ax=ax)
        ax.contour(self.imLabel[minr:maxr, minc:maxc] == group_id + 1,
                   levels=1, colors='r')

        src = group.listSources.copy()
        cat = self.cat[np.in1d(self.cat['ID'], src)]
        y, x = subim.wcs.sky2pix(np.array([cat['DEC'], cat['RA']]).T).T
        ax.scatter(x, y, c="r")

    def plotHistArea(self, ax=None, nbins='auto'):
        """Plot histogram of group areas.

        ax : matplotlib axe
        nbins: number of bins for histogram
        """
        ax = get_fig_ax(ax)
        ax.hist([group.region.area for group in self.groups], bins=nbins)

    def plotHistNbS(self, ax=None, nbins='auto'):
        """Plot histogram of group number of sources.

        ax : matplotlib axe
        nbins: number of bins for histogram
        """
        ax = get_fig_ax(ax)
        ax.hist([group.nbSources for group in self.groups], bins=nbins)
