"""
@author: raphael.bacher@gipsa-lab.fr
"""

import multiprocessing
import numpy as np
import pathlib
import tqdm
from astropy.table import Table, vstack
from mpdaf import CPU

from .deblend import deblendGroup
from .deblend_utils import (calcMainKernelTransfert, get_fig_ax, cmap,
                            extractHST)
from .grouping import doGrouping, getObjsInBlob
from .parameters import Params


def prepare_inputs(cube, hstimages, segmap, blob_mask, bbox, imLabel, cat):
    """
    Extract data for each group before the multiprocessing (avoid large
    datasets copies)
    """
    subcube = cube[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
    subsegmap = extractHST(segmap, subcube[0])
    subhstimages = [extractHST(hst, subcube[0]) for hst in hstimages]

    for obj in [subcube, subsegmap] + subhstimages:
        # need to copy some header infos, because wcs info are not
        # completly copied during an image resize
        obj.data_header.update(obj.wcs.to_header())

    sub_blob_mask = blob_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    imMUSE = cube[0]

    subimMUSE = imMUSE[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    listObjInBlob, listHSTObjInBlob = getObjsInBlob(
        'ID', cat, sub_blob_mask, subimMUSE, subsegmap.data.filled(0))

    return subcube, subhstimages, subsegmap, listObjInBlob, listHSTObjInBlob


class ODHIN():

    def __init__(self, cube, hstimages, segmap, cat, output_dir, params=None,
                 imMUSE=None, imHST=None, main_kernel_transfert=None):
        """
        Main class for deblending process

        Parameters
        ----------
        cube : mpdaf.obj.Cube
            The mpdaf Cube object to be deblended
        hstimages: list of mpdaf.obj.Image
            HST mpdaf images
        segmap : mpdaf.obj.Image
            segmentation map at HST resolution
        cat: mpdaf.sdetect.Catalog
            catalog of sources
        output_dir: str
            if not None, results of each group is saved in this directory
        params : class with all parameters
            defined in module parameters
        imMUSE (opt):
            white image MUSE (if none created by summation on the cube)
        imHST (opt):
            reference HST image (if none the first of hstimages is taken)
        main_kernel_transfert (opt):
            kernel HST->MUSE to be used for preprocessing (grouping).
            If none provided build transfer kernel from default parameters

        Attributes
        ----------
        table_groups
        table_sources
        """

        self.cube = cube
        self.hstimages = hstimages
        self.segmap = segmap
        self.cat = cat
        self.params = params
        self.output_dir = pathlib.Path(output_dir)
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

        names = ('G_ID', 'nbSources', 'listIDs', 'Area', 'Xi2',
                 'Condition Number')
        groups = [[i, group.nbSources, tuple(group.listSources),
                   group.region.area, 0, 0]
                  for i, group in enumerate(self.groups)]
        self.table_groups = Table(names=names, rows=groups,
                                  dtype=(int, int, tuple, float, float, float))
        self.table_groups.add_index('G_ID')

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

            for i in listGroupToDeblend:
                reg = self.groups[i].region
                blob = (self.imLabel == i + 1)
                # args: subcube, subhstimages, subsegmap, listObjInBlob,
                # listHSTObjInBlob
                args = prepare_inputs(
                    self.cube, self.hstimages, self.segmap, blob, reg.bbox,
                    self.imLabel, self.cat)
                outfile = str(self.output_dir / f'group_{i:05d}.fits')
                pool.apply_async(deblendGroup,
                                 args=args+(i, outfile), callback=update)

            pool.close()
            pool.join()
        else:
            for i in listGroupToDeblend:
                reg = self.groups[i].region
                blob = (self.imLabel == i + 1)
                # args: subcube, subhstimages, subsegmap, listObjInBlob,
                # listHSTObjInBlob
                args = prepare_inputs(
                    self.cube, self.hstimages, self.segmap, blob, reg.bbox,
                    self.imLabel, self.cat)
                outfile = str(self.output_dir / f'group_{i:05d}.fits')
                deblendGroup(*args, i, outfile)

        self.build_result_table()

    def build_result_table(self):
        tables = [Table.read(f, hdu='TAB_SOURCE')
                  for f in self.output_dir.glob('group_*.fits')]
        self.table_sources = vstack(tables)
        return self.table_sources

        # self.table_groups.loc[group_id]['Xi2'] = xi2
        # self.table_groups.loc[group_id]['Condition Number'] = cond_number

        # self.results['Table Groups'] = self.table_groups
        # self.results['Table Sources'] = self.table_sources

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
