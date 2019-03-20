"""
@author: raphael.bacher@gipsa-lab.fr
"""

import logging
import multiprocessing
import numpy as np
import pathlib

from astropy.io import fits
from astropy.table import Table, vstack, join
from mpdaf import CPU
from mpdaf.obj import Cube, Image
from mpdaf.sdetect import Catalog

from .deblend import deblendGroup
from .utils import (calcMainKernelTransfert, get_fig_ax, cmap,
                    extractHST, check_segmap_catalog, ProgressBar)
from .grouping import doGrouping
from .parameters import Params, load_settings


def _worker_deblend(group, outfile, conf):
    try:
        deblendGroup(group, outfile, conf)
    except Exception:
        logger = logging.getLogger(__name__)
        logger.error('group %d, failed', group.GID, exc_info=True)


class ODHIN:
    """
    Main class for the deblending process.

    Parameters
    ----------
    settings_file : str
        Settings file.
    output_dir: str
        if not None, results of each group is saved in this directory

    Attributes
    ----------
    table_groups
    table_sources
    """

    def __init__(self, settings_file, output_dir):
        self.logger = logging.getLogger(__name__)
        self.logger.debug('loading settings from %s', settings_file)
        self.output_dir = pathlib.Path(output_dir)
        self.groups = None

        self.settings_file = settings_file
        self.conf = load_settings(settings_file)

        # if nothing provided take white image of the cube
        if 'white' in self.conf:
            self.imMUSE = Image(self.conf['white'])
        else:
            cube = Cube(self.conf['cube'])
            self.imMUSE = cube.sum(axis=0)

        self.segmap = extractHST(Image(self.conf['segmap']), self.imMUSE)

        # catalog: check for potential discrepancy between the catalog and the
        # segmentation map (as in Rafelski15)
        self.cat = Catalog.read(self.conf['catalog'])
        self.cat.add_index('ID')
        self.cat = check_segmap_catalog(self.segmap, self.cat)

        self.params = Params(**self.conf.get('params', {}))

        # Reference HR image
        ref_info = self.conf['hr_bands'][self.conf['hr_ref_band']]
        self.imHST = extractHST(Image(ref_info['file']), self.imMUSE)
        self.imHST.primary_header['photflam'] = ref_info.get('photflam', 1)

    @staticmethod
    def set_loglevel(level):
        logger = logging.getLogger()
        logger.setLevel(level)
        logger.handlers[0].setLevel(level)

    def grouping(self, verbose=True, cut=None):
        """
        Segment all sources in a number of connected (at the MUSE resolution)
        groups and build a table of the groups.
        """
        if cut is not None:
            self.params.cut = cut

        # if nothing provided build transfer kernel from default parameters
        if 'kernel_transfert' in self.conf:
            kernel_transfert = fits.getdata(self.conf['kernel_transfert'])
        else:
            kernel_transfert = calcMainKernelTransfert(self.params, self.imHST)

        self.groups, self.imLabel = doGrouping(
            self.imHST, self.segmap, self.imMUSE, self.cat,
            kernel_transfert, params=self.params, verbose=verbose)

        names = ('G_ID', 'nbSources', 'listIDs', 'Area')
        groups = [[group.GID, group.nbSources, tuple(group.listSources),
                   group.region.area]
                  for i, group in enumerate(self.groups)]
        self.table_groups = Table(names=names, rows=groups,
                                  dtype=(int, int, tuple, float))
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

        self.output_dir.mkdir(exist_ok=True)

        to_process = []
        for i in listGroupToDeblend:
            group = self.groups[i]
            if len(group.listSources) == 1:
                self.logger.warning('skipping group %d, no sources in group',
                                    group.GID)
                continue

            self.logger.debug('deblending group %d', group.GID)
            # args: subcube, subsegmap
            outfile = str(self.output_dir / f'group_{group.GID:05d}.fits')
            to_process.append((group, outfile, self.conf))

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
        self.logger.debug('using %d cpus', cpu_count)
        if cpu_count > 1:
            pool = multiprocessing.Pool(processes=cpu_count)
            if verbose:
                ntasks = len(listGroupToDeblend)
                # add progress bar
                pbar = ProgressBar(total=ntasks)

                def update(*a):
                    pbar.update()
            else:
                def update(*a):
                    pass

            for args in to_process:
                pool.apply_async(_worker_deblend, args=args, callback=update)

            pool.close()
            pool.join()
        else:
            for args in to_process:
                deblendGroup(*args)

        self.build_result_table()

    def build_result_table(self):
        tables = vstack([Table.read(f, hdu='TAB_SOURCES')
                         for f in self.output_dir.glob('group_*.fits')])
        cat = Table([[str(x) for x in self.cat['ID']], self.cat['RA'],
                     self.cat['DEC']], names=('ID', 'RA', 'DEC'))
        self.table_sources = join(tables, cat, keys=['ID'], join_type='left')
        return self.table_sources

    def plotGroups(self, ax=None, groups=None, linewidth=1):
        """
        ax : matplotlib axis
        groups: list of groups
        """
        import matplotlib.patches as mpatches
        ax = get_fig_ax(ax)
        cm = cmap(self.imLabel.max(), random_state=12345)
        ax.imshow(self.imLabel, cmap=cm)
        if groups is None:
            groups = self.groups
        for group in groups:
            minr, maxr = group.region.sy.start, group.region.sy.stop
            minc, maxc = group.region.sx.start, group.region.sx.stop
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red',
                                      linewidth=linewidth)
            ax.add_patch(rect)

    def plotAGroup(self, ax=None, group_id=None):
        """
        ax : matplotlib axis
        group_id: group id
        """
        assert group_id is not None
        ax = get_fig_ax(ax)
        group = self.groups[group_id]
        reg = group.region
        subim = self.imMUSE[reg.sy, reg.sx]
        subim.plot(ax=ax)
        ax.contour(self.imLabel[reg.sy, reg.sx] == group.GID + 1,
                   levels=1, colors='r')

        src = group.listSources.copy()
        if 'bg' in src:
            src.remove('bg')
        cat = self.cat[np.in1d(self.cat['ID'], src)]
        y, x = subim.wcs.sky2pix(np.array([cat['DEC'], cat['RA']]).T).T
        ax.scatter(x, y, c="r")

    def plotHistArea(self, ax=None, nbins='auto'):
        """Plot histogram of group areas.

        ax : matplotlib axis
        nbins: number of bins for histogram
        """
        ax = get_fig_ax(ax)
        ax.hist([group.region.area for group in self.groups], bins=nbins)

    def plotHistNbS(self, ax=None, nbins='auto'):
        """Plot histogram of group number of sources.

        ax : matplotlib axis
        nbins: number of bins for histogram
        """
        ax = get_fig_ax(ax)
        ax.hist([group.nbSources for group in self.groups], bins=nbins)
