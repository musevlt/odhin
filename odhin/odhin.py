"""
@author: raphael.bacher@gipsa-lab.fr
"""

import logging
import multiprocessing
import pathlib
import pickle
import warnings
from datetime import datetime

import numpy as np

from astropy.io import fits
from astropy.table import Column, Table, join, vstack
from mpdaf import CPU
from mpdaf.obj import Cube, Image
from mpdaf.sdetect import Catalog
from mpdaf.tools import MpdafUnitsWarning, progressbar

from .deblend import deblendGroup
from .grouping import doGrouping
from .parameters import Params, load_settings
from .utils import (
    calcMainKernelTransfert,
    check_segmap_catalog,
    cmap,
    extractHST,
    get_fig_ax,
)

# Ignore units warnings from MPDAF
warnings.simplefilter('ignore', MpdafUnitsWarning)


def _worker_deblend(group, outfile, conf, imLabel, timestamp):
    try:
        deblendGroup(group, outfile, conf, imLabel, timestamp)
    except Exception:
        logger = logging.getLogger(__name__)
        logger.error('group %d, failed', group.ID, exc_info=True)


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
    table_groups : `astropy.table.Table`
        Table computed by `ODHIN.grouping`, containing the information about
        groups.
    table_sources : `astropy.table.Table`
        Table computed by `ODHIN.deblend`, containing the information about
        the deblended sources.

    """

    def __init__(self, settings_file, output_dir, idname='ID', raname='RA',
                 decname='DEC'):
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

        self.segmap = extractHST(Image(self.conf['segmap']), self.imMUSE,
                                 integer_mode=True)

        # catalog: check for potential discrepancy between the catalog and the
        # segmentation map (as in Rafelski15)
        self.cat = Catalog.read(self.conf['catalog'])
        self.cat.add_index(idname)
        self.cat.meta['idname'] = self.idname = idname
        self.cat.meta['raname'] = self.raname = raname
        self.cat.meta['decname'] = self.decname = decname
        self.cat = check_segmap_catalog(self.segmap, self.cat,
                                        idname=self.idname)

        self.params = Params(**self.conf.get('params', {}))

        # Reference HR image
        ref_info = self.conf['hr_bands'][self.conf['hr_ref_band']]
        self.imHST = extractHST(Image(ref_info['file']), self.imMUSE)
        self.imHST.primary_header['photflam'] = ref_info.get('photflam', 1)

    @staticmethod
    def set_loglevel(level):
        """Change the logging level."""
        logger = logging.getLogger()
        logger.setLevel(level)
        logger.handlers[0].setLevel(level)

    def dump(self, filename):
        """Dump the ODHIN object to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, filename):
        """Load an ODHIN object from a pickle file."""
        with open(filename, 'rb') as f:
            self = pickle.load(f)
        # recreate the group_id index, otherwise it crashes with the default
        # index implementation
        self.table_groups.remove_indices('group_id')
        self.table_groups.add_index('group_id')
        return self

    def grouping(self, verbose=True, cut=None):
        """Segment all sources in a number of connected (at the MUSE
        resolution) groups and build a table of the groups.

        Parameters
        ----------
        verbose : bool
            If True, show a progress bar.
        cut : (float, float)
            Threshold on the convolved intensity map, to get the segmentation
            image.

        """
        if cut is not None:
            self.params.cut = cut

        # if nothing provided build transfer kernel from default parameters
        if 'kernel_transfert' in self.conf:
            kernel_transfert = fits.getdata(self.conf['kernel_transfert'])
        else:
            kernel_transfert = calcMainKernelTransfert(self.params, self.imHST)

        self.groups, self.imLabel, self.missing_ids = doGrouping(
            self.imHST, self.segmap, self.imMUSE, self.cat,
            kernel_transfert, self.params, idname=self.idname, verbose=verbose
        )

        rows = [[group.ID, group.nbSources, tuple(group.listSources),
                 group.region.area, group.step]
                for i, group in enumerate(self.groups)]
        self.table_groups = Table(
            names=('group_id', 'nb_sources', 'list_ids', 'area', 'step'),
            rows=rows,
            dtype=(int, int, tuple, float, int)
        )
        self.table_groups.add_index('group_id')

    def deblend(self, listGroupToDeblend=None, njobs=None, verbose=True):
        """Parallelized deblending on a list of groups

        Parameters
        ----------
        listGroupToDeblend : list
            List of group IDs to process. If not provided, all groups are
            processed, starting with the ones with the highest number of
            sources.
        njobs : int
            Number of process to run in parallel.
        verbose : bool
            If True, show a progress bar.

        """
        if self.groups is None:
            raise ValueError("No groups were defined. Please call the "
                             ".grouping() method before doing a deblend")

        # if no special groups are listed, do on all groups
        klist = []
        slist = []
        if listGroupToDeblend is None:
            for k,group in enumerate(self.groups):
                if len(group.listSources) == 1:
                    self.logger.warning('skipping group %d, no sources in group',
                                    group.ID)
                    continue
                klist.append(k)
                slist.append(len(group.listSources))
            klist = np.array(klist)
            ksort = np.argsort(slist)[::-1]
            klist = klist[ksort]
            listGroupToDeblend = klist

        self.output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().isoformat()

        to_process = []
        for i in listGroupToDeblend:
            group = self.groups[i]
            if len(group.listSources) == 1:
                self.logger.warning('skipping group %d, no sources in group',
                                    group.ID)
                continue

            outfile = str(self.output_dir / f'group_{group.ID:05d}.fits')
            to_process.append((group, outfile, self.conf, self.imLabel, timestamp))

        # Determine the number of processes:
        # - default: all CPUs except one.
        # - mdaf.CPU
        # - cpu_count parameter
        cpu_count = multiprocessing.cpu_count() - 1
        if CPU > 0 and CPU < cpu_count:
            cpu_count = CPU
        if njobs is not None and njobs < cpu_count:
            cpu_count = njobs

        cpu_count = min(cpu_count, len(listGroupToDeblend))
        self.logger.debug('using %d cpus', cpu_count)
        if cpu_count > 1:
            pool = multiprocessing.Pool(processes=cpu_count)
            if verbose:
                ntasks = len(to_process)
                # add progress bar
                pbar = progressbar(total=ntasks)

                def update(*a):
                    pbar.update()
            else:
                update = None

            for args in to_process:
                pool.apply_async(_worker_deblend, args=args, callback=update)

            pool.close()
            pool.join()
        else:
            for args in to_process:
                deblendGroup(*args)

        self.build_result_table()

    def build_result_table(self):
        """Build the result table from the sources.

        This is called at the end of the `~ODHIN.deblend` method.

        """
        tables = []
        for f in self.output_dir.glob('group_*.fits'):
            t = Table.read(f, hdu='TAB_SOURCES')
            t['timestamp'] = fits.getval(f, 'ODH_TS')
            tables.append(t)

        tables = vstack(tables)
        cat = Table([[str(x) for x in self.cat[self.idname]],
                     self.cat[self.raname], self.cat[self.decname]],
                    names=('id', 'ra', 'dec'))

        # join with input catalog (inner join to get only the processed ids,
        # and without the bg_* rows)
        self.table_sources = join(tables, cat, keys=['id'], join_type='inner')
        # cast id column to integer
        self.table_sources.replace_column(
            'id', Column(data=[int(x) for x in self.table_sources['id']])
        )
        self.table_sources.sort('group_id')
        return self.table_sources

    def plotGroups(self, ax=None, groups=None, linewidth=1):
        """Plot the segmentation map and groups.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to use for the plot.
        groups : list
            List of groups.

        """
        import matplotlib.patches as mpatches
        ax = get_fig_ax(ax)
        cm = cmap(self.imLabel.max(), random_state=12345)
        ax.imshow(self.imLabel, cmap=cm, origin='lower')
        if groups is None:
            groups = self.groups
        for group in groups:
            minr, maxr = group.region.sy.start, group.region.sy.stop
            minc, maxc = group.region.sx.start, group.region.sx.stop
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red',
                                      linewidth=linewidth)
            ax.add_patch(rect)

    def plotAGroup(self, ax=None, group_id=None, cmap='Greys', **kwargs):
        """Plot a group, with sources positions and contour of the label image.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to use for the plot.
        group_id : int
            Group id.
        cmap : str
            Colormap for the image plot.
        kwargs : dict
            Passed to Image.plot.

        """
        assert group_id is not None
        ax = get_fig_ax(ax)
        group = self.groups[group_id - 1]
        reg = group.region
        subim = self.imMUSE[reg.sy, reg.sx]
        subim.plot(ax=ax, cmap='Greys', **kwargs)
        ax.contour(self.imLabel[reg.sy, reg.sx] == group.ID, levels=1, colors='r')

        src = group.listSources.copy()
        if 'bg' in src:
            src.remove('bg')
        self.cat.plot_symb(ax, subim.wcs, label=True, esize=0.4)
        cat = self.cat[np.in1d(self.cat[self.idname], src)]
        y, x = subim.wcs.sky2pix(np.array([cat[self.decname], cat[self.raname]]).T).T
        ax.scatter(x, y, c="r")

    def plotHistArea(self, ax=None, nbins=None):
        """Plot histogram of group areas.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to use for the plot.
        nbins : str or int
            Number of bins for `matplotlib.pyplot.hist`.

        """
        ax = get_fig_ax(ax)
        if nbins is None:
            nbins = int(self.table_groups['area'].max()) // 50
        ax.hist([group.region.area for group in self.groups], bins=nbins)
        ax.set_title('Histogram of group areas')

    def plotHistNbS(self, ax=None, nbins=None):
        """Plot histogram of the number of sources per group.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to use for the plot.
        nbins : str or int
            Number of bins for `matplotlib.pyplot.hist`.

        """
        ax = get_fig_ax(ax)
        if nbins is None:
            nbins = self.table_groups['nb_sources'].max()
        ax.hist([group.nbSources for group in self.groups], bins=nbins)
        ax.set_title('Histogram of sources number per group')
