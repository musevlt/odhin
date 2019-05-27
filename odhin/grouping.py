"""
@author: raphael.bacher@gipsa-lab.fr

Store methods for
- grouping sources to be deblended
- exploring groups
- modifying groups
"""

import logging
from itertools import chain

import numpy as np
from mpdaf.tools import progressbar
from skimage.measure import label, regionprops

from .utils import createIntensityMap

__all__ = ('SourceGroup', 'RegionAttr', 'doGrouping', 'getObjsInBlob')


class SourceGroup:

    __slots__ = ('ID', 'listSources', 'listHST_ID', 'region', 'step',
                 'nbSources')

    def __init__(self, ID, listSources, listHST_ID, region, step):
        self.ID = ID
        self.listSources = listSources
        self.listHST_ID = listHST_ID
        self.step = step
        self.region = region  # RegionAttr region
        self.nbSources = len(listSources)

    def __repr__(self):
        label = 'sources' if self.nbSources > 1 else 'source'
        return f'<SourceGroup({self.ID}, {self.nbSources} {label})>'


class RegionAttr:
    """
    Get region attributes from skimage region properties
    """

    __slots__ = ('area', 'centroid', 'sx', 'sy', 'ra', 'dec')

    def __init__(self, area, centroid, sy, sx):
        self.area = area
        self.centroid = centroid
        self.sy = sy
        self.sx = sx

    @classmethod
    def from_skimage(cls, reg):
        min_row, min_col, max_row, max_col = reg.bbox
        sy = slice(min_row, max_row)
        sx = slice(min_col, max_col)
        return cls(reg.area, reg.centroid, sy, sx)

    def compute_sky_centroid(self, wcs):
        self.dec, self.ra = wcs.pix2sky(self.centroid)[0]

    @property
    def bbox_area(self):
        return (self.sy.stop - self.sy.start) * (self.sx.stop - self.sx.start)

    def ensureMinimalBbox(self, min_width, imLabel, min_sky_pixels, margin):
        """
        Ensures that region respects a minimal area and contains at least
        `min_sky_pixels` sky pixels.
        """
        # First add margin around bounding box
        ny, nx = imLabel.shape
        self.sy = slice(int(max(self.sy.start - margin, 0)),
                        int(min(self.sy.stop + margin, ny)))
        self.sx = slice(int(max(self.sx.start - margin, 0)),
                        int(min(self.sx.stop + margin, nx)))

        # then check minimal area
        if self.bbox_area < min_width**2:
            half_width = min_width // 2
            self.sy = slice(int(max(self.centroid[0] - half_width, 0)),
                            int(min(self.centroid[0] + half_width, ny)))
            self.sx = slice(int(max(self.centroid[1] - half_width, 0)),
                            int(min(self.centroid[1] + half_width, nx)))

        # then check minimal number of sky pixels
        nb_pixels = np.sum(imLabel[self.sy, self.sx] == 0)
        while nb_pixels < min_sky_pixels:
            min_width = min_width + 1
            half_width = min_width // 2
            self.sy = slice(int(max(self.centroid[0] - half_width, 0)),
                            int(min(self.centroid[0] + half_width, ny)))
            self.sx = slice(int(max(self.centroid[1] - half_width, 0)),
                            int(min(self.centroid[1] + half_width, nx)))
            nb_pixels = np.sum(imLabel[self.sy, self.sx] == 0)

    def convertToHR(self, imHR, imLR):
        """Convert the bounding box from low resolution (MUSE) to
        high resolution (HST).
        """
        # compute coordinates of bottom left corners
        pos = np.array([[self.sy.start, self.sx.start],
                        [self.sy.stop, self.sx.stop]]) - 0.5
        # get HR pixel indices
        hrpix = imHR.wcs.sky2pix(imLR.wcs.pix2sky(pos), nearest=True)
        sy, sx = (slice(*x) for x in hrpix.T)
        return sy, sx


def doGrouping(imHR, segmap, imMUSE, cat, kernel, params, idname='ID', verbose=True):
    """Compute groups of connected (at the MUSE resolution) sources.

    The grouping is done in 2 steps, with 2 thresholds. The first one allows to
    get the groups with bright sources, and the second is needed to get all
    sources included faint ones.

    """
    logger = logging.getLogger(__name__)

    if len(params.cut) != 2:
        raise ValueError(f'the cut param must contain 2 values')

    groups = []
    im_label_comb = np.zeros(imMUSE.shape, dtype=int)

    for it in range(2):
        intensityMapLRConvol = createIntensityMap(imHR, segmap, imMUSE, kernel, params)
        im_label = label(intensityMapLRConvol > params.cut[it])

        # compute offset before adding the label image
        offset_label = im_label_comb.max()

        # combine label images
        im_label_comb += np.where(im_label > 0, im_label + offset_label, 0)

        regions = regionprops(im_label)

        if verbose:
            regions = progressbar(regions)

        for skreg in regions:
            # Build a RegionAttr object from a skimage region
            region = RegionAttr.from_skimage(skreg)
            region.compute_sky_centroid(imMUSE.wcs)
            region.ensureMinimalBbox(params.min_width, im_label,
                                     params.min_sky_pixels, params.margin_bbox)
            blob_mask = (im_label == skreg.label)
            sub_blob_mask = blob_mask[region.sy, region.sx]
            subimMUSE = imMUSE[region.sy, region.sx]
            hy, hx = region.convertToHR(segmap, imMUSE)
            subsegmap = segmap.data[hy, hx]

            listSources, hstids = getObjsInBlob(idname, cat, sub_blob_mask,
                                                subimMUSE, subsegmap)

            if len(listSources) == 1:
                # FIXME: this should not happen. It seems to happen when
                # a source is close to an edge, and because the HR to LR
                # resampling remove the source flux on the edge spaxels.
                # Should investigate more!
                logger.warning('found no sources in group %d', skreg.label)

            gid = skreg.label + offset_label
            groups.append(SourceGroup(gid, listSources, hstids, region, it + 1))

        # build the list of all IDs that are included in a group
        listSources = (grp.listSources for grp in groups)
        ids_in_groups = set(
            int(i) for i in chain.from_iterable(listSources) if i != 'bg'
        )
        area = np.array([grp.region.area for grp in groups])

        # find the IDs that are not in a group
        tbl = cat.select(imMUSE.wcs, margin=0)
        missing_ids = sorted(set(tbl[idname].tolist()) - ids_in_groups)
        logger.info(
            'Step %d: %d groups, %d sources, %d missing sources, area min=%d max=%d',
            it + 1, len(groups), len(ids_in_groups), len(missing_ids),
            area.min(), area.max()
        )

        if it == 0:
            # mask the HR image for the next iteration
            missing_map = np.logical_or.reduce(
                [segmap._data == i for i in missing_ids]
            ).astype(int)

            imHR = imHR * missing_map

    return groups, im_label_comb


def getObjsInBlob(idname, cat, sub_blob_mask, subimMUSE, subsegmap):
    """Return the index and IDs of sources in the blobs.

    Returns
    -------
    listHSTObjInBlob : list of int
        List of catalog IDs connected to the blob.
    listHST_ID : list of int
        List of all catalog IDs in the cutout.

    """
    listHST_ID = np.unique(subsegmap)
    listHST_ID = listHST_ID[listHST_ID > 0]

    subcat = cat.loc[idname, listHST_ID]
    center = np.array([subcat['DEC'], subcat['RA']]).T
    centerMUSE = subimMUSE.wcs.sky2pix(center, nearest=True).T
    idx = sub_blob_mask[centerMUSE[0], centerMUSE[1]]

    # listObjInBlob = [0] + list(np.where(idx)[0] + 1)
    listHSTObjInBlob = ['bg'] + list(listHST_ID[idx])
    listHST_ID = ['bg'] + list(listHST_ID)
    return listHSTObjInBlob, listHST_ID
