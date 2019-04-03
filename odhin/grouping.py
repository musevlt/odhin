"""
@author: raphael.bacher@gipsa-lab.fr

Store methods for
- grouping sources to be deblended
- exploring groups
- modifying groups
"""

import logging
import numpy as np
from skimage.measure import regionprops, label

from .utils import createIntensityMap, ProgressBar

__all__ = ('SourceGroup', 'RegionAttr', 'doGrouping', 'getObjsInBlob')


class SourceGroup:

    __slots__ = ('ID', 'listSources', 'region', 'nbSources', 'listHST_ID')

    def __init__(self, ID, listSources, listHST_ID, region):
        self.ID = ID
        self.listSources = listSources
        self.listHST_ID = listHST_ID
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


def doGrouping(imHR, segmap, imMUSE, cat, kernel_transfert, params,
               verbose=True):
    """Segment all sources in a number of connected (at the MUSE resolution)
    groups.
    """
    logger = logging.getLogger(__name__)
    intensityMapLRConvol = createIntensityMap(imHR, segmap, imMUSE,
                                              kernel_transfert, params)

    imLabel = label(intensityMapLRConvol > params.cut)
    groups = []
    regions = regionprops(imLabel)
    if verbose:
        regions = ProgressBar(regions)

    for skreg in regions:
        # Build a RegionAttr object from a skimage region
        region = RegionAttr.from_skimage(skreg)
        region.compute_sky_centroid(imMUSE.wcs)
        region.ensureMinimalBbox(params.min_width, imLabel,
                                 params.min_sky_pixels, params.margin_bbox)
        blob_mask = (imLabel == skreg.label)
        sub_blob_mask = blob_mask[region.sy, region.sx]
        subimMUSE = imMUSE[region.sy, region.sx]
        hy, hx = region.convertToHR(segmap, imMUSE)
        subsegmap = segmap.data[hy, hx]

        listSources, hstids = getObjsInBlob('ID', cat, sub_blob_mask,
                                            subimMUSE, subsegmap)

        if len(listSources) == 1:
            # FIXME: this should not happen. It seems to happen when a source
            # is close to an edge, and because the HR to LR resampling remove
            # the source flux on the edge spaxels. Should investigate more!
            logger.warning('found no sources in group %d', skreg.label - 1)

        groups.append(SourceGroup(skreg.label - 1, listSources, hstids,
                                  region))

    return groups, imLabel


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
