"""
@author: raphael.bacher@gipsa-lab.fr

Store methods for
- grouping sources to be deblended
- exploring groups
- modifying groups
"""

import numpy as np
from skimage.measure import regionprops, label
from tqdm import tqdm

from .deblend_utils import createIntensityMap, modifSegmap


class SourceGroup:

    __slots__ = ('GID', 'listSources', 'region', 'nbSources')

    def __init__(self, GID, listSources, region):
        self.GID = GID
        self.listSources = listSources
        self.region = region  # RegionAttr region
        self.nbSources = len(listSources)


class RegionAttr:
    """
    Get region attributes from skimage region properties
    """

    __slots__ = ('area', 'centroid', 'bbox')

    def __init__(self, area, centroid, bbox):
        self.area = area
        self.centroid = centroid
        self.bbox = list(bbox)

    @classmethod
    def from_skimage(cls, reg):
        return cls(reg.area, reg.centroid, reg.bbox)


def doGrouping(cube, imHR, segmap, imMUSE, cat, kernel_transfert, params,
               verbose=True):
    """
    Segment all sources in a number of connected (at the MUSE resolution)
    groups
    """
    # needed because of potential discrepancy between the catalog and the
    # segmentation map (as in Rafelski15)
    segmap = modifSegmap(segmap, cat)

    intensityMapLRConvol = createIntensityMap(imHR, segmap, imMUSE,
                                              kernel_transfert, params)

    cut = params.cut
    imLabel = label(intensityMapLRConvol > cut)
    groups = []
    regions = regionprops(imLabel)
    if verbose:
        regions = tqdm(regions)

    for skreg in regions:
        # Build a RegionAttr object from a skimage region
        region = RegionAttr.from_skimage(skreg)
        ensureMinimalBbox(region, params.min_width, imLabel,
                          params.min_sky_pixels, params.margin_bbox)
        bbox = region.bbox  # order is row0,column0,row1,column1

        bboxHR = convertBboxToHR(bbox, segmap, imMUSE)
        subsegmap = segmap.data[bboxHR[0]:bboxHR[2], bboxHR[1]:bboxHR[3]]
        blob_mask = (imLabel == skreg.label)
        sub_blob_mask = blob_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        subimMUSE = imMUSE[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        sources = getObjsInBlob('ID', cat, sub_blob_mask, subimMUSE,
                                subsegmap)[1]
        groups.append(SourceGroup(skreg.label - 1, sources, region))

    return groups, imLabel


def ensureMinimalBbox(region, width, imLabel, min_sky_pixels, margin_bbox):
    """
    Ensures that region respects a minimal area and contains at least
    `min_sky_pixels` sky pixels.
    """
    # First add margin around bounding box
    nx, ny = imLabel.shape
    region.bbox[0] = int(max(region.bbox[0] - margin_bbox, 0))
    region.bbox[1] = int(max(region.bbox[1] - margin_bbox, 0))
    region.bbox[2] = int(min(region.bbox[2] + margin_bbox, nx))
    region.bbox[3] = int(min(region.bbox[3] + margin_bbox, ny))
    area = ((region.bbox[2] - region.bbox[0]) *
            (region.bbox[3] - region.bbox[1]))

    # then check minimal area
    if area < width**2:
        region.bbox[0] = int(max(region.centroid[0] - width // 2, 0))
        region.bbox[1] = int(max(region.centroid[1] - width // 2, 0))
        region.bbox[2] = int(min(region.centroid[0] + width // 2, nx))
        region.bbox[3] = int(min(region.centroid[1] + width // 2, ny))

    nb_pixels = np.sum(imLabel[region.bbox[0]:region.bbox[2],
                               region.bbox[1]:region.bbox[3]] == 0)

    # then check minimal number of sky pixels
    while nb_pixels < min_sky_pixels:
        width = width + 1
        region.bbox[0] = int(max(region.centroid[0] - width // 2, 0))
        region.bbox[1] = int(max(region.centroid[1] - width // 2, 0))
        region.bbox[2] = int(min(region.centroid[0] + width // 2, nx))
        region.bbox[3] = int(min(region.centroid[1] + width // 2, ny))
        nb_pixels = np.sum(imLabel[region.bbox[0]:region.bbox[2],
                                   region.bbox[1]:region.bbox[3]] == 0)


# def getObjsInBlob(keys_cat, cat, sub_blob_mask, subimMUSE, subsegmap):
#     """

#     output
#     ------
#     listObjInBlob : list of simple indices (from 0 to nb of sources in
#     bounding box) of objects in the bounding box connected to the blob
#     listObjInBlob : list of catalog indices  of objects in the bounding
#     box connected to the blob
#     """
#     listObjInBlob = [0]
#     listHSTObjInBlob = ['bg']
#     labelHR = _getLabel(subsegmap)
#     nbSources = np.max(labelHR) + 1
#     listHST_ID = ['bg'] + [int(subsegmap[labelHR == k][0])
#                            for k in range(1, nbSources)]
#     for k in range(1, len(listHST_ID)):
#         if listHST_ID[k] in keys_cat:
#             row = cat.loc['ID', listHST_ID[k]]
#             center = (row['DEC'], row['RA'])
#             centerMUSE = subimMUSE.wcs.sky2pix([center], nearest=True)[0]
#             if sub_blob_mask[centerMUSE[0], centerMUSE[1]]:  # y,x
#                 listObjInBlob.append(k)
#                 listHSTObjInBlob.append(listHST_ID[k])
#     return listObjInBlob, listHSTObjInBlob


def getObjsInBlob(idname, cat, sub_blob_mask, subimMUSE, subsegmap):
    """Return the index and IDs of sources in the blobs.

    output
    ------
    listObjInBlob : list of simple indices (from 0 to nb of sources in
    bounding box) of objects in the bounding box connected to the blob
    listObjInBlob : list of catalog indices  of objects in the bounding
    box connected to the blob
    """
    listHST_ID = np.unique(subsegmap)
    listHST_ID = listHST_ID[listHST_ID > 0]

    subcat = cat.loc[idname, listHST_ID]
    center = np.array([subcat['DEC'], subcat['RA']]).T
    centerMUSE = subimMUSE.wcs.sky2pix(center, nearest=True).T
    idx = sub_blob_mask[centerMUSE[0], centerMUSE[1]]

    listObjInBlob = [0] + list(np.where(idx)[0] + 1)
    listHSTObjInBlob = ['bg'] + list(listHST_ID[idx])
    return listObjInBlob, listHSTObjInBlob


def convertBboxToHR(bbox, imHR, imLR):
    """
    Convert the bounding box from low resolution (MUSE) to high resolution
    (HST).
    """
    y0, x0, y1, x1 = bbox  # row then column
    pos = np.array([[y0, x0], [y1, x1]])  # still y,x (row,column)
    return imHR.wcs.sky2pix(imLR.wcs.pix2sky(pos), nearest=True).flatten()
