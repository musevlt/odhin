# -*- coding: utf-8 -*-
"""
@author: raphael.bacher@gipsa-lab.fr
"""

from .single_deblend import Deblending
from .deblend_utils import extractHST
from .grouping import getObjsInBlob


def getInputs(cube, hstimages, segmap, blob_mask, bbox, imLabel, cat):
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


def deblendGroup(subcube, subhstimages, subsegmap, listObjInBlob,
                 listHSTObjInBlob, group_id, outfile):
    debl = Deblending(subcube, subhstimages)
    debl.createIntensityMap(subsegmap.data.filled(0.))
    debl.findSources()
    debl.write(outfile, listObjInBlob, listHSTObjInBlob, group_id)
