# -*- coding: utf-8 -*-
"""
@author: raphael.bacher@gipsa-lab.fr
"""
import os
from mpdaf.obj import Cube, Image, Spectrum
import numpy as np
from .parameters import Params
from .single_deblend import Deblending
from .eval_utils import calcCondNumber,calcXi2_tot,calcXi2_source
from astropy import table
from .deblend_utils import extractHST
from .grouping import convertBboxToHR,getObjsInBlob


def getInputs(cube, hstimages, segmap, blob_mask, bbox, imLabel, cat):
    """
    Extract data for each group before the multiprocessing (avoid large datasets copies)
    """
    subcube = cube[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
    copyHeaderWCSInfo(subcube) # need to copy some header infos
    
    subsegmap = extractHST(segmap, subcube[0])
    copyHeaderWCSInfo(subsegmap)
    
    subhstimages = [extractHST(hst, subcube[0]) for hst in hstimages]
    for im in subhstimages:
        copyHeaderWCSInfo(im)
    
    sub_blob_mask = blob_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    imMUSE = cube[0]

    subimMUSE = imMUSE[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    keys_cat = cat['ID']
    listObjInBlob,listHSTObjInBlob = getObjsInBlob(
        keys_cat,cat,sub_blob_mask, subimMUSE,subsegmap.data.filled(0))
    
    return subcube, subhstimages, subsegmap, listObjInBlob,listHSTObjInBlob


def getRes(debl, listObjInBlob, listHSTObjInBlob, group_id=0,write_dir=None):
    cond_number = calcCondNumber(debl, listObjInBlob)
    xi2_tot = calcXi2_tot(debl)
    dic_spec = debl.getsp()
    dic_spec[f'bg_{group_id}'] = dic_spec.pop('bg')
    
    # remove spectra from objects not in the blob
    listKToRemove=[]
    for k in dic_spec.keys():
        if k not in (listHSTObjInBlob+[f'bg_{group_id}']):
            listKToRemove.append(k)
    for k in listKToRemove:
        dic_spec.pop(k)
    
    # build sources table
    data_rows=[]
    for k in listObjInBlob:
        ID = debl.listHST_ID[k]
        xi2 = calcXi2_source(debl,k)
        data_rows.append((ID, xi2, cond_number,group_id)) 
    t = table.Table(rows=data_rows, names=('ID', 'Xi2','Condition Number','G_ID'))
    if write_dir is None:
        return t,dic_spec,debl.cube,debl.estimatedCube,group_id,cond_number,xi2_tot
    else:
        debl.cube.write(os.path.join(write_dir,"cube"+"_orig_%s"%(group_id) + ".fits"))
        debl.estimatedCube.write(os.path.join(write_dir,"cube"+"_estim_%s"%(group_id) + ".fits"))
        return t,dic_spec,group_id,cond_number,xi2_tot


def copyHeaderWCSInfo(new):
    """
    needed because wcs info are not completly copied during an image resize
    """
    new_header = new.wcs.to_header()
    for key in new_header:
        new.data_header[key] = new_header[key]

def deblendGroup(subcube,subhstimages,subsegmap,listObjInBlob,listHSTObjInBlob,group_id,write_dir=None ):    
    debl = Deblending(subcube,subhstimages)
    debl.createIntensityMap(subsegmap.data.filled(0.))
    debl.findSources()
    return(getRes(debl,listObjInBlob,listHSTObjInBlob,group_id,write_dir))

