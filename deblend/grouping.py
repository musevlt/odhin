"""
@author: raphael.bacher@gipsa-lab.fr

Store methods for 
- grouping sources to be deblended
- exploring groups
- modifying groups
"""
from skimage.measure import regionprops, label
from photutils import SegmentationImage
import matplotlib.patches as mpatches
from .deblend_utils import _getLabel,createIntensityMap,modifSegmap
import numpy as np
import multiprocessing
import tqdm


class SourceGroup():

    def __init__(self, GID, listSources, region):
        self.GID = GID
        self.listSources = listSources
        self.region = region #RegionAttr region
        self.nbSources = len(listSources)
        self.condNumber = 0
        self.Xi2 = 0

class RegionAttr():
    """
    Get region attributes from skimage region properties
    """
    def __init__(self, area, centroid, bbox):
        self.area = area
        self.centroid = centroid
        self.bbox = bbox
        
def getRegionAttr(sk_region):
    """
    build regionAttr from skimage region
    
    Intut:
    ------
    sk_region : skimage region
    
    Output:
    ------
    region : RegionAttr
    """
    region = RegionAttr(sk_region.area, sk_region.centroid, sk_region.bbox)
    return region
    
def doGrouping(cube, imHR, segmap, imMUSE, cat, kernel_transfert, params, cut,verbose=True):
    """
    Segment all sources in a number of connected (at the MUSE resolution) groups
    """
    segmap = modifSegmap(segmap, cat) # needed because of potential discrepancy between the catalog and the segmentation map (as in Rafelski15)
    intensityMapLRConvol = createIntensityMap(imHR, segmap, imMUSE, kernel_transfert, params)
    imLabel = label(intensityMapLRConvol > cut)
    listGroups=[]

    if verbose:
        ntasks = len(regionprops(imLabel))
        pbar = tqdm.tqdm(total=ntasks)
        
    keys_cat = cat['ID']
    listRegions = regionprops(imLabel)
    
    for i,sk_region in enumerate(listRegions):
        region = getRegionAttr(sk_region) 
        blob_mask = (imLabel==i+1)
        ensureMinimalBbox(region,params.min_width,imLabel,params.min_sky_pixels,params.margin_bbox)
        bbox = region.bbox # order is row0,column0,row1,column1
        
        bboxHR = convertBboxToHR(bbox,segmap,imMUSE)
        subsegmap = segmap.data[ bboxHR[0]:bboxHR[2],bboxHR[1]:bboxHR[3]]
        sub_blob_mask = blob_mask[bbox[0]:bbox[2],bbox[1]:bbox[3] ]
        subimMUSE = imMUSE[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        listSources = getObjsInBlob(keys_cat,cat,sub_blob_mask, subimMUSE,subsegmap)[1]
        listGroups.append(SourceGroup(GID=i,listSources=listSources,region=region))
        if verbose:
            pbar.update(1)
        
    return listGroups,imLabel

def ensureMinimalBbox(region,width,imLabel,min_sky_pixels,margin_bbox):
    """
    Ensures that region respects a minimal area and contains at least `min_sky_pixels` sky pixels 
    """
    # First add margin around bounding box
    bbox=[0,0,0,0] #init
    bbox[0] = int(np.maximum(region.bbox[0]-margin_bbox,0))
    bbox[1] = int(np.maximum(region.bbox[1]-margin_bbox,0))
    bbox[2] = int(np.minimum(region.bbox[2]+margin_bbox,imLabel.shape[0]))
    bbox[3] = int(np.minimum(region.bbox[3]+margin_bbox,imLabel.shape[1]))
    region.bbox=(bbox[0],bbox[1],bbox[2],bbox[3])
    region.area = (region.bbox[2]-region.bbox[0])*region.bbox[3]-region.bbox[1]
    
    if region.area < width**2: #then check minimal area
        bbox=[0,0,0,0]
        bbox[0] = int(np.maximum(region.centroid[0]-width//2,0))
        bbox[1] = int(np.maximum(region.centroid[1]-width//2,0))
        bbox[2] = int(np.minimum(region.centroid[0]+width//2,imLabel.shape[0]))
        bbox[3] = int(np.minimum(region.centroid[1]+width//2,imLabel.shape[1]))
        region.bbox=(bbox[0],bbox[1],bbox[2],bbox[3])
        region.area = (region.bbox[2]-region.bbox[0])*region.bbox[3]-region.bbox[1]
        
    nb_pixels = np.sum(imLabel[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]] == 0)
    while nb_pixels < min_sky_pixels: # then check minimal number of sky pixels
        width = width+1
        bbox=[0,0,0,0]
        bbox[0] = int(np.maximum(region.centroid[0]-width//2,0))
        bbox[1] = int(np.maximum(region.centroid[1]-width//2,0))
        bbox[2] = int(np.minimum(region.centroid[0]+width//2,imLabel.shape[0]))
        bbox[3] = int(np.minimum(region.centroid[1]+width//2,imLabel.shape[1]))
        nb_pixels = np.sum(imLabel[bbox[0]:bbox[2],bbox[1]:bbox[3]] == 0)
        region.bbox=(bbox[0],bbox[1],bbox[2],bbox[3])
        region.area = (region.bbox[2]-region.bbox[0])*region.bbox[3]-region.bbox[1]
            
    

def getObjsInBlob(keys_cat, cat, sub_blob_mask, subimMUSE, subsegmap):
    """
    
    output
    ------
    listObjInBlob : list of simple indices (from 0 to nb of sources in bounding box) of objects in the bounding box connected to the blob
    listObjInBlob : list of catalog indices  of objects in the bounding box connected to the blob
    """
    listObjInBlob = [0]
    listHSTObjInBlob = ['bg']
    labelHR = _getLabel(subsegmap)
    nbSources = np.max(labelHR)+1
    listHST_ID = ['bg'] + [int(subsegmap[labelHR == k][0])
                       for k in range(1, nbSources)]
    for k in range(1, len(listHST_ID)):
        if listHST_ID[k] in keys_cat:
            row = cat.loc['ID', listHST_ID[k]]
            center = (row['DEC'], row['RA'])
            centerMUSE = subimMUSE.wcs.sky2pix([center], nearest=True)[0]
            if sub_blob_mask[centerMUSE[0], centerMUSE[1]]: #y,x
                listObjInBlob.append(k)
                listHSTObjInBlob.append(listHST_ID[k])
    return listObjInBlob,listHSTObjInBlob
    

def convertBboxToHR(bbox,imHR,imLR):
    """
    convert the bounding box from low resolution (MUSE) to high resolution (HST)
    """
    y0,x0 = bbox[0],bbox[1] #row then column
    y1,x1 = bbox[2],bbox[3]
        
    bboxHR = imHR.wcs.sky2pix(imLR.wcs.pix2sky(np.array([[y0,x0],[y1,x1]])),nearest=True).flatten() #still y,x (row,column)
    
    return bboxHR


