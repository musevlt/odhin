# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:50:47 2016

@author: raphael
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:16:46 2016

@author: raphael
"""
import numpy as np
import numba

@numba.jit(nopython=True)
def getAbundance(imHR,dim):
    #simple case: (0,0) -> (0,0)
    di=imHR.shape[0]/float(dim[0])
    dj=imHR.shape[1]/float(dim[1])
    abundanceMap0=np.zeros((dim[0],imHR.shape[0]))
    abundanceMap1=np.zeros((dim[1],imHR.shape[1]))
    i=1

    for k in xrange(dim[0]):
        r=0
        i=i-1
        while r<np.floor(di):
            dr=min(1-abundanceMap0[max(0,k-1)][i],di-r)
            abundanceMap0[k][i]=dr
            r+=dr
            i+=1
        if (r<di) and (i<abundanceMap0.shape[1]):
            abundanceMap0[k][i]= di -r
    j=1
    for k in xrange(dim[1]):
        r=0
        j=j-1
        while r<np.floor(di):
            dr=min(1-abundanceMap1[max(0,k-1)][j],dj-r)
            abundanceMap1[k][j]=dr
            r+=dr
            j+=1
        if r<dj and j<abundanceMap1.shape[1]:
            abundanceMap1[k][j]= dj -r
    return abundanceMap0,abundanceMap1

@numba.jit(nopython=True)
def numba_outer(a,b):
    m = a.shape[0]
    n = b.shape[0]
    out=np.empty((m, n))
    for i in range(m):
        for j in range(n):
            out[i, j] = a[i]*b[j]
    return out

@numba.jit
def downsampling(cubeHR,dim,returnMatrix=False):
    if len(cubeHR.shape)==2:
        cubeHR=cubeHR[None,:,:]
    cubeLR=np.zeros((cubeHR.shape[0],dim[0],dim[1]))
    imHR=cubeHR[0]
    di=imHR.shape[0]/float(dim[0])
    dj=imHR.shape[1]/float(dim[1])
    abundanceMap0,abundanceMap1=getAbundance(imHR,dim)
    downsampleMatrix=np.zeros((imHR.shape[0],imHR.shape[1],cubeLR[0].size))
    for i in xrange(dim[0]):
        for j in xrange(dim[1]):
            downsampleMatrix[max(i*int(np.floor(di))-1,0):(i+1)*int(np.ceil(di))+1,max(0,j*int(np.floor(dj))-1):(j+1)*int(np.ceil(dj))+1,j+dim[0]*i]=numba_outer(abundanceMap0[i,max(0,i*int(np.floor(di))-1):(i+1)*int(np.ceil(di))+1],abundanceMap1[j,max(0,j*int(np.floor(dj))-1):(j+1)*int(np.ceil(dj))+1])
    for k,imHR in enumerate(cubeHR):
        for i in xrange(dim[0]):
            for j in xrange(dim[1]):
                #imLR[i,j]=np.sum(np.outer(abundanceMap0[i,i*int(np.floor(di))-1:(i+1)*int(np.ceil(di))+1],abundanceMap1[j,j*int(np.floor(dj))-1:(j+1)*int(np.ceil(dj))+1])*imHR[i*int(np.floor(di))-1:(i+1)*int(np.ceil(di))+1,j*int(np.floor(dj))-1:(j+1)*int(np.ceil(dj))+1])
                cubeLR[k,i,j]=np.sum(numba_outer(abundanceMap0[i,max(i*int(np.floor(di))-1,0):(i+1)*int(np.ceil(di))+1],abundanceMap1[j,max(0,j*int(np.floor(dj))-1):(j+1)*int(np.ceil(dj))+1])*imHR[max(0,i*int(np.floor(di))-1):(i+1)*int(np.ceil(di))+1,max(0,j*int(np.floor(dj))-1):(j+1)*int(np.ceil(dj))+1])

        downsampleMatrix=downsampleMatrix.reshape(imHR.size,cubeLR[0].size)
        #handle nan values
    cubeLR[np.where(np.isnan(cubeLR))]=np.median(cubeLR[np.where(np.isnan(cubeLR)==False)])
    if len(cubeHR.shape)==2:
        cubeLR=cubeLR[0,:,:]
    if returnMatrix==True:
        return cubeLR,downsampleMatrix,abundanceMap0.T,abundanceMap1.T

    return cubeLR

