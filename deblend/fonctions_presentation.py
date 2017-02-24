# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 07:06:46 2016

@author: raphael.bacher@gipsa-lab.fr
"""

from astropy.io import fits as pyfits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import astropy.io
plt.rcParams['image.cmap']="coolwarm"
plt.rcParams['image.origin']='lower'
plt.rcParams['image.interpolation']='nearest'
from matplotlib.colors import LogNorm
plt.rcParams['figure.figsize']=(15,15)
from mpdaf.obj import Cube,Image
from mpdaf.sdetect import Source
from deblend import main_deblending, deblend_utils

from skimage.measure import regionprops
from mpdaf.sdetect import Catalog
from astropy.table import Table
import math
from math import pi
from mpdaf.tools.astropycompat import zscale as plt_zscale

from scipy.interpolate import interp1d
import scipy.signal as ssl
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

def plotMainDeblend(debl,imMUSE=None,tab=None,saveDir='.',show=True,savePdf=False,name="simple",savePng=False,otherNeighbors=None,factorMUSE_HST=12500):
    if savePdf==True:
        plt.rc("text", usetex=False)

    ID=debl.src.ID
    if show==True:
        print '-'*40
        print "\n Objet %s \n"%ID

    with PdfPages(saveDir+'/deblend_%s_%s.pdf'%(name,ID)) as pdf:

        if tab is not None :
            if show==True:
                tab[tab['ID']==ID].pprint(max_width=-1)
            if savePdf==True:
                text=""
                for i,col in enumerate(list(tab[tab['ID']==ID][0])):
                    text=text+tab.colnames[i] + ' : ' + str(col)+"\n"
                fig=plt.figure(figsize=(15,3))
                plt.text(0.,0.,text,fontsize=15)
                plt.axis('off')
                pdf.savefig(fig)
                plt.clf()

        if debl.background==True:
            listHST_ID= [int(debl.segmap[debl.labelHR==k][0]) for k in xrange(1,debl.nbSources)]
        else:
            listHST_ID= [int(debl.segmap[debl.labelHR==k][0]) for k in xrange(1,debl.nbSources+1)]

        listNeighbors=debl.src.RAF_ID.split(",")
        listNeighbors=[int(c) for c in listNeighbors]


        #get central source
        if debl.segmap[debl.shapeHR[0]/2,debl.shapeHR[0]/2]==0:
            HST_ID_center=listNeighbors[0]
        else:
            HST_ID_center=debl.segmap[debl.shapeHR[0]/2,debl.shapeHR[0]/2]

        #get Main line pos and choose hst image of ref
        l_main=[x for (y, x) in sorted([(row['SNR_REF'],row['LBDA_OBS']) for row in debl.src.lines])][-1]
        if l_main < 7000:
            ref_hst='HST_F606W'
        elif l_main < 8000:
            ref_hst='HST_F775W'
        else:
            ref_hst='HST_F850LP'

        #remove source if not close to central source
        if otherNeighbors is None:
            otherNeighbors=[]
            for HST_ID in listHST_ID:
                if (np.sum(debl.segmap[30:-30,30:-30]==HST_ID)>0) and HST_ID not in listNeighbors:
                    otherNeighbors.append(HST_ID)



        #fig,ax=plt.subplots(len(otherNeighbors+listNeighbors)/3+2,3)
        fig=plt.figure()
        gs = gridspec.GridSpec(len(otherNeighbors+listNeighbors)/3+2,3)
        #fig.suptitle("HST Images and abundances map",fontsize=20)
        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[0, 1])
        ax2 = plt.subplot(gs[0, 2])
        if imMUSE is not None:
            imMUSE.plot(zscale=True,ax=ax0)
            start=imMUSE.wcs.sky2pix(debl.src.images['MASK_OBJ'].wcs.get_start())[0]
            rect = patches.Rectangle((start[1],start[0]),debl.src.images['MASK_OBJ'].shape[0],\
                                     debl.src.images['MASK_OBJ'].shape[1],linewidth=1,edgecolor='r',facecolor='none')
            ax0.add_patch(rect)

        ax1.imshow(debl.labelHR)
        for region in regionprops(debl.labelHR):
            ax1.annotate(
                listHST_ID[region.label-1],
                xy = (region.centroid[1],region.centroid[0]), xytext = (-20, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

        debl.src.images[ref_hst].plot(zscale=True,ax=ax2)
        ax2.contour(debl.labelHR,np.arange(np.max(debl.labelHR)+1), colors='black', linewidths=1)
        ax2.set_title(ref_hst)
#        if saveDir is not None:
#            if savePdf==True:
#                pdf.savefig(fig)
#            else:
#                fig.savefig(saveDir+'/src_desc_%s.png'%ID)
#
#        if show ==True:
#            plt.show()
#        else:
#            plt.clf()

        #plt.suptitle("Abundance maps",fontsize=20,y=0.8)
        for k,j in enumerate(otherNeighbors+listNeighbors):

            ax=plt.subplot(gs[k/3+1,k%3])
            ax.set_title("Abundance map src %s"%j)
            try:
                ax.imshow(debl.finalAbundanceMap[listHST_ID.index(j)].reshape(debl.shapeLR))
            except:
                ax.imshow(debl.listAbundanceMapLRConvol[0][0][listHST_ID.index(j)].reshape(debl.shapeLR))
        plt.tight_layout()

        if savePdf==True:
            pdf.savefig(fig)
        elif savePng==True:
            fig.savefig(saveDir+'/src_desc_%s_%s.png'%(name,ID),bbox_inches='tight')
        if show ==True:
            plt.show()
        else:
            plt.clf()

        fig=plt.figure(figsize=(10,5))
        plt.suptitle("Residuals", fontsize=20)
        plt.subplot(241)
        plt.title("White image MUSE")
        im_muse = np.sum(debl.cubeLR, axis=0)
        # factor=np.max(im_muse[5:-5,5:-5])
        factor = 1
        plt.imshow(im_muse[5:-5, 5:-5]/factor)
        plt.colorbar(fraction=0.046)
        plt.subplot(242)
        plt.title("White image estimated")
        plt.imshow(np.sum(debl.cubeRebuilt, axis=0)[5:-5, 5:-5]/factor)
        plt.colorbar(fraction=0.046)
        plt.subplot(243)
        plt.title("Residuals")
        plt.imshow(np.sum(debl.residus, axis=0)[5:-5, 5:-5]/factor,vmax=np.max(np.sum(debl.cubeLR, axis=0)[5:-5, 5:-5]))
        plt.colorbar(fraction=0.046)
#        if saveDir is not None:
#            if savePdf==True:
#                pdf.savefig(plt.gcf())
#            else:
#                plt.savefig(saveDir+'/src_residuals_main_%s.png'%ID)
#        if show ==True:
#            plt.show()
#        else:
#            plt.clf()
        if 'MUSE_CUBE_CONT' in debl.src.cubes.keys():
            plt.subplot(245)
            plt.title("Continuum Residuals")
            cubeLR_cont=debl.src.cubes['MUSE_CUBE_CONT'].data
            cubeRebuilt_cont=debl.cubeRebuiltCont
            plt.imshow(np.sum(cubeLR_cont-cubeRebuilt_cont,axis=0)[5:-5,5:-5]/factor)
            plt.colorbar(fraction=0.046)
            plt.subplot(246)
            plt.title("Line Residuals")
            plt.imshow(np.sum(debl.cubeLR-cubeLR_cont-debl.cubeRebuilt+cubeRebuilt_cont,axis=0)[5:-5,5:-5]/factor)
            plt.colorbar(fraction=0.046)
            plt.subplot(247)
            plt.title("Residuals MUSE-HST775")
            hst_im=debl.listImagesHR[1].copy()
            hst_im.data=ssl.fftconvolve(hst_im.data,debl.listPSF[7],mode='same')
            hst_im=hst_im.resample(debl.shapeLR,debl.wcs.get_start()+debl.shift,
                                   newstep=0.2,flux=True,antialias=False).data
            muse_im=np.sum(debl.cubeLR*debl.filtResp[1][:,np.newaxis,np.newaxis],axis=0)
            #factorMUSE_HST=np.sum(muse_im[10:-10,10:-10])/np.sum(hst_im[10:-10,10:-10])
            plt.imshow((muse_im[5:-5,5:-5]-hst_im[5:-5,5:-5]*factorMUSE_HST)/np.max(muse_im[5:-5,5:-5]) )

            plt.colorbar(fraction=0.046)
            plt.subplot(248)
            plt.title("Residuals Rebuilt-HST775")
            rebuilt_im=np.sum(debl.cubeRebuilt*debl.filtResp[1][:,np.newaxis,np.newaxis],axis=0)
            plt.imshow((rebuilt_im[5:-5,5:-5]-hst_im[5:-5,5:-5]*factorMUSE_HST)/np.max(muse_im[5:-5,5:-5]) )

            plt.colorbar(fraction=0.046)
        plt.tight_layout()
        if savePdf==True:
            pdf.savefig(fig)
        elif savePng==True:

            fig.savefig(saveDir+'/src_residuals_%s_%s.png'%(name,ID),bbox_inches='tight')
        if show ==True:
            plt.show()
        else:
            plt.clf()



        listNB=[im.replace('NB_','') for im in debl.src.images.keys() if "NB" in im]
        listNB=[im for im in listNB if (im in debl.src.lines['LINE']) or (im=='OII3727')]
        listNB=listNB[:5]#limit number of NB images
        listLines=[]
        for line in listNB:
            if line=='OII3727':#inconsistency between NB and lines
                listLines.append(debl.src.lines[debl.src.lines['LINE']=="OII3726"]['LBDA_OBS'][0])
            else:
                listLines.append(debl.src.lines[debl.src.lines['LINE']==line]['LBDA_OBS'][0])
        listNB=[x for (y, x) in sorted(zip(listLines, listNB))]
        listLines=sorted(listLines)
        debl.src.lines['LBDA_OBS']


        fig,ax=plt.subplots(1+len(otherNeighbors),3, gridspec_kw = {'width_ratios':[3, 1,1]},figsize=(20,4*(len(otherNeighbors)+1)))
        refSpec=debl.src.header['REFSPEC']
        #fig.suptitle("Spectra",fontsize=20)
        if len(otherNeighbors)==0:
            debl.src.spectra[refSpec].convolve(np.ones((5,))/5).plot(label=refSpec,ax=ax[0])
            for line in listLines:
                ax[0].axvline(line,linestyle='--',c='r')

            debl.src.spectra[refSpec].subspec(l_main-40,l_main+40).plot(ax=ax[1],label=refSpec)

            spectraTot=debl.getsp()
            for i in listNeighbors:
                spectraTot[i].convolve(np.ones((5,))/5).plot(ax=ax[0],label=i)
                ax[0].legend()
                for line in listLines:
                    ax[0].axvline(line,linestyle='--',c='r')
                spectraTot[i].subspec(l_main-40,l_main+40).plot(ax=ax[1],label=i)
            ax[2].imshow(debl.labelHR)
            for region in regionprops(debl.labelHR):
                if listHST_ID[region.label-1] in listNeighbors:
                    ax[2].annotate(
                        listHST_ID[region.label-1],
                        xy = (region.centroid[1],region.centroid[0]), xytext = (-20, 20),
                        textcoords = 'offset points', ha = 'right', va = 'bottom',
                        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        else:
            debl.src.spectra[refSpec].plot(label=refSpec,ax=ax[0,0])
            for line in listLines:
                ax[0,0].axvline(line,linestyle='--',c='r')

            debl.src.spectra[refSpec].subspec(l_main-40,l_main+40).plot(ax=ax[0,1],label=refSpec)

            spectraTot=debl.getsp()
            for i in listNeighbors:
                spectraTot[i].plot(ax=ax[0,0],label=i)
                ax[0,0].legend()
                for line in listLines:
                    ax[0,0].axvline(line,linestyle='--',c='r')
                spectraTot[i].subspec(l_main-40,l_main+40).plot(ax=ax[0,1],label=i)
            ax[0,2].imshow(debl.labelHR)
            for region in regionprops(debl.labelHR):
                if listHST_ID[region.label-1] in listNeighbors:
                    ax[0,2].annotate(
                        listHST_ID[region.label-1],
                        xy = (region.centroid[1],region.centroid[0]), xytext = (-20, 20),
                        textcoords = 'offset points', ha = 'right', va = 'bottom',
                        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

        if len(otherNeighbors)>0:
            #fig,ax1=plt.subplots(len(otherNeighbors),3,gridspec_kw = {'width_ratios':[3, 1,1]},figsize=(15,4*len(otherNeighbors)))
            for n,i in enumerate(otherNeighbors):

                spectraTot[i].plot(ax=ax[n+1,0],label=i)
                for line in listLines:
                    ax[n+1,0].axvline(line,linestyle='--',c='r')
                    ax[n+1,0].legend()
                spectraTot[i].subspec(l_main-40,l_main+40).plot(ax=ax[n+1,1],label=i)
                ax[n+1,2].imshow(debl.labelHR)
                for region in regionprops(debl.labelHR):
                    if listHST_ID[region.label-1]==i:
                        ax[n+1,2].annotate(
                            listHST_ID[region.label-1],
                            xy = (region.centroid[1],region.centroid[0]), xytext = (-20, 20),
                            textcoords = 'offset points', ha = 'right', va = 'bottom',
                            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        if savePdf==True:
            pdf.savefig(fig)
        elif savePng==True:
            fig.savefig(saveDir+'/src_spectra_%s_%s.png'%(name,ID),bbox_inches='tight')
        if show ==True:
            plt.show()
        else:
            plt.clf()

        if len(listNB)>0:

#            if len(listNB)<3:
#                plt.figure(figsize=(5,5))
#            else:
            plt.figure(figsize=(5*len(listNB),10))
            gs = gridspec.GridSpec(2,len(listNB),wspace=0.,hspace=0.2)
            #plt.suptitle("Narrow band Images",fontsize=20)
            for i in xrange(len(listNB)):
                width=int(debl.src.tables['NB_PAR'][debl.src.tables['NB_PAR']['LINE']=='NB_'+listNB[i]]['WIDTH'][0]/1.25)
                l=int(debl.wave.pixel(listLines[i],nearest=True))
                ax0 = plt.subplot(gs[0, i])
                #plt.subplot(2,len(listNB),i+1)
                ax0.set_title('NB_'+listNB[i])
                debl.src.images['NB_'+listNB[i]].plot(zscale=True,ax=ax0,colorbar='v')
                vmin, vmax = plt_zscale(debl.src.images['NB_'+listNB[i]].data)
                ax1 = plt.subplot(gs[1, i])
#                plt.subplot(2,len(listNB),1+i+len(listNB))
                ax1.set_title('NB_'+listNB[i])
                im=np.sum(debl.cubeRebuilt[l-width:l+width],axis=0)-np.sum(debl.cubeRebuiltCont[l-width:l+width],axis=0)
                #vmin, vmax = plt_zscale(im)
                im0=ax1.imshow(im,vmin=vmin, vmax=vmax)
                plt.colorbar(im0,ax=ax1)

            #plt.tight_layout()
            if savePdf==True:
                pdf.savefig(plt.gcf())
            elif savePng==True:
                plt.savefig(saveDir+'/src_NBimages_%s_%s.png'%(name,ID),bbox_inches='tight')
            if show ==True:
                plt.show()
            else:
                plt.clf()

    plt.close('all')



def convertFilt(filt,cube=None,x=None):
    """
    Resample response of HST filter on the spectral grid of MUSE

    filt: 2d array (wavelength,amplitude) of response filter
    cube: mpdaf cube, provides the wave sampling grid
    """
    if cube is not None:
        x = cube.wave.coord()
    f=interp1d(filt[:,0],filt[:,1],fill_value=0,bounds_error=False)
    return np.array(f(x))

