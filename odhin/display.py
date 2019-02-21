# module to display deblend results
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from muse_analysis.udf.sources import UDFSource
from mpdaf.obj import Image,Spectrum,Cube
from mpdaf.sdetect import Catalog
from itertools import cycle

#plt.rcParams[''] =

class Display(object):
    def __init__(self, tsrc=None, dr1=None, minconfid=1, rafelski=None, hstima=None):
        if dr1 is None:
            dr1 = Catalog.read('/muse/UDF/public/catalogs/DR1/udf10_dr1.team.20170829.fits')
        dr1 = dr1[dr1['CONFID']>minconfid]
        ref_ids = []
        ref_rafids = []
        for e in dr1:
            for r in e['RAF_ID'].split(','):
                ref_ids.append(e['ID'])
                ref_rafids.append(int(r))
        self.dr1 = dr1
        self.dr1_ids = ref_ids
        self.dr1_rafids = ref_rafids
        if rafelski is None:
            self.rafelski = Catalog.read('/muse/UDF/public/catalogs/udf10/hst/udf10_uvudf_rafelski_2015.fits')
        if hstima is None:
            hstdir = '/muse/UDF/private/HST/'
            self.imHST = Image(hstdir+"hlsp_xdf_hst_acswfc-30mas_hudf_f775w_v1_sci.fits")
            self.segmap = Image(hstdir+"SegMap/segmentation_map_rafelski_2015.fits")
        if tsrc is None:
            self.tsrc = Catalog.read('table_sources.fits')
            self.tgrp = Catalog.read('table_groups.vot')

    def find_museid(self, rafid):
        if rafid not in self.dr1_rafids:
            return None
        return self.dr1_ids[self.dr1_rafids.index(rafid)]

    def add_museid(self):
        self.tsrc['MUSE_ID'] = 0
        for s in self.tsrc:
            if s['ID'] == 'bg':
                continue
            iden = self.find_museid(int(s['ID']))   
            if iden is not None:
                s['MUSE_ID'] = iden

    def select_groups_with_museid(self):
        grp_wmuse = np.unique(self.tsrc[self.tsrc['MUSE_ID']>0]['G_ID'])
        tgrp_wmuse = self.tgrp[np.in1d(self.tgrp['G_ID'],grp_wmuse)]
        gids = []
        ns = []
        areas = []
        conds = []
        idmuses = []
        zmuses = []
        abmuses = []
        tmuses = []
        for g in tgrp_wmuse:
            museiden = []
            zm = []
            tm = []
            ab = []
            for raf in g['listIDs'].replace(')','').split(',')[1:]:
                iden = self.find_museid(int(raf))
                if iden is not None:
                    museiden.append(iden)
                    r = self.dr1[self.dr1['ID']==iden][0]
                    zm.append(np.around(r['Z_MUSE'],2))
                    tm.append(r['TYPE'])
                    ab.append(np.around(r['HST_F775W'],2))
            gids.append(g['G_ID'])
            ns.append(g['nbSources']-1)
            areas.append(g['Area'])
            conds.append(g['Condition_Number'])
            idmuses.append(museiden)
            zmuses.append(zm)
            tmuses.append(tm)
            abmuses.append(ab)
        ntab = Catalog(data=[gids,ns,areas,conds,idmuses,tmuses,zmuses,abmuses], names=['G_ID','NS','AREA','COND','MUSE_ID','MUSE_TYPE','MUSE_Z','MUSE_MAG'])
        ntab['COND'].format = '.2f'
        return ntab

    def display2(self, fig, gid, cat, refspec=['MUSE_PSF_SKYSUB','MUSE_PSF_SKYSUB'], lzoom=[None,None,None], dl=50, showima=True):
        if lzoom is None:
            lzoom = [None,None,None]
        elif isinstance(lzoom, (int,float)):
            lzoom = [lzoom,lzoom,lzoom]
        r = cat[cat['G_ID']==gid][0]
        muse_id = r['MUSE_ID']
        t = self.tsrc[self.tsrc['G_ID']==gid]
        dirg = f'groups/Group{gid:05d}'
        splist = [[raf,Spectrum(f"{dirg}/sp{int(raf):05d}.fits")] for raf in t['ID'] if raf != 'bg']
        splist.append(['bg',Spectrum(f"{dirg}/bg.fits")])
        if not showima:
            gs = gridspec.GridSpec(2,6)
            axd1 = fig.add_subplot(gs[0,0:2])
            axm1 = fig.add_subplot(gs[1,0:2], sharey=axd1, sharex=axd1)
            axd2 = fig.add_subplot(gs[0,2:4])
            axm2 = fig.add_subplot(gs[1,2:4], sharey=axd2, sharex=axd2)
            axdb = fig.add_subplot(gs[0,4:6]) 
            axdict = dict(axd1=axd1, axm1=axm1, axd2=axd2, axm2=axm2, axdb=axdb)
        else:
            gs = gridspec.GridSpec(3,6)
            axd1 = fig.add_subplot(gs[0,0:2])
            axm1 = fig.add_subplot(gs[1,0:2], sharey=axd1, sharex=axd1)
            axd2 = fig.add_subplot(gs[0,2:4])
            axm2 = fig.add_subplot(gs[1,2:4], sharey=axd2, sharex=axd2)
            axdb = fig.add_subplot(gs[2,0:2]) 
            axh1 = fig.add_subplot(gs[0:2,4:6]) 
            axob = fig.add_subplot(gs[2,2]) 
            axfi = fig.add_subplot(gs[2,3]) 
            axre = fig.add_subplot(gs[2,4]) 
            axh2 = fig.add_subplot(gs[2,5])
            axdict = dict(axd1=axd1, axm1=axm1, axd2=axd2, axm2=axm2, axdb=axdb,
                           axh1=axh1, axob=axob, axfi=axfi, axre=axre, axh2=axh2)
        for ax,(raf,sp),lz in zip([axd1,axd2,axdb],splist,lzoom):
            _plot_spec(ax, sp, lz, dl, f'DEBLEND RAF_ID {raf}')
        srclist = [[int(raf),_get_udfsource(int(raf))] for raf in t['ID'] if raf != 'bg']
        for ax,(raf,src),lz,ref in zip([axm1,axm2],srclist,lzoom[0:2],refspec):
            _plot_spec(ax, src.spectra[ref], lz, dl, f'MUSE MUSE_ID {self.find_museid(raf)}')
        if showima:
            obs = Cube(f'{dirg}/obscube.fits')
            imaobs = obs.mean(axis=0)
            _plot_ima(axob, imaobs, catalog=self.rafelski, label=False, text='Obs')
            fit = Cube(f'{dirg}/fitcube.fits')
            imafit = fit.mean(axis=0)
            _plot_ima(axfi, imafit, catalog=self.rafelski, label=False, text='Fit')
            res = obs - fit
            imares = res.mean(axis=0)
            _plot_ima(axre, imares, catalog=self.rafelski, label=False, text='Residual')
            srange = imaobs.wcs.get_range()
            center = (0.5*(srange[0]+srange[2]), 0.5*(srange[1]+srange[3]))
            size = imaobs.shape[0]*0.2
            hst = self.imHST.subimage(center,size=size,minsize=0)
            hst2 = self.imHST.subimage(center,size=2*size,minsize=0)
            seg = self.segmap.subimage(center,size=size,minsize=0)
            _plot_ima(axh2, hst2, zscale=True, catalog=self.rafelski, label=False, text='F775W')
            _plot_ima(axh1, hst, catalog=self.rafelski, segmap=seg, text='F775W') 
        fig.suptitle(f"Grp {gid} Ns {r['NS']} Area {r['AREA']:.0f} Cond {r['COND']:.1f} MUSE ID {r['MUSE_ID']} Type {r['MUSE_TYPE']} Z {r['MUSE_Z']} Mag {r['MUSE_MAG']}",
                fontsize=10)
        return axdict

def _plot_ima(ax, ima, segmap=None, cuts=(None,None), catalog=None, label=True, cmap='Greys', scale='arcsinh', zscale=False, text=None,
              fontsize=8):
    ima.plot(ax=ax, cmap=cmap, vmin=cuts[0], vmax=cuts[1], scale=scale, zscale=zscale)
    if segmap is not None:
        cycol = cycle('bgrcmy')
        scat = catalog.select(ima.wcs)
        for iden in scat['ID']:
            _show_mask(ax, segmap, levels=[iden - 0.1, iden + 0.1],
                      col=next(cycol), alpha=0.5, surface=True)
    if catalog is not None:
        catalog.plot_symb(ax, ima.wcs, label=label, esize=0.1, ecol='r')
    if text is not None:
        ax.set_title(text, fontsize=fontsize)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def _plot_spec(ax, sp, lz=None, dl=50, text=None, tloc=(0.05,0.90), fontsize=8):
    if lz is not None:
        sp.plot(ax=ax, noise=True, lmin=lz-dl, lmax=lz+dl)
    else:
        sp.plot(ax=ax, noise=True)
    ax.set_xlabel('')
    ax.set_ylabel('')
    if text is not None:
        ax.text(tloc[0], tloc[1], text, ha='left', transform=ax.transAxes,fontsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)

def _get_udfsource(rafid):
    dirsp = '/muse/UDF/private/analysis/MUSEX/export/muse-udf-10/hstprior_udf10/sources'
    src = UDFSource.from_file(f'{dirsp}/source-{rafid:05d}.fits')
    return src

def _show_mask(ax, segmap, col='r', levels=[0], alpha=0.4, surface=False):
    if surface:
        ax.contourf(segmap._data.astype(float), levels=levels, origin='lower', colors=col,
                    alpha=alpha, extent=[-0.5, segmap.shape[0] - 0.5, -0.5, segmap.shape[1] - 0.5])
    else:
        ax.contour(segmap._data.astype(float), levels=levels,
                   origin='lower', colors=col, alpha=alpha)


