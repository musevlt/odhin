# -*- coding: utf-8 -*-

import numpy as np

import scipy.signal as ssl
import scipy.stats as sst
import itertools
import sklearn.linear_model as sklm
import sklearn.model_selection as skms
from scipy.ndimage.morphology import binary_dilation,grey_dilation
import numba
import time

def glasso_bic(X,Y,ng=2,nonneg=False,intercept=True,multivar=True,listMask=None,
               returnCriterion=False, greedy=True,
               averaged=True,maskOnly=False):
    coeff=np.zeros((X.shape[1],Y.shape[1]))
    criterion=[]
    intercepts = np.zeros((1,Y.shape[1]))
    if listMask is None:
        for k in xrange(Y.shape[1]):
            res= lasso_bic(X,Y[:,np.maximum(0,k-ng):k+ng+1],intercept=intercept,
                               multivar=multivar,averaged=averaged,greedy=greedy)
            coeff[:,k]= res[0][:,np.minimum(k,ng)]
            intercepts[:,k]=res[1][np.minimum(k,ng)]
            criterion.append(res[2])
    else:
        for mask in listMask:
            res=lasso_bic(X,Y[:,mask],intercept=intercept,multivar=multivar,
                          averaged=averaged,greedy=greedy)
            coeff[:,mask]=res[0]
            intercepts[:,mask]=res[1]
            criterion.append(res[2])
        if maskOnly==False:
            if listMask==[]:
                listMask=[[False for l in xrange(coeff.shape[1])]]
            for l in np.nonzero(~np.sum(listMask,axis=0).astype(bool))[0]:
                #use ng following slices for robustness of support

                res=lasso_bic(X,Y[:,np.maximum(l-ng,0):l+ng+1],intercept=intercept,
                                  multivar=multivar,averaged=averaged,greedy=greedy)
                coeff[:,l]=res[0][:,np.minimum(l,ng)]
                intercepts[:,l]=res[1][np.minimum(l,ng)]
                criterion.append(res[2])

    if returnCriterion:
        return coeff,intercepts,criterion
    return coeff,intercepts


def lasso_bic(X,Y,criterion='bic',intercept=True,multivar=True,greedy=False,averaged=True):
    if averaged==True: #work on averaged data for model selection
        Y_all=Y.copy()
        Y=np.mean(Y,axis=1)[:,None]

    n_samples = X.shape[0]
    n_models = X.shape[1]
    n_targets = Y.shape[1]
    coef_path_ = []#np.zeros((n_models,n_targets,n_models+1))
    listComb=[]
    if greedy==False:
        for k in xrange(1,n_models+1):
            listComb+=[i for i in itertools.combinations(np.arange(n_models), k)]
    else:
        listComb=[[]]
        listModels=range(n_models)
        lprod=[np.mean(np.dot(X[:,i],Y)) for i in listModels]
        a=np.argmax(np.abs(lprod))
        residuals=Y-(lprod[a]/np.linalg.norm(X[:,a])*X[:,a])[:,None]
        listModels.pop(a)
        listComb[0]=[a]
        for k in xrange(1,n_models):
            lprod=[np.mean(np.dot(X[:,i],residuals)) for i in listModels]
            a=np.argmax(np.abs(lprod))
            a_m=listModels[a]
            residuals=Y-(lprod[a]/np.linalg.norm(X[:,a_m])*X[:,a_m])[:,None]
            listModels.pop(a)
            listComb.append(listComb[k-1]+[a_m])


    if intercept == True:
        X_offset = np.average(X, axis=0)
        Y_offset = np.average(Y, axis=0)
        X=X-X_offset
        Y= Y-Y_offset
        #X,Y,X_offset,Y_offset,X_scale=sklm.base._preprocess_data(X,Y,fit_intercept=True)

    for ind in listComb:
        coef_path_.append(np.linalg.lstsq(X[:,ind],Y)[0])

    if criterion == 'aic':
        K = 2  # AIC
    elif criterion == 'bic':
        K = np.log(n_samples*n_targets)  # BIC
    else:
        raise ValueError('criterion should be either bic or aic')



    mean_squared_error=[]
    for k in xrange(len(coef_path_)):
        R = Y - np.dot(X[:,listComb[k]], coef_path_[k])  # residuals
        mean_squared_error.append(np.mean(R ** 2,axis=0))
        #mean_squared_error.append(np.mean(R ** 2))
    mean_squared_error=np.array(mean_squared_error)
    df = np.zeros(len(coef_path_), dtype=np.int)  # Degrees of freedom
    for k, coef in enumerate(coef_path_):
        # get the number of degrees of freedom equal to:

        df[k] = coef.size
        if multivar:
            df[k]=df[k]+n_targets
        else:
            df[k]=df[k]+1
        if intercept == True:
            df[k] = df[k]+n_targets

    if multivar==True:
        criterion_ = n_samples * np.sum(np.log(mean_squared_error),axis=1) + K * df
    else:
        criterion_ = n_samples * n_targets* np.log(np.mean(mean_squared_error,axis=1)) + K * df

    n_best = np.argmin(criterion_)

    if multivar==True:

        r0=n_samples * np.sum(np.log(np.mean(Y**2,axis=0)))+K*n_targets
    else:
        r0=n_samples * n_targets * np.log(np.mean(Y**2))+K
    if intercept:
        r0=r0+n_targets*K

    if averaged==True: # get back to whole dataset
        Y=Y_all
        n_targets=Y.shape[1]
        if intercept == True:
            Y_offset = np.average(Y, axis=0)
            Y=Y-Y_offset


    coeff = np.zeros((n_models,n_targets))

    if criterion_[n_best] < r0: # if not, all regressors stay at 0
        if averaged:
            coeff[listComb[n_best],:] = np.linalg.lstsq(X[:,listComb[n_best]],Y)[0]
        else:
            coeff[listComb[n_best],:] = coef_path_[n_best]


    if intercept == True:
        intercepts = Y_offset - np.dot(X_offset, coeff)
    else:
        intercepts=np.zeros_like(Y[0])

    return coeff,intercepts,np.concatenate([np.array([r0]),criterion_])

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

def getLinesSupportList(listSpe,w=10,wmin=1,wmax=20,alpha=1.4,beta=1.2,
                       n_sig=1,f=0.6,f2=0.7,returnAll=False,filt=None,localConstraint=True):
    """

    """
    if filt is None:
        filt=sst.norm.pdf(np.linspace(-2*w,2*w,4*w+1),scale=w)
        fitl=filt/np.linalg.norm(filt)
    listMask=[]
    for l in xrange(len(listSpe)):
        spe=listSpe[l]
        sig=1.489*mad(spe)
        spe_filt=ssl.fftconvolve(spe,filt,mode='same')
        sig_filt=1.489*mad(spe_filt)
        med_filt=np.median(spe_filt)
        lRejected=0



        mask0 = np.ones(2*w+1 )
        mask0[w]= 0
        B= grey_dilation(np.abs(spe_filt),footprint=mask0)
        listArgExtrema = np.nonzero(np.abs(spe_filt)>B)[0]

        #listArgMax=ssl.argrelmax(spe_filt,order=w)[0]
        #listArgMin=ssl.argrelmin(spe_filt,order=w)[0]


        #listMax=spe_filt[listArgMax]
        #listMin=spe_filt[listArgMin]

        #listArgExtrema = [x for (y,x) in sorted(zip(np.concatenate([listMax,np.abs(listMin)]),
        #                                            np.concatenate([listArgMax,listArgMin])), key=lambda pair: pair[0],
        #                                        reverse=True)]
        listExtrema = spe_filt[listArgExtrema]
        listKernel=genKernels(listWidth=np.concatenate([np.array([0.1]),np.arange(1,2*wmax+2,2)]),n=2*wmax+1)
        nThresh=np.sum(np.abs(spe_filt[listArgExtrema])>alpha*sig_filt)
        for k,m in zip(listArgExtrema,listExtrema):
            if (np.abs(spe_filt[k])>alpha*sig_filt) and ((localConstraint==False) or (spe[np.maximum(k-1,0):k+2]>np.sign(spe[k])*sig).all()):
                mask=np.zeros_like(spe).astype(bool)
                kmin=np.maximum(k-wmax,0)
                if k-wmax<0:
                    line=np.concatenate([np.zeros(wmax-k),spe[kmin:k+wmax+1]])
                elif wmax+k+1>len(spe):
                    line=np.concatenate([spe[kmin:k+wmax+1],np.zeros(k+wmax+1-len(spe))])
                else:
                    line=spe[kmin:k+wmax+1]

                line=line/np.linalg.norm(line)
                width=calcWidth(line,listKernel=listKernel,n_sig=n_sig,
                                listWidth=np.concatenate([np.array([0.1]),np.arange(1,2*wmax+2,2)]))
                width=int(width)
                if width>=2*wmin+1:
                    #a,b=k-width/2,k+width/2+1
                    if len(np.nonzero(spe[np.maximum(k-width,0):k]<beta*sig)[0])>0:
                        a=np.maximum(k-width,0)+np.nonzero(spe[np.maximum(k-width,0):k]<beta*sig)[0][-1]
                    else:
                        a=np.maximum(k-width,0)
                    if len(np.nonzero(spe[k:k+width+1]<beta*sig)[0])>0:
                        b=k+np.nonzero(spe[k:k+width+1]<beta*sig)[0][0]
                    else:
                        b=k+width+1
                else:
                    lRejected+=1
                    continue

                if np.any([np.sum(x[a:b])>(b-a)*f for x in listMask]):
                    #if already covered at more than a fraction f by an existing mask don't add current mask
                    continue
                #if new mask contains more than fraction f2 of old mask, remove old mask
#                 listOldMask=[listMask[m].all() for m in np.nonzero([np.sum(x[a:b])>f2*np.sum(x) for x in listMask])[0]]

#                 for oldMask in listOldMask:
#                     listMask.remove(oldMask)
                mask[a:b]=True
                listMask.append(mask)
    if returnAll==True:
        #return listMask,listArgExtrema,listExtrema,nThresh,lRejected,len(listExtrema)
        return listMask,lRejected,len(listExtrema),nThresh,listExtrema,listArgExtrema,spe_filt,sig_filt,sig
    return listMask

def genKernels(listWidth=np.arange(5,42,2),n=41,n_sig=2):
    listKernel=[]
    x=np.linspace(-20,20,n)
    for k in listWidth:
        g=sst.norm.pdf(x,scale=k/(n_sig*2.))
        listKernel.append(g/np.linalg.norm(g))
    return listKernel

def calcWidth(spe,listKernel=None,n_sig=1,listWidth=np.arange(5,42,2)):

    if listKernel is None:
        listKernel=[]
        x=np.linspace(-20,20,listWidth[-1])
        for k in listWidth:
            g=sst.norm.pdf(x,scale=k/(n_sig*2.))
            listKernel.append(g/np.linalg.norm(g))
    listCorr=[]
    for g in listKernel:
        listCorr.append(np.dot(spe,g))
    res=listWidth[np.argmax(np.abs(listCorr))]
    return res

def oneSigRule(LRCV):
    ind=np.argmin(np.mean(LRCV.mse_path_,axis=1))
    min_mse=np.mean(LRCV.mse_path_,axis=1)[ind]
    if type(LRCV.cv)==int:
        cv=LRCV.cv
    elif type(LRCV.cv)==skms._split.KFold:
        cv=LRCV.cv.n_splits
    min_mse_std=np.std(LRCV.mse_path_,axis=1)[ind]/np.sqrt(cv)
    return np.max([LRCV.alphas_[i] for i in xrange(len(LRCV.alphas_)) if np.mean(LRCV.mse_path_[i])<min_mse+min_mse_std])

def oneSigRuleRidge(LRCV):
    ind=np.argmin(np.mean(np.mean(LRCV.cv_values_,axis=0),axis=0))
    min_mse=np.mean(np.mean(LRCV.cv_values_,axis=0),axis=0)[ind]
    cv=LRCV.cv_values_.shape[0]*LRCV.cv_values_.shape[1]
    min_mse_std=np.std(np.std(LRCV.cv_values_,axis=0),axis=0)[ind]/np.sqrt(cv)
    return np.max([LRCV.alphas[i] for i in xrange(len(LRCV.alphas)) if np.mean(LRCV.cv_values_[:,:,i])<min_mse+min_mse_std])


def glasso_cv(X,Y,ng=9,cv=10,n_alphas=100,eps=1e-3,recompute=True,oneSig=True,
              listMask=None,returnAlpha=False,intercept=True,maskOnly=False):

    coeff=np.zeros((X.shape[1],Y.shape[1]))
    intercepts=np.zeros((1,Y.shape[1]))
    alphas_ = []
    #LAMTCV_l_slid_alphas=[]
    kf=skms.KFold(n_splits=cv,shuffle=True)
    LAMTCV_slid=sklm.MultiTaskLassoCV(n_alphas=n_alphas,eps=eps,n_jobs=1,cv=kf,fit_intercept=intercept)
    if listMask is not None:
        for mask in listMask:
            LAMTCV_slid.fit(X,Y[:,mask])
            if oneSig == True:
                alpha=oneSigRule(LAMTCV_slid)
                LAMT_slid=sklm.MultiTaskLasso(alpha=alpha,fit_intercept=intercept)
                LAMT_slid.fit(X,Y[:,mask])
                coeff[:,mask]=LAMT_slid.coef_.T
                if intercept:
                    intercepts[:,mask]=LAMT_slid.intercept_.T
                alphas_.append(alpha)
            else:
                coeff[:,mask]=LAMTCV_slid.coef_.T
                if intercept:
                    intercepts[:,mask]=LAMTCV_slid.intercept_.T
                alphas_.append(LAMTCV_slid.alpha_)
        if maskOnly==False:
            if listMask==[]:
                listMask=[[False for l in xrange(coeff.shape[1])]]
            for l in np.nonzero(~np.sum(listMask,axis=0).astype(bool))[0]:
                LAMTCV_slid.fit(X,Y[:,np.maximum(l-ng,0):l+ng+1])
                if oneSig == True:
                    alpha=oneSigRule(LAMTCV_slid)

                    LAMT_slid=sklm.MultiTaskLasso(alpha=alpha,fit_intercept=intercept)
                    LAMT_slid.fit(X,Y[:,np.maximum(l-ng,0):l+ng+1])
                    coeff[:,l]=LAMT_slid.coef_.T[:,np.minimum(ng,l)]
                    if intercept:
                        intercepts[:,l]=LAMT_slid.intercept_.T[np.minimum(ng,l)]
                    alphas_.append(alpha)
                else:
                    coeff[:,l]=LAMTCV_slid.coef_.T[:,np.minimum(ng,l)]
                    if intercept:
                        intercepts[:,l]=LAMTCV_slid.intercept_.T[np.minimum(ng,l)]
                    alphas_.append(LAMTCV_slid.alpha_)
    else:
        for k in xrange(Y.shape[1]):
            LAMTCV_slid.fit(X,Y[:,np.maximum(k-ng,0):k+1+ng])
            if oneSig == True:
                alpha=oneSigRule(LAMTCV_slid)
                LAMT_slid=sklm.MultiTaskLasso(alpha=alpha)
                LAMT_slid.fit(X,Y[:,np.maximum(k-ng,0):k+1+ng])
                coeff[:,k]=LAMT_slid.coef_.T[:,np.minimum(ng,k)]
                if intercept:
                    intercepts[:,k]=LAMT_slid.intercept_.T[np.minimum(ng,k)]
                alphas_.append(alpha)
            else:
                coeff[:,k]=LAMTCV_slid.coef_.T[:,np.minimum(ng,k)]
                if intercept:
                    intercepts[:,k]=LAMTCV_slid.intercept_.T[np.minimum(ng,k)]
                alphas_.append(LAMTCV_slid.alpha_)
        #LAMTCV_l_slid_alphas.append(LAMTCV_l_slid.alpha_)
    if recompute:
        LR_sup = sklm.LinearRegression(fit_intercept=intercept)
        coeff2=np.zeros_like(coeff)
        for k in xrange(coeff.shape[1]):
            LR_sup.fit(np.dot(X,np.diag((coeff!=0)[:,k])),Y[:,k])
            coeff2[:,k]=LR_sup.coef_.T
            if intercept:
                intercepts[:,k]=LR_sup.intercept_.T
        coeff=coeff2
    if returnAlpha == True:
        return coeff, intercepts, alphas_
    return coeff,intercepts

def glasso(X,Y,ng=9,alpha=0.0001):

    coeff=np.zeros((X.shape[1],Y.shape[1]))
    LAMT_slid=sklm.MultiTaskLasso(alpha=alpha)
    for k in xrange(int(np.ceil(Y.shape[1]/float(ng)))):
        LAMT_slid.fit(X,Y[:,k*ng:(k+1)*ng])
        coeff[:,k*ng:(k+1)*ng]=LAMT_slid.coef_.T
    return coeff

def gridge_bic(X,Y,alphas=np.logspace(-7,3,100),intercept=True,multivar=False,
               averaged=True):
    if averaged==True: #work on averaged data for model selection
        Y_all=Y.copy()
        Y=np.mean(Y,axis=1)[:,None]

    n_samples = X.shape[0]
    n_models = X.shape[1]
    n_targets = Y.shape[1]
    #coef_path_ = []#np.zeros((n_models,n_targets,n_models+1))


    if intercept == True:
        X_offset = np.average(X, axis=0)
        Y_offset = np.average(Y, axis=0)
        X=X-X_offset
        Y= Y-Y_offset
        #X,Y,X_offset,Y_offset,X_scale=sklm.base._preprocess_data(X,Y,fit_intercept=True)


    K = np.log(n_samples*n_targets)  # BIC
    U,sval,V=np.linalg.svd(X,full_matrices=False)
    df = np.zeros(len(alphas))  # Degrees of freedom
    mean_squared_error=[]
    UtY=np.dot(U.T,Y)
    #print V.T.shape,np.diag(sval/(sval**2+1)).shape,UtY.shape,Y.shape,U.T.shape
    for k,alpha in enumerate(alphas):
        #Ridge=sklm.Ridge(alpha,fit_intercept=intercept)
        #Ridge.fit(X,Y,)
        #R = Y - np.dot(X, Ridge.coef_.T)  # residuals
        coef_=np.dot(np.dot(V.T,np.diag(sval/(sval**2+alpha))),UtY)
        R = Y - np.dot(X, coef_)  # residuals
        mean_squared_error.append(np.mean(R ** 2,axis=0))
        df[k] = np.sum(sval**2/(alpha+sval**2))

    mean_squared_error=np.array(mean_squared_error)

    if multivar==True:
        criterion_ = n_samples * np.sum(np.log(mean_squared_error),axis=1) + K * df
    else:
        criterion_ = n_samples * n_targets* np.log(np.mean(mean_squared_error,axis=1)) + K * df

    n_best = np.argmin(criterion_)

    return alphas[n_best]

def gridge_cv(X,Y,ng=1,alphas=np.logspace(-5,2,100),block=True,intercept=True,
              oneSig=False,method='gcv_spe',sig2=None,support=None):

    coeff=np.zeros((X.shape[1],Y.shape[1]))
    intercepts=np.zeros((1,Y.shape[1]))
    RCV_slid=sklm.RidgeCV(alphas=alphas,fit_intercept=intercept,normalize=True,
                          store_cv_values=True)
    listAlpha=np.zeros((Y.shape[1]))
    listAlphaMin=np.zeros((Y.shape[1]))
    listRSS=[]
    listSig2=[]
    if block:
        if intercept:
            X_centr=X-np.mean(X,axis=0)
            Y_centr=Y-np.mean(Y,axis=0)
        else:
            X_centr=X
            Y_centr=Y
        for x in xrange(X_centr.shape[1]):
            X_centr[:,x]=X_centr[:,x]/np.linalg.norm(X_centr[:,x])
        for k in xrange(int(np.ceil(Y.shape[1]/float(ng)))):

            if method=="bic":
                alpha=gridge_bic(X_centr,Y_centr[:,k*ng:(k+1)*ng],alphas)
                #alpha=np.maximum(alpha,alphaMin)
                Ridge=sklm.Ridge(alpha=alpha,fit_intercept=intercept,normalize=True)
                Ridge.fit(X,Y[:,k*ng:(k+1)*ng])
                coeff[:,k*ng:(k+1)*ng]=Ridge.coef_.T
                if intercept:
                    intercepts[:,k*ng:(k+1)*ng]=Ridge.intercept_.T
            elif method=='gcv_spe':

                alpha,rss=gridge_gcv_spectral(X_centr,Y_centr[:,k*ng:(k+1)*ng],alphas=alphas,Sig2=sig2[k*ng:(k+1)*ng],support=support)
                if sig2 is not None:
                    rss_sig=(rss*1/sig2[k*ng:(k+1)*ng][None,None,:]).mean(axis=1).mean(axis=1)
                    if np.any(rss_sig<1):
                        alpha_tmp=alphas.copy()
                        alpha_tmp[rss_sig>1]=0
                        m=np.minimum(len(alphas)-1,np.argmax(alpha_tmp)+1)
                    else:
                        m=0
                    alphaMin=alphas[m]
                else:
                    alphaMin=0
                listAlphaMin[k*ng:(k+1)*ng]=alphaMin
                #listAlpha[k*ng:(k+1)*ng]=alpha
                listRSS.append(rss.mean(axis=2).mean(axis=1))
                listSig2.append(rss_sig)
                if rss_sig[-1]>1:
                    #alpha=np.maximum(alpha,alphaMin)
                    pass

                    #alpha=np.minimum(alpha,alphaMax)
                listAlpha[k*ng:(k+1)*ng]=alpha

                Ridge=sklm.Ridge(alpha=alpha,fit_intercept=intercept,normalize=True)
                Ridge.fit(X,Y[:,k*ng:(k+1)*ng])
                coeff[:,k*ng:(k+1)*ng]=Ridge.coef_.T
                if intercept:
                    intercepts[:,k*ng:(k+1)*ng]=Ridge.intercept_.T

            else:
                RCV_slid.fit(X,Y[:,k*ng:(k+1)*ng])
                if sig2 is not None:
                    alphaMin=alphas[np.minimum(len(alphas),np.argmax(alphas[RCV_slid.cv_values_.mean(axis=0).mean(axis=0)<sig2])+1)]
                else:
                    alphaMin=0
                if oneSig==True:
                    alpha=oneSigRuleRidge(RCV_slid)
                else:
                    alpha=RCV_slid.alpha_
                alpha=np.maximum(alpha,alphaMin)
                listAlpha[k*ng:(k+1)*ng]=alpha
                Ridge=sklm.Ridge(alpha=alpha,fit_intercept=intercept,normalize=True)
                Ridge.fit(X,Y[:,k*ng:(k+1)*ng])
                coeff[:,k*ng:(k+1)*ng]=Ridge.coef_.T
                if intercept:
                    intercepts[:,k*ng:(k+1)*ng]=Ridge.intercept_.T


    else:
        for k in xrange(Y.shape[1]):
            RCV_slid.fit(X,Y[:,np.maximum(k-ng,0):k+1+ng])
            if intercept:
                intercepts[:,k]=RCV_slid.intercept_.T[np.minimum(ng,k)]
            coeff[:,k]=RCV_slid.coef_.T[:,np.minimum(ng,k)]
    #print RCV_slid.alpha_,alpha
    return coeff,intercepts,listAlpha,listAlphaMin,listRSS,listSig2

def gridge_kgcv(X,Y,shape,support,alphas=np.logspace(-7,3,100),
                intercept=True,w=1,n=10,mean=False):
    rss=[]
    if mean==True:
        Y=np.mean(Y,axis=1)[:,None]

    Ys=Y[support]
    Xs=X[support]
    pos=np.arange(X.shape[0])[support]

    #print Xs.shape,Ys.shape
    U,sval,V=np.linalg.svd(Xs,full_matrices=False)
    UtY=np.dot(U.T,Ys)
    if w==0:
        listInd=np.nonzero(support)[0]
    else:
        listInd=np.random.choice(np.nonzero(support)[0],n,replace=False)
    listInd=np.nonzero(support)[0]
    listXk=dict({})
    listYk=dict({})
    listInd2=dict({})
    for k in listInd:
        if support[k]:
            #Xk,Yk=getNeighbors(X,Y,k,w,shape)
            ind=getNeighbors(X,Y,k,w,shape,listInd)
            ind2=[i for i,j in enumerate(pos) if j in ind]
            #ind2=[np.nonzero(pos==i)[0] for i in ind]
            listXk[k]= X[ind,:]
            listYk[k]= Y[ind,:]
            listInd2[k] = ind2
    rss=np.zeros((len(alphas),len(listInd),Y.shape[1]))
    listEye=[np.eye(i) for i in xrange((2*w+1)**2+1)]
    eye=np.eye(Xs.shape[0])
    for a,alpha in enumerate(alphas):

        #XtX=np.linalg.inv(np.dot(X.T,X)+alpha*np.eye(X.shape[1]))
        #beta=np.dot(np.dot(XtX,X.T),Y)
        #XtX=np.dot(np.dot(V.T,np.diag(1/(sval**2+alpha))),V)
        XtX=np.dot(V.T,_diag_dot(1/(sval**2+alpha),V))
        #for i in xrange(Y.shape[1]):
        #beta=np.dot(np.dot(V.T,np.diag(sval/(sval**2+alpha))),UtY)
        Xbeta=np.dot(U,_diag_dot(sval**2/(sval**2+alpha),UtY))
        residus=Ys-Xbeta
        USig=U*sval/(sval**2+alpha)
        S=np.dot(USig,U.T)

        #rss[a]=[:,2*(w**2+w)]
        for i,k in enumerate(listInd):
            #if np.sum(X[k])>0.01:
            if support[k]:
                #Xk,Yk=getNeighbors(X,Y,k,w,shape)
                Xk=listXk[k]
                Yk=listYk[k]
                #rss[a].append(np.mean((np.dot(np.linalg.inv(np.eye(len(Xk))-np.dot(np.dot(Xk,XtX),Xk.T)),(Yk-np.dot(Xk,beta))))**2))
                if w==-1: #gcv
                    rss[a].append(((Yk-np.dot(Xk,beta))/(1-trS))**2)
                else:
                    #res=(np.dot(np.linalg.inv(np.eye(len(Xk))-np.dot(np.dot(Xk,XtX),Xk.T)),(Yk-np.dot(Xk,beta))))**2
                    #res=np.linalg.solve(np.eye(len(Xk))-np.dot(np.dot(Xk,XtX),Xk.T),(Yk-np.dot(Xk,beta)))**2
                    #rss[a][i]=(np.linalg.solve(listEye[len(Xk)]-np.dot(np.dot(Xk,XtX),Xk.T),residus[listInd2[k]])**2)[2*(w**2+w)]
                    rss[a][i]=(np.linalg.solve(listEye[len(Xk)]-S[listInd2[k],listInd2[k]],residus[listInd2[k]])**2)[2*(w**2+w)]




    return alphas[np.argmin(np.mean(rss.mean(axis=1),axis=1))],np.array(rss)



def gridge_gcv_spectral(X,Y,support,alphas=np.logspace(-7,3,100),
                w=0,n=10,Sig2=None):

    Ys=Y[support]
    Xs=X[support]
    if Sig2 is None:
        Sig2=np.ones(Y.shape[1])
    sumSig2=Sig2[:-2]+Sig2[2:]
    if (1==0):
        RCV=sklm.RidgeCV(alphas=alphas,fit_intercept=False,store_cv_values=True)
        RCV.fit(Xs,Ys)
        return RCV.alpha_,np.swapaxes(np.swapaxes(RCV.cv_values_,0,2),1,2)
    U,sval,V=np.linalg.svd(Xs,full_matrices=False)
    UtY=np.dot(U.T,Ys)
    listInd=np.nonzero(support)[0]

    rss=np.zeros((len(alphas),len(listInd),Y.shape[1]))
    for a,alpha in enumerate(alphas):
        #w = ((sval**2 + alpha) ** -1) - (alpha ** -1)
        #c = np.dot(U, _diag_dot(w, UtY)) + (alpha ** -1) * Ys
        #G_diag = _decomp_diag(w, U) + (alpha ** -1)
        #if len(Ys.shape) != 1:
        #    G_diag = G_diag[:, np.newaxis]
        #rss[a,:,:]=(c / G_diag) ** 2


        #XtX=np.dot(np.dot(V.T,np.diag(1/(sval**2+alpha))),V)
        #XtX=np.dot(V.T*1/(sval**2+alpha),V)
        #S=np.dot(Xs,np.dot(XtX,Xs.T))
        #USig=U*sval/(sval**2+alpha)
        #S=np.dot(USig,U.T)

        S=(U * _diag_dot(sval**2/(sval**2+alpha),U.T).T).sum(-1) # get diag values of np.dot(Xs,np.dot(XtX,Xs.T))
        Xbeta=np.dot(U,_diag_dot(sval**2/(sval**2+alpha),UtY))
        rss[a,:,:]=((Ys-Xbeta)/(1-S)[:,None])**2

        #Y_m=np.zeros_like(Ys)
        #Y_m[:,0] = Ys[:,1]
        #Y_m[:,1:-1]=(Ys[:,:-2]+Ys[:,2:])/2
        #Y_m[:,-1] = Ys[:,-2]
        #rss[a,:,:]=((Y_m-Xbeta)/(1-S)[:,None])**2


        #beta=np.dot(np.dot(V.T,np.diag(sval/(sval**2+alpha))),UtY)
        #beta=np.dot(V.T*sval/(sval**2+alpha),UtY)
        #Xbeta=np.dot(Xs,beta)
        #res_left=((Ys[:,:-1]-Xbeta[:,1:])/(1-S)[:,None])**2
        #res_right=((Ys[:,1:]-Xbeta[:,:-1])/(1-S)[:,None])**2
        #rss[a,:,0]=res_right[:,0]
        #rss[a,:,1:-1]=(res_left[:,:-1]*1*Sig2[2:]+res_right[:,1:]*Sig2[:-2])/sumSig2
        #rss[a,:,1:-1]=(res_left[:,:-1]+res_right[:,1:])/2
        #rss[a,:,-1]=res_left[:,-1]

    #return alphas[np.argmin(np.mean(np.average(rss,axis=2,weights=1/Sig2),axis=1))],rss
    return alphas[np.argmin(np.mean(np.average(rss,axis=2,weights=1/Sig2),axis=1))],rss

def _diag_dot( D, B):
    # compute dot(diag(D), B)
    if len(B.shape) > 1:
        # handle case where B is > 1-d
        D = D[(slice(None), ) + (np.newaxis, ) * (len(B.shape) - 1)]
    return D * B

def _decomp_diag(v_prime, Q):
    # compute diagonal of the matrix: dot(Q, dot(diag(v_prime), Q^T))
    return (v_prime * Q ** 2).sum(axis=-1)

def getNeighbors(X,Y,k,n,shape,listInd):
    #Xk=[]
    #Yk=[]
    listI=[]
    for i in xrange(-n,n+1):
        for j in xrange(-n,n+1):
            #if (0<= k+i+j*shape[0]) and (k+i+j*shape[0]<X.shape[0]):
            if k+i+j*shape[0] in listInd:
                #if np.sum(X[int(k+i+j*shape[0])])>0.:
                listI.append(int(k+i+j*shape[0]))
                    #Xk.append(X[int(k+i+j*shape[0])])
                    #Yk.append(Y[int(k+i+j*shape[0])])
    return listI
    #return np.array(Xk),np.array(Yk)

def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by symmetry.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        #y[:j,i] = x[0]
        y[:j,i] = x[:j][::-1]
        y[:-j,-(i+1)] = x[j:]
        #y[-j:,-(i+1)] = x[-1]
        y[-j:,-(i+1)] = x[len(x)-j:][::-1]
    return np.median (y, axis=1)


def regulDeblendFunc(X,Y,mask=True,split=True,two_steps=False,l_method='glasso_bic',ng=1,c_method='RCV',g_method=None,cv=5,
    intercept=True,n_alphas=100,eps=1e-3,alpha=0.0001,oneSig=True,support=None,filt=None,trueLines=None,
                  multivar=True,recompute=True,filt_w=101,maskOnly=False,smooth=None,corrflux=True,Y_sig2=None):

    if split:

        Y_c=np.vstack([medfilt(y,filt_w) for y in Y])
        Y_l=Y-Y_c
        if trueLines is not None:
            listMask=trueLines
        elif mask==True:
            listSpe=[]

            for i in xrange(X.shape[1]-1):
                listSpe.append(np.dot(X[:,i:i+1].T,Y_l)[0])
            if intercept:
                listSpe.append(np.dot(X[:,X.shape[1]-1:].T,Y_l)[0])
            listMask=getLinesSupportList(listSpe,w=2,wmax=20,wmin=2,alpha=2.5,filt=None)
            #listMask=getLinesSupportMat(Y_l,mat_sig=Y_sig,w=3,alpha=3.5,wmax=10,wmin=5,beta=1.2,wmin2=3,f2=0.5,f=0.3,filt=filt)
        else:
            listMask=None

        if l_method == 'glasso_bic':
            l_coeff,l_intercepts =glasso_bic(X,Y_l,ng=ng,listMask=listMask,intercept=intercept,
                                           multivar=multivar,maskOnly=maskOnly)

        elif l_method == 'glasso_cv':
            if support is not None:
                X1=X[support,:]
                Y_l1=Y_l[support,:]
            else:
                X1=X
                Y_l1=Y_l
            l_coeff,l_intercepts = glasso_cv(X1,Y_l1,ng=ng,cv=cv,recompute=recompute,
                                           n_alphas=n_alphas,eps=eps,listMask=listMask,
                                           oneSig=oneSig,intercept=intercept,maskOnly=maskOnly)

        if two_steps==True:
            Y_c=Y-np.dot(X,l_coeff)-l_intercepts
        if smooth is not None:
            Y_c=ssl.fftconvolve(Y_c,smooth,mode='same')
        if support is not None:
            X1=X[support,:]
            Y_c1=Y_c[support,:]
        else:
            X1=X
            Y_c1=Y_c
        if c_method == 'RCV':

            RCV = sklm.RidgeCV(alphas=np.logspace(-8,5,200),normalize=True,cv=cv,
                               store_cv_values=True,fit_intercept=intercept)
            RCV.fit(X1,Y_c1)
            c_coeff=RCV.coef_.T
            if intercept:
                c_intercepts=RCV.intercept_.T
            else:
                c_intercepts=np.zeros(Y.shape[1])
            path=RCV.cv_values_
        elif c_method == 'Ridge':
            Ridge = sklm.Ridge(alpha=alpha,normalize=True,fit_intercept=intercept)
            Ridge.fit(X,Y_c)
            c_coeff=Ridge.coef_.T
            if intercept:
                c_intercepts=Ridge.intercept_.T
            else:
                c_intercepts=np.zeros(Y.shape[1])
        elif c_method == 'LR':
            LR = sklm.LinearRegression(normalize=True,fit_intercept=intercept)
            LR.fit(X,Y_c)
            c_coeff=LR.coef_.T
            if intercept:
                c_intercepts=LR.intercept_.T
            else:
                c_intercepts=np.zeros(Y.shape[1])
        elif c_method =='gridge_cv':
            c_coeff,c_intercepts,c_alphas,c_alphas_min,listRSS,listSig2 =gridge_cv(X,Y_c,ng=ng,intercept=intercept,support=support,sig2=Y_sig2)
        if corrflux==True:
            c_coeff=corrFlux(X,Y_c,c_coeff)
        res = c_coeff + l_coeff
        intercepts=c_intercepts+l_intercepts
    elif g_method =='glasso_bic':
        res,intercepts =glasso_bic(X,Y,ng=ng,listMask=None,intercept=intercept,multivar=multivar)
    elif g_method =='RCV':
        RCV = sklm.RidgeCV(alphas=np.logspace(-7,3,100),normalize=True,fit_intercept=intercept)
        RCV.fit(X,Y)
        res=RCV.coef_.T
        intercepts=RCV.intercept_.T
    elif g_method =='gridge_cv':
        res,intercepts =gridge_cv(X,Y,ng=ng,intercept=intercept)
    if mask==True:
        if c_method == 'RCV':
            return res,intercepts,listMask,c_coeff,l_coeff,path
        else:
            return res,intercepts,listMask,c_coeff,l_coeff,Y,Y_l,Y_c,c_alphas,c_alphas_min,listRSS,listSig2
    return res,intercepts

def corrFlux(X,Y,beta,mask=None):
    """
    Correct coefficients
    """
    if (type(mask)==np.bool_) or (mask is None):
        mask=np.zeros(Y.shape[1]).astype(bool)

    beta_c=beta.copy()
    beta_m=beta[:,~mask]
    Y_t=np.dot(Y[:,~mask],np.linalg.pinv(beta_m))
    for i in xrange(X.shape[1]):
        a=np.dot(Y_t[:,i],X[:,i])/np.linalg.norm(X[:,i])**2
        beta_c[i,~mask]=beta_c[i,~mask]*a
    return beta_c
