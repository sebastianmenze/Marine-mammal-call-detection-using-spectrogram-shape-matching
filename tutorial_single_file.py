# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 16:18:16 2021

@author: Administrator
"""

from functools import partial
import multiprocessing  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob 
import os
from matplotlib.path import Path

from scipy.io import wavfile
from scipy import signal
from skimage.transform import rescale, resize, downscale_local_mean

from skimage import data, filters, measure, morphology
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
from skimage.morphology import disk  # noqa

def find_regions(audiopath,f_lim,fft_size,db_threshold,minimum_patcharea,startlabel):
    
    # audiopath=r"D:\passive_acoustics\detector_delevopment\pitchtrack\aural_2017_04_05_17_40_00.wav"
    starttime= dt.datetime.strptime( audiopath.split('\\')[-1], 'aural_%Y_%m_%d_%H_%M_%S.wav' )
   
    fs, x = wavfile.read(audiopath)

    # fft_size=2**14
    f, t, Sxx = signal.spectrogram(x, fs, window='hamming',nperseg=fft_size,noverlap=0.9*fft_size)
    
    # f_lim=[10,200]
    # db_threshold=10
    # minimum_patcharea=5*5
    # startlabel=0
    
    ix_f=np.where((f>=f_lim[0]) & (f<=f_lim[1]))[0]
    spectrog = 10*np.log10(Sxx[ ix_f[0]:ix_f[-1],: ] )   
    
    # filter out background
    rectime= pd.to_timedelta( t ,'s')
    spg=pd.DataFrame(np.transpose(spectrog),index=rectime)
    bg=spg.resample('3min').mean().copy()
    bg=bg.resample('1s').interpolate(method='time')
    bg=    bg.reindex(rectime,method='nearest')
    
    background=np.transpose(bg.values)   
    z=spectrog-background
   
    # plt.figure(1)
    # plt.clf()
    # plt.subplot(211)
    # plt.imshow(background,aspect='auto',origin='lower')
    # plt.subplot(212)
    # plt.imshow(z,aspect='auto',origin='lower')
    # plt.clim([10,z.max()])

    
    # Binary image, post-process the binary mask and compute labels
    mask = z > db_threshold
    mask = morphology.remove_small_objects(mask, 50,connectivity=30)
    mask = morphology.remove_small_holes(mask, 50,connectivity=30)
    
    mask = closing(mask,  disk(3) )
    # op_and_clo = opening(closed,  disk(1) )
    
    labels = measure.label(mask)
      
    probs=measure.regionprops_table(labels,spectrog,properties=['label','area','mean_intensity','orientation','major_axis_length','minor_axis_length','weighted_centroid','bbox'])
    df=pd.DataFrame(probs)
    
    # # plot spectrogram and shapes
    # plt.figure(1)
    # plt.clf()
    # plt.imshow(z,aspect='auto',origin='lower',cmap='gist_yarg')
    # plt.clim([0,20])
    # for index in range(len(df)):
    #     label_i = df.loc[index,'label']
    #     contour = measure.find_contours(labels == label_i, 0.5)[0]
    #     y, x = contour.T
    #     plt.plot(x,y)    
            
    # get corect f anf t
    ff=f[ ix_f[0]:ix_f[-1] ]
    ix=df['bbox-0']>len(ff)-1
    df.loc[ix,'bbox-0']=len(ff)-1
    ix=df['bbox-2']>len(ff)-1
    df.loc[ix,'bbox-2']=len(ff)-1
    
    df['f-1']=ff[df['bbox-0']] 
    df['f-2']=ff[df['bbox-2']] 
    df['f-width']=df['f-2']-df['f-1']
    
    ix=df['bbox-1']>len(t)-1
    df.loc[ix,'bbox-1']=len(t)-1
    ix=df['bbox-3']>len(t)-1
    df.loc[ix,'bbox-3']=len(t)-1
    
    df['t-1']=t[df['bbox-1']] 
    df['t-2']=t[df['bbox-3']] 
    df['duration']=df['t-2']-df['t-1']
        
    tt=df['t-1']
    df['realtime']=starttime + pd.to_timedelta( tt ,'s')
    
    indices=np.where( (df['area']<minimum_patcharea) | (df['bbox-3']-df['bbox-1']<3)  )[0]
    df=df.drop(indices)
    df=df.reset_index()    
    
    df['id']= startlabel + np.arange(len(df))
    
    # get region dict
    sgram={}
    patches={}
    p_t_dict={}
    p_f_dict={}
    
    for ix in range(len(df)):
        m=labels== df.loc[ix,'label']
        ix1=df.loc[ix,'bbox-1']
        ix2=df.loc[ix,'bbox-3']
        jx1=df.loc[ix,'bbox-0']
        jx2=df.loc[ix,'bbox-2'] 
    
        patch=m[jx1:jx2,ix1:ix2]
        pt=t[ix1:ix2]
        pt=pt-pt[0]     
        pf=ff[jx1:jx2]
        
        # contour = measure.find_contours(m, 0.5)[0]
        # y, x = contour.T
                   
        patches[ df['id'][ix]  ] = patch
        p_t_dict[ df['id'][ix]  ] = pt
        p_f_dict[ df['id'][ix]  ] = pf
           
        ix1=ix1-10
        if ix1<=0: ix1=0
        ix2=ix2+10
        if ix2>=spectrog.shape[1]: ix2=spectrog.shape[1]-1       
        sgram[ df['id'][ix]  ] = spectrog[:,ix1:ix2]

    return df, patches,p_t_dict,p_f_dict ,sgram


def match_shape(df, patches ,p_t_dict,p_f_dict ,shape_t, shape_f,shape_label):


    # kernel_csv=r"D:\passive_acoustics\detector_delevopment\specgram_corr\kernel_dcall.csv"
    # df_shape=pd.read_csv(kernel_csv,index_col=0)
    # shape_t=df_shape['Timestamp'].values - df_shape['Timestamp'].min()
    # shape_f=df_shape['Frequency'].values
    # shape_label='dcall'
      
    df.index=df['id']    

    score_smc=[]
    score_ioubox=[] 
    smc_rs=[]

    for ix in df['id'].values:   
        
        # breakpoint()
        patch=patches[ix]
        pf=p_f_dict[ix]
        pt=p_t_dict[ix]
        pt=pt-pt[0]
        
        
        if df.loc[ix,'f-1'] < shape_f.min():
            f1= df.loc[ix,'f-1'] 
        else:
            f1= shape_f.min()
        if df.loc[ix,'f-2'] > shape_f.max():
            f2= df.loc[ix,'f-2'] 
        else:
            f2= shape_f.max()      
            
        # f_lim=[ f1,f2  ]
                
        time_step=np.diff(pt)[0]
        f_step=np.diff(pf)[0]
        k_f=np.arange(f1,f2,f_step )
        
        if pt.max()>shape_t.max():
            k_t=pt
            # k_t=np.arange(0,pt.max(),time_step)

        else:
            k_t=np.arange(0,shape_t.max(),time_step)
            # k_length_seconds=shape_t.max()
            # k_t=np.linspace(0,k_length_seconds,int(np.ceil(k_length_seconds/time_step)) )
    
            
        # generate kernel  
        # ix_f=np.where((p_f>=f_lim[0]) & (p_f<=f_lim[1]))[0]
        # k_f=p_f[ix_f[0]:ix_f[-1]]
        # k_f=np.linspace(f_lim[0],f_lim[1], int( (f_lim[1]-f_lim[0]) /f_step)  )
        
        kk_t,kk_f=np.meshgrid(k_t,k_f)   
        kernel=np.zeros( [ k_f.shape[0] ,k_t.shape[0] ] ) 
        
        x, y = kk_t.flatten(), kk_f.flatten()
        points = np.vstack((x,y)).T 
        p = Path(list(zip(shape_t, shape_f))) # make a polygon
        grid = p.contains_points(points)
        mask = grid.reshape(kk_t.shape) # now you have a mask with points inside a polygon  
        kernel[mask]=1
        
        patch_comp=np.zeros( [ k_f.shape[0] ,k_t.shape[0] ] ) 
        
        # ixp2=np.where((k_t>=pt[0])  & (k_t<=pt[-1]))[0]     
        # ixp1=np.where((k_f>=pf[0])  & (k_f<=pf[-1]))[0]
        
        
        ixp_f=np.where(k_f>=pf[0])[0][0]
                               
        patch_comp[ixp_f:ixp_f+len(pf) , 0:len(pt) ]=patch
        
          
        smc =  np.sum( patch_comp.astype('bool') == kernel.astype('bool') ) /  len( patch_comp.flatten() )
        score_smc.append(smc )

        # shift, error, diffphase = phase_cross_correlation(patch_comp, kernel)
        # score_rms.append(error )
      
        ### iou bounding box
        
        iou_kernel=np.zeros( [ k_f.shape[0] ,k_t.shape[0] ] ) 
        ixp2=np.where((k_t>=shape_t.min())  & (k_t<=shape_t.max()))[0]     
        ixp1=np.where((k_f>=shape_f.min())  & (k_f<=shape_f.max()))[0]     
        iou_kernel[ ixp1[0]:ixp1[-1] , ixp2[0]:ixp2[-1] ]=1

        iou_patch=np.zeros( [ k_f.shape[0] ,k_t.shape[0] ] ) 
        ixp2=np.where((k_t>=pt[0])  & (k_t<=pt[-1]))[0]     
        ixp1=np.where((k_f>=pf[0])  & (k_f<=pf[-1]))[0]  
        iou_patch[ ixp1[0]:ixp1[-1] , ixp2[0]:ixp2[-1] ]=1
       
        intersection=  iou_kernel.astype('bool') & iou_patch.astype('bool')
        union=  iou_kernel.astype('bool') | iou_patch.astype('bool')
        iou_bbox =  np.sum( intersection ) /  np.sum( union )
        score_ioubox.append(iou_bbox)
        
        #####
        # bb= iou_kernel.astype('bool')| iou_patch.astype('bool')
        # plt.figure(3)
        # plt.clf()   
        # plt.subplot(311)    
        # plt.title('Bounding box kernel')
        # plt.imshow(iou_kernel,aspect='auto',origin='lower',cmap='gist_yarg')
        # plt.subplot(312)    
        # plt.title('Bounding box extracted region')
        # plt.imshow(iou_patch,aspect='auto',origin='lower',cmap='gist_yarg')    
        # plt.subplot(313)  
        # plt.title('Bounding boxes Intersection (red) over Union (black)')

        # bb= iou_kernel.astype('bool')| iou_patch.astype('bool')      
        # plt.imshow(bb,aspect='auto',origin='lower',cmap='gist_yarg')    
        # bb= iou_kernel.astype('bool')& iou_patch.astype('bool')      
       
        # plt.contour(bb,[1],colors='r')    
        # plt.tight_layout()    
        # # plt.savefig('example_iou_dcall.jpg',dpi=200) 
        ######
        
        patch_rs = resize(patch, (50,50))
        n_resize=50       
        k_t=np.linspace(0,shape_t.max(),n_resize )
        k_f=np.linspace(shape_f.min(), shape_f.max(),n_resize )   
        kk_t,kk_f=np.meshgrid(k_t,k_f)   
        # kernel=np.zeros( [ k_f.shape[0] ,k_t.shape[0] ] )
        x, y = kk_t.flatten(), kk_f.flatten()
        points = np.vstack((x,y)).T 
        p = Path(list(zip(shape_t, shape_f))) # make a polygon
        grid = p.contains_points(points)
        kernel_rs = grid.reshape(kk_t.shape) # now you have a mask with points inside a polygon  
        smc_rs.append(  np.sum( kernel_rs.astype('bool') == patch_rs.astype('bool') ) /  len( patch_rs.flatten() ) )

        # plt.figure(0)
        # plt.clf()   
        # plt.subplot(311)    
        # plt.title('Template shape')
        # plt.imshow(kernel_rs,aspect='auto',origin='lower',cmap='gist_yarg')
        # plt.subplot(312)   
        # plt.title('Extracted shape')
        # plt.imshow(patch_rs,aspect='auto',origin='lower',cmap='gist_yarg')    
        # plt.subplot(313)  
        
        # bb= patch_rs.astype('bool')== kernel_rs.astype('bool')
        # plt.title('Matching pixels')   
        # plt.imshow(bb,aspect='auto',origin='lower',cmap='gist_yarg')    
        # plt.tight_layout()    
        # # plt.savefig('example_smc_dcall_2.jpg',dpi=200) 

    # corr = signal.correlate2d(patch, kernel)

        # cc=  np.corrcoef(patch.flatten(),kernel.flatten() )[0,1]
        # if np.isnan(cc):
        #     cc=0
        # score_corr.append( cc )
        
    score_smc=np.array(score_smc)
    # score_smc[~ix_boundaries]=0         
    df[shape_label+'_smc']=score_smc
    
    smc_rs=np.array(smc_rs)
    # score_smc[~ix_boundaries]=0         
    df[shape_label+'_smc_rs']=smc_rs
   
    score_ioubox=np.array(score_ioubox)
    df[shape_label+'_ioubox']=score_ioubox
      

    df[shape_label+'_score'] =score_ioubox * smc_rs
    
    #     #### plot sgrams  
    # chosen=np.flip( df[shape_label+'_score'].argsort() )
    # chosen=chosen[:40]
    # n=np.ceil( np.sqrt( len(chosen)+1)  )
    
    # plt.figure(7)
    # plt.clf()
    
    # k=1
    # for ix in chosen:
    #     plt.subplot(n,n,k)    
    #     k=k+1
        
    #     patch=sgram[ix]
    #     plt.imshow(patch,aspect='auto',origin='lower',cmap='inferno')
      
    #     tlt='Score: '+ str(  df.loc[ix,shape_label+'_score'].round(2))+'\nIoU: '+ str(  df.loc[ix,shape_label+'_ioubox'].round(2)) +'\nSMC: ' + str(  df.loc[ix,shape_label+'_smc_rs'].round(2)) 
    #     plt.title(tlt)
    # plt.tight_layout()       
    # # plt.savefig('example_sgrams_and_score.jpg',dpi=200) 
    #     ####
    
         
    return df


#%%

audiopath=r"aural_2017_04_05_17_40_00.wav"

df, patches ,p_t_dict,p_f_dict,sgram = find_regions(audiopath,[10,200],2**14,10,5*5,0)

kernel_csv=r"D:\passive_acoustics\detector_delevopment\specgram_corr\kernel_dcall.csv"
df_shape=pd.read_csv(kernel_csv,index_col=0)
shape_t_dcall=df_shape['Timestamp'].values - df_shape['Timestamp'].min()
shape_f_dcall=df_shape['Frequency'].values

df=match_shape(df,patches ,p_t_dict,p_f_dict, shape_t_dcall, shape_f_dcall,'dcall')


