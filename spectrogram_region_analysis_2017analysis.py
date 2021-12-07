# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 22:53:01 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:10:37 2021

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
import pickle
#%%
folder=r'D:\passive_acoustics\detector_delevopment\shapematching_2017\*_shapematching_mf.csv'
sm=pd.DataFrame()
csv_names=glob.glob(folder)
for path in csv_names:
    smnew=pd.read_csv(path)
    ix=np.where( (smnew['dcall_score']>0) | (smnew['fw_downsweep_score']>0) | (smnew['srw_score']>0) | (smnew['fw20_score']>0) )[0]  
    
    sm=pd.concat([sm,smnew.loc[ix,:]],ignore_index=True)
sm['realtime']=pd.to_datetime(sm['realtime'])

#%%

# sm=pd.read_csv('shapematching_2017_mf.csv')
# sm['realtime']=pd.to_datetime(sm['realtime'])

# sm['dcall_score']=sm['dcall_ioubox']*(sm['dcall_smc_rs']-0.5)/0.5
# sm['fw_downsweep_score']=sm['fw_downsweep_ioubox']*(sm['fw_downsweep_smc_rs']-0.5)/0.5
# sm['srw_score']=sm['srw_ioubox']*(sm['srw_smc_rs']-0.5)/0.5
# sm['fw20_score']=sm['fw20_ioubox']*(sm['fw20_smc_rs']-0.5)/0.5

# sm.to_csv('shapematching_2017_mf.csv')


#%%
plt.figure(0)
plt.clf()
plt.subplot(221)
plt.hist(sm['dcall_score'],50)
plt.title('D call')
plt.xlabel('score')
plt.xscale('log')
plt.yscale('log')

plt.subplot(222)
plt.hist(sm['srw_score'],50)
plt.title('SRW')
plt.xlabel('score')
plt.xscale('log')
plt.yscale('log')


plt.subplot(223)
plt.hist(sm['fw_downsweep_score'],50)
plt.title('FW ds')
plt.xlabel('score')
plt.xscale('log')
plt.yscale('log')


plt.subplot(224)
plt.hist(sm['fw20_score'],50)
plt.title('FW20')
plt.xlabel('score')
plt.xscale('log')
plt.yscale('log')

sm['fw20_score'].mean()

sm['fw20_score'].max()
sm['dcall_score'].max()
sm['srw_score'].max()
sm['fw_downsweep_score'].max()


#%%
# plt.figure(0)
# plt.clf()

# ix=np.where( sm['fw_downsweep_score']>0.2 )[0]  

# p_time=sm.loc[ix,'realtime']
# # p_val=np.rad2deg( sm.loc[ix,'orientation'] )
# p_val= sm.loc[ix,'mean_intensity'] 

# plt.plot(p_time,p_val,'.k')

#%%

####
audio_folder=r'I:\postdoc_krill\pam\2017_aural\**'    
audiopath_list=glob.glob(audio_folder+'\*.wav',recursive=True)
timevec=[]

n_dcall_per_min=[]
n_fwdownsweep_per_min=[]
n_fw20_per_min=[]
n_srw_per_min=[]

for audiopath in audiopath_list: 
    starttime=pd.Timestamp( dt.datetime.strptime( audiopath.split('\\')[-1], 'aural_%Y_%m_%d_%H_%M_%S.wav' ) )
    endtime=starttime + pd.Timedelta('8min')
    
   
    threshold=0.2
    n_dcall_per_min.append( np.sum( (sm['dcall_score']>threshold) & (sm['realtime']>starttime) & (sm['realtime']<endtime)   )/ 8.0 ) 
    n_fwdownsweep_per_min.append( np.sum( (sm['fw_downsweep_score']>threshold) & (sm['realtime']>starttime) & (sm['realtime']<endtime)   ) / 8.0) 
    n_srw_per_min.append( np.sum( (sm['srw_score']>threshold) & (sm['realtime']>starttime) & (sm['realtime']<endtime)   ) / 8.0) 
    n_fw20_per_min.append( np.sum( (sm['fw20_score']>threshold) & (sm['realtime']>starttime) & (sm['realtime']<endtime)   )  / 8.0)

    timevec.append(starttime)
timevec=pd.Series(timevec)


# plt.figure(0)
# plt.clf()
# plt.subplot(411)
# plt.plot(timevec,n_dcall_per_min,label='dcall')
# plt.legend()
# plt.subplot(412)

# plt.plot(timevec,n_fwdownsweep_per_min,label='FW downsweep')
# plt.legend()
# plt.subplot(413)

# plt.plot(timevec,n_fw20_per_min,label='FW 20hz')
# plt.legend()
# plt.subplot(414)

# plt.plot(timevec,n_srw_per_min,label='SRW')

# plt.legend()

#%%


plt.figure(0)
plt.clf()
plt.subplot(411)
plt.grid()

plt.plot(timevec,n_dcall_per_min,label='dcall')
plt.title('Blue whale D calls')
plt.ylabel('Calls per min')
plt.subplot(412)
plt.grid()
plt.plot(timevec,n_fwdownsweep_per_min,label='FW downsweep')
plt.title('Fin whale down-sweeps')
plt.ylabel('Calls per min')
plt.subplot(413)
plt.grid()

plt.plot(timevec,n_fw20_per_min,label='FW 20hz')
plt.title('Fin whale 20-Hz calls')
plt.ylabel('Calls per min')
plt.subplot(414)
plt.grid()

plt.plot(timevec,n_srw_per_min,label='SRW')

plt.title('Southern right whale calls')
plt.ylabel('Calls per min')
plt.tight_layout()

# plt.savefig('shapematching_detection_timeseries_2017.jpg',dpi=200)

#%%

ix=sm['fw_downsweep_score']>threshold

plt.figure(1)
plt.clf()
plt.subplot(321)
plt.hist(sm.loc[ix,'f-1'],50 )
plt.title('Lower frequeny bound in Hz')
plt.ylabel('counts')

plt.subplot(322)
plt.hist(sm.loc[ix,'f-2'],50 )
plt.title('Upper frequeny bound in Hz')
plt.ylabel('counts')

plt.subplot(323)
plt.hist(sm.loc[ix,'f-width'],50 )
plt.title('Bandwidth in Hz')
plt.ylabel('counts')

plt.subplot(324)
plt.hist(sm.loc[ix,'duration'],50 )
plt.title('Duration in s')
plt.ylabel('counts')

plt.subplot(325)
plt.hist(np.rad2deg(sm.loc[ix,'orientation']),50 )
plt.title('Sweep Angle')
plt.ylabel('counts')

plt.subplot(326)
plt.hist(sm.loc[ix,'mean_intensity'],50 )
plt.title('Intensity')
plt.ylabel('counts')

plt.tight_layout()

# plt.savefig('shapematching_fw_downsweep_characteristics_histogram_2017.jpg',dpi=200)


ix=sm['srw_score']>threshold

plt.figure(2)
plt.clf()
plt.subplot(321)
plt.hist(sm.loc[ix,'f-1'],50 )
plt.title('Lower frequeny bound in Hz')

plt.subplot(322)
plt.hist(sm.loc[ix,'f-2'],50 )
plt.title('Upper frequeny bound in Hz')

plt.subplot(323)
plt.hist(sm.loc[ix,'f-width'],50 )
plt.title('Bandwidth in Hz')

plt.subplot(324)
plt.hist(sm.loc[ix,'duration'],50 )
plt.title('Duration in s')

plt.subplot(325)
plt.hist(np.rad2deg(sm.loc[ix,'orientation']),50 )
plt.title('Sweep Angle')

plt.subplot(326)
plt.hist(sm.loc[ix,'mean_intensity'],50 )
plt.title('SRW Intensity')

plt.tight_layout()


ix=sm['fw20_score']>threshold

plt.figure(3)
plt.clf()
plt.subplot(321)
plt.hist(sm.loc[ix,'f-1'],50 )
plt.title('Lower frequeny bound in Hz')

plt.subplot(322)
plt.hist(sm.loc[ix,'f-2'],50 )
plt.title('Upper frequeny bound in Hz')

plt.subplot(323)
plt.hist(sm.loc[ix,'f-width'],50 )
plt.title('Bandwidth in Hz')

plt.subplot(324)
plt.hist(sm.loc[ix,'duration'],50 )
plt.title('Duration in s')

plt.subplot(325)
plt.hist(np.rad2deg(sm.loc[ix,'orientation']),50 )
plt.title('Sweep Angle')

plt.subplot(326)
plt.hist(sm.loc[ix,'mean_intensity'],50 )
plt.title('FW 20 Intensity')

plt.tight_layout()


ix=sm['dcall_score']>threshold

plt.figure(4)
plt.clf()
plt.subplot(321)
plt.hist(sm.loc[ix,'f-1'],50 )
plt.title('Lower frequeny bound in Hz')

plt.subplot(322)
plt.hist(sm.loc[ix,'f-2'],50 )
plt.title('Upper frequeny bound in Hz')

plt.subplot(323)
plt.hist(sm.loc[ix,'f-width'],50 )
plt.title('Bandwidth in Hz')

plt.subplot(324)
plt.hist(sm.loc[ix,'duration'],50 )
plt.title('Duration in s')

plt.subplot(325)
plt.hist(np.rad2deg(sm.loc[ix,'orientation']),50 )
plt.title('Sweep Angle')

plt.subplot(326)
plt.hist(sm.loc[ix,'mean_intensity'],50 )
plt.title('D call Intensity')

plt.tight_layout()

#%%

# ix=sm['dcall_score']>0.01

# ix=sm['fw20_score']>0.01
# ix=sm['srw_score']>0.1
ix=sm['fw_downsweep_score']>threshold

plt.figure(5)
plt.clf()

plt.subplot(411)
plt.grid()
plt.title('Intensity')
plt.plot(sm.loc[ix,'realtime'],sm.loc[ix,'mean_intensity'],'.k' )

plt.subplot(412)
plt.grid()
plt.title('Duration')
plt.plot(sm.loc[ix,'realtime'],sm.loc[ix,'duration'],'.k' )

plt.subplot(413)
plt.grid()
plt.title('Width')
plt.plot(sm.loc[ix,'realtime'],sm.loc[ix,'f-width'],'.k' )

plt.subplot(414)
plt.grid()
plt.title('F 1')
plt.plot(sm.loc[ix,'realtime'],sm.loc[ix,'f-1'],'.k' )


plt.tight_layout()

#%%
score=sm['fw_downsweep_score']

ix=score>threshold

# timeseries = pd.Series( pd.date_range(start=sm.realtime.min(),end=sm.realtime.max(),freq='1d') )


plt.figure(5)
plt.clf()

plt.subplot(611)
plt.grid()
plt.title('Intensity')

plt.plot(sm.loc[ix,'realtime'],sm.loc[ix,'mean_intensity'],'.k' )
values = sm.loc[ix,'mean_intensity']
values.index=sm.loc[ix,'realtime']
values_dailymean=values.resample('1d').mean()
plt.plot(values_dailymean,'-r')


plt.subplot(612)
plt.grid()
plt.title('Duration')
plt.plot(sm.loc[ix,'realtime'],sm.loc[ix,'duration'],'.k' )
values = sm.loc[ix,'duration']
values.index=sm.loc[ix,'realtime']
values_dailymean=values.resample('1d').mean()
plt.plot(values_dailymean,'-r')

plt.subplot(613)
plt.grid()
plt.title('Width')
plt.plot(sm.loc[ix,'realtime'],sm.loc[ix,'f-width'],'.k' )
values = sm.loc[ix,'f-width']
values.index=sm.loc[ix,'realtime']
values_dailymean=values.resample('1d').mean()
plt.plot(values_dailymean,'-r')

plt.subplot(614)
plt.grid()
plt.title('F 1')
plt.plot(sm.loc[ix,'realtime'],sm.loc[ix,'f-1'],'.k' )
values = sm.loc[ix,'f-1']
values.index=sm.loc[ix,'realtime']
values_dailymean=values.resample('1d').mean()
plt.plot(values_dailymean,'-r')

plt.subplot(615)
plt.grid()
plt.title('F 2')
plt.plot(sm.loc[ix,'realtime'],sm.loc[ix,'f-2'],'.k' )
values = sm.loc[ix,'f-2']
values.index=sm.loc[ix,'realtime']
values_dailymean=values.resample('1d').mean()
plt.plot(values_dailymean,'-r')

plt.subplot(616)
plt.grid()
plt.title('Score')
plt.plot(sm.loc[ix,'realtime'],score[ix],'.k' )
values = score[ix]
values.index=sm.loc[ix,'realtime']
values_dailymean=values.resample('1d').mean()
plt.plot(values_dailymean,'-r')

plt.tight_layout()

# plt.savefig('shapematching_fw20_characteristics_2017.jpg',dpi=200)


#%% plot sgrams

## best
ix=sm['dcall_score']>0.1

chosen=np.flip( sm['dcall_score'].argsort() )

chosen=chosen[:40]

n=np.ceil( np.sqrt( len(chosen)+1)  )

pklname='shapematching_2017\\'+sm.loc[chosen.iloc[0],'filename'].split('\\')[-1][:-4] + '_shapematching_sgram.pkl'

plt.figure(7)
plt.clf()

k=1
for ix in chosen:

    plt.subplot(n,n,k)    
    k=k+1
    
    pklname='shapematching_2017\\'+sm.loc[ix,'filename'].split('\\')[-1][:-4] + '_shapematching_sgram.pkl'
    
    sgram = pickle.load( open( pklname, "rb" ) )
    patch_id=sm.loc[ix,'id']
    
    patch=sgram[patch_id]
    plt.imshow(patch,aspect='auto',origin='lower',cmap='inferno')
  
    # patch=patches[ix]    
    # plt.contour(patch.astype('int'),[0.5],color='k')
    tlt='score: '+ str(  sm.loc[ix,'dcall_score'].round(2)) 
    plt.title(tlt)
plt.tight_layout()    

#%% worst
ix=np.where(sm['dcall_score'].values>0.2)[0]

score= sm.loc[ix,'dcall_score'].values 

chosen=ix[np.argsort(score)]
# chosen=np.flip( ix[np.argsort(score)] )

chosen=chosen[:40]

n=np.ceil( np.sqrt( len(chosen)+1)  )

# pklname='shapematching_2017\\'+sm.loc[chosen.iloc[0],'filename'].split('\\')[-1][:-4] + '_shapematching_sgram.pkl'

plt.figure(8)
plt.clf()

k=1
for index in chosen:

    plt.subplot(n,n,k)    
    k=k+1
    
    pklname='shapematching_2017\\'+sm.loc[index,'filename'].split('\\')[-1][:-4] + '_shapematching_sgram.pkl'
    
    sgram = pickle.load( open( pklname, "rb" ) )
    patch_id=sm.loc[index,'id']
    
    patch=sgram[patch_id]
    plt.imshow(patch,aspect='auto',origin='lower',cmap='inferno')
  
    # patch=patches[ix]    
    # plt.contour(patch.astype('int'),[0.5],color='k')
    tlt='score: '+ str(  sm.loc[index,'dcall_score'].round(3)) 
    plt.title(tlt)
plt.tight_layout()  

# plt.savefig('shapematching_worst40calls_2017_dcall.jpg',dpi=200)
  

#%% worst
ix=np.where(sm['srw_score'].values>0.2)[0]

score= sm.loc[ix,'srw_score'].values 

chosen=ix[np.argsort(score)]
# chosen=np.flip( ix[np.argsort(score)] )

chosen=chosen[:40]

n=np.ceil( np.sqrt( len(chosen)+1)  )

# pklname='shapematching_2017\\'+sm.loc[chosen.iloc[0],'filename'].split('\\')[-1][:-4] + '_shapematching_sgram.pkl'

plt.figure(8)
plt.clf()

k=1
for index in chosen:

    plt.subplot(n,n,k)    
    k=k+1
    
    pklname='shapematching_2017\\'+sm.loc[index,'filename'].split('\\')[-1][:-4] + '_shapematching_sgram.pkl'
    
    sgram = pickle.load( open( pklname, "rb" ) )
    patch_id=sm.loc[index,'id']
    
    patch=sgram[patch_id]
    plt.imshow(patch,aspect='auto',origin='lower',cmap='inferno')
  
    # patch=patches[ix]    
    # plt.contour(patch.astype('int'),[0.5],color='k')
    tlt='score: '+ str(  sm.loc[index,'srw_score'].round(3)) 
    plt.title(tlt)
plt.tight_layout()    

#%% worst
ix=np.where(sm['fw20_score'].values>0.2)[0]
len(ix)
score= sm.loc[ix,'fw20_score'].values 

chosen=ix[np.argsort(score)]

chosen=chosen[:40]

n=np.ceil( np.sqrt( len(chosen)+1)  )

# pklname='shapematching_2017\\'+sm.loc[chosen.iloc[0],'filename'].split('\\')[-1][:-4] + '_shapematching_sgram.pkl'

plt.figure(8)
plt.clf()

k=1
for index in chosen:

    plt.subplot(n,n,k)    
    k=k+1
    
    pklname='shapematching_2017\\'+sm.loc[index,'filename'].split('\\')[-1][:-4] + '_shapematching_sgram.pkl'
    
    sgram = pickle.load( open( pklname, "rb" ) )
    patch_id=sm.loc[index,'id']
    
    patch=sgram[patch_id]
    plt.imshow(patch,aspect='auto',origin='lower',cmap='inferno')
  
    # patch=patches[ix]    
    # plt.contour(patch.astype('int'),[0.5],color='k')
    tlt='score: '+ str(  sm.loc[index,'fw20_score'].round(3)) 
    plt.title(tlt)
    # plt.yscale('log')
plt.tight_layout()  

#%%

#%% worst
ix=np.where(sm['fw_downsweep_score'].values>0.2)[0]
len(ix)
score= sm.loc[ix,'fw_downsweep_score'].values 

chosen=ix[np.argsort(score)]
# chosen=np.flip( ix[np.argsort(score)] )

chosen=chosen[:40]

n=np.ceil( np.sqrt( len(chosen)+1)  )

# pklname='shapematching_2017\\'+sm.loc[chosen.iloc[0],'filename'].split('\\')[-1][:-4] + '_shapematching_sgram.pkl'

plt.figure(8)
plt.clf()

k=1
for index in chosen:

    plt.subplot(n,n,k)    
    k=k+1
    
    pklname='shapematching_2017\\'+sm.loc[index,'filename'].split('\\')[-1][:-4] + '_shapematching_sgram.pkl'
    
    sgram = pickle.load( open( pklname, "rb" ) )
    patch_id=sm.loc[index,'id']
    
    patch=sgram[patch_id]
    plt.imshow(patch,aspect='auto',origin='lower',cmap='inferno')
  
    # patch=patches[ix]    
    # plt.contour(patch.astype('int'),[0.5],color='k')
    tlt='score: '+ str(  sm.loc[index,'fw_downsweep_score'].round(3)) 
    plt.title(tlt)
    # plt.yscale('log')
plt.tight_layout()  

# plt.savefig('shapematching_best40calls_2017_fwdownsweep.jpg',dpi=200)


#%% check for 24h cycle

hours=np.arange(0,24)
plt.figure(12)
plt.clf()

plt.subplot(221)
callsperhour=[]
ix=np.where(sm['fw20_score'].values>0.2)[0]
a=pd.to_datetime(sm.loc[ix,'realtime'].values)
for hour in hours:
    ix=a.hour==hour
    callsperhour.append(  np.sum( ix ) )
plt.bar( hours,callsperhour )
plt.title('FW 20')
plt.xlabel('Hour')
plt.ylabel('N')

plt.subplot(222)
callsperhour=[]
ix=np.where(sm['fw_downsweep_score'].values>0.2)[0]
a=pd.to_datetime(sm.loc[ix,'realtime'].values)
for hour in hours:
    ix=a.hour==hour
    callsperhour.append(  np.sum( ix ) )
plt.bar( hours,callsperhour )
plt.title('FW downsweep')
plt.xlabel('Hour')
plt.ylabel('N')

plt.subplot(223)
callsperhour=[]
ix=np.where(sm['srw_score'].values>0.2)[0]
a=pd.to_datetime(sm.loc[ix,'realtime'].values)
for hour in hours:
    ix=a.hour==hour
    callsperhour.append(  np.sum( ix ) )
plt.bar( hours,callsperhour )
plt.title('SRW')
plt.xlabel('Hour')
plt.ylabel('N')

plt.subplot(224)
callsperhour=[]
ix=np.where(sm['dcall_score'].values>0.2)[0]
a=pd.to_datetime(sm.loc[ix,'realtime'].values)
for hour in hours:
    ix=a.hour==hour
    callsperhour.append(  np.sum( ix ) )
plt.bar( hours,callsperhour )
plt.title('D calls')
plt.xlabel('Hour')
plt.ylabel('N')
plt.tight_layout()  


# plt.savefig('shapematching_24cycles_2017.jpg',dpi=200)
