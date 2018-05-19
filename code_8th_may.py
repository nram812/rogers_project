#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 15:16:27 2018

@author: neeleshrampal
"""
import numpy as np
x='/Users/neeleshrampal/OneDrive/Honours Research/colocation/Cloud_phase_dataset_water_reset1.npy'
y='/Users/neeleshrampal/OneDrive/Honours Research/colocation/Cloud_phase_dataset_ice_reset1.npy'
#x refers to the water dataset, while y refers to the ice dataset
water=np.load(x)
ice=np.load(y)
figure()
plot(np.nansum(ice[3,0,0],axis=0))
p=np.nansum(ice[0,0,0],axis=0)/np.nansum(ice[0,0,0]+water[0,0,0],axis=0)

p_temp=(ice[0,0,0])#number of ice in a event
clust=np.nansum(ice[0,0,0]+water[0,0,0],axis=1)
where=np.where(clust>0)
p=p[where]
p_temp=p_temp[where]
p_temp2=(ice[0,0,0]+water[0,0,0])#
p_temp2=p_temp2[where]
sum1=array([p_temp[i,:]**2 for i in range(23825)])
sum2=array([p_temp2[i,:]*p_temp[i,:] for i in range(23825)])
sum3=array([p_temp2[i,:]**2 for i in range(23825)])
se=(len(where[0])/sum(clust))*np.sqrt((np.nansum(sum1,axis=0)-2*p*np.nansum(sum2,axis=0)+np.nansum(sum3,axis=0)*p**2)/(23825*(23825-1)))