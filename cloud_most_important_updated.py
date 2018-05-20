#lets run the cloud statistics for the entire region fo the MODIS colocated track as a function of season. 
dir1='F:/SH_data2/'
import os 
import numpy as np
from pyhdf import SD
from calipso_run_updated import _MODIS
import numpy as np
import numpy
import pylab as py
from calipso_run_updated_to_analyse import Cal2
from calipso_run_updated_to_analyse import calipso_sort
from calipso_run_updated_to_analyse import _MODIS
def btd(mod):
    from pyhdf import SD
    f=SD.SD(mod)
    sds=f.select('Brightness_Temperature')
    #sds=f.select('Brightness_Temperature')
    btd=np.repeat(np.repeat((sds.get()[:,0:406,:]+15000)*0.01,5,axis=1),5,axis=2).reshape(7,2030,15)
    """find a way to improve the phase discrimination at 1km"""
    diff=btd[0]-btd[1]
    sds=f.select('cloud_top_temperature_1km')
    ctt=(sds.get()+15000)*0.01
    sds=f.select('Cloud_Phase_Optical_Properties')
    phase1=sds.get()
    sds=f.select('Cloud_Phase_Infrared_1km')
    phase2=sds.get()
    sds=f.select('Latitude')
    lat=np.repeat(np.repeat(sds.get(),5,axis=0),5,axis=1)
    sds=f.select('Longitude')
    lon=np.repeat(np.repeat(sds.get(),5,axis=0),5,axis=1)
    sds=f.select('cloud_top_height_1km')
    cth=sds.get()
    sds=f.select('Cloud_Optical_Thickness')
    tau=sds.get()*0.009999999776482582
    

    #sds=f.select('EV_1KM_Emissive')
    #band_7_r=(sds.get()[-5]+sds.attributes()['radiance_offsets'][-5])*sds.attributes()['radiance_scales'][-5]
    return diff,ctt,phase1,phase2,cth,lat,lon,tau
def co_locate(cal,mod):
    from calipso_run_updated_to_analyse import Cal2
    try:
        c=Cal2(cal)
        from pyhdf import SD
        #lat=c.coords()[0]
        #lon=c.coords()[1]
        lat,lon,Image2,Image3=c.file_sort_s()
        #c=np.where((lat>-68)&(lat<-48))
        #lat=lat[c]
        #lon=lon[c]
    
        #different co-ordinate resolutions of each product
        #here we are using the r5km resolution
        #btd=c.btd_10()
        #both products are offset by 20 pixels, meaning the temperature products are offset by 20kms. 
    
        #c.close()
        
        f=SD.SD(mod)
        subdataset_name='Latitude'
        sds=f.select(subdataset_name)
            #lat2=np.repeat(np.repeat(sds.get()[:406,:270],5,axis=0),5,axis=1).reshape(2030,1350,order='C')
        #sds=f.select('Longitude')
        #lon2=np.repeat(np.repeat(sds.get()[:406,:270],5,axis=0),5,axis=1).reshape(2030,1350,order='C')
        #x=np.zeros([2030,1354])
        lat2=sds.get()[:406,:3]
        sds=f.select('Longitude')
        lon2=sds.get()[:406,:3]
        #x=np.zeros([2030,1354])
        cal_index=[]
        f.end()
        #x=np.zeros([2030,1354])
        iterr=[]
        iter2=[]
        lat1=[]
        lon1=[]
        coords_x=[]
        coords_y=[]
        for i in range(len(lat)):
            c1=abs(lat2-lat[i])
    
            #print i
            c2=abs(lon2-lon[i])
            c3=np.sqrt(c1**2+c2**2)
            #print c1.min()
            c=np.where((c3==c3.min())&(c3.min()<0.025))
    
            if len(c[0])>0:
                lat1=lat1+[i for i in range(5*i,5*i+5)]
    
                #iter2.append(c[0])
                #btd1.append(btd[i])
                coords_x=coords_x+np.arange(5*c[0],5*c[0]+5).tolist()
                coords_y=coords_y+np.repeat(5*c[1],5).tolist()
        return np.array([coords_x,coords_y]),lat1
    except:
        print 'did not work'
        return [],[]
        pass

        
import os
bins=np.arange(220,280,1)

hist_water_ir=np.zeros([12,len(bins)-1])
hist_water_ir2=np.zeros([12,len(bins)-1])
hist_water_ir3=np.zeros([12,len(bins)-1])
hist_water_mir1=np.zeros([12,len(bins)-1])
hist_water_mir2=np.zeros([12,len(bins)-1])
hist_water_mir3=np.zeros([12,len(bins)-1])
hist_water_o=np.zeros([12,len(bins)-1])
hist_water_o2=np.zeros([12,len(bins)-1])
hist_water_o3=np.zeros([12,len(bins)-1])
hist_water_cal=np.zeros([12,len(bins)-1])
hist_water_cal2=np.zeros([12,len(bins)-1])
hist_water_cal3=np.zeros([12,len(bins)-1])

hist_ice_ir=np.zeros([12,len(bins)-1])
hist_ice_ir2=np.zeros([12,len(bins)-1])
hist_ice_ir3=np.zeros([12,len(bins)-1])
hist_ice_mir1=np.zeros([12,len(bins)-1])
hist_ice_mir2=np.zeros([12,len(bins)-1])
hist_ice_mir3=np.zeros([12,len(bins)-1])
hist_ice_o=np.zeros([12,len(bins)-1])
hist_ice_o2=np.zeros([12,len(bins)-1])
hist_ice_o3=np.zeros([12,len(bins)-1])

hist_ice_cal=np.zeros([12,len(bins)-1])
hist_ice_cal2=np.zeros([12,len(bins)-1])
hist_ice_cal3=np.zeros([12,len(bins)-1])
"""need to store the success of each retrieval"""
x=numpy.load("C:/Users/Neelesh/colocation.npy")
uce=0
bins2=np.arange(-2,2,0.1)
bins3=np.arange(0,50,1)
"""the total number of calipso retrievals, for each setting"""
no_cal={}
no_cal['liq']=[]
no_cal['ice']=[]
no_cal['liq2']=[]
no_cal['ice2']=[]
no_cal['liq3']=[]
no_cal['ice3']=[]
"""the disagreement consists of the agreement for water and the disagreement for ice as it has a dimension of 2"""
mir={}
mir['m_liq_cal_liq']=[]
mir['m_ice_cal_liq']=[]
mir['m_un_cal_liq']=[]
mir['m_liq_cal_ice']=[]
mir['m_ice_cal_ice']=[]
mir['m_un_cal_ice']=[]
mir['m2_liq_cal_liq']=[]
mir['m2_ice_cal_liq']=[]
mir['m2_un_cal_liq']=[]
mir['m2_liq_cal_ice']=[]
mir['m2_ice_cal_ice']=[]
mir['m2_un_cal_ice']=[]
mir['m3_liq_cal_liq']=[]
mir['m3_ice_cal_liq']=[]
mir['m3_un_cal_liq']=[]
mir['m3_liq_cal_ice']=[]
mir['m3_ice_cal_ice']=[]
mir['m3_un_cal_ice']=[]
ir={}
ir['m_liq_cal_liq']=[]
ir['m_ice_cal_liq']=[]
ir['m_un_cal_liq']=[]
ir['m_liq_cal_ice']=[]
ir['m_ice_cal_ice']=[]
ir['m_un_cal_ice']=[]
ir['m2_liq_cal_liq']=[]
ir['m2_ice_cal_liq']=[]
ir['m2_un_cal_liq']=[]
ir['m2_liq_cal_ice']=[]
ir['m2_ice_cal_ice']=[]
ir['m2_un_cal_ice']=[]
ir['m3_liq_cal_liq']=[]
ir['m3_ice_cal_liq']=[]
ir['m3_un_cal_liq']=[]
ir['m3_liq_cal_ice']=[]
ir['m3_ice_cal_ice']=[]
ir['m3_un_cal_ice']=[]
o={}
o['m_liq_cal_liq']=[]
o['m_ice_cal_liq']=[]

o['m_liq_cal_ice']=[]
o['m_ice_cal_ice']=[]

o['m2_liq_cal_liq']=[]
o['m2_ice_cal_liq']=[]

o['m2_liq_cal_ice']=[]
o['m2_ice_cal_ice']=[]

o['m3_liq_cal_liq']=[]
o['m3_ice_cal_liq']=[]

o['m3_liq_cal_ice']=[]
o['m3_ice_cal_ice']=[]

btd_agree={}
btd_agree['cal_liq_mod_liq']=np.zeros([len(x),len(bins2)-1])
btd_agree['cal_liq_mod_ice']=np.zeros([len(x),len(bins2)-1])
btd_agree['cal_liq_mod_un']=np.zeros([len(x),len(bins2)-1])

btd_agree['cal_ice_mod_liq']=np.zeros([len(x),len(bins2)-1])
btd_agree['cal_ice_mod_ice']=np.zeros([len(x),len(bins2)-1])
btd_agree['cal_ice_mod_un']=np.zeros([len(x),len(bins2)-1])

tau_agree={}
tau_agree['cal_liq_mod_liq']=np.zeros([len(x),len(bins3)-1])
tau_agree['cal_liq_mod_ice']=np.zeros([len(x),len(bins3)-1])
tau_agree['cal_liq_mod_un']=np.zeros([len(x),len(bins3)-1])
#you require the optical depth for each image in particualr
tau_agree['cal_ice_mod_liq']=np.zeros([len(x),len(bins3)-1])
tau_agree['cal_ice_mod_ice']=np.zeros([len(x),len(bins3)-1])
tau_agree['cal_ice_mod_un']=np.zeros([len(x),len(bins3)-1])




for i in range(0,len(x)):
    mod=x[i][0]
    cal=x[i][1]
    z1=_MODIS(mod).date.month-1
    try:
        x1,y1=co_locate(cal,mod)
    except:
        x1,y1=[],[]
        continue
    
    if len(y1)>0:
        try:
            cloud_top_phase,ctp2,ctp3,cth1=calipso_sort(cal,300,3500,'s')
            """have change this condition"""
            diff,ctt,phase1,phase2,cth,lat,lon,tau=btd(mod)
            """ctp3 is only boundary layer clouds"""
            """cloud_top_phase is simply the phase restricted by standard deviations"""
    
            ctt=ctt[x1[0],x1[1]]
            cth=cth[x1[0],x1[1]]
            diff=diff[x1[0],x1[1]]
            cloud_top_phase=cloud_top_phase[y1]
            ctp2=ctp2[y1]
            ctp3=ctp3[y1]
            phase1=phase1[x1[0],x1[1]]
            phase2=phase2[x1[0],x1[1]]
            tau=tau[x1[0],x1[1]]
            x12=np.zeros(diff.shape)*np.nan
            c=np.where((diff>0.45)&(ctt>150)&(cloud_top_phase>0))
            """optimized thresholds, should consider an adjustment"""
            c2=np.where((diff<-0.33)&(ctt>150)&(cloud_top_phase>0))
            c3=np.where((ctt<150)&(diff>-0.35)&(diff<0.45))
            """using the agreement of the MODIS cloud mask"""
            x12[c]=2
            x12[c2]=1
            x12[c3]=0
            
            
            v=np.where((ctt>150)&(cloud_top_phase>0))#does not need a cloud top height condition 
            #since this is included already in the cloud top phase
            """this condition only chooses the data in which all the cloud masks agree"""
            v2=np.where((ctt>150)&(ctp3>0))
            v3=np.where((ctt>150)&(ctp3>0))
            """the zero refers to the ambigous classification"""
            """these represent the other conditions associated with the other cloud top phase products"""
            
            #v=np.where(cth1[y1]<3500)
            """appeend the calipso water and ice pixels that MODIS appends and the brightness temperature of each of these temperature to identify why these pixels were incorrectly identified"""
            ctt1=ctt[v]
            cth1=cth[v]
            phase11=phase1[v]
            phase21=phase2[v]
            cloud_top_phase=cloud_top_phase[v]
            x121=x12[v]
            diff11=diff[v]
            tau1=tau[v]
            
            ctt2=ctt[v2]
            cth2=cth[v2]
            phase12=phase1[v2]
            phase22=phase2[v2]
            ctp2=ctp2[v2]
            x122=x12[v2]
            diff22=diff[v2]
            tau2=tau[v2]
            
            ctt3=ctt[v3]
            cth3=cth[v3]
            phase13=phase1[v3]
            phase23=phase2[v3]
            ctp3=ctp3[v3]
            x123=x12[v3]
            diff33=diff[v3]
            tau3=tau[v3]
            
            
            
            #x12=x12[v]

            #x12=x12[v]
            """the statistics have been analysed with a smooth cloud top height and where both cloud masks agree"""
            
            """cloud_top_phase"""
            c=np.where(cloud_top_phase==1)
            no_cal['liq'].append(len(c[0]))
            c21=np.where(x121[c]==1)
            mir['m_liq_cal_liq'].append(len(c21[0]))
            c22=np.where(x121[c]==2)
            mir['m_ice_cal_liq'].append(len(c22[0]))
            c23=np.where(x121[c]==0)
            mir['m_un_cal_liq'].append(len(c23[0]))
            
            c21=np.where(phase11[c]==2)
            o['m_liq_cal_liq'].append(len(c21[0]))
            c22=np.where(phase11[c]==3)
            o['m_ice_cal_liq'].append(len(c22[0]))
            
            c21=np.where(phase21[c]==1)
            ir['m_liq_cal_liq'].append(len(c21[0]))
            c22=np.where(phase21[c]==2)
            ir['m_ice_cal_liq'].append(len(c22[0]))
            c23=np.where(phase21[c]==0)
            

            """agree and 0 corresponds to the agreement of water"""
            btd_agree['cal_liq_mod_liq'][i]=btd_agree['cal_liq_mod_liq'][i]+np.histogram(diff11[c][c21],bins=bins2)[0]
            """disagree and 0 index corresponds to a disagreement(liquid calipso and ice x121)"""
            btd_agree['cal_liq_mod_ice'][i]=btd_agree['cal_liq_mod_ice'][i]+np.histogram(diff11[c][c22],bins=bins2)[0]
            btd_agree['cal_liq_mod_un'][i]=btd_agree['cal_liq_mod_un'][i]+np.histogram(diff11[c][c23],bins=bins2)[0]
            tau_agree['cal_liq_mod_liq'][i]=tau_agree['cal_liq_mod_liq'][i]+np.histogram(tau1[c][c21],bins=bins3)[0]
            """disagree and 0 index corresponds to a disagreement(liquid calipso and ice x121)"""
            tau_agree['cal_liq_mod_ice'][i]=tau_agree['cal_liq_mod_ice'][i]+np.histogram(tau1[c][c22],bins=bins3)[0]
            tau_agree['cal_liq_mod_un'][i]=tau_agree['cal_liq_mod_un'][i]+np.histogram(tau1[c][c23],bins=bins3)[0]
            
            
            c=np.where(cloud_top_phase==2)     
            no_cal['ice'].append(len(c[0]))
            c21=np.where(x121[c]==1)
            mir['m_liq_cal_ice'].append(len(c21[0]))
            c22=np.where(x121[c]==2)
            mir['m_ice_cal_ice'].append(len(c22[0]))
            c23=np.where(x121[c]==0)
            mir['m_un_cal_ice'].append(len(c23[0]))
            
            c21=np.where(phase11[c]==2)
            o['m_liq_cal_ice'].append(len(c21[0]))
            c22=np.where(phase11[c]==3)
            o['m_ice_cal_ice'].append(len(c22[0]))
            
            c21=np.where(phase21[c]==1)
            ir['m_liq_cal_ice'].append(len(c21[0]))
            c22=np.where(phase21[c]==2)
            ir['m_ice_cal_ice'].append(len(c22[0]))
            
            

            """agree and 0 corresponds to the agreement of water"""
            btd_agree['cal_ice_mod_liq'][i]=btd_agree['cal_liq_mod_liq'][i]+np.histogram(diff11[c][c21],bins=bins2)[0]
            """disagree and 0 index corresponds to a disagreement(liquid calipso and ice x121)"""
            btd_agree['cal_ice_mod_ice'][i]=btd_agree['cal_liq_mod_ice'][i]+np.histogram(diff11[c][c22],bins=bins2)[0]
            btd_agree['cal_ice_mod_un'][i]=btd_agree['cal_liq_mod_un'][i]+np.histogram(diff11[c][c23],bins=bins2)[0]
            tau_agree['cal_ice_mod_liq'][i]=tau_agree['cal_liq_mod_liq'][i]+np.histogram(tau1[c][c21],bins=bins3)[0]
            """disagree and 0 index corresponds to a disagreement(liquid calipso and ice x121)"""
            tau_agree['cal_ice_mod_ice'][i]=tau_agree['cal_liq_mod_ice'][i]+np.histogram(tau1[c][c22],bins=bins3)[0]
            tau_agree['cal_ice_mod_un'][i]=tau_agree['cal_liq_mod_un'][i]+np.histogram(tau1[c][c23],bins=bins3)[0]
            






            """add the data about the histograms here"""
            """ctp2"""
            c=np.where(ctp2==1)
            no_cal['liq2'].append(len(c[0]))
            c21=np.where(x122[c]==1)
            mir['m2_liq_cal_liq'].append(len(c21[0]))
            c22=np.where(x122[c]==2)
            mir['m2_ice_cal_liq'].append(len(c22[0]))
            c23=np.where(x122[c]==0)
            mir['m2_un_cal_liq'].append(len(c23[0]))
            
            c21=np.where(phase12[c]==2)
            o['m2_liq_cal_liq'].append(len(c21[0]))
            c22=np.where(phase12[c]==3)
            o['m2_ice_cal_liq'].append(len(c22[0]))
            
            c21=np.where(phase22[c]==1)
            ir['m2_liq_cal_liq'].append(len(c21[0]))
            c22=np.where(phase22[c]==2)
            ir['m2_ice_cal_liq'].append(len(c22[0]))
            
            
      
            
            c=np.where(ctp2==2)     
            no_cal['ice2'].append(len(c[0]))
            c21=np.where(x122[c]==1)
            mir['m2_liq_cal_ice'].append(len(c21[0]))
            c22=np.where(x122[c]==2)
            mir['m2_ice_cal_ice'].append(len(c22[0]))
            c23=np.where(x122[c]==0)
            mir['m2_un_cal_ice'].append(len(c23[0]))
            
            c21=np.where(phase12[c]==2)
            o['m2_liq_cal_ice'].append(len(c21[0]))
            c22=np.where(phase12[c]==3)
            o['m2_ice_cal_ice'].append(len(c22[0]))
            
            c21=np.where(phase22[c]==1)
            ir['m2_liq_cal_ice'].append(len(c21[0]))
            c22=np.where(phase22[c]==2)
            ir['m2_ice_cal_ice'].append(len(c22[0]))
            
            """ctp3"""
            c=np.where(ctp3==1)
            no_cal['liq3'].append(len(c[0]))
            c21=np.where(x123[c]==1)
            mir['m3_liq_cal_liq'].append(len(c21[0]))
            c22=np.where(x123[c]==2)
            mir['m3_ice_cal_liq'].append(len(c22[0]))
            c23=np.where(x123[c]==0)
            mir['m3_un_cal_liq'].append(len(c23[0]))
            
            c21=np.where(phase13[c]==2)
            o['m3_liq_cal_liq'].append(len(c21[0]))
            c22=np.where(phase13[c]==3)
            o['m3_ice_cal_liq'].append(len(c22[0]))
            
            c21=np.where(phase23[c]==1)
            ir['m3_liq_cal_liq'].append(len(c21[0]))
            c22=np.where(phase23[c]==2)
            ir['m3_ice_cal_liq'].append(len(c22[0]))
            
            
      
            
            c=np.where(ctp3==2)     
            no_cal['ice3'].append(len(c[0]))
            c21=np.where(x123[c]==1)
            mir['m3_liq_cal_ice'].append(len(c21[0]))
            c22=np.where(x123[c]==2)
            mir['m3_ice_cal_ice'].append(len(c22[0]))
            c23=np.where(x123[c]==0)
            mir['m3_un_cal_ice'].append(len(c23[0]))
            
            c21=np.where(phase13[c]==2)
            o['m3_liq_cal_ice'].append(len(c21[0]))
            c22=np.where(phase13[c]==3)
            o['m3_ice_cal_ice'].append(len(c22[0]))
            
            c21=np.where(phase23[c]==1)
            ir['m3_liq_cal_ice'].append(len(c21[0]))
            c22=np.where(phase23[c]==2)
            ir['m3_ice_cal_ice'].append(len(c22[0]))
            
            c=np.where(x121==2)
            c1=np.where(x121==1)
            hist_ice_mir1[z1]=hist_ice_mir1[z1]+np.histogram(ctt1[c],bins=bins)[0]
            hist_water_mir1[z1]=hist_water_mir1[z1]+np.histogram(ctt1[c1],bins=bins)[0]
            c=np.where(x122==2)
            c1=np.where(x122==1)
            hist_ice_mir2[z1]=hist_ice_mir2[z1]+np.histogram(ctt2[c],bins=bins)[0]
            hist_water_mir2[z1]=hist_water_mir2[z1]+np.histogram(ctt2[c1],bins=bins)[0] 
            c=np.where(x123==2)
            c1=np.where(x123==1)
            hist_ice_mir3[z1]=hist_ice_mir3[z1]+np.histogram(ctt3[c],bins=bins)[0]
            hist_water_mir3[z1]=hist_water_mir3[z1]+np.histogram(ctt3[c1],bins=bins)[0]     
            c=np.where(phase21==2)
            c1=np.where(phase21==1)
            hist_ice_ir[z1]=hist_ice_ir[z1]+np.histogram(ctt1[c],bins=bins)[0]
            hist_water_ir[z1]=hist_water_ir[z1]+np.histogram(ctt1[c1],bins=bins)[0] 
            c=np.where(phase22==2)
            c1=np.where(phase22==1)
            hist_ice_ir2[z1]=hist_ice_ir2[z1]+np.histogram(ctt2[c],bins=bins)[0]
            hist_water_ir2[z1]=hist_water_ir2[z1]+np.histogram(ctt2[c1],bins=bins)[0] 
            c=np.where(phase23==2)
            c1=np.where(phase23==1)
            hist_ice_ir3[z1]=hist_ice_ir3[z1]+np.histogram(ctt3[c],bins=bins)[0]
            hist_water_ir3[z1]=hist_water_ir3[z1]+np.histogram(ctt3[c1],bins=bins)[0] 
            c=np.where(phase11==3)
            c1=np.where(phase11==2)
            hist_ice_o[z1]=hist_ice_o[z1]+np.histogram(ctt1[c],bins=bins)[0]
            hist_water_o[z1]=hist_water_o[z1]+np.histogram(ctt1[c1],bins=bins)[0]
            c=np.where(phase12==3)
            c1=np.where(phase12==2)
            hist_ice_o2[z1]=hist_ice_o2[z1]+np.histogram(ctt2[c],bins=bins)[0]
            hist_water_o2[z1]=hist_water_o2[z1]+np.histogram(ctt2[c1],bins=bins)[0]
            c=np.where(phase13==3)
            c1=np.where(phase13==2)
            hist_ice_o3[z1]=hist_ice_o3[z1]+np.histogram(ctt3[c],bins=bins)[0]
            hist_water_o3[z1]=hist_water_o3[z1]+np.histogram(ctt3[c1],bins=bins)[0]
            c=np.where(cloud_top_phase==2)
            c1=np.where(cloud_top_phase==1)
            hist_ice_cal[z1]=hist_ice_cal[z1]+np.histogram(ctt1[c],bins=bins)[0]
            hist_water_cal[z1]=hist_water_cal[z1]+np.histogram(ctt1[c1],bins=bins)[0]
            c=np.where(ctp2==2)
            c1=np.where(ctp2==1)
            hist_ice_cal2[z1]=hist_ice_cal2[z1]+np.histogram(ctt2[c],bins=bins)[0]
            hist_water_cal2[z1]=hist_water_cal2[z1]+np.histogram(ctt2[c1],bins=bins)[0]
            c=np.where(ctp3==2)
            c1=np.where(ctp3==1)
            hist_ice_cal3[z1]=hist_ice_cal3[z1]+np.histogram(ctt3[c],bins=bins)[0]
            hist_water_cal3[z1]=hist_water_cal3[z1]+np.histogram(ctt3[c1],bins=bins)[0]
            if i%100==0:
                np.save('hist_ice_mir',[hist_ice_mir1,hist_ice_mir2,hist_ice_mir3])
                np.save('hist_water_mir',[hist_water_mir1,hist_water_mir2,hist_water_mir3])
                
                np.save('hist_ice_ir',[hist_ice_ir,hist_ice_ir2,hist_ice_ir3])
                np.save('hist_water_ir',[hist_water_ir,hist_water_ir2,hist_water_ir3])
                
                np.save('hist_water_o',[hist_water_o,hist_water_o2,hist_water_o])
                np.save('hist_ice_o',[hist_ice_o,hist_ice_o2,hist_ice_o3])
                
                np.save('hist_water_cal',[hist_water_cal,hist_water_cal2,hist_water_cal3])
                np.save('hist_ice_cal',[hist_ice_cal,hist_ice_cal2,hist_ice_cal3])
                
                np.save('btd_agree',btd_agree)
                np.save('tau_agree',tau_agree)
                np.save('mir',mir)
                np.save('ir',ir)
                np.save('o',o)
        
            uce=uce+1
            print uce,'working cuz',i
        except:
            continue
    
    else: 
        print 'not working',uce,i
        uce=uce+1
        
