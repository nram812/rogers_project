#this document contains all the calipso functions
def plotting_para():
    plt.style.use('seaborn-white')
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern Roman'
    plt.rcParams['font.monospace'] = 'Computer Modern Typewriter'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12
def btd1(mod,rad):
    from pyhdf import SD
    f=SD.SD(mod)
    sds=f.select('Brightness_Temperature')
    #sds=f.select('Brightness_Temperature')
    btd=np.repeat(np.repeat((sds.get()[:,0:406,0:270]+15000)*0.01,5,axis=1),5,axis=2).reshape(7,2030,1350)
    diff=btd[0]-btd[1]
    sds=f.select('cloud_top_temperature_1km')
    ctt=(sds.get()+15000)*0.01
    sds=f.select('Cloud_Phase_Optical_Properties')
    phase1=sds.get()
    sds=f.select('Cloud_Phase_Infrared_1km')
    phase2=sds.get()
    f=SD.SD(rad)
   
    sds=f.select('EV_250_Aggr1km_RefSB')
    
    band_600_r=sds.get()[0]*sds.attributes()['reflectance_scales'][0]
    band_800_r=sds.get()[1]*sds.attributes()['reflectance_scales'][1]
    
    #sds=f.select('EV_500_Aggr1km_RefSB')
    
    #plot the brightness temperature differences too. 
    sds=f.select('EV_500_Aggr1km_RefSB')
    band_6_r=sds.get()[-1]*sds.attributes()['reflectance_scales'][-1]
    
    #sds=f.select('EV_1KM_Emissive')
    #band_7_r=(sds.get()[-5]+sds.attributes()['radiance_offsets'][-5])*sds.attributes()['radiance_scales'][-5]
    return diff,band_600_r-band_6_r,band_6_r/band_600_r,band_600_r-band_800_r,band_800_r/band_600_r,band_600_r,band_6_r,ctt,phase1,phase2
def btd(mod):
    from pyhdf import SD
    f=SD.SD(mod)
    sds=f.select('Brightness_Temperature')
    #sds=f.select('Brightness_Temperature')
    btd=np.repeat(np.repeat((sds.get()[:,0:406,:]+15000)*0.01,5,axis=1),5,axis=2).reshape(7,2030,1350)
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
    

    #sds=f.select('EV_1KM_Emissive')
    #band_7_r=(sds.get()[-5]+sds.attributes()['radiance_offsets'][-5])*sds.attributes()['radiance_scales'][-5]
    return diff,ctt,phase1,phase2,cth,lat,lon
def co_locate(cal,mod):
    #from calipso_run_updated_to_analyse import Cal2
    try:
        c=Cal2(cal)
        from pyhdf import SD
        #lat=c.coords()[0]
        #lon=c.coords()[1]
        lat,lon,Image2,Image3=c.file_sort()
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
        lat2=sds.get()[:406,:270]
        """note have changed the shape for a larger image"""
        sds=f.select('Longitude')
        lon2=sds.get()[:406,:270]
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
from pyhdf.SD import SD, SDC
import warnings
class _MODIS:
    """b 
    Returns the MODIS file as a datetime array to colocating the files
    
    """
    
    def __init__(self, filename):
        warnings.simplefilter('ignore', DeprecationWarning)
        
        import datetime    

        self.filename = filename
        
        # time of orbit start
        #self.orbit = filename[-15:-4]
        #self.z = self.orbit[-2:]  # zn or zd
        self.year = int(filename[-34:-30])
        x=(datetime.datetime(self.year,1,1)+datetime.timedelta(int(filename[-30:-27])-1))
        self.month =x.month
        self.hour = int(filename[-26:-24])
        self.minutes = int(filename[-24:-22])
        self.day=x.day
        #self.seconds = int(filename[-15:-13])
        # date tag + orbit start
            
        self.date = datetime.datetime(self.year, self.month, self.day,
                                      self.hour, self.minutes)
#data=sds.get()
import datetime
import numpy as np
class _Cal3:
    """b 
    Trying to open a non-existing CALIOP file gives an exception
    """

    def __init__(self, filename):
        #warnings.simplefilter('ignore', DeprecationWarning)

        
        self.filename = filename
        if filename[-10:-4]=='Subset':
        # time of orbit start
        #self.orbit = filename[-15:-4]
        #self.z = self.orbit[-2:]  # zn or zd
            self.year = int(filename[-32:-28])
            self.month = int(filename[-27:-25])
            self.day = int(filename[-24:-22])
            self.hour = int(filename[-21:-19])
            self.minutes = int(filename[-18:-16])
            #self.seconds = int(filename[-15:-13])
            # date tag + orbit start
            
            self.date = datetime.datetime(self.year, self.month, self.day,
                                      self.hour, self.minutes)
        #elif filename[-42:-32]=='Track-Beta':
            
        else:
            j=7
                    # time of orbit start
        #self.orbit = filename[-15:-4]
        #self.z = self.orbit[-2:]  # zn or zd
            self.year = int(filename[-32+j:-28+j])
            self.month = int(filename[-27+j:-25+j])
            self.day = int(filename[-24+j:-22+j])
            self.hour = int(filename[-21+j:-19+j])
            self.minutes = int(filename[-18+j:-16+j])
            #self.seconds = int(filename[-15+j:-13+j])
            # date tag + orbit start
            self.date = datetime.datetime(self.year, self.month, self.day,
                                      self.hour, self.minutes)

def calipso_sort(filename,interval,threshold):
    import warnings
    warnings.simplefilter('ignore', RuntimeWarning)
    """threshold corresponds to the thresholds on the interval"""
    #opening the filename, where filenameis the calispo
    c=Cal2(filename)
    #lon,lat=c.coords()
    import numpy as np
    #OUTPUTS the image and its quality
    lat,lon,Image2,Image2_qa=c.file_sort()
    #repeating the longitude and latitude arrays for 1km resolution
    lat=np.repeat(lat,5)
    lon=np.repeat(lon,5)
    #this is the height profile from the calipso data
    height=np.array((range(-500,8200,30)+range(8200,20200,60)+range(20200,30100,180))*Image2.shape[1]).reshape(Image2.shape[1],545)
    #only choosing the regions that have high quality
    c=np.where((Image2_qa<6) & (Image2_qa>0))
    Image2[c]=6
    #find the cloud top phase
    p=np.ma.masked_array(Image2,mask=np.isnan(Image2))
    p2=np.cumsum(p,axis=0)
    c4=np.where((p2==np.nanmax(p2,axis=0)) & (p2>0))
    c5=np.where((p2==np.nanmin(p2,axis=0)) & (p2>0))
    #height_array[c]=np.nan
    height1=np.zeros(p2.shape)*np.nan
    height1[c4]=height.T[c4]
    height1_min=np.zeros(p2.shape)*np.nan
    height1_min[c5]=height.T[c5]

    height_array=np.nanmax(height1,axis=0)
    height_min_array=np.nanmax(height1_min,axis=0)
    cloud_top_phase=np.zeros(Image2.shape)*np.nan
    cloud_top_phase[c4]=Image2[c4]
    cloud_top_phase=np.nanmax(cloud_top_phase,axis=0)
    prop=np.zeros(cloud_top_phase.shape)*np.nan
    new_top=np.zeros(cloud_top_phase.shape)*np.nan

    prop2=[len(np.where(cloud_top_phase[i:i+interval]==2)[0])*1.0/((len(np.where(cloud_top_phase[i:i+interval]==2)[0])+len(np.where(cloud_top_phase[i:i+interval]==1)[0]))) if (((len(np.where(cloud_top_phase[i:i+interval]==2)[0])>0 ) or (len(np.where(cloud_top_phase[i:i+interval]==1)[0])>0)) & ((len(cloud_top_phase[i:i+interval][cloud_top_phase[i:i+interval]>=0.0])>interval*(4/5.0))  & (np.nanstd(height_array[i:i+interval])<threshold))) else np.nan for i in range(prop.shape[0]) ]
    #prop2=[np.array(
    diff_thick=height_array-height_min_array
    prop2=np.array(prop2)
    c=np.where(prop2>=0.0)
    
    
    
#    import time
#    start=time.time()
#    for i in range(prop.shape[0]):
#        reg=height_array[i:i+interval]
#        x1=np.nanstd(reg)
#        
#        if x1<400:
#            #print 'chur'
#            reg2=cloud_top_phase[i:i+interval]
#            c=np.where(reg2==2)
#            c1=np.where(reg2==1)
#            prop[i]=height_array[i]
#            new_top[i]=cloud_top_phase[i]
#            #Image4[c4[0][np.where(c4[1]==i)],i]=90
#            #prop2[i]=np.nan
#
#            if ((len(c1[0])>0 ) or (len(c[0])>0)): #& (len(reg[reg>=0.0])>35)):
#
#              prop2[i]=len(c[0])*1.0/(1.0*(len(c[0])+len(c1[0])))
#        print time.time()-start

    

    #c=np.where(height_array>4000)
    z=[]
    index=[]
    c=np.where((height_array>4000)|(height_array<400))
    new_top[c]=np.nan
    prop2[c]=np.nan
    
    for i in range(cloud_top_phase.shape[0]):
        reg=prop2[i:i+interval]
        std2=height_array[i:i+interval]
        thick=diff_thick[i:i+interval]
        hmin1=height_min_array[i:i+interval]
        c1=np.where((reg<0.25) & (reg>0.0))
        c2=np.where((reg>0.7) & (reg<1.0))
        """the mean cloud_height is within a threshold"""
        if ((len(c1[0])>0 )& (len(c2[0])>0) & (len(reg[reg>=0.0])>interval*(4/5.0))&(len(std2[std2>5000])==0)&(len(thick[thick>2000])==0)&(np.nanmean(hmin1)>200)):
            z.append(str(filename))
            index.append(i)
            print 'uce'
    if len(z)>0:
        z=z[0]
            
    
    
    

        
    
    
    
    
    
    return Image2,height_array,cloud_top_phase,index,lat,lon



class _Cal:
    """b 
    Trying to open a non-existing CALIOP file gives an exception
    """

    def __init__(self, filename):
        warnings.simplefilter('ignore', DeprecationWarning)

        self.hdf = SD(filename, SDC.READ)
        self.filename = filename
        if filename[-10:-4]=='Subset':
        # time of orbit start
        #self.orbit = filename[-15:-4]
        #self.z = self.orbit[-2:]  # zn or zd
            self.year = int(filename[-32:-28])
            self.month = int(filename[-27:-25])
            self.day = int(filename[-24:-22])
            self.hour = int(filename[-21:-19])
            self.minutes = int(filename[-18:-16])
            self.seconds = int(filename[-15:-13])
            # date tag + orbit start
            self.id = filename[-25:-4]
            self.date = datetime.datetime(self.year, self.month, self.day,
                                      self.hour, self.minutes, self.seconds)
        #elif filename[-42:-32]=='Track-Beta':
            
        else:
            j=7
                    # time of orbit start
        #self.orbit = filename[-15:-4]
        #self.z = self.orbit[-2:]  # zn or zd
            self.year = int(filename[-32+j:-28+j])
            self.month = int(filename[-27+j:-25+j])
            self.day = int(filename[-24+j:-22+j])
            self.hour = int(filename[-21+j:-19+j])
            self.minutes = int(filename[-18+j:-16+j])
            self.seconds = int(filename[-15+j:-13+j])
            # date tag + orbit start
            self.id = filename[-25+j:-4+j]
            self.date = datetime.datetime(self.year, self.month, self.day,
                                      self.hour, self.minutes, self.seconds)

    def __repr__(self):
        return self.filename

    def close(self):
        self.hdf.end()
        self.hdf = None

    # IO

    def _read_var(self, var, idx=None):
        """
        read a variable (1D or 2D) in HDF file
        """
    
        hdfvar = self.hdf.select(var)
        if idx is None:
            data = hdfvar[:]
        else:
            if len(hdfvar.dimensions()) == 1:
                data = hdfvar[idx[0]:idx[1]]
            else:
                data = hdfvar[idx[0]:idx[1], :]
        hdfvar.endaccess()
        return data



# Useful maths

def _vector_average(v0, navg, missing=None, valid=None):
    """
    v = _vector_average (v0, navg)
    moyenne le vector v0 tous les navg points.
    """

    v0 = v0.squeeze()

    assert v0.ndim == 1, 'in _vector_average, v0 should be a vector'
    if navg == 1:
        return v0

    n = np.floor(1. * np.size(v0, 0) / navg)
    v = np.zeros(n)
    if valid is None:
        valid = np.ones_like(v0)

    for i in np.arange(n):
        n0 = i * navg
        vslice = v0[n0:n0 + navg - 1]
        validslice = valid[n0:n0 + navg - 1]
        if missing is None:
            idx = (validslice != 0) & (vslice > -999.)
            if idx.sum() == 0:
                v[i] = -9999.
            else:
                v[i] = vslice[idx].mean()
        else:
            idx = (vslice != missing) & (validslice != 0) & (vslice > -999.)
            if idx.sum() == 0:
                v[i] = None
            else:
                v[i] = vslice[idx].mean()
    return v


def _array_std(a0, navg, valid=None):
    a0 = a0.squeeze()
    assert a0.ndim == 2, 'in _array_std, a0 should be a 2d array'
    if navg == 1:
        return np.zeros_like(a0)
    n = np.size(a0, 0) / navg
    a = np.zeros([n, np.size(a0, 1)])

    if valid is None:
        valid = np.ones(np.size(a0, 0))

    for i in np.arange(n):
        n0 = i * navg
        aslice = a0[n0:n0 + navg - 1, :]
        validslice = valid[n0:n0 + navg - 1]
        idx = (validslice > 0)
        if idx.sum() == 0:
            a[i, :] = -9999.
        else:
            a[i, :] = np.std(aslice[idx, :], axis=0)
    return a


def _array_average(a0, navg, weighted=False, valid=None, missing=None):
    """
    a = _array_average (a0, navg, weighted=False)
    moyenne le tableau a0 le long des x tous les navg profils.
    missing = valeur a ignorer (genre -9999), ou None  
    precising a missing value might slow things down...      
    """

    a0 = a0.squeeze()

    assert a0.ndim == 2, 'in _array_average, a0 should be a 2d array'
    if navg == 1:
        return a0

    if weighted and (navg % 2 != 1):
        weighted = False
        print("_array_average: navg is even, turning weights off")

    # create triangle-shaped weights
    if weighted:
        w = np.zeros(navg)
        w[:navg / 2. + 1] = np.r_[1:navg / 2. + 1]
        w[navg / 2. + 1:] = np.r_[int(navg / 2.):0:-1]
    else:
        w = None

    # create averaged array a with number of averaged profiles n
    n = np.floor(1. * np.size(a0, 0) / navg)
    a = np.zeros([n, np.size(a0, 1)])
    if valid is None:
        # si on n'a pas d'info sur les profils valides, on suppose qu'ils le sont tous
        valid = np.ones(np.size(a0, 0))

    for i in np.arange(n):
        n0 = i * navg
        aslice = a0[n0:n0 + navg - 1, :]
        validslice = valid[n0:n0 + navg - 1]

        if missing is None:

            idx = (validslice != 0)  # & (np.all(aslice > -999., axis=1))

            if idx.sum() == 0:
                # no valid profiles in the slice according to valid profiles data
                a[i, :] = -9999.
            else:
                aslice = aslice[idx, :]
                # there might be invalid points somewhere in those profiles
                # find number of valid points along the vertical
                npts = np.sum(aslice > -999., axis=0)
                # sum only the valid points along the vertical
                aslice[aslice < -999.] = 0
                asliceavg = np.sum(aslice, axis=0) / npts
                # mark averaged points with no valid raw points as invalid
                asliceavg[npts == 0] = -9999.
                a[i, :] = asliceavg
        else:
            aslice = np.ma.masked_where(missing == aslice, aslice)
            a[i, :] = np.ma.mean(aslice, axis=0, weights=w)
    return a


def _remap_y(z0, y0, y):
    """ z = remap (z0, y0, y)
            interpole les donnees du tableau z0 sur un nouveau y.
            utile pour regridder les donnees meteo genre temp
    """

    z = np.zeros([np.size(z0, 0), np.size(y)])
    for i in np.arange(np.size(z, 0)):
        z[i, :] = np.interp(y, np.flipud(y0), np.flipud(z0[i, :]))
    return z
# -*- coding: utf-8 -*-

#!/usr/bin/env python
#encoding:utf-8

"""
CALIPSO Level 2 data file class
V. Noel 2008-2014
LMD/CNRS
"""

import numpy as np
import datetime



class Cal2(_Cal):
    """
    Class to process CALIOP Level 2 files.
    No averaging is possible here given the qualitative nature of variables. (can't average altitudes etc.)
    example use:
        
        from calipso2 import Cal2
        
        c = Cal2(filename)
        lon, lat = c.coords()
        nl, base, top = c.layers()
        ...
        c.close()
        
    """

    def __init__(self, filename):
        _Cal.__init__(self, filename)
        lat = self._read_var('Latitude')
        # identify 333m or more level 2 files
        if lat.shape[1] == 1:
            self.havg = 0.333
            self.iavg = 0
        else:
            self.havg = 5.
            self.iavg = 1
    def close(self):
        self.hdf.end()
        self.hdf = None
    def coords(self, idx=None):
        """
        Returns longitude and latitude for profiles.
        shape [nprof]
        # the [:,1] is because level 2 vectors contain 3 values, that describe
        # the parameter for the first profile of the averaged section, the last profile,
        # and an average. The middle value is the average, that's what we use.
        """
        import numpy as np
        
        lat = self._read_var('Latitude', idx=idx)[:, self.iavg]
        lon = self._read_var('Longitude', idx=idx)[:, self.iavg]

        return lat, lon
        
    
    def coords_bounds(self):
        """
        returns the lat and lon boundaries for each 5-km profile.
        """
        if self.havg < 1.:
            raise BaseException('333m file == no boundaries')
        lat = self._read_var('Latitude')
        lat0, lat1 = lat[:, 0], lat[:,-1]
        lon = self._read_var('Longitude')
        lon0, lon1 = lon[:, 0], lon[:,-1]
        return (lon0, lat0), (lon1, lat1)

    def time(self, idx=None):
        """
        returns profile time (TAI)
        shape [nprof]
        """
        return self._read_var('Profile_Time', idx=idx)[:][:, self.iavg]

    def file_sort(self,idx=None):
        import numpy as np
        lat = self._read_var('Latitude', idx=idx)[:, self.iavg]
        lon = self._read_var('Longitude', idx=idx)[:, self.iavg]
        #lat=np.repeat(lat,5)
        #lon=np.repeat(lon,5)
        
        c_uce=np.where((lat<-40) &(lat>-68))#new change
        
        
        f = self.flag(idx=idx)
        
        
        cloudflag = (f & 384) >> 7
        calipso_data=cloudflag[c_uce,:][0]
        #calipso_data=np.array(cloudflag,dtype=float)
        #calipso_data[c_uce,:]=np.nan
        data1=calipso_data[:, 0:165]
        data2=calipso_data[:, 165:1165]
        data3=calipso_data[:, 1165:]
        data1d=data3.reshape(15*data3.shape[0],290,order='C')
        data2d=np.repeat(data2.reshape(5*data2.shape[0],200,order='C'),3,axis=0)
        data3d=np.repeat(data1.reshape(3*data1.shape[0],55,order='C'),5,axis=0)
        Image=np.zeros([15*data1.shape[0],545])
        Image[:,0:290]=data1d.T[::-1].T
        Image[:,290:490]=data2d.T[::-1].T
        Image[:,490:]=data3d.T[::-1].T
        Image1=Image.reshape(3,data1.shape[0]*5,545,order='F')
        Image3=np.nansum(Image1,axis=0)
        
        cloudflag = (f & 96) >> 5

        
        calipso_data=cloudflag[c_uce,:][0]
        #calipso_data[c_uce,:]=np.nan
        data1=calipso_data[:, 0:165]
        data2=calipso_data[:, 165:1165] 
        data3=calipso_data[:, 1165:] 
        data1d=data3.reshape(15*data3.shape[0],290,order='C')
        data2d=np.repeat(data2.reshape(5*data2.shape[0],200,order='C'),3,axis=0)
        data3d=np.repeat(data1.reshape(3*data1.shape[0],55,order='C'),5,axis=0)
        Image=np.zeros([15*data1.shape[0],545])
        Image[:,0:290]=data1d.T[::-1].T
        Image[:,290:490]=data2d.T[::-1].T
        Image[:,490:]=data3d.T[::-1].T
        c=np.where((Image==3) | (Image==1))
        Image[c]=1
        c=np.where(Image==0)
        Image[c]=np.nan
        c=np.where(Image==2)
        Image[c]=0
        Image1=Image.reshape(3,Image.shape[0]/3,545,order='F')
        #Image2=np.nansum(Image1,axis=0)

        
        
        Image2=np.sum(Image1,axis=0).T
        c=np.where(Image2>=2)
        Image2[c]=2
        c=np.where(Image2<2)
        Image2[c]=1
        #lat[c_uce]=np.nan
        #lon[c_uce]=np.nan
        #Image2[:,c_uce]=np.nan
        

        return lat[c_uce],lon[c_uce],Image2,Image3.T

    def time_bounds(self):
        '''
        returns the time boundaries for each 5-km profile.
        '''
        if self.havg < 1:
            raise BaseException('333m file == no boundaries')
        time = self._read_var('Profile_Time')
        return time[:,0], time[:,2]
    

    def utc_time(self, idx=None):
        """
        Returns utc time value (decimal time)
        """
        time = self._read_var('Profile_UTC_Time', idx=idx)[:, self.iavg]
        return time

    def datetime(self, idx=None):
        """
        Returns an array of datetime objects based on utc_time values
        """
        time = self.time(navg=navg)
        datetimes = netCDF4.num2date(time, units='seconds since 1993-01-01')
        return np.array(datetimes)

    def datetime2(self, idx=None):
        """
        Returns an array of datetime objects based on utc_time values
        this version is 5 times faster than the datetime function above. 
        Is it worth it ? not sure.
        """

        def _decdate_to_ymd(decdate):

            year = np.floor(decdate / 10000.)
            remainder = decdate - year * 10000
            month = np.floor(remainder / 100.)
            day = np.floor(remainder - month * 100)

            return year + 2000, month, day

        utc = self.utc_time(idx=idx)
        seconds_into_day = ((utc - np.floor(utc)) * 24. * 3600.)

        # first let's check if the orbit spans more than a single date
        y0, m0, d0 = _decdate_to_ymd(utc[0])
        y1, m1, d1 = _decdate_to_ymd(utc[-1])
        if d0 == d1:
            # orbit spans a single date
            # we can be a little faster
            datetimes = np.array(datetime.datetime(int(y0), int(m0), int(d0), 0, 0, 0)) + np.array(
                [datetime.timedelta(seconds=int(ss)) for ss in seconds_into_day])
        else:
            # orbits spans more than a day, we have to compute everything
            print('multi date', y0, m0, d0, y1, m1, d1)
            y, m, d = _decdate_to_ymd(utc)
            datetimes = [datetime.datetime(int(yy), int(mm), int(dd), 0, 0, 0) + datetime.timedelta(seconds=int(ss)) for
                         yy, mm, dd, ss in zip(y, m, d, seconds_into_day)]

        return np.array(datetimes)

    def statistics_532(self):

        var = self._read_var('Attenuated_Backscatter_Statistics_532')
        stats = dict()
        stats['min'] = var[:, 0::6]
        stats['max'] = var[:, 1::6]
        stats['mean'] = var[:, 2::6]
        stats['std'] = var[:, 3::6]
        stats['centroid'] = var[:, 4::6]
        stats['skewness'] = var[:, 5::6]

        return stats

    def off_nadir_angle(self, idx=None):
        """
        Returns the off-nadir-angle, in deg, for profiles.
        shape [nprof]
        """
        return self._read_var('Off_Nadir_Angle', idx=idx)

    def tropopause_height(self, idx=None):
        """
        Returns the ancillary tropopause height, in km, for profiles.
        shape [nprof]
        """
        return self._read_var('Tropopause_Height', idx=idx)[:, 0]

    def tropopause_temperature(self, idx=None):
        """
        Returns the ancillary tropopause temperature, in degC, for profiles
        shape [nprof]
        """
        return self._read_var('Tropopause_Temperature', idx=idx)

    def dem_surface_elevation(self, idx=None):
        """
        Returns the ancillary surface elevation (from digital elevation model)
        in km, for profiles
        shape [nprof]
        """
        return self._read_var('DEM_Surface_Elevation', idx=idx)

    def lidar_surface_elevation(self, idx=None):
        """
        returns 8 values per profile
        min, max, mean, stddev for upper boundary of the surface echo.
        min, max, mean, stddev for lower boundary of the surface echo.
        en kilometres 
        """

        return self._read_var('Lidar_Surface_Elevation', idx=idx)

    def IGBP_Surface_Type(self, idx=None):
        """
        IGBP_Surface_Type:format = "Int_8" ;
        IGBP_Surface_Type:valid_range = "1....18" ;
        IGBP_Surface_Type:fillvalue = '\367' ;
        IGBP_Surface_Type:range_value = "evergreen needleleaf forest, evergreen broadleaf forest, deciduous needleleaf forest, deciduous broadleaf forest, mixed forest, closed shrublands, open shrublands,woody savannas, savannas, grasslands, permanent wetlands, croplands, urban and built-up,cropland/natural vegetation mosaic, snow and ice, barren or sparsely vegetated, water bodies, tundra" ;
        water = 17
        """
        return np.squeeze(self._read_var('IGBP_Surface_Type', idx=idx))
        

    # Layer info

    def layers(self, idx=None):
        """
        Returns layer information by profile:
        nl = number of layers, shape [nprof]
        top = layer top, shape [nprof, nlaymax]
        base = layer base, shape [nprof, nlaymax]
        """

        nl = self._read_var('Number_Layers_Found', idx=idx)[:, 0]
        top = self._read_var('Layer_Top_Altitude', idx=idx)
        base = self._read_var('Layer_Base_Altitude', idx=idx)
        return nl, base, top

    def layers_pressure(self, idx=None):
        """
        returns layer pressure by profile
        units : hPa
        """

        ptop = self._read_var('Layer_Top_Pressure', idx=idx)
        pbase = self._read_var('Layer_Base_Pressure', idx=idx)
        return pbase, ptop

    def layers_number(self, idx=None):
        """
        Returns the number of layer found by profile
        shape [nprof]
        """
        return self._read_var('Number_Layers_Found', idx=idx)[:, 0]

    def midlayer_temperature(self, idx=None):
        """
        Returns the midlayer temperature by layer, in degrees C
        shape [nprof, nlaymax]
        """
        return self._read_var('Midlayer_Temperature', idx=idx)

    def layer_base_temperature(self, idx=None):
        """
        Returns the layer base temperature by layer, in degrees C
        shape [nprof, nlaymax]
        """
        return self._read_var('Layer_Base_Temperature', idx=idx)

    def flag(self, idx=None):
        """
        Returns the feature classification flag by layer
        shape [nprof, nlaymax]
        cf https://eosweb.larc.nasa.gov/sites/default/files/project/calipso/quality_summaries/CALIOP_L2VFMProducts_3.01.pdf
        """
        return self._read_var('Feature_Classification_Flags', idx=idx)

    def layer_type(self, idx=None):
        """
        Returns the layer type from the feature classification flag
        shape [nprof, nlaymax]
        
        0 = invalid (bad or missing data) 
        1 = "clear air"
        2 = cloud
        3 = aerosol
        4 = stratospheric feature
        5 = surface
        6 = subsurface
        7 = no signal (totally attenuated)
        
        """
        f = self.flag(idx=idx)
        # type flag : bits 1 to 3
        typeflag = (f & 7)
        return typeflag

    def layer_subtype(self, idx=None):
        """
        Returs the layer subtype, as identified from the feature
        classification flag
        shape [nprof, nlaymax]
        
        for clouds (feature type == layer_type == 2)
        0 = low overcast, transparent
        1 = low overcast, opaque
        2 = transition stratocumulus
        3 = low, broken cumulus
        4 = altocumulus (transparent) 5 = altostratus (opaque)
        6 = cirrus (transparent)
        7 = deep convective (opaque)
        """
        f = self.flag(idx=idx)
        # subtype flag : bits 10 to 12
        subtype = (f & 3584) >> 9
        return subtype

    def layer_type_qa(self, idx=None):
        """
        Returs the quality flag for the layer type, as identified from the
        feature classification flag
        shape [nprof, nlaymax]
        """
        f = self.flag(idx=idx)
        typeflag_qa = (f & 24) >> 3
        return typeflag_qa

    def phase(self, idx=None):
        """
        
        Returs the layer thermodynamical phase, as identified from the
        feature classification flag
        shape [nprof, nlaymax]
        
        0 = unknown / not determined 1 = randomly oriented ice
        2 = water
        3 = horizontally oriented ice
        """
        import numpy as np
       
        f = self.flag(idx=idx)
        #lat = self._read_var('Latitude', idx=idx)[:, self.iavg]
        #c=np.where((lat<-48)& (lat>-68))
        #lon = self._read_var('Longitude', idx=idx)[:, self.iavg]
        # 96 = 0b1100000, bits 6 to 7
        cloudflag = (f & 96) >> 5
        return cloudflag

    def phase_qa(self, idx=None):
        """
        Returns the quality flag for the layer thermodynamical phase,
        as identified from the feature classification flag
        shape [nprof, nlaymax]
        
        0 = none
        1 = low
        2 = medium 3 = high
        """

        f = self.flag(idx=idx)
        cloudflag_qa = (f & 384) >> 7
        return cloudflag_qa

    def opacity_flag(self, idx=None):
        """
        Returns the opacity flag by layer.
        shape [nprof, nlaymax]
        """

        return self._read_var('Opacity_Flag', idx=idx)

    def horizontal_averaging(self, idx=None):
        """
        Returns the horizontal averaging needed to detect a given layer.
        shape [nprof, nlaymax]
        """
        return self._read_var('Horizontal_Averaging', idx=idx)

    def iatb532(self, idx=None):
        """
        Returns the integrated attenuated total backscatter at 532 nm
        along the layer thickness.
        shape [nprof, nlaymax]
        """
        return self._read_var('Integrated_Attenuated_Backscatter_532', idx=idx)

    def ivdp(self, idx=None):
        """
        Returns the volumic depolarization ratio for the entire layer
        thickness, obtained by the
        ratio of integrated perpendicular backscatter on the integrated
        parallel backscatter at 532 nm
        shape [nprof, nlaymax]
        """
        return self._read_var('Integrated_Volume_Depolarization_Ratio', idx=idx)

    def ipdp(self, idx=None):
        """
        Returns the particulate depolarization ratio for the entire
        layer thickness, i.e. the volumic
        depolarization ratio corrected to remove its molecular component.
        shape [nprof, nlaymax]
        """
        return self._read_var('Integrated_Particulate_Depolarization_Ratio', idx=idx)

    def icr(self, idx=None):
        """
        Returns the integrated attenuated color ratio for the entire
        layer thickness.
        shape [nprof, nlaymax]
        """
        return self._read_var('Integrated_Attenuated_Total_Color_Ratio', idx=idx)

    def ipcr(self, idx=None):
        """
        Returns the integrated color ratio for the entire layer thickness
        shape [nprof, nlaymax]
        """
        return self._read_var('Integrated_Particulate_Color_Ratio', idx=idx)

    def od(self, idx=None):
        """
        Returns the optical depth found by layer.
        shape [nprof, nlaymax]
        """
        return self._read_var('Feature_Optical_Depth_532', idx=idx)
    def blk_btd(self,idx=None):
        
        """
        Returns the black body brightness temperature for each individual pixel along
        the track of the lidar"""
        return self._read_var('Blackbody_Brightness_Temperature',idx=None)
    def btd_08(self,idx=None):
        
        """
        Returns the black body brightness temperature for each individual pixel along
        the track of the lidar"""
        return self._read_var('Brightness_Temperature_08_65',idx=None)
    def btd_10(self,idx=None):
        
        """
        Returns the black body brightness temperature for each individual pixel along
        the track of the lidar"""
        return self._read_var('Brightness_Temperature_10_60',idx=None)
    def btd_12(self,idx=None):
        
        """
        Returns the black body brightness temperature for each individual pixel along
        the track of the lidar"""
        return self._read_var('Brightness_Temperature_12_05',idx=None)
    def emis_08(self,idx=None):
        
        """
        Returns the black body brightness temperature for each individual pixel along
        the track of the lidar"""
        return self._read_var('Effective_Emissivity_08_65',idx=None)
    def emis_10(self,idx=None):
        
        """
        Returns the black body brightness temperature for each individual pixel along
        the track of the lidar"""
        return self._read_var('Effective_Emissivity_10_60',idx=None)
    def emis_12(self,idx=None):
        
        """
        Returns the black body brightness temperature for each individual pixel along
        the track of the lidar"""
        return self._read_var('Effective_Emissivity_12_05',idx=None)
    def part_size(self,idx=None):
        
        """
        Returns the black body brightness temperature for each individual pixel along
        the track of the lidar"""
        return self._read_var('Effective_Particle_Size',idx=None)
    def qa(self,idx=None):
        
        """
        Returns the black body brightness temperature for each individual pixel along
        the track of the lidar"""
        return self._read_var('IIR_Data_Quality_Flag',idx=None)
    def iwp(self,idx=None):
        
        """
        Returns the black body brightness temperature for each individual pixel along
        the track of the lidar"""
        return self._read_var('Ice_Water_Path',idx=None)
    def li_qa(self,idx=None):
        
        """
        Returns the black body brightness temperature for each individual pixel along
        the track of the lidar"""
        return self._read_var('LIDAR_Data_Quality_Flag',idx=None)
    def multi_cloud(self,idx=None):
        
        """
        Returns the black body brightness temperature for each individual pixel along
        the track of the lidar"""
        return self._read_var('Multi_Layer_Cloud_Flag',idx=None)

    
        
