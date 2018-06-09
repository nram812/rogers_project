

from pyhdf import SD
import os
import tables as pt
import numpy as np
import time as ti


def file_lists():
    tau_list = os.listdir('Tau/')
    Rad_list = os.listdir('Rad/')
    if check_sort(tau_list,Rad_list) == True:
        return tau_list,Rad_list
    else:
        return 'Unsorted'
    
    
def check_sort(tau_list,Rad_list):
    test = []
    for i in range(len(tau_list)):
        if Rad_list[i][8:26] == tau_list[i][8:26]:
            test.append(True)
            
    if len(test) == len(tau_list):
        return True
    else:
        return False

def access_dataset(dataset_name,SD_file):
    dataset_d = SD_file.select(dataset_name)
    scale = dataset_d.attributes()['scale_factor']
    offset = dataset_d.attributes()['add_offset']
    
    dataset_data = scale*(dataset_d.get()[:]-offset)
    return dataset_data



def interpolation_fix(x):
    """gradient test for inbuilt interpolation in azimuthal angle data
    Replaces bad pixels with nearest non-bad pixels."""
    q = np.where(np.abs(np.gradient(x,axis=1)) >= 1.0)
    #bad_data = x[q]
    if len(q[0]) <> 0:
        width = 12
        new_data = x.copy()
        new_data[q] = x[(q[0],q[1]-width)]
        return new_data
    else:
        return x


def find_bin(value,Thresholds):
    histogram = np.histogram(value,bins=np.array([0]+list(Thresholds)))
    bin_no = np.where(histogram[0]==1)[0][0]
    """Find the bin a specific value is in for each variable"""
    return bin_no
    
def digitizing(array,Thresholds):
    bins = np.array([0]+list(Thresholds))   
    return np.digitize(array,bins)-1

def Open_Datasets(rad_file,tau_file):
    
    f1 = SD.SD('Rad/{0}'.format(rad_file))
    g1 = f1.select('EV_250_RefSB')
    Radiance_620_d = g1.get()[0,:,:]
    scale = g1.attributes()['reflectance_scales'][0]
    offset = g1.attributes()['reflectance_offsets'][0]
    Radiance_620 = scale*(Radiance_620_d-offset)
    
    LAT = f1.select('Latitude').get()
    LON = f1.select('Longitude').get()
    
    
    f2 = SD.SD('Tau/{0}'.format(tau_file)) 
    COT = access_dataset('Cloud_Optical_Thickness_37',f2) 
    COTERR = access_dataset('Cloud_Optical_Thickness_Uncertainty_37',f2) #Percentage Uncertainty in COT_data
    SZA = access_dataset('Solar_Zenith',f2)
    SA = access_dataset('Solar_Azimuth',f2)
    VZA = access_dataset('Sensor_Zenith',f2)
    VA = access_dataset('Sensor_Azimuth',f2)
    REFF = access_dataset('Cloud_Effective_Radius_37',f2) #check - tick
    CTH = access_dataset('Cloud_Top_Height',f2) #CHECK - tick
    CWV = access_dataset('Above_Cloud_Water_Vapor_094',f2)
    #CloudMask = access_dataset('Cloud_Mask_1km',f2)
    
    RAZ = interpolation_fix(SA)-interpolation_fix(VA)
    
    return [VZA,LAT,LON,COT,COTERR,SZA,RAZ,REFF,CTH,CWV,Radiance_620]

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh)
    
def Calculate_Zeta(Radiance_620, COTshape):
    """return the downsampled heterogeneity value (<R>/sigma_R)
    so it has the same resolution as the tau file for Radiance_620"""
    new_shape = (COTshape[0]/2,COTshape[1]/2)
    level1 = rebin(Radiance_620,new_shape)
    level2 = np.swapaxes(level1,1,2)
    level3 = level2.reshape(new_shape[0],new_shape[1],64)
    
    level4 = np.std(level3,axis=2)/np.mean(level3,axis=2)
    level5 = np.mean(level3,axis=2)
    return level4,level5
    
    
def upsample(x,shape_factor):
    return np.array([val for val in x for _ in range(shape_factor**2)])
    
    
def ROI(x,i,j):
    if x.shape[1] == 1354: #check number
        return (x[i*100:100*(i+1),500+j*100:500+(j+1)*100]).ravel()
    elif x.shape[1] ==  270:
        return upsample((x[i*20:20*(i+1),100+j*20:100+(j+1)*20]).ravel(),1354/270)
    elif x.shape[1] ==677:
        return upsample((x[i*50:50*(i+1),250+j*50:250+(j+1)*50]).ravel(),1354/677)

def classify_texture(ROIS,VZT,SZAT,RAT,BT,TT):
    Digitised = np.zeros((10000,4))
    
    
    test = np.zeros((10000))
    
    
    #sza = point[1]#['sza']
    #vza = point[0]#['vza']
    #raz = point[2]#['raz']
    #zeta = point[4]#['zeta']
    #mbrf = point[3]#['mbrf']
    
    
    CN1 = 4#index of corresponding Camera Number (1) bin  - for Nadir, since the cameras are all for VZA that are small . . .
    ABCN = 0#index of corresponding AB Camera Number bin - guess
    ST = 0#index of surface type corresponding to ocean              These last three give a small dependence so . . .
    CP = 0#index corresponding to liquid - this is a guess. . . .
    
    Digitised[:,0] = digitizing(ROIS[0],VZT[:,CN1])
    Digitised[:,1] = digitizing(ROIS[5],SZAT)
    #Digitised[:,2] = digitizing(np.abs(raz),RAT[:,SZA])
    
    for i in range(RAT.shape[1]):
        Digitised[np.where(Digitised[:,2]==i),2] = digitizing(ROIS[6][np.where(Digitised[:,2]==i)],RAT[:,i])
    
    

    for i in range(10000):
        Digitised[i,3] = digitizing(ROIS[-2][i],BT[:,CN1,ST,CP,Digitised[i,2],Digitised[i,0],Digitised[i,1]])
        test[i] = TT[ABCN,Digitised[i,3],Digitised[i,2],Digitised[i,0],Digitised[i,1]]
        
    Texture = ROIS[-1]>=test
        
    return Texture.astype(int)
    
    
    
AZM_Data = SD.SD('/Users/jesseloveridge/Documents/Summer Project/MISR_AM1_AZM_F01_01.hdf') #Thresholds from MISR dataset
VZT = AZM_Data.select('Viewing Zenith Angle Thresholds')[:]
SZAT = AZM_Data.select('Solar Zenith Angle Thresholds')[:] 
RAT = AZM_Data.select('Relative Azimuth Thresholds')[:,:]
BT = AZM_Data.select('Brightness Thresholds')[:]
TT = AZM_Data.select('Texture Thresholds')[:]


os.chdir('/Volumes/Promise1/Jesse/MODIS_DATA/') #file lists requires this directory to be changed
tau_list,rad_list = file_lists()


    #use a for loop here to provide indices for the file lists.

    tau_file = tau_list[index]
    rad_file = rad_list[index]
        
    Datasets = Open_Datasets(rad_file,tau_file)
            
    ZETA,RAD_MEAN = Calculate_Zeta(Datasets[-1],Datasets[3].shape)
    Datasets_list = Datasets[0:-1]+[RAD_MEAN,ZETA]
            
            
        
            
    for j in range(20):
        for k in range(3):
                    
            ROIS  = []
            for data in Datasets_list:
                ROIS.append(ROI(data,j,k))
                        
                textures = classify_texture(ROIS,VZT,SZAT,RAT,BT,TT)
                
                
                
