#Note the files below are downloaded from the onedrive as they are not visible on the computer
#
#TODO run these files tomorrow to improve the figures, note the files have been downloaded
#note the files have been added to "research paper folder on windowns


"""here is code for sampling a transition in cloud phase"""

cal_file='CAL_LID_L2_VFM-Standard-V4-10.2008-08-21T13-55-31ZD.hdf'#'CAL_LID_L2_VFM-Standard-V4-10.2015-09-01T12-20-46ZD.hdf'
mod_file='MYD06_L2.A2008234.1355.006.2013350215759.hdf'#MYD021KM.A2015244.1220.006.2015245152631.hdf'
mod_l2_file='MYD021KM.A2008234.1355.006.2012069224530.hdf'#'MYD06_L2.A2015244.1220.006.2015245162222.hdf'
mac_dir='/Users/neeleshrampal/Downloads/'

desktop_dir='C:/Users/Neelesh/OneDrive/Research Paper/Data_files/'#this has been edited
cal_file=desktop_dir+cal_file
mod_file=desktop_dir+mod_file
mod_l2_file=desktop_dir+mod_l2_file
rad=mod_l2_file
mod=mod_file
#the directories are changed for the different files
#edited for macc
#the code for the colocate function is shown below
def co_locate(cal, mod):
    # from calipso_run_updated_to_analyse import Cal2
    try:
        from file_god_contains_functions import Cal2
        """module needs to be imported to avoid error issues"""
        c = Cal2(cal)
        from pyhdf import SD
        # lat=c.coords()[0]
        # lon=c.coords()[1]
        import numpy as np
        lat, lon, Image2, Image3 = c.file_sort()
        # c=np.where((lat>-68)&(lat<-48))
        # lat=lat[c]
        # lon=lon[c]

        # different co-ordinate resolutions of each product
        # here we are using the r5km resolution
        # btd=c.btd_10()
        # both products are offset by 20 pixels, meaning the temperature products are offset by 20kms.

        # c.close()

        f = SD.SD(mod)
        subdataset_name = 'Latitude'
        sds = f.select(subdataset_name)
        # lat2=np.repeat(np.repeat(sds.get()[:406,:270],5,axis=0),5,axis=1).reshape(2030,1350,order='C')
        # sds=f.select('Longitude')
        # lon2=np.repeat(np.repeat(sds.get()[:406,:270],5,axis=0),5,axis=1).reshape(2030,1350,order='C')
        # x=np.zeros([2030,1354])
        lat2 = sds.get()[:406, :270]
        """note have changed the shape for a larger image"""
        sds = f.select('Longitude')
        lon2 = sds.get()[:406, :270]
        # x=np.zeros([2030,1354])
        cal_index = []
        f.end()
        # x=np.zeros([2030,1354])
        iterr = []
        iter2 = []
        lat1 = []
        lon1 = []
        coords_x = []
        coords_y = []
        for i in range(len(lat)):
            c1 = abs(lat2 - lat[i])

            # print i
            c2 = abs(lon2 - lon[i])
            c3 = np.sqrt(c1 ** 2 + c2 ** 2)
            # print c1.min()
            c = np.where((c3 == c3.min()) & (c3.min() < 0.2))

            if len(c[0]) > 0:
                lat1 = lat1 + [i for i in range(5 * i, 5 * i + 5)]

                # iter2.append(c[0])
                # btd1.append(btd[i])
                coords_x = coords_x + np.arange(5 * c[0], 5 * c[0] + 5).tolist()
                coords_y = coords_y + np.repeat(5 * c[1], 5).tolist()
        return np.array([coords_x, coords_y]), lat1
    except:
        print 'did not work'
        return [], []
        pass

mod='C:\Users\Neelesh\Desktop\MYD06_L2.A2015244.1220.006.2015245162222.hdf'
rad='C:\Users\Neelesh\Downloads\MYD021KM.A2015244.1220.006.2015245152631.hdf'

cal_file='C:\Users\Neelesh\Desktop\CAL_LID_L2_VFM-Standard-V4-10.2015-09-01T12-20-46ZD.hdf'
mod_file=mod
cal=cal_file
x,y=co_locate(cal_file,mod_file)
#the positions are outputted in the file shown.
x1=x[0]
x2=x[1]#x1, x2 are the respective positions of the modis images
#it is neccessary to use some of t

import pylab as py
import numpy as np


#TODO here we are recreating the file to correctly sample the file


#x=coords_x
#y=coords_y
#z=lat1
def btd1(mod, rad):
    from pyhdf import SD
    f = SD.SD(mod)
    sds = f.select('Brightness_Temperature')
    # sds=f.select('Brightness_Temperature')
    #note that the shape of the file, is currently (406,270) this shape can be altered depedending on the file type.
    btd = np.repeat(np.repeat((sds.get()[:, 0:406, 0:270] + 15000) * 0.01, 5, axis=1), 5, axis=2).reshape(7, 2030, 1350)
    #data is over sampled
    #TODO include over sampling
    diff = btd[0] - btd[1]
    sds = f.select('cloud_top_temperature_1km')
    ctt = (sds.get() + 15000) * 0.01
    sds = f.select('Cloud_Phase_Optical_Properties')
    phase1 = sds.get()
    sds = f.select('Cloud_Phase_Infrared_1km')
    phase2 = sds.get()
    f = SD.SD(rad)

    sds = f.select('EV_250_Aggr1km_RefSB')

    red = (sds.get()[0]+sds.attributes()['reflectance_offsets'][0]) * sds.attributes()['reflectance_scales'][0]
    sds = f.select('EV_1KM_RefSB')
    _800 = (sds.get()[-1]+sds.attributes()['reflectance_offsets'][-1]) * sds.attributes()['reflectance_scales'][-1]





    blue = (sds.get()[2]+sds.attributes()['reflectance_offsets'][2]) * sds.attributes()['reflectance_scales'][2]
    grn = (sds.get()[3]+sds.attributes()['reflectance_offsets'][3]) * sds.attributes()['reflectance_scales'][3]
    sds=f.select('EV_500_Aggr1km_RefSB')
    _2105_band = (sds.get()[-1] + sds.attributes()['reflectance_offsets'][-1]) * sds.attributes()['reflectance_scales'][-1]
    _1600_band = (sds.get()[-2] + sds.attributes()['reflectance_offsets'][-2]) * sds.attributes()['reflectance_scales'][-2]
    _1200_band = (sds.get()[-3] + sds.attributes()['reflectance_offsets'][-3]) * sds.attributes()['reflectance_scales'][-3]

    #TODO add this to your log book to remember that you updated your plots with a linear filter to amplify colour
    colour=np.zeros([3,2030,1354])
    colour[0]=red
    colour[1]=grn
    colour[2]=blue
    #c=np.where(colour>0.28)
    #a1=0.28
    #a2=0.37
    ##max_val=colour[:,:,:].max()
    #c2=(a2-a1)/(max_val-a1)
    #c1=(1-c2)*a1
    #colour[c]=c1+c2*colour[c]
    return colour, phase1,phase2,diff,_2105_band,_1600_band,_1200_band,_800


mod=mod_file
rad=mod_l2_file
colour,opt,infrared,diff,_2100,_1600,_1200,_800=btd1(mod,rad)
from file_god_contains_functions import *
Image2,height_array,phase,index,lat1,lon1=calipso_sort(cal,400,3500)
lat = np.repeat(lat1, 5, axis=0)
lon = np.repeat(lon1, 5, axis=0)
y=np.array(y)
#c=np.where((y<970)&(y>710))#location of transition case study 1
c=np.where((y>650))#location of transition case study 1
py.figure()
py.imshow(Image2[:,y])
py.show()
width,height=plt.figaspect(0.8)

fig,ax1=plt.subplots(figsize=(width,height),dpi=100)
#from pylab import *
c1=np.where(phase[c]==1)
c2=np.where(phase[c]==2)
ax1.plot(_800[x1[c],x2[c]][c1],_2100[x1[c],x2[c]][c1],'bo')
ax1.plot(_800[x1[c],x2[c]][c2],_2100[x1[c],x2[c]][c2],'ro')
ax1.legend(['CALIOP Liquid','CALIOP Ice'])
ax1.set_xlabel('0.85$\mu m$ Reflectivity')
ax1.set_ylabel('2.1$\mu m$ Reflectivity')
fig.show()

py.show()
py.plot(1/(colour[0][x1[c],x2[c]]/_2100[x1[c],x2[c]]))
py.show()
py.figure()
py.plot(Image2[:,y[c[0]]])
py.show()
#matching the resolution of the images


#figure()
#pcolormesh(Y,X,infrared[:,:1350])
#the function below creates the plotting parameters and is found in file_god_contains_functions
#TODO note that MODIS tends to over classify ice around the edges
from file_god_contains_functions import *
import matplotlib.pyplot as plt
#plotting_para()

width, height = plt.figaspect(0.8)
fig,ax1=plt.subplots(figsize=(width,height),dpi=100)
ax1.imshow(colour[:,:,:1350].transpose(1,2,0)*25)
fig.show()
ax1.set_title('RGB Image of the Cloud Phase Transition')
X, Y = np.repeat(np.repeat(lat2, 5, axis=0), 5, axis=1).reshape(2030, 1350, order='F'),np.repeat(np.repeat(lon2, 5, axis=0), 5, axis=1).reshape(2030, 1350, order='F')
#the above command defines the grib
ax1[0,0].pcolormesh(Y,X,colour[1,:,:1350])
ax1[0,0].plot(lon1[2500:],lat1[2500:])
colour[:,x1,x2]=12
ax1[0,1].imshow(colour[:,:,:1350].transpose(1,2,0)*6)
clim=(0,0.2)
fig.show()
fig, ax11 = plt.subplots(nrows=1, figsize=(width, height), dpi=100)
# vis_image[x1[0][700:850],x1[1][700:850]]=0.4
# c=ax11.imshow(vis_image,cmap='viridis')
c = ax11.pcolormesh(Y,X,colour[1][:,:1350],cmap='viridis')
#red light is used since it contains the longest wavelenght
colorbar(c)
c.set_clim(0.0, 0.25)
ax11.plot(lon1[y], lat1[y], 'y-')
ax11.plot(lon1[y][400:750], lat1[y][400:750], 'r-')
ax11.legend(['CALIOP Track', 'Phase Transition'])
ax11.set_xlabel('Longitude')
ax11.set_ylabel('Latitude')
ax11.legend(['CALIOP Track', 'Phase Transition'])
ax11.set_xlabel('Longitude')
ax11.set_ylabel('Latitude')
fig.show()

fig, ax11 = plt.subplots(nrows=1, figsize=(width, height), dpi=100)

imshow(Image2[:, y1], cmap='viridis')
py.figure()
py.imshow(colour.transpose(1,2,0),clim=[0,0.25])
clim(0, 0.25)
colorbar()
ylabel('0.6$\mu m$ reflectivity')














    #you need to filter the values that are nan
    figure()


    rgb=np.dstack([colour[0],colour[1],colour[2]])
    color_tuples = rgb.transpose((1,0,2)).reshape((rgb.shape[0]*rgb.shape[1],rgb.shape[2]))

    #color_tuples=color_tuples.reshape(2030,1354,3)
    plt.imshow(colour[:,:,:1350].transpose(1,2,0)*5,clim=(0,15))
    #m.set_array(None)
    plt.show()

    show()
    # plot the brightness temperature differences too.
    sds = f.select('EV_500_Aggr1km_RefSB')
    band_6_r = sds.get()[-1] * sds.attributes()['reflectance_scales'][-1]
#note the function has been written for simplicity

def figures():
    from mpl_toolkits.basemap import Basemap


    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure()
    # setup north polar stereographic basemap.
    # The longitude lon_0 is at 6-o'clock, and the
    # latitude circle boundinglat is tangent to the edge
    # of the map at lon_0. Default value of lat_ts
    # (latitude of true scale) is pole.
    m = Basemap(projection='spstere', boundinglat=-30, lon_0=90, resolution='l')
    X,Y=np.repeat(np.repeat(lat2,5,axis=0),5,axis=1).reshape(2030,1350, order='F'),np.repeat(np.repeat(lon2,5,axis=0),5,axis=1).reshape(2030,1350,order='F')
    #m.fillcontinents(color='coral', lake_color='aqua')
    m.pcolormesh(Y,X,colour[1,:,:1350])
    plt.show()
    #doesnt quite work
    #TODO fix the bug in this code


    # draw parallels and meridians.
    m.drawparallels(np.arange(-80., 81., 20.))
    m.drawmeridians(np.arange(-180., 181., 20.))
    m.drawmapboundary(fill_color='aqua')
    # draw tissot's indicatrix to show distortion.
    ax = plt.gca()
    for y in np.linspace(19 * m.ymin / 20, m.ymin / 20, 10):
        for x in np.linspace(19 * m.xmin / 20, m.xmin / 20, 10):
            lon, lat = m(x, y, inverse=True)
            poly = m.tissot(lon, lat, 2.5, 100, \
                            facecolor='green', zorder=10, alpha=0.5)
    plt.title("South Polar Stereographic Projection")

import matplotlib.pyplot as plt
import numpy as np
#TODO make new scriupts, please accersss al the files siuch as Tau_agree as these contain value infromation about the types of clouds.
#tau=np.load('/Users/neeleshrampal/OneDrive/Run_8th/tau_agree_btd_un.npy')
