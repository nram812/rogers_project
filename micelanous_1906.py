#Note the files below are downloaded from the onedrive as they are not visible on the computer
#
#TODO run these files tomorrow to improve the figures, note the files have been downloaded
#note the files have been added to "research paper folder on windowns
mod='C:\Users\Neelesh\OneDrive\_2017\Honours Project\Colocation_Data\COLOC_2015_09_01(not done)\MYD06_L2.A2015244.1220.006.2015245162222.hdf'
rad='C:\Users\Neelesh\OneDrive\_2017\Honours Project\Colocation_Data\COLOC_2015_09_01(not done)\MYD021KM.A2015244.1220.006.2015245152631.hdf'
cal_file='C:\Users\Neelesh\OneDrive\_2017\Honours Project\Colocation_Data\COLOC_2015_09_01(not done)\CAL_LID_L2_VFM-Standard-V4-10.2015-09-01T12-20-46ZD.hdf'
mod_file=mod
cal=cal_file

cal = 'C:\Users\Neelesh\OneDrive\_2017\Honours Project\Colocation_Data\COLOC_2015_09_01(not done)\CAL_LID_L2_VFM-Standard-V4-10.2015-09-01T12-20-46ZD.hdf'
mod = "C:\Users\Neelesh\OneDrive\_2017\Honours Project\Colocation_Data\COLOC_2015_09_01(not done)\MYD06_L2.A2015244.1220.006.2015245162222.hdf"  # here is th eforum

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
cal=cal_file
#the above are simply just the directories
fig1 = case_study_btd_sim(ax3, cal, mod, [-60, -62], [1000, 3500], '(c)', 'case_study_2_c.pdf')
fig2 = case_study_btd_sim(ax3, cal, mod, [-59.5, -63], [-500, 3000], '(c)', 'case_study_1_c.pdf')
%run file_god_contains_functions.py
import matplotlib.gridspec as gridspec

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
    lat=np.repeat(np.repeat(f.select('Latitude').get(),5,axis=0),5,axis=1)
    lon = np.repeat(np.repeat(f.select('Longitude').get(), 5, axis=0), 5, axis=1)

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
    #c=np.where(colour>0.15)
    #a1=0.15
    #a2=0.37
    #max_val=colour[:,:,:].max()
    #c2=(a2-a1)/(max_val-a1)
    #c1=(1-c2)*a1
    #colour[c]=c1+c2*colour[c]
    return colour, phase1,phase2,diff,_2105_band,_1600_band,_1200_band,_800,ctt,lat,lon
def btd2(mod):
    from pyhdf import SD
    f = SD.SD(mod)
    sds = f.select('Brightness_Temperature')
    # sds=f.select('Brightness_Temperature')
    btd = np.repeat(np.repeat((sds.get()[:, 0:406, 0:270] + 15000) * 0.01, 5, axis=1), 5, axis=2).reshape(7, 2030, 1350)
    diff = btd[0] - btd[1]
    sds = f.select('cloud_top_temperature_1km')
    ctt = (sds.get() + 15000) * 0.01
    sds = f.select('Cloud_Phase_Optical_Properties')
    phase1 = sds.get()
    sds = f.select('Cloud_Phase_Infrared_1km')
    phase2 = sds.get()
    # f=SD.SD(rad)
    btd11=btd[1]

    # sds=f.select('EV_250_Aggr1km_RefSB')

    # band_600_r=sds.get()[0]*sds.attributes()['reflectance_scales'][0]
    # band_800_r=sds.get()[1]*sds.attributes()['reflectance_scales'][1]

    # sds=f.select('EV_500_Aggr1km_RefSB')

    # plot the brightness temperature differences too.
    # sds=f.select('EV_500_Aggr1km_RefSB')
    # band_6_r=sds.get()[-1]*sds.attributes()['reflectance_scales'][-1]

    # sds=f.select('EV_1KM_Emissive')
    # band_7_r=(sds.get()[-5]+sds.attributes()['radiance_offsets'][-5])*sds.attributes()['radiance_scales'][-5]
    return diff, btd11
def case_study_btd_sim(ax,cal,mod,lat_range,h_range,title_label_3,figname):
    import matplotlib
    #plotting_para()
    import numpy as np
    latmin=lat_range[1]
    latmax=lat_range[0]
    hmin=h_range[0]
    hmax=h_range[1]
    from pylab import Line2D
    import pylab as py
    diff,ctt=btd2(mod)
    c=Cal2(cal)
    lat,lon,Image2,Image3=c.file_sort()
    import matplotlib.pyplot as plt
    ##plt.figure()
    #plt.imshow(Image2,cmap='viridis')
    #plt.show()
    x=range(-500,8200,30)+range(8200,20200,60)+range(20200,30100,180)
    y=np.repeat(lat,5)
    laty=y
    X,Y=np.meshgrid(x,y)
    import numpy.ma as ma
    Z=Image2
    Zm = ma.masked_where(np.isnan(Z),Z)
    #plt.pcolormesh(X,Y,Zm.T)

    #import prettyplotlib as ppl
    #from prettyplotlib import plt
    import numpy as np
    import string
    #width,height=plt.figaspect(0.8)
    #fig,ax1=plt.subplots(figsize=(width,height),dpi=100)
    colorsList = ['indigo','#FFE600']
    CustomCmap = matplotlib.colors.ListedColormap(colorsList)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    red_patch = mpatches.Patch(color='#FFE600', label='CALIOP Ice')
    blue_patch = mpatches.Patch(color='indigo', label='CALIOP Liquid')
    red_patch1 = Line2D(range(1), range(1), color='r', marker='o', markersize=4, markerfacecolor="r",
                        linewidth=0, label='MODIS BTD')  # mpatches.Circle((3,3),color='#0067acff',marker='o')

    #blue_patch1 = mpatches.Patch(color='r', label='Undetermined')
    x,y=co_locate(cal,mod)
    ax.tick_params(axis=u'both', which=u'both', length=3)
    #x,y=co_locate(cal,mod)
    im=ax.pcolormesh(Y[y,:][::-1],X[y,:][::-1],Zm.T[y,:][::-1],cmap=CustomCmap,alpha=0.6,edgecolor='None')
    leg=ax.legend(handles=[red_patch,blue_patch,red_patch1], loc='upper right', ncol=1, fontsize=9,handletextpad=0.4, columnspacing=0.4,frameon=True)
    leg.get_frame().set_edgecolor('k')
    ax.set_title(title_label_3,fontsize=12)
    ax.set_ylim([hmin,hmax])
    ax.set_xlabel('Latitude ($\degree$)')
    ax.set_ylabel('CALIOP Cloud Height (km)')
    xticks = np.arange(latmin,latmax+0.5,0.5)[::-1]
    yticks = range(hmin,hmax,500)

    xticklabels=xticks
    yticklabels=np.arange(hmin/1000.0,hmax/1000.0,0.5)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    #fig.show()
    ax.tick_params(axis=u'both', which=u'both', length=3)
    #ax1.legend()
    #ax1.set_yticklabels(np.arange(0,500,3500))
    axn=ax.twinx()
    axn.plot(laty[y],diff[x[0],x[1]],'ro',markersize=4,linewidth=2)
    axn.set_ylabel('BTD (K)',color='red')
    axn.set_ylim([-0.7,0.7])
    axn.set_xlim([latmin,latmax])
    axn.tick_params(axis='y',colors='red', length=3)
    axn.spines['right'].set_color('red')
    #asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    #ax.set_aspect(asp)
    #ax.spines['top'].set_color('red')
    #ax2.xaxis.label.set_color('red')
    #ax2.tick_params(axis='x', colors='red')
    #fig.show()
    #fig.savefig(figname)
    return [ax,axn]
    """Execution"""
import os
def plots_for_each_casestudy(cal,mod,rad,lat_range,h_range,title_label_3,figname,loc,dpi,figaspect):
    import numpy as np


    #here we extract the data from the required cloud fields
    import matplotlib.pyplot as plt
    diff,ctt=btd2(mod)
    from file_god_contains_functions import calipso_sort
    colour, phase1,phase2,diff,_2105_band,_1600_band,_1200_band,_800,ctt,lat,lon=btd1(mod,rad)
    Image2,height_array,phase,index,lat1,lon1=calipso_sort(cal,400,3500)
    x,y=co_locate(cal,mod)
    if loc==1:
        y=np.array(y)
        c=np.where((y<970)&(y>710))
    else:
        y = np.array(y)
        c = np.where((y > 690))
    #location of transition case study 1
    ##location of transition case study 1
    #c=range(670,770)+range(880,980)
    x1=x[0]
    x2=x[1]#x1, x2 are the respective positions of the modis images
    #it is neccessary to use some of t

    #creating the figure
    #width,height=plt.figaspect(figaspect)
    #fig,ax = plt.subplots(2,2,figsize=(figaspect[0],figaspect[1]),dpi=dpi)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.08])
    ax1 = plt.subplot(gs[0,0])
    ax2 = plt.subplot(gs[0,1])
    ax3=plt.subplot(gs[1,0])
    ax4=plt.subplot(gs[1, 1])
    #x1=ax[0,0]#fig.add_subplot(221)
    #ax2=ax[0,1]#fig.add_subplot(222)
    #ax3=ax[1,0]#fig.add_subplot(223)
    #ax4=ax[1,1]#fig.add_subplot(224)
    #ax=[ax1,ax2,ax3,ax4]
   # plt.subplots_adjust(wspace = 0.33,hspace=0.33)


    #begin our plots with the satellite imagery in ax1


    a=ax1.pcolormesh(lon[:,:1350],lat[:,:1350],colour[0,:,:1350],vmin=0,vmax=0.3,cmap='Greys_r')
    cbar=plt.colorbar(a,ticks=[0,0.1,0.2,0.3,0.4],ax=ax1)
    ax1.plot(lon1[y],lat1[y],label='CALIOP Track',color='r')
    ax1.plot(lon1[y[c]],lat1[y[c]],label='Phase Transition',color='b')
    leg=ax1.legend(loc='upper right',frameon=True)
    cbar.set_label('0.6 $\mu m$  Reflectivity',size=8)
    cbar.ax.tick_params(labelsize=8)
    #leg=ax1.legend(['CALIOP Liquid','CALIOP Ice'],loc='upper right',frameon=True)
    leg.get_frame().set_edgecolor('k')
    ax1.set_title('(a)',fontsize=12)
    ax1.set_xlabel('Longitude ($\degree$)')
    ax1.set_ylabel('Latitude ($\degree$)')
    #asp = np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0]
    #ax1.set_aspect(asp)


    #the zoomed in imagery in the second axis

    ax2.set_title('(b)',fontsize=12)
    im=ax2.imshow(colour[0,1200:1700,250:950],vmin=0,vmax=0.28,cmap='Greys_r',aspect='auto')
    cbar = plt.colorbar(im, ticks=[0, 0.1, 0.2],ax=ax2)
    cbar.set_label('0.6 $\mu m$  Reflectivity',size=8)
    cbar.ax.tick_params(labelsize=8)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    #asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    #ax2.set_aspect(asp)


    #the third image is the complex profile of cloud phase
    fig1=case_study_btd_sim(ax3,cal,mod,lat_range,h_range,title_label_3,figname)

    #lastly we can sample the data on the location of the transition section
    diff, ctt=btd2(mod)
    #from pylab import *
    c1=np.where(phase[c]==1)
    c2=np.where(phase[c]==2)
    ax4.plot(ctt[x1[c],x2[c]][c1],diff[x1[c],x2[c]][c1],'bo',markersize=5)
    ax4.plot(ctt[x1[c],x2[c]][c2],diff[x1[c],x2[c]][c2],'ro',markersize=5)
    ax4.set_xlim(240,255)
    ax4.set_ylim(-0.7,0.7)
    ax4.tick_params(axis=u'both', which=u'both', length=3)
    xticks=np.arange(240,256,5)
    ax4.set_xticks(xticks)
    ax4.set_xticklabels(xticks)
    yticks=np.linspace(-0.6,0.6,7)
    ax4.set_yticks(yticks)
    ax4.set_yticklabels(yticks)
    leg=ax4.legend(['CALIOP Liquid','CALIOP Ice'],loc='upper right',frameon=True)
    leg.get_frame().set_edgecolor('k')
    ax4.set_xlabel('MODIS 11 $\mu m$ Brightness Temperature (K)')
    #ax4.set_ylabel('BTD (K)')
    ax4.set_title('(d)',fontsize=12)
   #asp = np.diff(ax4.get_xlim())[0] / np.diff(ax4.get_ylim())[0]
    #ax4.set_aspect(asp)

    plt.show()
plots_for_each_casestudy(cal, mod, rad,[-60, -62], [1000, 3500], '(c)', 'case_study_1_c.pdf',1,100,[10,10])
#matching the resolution of the images

plots_for_each_casestudy(cal, mod, rad,[-59.5, -63], [-500, 3500], '(c)', 'case_study_1_c.pdf',2,100,0.8)
#figure()
#pcolormesh(Y,X,infrared[:,:1350])
#the function below creates the plotting parameters and is found in file_god_contains_functions
#TODO note that MODIS tends to over classify ice around the edges
from file_god_contains_functions import *
import matplotlib.pyplot as plt
#plotting_para()
"""here is the script for creating a colour image"""

#colour image
mod='C:\Users\Neelesh\OneDrive\_2017\Honours Project\Colocation_Data\COLOC_2015_09_01(not done)\MYD06_L2.A2015244.1220.006.2015245162222.hdf'
rad='C:\Users\Neelesh\OneDrive\_2017\Honours Project\Colocation_Data\COLOC_2015_09_01(not done)\MYD021KM.A2015244.1220.006.2015245152631.hdf'
cal="C:\Users\Neelesh\OneDrive\Research Paper\Data_files\CAL_LID_L2_VFM-Standard-V4-10.2008-08-21T13-55-31ZD.hdf"
mod="C:\Users\Neelesh\OneDrive\Research Paper\Data_files\MYD06_L2.A2008234.1355.006.2013350215759.hdf"
rad="C:\Users\Neelesh\OneDrive\Research Paper\Data_files\MYD021KM.A2008234.1355.006.2012069224530.hdf"
# cbar.ax1.tick_params(labelsize=10)
fig.show()
fig.savefig('visible_image_case_study_2.pdf')





"""here is the script for showing the satellite overpass"""



plt.savefig('case_study12_location.pdf')









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
    #X,Y=np.repeat(np.repeat(lat2,5,axis=0),5,axis=1).reshape(2030,1350, order='F'),np.repeat(np.repeat(lon2,5,axis=0),5,axis=1).reshape(2030,1350,order='F')
    m.fillcontinents(color='coral', lake_color='aqua')
    #m.pcolormesh(Y,X,colour[1,:,:1350])
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
