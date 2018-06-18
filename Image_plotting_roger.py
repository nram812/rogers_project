#from file_god_contains_functions import *
plotting_para()
import numpy as np
cal="C:\Users\Neelesh\OneDrive\Research Paper\Data_files\CAL_LID_L2_VFM-Standard-V4-10.2008-08-21T13-55-31ZD.hdf"
mod="C:\Users\Neelesh\OneDrive\Research Paper\Data_files\MYD06_L2.A2008234.1355.006.2013350215759.hdf"
rad="C:\Users\Neelesh\OneDrive\Research Paper\Data_files\MYD021KM.A2008234.1355.006.2012069224530.hdf"
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

    # sds=f.select('EV_250_Aggr1km_RefSB')

    # band_600_r=sds.get()[0]*sds.attributes()['reflectance_scales'][0]
    # band_800_r=sds.get()[1]*sds.attributes()['reflectance_scales'][1]

    # sds=f.select('EV_500_Aggr1km_RefSB')

    # plot the brightness temperature differences too.
    # sds=f.select('EV_500_Aggr1km_RefSB')
    # band_6_r=sds.get()[-1]*sds.attributes()['reflectance_scales'][-1]

    # sds=f.select('EV_1KM_Emissive')
    # band_7_r=(sds.get()[-5]+sds.attributes()['radiance_offsets'][-5])*sds.attributes()['radiance_scales'][-5]
    return diff, ctt
def case_study_btd_sim(cal,mod,lat_range,h_range,title_label,figname):
    import matplotlib
    plotting_para()
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
    width,height=plt.figaspect(0.8)
    fig,ax1=plt.subplots(figsize=(width,height),dpi=100)
    colorsList = ['indigo','#FFE600']
    CustomCmap = matplotlib.colors.ListedColormap(colorsList)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    red_patch = mpatches.Patch(color='#FFE600', label='CALIOP Ice')
    blue_patch = mpatches.Patch(color='indigo', label='CALIOP Liquid')
    red_patch1 = Line2D(range(1), range(1), color='r', marker='o', markersize=6, markerfacecolor="r",
                        linewidth=0, label='MODIS BTD')  # mpatches.Circle((3,3),color='#0067acff',marker='o')

    #blue_patch1 = mpatches.Patch(color='r', label='Undetermined')
    x,y=co_locate(cal,mod)
    ax1.tick_params(axis=u'both', which=u'both', length=3)
    #x,y=co_locate(cal,mod)
    im=ax1.pcolormesh(Y[y,:][::-1],X[y,:][::-1],Zm.T[y,:][::-1],cmap=CustomCmap,alpha=0.6,edgecolor='None')
    ax1.legend(handles=[red_patch,blue_patch,red_patch1], loc='upper right', ncol=1, fontsize=10,handletextpad=0.4, columnspacing=0.4)
    ax1.set_title(title_label,fontsize=12)
    ax1.set_ylim([hmin,hmax])
    ax1.set_xlabel('Latitude ($\degree$)')
    ax1.set_ylabel('CALIOP Cloud Height (m)')
    xticks = np.arange(latmin,latmax+0.5,0.5)[::-1]
    yticks = range(hmin,hmax,500)

    xticklabels=xticks
    yticklabels=range(hmin,hmax,500)

    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticklabels)

    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    #fig.show()
    ax1.tick_params(axis=u'both', which=u'both', length=3)
    #ax1.legend()
    #ax1.set_yticklabels(np.arange(0,500,3500))
    ax2=ax1.twinx()
    ax2.plot(laty[y],diff[x[0],x[1]],'ro',markersize=6,linewidth=2)
    ax2.set_ylabel('MODIS BTD (K)',color='red')
    ax2.set_ylim([-0.6,0.6])
    ax2.set_xlim([latmin,latmax])
    ax2.tick_params(axis='y',colors='red', length=3)
    ax2.spines['right'].set_color('red')
    #ax.spines['top'].set_color('red')
    #ax2.xaxis.label.set_color('red')
    #ax2.tick_params(axis='x', colors='red')
    fig.show()
    #fig.savefig(figname)
    return fig
    """Execution"""
import os
os.chdir('C:/Users/Neelesh/OneDrive')
fig1=case_study_btd_sim(cal, mod, [-60, -62], [1000, 3500], '(c)', 'case_study_2_c.pdf')
fig2=case_study_btd_sim(cal, mod, [-59.5, -63], [-500, 3000], '(c)', 'case_study_1_c.pdf')
cal = 'C:\Users\Neelesh\OneDrive\_2017\Honours Project\Colocation_Data\COLOC_2015_09_01(not done)\CAL_LID_L2_VFM-Standard-V4-10.2015-09-01T12-20-46ZD.hdf'
mod = "C:\Users\Neelesh\OneDrive\_2017\Honours Project\Colocation_Data\COLOC_2015_09_01(not done)\MYD06_L2.A2015244.1220.006.2015245162222.hdf"  # here is th eforum
def colour_image(cal,mod):
    def btd1(mod, rad):
        from pyhdf import SD
        f = SD.SD(mod)
        sds = f.select('Brightness_Temperature')
        # sds=f.select('Brightness_Temperature')
        # note that the shape of the file, is currently (406,270) this shape can be altered depedending on the file type.
        btd = np.repeat(np.repeat((sds.get()[:, 0:406, 0:270] + 15000) * 0.01, 5, axis=1), 5, axis=2).reshape(7, 2030,
                                                                                                              1350)
        # data is over sampled
        # TODO include over sampling
        diff = btd[0] - btd[1]
        sds = f.select('cloud_top_temperature_1km')
        ctt = (sds.get() + 15000) * 0.01
        sds = f.select('Cloud_Phase_Optical_Properties')
        phase1 = sds.get()
        sds = f.select('Cloud_Phase_Infrared_1km')
        phase2 = sds.get()
        lat=f.select('Latitude').get()
        lon =f.select('Longitude').get()
        f = SD.SD(rad)

        sds = f.select('EV_250_Aggr1km_RefSB')

        red = (sds.get()[0] + sds.attributes()['reflectance_offsets'][0]) * sds.attributes()['reflectance_scales'][0]
        sds = f.select('EV_1KM_RefSB')

        blue = (sds.get()[2] + sds.attributes()['reflectance_offsets'][2]) * sds.attributes()['reflectance_scales'][2]
        grn = (sds.get()[3] + sds.attributes()['reflectance_offsets'][3]) * sds.attributes()['reflectance_scales'][3]
        # sds=f.select('EV_500_Aggr1km_RefSB')
        # TODO add this to your log book to remember that you updated your plots with a linear filter to amplify colour
        colour = np.zeros([3, 2030, 1354])
        colour[0] = red
        colour[1] = grn
        colour[2] = blue
        # c=np.where(colour>0.28)
        # a1=0.28
        # a2=0.37
        # max_val=colour[:,:,:].max()
        # c2=(a2-a1)/(max_val-a1)
        # c1=(1-c2)*a1
        # colour[c]=c1+c2*colour[c]
        return colour, phase1, phase2, diff, ctt,lat,lon

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
                c = np.where((c3 == c3.min()) & (c3.min() < 0.025))

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
            c = np.where((c3 == c3.min()) & (c3.min() < 0.025))

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
x, y = co_locate(cal, mod)
mod=mod_file
rad=mod_l2_file
colour,opt,infrared,diff,ctt=btd1(mod,rad)
Image2,height_array,phase,index,lat1,lon1=calipso_sort(cal,400,3500)
lat = np.repeat(lat1, 5, axis=0)
lon = np.repeat(lon1, 5, axis=0)

#please see micelanous 1906 for further details











"""just need to figure out how to adjust the x tick labels"""


# can look into more detail regarding colocation and visual image