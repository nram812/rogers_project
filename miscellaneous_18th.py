#Note the files below are downloaded from the onedrive as they are not visible on the computer
#TODO run these files tomorrow to improve the figures, note the files have been downloaded
#note the files have been added to "research paper folder on windowns

cal_file='CAL_LID_L2_VFM-Standard-V4-10.2015-09-01T12-20-46ZD.hdf'
mod_file='MYD021KM.A2015244.1220.006.2015245152631.hdf'
mod_l2_file='MYD06_L2.A2015244.1220.006.2015245162222.hdf'
mac_dir='/Users/neeleshrampal/Downloads/'
desktop_dir='C:/Users/Neelesh/OneDrive/Research Paper/'
cal_file=desktop_dir+cal_file
mod_file=desktop_dir+mod_file
mod_l2_file=desktop_dir+mod_l2_file
#the directories are changed for the different files
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
x,y=co_locate(cal_file,mod_file)
#the positions are outputted in the file shown.
x1=x[0]
x2=x[1]
#x1, x2 are the respective positions of the modis images




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




    blue = (sds.get()[2]+sds.attributes()['reflectance_offsets'][2]) * sds.attributes()['reflectance_scales'][2]
    grn = (sds.get()[3]+sds.attributes()['reflectance_offsets'][3]) * sds.attributes()['reflectance_scales'][3]
    # sds=f.select('EV_500_Aggr1km_RefSB')
    #TODO add this to your log book to remember that you updated your plots with a linear filter to amplify colour
    colour=np.zeros([3,2030,1354])
    colour[0]=red
    colour[1]=grn
    colour[2]=blue
    c=np.where(colour>0.15)
    a1=0.15
    a2=0.25
    max_val=colour.max()
    c2=(a2-a1)/(max_val-0.15)
    c1=a2-c2*max_val
    colour[c]=c1+c2*colour[c]
    #you need to filter the values that are nan
    figure()
    imshow(colour.T*5)
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
    m.drawcoastlines()
    X,Y=m(np.repeat(np.repeat(lat2,5,axis=0),5,axis=1).reshape(2030,1350, order='F'),np.repeat(np.repeat(lon2,5,axis=0),5,axis=1).reshape(2030,1350,order='F'))
    #m.fillcontinents(color='coral', lake_color='aqua')
    m.pcolor(Y,X,colour[1,:,:1350])
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


