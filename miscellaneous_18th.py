cal_file='/Users/neeleshrampal/Downloads/CAL_LID_L2_VFM-Standard-V4-10.2015-09-01T12-20-46ZD.hdf'
mod_file='/Users/neeleshrampal/Downloads/MYD021KM.A2015244.1220.006.2015245152631.hdf'

mod_l2_file='/Users/neeleshrampal/Downloads/MYD06_L2.A2015244.1220.006.2015245162222.hdf'
x,y=co_locate(cal_file,mod_file)

#x=coords_x
#y=coords_y
#z=lat1
def btd1(mod, rad):
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
    f = SD.SD(rad)

    sds = f.select('EV_250_Aggr1km_RefSB')

    red = (sds.get()[0]+sds.attributes()['reflectance_offsets'][0]) * sds.attributes()['reflectance_scales'][0]
    sds = f.select('EV_1KM_RefSB')




    blue = (sds.get()[2]+sds.attributes()['reflectance_offsets'][2]) * sds.attributes()['reflectance_scales'][2]
    grn = (sds.get()[3]+sds.attributes()['reflectance_offsets'][3]) * sds.attributes()['reflectance_scales'][3]
    # sds=f.select('EV_500_Aggr1km_RefSB')
    colour=np.zeros([3,2030,1354])
    colour[0]=red
    colour[1]=grn
    colour[2]=blue
    figure()
    imshow(colour.T*3)
    show()
    # plot the brightness temperature differences too.
    sds = f.select('EV_500_Aggr1km_RefSB')
    band_6_r = sds.get()[-1] * sds.attributes()['reflectance_scales'][-1]



