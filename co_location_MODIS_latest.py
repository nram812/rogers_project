"""testing file to analyse solar zenith depedance"""
"""this script examines the distributions of MODIS along each path, comparisons can be made to CALIOP later"""

        #print file_dir

#note that the temperature bins have been investigated
"""here you are using the MODIS heights please make a note of this"""
def MODIS_file(filedir, height_lim, t_bins):
    from pyhdf import SD
    import numpy as np

    """new edits include examining the frequency
     of clear sky retrievals, underdetermined etc and the frequency of low clouds"""
    #note that all the orbital paths are small

    f = SD.SD(file_dir)
    sds = f.select('cloud_top_temperature_1km')
    temp = 0.01 * (sds.get()[:2030,0] + 15000)  # sampling on the same grid as latitude
    sds = f.select('Cloud_Phase_Optical_Properties')
    phase = sds.get()[:2030, :0]

    sds = f.select('Solar_Zenith_Day')
    zenith = 0.009999999776482582 * np.repeat(np.repeat(sds.get(), 5, axis=0), 5, axis=1).reshape(2030, 1350)
    # consider getting rid of the fill values
    sds = f.select('cloud_top_height_1km')
    height = sds.get()[:2030, :0]
    sds = f.select('Latitude')
    lat = np.repeat(sds.get()[:406,0], 5, axis=0).reshape(2030)
    sds=f.select('Brightness_Temperature')
    bt=np.repeat(sds.get()[:,:406,0], 5, axis=1).reshape(7,2030)#scaling factor is requried
    bt=(bt+15000)*0.009999999776482582 #correction applied
    btd=bt[0]-bt[1]
    new_alg=np.zeros([2030])

    # the above code extracts the relevant datasets from each MODIS file

    # filtering cloud by altitude
    heights = np.where((height < height_lim) & (
                (lat < -45) & (lat > -65)) & (temp>220))  # less than 4000 meteters
    # filtered by both positive and negative zenith angles
    # sample the new fields based on these thresholds
    temp = temp[heights]
    phase = phase[heights]
    zenith = zenith[heights]
    height = height[heights]
    btd=btd[heights]

    ice_loc = np.where(phase == 3)  # location of pixels in the liquid and ice fields
    liq_loc = np.where(phase == 2)
    #clr_loc = np.where(phase == 1)
    no_retrieval = np.where(phase == 0) #this is likely the case of a solar zenith angle deficiey
    pixel_counts=len(no_retrieval[0])
    """pixel counts need to be stored"""

    hist_un = np.histogram(temp[no_retrieval], bins=t_bins)[0]
    hist_water = np.histogram(temp[liq_loc], bins=t_bins)[0]
    hist_ice = np.histogram(temp[ice_loc], bins=t_bins)[0]
    """note a more rigorous cluter analysis should be presented"""
    ice_loca = np.where(btd > 0.18)  # location of pixels in the liquid and ice fields
    liq_loca = np.where(btd < -0.28)
    un_loca = np.where((btd>-0.28)&(btd<0.18))
    hist_un_cal = np.histogram(temp[un_loca], bins=t_bins)[0]
    hist_water_cal = np.histogram(temp[liq_loca], bins=t_bins)[0]
    hist_ice_cal = np.histogram(temp[ice_loca], bins=t_bins)[0]


    # histograms of the temperatures of ice and liquid pixel numbers

    return hist_water, hist_ice, hist_un ,hist_water_cal,hist_ice_cal,hist_un_cal,pixel_counts
import numpy as np
import os
dir1='F:/SH_data2/'
hist_water, hist_ice, hist_un, hist_water_cal, hist_ice_cal, hist_un_cal, pixel_counts=np.zeros()
for filename in os.listdir(dir1):
    if 'hdf' in filename:
        file_dir=dir1+filename
        hist_water, hist_ice, hist_un, hist_water_cal, hist_ice_cal, hist_un_cal, pixel_counts=MODIS_file(file_dir, 3200, np.arange(230,290,1))
def main():
    import os
    import warnings
    import numpy as np
    ice = np.zeros(len(np.arange(240, 290, 1)))[:-1]
    water = np.zeros(len(np.arange(240, 290, 1)))[:-1]
    zenith_ice = np.zeros(len(np.arange(-90, 90, 1)))[:-1]
    zenith_liq = np.zeros(len(np.arange(-90, 90, 1)))[:-1]
    counter = 0
    for filename in os.listdir(
            "F:/Large Scale MODIS files for Images/modis FLES/Data_dwn_18th_april/terra_jan_verification/"):
        print filename
        try:
            water1, ice1, t1, zen_ice, zen_liq = MODIS_file(
                "F:/Large Scale MODIS files for Images/modis FLES/Data_dwn_18th_april/terra_jan_verification/" + filename,
                4000, 45, np.arange(240, 290, 1))
            ice = ice + ice1
            water = water + water1
            zenith_ice = zenith_ice + zen_ice
            zenith_liq = zenith_liq + zen_liq

            counter = counter + 1
            print counter
            if counter % 10 == 0:
                np.save('MODIS_DATA_jan_VERIFICATION_sza_45', [ice, water, zenith_ice, zenith_liq])
                print 'data_saved'
        except:
            counter = counter + 1
            print 'error'


def main2():
    import os
    import warnings
    import numpy as np
    ice = np.zeros(len(np.arange(240, 290, 1)))[:-1]
    water = np.zeros(len(np.arange(240, 290, 1)))[:-1]
    zenith_ice = np.zeros(len(np.arange(-90, 90, 1)))[:-1]
    zenith_liq = np.zeros(len(np.arange(-90, 90, 1)))[:-1]
    counter = 0
    for filename in os.listdir("F:/Large Scale MODIS files for Images/modis FLES/Data_dwn_18th_april/MODIS_august/"):
        print filename
        try:
            water1, ice1, t1, zen_ice, zen_liq = MODIS_file(
                "F:/Large Scale MODIS files for Images/modis FLES/Data_dwn_18th_april/MODIS_august/" + filename, 4000,
                81, np.arange(240, 290, 1))
            ice = ice + ice1
            water = water + water1
            zenith_ice = zenith_ice + zen_ice
            zenith_liq = zenith_liq + zen_liq
            counter = counter + 1
            print counter
            if counter % 10 == 0:
                np.save('MODIS_DATA_aUGUST', [ice, water, zenith_ice, zenith_liq])
                print 'data_saved'
        except:
            counter = counter + 1
            print 'error'


figure()
##main()
plot(t_bins[:-1], ice * 1.0 / sum(ice))
plot(t_bins[:-1], water * 1.0 / sum(water))
# figure()
# plot(np.arange(-90,90,1)[:-1],zeni/sum(zeni))
# plot(np.arange(-90,90,1)[:-1],zenl/sum(zenl))
# ice retrievals are sharply peaked at 75 degree viewing angle whereas a liquid is peaked differently
# also note that the height of the jump is depedant on the low cloud height filter (look at the work you did a couple of summers ago).
# colocate observations with CALIOP and store the viewing angles


# note in your figure of glaciation probability, the low altitude cloud shown is of CALIOP's height which is different to the results shown in main() and main2()
# in the paper show that estimates

# we would expect similar distributions of liquid and ice as a function of solar zenith angle, as phase is an independant quantity, however the distributions differ during the winter (note need to check summer)

# write code that computes the glaciation probability as a function of solar zenith angle
# main2()
"""Note when the zenith angle is set to a maximum of pm 60 during the winter there are no retrievals over the entire southern ocean"""  # seems weird
main()

# in the marchant paper there is still ambiguity in the phase relationship, as in that too, there is overlap in the cloud effective radius tests
# Hence your method is somewhat similar
# the colocation proceedure only uses 2 months in particular years nov and july , considers all types of cloud also.
# also create contingency tables for MODIS
# you also would need to do the entire Southern Ocean to verify that your method is correct.