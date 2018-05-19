"""testing file to analyse solar zenith depedance"""
def MODIS_file(filename,height_lim,zen_lim,t_bins):
    from pyhdf import SD
    import numpy as np

    
    f=SD.SD(filename)
    sds=f.select('cloud_top_temperature_1km')
    temp=0.01*(sds.get()[:2030,:1350]+15000)#sampling on the same grid as latitude
    sds=f.select('Cloud_Phase_Optical_Properties')
    phase=sds.get()[:2030,:1350]
    
    sds=f.select('Solar_Zenith_Day')
    zenith=0.009999999776482582*np.repeat(np.repeat(sds.get(),5,axis=0),5,axis=1).reshape(2030,1350)
    #consider getting rid of the fill values
    sds=f.select('cloud_top_height_1km')
    height=sds.get()[:2030,:1350]
    sds=f.select('Latitude')
    lat=np.repeat(np.repeat(sds.get(),5,axis=0),5,axis=1).reshape(2030,1350)
    #the above code extracts the relevant datasets from each MODIS file
    
    #filtering cloud by altitude
    heights=np.where((height<height_lim)&(((zenith<zen_lim)&(zenith>-zen_lim)))&((lat<-45)&(lat>-65)))#less than 4000 meteters
    #filtered by both positive and negative zenith angles
    #sample the new fields based on these thresholds
    temp=temp[heights]
    phase=phase[heights]
    zenith=zenith[heights]
    height=height[heights]
    lat=lat[heights]
    ice_loc=np.where(phase==3)#location of pixels in the liquid and ice fields
    liq_loc=np.where(phase==2)
    hist_water=np.histogram(temp[liq_loc],bins=t_bins)[0]
    
    hist_ice=np.histogram(temp[ice_loc],bins=t_bins)[0]
    #histograms of the temperatures of ice and liquid pixel numbers
    
    return hist_water,hist_ice,t_bins,np.histogram(zenith[ice_loc],bins=np.arange(-90,90,1))[0],np.histogram(zenith[liq_loc],bins=np.arange(-90,90,1))[0]


def main():
    import os
    import warnings
    import numpy as np
    ice=np.zeros(len(np.arange(240,290,1)))[:-1]
    water=np.zeros(len(np.arange(240,290,1)))[:-1]
    zenith_ice=np.zeros(len(np.arange(-90,90,1)))[:-1]
    zenith_liq=np.zeros(len(np.arange(-90,90,1)))[:-1]
    counter=0
    for filename in os.listdir("F:/Large Scale MODIS files for Images/modis FLES/Data_dwn_18th_april/terra_jan_verification/"):
        print filename
        try :
            water1,ice1,t1,zen_ice,zen_liq=MODIS_file("F:/Large Scale MODIS files for Images/modis FLES/Data_dwn_18th_april/terra_jan_verification/"+filename,4000,45,np.arange(240,290,1))
            ice=ice+ice1
            water=water+water1
            zenith_ice=zenith_ice+zen_ice
            zenith_liq=zenith_liq+zen_liq
            
            counter=counter+1
            print counter
            if counter%10==0:
                np.save('MODIS_DATA_jan_VERIFICATION_sza_45',[ice,water,zenith_ice,zenith_liq])
                print 'data_saved'
        except:
            counter=counter+1
            print 'error'
def main2():
    import os
    import warnings
    import numpy as np
    ice=np.zeros(len(np.arange(240,290,1)))[:-1]
    water=np.zeros(len(np.arange(240,290,1)))[:-1]
    zenith_ice=np.zeros(len(np.arange(-90,90,1)))[:-1]
    zenith_liq=np.zeros(len(np.arange(-90,90,1)))[:-1]
    counter=0
    for filename in os.listdir("F:/Large Scale MODIS files for Images/modis FLES/Data_dwn_18th_april/MODIS_august/"):
        print filename
        try :
            water1,ice1,t1,zen_ice,zen_liq=MODIS_file("F:/Large Scale MODIS files for Images/modis FLES/Data_dwn_18th_april/MODIS_august/"+filename,4000,81,np.arange(240,290,1))
            ice=ice+ice1
            water=water+water1
            zenith_ice=zenith_ice+zen_ice
            zenith_liq=zenith_liq+zen_liq
            counter=counter+1
            print counter
            if counter%10==0:
                np.save('MODIS_DATA_aUGUST',[ice,water,zenith_ice,zenith_liq])
                print 'data_saved'
        except:
            counter=counter+1
            print 'error'
figure()
##main()
plot(t_bins[:-1],ice*1.0/sum(ice))  
plot(t_bins[:-1],water*1.0/sum(water))  
#figure()
#plot(np.arange(-90,90,1)[:-1],zeni/sum(zeni))
#plot(np.arange(-90,90,1)[:-1],zenl/sum(zenl))
#ice retrievals are sharply peaked at 75 degree viewing angle whereas a liquid is peaked differently
#also note that the height of the jump is depedant on the low cloud height filter (look at the work you did a couple of summers ago).
#colocate observations with CALIOP and store the viewing angles


#note in your figure of glaciation probability, the low altitude cloud shown is of CALIOP's height which is different to the results shown in main() and main2()
#in the paper show that estimates 

#we would expect similar distributions of liquid and ice as a function of solar zenith angle, as phase is an independant quantity, however the distributions differ during the winter (note need to check summer)

#write code that computes the glaciation probability as a function of solar zenith angle
#main2()
"""Note when the zenith angle is set to a maximum of pm 60 during the winter there are no retrievals over the entire southern ocean""" #seems weird
main()

#in the marchant paper there is still ambiguity in the phase relationship, as in that too, there is overlap in the cloud effective radius tests
#Hence your method is somewhat similar
#the colocation proceedure only uses 2 months in particular years nov and july , considers all types of cloud also.
#also create contingency tables for MODIS
# you also would need to do the entire Southern Ocean to verify that your method is correct.