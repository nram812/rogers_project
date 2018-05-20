def cal_plot(cal,mod):
    from file_god_contains_functions1 import *
    cal='F:/Calipso_Files/VFM_northern/CAL_LID_L2_VFM-Standard-V4-10.2008-06-25T14-04-04ZD.hdf'
    mod='C:/Users/Neelesh/OneDrive/Honours Project/Colocation_Data/COLOC 2008_06_25/MYD06_L2.A2008177.1400.006.2013348153940.hdf'
    rad='C:/Users/Neelesh/OneDrive/Honours Project/Colocation_Data/COLOC 2008_06_25/MYD021KM.A2008177.1400.006.2012069014813.hdf'
    diff,ctt,phase1,phase2,cth,lat,lon,tau=btd_small(mod)
    x1,y1=co_locate2(cal,mod)
    Image2,height_array,phase,index,lat1,lon1=calipso_sort(cal,400,3500)
    #band_600_r-band_6_r,band_6_r/band_600_r,band_600_r-band_800_r,band_800_r/band_600_r,band_600_r,band_6_r
    x=btd1(rad)
    vis_image=x[-2][0:2030,:1350]
    lat=np.repeat(np.repeat(lat,5,axis=0),5,axis=1)
    lon=np.repeat(np.repeat(lon,5,axis=0),5,axis=1)
    width,height=plt.figaspect(0.8)
    from mpl_toolkits.basemap import Basemap, cm
    # requires netcdf4-python (netcdf4-python.googlecode.com)
    fig, ax11 = plt.subplots(nrows=1,figsize=(width,height), dpi=100)
    #vis_image[x1[0][700:850],x1[1][700:850]]=0.4
    #c=ax11.imshow(vis_image,cmap='viridis')
    c=ax11.pcolormesh(lon,lat,vis_image,cmap='viridis')
    colorbar(c)
    c.set_clim(0.0,0.48)
    ax11.plot(lon1[y1],lat1[y1],'y-')
    ax11.plot(lon1[y1][400:750],lat1[y1][400:750],'r-')
    ax11.legend(['CALIOP Track','Phase Transition'])
    ax11.set_xlabel('Longitude')
    ax11.set_ylabel('Latitude')
ax11.legend(['CALIOP Track','Phase Transition'])
    ax11.set_xlabel('Longitude')
    ax11.set_ylabel('Latitude')
    
    
    fig, ax11 = plt.subplots(nrows=1,figsize=(width,height), dpi=100)
   
    imshow(Image2[:,y1],cmap='viridis')
    py.figure()
    py.imshow(vis_image,cmap='viridis')
    clim(0,0.48)
    colorbar()
    ylabel('0.6$\mu m$ reflectivity')
    
    
    m =\
Basemap(lat_0=-60,lon_0=50,\
            llcrnrlat=-80,urcrnrlat=-50,\
            llcrnrlon=5,urcrnrlon=80,projection='mill')
# add wrap-around point in longitude.

# draw coastlines, state and country boundaries, edge of map.
    m.drawcoastlines()
    lons,lats=m(lon,lat)
    #m.drawparallels(np.arange(40.,66.,25.),labels=[1,0,0,0],fontsize=10)
    m.drawmeridians(np.arange(-180,180,15),labels=[1,0,0,0],fontsize=10)
    m.drawparallels(np.arange(-95.,-40.,15.),labels=[1,0,0,0],fontsize=10)
    m.pcolormesh(lons,lats,vis_image,cmap='viridis')
    clim(0,0.3)
    
    #plt.figure()
    #plt.imshow(Image2[:,:],cmap='viridis')
    #"""saturation in the bands"""
    #x=np.zeros([2030,1350])
    #c=np.where(diff>=0.35)
    #c2=np.where(diff<=-0.25)
    #x[c]=2
    #x[c2]=1
    #x[x1[0],x1[1]]=np.nan
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    
    import matplotlib
    colorsList = ['indigo','#FFE600']
    CustomCmap = matplotlib.colors.ListedColormap(colorsList)
    
    width, height = plt.figaspect(0.8)
    #fig = plt.figure(figsize=(width,height), dpi=100)
    fig, ax11 = plt.subplots(nrows=1,figsize=(width,height), dpi=100)
    ax1=ax11#[1]
    c=np.where(phase[y1]==1)
    ax1.plot(lat1[y1][c][::5],diff[x1[0],x1[1]][c][::5], color='r',marker='o',linewidth=0)
    xlim(-58,-62)
    c=np.where(phase[y1]==2)
    ax1.plot(lat1[y1][c][::5],diff[x1[0],x1[1]][c][::5], color='#0EBFE9',marker='o',linewidth=0)
    xlim(-58,-62)
    ax1.set_xlabel('Latitude')
   # # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('BTD(K)')
    
    ax2 = ax1
    #xtickloc = ax1.get_yticks()
    #ax2.set_yticks(xtickloc)
    ytick1=np.arange(-500,8200,1000)
    ax2.set_yticklabels([str(y) for y in ytick1])
    red_patch = mpatches.Patch(color='#FFE600', label='Ice')
    blue_patch = mpatches.Patch(color='indigo', label='Liquid')
    blue_patch1 = Line2D(range(1), range(1), color='#BFEFFF', marker='o',markersize=10, markerfacecolor='#0EBFE9',linewidth=0)#mpatches.Circle((3,3),color='#0067acff',marker='o')
    red_patch1 = Line2D(range(1), range(1), color='r', marker='o',markersize=10, markerfacecolor="r",linewidth=0)#mpatches.Circle((3,3),color='#0067acff',marker='o')
    
    #handles = handlesL + handlesR
    #labels = labelsL + labelsR
    plt.legend(handles=[red_patch,blue_patch,blue_patch1,red_patch1], loc='upper right', ncol=2, fontsize=12,
            handletextpad=0.4, columnspacing=0.4)
    #plt.legend(handles=[blue_patch,red_patch])
    ax2.imshow(Image2[0:290,:][::-1],extent=[lat1[0],lat1[-1],-0.5,8.200],cmap=CustomCmap,alpha=0.55,clim=[1,2])
    ax2.set_ylabel('Cloud Height (km)')
    fig.tight_layout()
    plt.show()
#import matplotlib
cal='C:/Users/Neelesh/OneDrive/Honours Project/Colocation_Data/COLOC_08_21/CAL_LID_L2_VFM-Standard-V4-10.2008-08-21T13-55-31ZD.hdf'
#
mod='C:/Users/Neelesh/OneDrive/Honours Project/Colocation_Data/COLOC_08_21/MYD06_L2.A2008234.1355.006.2013350215759.hdf'
#cmap = matplotlib.cm.get_cmap('Blues')
#
#rgba = cmap(0.5)
#print(rgba)
#plt.close('all')
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
diff,ctt,phase1,phase2,cth,lat,lon,tau,btd1=btd(mod)
x1,y1=co_locate2(cal,mod)
phase,lat1,lon1=calipso_sort(cal,400,3500)[0],calipso_sort(cal,400,3500)[-2],calipso_sort(cal,400,3500)[-1]

"""saturation in the bands"""
x=np.zeros([2030,1350])
x[x1[0],x1[1]]=np.nan
c=np.where(diff>=0.10)
c2=np.where(diff<=-0.10)
c3=np.where((diff>-0.1) &(diff<0.1))
x[c]=2
x[c2]=1
x[c3]=3

py.figure()
py.pcolormesh(lon,lat,diff)
ylabel('Latitude')
xlabel('Longitude')
py.plot(lon1[y1],lat1[y1],'r--')
py.plot(lon1[y1][700:900],lat1[y1][700:900],'b-')

phase21=phase[:,y1]
phase4=np.zeros(phase21.shape)*np.nan
for i in range(phase21.shape[1]):
    c=np.where(phase21[:,i]>0)
    phase4[c[0],i]=abs(phase1[x1[0],x1[1]][i]-1)
phase3=np.zeros(phase21.shape)*np.nan
for i in range(phase21.shape[1]):
    c=np.where(phase21[:,i]>0)
    phase3[c[0],i]=abs(x[x1[0],x1[1]][i])
c=np.where((phase3<0))
phase3[c]=nan
#
width, height = plt.figaspect(1.8)
colorsList = ['indigo','#FFE600','r']
CustomCmap = matplotlib.colors.ListedColormap(colorsList)

#fig = plt.figure(figsize=(width,height), dpi=100)
fig, ax1 = plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True,figsize=(width,height), dpi=100)
plt.subplots_adjust(wspace=0.001, hspace=0.1)
ax1[0].imshow(phase21[0:290,:][::-1],extent=[lat1[y1][-1],lat1[y1][0],-0.500,8.200],cmap=CustomCmap,clim=[1,4])
#ax1[1].imshow(phase3[0:290,:][::-1],extent=[lat1[y1][-1],lat1[y1][0],-0.500,8.200],cmap=CustomCmap,clim=[1,4])
ax1[1].imshow(phase4[0:290,:][::-1],extent=[lat1[y1][-1],lat1[y1][0],-0.500,8.200],cmap=CustomCmap,clim=[1,4])
#grid1=[i.grid() for i in ax1]
set1=[i.set_ylabel('Cloud Height (km)') for i in ax1]
ax1[1].set_xlabel('Latitude')

red_patch = mpatches.Patch(color='#FFE600', label='CALLIOP Ice')
blue_patch = mpatches.Patch(color='indigo', label='CALLIOP Liquid')
blue_patch1 = mpatches.Patch(color='r', label='Undetermined')
ax1[0].legend(handles=[red_patch,blue_patch,blue_patch1], loc='upper right', ncol=1, fontsize=8,
           handletextpad=0.4, columnspacing=0.4)
ax1[0].set_title('(a)',fontsize=12)
red_patch = mpatches.Patch(color='#FFE600', label='MODIS Revised Ice')
blue_patch = mpatches.Patch(color='indigo', label='MODIS Revised Liquid')
blue_patch1 = mpatches.Patch(color='r', label='Undetermined')
#ax1[1].legend(handles=[red_patch,blue_patch,blue_patch1], loc='upper right', ncol=1, fontsize=8,
           #handletextpad=0.4, columnspacing=0.4)
#ax1[1].set_title('(b)',fontsize=12)
red_patch = mpatches.Patch(color='#FFE600', label='MODIS Ice')
blue_patch = mpatches.Patch(color='indigo', label='MODIS Liquid')
blue_patch1 = mpatches.Patch(color='r', label='Undetermined')
ax1[1].legend(handles=[red_patch,blue_patch,blue_patch1], loc='upper right', ncol=1, fontsize=8,
           handletextpad=0.4, columnspacing=0.4)
ax1[1].set_title('(b)',fontsize=12)
fig.tight_layout()
plt.show()
#xtickloc = ax1.get_yticks()
#ax2.set_yticks(xtickloc)
#ytick1=np.arange(-500,8200,1000)
#ax2.set_yticklabels([str(y) for y in ytick1])
red_patch = mpatches.Patch(color='#4682b4', label='Ice')
blue_patch = mpatches.Patch(color='#0067acff', label='Liquid')
blue_patch1 = Line2D(range(1), range(1), color='#0067acff', marker='o',markersize=10, markerfacecolor='#0067acff')#mpatches.Circle((3,3),color='#0067acff',marker='o')
red_patch1 = Line2D(range(1), range(1), color='#4682b4', marker='o',markersize=10, markerfacecolor="slategray")#mpatches.Circle((3,3),color='#0067acff',marker='o')

#handles = handlesL + handlesR
#labels = labelsL + labelsR
plt.legend(handles=[red_patch,blue_patch,red_patch1,blue_patch1], loc='lower center', ncol=2, fontsize=8,
           handletextpad=0.4, columnspacing=0.4)
#plt.legend(handles=[blue_patch,red_patch])
ax2.imshow(Image2[0:290,y1][::-1],extent=[lat[y1][0],lat[y1][-1],-0.5,8.200],cmap='Blues_r')
ax2.set_ylabel('Cloud Height (km)')
fig.tight_layout()
plt.show()
#
#py.figure()
#py.imshow(x)
#py.figure()
#py.imshow(phase2)
#py.figure()
#py.pcolormesh(lon,lat,diff)
#py.figure()
#title('Calipso Algorithm')
#py.subplot(311)
#
#py.subplot(312)
#title('My Algorithm')
#py.imshow(phase3[0:290,:],extent=[lat1[y1][-1],lat1[y1][0],8.200,-0.500])
#py.subplot(313)
#title('MODIS cloud phase')
#py.imshow(phase4[0:290,:],extent=[lat1[y1][-1],lat1[y1][0],8.200,-0.500])
#width, height = plt.figaspect(1.1)
#
#fig = plt.figure(figsize=(width,height), dpi=100)
fig, ax1 = plt.subplots()
py.plot(c2[::10],g1[3],color='#00bfff',marker='o',markersize=1.5,linewidth=3.5)
py.plot(c2[::10],g2[3],color='#22316C',marker='o',markersize=1.5,linewidth=3.5)
ax1.grid()
ax1.set_xlabel('Brightness Temperature Difference (K)')
ax1.set_ylabel('Probability of Phase')
ax1.legend(['Ice','Liquid'],loc='upper centrer', ncol=1, fontsize=10,
           handletextpad=0.4, columnspacing=0.4)

#py.plot(bins[:],g1[0])
#py.plot(bins[:],g1[1])
#py.plot(bins[:],g1[2])
py.plot(bins[:],g1[3])
py.plot(bins[:],g2[3],'r')

















#plotting a mixed Phase Image