#from file_god_contains_functions import *
plotting_para()
import numpy as np
cal="C:\Users\Neelesh\OneDrive\Research Paper\Data_files\CAL_LID_L2_VFM-Standard-V4-10.2008-08-21T13-55-31ZD.hdf"
mod="C:\Users\Neelesh\OneDrive\Research Paper\Data_files\MYD06_L2.A2008234.1355.006.2013350215759.hdf"
rad="C:\Users\Neelesh\OneDrive\Research Paper\Data_files\MYD021KM.A2008234.1355.006.2012069224530.hdf"

diff,ctt=btd1(mod,rad)
c=Cal2(cal)
lat,lon,Image2,Image3=c.file_sort()
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(Image2,cmap='viridis')
plt.show()
x=range(-500,8200,30)+range(8200,20200,60)+range(20200,30100,180)
y=np.repeat(lat,5)
laty=y
X,Y=np.meshgrid(x,y)
import numpy.ma as ma
Z=Image2
Zm = ma.masked_where(np.isnan(Z),Z)
#plt.pcolormesh(X,Y,Zm.T)
width,height=plt.figaspect(0.8)
fig,ax1=plt.subplots(figsize=(width,height),dpi=300)
x,y=co_locate(cal,mod)
ax1.pcolormesh(Y[y,:],X[y,:],Zm.T[y,:],cmap='viridis')
ax1.set_yticklabels(np.arange(0,500,3500))
ax2=ax1.twinx()
ax2.plot(laty[y],diff[x[0],x[1]])
"""just need to figure out how to adjust the x tick labels"""



plt.show()