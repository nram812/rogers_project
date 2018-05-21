from netCDF4 import Dataset
from pyhdf import SD


def plotting_para():
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-white')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern Roman'
    plt.rcParams['font.monospace'] = 'Computer Modern Typewriter'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12
#the function above, is able to vary the plotting parameters
fig, ax11 = plt.subplots(nrows=1,figsize=(width,height), dpi=100)
colorsList = ['indigo', '#FFE600']
CustomCmap = matplotlib.colors.ListedColormap(colorsList)
def plotting_para_hidden():
    width, height = plt.figaspect(0.8)
    # fig = plt.figure(figsize=(width,height), dpi=100)
    fig, ax11 = plt.subplots(nrows=1, figsize=(width, height), dpi=100)
    ax1 = ax11  # [1]
    c = np.where(phase[y1] == 1)
    ax1.plot(lat1[y1][c][::5], diff[x1[0], x1[1]][c][::5], color='r', marker='o', linewidth=0)
    xlim(-58, -62)
    c = np.where(phase[y1] == 2)
    ax1.plot(lat1[y1][c][::5], diff[x1[0], x1[1]][c][::5], color='#0EBFE9', marker='o', linewidth=0)
    xlim(-58, -62)
    ax1.set_xlabel('Latitude')
    # # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('BTD(K)')

    ax2 = ax1
    # xtickloc = ax1.get_yticks()
    # ax2.set_yticks(xtickloc)
    ytick1 = np.arange(-500, 8200, 1000)
    ax2.set_yticklabels([str(y) for y in ytick1])
    red_patch = mpatches.Patch(color='#FFE600', label='Ice')
    blue_patch = mpatches.Patch(color='indigo', label='Liquid')
    blue_patch1 = Line2D(range(1), range(1), color='#BFEFFF', marker='o', markersize=10, markerfacecolor='#0EBFE9',
                         linewidth=0)  # mpatches.Circle((3,3),color='#0067acff',marker='o')
    red_patch1 = Line2D(range(1), range(1), color='r', marker='o', markersize=10, markerfacecolor="r",
                        linewidth=0)  # mpatches.Circle((3,3),color='#0067acff',marker='o')

    # handles = handlesL + handlesR
    # labels = labelsL + labelsR
    plt.legend(handles=[red_patch, blue_patch, blue_patch1, red_patch1], loc='upper right', ncol=2, fontsize=12,
               handletextpad=0.4, columnspacing=0.4)
    # plt.legend(handles=[blue_patch,red_patch])
    ax2.imshow(Image2[0:290, :][::-1], extent=[lat1[0], lat1[-1], -0.5, 8.200], cmap=CustomCmap, alpha=0.55, clim=[1, 2])
    ax2.set_ylabel('Cloud Height (km)')
    fig.tight_layout()
    plt.show()
#note the function below needs to be updated, as the parameters have been taken from
# an old function

"""the functions above are for plotting purposes , the code below is to plot histogram plots for research"""
import numpy as np
#str1='/Users/neeleshrampal/OneDrive/Honours Research/colocation/Cloud_phase_dataset_ice_reset1.npy'
#str2='/Users/neeleshrampal/OneDrive/Honours Research/colocation/Cloud_phase_dataset_water_reset1.npy'
str2='C:/Users/Neelesh/OneDrive/Honours Research/colocation/Cloud_phase_dataset_water_reset1.npy'
str1='C:/Users/Neelesh/OneDrive/Honours Research/colocation/Cloud_phase_dataset_ice_reset1.npy'
#note that the above two files are converted for the MACOS system and the below commands need to be chanaged for PC
x=np.load(str1)#ice dataset
x1=np.load(str2)#water dataset
#let us create the a similar figure to that of MODIS algorithm
y=np.nansum(x,axis=3)#a sum over all possible files
y1=np.nansum(x1,axis=3)
"""it is entirely neccessary to sum over the axis as there each index (axis 3) represents an independant height layer"""

#  simple function to do plotting of two variables
def plot(x,y):
    import matplotlib.pyplot as plt
    plt.figure()
    if y<>[]:

        plt.plot(x,y)
        plt.show()
    else:
        plt.plot(x)
        plt.show()
#the figure here makes it easier to plot normal figures, two examples shown below
#plot(temp,np.nansum(y[i],axis=(0,1))/np.nansum(np.nansum(y[i],axis=(0,1))))
#plot(temp,y[i,j,0]/sum(y[i,j,0]))
#TODO check sum of all altitudes
y11=sum(y,axis=2)#sum(y,axis=2)
y12=sum(y1,axis=2)
temp=np.arange(231,290,1)
i=0
j=0# hemisphere
k=2
plotting_para()
fig, ax1 = plt.subplots(dpi=100)#how to use figsie paramter

#the bar graphs are shaped (x position, y position, width, align=(center, edge))
ax1.bar(temp,y11[i,j]/sum(y11[i,j]),0.7,alpha=0.8,color='b')
ax1.bar(temp,y12[i,j]/sum(y12[i,j]),0.7,alpha=0.7,color='r')
#TODO more collocation work as results are not the same
ax1.set_xlabel('MODIS Cloud Top Temperature 1km')
ax1.set_ylabel('Normalised Probability')
ax1.legend(['CALIOP Ice','CALIOP Liquid'])
ax1.set_title('Temperature distributions of Liquid and Ice clouds in Low Altitude SO caliop')
fig.show()
"""note that the figure for clouds over the southern ocean, has more probable cloud around 270 K which is
due to the higher frequency of low cloud that exists over the SO"""

#TODO create
x='this'
if x in 'this is great':
    print 'chur' #just an example of how to plot things
import matplotlib.pyplot as plt

#TODO create a figure one with high, and low fig,ax1=plt.subplots()
bins=temp
"""what is the width and height used in each of the other figures"""
#width,height=20,21#TODO check what this is "

#TODO altitude and use the figure to explain how the distribution changes
#between high and low cloud

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
#x = np.arange(4)
#money = [1.5e5, 2.5e6, 5.5e6, 2.0e7]
#TODO HDBFHDBF

def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x * 1e-6)


formatter = FuncFormatter(millions)


#ax.yaxis.set_major_formatter(formatter)
plt.bar(temp,y[i,j,2]/sum(y[i,j,2]),3)
#plt.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))
plt.show()
x = np.arange(4)
money = [1.5e5, 2.5e6, 5.5e6, 2.0e7]


def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.1fM' % (x * 1e-6)


formatter = FuncFormatter(millions)

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
plt.bar(x, money)
plt.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))
plt.show()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')