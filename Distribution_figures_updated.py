from netCDF4 import Dataset
from pyhdf import SD


def plotting_para():
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-white')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern Roman'
    plt.rcParams['font.monospace'] = 'Computer Modern Typewriter'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['axes.titlesize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
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

#TODO add the optical depth data
str2='C:/Users/Neelesh/OneDrive/Honours Research/colocation/Cloud_phase_dataset_water_reset1.npy'
str1='C:/Users/Neelesh/OneDrive/Honours Research/colocation/Cloud_phase_dataset_ice_reset1.npy'
#files_list='C:/Users/Neelesh/Downloads/files_list_tot.npy'
#the file_list contains all the files for the analysis.
import numpy as np
#file_list=np.load(files_list)
from file_god_contains_functions import _Cal3
#here we will split the data by season to stratify and understand the data more, here we are extracting the months of each data
month_list=[]
for i in file_list:
    month_list.append(_Cal3(i).month)
month_list=np.array(month_list)
c_sum=np.where((month_list<3)|(month_list>10))#summer and winter months extraction
c_wint=np.where((month_list<8)&(month_list>5))




#note that the above two files are converted for the MACOS system and the below commands need to be chanaged for PC
import numpy as np
x=np.load(str1)#ice dataset
x1=np.load(str2)#water dataset
#let us create the a similar figure to that of MODIS algorithm
 y=np.nansum(x[:,:,:,c_wint[0]],axis=3)#a sum over all possible files
 y1=np.nansum(x1[:,:,:,c_wint[0]],axis=3)
# """it is entirely neccessary to sum over the axis as there each index (axis 3) represents an independant height layer"""
 """here we have added another file for seasonal variations"""
import numpy as np
#season=np.load('/Users/neeleshrampal/OneDrive/Honours Research/colocation/Southern_Colocation_reset1.npy')
season=np.load('C:/Users/Neelesh/OneDrive/Honours Research/colocation/Southern_Colocation_reset1.npy')

 simple function to do plotting of two variables
# def plot(x,y):
#     import matplotlib.pyplot as plt
#     plt.figure()
#     if y<>[]:
#
#         plt.plot(x,y)
#         plt.show()
#     else:
#         plt.plot(x)
#         plt.show()
the figure here makes it easier to plot normal figures, two examples shown below
plot(temp,np.nansum(y[i],axis=(0,1))/np.nansum(np.nansum(y[i],axis=(0,1))))
plot(temp,y[i,j,0]/sum(y[i,j,0]))
TODO check sum of all altitudes
#y11=sum(y,axis=2)#sum(y,axis=2)
#y12=sum(y1,axis=2)
temp=np.arange(231,290,1)
i=1
j=0# hemisphere
k=0
sum(season[0]['liq'][2020])

#examine the cases with max amount of ice
np.where(np.array(season[0]['ice'])==max(np.array(season[0]['ice'])))
#open the files list
#index is 2020
season[1]['cal_ice'][0][2020]

sum(season[1]['cal_ice'][0])*1.0/sum(season[0]['ice'])um(season[1]['cal_liq'][1])+sum(season[1]['cal_liq'][2])
#TODO CHECK YOUR CODE, MAYBE ICE IS BETTER THAN YOU THINK, check index of ice to see if misclassified is actually occuring
from file_god_contains_functions import *
plotting_para()
fig, ax1 = plt.subplots(dpi=100)#how to use figsie paramter

#the bar graphs are shaped (x position, y position, width, align=(center, edge))
ax1.bar(temp,y[i,j,k]/sum(y[i,j,k]),0.7,alpha=0.8,color='b')
ax1.bar(temp,y1[i,j,k]/sum(y1[i,j,k]),0.7,alpha=0.5,color='r')
#TODO more collocation work as results are not the same
ax1.set_xlabel('MODIS Cloud Top Temperature 1km')
ax1.set_ylabel('Normalised Probability')
ax1.legend(['MODIS Ice','MODIS Liquid'])
ax1.set_title('Distributions of CTP as function of temperature MODIS SO')
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
#between high and low cloudfrom matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
#x = np.arange(4)
#money = [1.5e5, 2.5e6, 5.5e6, 2.0e7]
#TODO HDBFHDBF




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
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(formatter)
    plt.bar(x, money)
    plt.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))
    plt.show()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
formatter = FuncFormatter(millions)


u,v=np.meshgrid(np.arange(-10,10,0.01),np.arange(-10,10,0.01))
z=u**v
figure()
imshow(z)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
Axes3D.plot_surface(u,v, z,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

c=np.where(np.nansum(x[0,0,0,:]+x1[0,0,0,:],axis=1)>0)
tot=np.array(season[0]['ice'])+np.array(season[0]['liq'])
c1=np.where(tot>0)
print len(c1[0])

equiv=np.nansum(x[0,0,0,:]+x1[0,0,0,:],axis=1)[c]
py.figure()
py.plot(equiv[0:][0:15000],tot[c1][0:23825][0:15000],'x')
py.show()
"""note something weird happens from 15000 onwards, some sort of break"""
from scipy.stats import linregress
x=linregress(equiv[0:],tot[c1][0:23825])
#lets make plots up till 15000








j=25000
str2='/Users/neeleshrampal/OneDrive/Honours Research/colocation/Cloud_phase_dataset_water_reset1.npy'
str1='/Users/neeleshrampal/OneDrive/Honours Research/colocation/Cloud_phase_dataset_ice_reset1.npy'
#c_sum=np.where((month_list<3)|(month_list>10))#summer and winter months extraction
#c_wint=np.where((month_list<8)&(month_list>5))
#sum(season[1]['cal_ice'][0])

#1.0*(sum(season[0]['cal_liq'][2])+sum(season[3]['cal_liq'][0]))/sum(season[0]['liq'])

#note that the above two files are converted for the MACOS system and the below commands need to be chanaged for PC
import numpy as np
x=np.load(str1)#ice dataset
x1=np.load(str2)#water dataset

temp=np.arange(230,290)[:-1]
#two stage sampling error
#ice=x[0,0,0,c[0]][0:j]
#c_sth=np.where(tot1>0)

import matplotlib.pyplot as plt
import numpy as np


width,height=plt.figaspect(0.8)#make sure it is consistent throughout
fig,ax11 = plt.subplots(figsize=(width,height),nrows=2,ncols=2,dpi=300,sharey=True)
alg=0
list=['CALIOP','','','']
list2=[' SO','Revised SO', 'Infrared SO','Optical Properties SO']
list3=[' NA','Revised NA', 'Infrared NA','Optical Properties NA']

j=25000
for ax1 in np.array(ax11).ravel().tolist():
     # choose your algorithm
    c = np.where(np.nansum(x[alg, 0, 0, :] + x1[alg, 0, 0, :], axis=1) > 0)
    glac_s = np.nansum(x[alg, 0, 0, c[0]][0:j], axis=0) / (
                 np.nansum(x1[alg, 0, 0, c[0]][0:j], axis=0) + np.nansum(x[alg, 0, 0, c[0]][0:j], axis=0))
    glac_n = np.nansum(x[alg, 1, 0, c[0]][0:j], axis=0) / (
                 np.nansum(x1[alg, 1, 0, c[0]][0:j], axis=0) + np.nansum(x[alg, 1, 0, c[0]][0:j], axis=0))


    # let us create the a similar figure to that of MODIS algorithm
    # y=np.nansum(x[:,:,:,c_wint[0]],axis=3)#a sum over all possible files
    # y1=np.nansum(x1[:,:,:,c_wint[0]],axis=3)
    p_sth = np.nansum(x[alg, 0, 0, c[0]][0:j], axis=0) / (
                np.nansum(x1[alg, 0, 0, c[0]][0:j], axis=0) + np.nansum(x1[alg, 0, 0, c[0]][0:j], axis=0))
    # sampling_error=1.96*np.sqrt(p_sth*(1-p_sth)/(np.nansum(x[alg,0,0,c[0]][0:j],axis=0)*1/p_sth))
    p_nth = np.nansum(x[alg, 1, 0, c[0]][0:j], axis=0) / (
                np.nansum(x1[alg, 1, 0, c[0]][0:j], axis=0) + np.nansum(x1[alg, 1, 0, c[0]][0:j], axis=0))
    # sampling_error_n=1.99*np.sqrt(p_nth*(1-p_nth)/(np.nansum(x[0,1,0,c[0]][0:j],axis=0)*1/p_nth))

    tot1 = x1[alg, 0, 0, c[0]][0:j] + x[alg, 0, 0, c[0]][0:j]
    tot2 = x1[alg, 1, 0, c[0]][0:j] + x[alg, 1, 0, c[0]][0:j]

    """MODIS ALGORITHM DETECTS MORE ICE"""
    v_sth = np.zeros(59)
    v_nth = np.zeros(59)
    for i in range(59):
        c_sth = np.where(tot1[:, i] > 0)
        n_samp = len(c_sth[0])
        m_bar = sum(tot1[:, i]) / n_samp
        # average number of samples in each cluster, lower temperature have a fewer samples,as averge number of samples
        # is less than other temperatures, might need to consider this.
        c_nth = np.where(tot2[:, i] > 0)
        n_sampn = len(c_nth[0])
        m_barn = sum(tot2[:, i]) / n_sampn
        v_sth[i] = np.nansum((x[alg, 0, 0, c[0]][0:j, i][c_sth] - p_sth[i] * tot1[:, i][c_sth]) ** 2, axis=0) / (
                    n_samp * (n_samp - 1) * m_bar ** 2)
        # se = np.sqrt(v_sth) * 1.96

        v_nth[i] = np.nansum((x[alg, 1, 0, c[0]][0:j, i][c_nth] - p_nth[i] * tot2[:, i][c_nth]) ** 2, axis=0) / (
                    n_sampn * (n_sampn - 1) * m_barn ** 2)

    n2 = np.nansum(tot2, axis=0)
    # v_nth=np.nansum((x[0,1,0,c[0]][0:j][i]-np.repeat(p_nth,tot2.shape[0]).reshape(tot2.shape[0],59)*tot2)**2,axis=0)/(n2*(n2-1)*m_bar2**2)
    sen = np.sqrt(v_nth) * 1.96
    se = np.sqrt(v_sth) * 1.96
    ax1.fill_between(temp, glac_s-se, glac_s+se,alpha=0.3,color='b')
    ax1.plot(temp,glac_s,'b-')
    ax1.fill_between(temp, glac_n-sen, glac_n+sen,alpha=0.3,color='r')
    ax1.plot(temp,glac_n,'r-')
    #ax1.set_ylabel('between y1 and 0')
    ax1.set_xlim(238,290)
    ax1.set_ylim(0,1)
    ax1.set_xticks(np.arange(240, 290, 10))
    ax1.legend([list[alg] + list2[alg], list[alg] + list3[alg]])
    if alg == 2:
         ax1.set_xlabel('MODIS Cloud Top Temperature 1km (K)')
         ax1.set_ylabel('Glaciation Probability')
     elif alg == 3:
         ax1.set_xlabel('MODIS Cloud Top Temperature 1km (K)')
     elif alg == 0:
         ax1.set_ylabel('Glaciation Probability')
     ax1.set_xlim(238, 290)
    fig.show()
    alg=alg+1
fig.savefig('glaciation_probability_dpi300.pdf')
py.figure()

py.plot(temp,glac_s+se,'r--')
py.plot(temp,glac_s-se,'r--')
py.plot(temp,glac_s,'r-')
py.plot(temp,glac_n+sen,'b--')
py.plot(temp,glac_n-sen,'b--')
py.plot(temp,glac_n,'b-')
py.xlim(230,290)
py.ylim(0,1.5)
#py.plot(np.nansum(x1[0,0,0,0:15000],axis=0))
py.show()
"""glaciation probability plots done"""
#you need to prepre the histogram plots and have them done today

