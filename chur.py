width, height = plt.figaspect(0.8)  # make sure it is consistent throughout
fig, ax11 = plt.subplots(figsize=(width, height), nrows=2, ncols=2, dpi=100,sharex=True,sharey=True)
alg=0
list=['CALIOP','','','']
list2=[' Ice','Revised Ice', 'Infrared Ice','Optical Properties Ice']
list3=[' Liquid','Revised Liquid', 'Infrared Liquid','Optical Properties Liquid']
for ax1 in np.array(ax11).ravel().tolist():
     # choose your algorithm
    ice_s = np.nansum(x[alg, 0, 0, c[0]][0:j], axis=0)
    water_s=np.nansum(x1[alg, 0, 0, c[0]][0:j], axis=0)
    ice_n = np.nansum(x[alg, 1, 0, c[0]][0:j], axis=0)
    water_n=np.nansum(x1[alg, 1, 0, c[0]][0:j], axis=0)


    tot1 = x1[alg, 0, 0, c[0]][0:j] + x[alg, 0, 0, c[0]][0:j]
    tot2 = x1[alg, 1, 0, c[0]][0:j] + x[alg, 1, 0, c[0]][0:j]

    ax1.fill_between(temp, 0, ice_s/sum(ice_s),alpha=0.3,color='b')
    ax1.fill_between(temp, 0, water_s/sum(water_s), alpha=0.3, color='r')
    ax1.set_xticks(np.arange(240,290,10))
    ax1.legend([list[alg]+list2[alg],list[alg]+list3[alg]])
    if alg==2:
        ax1.set_xlabel('MODIS Cloud Top Temperature 1km (K)')
        ax1.set_ylabel('Normalised Probability')
    elif alg==3:
        ax1.set_xlabel('MODIS Cloud Top Temperature 1km (K)')
    elif alg==0:
        ax1.set_ylabel('Normalised Probability')
    ax1.set_xlim(238,290)

    #ax1.set_yticks(np.linspace(0,0.125,4))
    #ax1.plot(temp,glac_s,'bx-')
    #ax1.fill_between(temp, glac_n-sen, glac_n+sen,alpha=0.3,color='r')
    #ax1.plot(temp,glac_n,'rx-')
    #ax1.set_ylabel('between y1 and 0')
    fig.show()
    alg=alg+1
fig.savefig('Distributions_almost_final.pdf')