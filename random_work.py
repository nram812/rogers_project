dataset='C:/Users/Neelesh/OneDrive/Rogers Project/Data_for_tbo/2002_2017Uwind.nc'
key='u10'
dataset='C:/Users/Neelesh/OneDrive/Rogers Project/Data_for_tbo/2002_2017sst.nc'
key='sst'
key='v10'
dataset='C:/Users/Neelesh/OneDrive/Rogers Project/Teleconnections/2000_2015_netcdf_vwind650.nc'
key='v'
#xp=seasonal(dataset,key)
sst=seasonal(dataset,key)
U=seasonal(dataset,key)
V=seasonal(dataset,key)4



diff_sst1=diff(sst,axis=0)
diff_sst2=diff(sst,axis=1)
dot=np.zeros([204,61,120])
for i in range(204):
    for j in range(60):
        for k in range(120):
            dot[i,j,k]=np.array([diff_sst1[i,j,k],diff_sst2[i,j,k]]).dot(np.array([U[i,j,k],V[i,j,k]]))
            #print i,j
            
0.48*-0.86+0.75*0.047
    seasonal_x=dot.reshape(12,17,61,120,order='F')
    mean_s=np.repeat(np.nanmean(seasonal_x,axis=1),17,axis=0).reshape(12,17,61,120,order='C').reshape(204,61,120,order='F')
    #this is the seasonal trend
    sst_anom=dot-mean_s
    figure()
    plot(run_mean[:,0],run_mean[:,1],'x')
    plot()
run_mean=np.zeros([204,2])
for i in range(204):
    run_mean[i,0]=np.nanmean(sst_anom[i:i+3,31,44])
    run_mean[i,1]=np.nanmean(sst_anom[i:i+3,31,60])
x33=np.cov(sst_anom[:,31,:].T)
cov=np.zeros([120,120])
for i in range(120):
    for j in range(120):
        cov[i,j]=np.corrcoef(sst_anom[:,31,i],sst_anom[:,31,j]).T[0,1]
        
        np.corrcoef(sst_anom[:,31,44],sst_anom[:,31,44]).T[0,1]