import numpy as np
import os

# go to current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
 

windCase   = '10B'

flatData   = np.loadtxt('rollerLevelData_'+windCase+'.out')

rollerData = flatData.reshape((-1,27,6)) # -1 indicates Python should work out this final value, 27 = number of rollers, 6 = number of roller level outputs at each timestep

# rollerData has three dimension: (time, roller number, roller variable). The six roller variables, in order, are:

#    1) roller load in Newtons (need to double check that!) time steps of 0.01s (same as global)

#    2) roller location in rads

#    3) elliptical semi-axes on inner and then outer raceway contacts (a_in, b_in, a_out,b_out) all in millimetres (again I should double check!)



globalData = np.loadtxt('globalBearingData_'+windCase+'.csv',skiprows=1,delimiter=',')

time    = globalData[:,0] # time in seconds     starts at 99.9999 seconds turbine starts to settle in

power   = globalData[:,1] # power in Watts

wSpeed  = globalData[:,2] # rotor average wind speed in m/s

pitch   = globalData[:,3] # blade pitch angle in rads

shaftSp = globalData[:,4] # shaft rotational speed (i.e. inner ring) in rad/s

Fa      = globalData[:,5] # bearing axial applied load in Newtons   

Fr      = globalData[:,6] # bearing radial applied load in Newtons

theta   = globalData[:,7] # inner ring position over time in rads


### Checks:
print(rollerData)

if rollerData[:,:,1].min() >= -np.pi and rollerData[:,:,1].max() <= np.pi:
    print("\nPi Test for rollerData[:,:,1]: passed\nMin: ", rollerData[:,:,1].min(), " >= pi = ", -np.pi, "\nMax: ",rollerData[:,:,1].max(), " <= pi = ", np.pi)
else:
    print("\n\n\nERROR\n\nPi Test FAILED for rollerData[:,:,1]\nMin: ", rollerData[:,:,1].min(), " >= pi = ", -np.pi, "\nMax: ",rollerData[:,:,1].max(), " <= pi = ", np.pi,"\n\n\n")

print(rollerData.shape, globalData.shape)
if rollerData.shape[0] == globalData.shape[0]:
    print("\nNumber of time steps test: passed\nrollerData.shape: ", rollerData.shape, "\nglobalData.shape: ", globalData.shape)
else:
    print("\n\n\nERROR\n\nNumber of time steps test: FAILED\nrollerData.shape: ", rollerData.shape, "\nglobalData.shape: ", globalData.shape)