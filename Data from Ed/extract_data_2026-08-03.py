import numpy as np
import os
import matplotlib.pyplot as plt

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

theta_force   = globalData[:,7] # inner ring position over time in rads

theta = - np.cumsum(shaftSp) * ( time[1] - time[0] ) # theta value using integration assuming constant time steps; negative sign infront due to convention of clockwise turning turbine looking from upwind
theta = theta % (2*np.pi)

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
    # ??? hot fix
    print("attempt hot fix: rollerData is trimmed")
    rollerData = rollerData[:-1,:,:]
    if rollerData.shape[0] == globalData.shape[0]:
        print("Hot fix worked\nNumber of time steps test: passed\nrollerData.shape: ", rollerData.shape, "\nglobalData.shape: ", globalData.shape)
    else: print("Hot fix failed!!!!!!!!!!!!!!!!!!!!!!!!!!!!")




##### Save pressure fields on a segment of the inner raceway
### Assumptions to remember:
# - theta in [-pi, pi]

### Parameters

number_of_closest_rollers = 3 # number of rollers accounted for
angle_offset = 0#np.pi/6 # specific segments of the inner raceway can be investigated utilizing this offset
theta = theta + angle_offset
#length_of_line_contact = 1 # ??? roller width ????
epsilon = 1e-9 # prevent division by zero if contact path vanishes
output_filename = f"inner_ring_segment_pressure_{windCase}_closest_rollers_{number_of_closest_rollers}_angle_offset_{round(angle_offset, 2)}"


# find closest rollers at each time step
# 1. calculate distance of roller to segment
roller_locations = rollerData[:, :, 1]
angle_diff = (roller_locations - theta[:, np.newaxis] + np.pi) % (2*np.pi) - np.pi # np.newaxis accounts for the time dimension;
                                                                                   # account for 2*pi periodicity; 
                                                                                   # addign and subtratcing np.pi shifts the frame of reference from [-pi,pi[ to [0,2pi[ and back
abs_angle_diff = np.abs(angle_diff) # absolute difference of the angle of each roller to the angle of the segment for each time step
# 2. determine the closes rollers to the segment
closest_rollers_idx = np.argsort(abs_angle_diff, axis=1)[:, :number_of_closest_rollers] # safe the indices of the closes rollers
# 3. create time indexing
time_idx = np.arange(rollerData.shape[0])[:, np.newaxis]

# Extraction of specifc variables with shape: time x number_of_closest_rollers
loads = rollerData[time_idx, closest_rollers_idx, 0]       # Roller load (N)
relative_locations = angle_diff[time_idx, closest_rollers_idx]    # Relative angular position (rad)
a_in = rollerData[time_idx, closest_rollers_idx, 2]        # Inner semi-axis a (mm)
b_in = rollerData[time_idx, closest_rollers_idx, 3]        # Inner semi-axis b (mm)


p_max = 3 * loads / (2 * np.pi * a_in * b_in + epsilon)

# ??? npz is saved as binary and more compressed but not as easyly readible ???
np.savez_compressed(
    output_filename,
    time=time,
    roller_indices=closest_rollers_idx,
    relative_angle_rad=relative_locations,
    load_N=loads,
    a_in_mm=a_in,
    b_in_mm=b_in,
    P_max_inner_MPa=p_max
)
print(f"Segment data successfully saved to {output_filename}")

