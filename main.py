import numpy as np
import cupy as cp
import functions as func

# Remarks:
#

# To do's:
# 

#import config
#print(config.__file__)
# choose the parameters for the simulation
from config import default_config as config

if __name__ == "__main__":

    print(cp.__version__)
    print(cp.cuda.runtime.runtimeGetVersion())

    x = config.create_surface_grid()
    z = config.create_depth_grid()
    #print(z)

    p = config.contact.pressure_default(x=x)

    func.plot_pressure(x=x, p=p)

    s_xx, s_zz, s_xz = func.compute_subsurface_stresses_trapezoid(x, z, p, z_batch_size=100)
    func.plot_stresses(x, z, [s_xx, s_zz, s_xz])

    s_xx, s_zz, s_xz = func.compute_subsurface_stresses_trapezoid(x, z, p, z_batch_size=100, individual_z_extrema=config.grid.individual_z_factors)
    func.plot_stresses(x, z, [s_xx, s_zz, s_xz], individual_z_extrema=config.grid.individual_z_factors)

    func.plot_stresses_with_pressure(x, z, p, [s_xx, s_zz, s_xz], individual_z_extrema=config.grid.individual_z_factors)

    print(np.shape(s_xx))
    print(np.shape(s_xx)[0]/2)
    print(s_xx[int(np.shape(s_xx)[0]/2),0:100])
    print(s_zz[int(np.shape(s_xx)[0]/2),0:100])
    print(s_xz[int(np.shape(s_xx)[0]/2),0:100])

ig     print("Done")

    