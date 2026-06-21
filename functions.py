import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.colors as mcolors


### Numerical approximation

def compute_subsurface_stresses_trapezoid(x: np.ndarray , z: np.ndarray , p: np.ndarray, z_batch_size: int = 1, individual_z_extrema: list[float] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ calcualte the subsurface stresses over the grid (x,z) corresponding to the pressure distribution applied at z=0
        Note: z[i] != 0 to avoid singularities

        x:            1D array of x-positions dim(Nx,)
        z:            1D array of z-depths   dim(Nz) or a list of three 1D arrays for each stress component with dim(Nz) indicated by individual_z_grids
            — must not contain 0! 
        p:            1D array of pressures  dim(Nx)
        z_batch_size: number of z-levels processed simultaneously
        individual_z_grids: see definition of z
    """

    z = np.abs(z) # the formulas expect strictly positive z-values

    if np.any( np.isclose(z, 0.0)):
        print("\n\nWarning: z is close to zero where teh stress kernels are singualar\n\n")


    # transfer inputs to GPU once
    x_gpu = cp.asarray(x)
    z_gpu = cp.asarray(z)
    p_gpu = cp.asarray(p)

    Nx = len(x)
    Nz = len(z)

    s_xx = cp.zeros((Nx, Nz))
    s_zz = cp.zeros((Nx, Nz))
    s_xz = cp.zeros((Nx, Nz))

    x_minus_s = x_gpu[:,None,None] - x_gpu[None,None,:]  # dim(Nx, 1, Nx)

    for j_start in range(0, Nz, z_batch_size):
        if individual_z_extrema is None:
            scale_z_xx = 1
            scale_z_zz = 1
            scale_z_xz = 1
        else:
            scale_z_xx = individual_z_extrema[1]/individual_z_extrema[0]
            scale_z_zz = individual_z_extrema[2]/individual_z_extrema[0]
            scale_z_xz = individual_z_extrema[3]/individual_z_extrema[0]
            
        
        j_end = min(j_start + z_batch_size, Nz)
        z_batch = z_gpu[j_start:j_end] # dim(z_batch_size) (goes up to but not including j_end)
        # creates grid to evaluate stresses
        #x_eval = x[:,None,None] # dim(Nx,1,1)
        z_batch_eval = z_batch[None,:,None] # dim(1,z_batch_size,1)
        #s_meshgrid = x[None,None,:] # dim(1,1,Ns) with s being the points at which pressure is applied
        #p_meshgrid = p[None,None,:] # dim(1,1,Np) with Np = Ns

        #x_minus_s = (x_eval - s_meshgrid) # dim(Nx,1,Ns)
        #denominator = ( x_minus_s**2 + z_batch_eval**2 )**2

        

        s_xz[:,j_start:j_end] = - ( 2 * (z_batch_eval[:,:,0]*scale_z_xx) **2 / np.pi) * cp.trapezoid( p_gpu[None,None,:] * x_minus_s    / ( x_minus_s**2 + (z_batch_eval*scale_z_xx)**2 )**2, x=x_gpu, axis=2)
        s_xx[:,j_start:j_end] = - ( 2 * (z_batch_eval[:,:,0]*scale_z_zz)     / np.pi) * cp.trapezoid( p_gpu[None,None,:] * x_minus_s**2 / ( x_minus_s**2 + (z_batch_eval*scale_z_zz)**2 )**2, x=x_gpu, axis=2)
        s_zz[:,j_start:j_end] = - ( 2 * (z_batch_eval[:,:,0]*scale_z_xz) **3 / np.pi) * cp.trapezoid( p_gpu[None,None,:]                / ( x_minus_s**2 + (z_batch_eval*scale_z_xz)**2 )**2, x=x_gpu, axis=2)

    
    return cp.asnumpy(s_xx), cp.asnumpy(s_zz), cp.asnumpy(s_xz)

# older version without gpu acceleration

# def compute_subsurface_stresses_trapezoid(x: np.ndarray , z: np.ndarray , p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """ calcualte the subsurface stresses over the grid (x,z) corresponding to the pressure distribution applied at z=0
#         Note: z[i] != 0 to avoid singularities

#         x:            1D array of x-positions dim(Nx,)
#         z:            1D array of z-depths   dim(Nz)  — must not contain 0
#         p:            1D array of pressures  dim(Nx)
#         z_batch_size: number of z-levels processed simultaneously
#     """
#     if np.any( np.isclose(z, 0.0)):
#         print("\n\nWarning: z is close to zero where teh stress kernels are singualar\n\n")
#     # creates grid to evaluate stresses
#     x_eval = x[:,None,None] # dim(Nx,1,1)
#     z_eval = z[None,:,None] # dim(1,Nz,1)
#     s_meshgrid = x[None,None,:] # dim(1,1,Ns) with s being the points at which pressure is applied
#     #p_meshgrid = p[None,None,:] # dim(1,1,Np) with Np = Ns

#     x_minus_s = (x_eval - s_meshgrid) # dim(Nx,1,Ns)
#     ds = x[1] - x[0] # only for equal spacing can be done better by parsing x instead of ds to trapezoid
#     denominator = ( x_minus_s**2 + z_eval**2 )**2

#     s_xx = -(2*z_eval[:,:,0]   / np.pi) * np.trapezoid( p * x_minus_s**2 / denominator, dx=ds, axis=2)
#     s_zz = -(2*z_eval[:,:,0]**3/ np.pi) * np.trapezoid( p                / denominator, dx=ds, axis=2)
#     s_xz = -(2*z_eval[:,:,0]**2/ np.pi) * np.trapezoid( p * x_minus_s    / denominator, dx=ds, axis=2)

#     return s_xx, s_zz, s_xz

#### All plotting

def plot_pressure(x,p):
    plt.plot(x,p)
    plt.xlabel("x-coordinate in m")
    plt.ylabel("pressure in Pa")
    plt.title("Profile of applied line pressure ")
    print("3")
    plt.show(block=False)

def plot_stresses(x, z, stresses: list[np.ndarray], labels: list[str] = [r"$\sigma_{\mathrm{xx}}$", r"$\sigma_{\mathrm{zz}}$", r"$\tau_{\mathrm{xz}}$"], individual_z_extrema: list[float] = [1,1,1,1]):
    
    fig, axs = plt.subplots(len(stresses), 1, figsize=(6, 3*len(stresses)))
    
    if len(stresses) == 1:
        axs = [axs]

    x_mesh, z_mesh = np.meshgrid(x, z, indexing='ij')

    for i, array in enumerate(stresses):
        current_fig = axs[i].pcolormesh(x_mesh, z_mesh * individual_z_extrema[i+1] / individual_z_extrema[0], array, shading='auto', cmap='coolwarm')
        axs[i].set_title(labels[i])
        axs[i].set_xlabel("x-coordinate in meter")
        axs[i].set_ylabel("z-coordinate in meter")
        cbar = fig.colorbar(current_fig, ax=axs[i])
        cbar.set_label(labels[i])
    

    plt.show(block=False)

def plot_stresses_with_pressure(x, z, p, stresses: list[np.ndarray], labels: list[str] = [r"$\sigma_{\mathrm{xx}}$", r"$\sigma_{\mathrm{zz}}$", r"$\tau_{\mathrm{xz}}$"], individual_z_extrema: list[float] = [1,1,1,1], vmin=None, vmax=None, cmap="turbo", col_spacing=0.1, title=None):
    """
    vmin/vmax control shared color scale
    col_spacing controls horizontal whitespace between columns.
    """

    if labels is None:
        labels = [r"$\sigma_{xx}$", r"$\sigma_{zz}$", r"$\tau_{xz}$"]

    n = len(stresses)


    x_mesh, z_mesh = np.meshgrid(x, z, indexing="ij")

    fig = plt.figure(figsize=(5 * n, 6), constrained_layout=False)
    
    fig.suptitle(
    title,
    fontsize=16,
    y=0.98
)

    # Arrangement of columns in the bottom row: [s_xx, colorbar, white space, s_zz, colorbar, white space, s_xz, colorbar] resulting in 5 columns
    gs = plt.GridSpec(2, 3 * n - 1, figure=fig, height_ratios=[1, 5], width_ratios=sum(([10, 1, 4] for _ in range(n)), [])[:-1], wspace=col_spacing, hspace=0.05)

    for i, stress in enumerate(stresses):
        # pressure plots
        ax_p = fig.add_subplot(gs[0, 3*i])

        ax_p.fill_between(x, 0, p, color="#01153E", alpha=0.8)

        ax_p.set_xlim(x.min(), x.max())
        ax_p.set_ylabel("p in Pa")

        ax_p.tick_params(axis="x", bottom=False, labelbottom=False)

        # reomove lines around plot
        ax_p.spines["top"].set_visible(False)
        ax_p.spines["right"].set_visible(False)
        ax_p.spines["bottom"].set_visible(False)

        # stress plots
        ax_s = fig.add_subplot(gs[1, 3*i])

        # hardcoded that zero is green on all colorbars
        if stress.max() < 0:
            vmax_updated = 0
            cmap_updated = mcolors.LinearSegmentedColormap.from_list("lower_half", plt.get_cmap(cmap)( np.linspace(0, 0.5, 256)))
        else:
            cmap_updated = cmap
            vmax_updated = vmax
        # # set green to zero on colorbar
        # print(stress.min(), stress.max())
        # if stress.min() > 0:
        #     norm_min = -0.01
        # else:
        #     norm_min = stress.min()
        # if stress.max() < 0:
        #     norm_max = 0.01
        # else:
        #     norm_max = stress.max()
        # norm = TwoSlopeNorm(
        # vmin=norm_min,   # autoscaled lower bound
        # vcenter=0,
        # vmax=norm_max,    # autoscaled upper bound
        # )

        pcm = ax_s.pcolormesh(x_mesh, z_mesh * individual_z_extrema[i+1] / individual_z_extrema[0], stress, shading="auto", cmap=cmap_updated, vmin=vmin, vmax=vmax_updated)

        #ax_s.set_title(labels[i])
        ax_s.set_xlabel("x in m")

        if i == 0:
            ax_s.set_ylabel("z in m")

        # dedicated column for the color bar
        cax = fig.add_subplot(gs[1, 3*i + 1])
        cbar = fig.colorbar(pcm, cax=cax)
        cbar.set_label(f"{labels[i]} in Pa", fontsize=12, labelpad=4,)

    plt.show(block=False)


# Convergence analysis z-grid ### ??? needs check ########################################################################################
def plot_sigma_zz_convergence(config, z_grid_sizes, z_batch_size=500):
    """
    Plot maximum |sigma_zz| as a function of the number
    of grid points in the z-direction.
    """

    x = config.create_surface_grid()
    p = config.contact.pressure_default(x)

    sigma_zz_max = []

    original_nz = config.grid.z_num_grid_points

    for nz in z_grid_sizes:
        print(f"Computing for nz = {nz}")

        config.grid.z_num_grid_points = nz
        z = config.create_depth_grid()

        _, s_zz, _ = compute_subsurface_stresses_trapezoid(
            x,
            z,
            p,
            z_batch_size=z_batch_size
        )

        sigma_zz_max.append(np.max(np.abs(s_zz)))

        print(f"nz = {nz:6d}, "
            f"max(|sigma_zz|) = {sigma_zz_max[-1]:.6e} Pa"
        )

    config.grid.z_num_grid_points = original_nz

    plt.figure(figsize=(7, 4))
    plt.plot(z_grid_sizes, sigma_zz_max, "o-")
    plt.xlabel("Number of grid points in z-direction")
    plt.ylabel(r"max $|\sigma_{zz}|$ [Pa]")
    plt.title(r"Convergence of $\sigma_{zz}$ with z-grid resolution")
    plt.grid(False)
    plt.tight_layout()
    plt.show(block=False)