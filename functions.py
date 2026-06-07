import numpy as np
import cupy as cp
import matplotlib.pyplot as plt


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
    plt.show(block=False)

def plot_stresses(x, z, stresses: list[np.ndarray], labels: list[str] = [r"$\sigma_{\mathrm{xx}}$", r"$\sigma_{\mathrm{zz}}$", r"$\tau_{\mathrm{xz}}$"], individual_z_extrema=[1,1,1,1]):
    
    fig, axs = plt.subplots(len(stresses), 1, figsize=(6, 3*len(stresses)))
    
    if len(stresses) == 1:
        axs = [axs]

    x_mesh, z_mesh = np.meshgrid(x, z, indexing='ij')

    for i, array in enumerate(stresses):
        current_fig = axs[i].pcolormesh(x_mesh, z_mesh * individual_z_extrema[i+1] / individual_z_extrema[0], array, shading='auto', cmap='viridis')
        axs[i].set_title(labels[i])
        axs[i].set_xlabel("x-coordinate in meter")
        axs[i].set_ylabel("z-coordinate in meter")
        cbar = fig.colorbar(current_fig, ax=axs[i])
        cbar.set_label(labels[i])
    
    plt.show(block=False)

def plot_stresses_with_pressure(x, z, p, stresses: list[np.ndarray],
                                 labels: list[str] = [r"$\sigma_{\mathrm{xx}}$",
                                                       r"$\sigma_{\mathrm{zz}}$",
                                                       r"$\tau_{\mathrm{xz}}$"],
                                 individual_z_extrema=[1,1,1,1]):

    if labels is None:
        labels = [
            r"$\sigma_{xx}$",
            r"$\sigma_{zz}$",
            r"$\tau_{xz}$"
        ]

    if individual_z_extrema is None:
        individual_z_extrema = [1]*(len(stresses)+1)

    n = len(stresses)

    fig = plt.figure(
        figsize=(5*n, 5),
        constrained_layout=False
    )

    gs = plt.GridSpec(
        2,
        2*n,
        figure=fig,
        height_ratios=[1, 6],
        width_ratios=sum(([20, 1] for _ in range(n)), []),
        hspace=0.0,
        wspace=0.15
    )

    x_mesh, z_mesh = np.meshgrid(
        x,
        z,
        indexing="ij"
    )

    for i, stress in enumerate(stresses):

        # ------------------
        # Pressure axis
        # ------------------
        ax_p = fig.add_subplot(gs[0, 2*i])

        ax_p.fill_between(
            x,
            p,
            0
        )

        # flip back to normal orientation
        # high pressure at top
        ax_p.set_ylim(max(p), 0)

        ax_p.set_ylabel("p")

        ax_p.tick_params(
            axis="x",
            bottom=False,
            labelbottom=False
        )

        ax_p.spines["top"].set_visible(False)
        ax_p.spines["right"].set_visible(False)

        # ------------------
        # Stress axis
        # ------------------
        ax_s = fig.add_subplot(
            gs[1, 2*i],
            sharex=ax_p
        )

        cax = fig.add_subplot(
            gs[1, 2*i + 1]
        )

        pcm = ax_s.pcolormesh(
            x_mesh,
            z_mesh *
            individual_z_extrema[i+1] /
            individual_z_extrema[0],
            stress,
            shading="auto",
            cmap="viridis"
        )

        ax_s.set_title(labels[i])

        ax_s.set_xlabel("x [m]")

        if i == 0:
            ax_s.set_ylabel("z [m]")

        cbar = fig.colorbar(
            pcm,
            cax=cax
        )

        cbar.set_label(labels[i])

    plt.show(block=False)